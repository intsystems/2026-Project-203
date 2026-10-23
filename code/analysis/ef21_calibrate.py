"""Small controlled EF21 experiments with a known population gradient.

The bound is evaluated retrospectively from the actual update. This calibrates
the diagnostics; it is not a certificate for a neural-network training run.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from common.optimizers import EF21MuonSign, EF21MuonUSign
from analysis.ef21_report import schedule_values


def run_case(steps=80, schedule="cosine", nesterov=True, downlink="sign",
             initialization="zero", lr=0.03, noise=0.1, seed=17, trace=None):
    torch.manual_seed(seed)
    shape = (6, 8)
    # Uneven coordinates, with a small dense component; fixed across ablations.
    target = torch.full(shape, 0.01, dtype=torch.float64)
    target[0, 0] = 1.0
    weights = torch.linspace(0.2, 1.0, target.numel()).reshape(shape)
    initial = torch.zeros_like(target) if initialization == "zero" else torch.randn_like(target)*0.05
    p = torch.nn.Parameter(initial.clone())
    cls = EF21MuonSign if downlink == "sign" else EF21MuonUSign
    opt = cls([p], lr=lr, momentum=0.95, nesterov=nesterov,
              scale_aspect=False, lmo_dtype=torch.float32)
    opt.capture_direction = True
    if trace is not None:
        if downlink != "sign":
            raise ValueError("the raw EF21-P trace requires sign downlink")
        opt.enable_compression_diagnostics(trace, audit_oracle=True)
    rates = schedule_values(schedule, steps, lr)

    def objective(x):
        return float(0.5*(weights*(x-target).square()).sum())

    values = []
    initial_value = objective(initial)
    for t, eta in enumerate(rates):
        x = opt._exact(p).clone() if downlink == "sign" else p.detach().clone()
        w = p.detach().clone()
        gx, gw = weights*(x-target), weights*(w-target)
        opt.param_groups[0]["lr"] = eta
        p.grad = gw+noise/math.sqrt(p.numel())*torch.randn_like(p)
        opt.step()
        state = opt.state[p]
        estimator, direction = state["grad_estimator"], state["last_direction"]
        r = float(torch.linalg.svdvals(direction).max())
        delta = max(0.0, float(torch.linalg.svdvals(estimator).sum()-(estimator*direction).sum()))
        tracking = float(torch.linalg.svdvals(estimator-gx).sum())
        error = (1+r)*tracking+delta
        s = float(torch.linalg.svdvals(gx).sum())
        sw = float(torch.linalg.svdvals(gw).sum())
        curvature = float(weights.max()*direction.square().sum())
        transfer = math.sqrt(min(shape))*float(weights.max())*float((x-w).norm())
        xnew = state["exact_model"] if downlink == "sign" else p.detach()
        upper = objective(x)-eta*s+eta*error+curvature/2*eta**2
        if objective(xnew) > upper+1e-10:
            raise AssertionError("fixed-radius descent inequality violated")
        values.append((eta, s, sw, error, curvature, transfer))
    mass = math.fsum(v[0] for v in values)
    weighted = lambda j: math.fsum(v[0]*v[j] for v in values)/mass
    stationarity = weighted(1)
    # Supremum curvature directly instantiates the fixed-L Lean schedule lemma.
    curvature_bound = max(v[4] for v in values)
    smooth_term = curvature_bound/2*math.fsum(eta*eta for eta in rates)/mass
    bound = initial_value/mass+weighted(3)+smooth_term
    if stationarity > bound+1e-10 or weighted(2) > bound+weighted(5)+1e-10:
        raise AssertionError("weighted stationarity or query-transfer inequality violated")
    return dict(
        schedule=schedule, nesterov=nesterov, downlink=downlink,
        initialization=initialization, steps=steps, seed=seed, noise_frobenius_std=noise,
        server_weighted_nuclear_gradient=stationarity,
        broadcast_weighted_nuclear_gradient=weighted(2),
        initial_gap_term=initial_value/mass, measured_error_term=weighted(3),
        smoothness_term=smooth_term, server_bound=bound,
        query_transfer_term=weighted(5), broadcast_bound=bound+weighted(5),
        checked=True,
        scope="known quadratic, actual CPU EF21 implementation; retrospective pathwise inequality, not an ex-ante stochastic rate",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--output", type=Path)
    p.add_argument("--trace", type=Path, help="Optional raw trace for one default calibration case")
    args = p.parse_args()
    if args.trace:
        results = [run_case(steps=args.steps, trace=args.trace)]
    else:
        results = [run_case(args.steps, schedule, nes, down, init)
                   for schedule in ("constant", "cosine") for nes in (False, True)
                   for down in ("exact", "sign") for init in ("zero", "random")]
    body = json.dumps(results, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
