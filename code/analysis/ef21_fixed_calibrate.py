"""Enumerate a finite stochastic EF21 problem and check the closed budget.

The noise law is independent Rademacher noise at each round and worker.
No Monte Carlo estimate substitutes for an expectation. The matrix LMO calls
the shared Newton-Schulz implementation. Compatible ties are fixed to +1.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import torch

from analysis.ef21_fixed_momentum import fixed_momentum_bound
from analysis.ef21_report import schedule_values
from common.optimizers import muon_lmo


def enumerate_case(steps=5, workers=1, nesterov=True, downlink="sign", schedule="cosine", roundoff=0.0):
    if workers < 1 or steps < 1 or workers*steps > 12:
        raise ValueError("enumeration requires 1 <= workers*steps <= 12")
    if downlink not in ("sign", "exact"):
        raise ValueError("unknown downlink")
    beta, sigma = 0.2, 0.1
    mu, theta = 1-beta, beta if nesterov else 0.0
    rates = schedule_values(schedule, steps, lr=0.03)
    mass = math.fsum(rates)
    weights = torch.tensor([[1.0, 0.7], [0.4, 0.9]], dtype=torch.float64)
    base = torch.tensor([[0.7, 0.15], [-0.2, 0.4]], dtype=torch.float64)
    targets = torch.stack([base+0.1*(j-(workers-1)/2) for j in range(workers)])
    mean_target = targets.mean(0)
    noise_direction = torch.zeros_like(base)
    noise_direction[0, 0] = sigma
    rounding = torch.full_like(base, roundoff/math.sqrt(base.numel()))

    def compress(value):
        return value.abs().mean()*torch.where(value >= 0, 1.0, -1.0)

    def nuclear(value):
        return float(torch.linalg.svdvals(value).sum())

    accum = dict(tracking=0.0, server_score=0.0, broadcast_score=0.0)
    target_increments = [0.0]*steps
    noise_variance = [0.0]*steps
    post_uplink = [0.0]*steps
    bmax, rmax, dmax, gapmax, gain_min = 0.0, 0.0, 0.0, 0.0, 1.0
    paths = 2**(steps*workers)
    for signs in itertools.product((-1.0, 1.0), repeat=steps*workers):
        x, w = torch.zeros_like(base), torch.zeros_like(base)
        momentum, estimator = torch.zeros_like(targets), torch.zeros_like(targets)
        z, previous_target = torch.zeros_like(targets), torch.zeros_like(targets)
        for t, eta in enumerate(rates):
            gx, gw = weights*(x-mean_target), weights*(w-mean_target)
            noise = torch.stack([signs[t*workers+j]*noise_direction for j in range(workers)])
            reply = weights*(w-targets)+noise
            momentum = mu*momentum+beta*reply+rounding
            target = (1-theta)*momentum+theta*reply+rounding
            z = mu*z+beta*noise
            y = (1-theta)*z+theta*noise
            noise_variance[t] += float(y.mean(0).square().sum())/paths
            target_increments[t] += float((target-previous_target).flatten(1).norm(dim=1).mean())/paths
            previous_target = target.clone()
            for j in range(workers):
                estimator[j] += compress(target[j]-estimator[j])+rounding
            post_uplink[t] += float((estimator-target).flatten(1).norm(dim=1).mean())/paths
            h = estimator.mean(0)
            direction = muon_lmo(h, dtype=torch.float64, scale_aspect=False)
            bmax = max(bmax, float(direction.norm()))
            rmax = max(rmax, float(torch.linalg.svdvals(direction).max()))
            hnorm, pairing = nuclear(h), float((h*direction).sum())
            dmax = max(dmax, hnorm-pairing)
            if hnorm:
                gain_min = min(gain_min, pairing/hnorm)
            accum["tracking"] += eta*float((h-gx).norm())/(mass*paths)
            accum["server_score"] += eta*nuclear(gx)/(mass*paths)
            accum["broadcast_score"] += eta*nuclear(gw)/(mass*paths)
            x -= eta*direction
            w = x.clone() if downlink == "exact" else w+compress(x-w)
            w += rounding
            gapmax = max(gapmax, float((x-w).norm()))
    initial_local = -weights*targets
    assumptions = dict(beta=beta, nesterov=nesterov, workers=workers, sigma=sigma,
        smoothness=1.0, local_lipschitz=1.0, direction_frobenius=bmax,
        direction_operator=rmax, nuclear_frobenius=math.sqrt(2),
        alpha_up=0.25, alpha_down=1.0 if downlink == "exact" else 0.25,
        initial_gap=float((0.5*weights*targets.square()).sum((1, 2)).mean()),
        initial_global_gradient=float(initial_local.mean(0).norm()),
        initial_local_gradient=float(initial_local.flatten(1).norm(dim=1).mean()),
        momentum_roundoff=roundoff, target_roundoff=roundoff,
        uplink_roundoff=roundoff, downlink_roundoff=roundoff,
    )
    bound = fixed_momentum_bound(rates, oracle_defect=max(0.0, dmax), **assumptions)
    relative_bound = fixed_momentum_bound(rates, oracle_gain=gain_min, **assumptions)
    assert accum["tracking"] <= bound["frobenius_tracking_bound"]+1e-10
    assert accum["server_score"] <= bound["server_stationarity_bound"]+1e-10
    assert accum["broadcast_score"] <= bound["broadcast_stationarity_bound"]+1e-10
    assert accum["server_score"] <= relative_bound["server_stationarity_bound"]+1e-10
    assert gapmax <= bound["downlink_gap_bound"]+1e-10
    h0 = float(initial_local.flatten(1).norm(dim=1).mean())
    q = math.sqrt(0.75)
    u_bound = 0.0
    for t in range(steps):
        impulse = (1-theta)*beta*mu**t*h0+(theta*h0 if t == 0 else 0)
        assert target_increments[t] <= bound["target_increment_forcing"]+impulse+1e-10
        u_bound = q*(u_bound+bound["target_increment_forcing"]+impulse)+roundoff
        assert post_uplink[t] <= u_bound+1e-10
        assert noise_variance[t] <= sigma*sigma/workers*bound["filter_variance_gain"]+1e-12
    return dict(
        nesterov=nesterov, downlink=downlink, schedule=schedule, steps=steps,
        workers=workers, enumerated_paths=paths, checked=True,
        additive_roundoff_per_update=roundoff,
        expected_frobenius_tracking=accum["tracking"],
        expected_server_stationarity=accum["server_score"],
        expected_broadcast_stationarity=accum["broadcast_score"],
        max_broadcast_gap=gapmax, per_round_noise_variance=noise_variance, bound=bound,
        relative_oracle_bound=relative_bound,
        scope="entire finite noise law enumerated; numerical checks of the analytical proof, with direction bounds over all enumerated histories",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [enumerate_case(args.steps, args.workers, nes, down, schedule)
               for nes in (False, True) for down in ("exact", "sign")
               for schedule in ("constant", "cosine")]
    body = json.dumps(results, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
