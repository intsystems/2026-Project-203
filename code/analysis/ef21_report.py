"""Report EF21-P traces and schedule factors without a torch dependency.

Run from code/: python -m analysis.ef21_report --help
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def schedule_values(kind, steps, lr=1.0, final_ratio=0.1, stable_fraction=0.55, horizon=None):
    if steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("positive steps and finite positive lr are required")
    if not 0 <= final_ratio <= 1 or not 0 <= stable_fraction < 1:
        raise ValueError("invalid decay fraction")
    horizon = steps if horizon is None else horizon
    if horizon < 1:
        raise ValueError("positive decay horizon required")
    if kind == "constant":
        return [lr]*steps
    if kind == "cosine":
        # utils.cosine_lr and centralized CosineAnnealingLR: t=0,...,T-1.
        return [lr*0.5*(1+math.cos(math.pi*t/steps)) for t in range(steps)]
    if kind == "stable-linear":
        # nanoGPT get_lr caps progress during its iteration extension.
        return [lr*(1-(1-final_ratio)*max(0.0, (min(0.9999, t/horizon)-stable_fraction)/
                                                  (1-stable_fraction))) for t in range(steps)]
    raise ValueError("unknown schedule")


def schedule_summary(values, momentum=0.95):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("nonempty finite nonnegative schedule required")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in [0,1)")
    mass, square_mass = math.fsum(values), math.fsum(x*x for x in values)
    if mass <= 0:
        raise ValueError("schedule has zero mass")
    beta = 1-momentum
    ema_gain = beta/(2-beta)
    nesterov_gain = beta*beta*(2-beta)**2 + beta*momentum**4/(2-beta)
    return dict(
        steps=len(values), sum_eta=mass, sum_eta_squared=square_mass,
        initial_gap_multiplier=1/mass, smoothness_multiplier=square_mass/mass,
        effective_steps=mass*mass/square_mass, beta=beta,
        frozen_query_ema_variance_gain=ema_gain,
        frozen_query_nesterov_variance_gain=nesterov_gain,
        interpretation="schedule algebra and frozen-query filter gains; not a stationarity certificate",
    )


def _number(row, name):
    value = row.get(name, "")
    if value in (None, ""):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def analyze_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        # Historical nanoGPT data include separate gradient-block-only rows.
        if row.get("schema") == "ef21p-v1":
            groups[(row.get("run_id", ""), row["layer"])].append(row)
        elif _number(row, "alpha_dn") is not None:
            groups[(row.get("run_id", ""), row["parameter"])].append(row)
    output = []
    for (run, layer), group in sorted(groups.items()):
        group.sort(key=lambda r: int(r["step"]))
        full = all(r.get("schema") == "ef21p-v1" for r in group)
        alpha_key = "alpha_input" if full else "alpha_dn"
        alphas = [(_number(r, alpha_key), int(r["step"])) for r in group]
        alphas = [(a, t) for a, t in alphas if a is not None]
        item = dict(run_id=run, layer=layer, samples=len(group),
                    first_step=int(group[0]["step"]), last_step=int(group[-1]["step"]),
                    observation="per_block_trace" if full else "sampled_group_medians",
                    alpha_min=min((a for a, _ in alphas), default=None),
                    alpha_last=alphas[-1][0] if alphas else None,
                    stationarity_certificate=False)
        if not full:
            item.update(last_relative_gap=_number(group[-1], "lag_XW"),
                        tracking_envelope=None,
                        limitation="sampled medians are not a per-layer uniform contraction bound; missing rounds prevent recurrence reconstruction")
        else:
            ids = [int(r["step"]) for r in group]
            valid = (ids == list(range(ids[0], ids[-1]+1)) and
                     all(int(r.get("sample_every", 1)) == 1 for r in group) and
                     all(str(r.get("finite")).lower() == "true" for r in group))
            envelope = math.sqrt(_number(group[0], "gap_before_sq") or 0)
            max_excess, max_closure, max_identity = 0.0, 0.0, 0.0
            previous_post = None
            for r in group:
                q, move = _number(r, "observed_q"), _number(r, "server_step_sq")
                before, post = _number(r, "gap_before_sq"), _number(r, "post_gap_sq")
                residual, cross = _number(r, "residual_sq"), _number(r, "gap_step_inner")
                if None in (q, move, before, post, residual, cross):
                    valid = False
                    continue
                if min(move, before, post, residual) < 0 or q < 0:
                    raise ValueError("negative norm energy or contraction multiplier")
                if previous_post is not None and not math.isclose(before, previous_post, rel_tol=1e-6, abs_tol=1e-20):
                    valid = False
                previous_post = post
                envelope = q*(envelope+math.sqrt(move))
                max_excess = max(max_excess, math.sqrt(post)-envelope)
                max_closure = max(max_closure, abs(residual-before-move-2*cross)/max(residual, before+move, 1e-300))
                defect = _number(r, "identity_defect")
                if defect is not None:
                    max_identity = max(max_identity, abs(defect))
            item.update(complete_segment=valid, tracking_envelope=envelope if valid else None,
                        max_envelope_excess=max_excess if valid else None,
                        max_energy_closure_relative_error=max_closure,
                        max_scaled_sign_identity_defect=max_identity,
                        last_relative_gap=_number(group[-1], "relative_gap"),
                        limitation="observed multipliers give a retrospective tracking bound, not future contraction or true-gradient control")
            for field in ("direction_operator", "oracle_pairing_defect"):
                measured = [_number(row, field) for row in group]
                item["max_"+field] = max((x for x in measured if x is not None), default=None)
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--schedule", choices=["constant", "cosine", "stable-linear"], default="cosine")
    parser.add_argument("--steps", type=int, default=2330)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--schedule-horizon", type=int, help="Decay horizon before nanoGPT's iteration extension")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"generated_schedule_example": schedule_summary(
        schedule_values(args.schedule, args.steps, args.lr, horizon=args.schedule_horizon), args.momentum),
        "schedule_parameters": dict(kind=args.schedule, base_lr=args.lr, steps=args.steps,
                                    decay_horizon=args.schedule_horizon or args.steps,
                                    filter_momentum_example=args.momentum)}
    if args.trace:
        with args.trace.open(newline="", encoding="utf-8") as f:
            result["compression"] = analyze_rows(list(csv.DictReader(f)))
    body = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
