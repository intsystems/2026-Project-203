"""One-command overnight run for the centralized ResNet-18 study.

    cd code
    python3 -m centralized.overnight --device cuda:0 --budget-hours 8 --download

Watch the first ~6 minutes: it runs the CPU test suite, prints the per-layer
learning-rate table, times two real epochs on *your* GPU, and then prints a
schedule saying exactly which phases fit in the budget. Once the schedule appears
you can go to bed. In the morning read ``results/overnight/REPORT.md``.

Design for an unattended night
------------------------------
* **Budget-aware.** Every phase is costed from the *measured* epoch time, and the
  deadline is checked before each individual job, so the run stops cleanly instead
  of being killed mid-way.
* **Crash-isolated.** Each job is a subprocess; a failure is logged and the run
  continues. One diverging learning rate cannot take down the night.
* **Resumable and incremental.** State is written to
  ``results/overnight/state.json`` after every job, and ``--resume`` skips
  everything already done.
* **Priority-ordered.** The phases are ordered so that stopping early still leaves
  a usable result: the alpha decision first (cheapest, most theoretical value),
  then eta_0 for every method, then finals **seed-major** -- seed 0 for all ten
  methods before seed 1 -- so an interrupted night yields a complete 1-seed table
  rather than a partial 3-seed one.

Phases
------
0. ``preflight``  CPU tests, scaling tables, 2-epoch timing.
1. ``gain``       ``--log-gain`` runs: does the accumulated update grow like
                  ``sqrt(t)`` (favouring ``unit-gain``) or ``t`` (favouring ``mup``)?
2. ``alpha``      sweep ``power:ALPHA`` x eta_0 -> pick the exponent.
3. ``aux``        is the optimal ``lr_aux`` method-independent? (off by default)
4. ``lr``         eta_0 per method under the chosen rule, equal budget each.
5. ``final``      full 50k training runs at the tuned values, seed-major.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from centralized.train import LMO_FAMILY
from common.lr_scaling import FAMILY_SIGN, describe_rule, resolve_rule
from common.utils import results_root
from centralized.tune import (ALL_METHODS, LEGACY_ANCHORS, ROOT, SCALED_ANCHOR_BOOST,
                              best_of, boundary_warning, geom_grid, run_one)

OUT_DIR = results_root() / "overnight"
STATE_PATH = OUT_DIR / "state.json"
REPORT_PATH = OUT_DIR / "REPORT.md"

#: Startup cost of one subprocess (imports, CUDA init, dataset scan), seconds.
JOB_OVERHEAD_S = 35.0

_stop = {"requested": False}


def _handle_sigint(signum, frame):        # pragma: no cover - interactive
    _stop["requested"] = True
    print("\n[interrupt] finishing the current job, then stopping cleanly...", flush=True)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def load_state() -> Dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] {STATE_PATH} is corrupt; starting fresh")
    return {"jobs": {}, "phases": {}, "started": None, "epoch_seconds": None}


def save_state(state: Dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Budget:
    """Deadline bookkeeping, with a per-job feasibility check."""

    def __init__(self, hours: float):
        self.start = time.time()
        self.deadline = self.start + hours * 3600.0
        self.hours = hours

    def left(self) -> float:
        return self.deadline - time.time()

    def fits(self, seconds: float) -> bool:
        return self.left() > seconds

    def report(self) -> str:
        left = max(0.0, self.left())
        return f"{timedelta(seconds=int(time.time() - self.start))} elapsed, " \
               f"{timedelta(seconds=int(left))} left"


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def preflight(args) -> Optional[float]:
    """Run the tests, print the scaling table, and time two real epochs.

    Returns the measured seconds per epoch, or ``None`` if the timing run failed
    (in which case nothing else should be attempted).
    """
    print("=" * 78)
    print(f"[{stamp()}] PREFLIGHT")
    print("=" * 78)

    print("\n[1/3] CPU test suite (no GPU, no downloads)")
    proc = subprocess.run([sys.executable, "-m", "tests.test_code"], cwd=ROOT,
                          capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1:] or ["(no output)"]
    print(f"      {tail[0]}")
    if proc.returncode != 0:
        print("      FAILED -- fix this before running overnight. Failing tests:")
        for line in proc.stdout.splitlines():
            if line.startswith(("FAIL", "ERROR")):
                print(f"        {line}")
        if not args.force:
            return None
        print("      --force given, continuing anyway")

    print(f"\n[2/3] per-layer learning-rate multipliers, rule '{args.lr_scaling}'")
    rule = resolve_rule(args.lr_scaling)
    shapes = [("conv1", (64, 3, 3, 3)), ("layer1.conv", (64, 64, 3, 3)),
              ("layer2.conv", (128, 128, 3, 3)), ("layer2.downsample", (128, 64, 1, 1)),
              ("layer3.conv", (256, 256, 3, 3)), ("layer4.conv", (512, 512, 3, 3)),
              ("layer4.downsample", (512, 256, 1, 1))]
    for family in (FAMILY_SIGN, "lmo"):
        print(describe_rule(rule, family, shapes))

    print(f"\n[3/3] timing 2 real epochs of {args.model} on {args.device}")
    r = run_one(_tune_args(args), lr=LEGACY_ANCHORS["muon"], lr_aux=args.lr_aux,
                lr_scaling=args.lr_scaling, method="muon", epochs=2,
                tag="preflight_timing")
    if r is None or "epoch_seconds" not in r:
        print("      timing run FAILED -- see the log above. Aborting.")
        return None
    sec = r["epoch_seconds"]
    print(f"      measured {sec:.1f}s per epoch")
    return sec


def _tune_args(args) -> Namespace:
    """The Namespace ``centralized.tune.run_one`` expects."""
    return Namespace(
        dataset=args.dataset, model=args.model, batch_size=args.batch_size,
        momentum=args.momentum, weight_decay=args.weight_decay,
        head_adamw=args.head_adamw, last_k=args.last_k, val_seed=args.val_seed,
        seed=args.seed, device=args.device, data=args.data,
        num_workers=args.num_workers,
        nondeterministic=not args.deterministic,
        download=False,   # preflight fetches the data once, up front
    )


def job_cost(sec_per_epoch: float, epochs: int, split: str = "tune") -> float:
    """Estimated wall-clock of one run.

    ``sec_per_epoch`` is measured on the tuning split (45k train + 15k of eval).
    A ``full``-split run trains on 50k and evaluates only the 10k test set, which
    is close enough to the same per-epoch cost; the ``+1`` is the epoch-0
    evaluation.
    """
    return JOB_OVERHEAD_S + sec_per_epoch * (epochs + 1)


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------


def phase_gain(args, state, budget, sec) -> None:
    """Measure the growth of the accumulated update's gain.

    ``sqrt(t)`` growth means successive sign steps stay incoherent (alpha = 1/2,
    ``unit-gain``); linear ``t`` growth means they align (alpha = 1, ``mup``). This
    is the one measurement that decides the exponent without reference to accuracy.
    """
    for method in ("signmuon", "muon"):
        tag = f"gain_{method}"
        if tag in state["jobs"]:
            continue
        cost = job_cost(sec, args.gain_epochs)
        if not budget.fits(cost) or _stop["requested"]:
            return
        print(f"[{stamp()}] gain/{method} ({args.gain_epochs} ep, "
              f"~{cost/60:.0f} min) | {budget.report()}")
        base = LEGACY_ANCHORS[method]
        if LMO_FAMILY[method].family == FAMILY_SIGN:
            base *= SCALED_ANCHOR_BOOST.get(args.lr_scaling, 1.0)
        r = run_one(_tune_args(args), lr=base, lr_aux=args.lr_aux,
                    lr_scaling=args.lr_scaling, method=method,
                    epochs=args.gain_epochs, tag=tag, extra=("--log-gain",))
        state["jobs"][tag] = r
        save_state(state)


def phase_alpha(args, state, budget, sec) -> Optional[float]:
    """Sweep ``power:ALPHA`` x eta_0 on one sign-family method; return the winner."""
    method = args.alpha_method
    results: Dict[str, List] = state["phases"].setdefault("alpha", {})
    for alpha in args.alpha_grid:
        rule = f"power:{alpha:g}"
        # eta_0 anchor scales like fan_in^alpha at the median layer (fan_in ~ 1152).
        base = LEGACY_ANCHORS[method] * (1152.0 ** alpha)
        for lr in geom_grid(base, decades=args.alpha_decades, points=args.alpha_points):
            tag = f"alpha{alpha:g}_{method}_lr{lr:.4g}"
            if tag in state["jobs"]:
                continue
            cost = job_cost(sec, args.tune_epochs)
            if not budget.fits(cost) or _stop["requested"]:
                return _alpha_verdict(results)
            print(f"[{stamp()}] alpha={alpha:g} {method} (~{cost/60:.0f} min) "
                  f"| {budget.report()}")
            r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux, lr_scaling=rule,
                        method=method, epochs=args.tune_epochs, tag=tag)
            state["jobs"][tag] = r
            results.setdefault(f"{alpha:g}", []).append(r)
            save_state(state)
    return _alpha_verdict(results)


def _alpha_verdict(results: Dict[str, List]) -> Optional[float]:
    scored = []
    for alpha, runs in results.items():
        best = best_of(runs)
        if best:
            scored.append((float(alpha), best))
    if not scored:
        return None
    scored.sort(key=lambda kv: -kv[1]["val_acc"])
    print("\n  --- alpha verdict ---")
    for alpha, best in scored:
        print(f"    alpha={alpha:<4g} val {best['val_acc']:.2f}%  (lr={best['lr']:.4g})")
    if len(scored) > 1 and scored[0][1]["val_acc"] - scored[1][1]["val_acc"] < 0.3:
        print("    gap < 0.3% -> not decisive on this architecture (ResNet-18 has "
              "little shape diversity); prefer the --log-gain measurement.")
    return scored[0][0]


def phase_lr(args, state, budget, sec, rule: str) -> Dict[str, Dict]:
    """Tune eta_0 per method under ``rule``, identical budget for every method."""
    out: Dict[str, Dict] = state["phases"].setdefault("lr", {})
    for method, scale_baselines in _lr_jobs(args):
        key = f"{method}{'+scaled' if scale_baselines else ''}"
        base = LEGACY_ANCHORS[method]
        cls = LMO_FAMILY.get(method)
        if (cls is not None and cls.family == FAMILY_SIGN) or scale_baselines:
            base *= SCALED_ANCHOR_BOOST.get(rule, 1.0)
        grid = geom_grid(base, decades=args.lr_decades, points=args.lr_points)
        runs = out.setdefault(key, {"runs": [], "grid": grid})["runs"]
        for lr in grid:
            tag = f"lr_{key.replace('+', '_')}_{rule.replace(':', '')}_{lr:.4g}"
            if tag in state["jobs"]:
                continue
            cost = job_cost(sec, args.tune_epochs)
            if not budget.fits(cost) or _stop["requested"]:
                return out
            print(f"[{stamp()}] lr/{key} (~{cost/60:.0f} min) | {budget.report()}")
            extra = ("--scale-baselines",) if scale_baselines else ()
            r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux,
                        lr_scaling=rule, method=method, epochs=args.tune_epochs,
                        tag=tag, extra=extra)
            state["jobs"][tag] = r
            runs.append(r)
            save_state(state)
        best = best_of(runs)
        if best:
            out[key]["best"] = best
            warn = boundary_warning(best, grid)
            print(f"  BEST {key}: eta_0={best['lr']:.6g}  val {best['val_acc']:.2f}%")
            if warn:
                print(warn)
                out[key]["boundary"] = warn.strip()
    return out


def _lr_jobs(args):
    """``(method, scale_baselines)`` pairs. SGD/Adam are run both ways."""
    for method in args.methods:
        yield method, False
        if method in ("sgd", "adam") and not args.skip_baseline_variants:
            yield method, True


def phase_final(args, state, budget, sec, rule: str, tuned: Dict[str, Dict]) -> None:
    """Full-50k runs at the tuned eta_0, seed-major so a cut night still completes
    a whole 1-seed table."""
    for seed in args.final_seeds:
        for method, scaled in _lr_jobs(args):
            key = f"{method}{'+scaled' if scaled else ''}"
            best = tuned.get(key, {}).get("best")
            if not best:
                continue
            tag = f"{key.replace('+', '_')}_{rule.replace(':', '')}_s{seed}"
            if tag in state["jobs"]:
                continue
            cost = job_cost(sec, args.final_epochs, split="full")
            if not budget.fits(cost) or _stop["requested"]:
                return
            print(f"[{stamp()}] final/{key} seed {seed} ({args.final_epochs} ep, "
                  f"~{cost/60:.0f} min) | {budget.report()}")
            r = run_one(_tune_args(args), lr=best["lr"], lr_aux=args.lr_aux,
                        lr_scaling=rule, method=method, epochs=args.final_epochs,
                        tag=tag, split="full", seed=seed,
                        extra=("--scale-baselines",) if scaled else ())
            state["jobs"][tag] = r
            state["phases"].setdefault("final", {})[tag] = r
            save_state(state)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_report(args, state, budget, sec, rule, alpha, tuned) -> None:
    lines = [
        "# Overnight run report",
        "",
        f"* started `{state.get('started')}`, wall clock {budget.report()}",
        f"* device `{args.device}`, {args.model} / {args.dataset}, "
        f"measured **{sec:.1f} s/epoch**" if sec else "* timing unavailable",
        f"* scaling rule **`{rule}`**, `--head-adamw {args.head_adamw}`, "
        f"lr_aux = {args.lr_aux:g}, momentum {args.momentum:g}, "
        f"weight decay {args.weight_decay:g}",
        f"* tuning: {args.tune_epochs} epochs on the 45k/5k split, selected on "
        f"**val_acc** (tail mean of {args.last_k}); the test set was not used for "
        f"any decision",
        "",
    ]

    gain = {k: v for k, v in state["jobs"].items() if k.startswith("gain_") and v}
    if gain:
        lines += ["## Gain diagnostic", "",
                  "Growth of `||X_t - X_0||_F / sqrt(fan_out)`. Per-layer detail is at "
                  "the end of each log; the recorded series are `gain_min/median/max`.",
                  ""]
        for k, v in gain.items():
            lines.append(f"* `{k}` -> log `{Path(v['log']).name}`, "
                         f"val {v.get('val_acc', float('nan')):.2f}%")
        lines += ["",
                  "**How to read it:** fit `log(gain_median)` against `log(epoch)`. A "
                  "slope near 0.5 supports `unit-gain` (alpha=1/2); near 1.0 supports "
                  "`mup` (alpha=1).", ""]

    if alpha is not None:
        lines += ["## Alpha sweep", "",
                  f"Best exponent on `{args.alpha_method}`: **alpha = {alpha:g}**.", ""]
        for a, runs in sorted(state["phases"].get("alpha", {}).items()):
            b = best_of(runs)
            if b:
                lines.append(f"* alpha={a}: val {b['val_acc']:.2f}% at "
                             f"eta_0={b['lr']:.4g}")
        lines.append("")

    if tuned:
        lines += ["## Tuned eta_0 (validation)", "",
                  "| method | eta_0 | val acc | configs | note |",
                  "| :--- | ---: | ---: | ---: | :--- |"]
        for key, d in tuned.items():
            b = d.get("best")
            if not b:
                continue
            lines.append(f"| `{key}` | {b['lr']:.6g} | {b['val_acc']:.2f}% | "
                         f"{len(d['runs'])} | {d.get('boundary', '')} |")
        lines += ["", "A `BOUNDARY` note means the optimum sat on a grid endpoint: "
                      "extend the grid and re-run that method before reporting it.", ""]

    finals = state["phases"].get("final", {})
    if finals:
        lines += ["## Final runs (full 50k, test set)", "",
                  "| run | test acc (tail mean) | epochs to target |",
                  "| :--- | ---: | ---: |"]
        for tag, v in sorted(finals.items()):
            if not v:
                continue
            lines.append(f"| `{tag}` | {v.get('test_acc', float('nan')):.2f}% | "
                         f"{v.get('epochs_to_target', '-')} |")
        lines += ["", "Aggregate across seeds with "
                      "`python3 -m aggregate --root results/centralized`.", ""]

    done = sum(1 for v in state["jobs"].values() if v)
    failed = sum(1 for v in state["jobs"].values() if not v)
    lines += ["## Next steps", "",
              f"{done} jobs completed, {failed} failed. Resume with:", "",
              "```bash",
              f"python3 -m centralized.overnight --device {args.device} "
              f"--budget-hours 8 --resume",
              "```", ""]
    if not finals:
        lines += ["No final runs completed. With the tuned eta_0 above, launch them "
                  "directly:", "",
                  "```bash",
                  f"python3 -m centralized.main --dataset {args.dataset} "
                  f"--model {args.model} --epochs {args.final_epochs} \\",
                  f"    --optimizer <method> --lr-scaling {rule} "
                  f"--head-adamw {args.head_adamw} \\",
                  f"    --lr <eta_0> --lr-aux {args.lr_aux:g} --seed 0 "
                  f"--device {args.device}",
                  "```", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "=" * 78)
    print("\n".join(lines))
    print("=" * 78)
    print(f"[{stamp()}] report written to {REPORT_PATH}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--budget-hours", type=float, default=8.0)
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--download", action="store_true",
                   help="Download CIFAR-10 if missing (do this on the first run)")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--lr-aux", type=float, default=1e-3)
    p.add_argument("--head-adamw", type=str, default="always",
                   choices=["auto", "always", "never"])
    p.add_argument("--lr-scaling", type=str, default="unit-gain")
    p.add_argument("--last-k", type=int, default=5)
    p.add_argument("--val-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers for training (Windows spawn is costly)")
    p.add_argument("--deterministic", action="store_true",
                   help="Disable cuDNN autotuning: bitwise reproducible but slower. "
                        "Off by default -- with multiple seeds we measure seed "
                        "variation, not bitwise determinism")

    p.add_argument("--methods", nargs="*", default=ALL_METHODS)
    p.add_argument("--phases", nargs="*",
                   default=["gain", "alpha", "lr", "final"],
                   choices=["gain", "alpha", "aux", "lr", "final"],
                   help="Which phases to run, in order (default omits 'aux')")
    p.add_argument("--tune-epochs", type=int, default=6,
                   help="Short proxy horizon for ranking learning rates. "
                        "Raise to 10-15 if you have more than 8 h")
    p.add_argument("--gain-epochs", type=int, default=20)
    p.add_argument("--final-epochs", type=int, default=30,
                   help="Epoch budget for the full-50k runs. The paper uses "
                        "75; 30 is enough to rank the methods overnight")
    p.add_argument("--final-seeds", nargs="*", type=int, default=[0],
                   help="Seed-major: all methods at seed 0, then seed 1, ... "
                        "so an interrupted night still leaves a complete table")
    p.add_argument("--lr-points", type=int, default=4)
    p.add_argument("--lr-decades", type=float, default=1.0)
    p.add_argument("--alpha-grid", nargs="*", type=float, default=[0.0, 0.5, 1.0],
                   help="0 is the paper's current global LR (the control), "
                        "1/2 is unit-gain, 1 is mup")
    p.add_argument("--alpha-points", type=int, default=5)
    p.add_argument("--alpha-decades", type=float, default=1.0,
                   help="Each alpha needs its own eta_0 optimum found, or the "
                        "comparison is confounded by a badly chosen rate")
    p.add_argument("--alpha-method", type=str, default="signmuon")
    p.add_argument("--skip-baseline-variants", action="store_true",
                   help="Do NOT additionally tune SGD/Adam under the sign rule "
                        "(by default both parameterizations are run and the better "
                        "one reported, so neither baseline can be accused of a handicap)")

    p.add_argument("--resume", action="store_true", help="Skip jobs already recorded")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Continue even if the CPU test suite fails")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the schedule and exit without training")
    return p.parse_args()


def print_schedule(args, sec: float, budget: Budget) -> None:
    n_alpha = len(args.alpha_grid) * args.alpha_points
    n_lr = sum(1 for _ in _lr_jobs(args)) * args.lr_points
    n_final = sum(1 for _ in _lr_jobs(args)) * len(args.final_seeds)
    plan = [
        ("gain", 2, args.gain_epochs),
        ("alpha", n_alpha, args.tune_epochs),
        ("lr", n_lr, args.tune_epochs),
        ("final", n_final, args.final_epochs),
    ]
    print("\n" + "=" * 78)
    print(f"SCHEDULE  (budget {args.budget_hours:g} h, measured {sec:.1f} s/epoch)")
    print("=" * 78)
    print(f"  {'phase':<8}{'jobs':>6}{'epochs':>8}{'hours':>8}{'cumulative':>12}   fits?")
    cum = 0.0
    for name, n, ep in plan:
        if name not in args.phases:
            print(f"  {name:<8}{'-':>6}{'-':>8}{'-':>8}{'-':>12}   skipped")
            continue
        hrs = n * job_cost(sec, ep) / 3600.0
        cum += hrs
        print(f"  {name:<8}{n:>6}{ep:>8}{hrs:>8.1f}{cum:>12.1f}   "
              f"{'yes' if cum <= args.budget_hours else 'NO -- will be cut'}")
    print("=" * 78)
    if cum > args.budget_hours:
        final_h = (n_final * job_cost(sec, args.final_epochs) / 3600.0
                   if "final" in args.phases else 0.0)
        tuning_h = cum - final_h
        spare_s = (args.budget_hours - tuning_h) * 3600.0
        print(f"  Total {cum:.1f} h exceeds the {args.budget_hours:g} h budget. The run "
              f"stops cleanly at the deadline;")
        print(f"  phases are ordered so that what completes is still usable, and "
              f"--resume continues tomorrow.")
        if final_h > 0 and spare_s > 0:
            n_fit = int(spare_s // job_cost(sec, args.final_epochs))
            fit_ep = int((spare_s / n_final - JOB_OVERHEAD_S) / sec - 1)
            print(f"  Tuning alone needs {tuning_h:.1f} h, leaving {spare_s / 3600:.1f} h: "
                  f"about {n_fit} of {n_final} final runs will")
            print(f"  complete (seed-major, so whole methods finish rather than "
                  f"fragments).")
            if fit_ep >= 10:
                print(f"  To fit all {n_final} instead, use --final-epochs {fit_ep}.")
        else:
            print(f"  Tuning alone needs {tuning_h:.1f} h. Reduce --lr-points or "
                  f"--tune-epochs, or raise --budget-hours.")
    else:
        print(f"  Total {cum:.1f} h fits with "
              f"{args.budget_hours - cum:.1f} h to spare.")
    print(flush=True)


def main() -> None:
    args = get_args()
    signal.signal(signal.SIGINT, _handle_sigint)

    state = load_state() if args.resume else {"jobs": {}, "phases": {}}
    state.setdefault("jobs", {})
    state.setdefault("phases", {})
    state["started"] = state.get("started") or datetime.now().isoformat(timespec="seconds")
    budget = Budget(args.budget_hours)

    if args.download:
        # Fetch once, here, rather than letting every child run pass --download:
        # a partially written archive shared by many runs is a bad failure mode.
        print(f"[{stamp()}] fetching {args.dataset} into {args.data} ...")
        from centralized.data import build_loaders
        build_loaders(args.dataset, args.data, batch_size=args.batch_size,
                      download=True, split="tune", num_workers=0)
        print(f"[{stamp()}] dataset ready")

    sec = state.get("epoch_seconds") if args.resume else None
    if sec is None:
        sec = preflight(args)
        if sec is None:
            print("Preflight failed; nothing was trained.")
            sys.exit(1)
        state["epoch_seconds"] = sec
        save_state(state)
    else:
        print(f"[{stamp()}] resuming with the previously measured {sec:.1f} s/epoch "
              f"({len(state['jobs'])} jobs already recorded)")

    print_schedule(args, sec, budget)
    if args.preflight_only or args.dry_run:
        print("Stopping here as requested (--preflight-only / --dry-run).")
        return

    rule = args.lr_scaling
    alpha = None
    tuned: Dict[str, Dict] = {}

    if "gain" in args.phases:
        print(f"\n{'=' * 78}\n[{stamp()}] PHASE gain\n{'=' * 78}")
        phase_gain(args, state, budget, sec)
    if "alpha" in args.phases and not _stop["requested"]:
        print(f"\n{'=' * 78}\n[{stamp()}] PHASE alpha\n{'=' * 78}")
        alpha = phase_alpha(args, state, budget, sec)
        if alpha is not None:
            # Use the measured exponent for everything downstream.
            rule = {0.0: "legacy", 0.5: "unit-gain", 1.0: "mup"}.get(
                round(alpha, 6), f"power:{alpha:g}")
            print(f"\n  -> using rule '{rule}' for the remaining phases")
            state["phases"]["chosen_rule"] = rule
            save_state(state)
    if "lr" in args.phases and not _stop["requested"]:
        print(f"\n{'=' * 78}\n[{stamp()}] PHASE lr  (rule '{rule}')\n{'=' * 78}")
        tuned = phase_lr(args, state, budget, sec, rule)
    if "final" in args.phases and not _stop["requested"]:
        print(f"\n{'=' * 78}\n[{stamp()}] PHASE final\n{'=' * 78}")
        phase_final(args, state, budget, sec, rule, tuned or state["phases"].get("lr", {}))

    save_state(state)
    write_report(args, state, budget, sec, rule, alpha,
                 tuned or state["phases"].get("lr", {}))


if __name__ == "__main__":
    main()
