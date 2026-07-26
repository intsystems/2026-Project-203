"""One-command overnight run for the centralized ResNet-18 study.

    cd code
    python3 -m centralized.overnight --device cuda:0 --budget-hours 0 \
        --final-seeds 0 1 2 --download

Watch the first ~6 minutes: it runs the CPU test suite, prints the per-layer
learning-rate table, times two real epochs on *your* GPU, and then prints a
schedule with a finish time per phase. Once the schedule appears you can leave it.

``--budget-hours 0`` means **no deadline**: every phase runs to completion and only
Ctrl-C stops the run. ``results/overnight/REPORT.md`` is rewritten after every phase
and after every final run, so it can be read at any time *without* stopping
anything. Ctrl-C stops cleanly and writes the report; a second Ctrl-C exits at once.

Design for an unattended night
------------------------------
* **Budget-aware.** Every phase is costed from the *measured* epoch time, and the
  deadline is checked before each individual job, so a bounded run stops cleanly
  instead of being killed mid-way. ``--budget-hours 0`` removes the deadline.
* **Crash-isolated.** Each job is a subprocess; a failure is logged and the run
  continues. One diverging learning rate cannot take down the night.
* **Resumable and incremental.** State is written to
  ``results/overnight/state.json`` after every job, and ``--resume`` skips
  everything already done.
* **Priority-ordered.** The phases are ordered so that stopping early costs the
  least: the diagnostic and eta_0 first, then the headline table, then the two
  ablations. Finals are **seed-major** -- every parameterization at seed 0 before
  any of them reaches seed 1 -- so an early stop yields a complete 1-seed table
  rather than a fragmentary 3-seed one. Put another way, a night that runs short
  loses the ablations and the error bars, never the table itself.

Every learning rate tried is a **1-2-5 lattice point** (``tune.round_grid``), so a
tuned value is quotable as ``0.02`` rather than ``0.0172354775``, grid extension
stays on the same lattice however many rounds it takes, and two methods anchored
at slightly different places search the *same* grid.

Phases
------
0. ``preflight``  CPU tests, scaling tables, 2-epoch timing.
1. ``gain``       ``--log-gain`` runs at a CONSTANT step size: does the accumulated
                  update grow like ``sqrt(t)`` (alpha = 1/2, ``unit-gain``) or like
                  ``t`` (alpha = 1, ``mup``)? Annealing would let the accumulation
                  saturate and the fit would measure the schedule, so this phase
                  passes ``--constant-lr`` and its runs must never be compared with
                  scheduled ones.
2. ``lr``         eta_0 per method under the chosen rule, equal budget each.
3. ``final``      full 50k training runs at the tuned values, seed-major.
4. ``verify``     re-run the top rates at the FINAL horizon: is the short-horizon
                  ranking horizon-stable? (the assumption a short proxy makes)
5. ``wd``         re-run the best few methods with weight decay switched on. The
                  primary table is unregularized -- that is the setting the theorems
                  analyse, the one the nanoGPT record #40 config uses, and the one
                  Mishra et al.'s own sweep selects -- so this phase supplies the
                  regularized number and shows whether the ordering moves.

``alpha`` (sweep ``power:ALPHA`` x eta_0) is available but no longer in the default
list. At a single width alpha is largely absorbed into eta_0, so the sweep came out
flat to within 0.3% and cannot decide the exponent; phase 1 measures it directly.
Pass ``--phases gain alpha lr final verify wd`` to run it anyway.

One rule per results tree
-------------------------
``--lr-scaling`` and ``--weight-decay-mode`` are independent flags, and nothing
stops two invocations with different settings from writing into the same
``results/centralized``. They are recorded per run, so nothing is corrupted -- but
a later reader comparing across invocations can easily attribute a weight-decay
difference to the scaling rule. ``centralized.export_article`` refuses to compare
groups that differ in more than the rule; if you deliberately want two arms, keep
them in separate result trees.

The ``lr_aux`` study is a separate tool: ``python3 -m centralized.tune --stage aux``.
"""

from __future__ import annotations

import argparse
import json
import math
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
                              best_of, boundary_warning, canonical_tag, extend_grid,
                              round_grid, run_one)

OUT_DIR = results_root() / "overnight"
STATE_PATH = OUT_DIR / "state.json"
REPORT_PATH = OUT_DIR / "REPORT.md"

#: Startup cost of one subprocess (imports, CUDA init, dataset scan), seconds.
JOB_OVERHEAD_S = 35.0

_stop = {"requested": False}


def _handle_sigint(signum, frame):        # pragma: no cover - interactive
    if _stop["requested"]:
        raise KeyboardInterrupt("second interrupt -- exiting now")
    _stop["requested"] = True
    print("\n[interrupt] stopping cleanly and writing the report. "
          "Press Ctrl-C again to exit immediately.", flush=True)


def record(state: Dict, tag: str, result) -> None:
    """Record a job's outcome, then persist the state.

    A job cut short by Ctrl-C is deliberately **not** recorded: ``--resume`` skips
    any tag already present, so recording an interrupted run would silently retire
    it. Genuine failures *are* recorded, because a configuration that diverges will
    diverge again and re-running it only wastes the next night.
    """
    if result is None and _stop["requested"]:
        print(f"  (interrupted before {tag} finished -- not recorded, "
              f"--resume will retry it)")
        return
    state["jobs"][tag] = result
    save_state(state)


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
        self.hours = hours
        #: ``hours <= 0`` means no deadline: every phase runs to completion and only
        #: Ctrl-C stops the run.
        self.unlimited = hours <= 0
        self.deadline = float("inf") if self.unlimited else self.start + hours * 3600.0

    def left(self) -> float:
        return float("inf") if self.unlimited else self.deadline - time.time()

    def fits(self, seconds: float) -> bool:
        return self.unlimited or self.left() > seconds

    def report(self) -> str:
        elapsed = timedelta(seconds=int(time.time() - self.start))
        if self.unlimited:
            return f"{elapsed} elapsed, no deadline (Ctrl-C to stop)"
        return f"{elapsed} elapsed, {timedelta(seconds=int(max(0.0, self.left())))} left"


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
        weight_decay_mode=args.weight_decay_mode,
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
    for method in args.gain_methods:
        key = canonical_tag(f"gain_{method}", epochs=args.gain_epochs)
        if key in state["jobs"]:
            continue
        tag = f"gain_{method}"
        cost = job_cost(sec, args.gain_epochs)
        if not budget.fits(cost) or _stop["requested"]:
            return
        print(f"[{stamp()}] gain/{method} ({args.gain_epochs} ep, "
              f"~{cost/60:.0f} min) | {budget.report()}")
        base = LEGACY_ANCHORS[method]
        if LMO_FAMILY[method].family == FAMILY_SIGN:
            base *= SCALED_ANCHOR_BOOST.get(args.lr_scaling, 1.0)
        # A third of the anchor: this is a diagnostic, not a performance run, and a
        # constant (un-annealed) rate at the full anchor risks instability, which
        # would corrupt the very series we are trying to fit.
        base /= 3.0
        r = run_one(_tune_args(args), lr=base, lr_aux=args.lr_aux,
                    lr_scaling=args.lr_scaling, method=method,
                    epochs=args.gain_epochs, tag=tag,
                    extra=("--log-gain", "--constant-lr"))
        record(state, key, r)


def phase_alpha(args, state, budget, sec) -> Optional[float]:
    """Sweep ``power:ALPHA`` x eta_0 on one sign-family method; return the winner.

    Each exponent gets its **own** eta_0 grid, anchored at ``fan_in^alpha`` for the
    median layer, because the exponent changes what eta_0 means. Without that the
    comparison would just be measuring whose anchor happened to be luckier.
    """
    method = args.alpha_method
    results: Dict[str, List] = state["phases"].setdefault("alpha", {})
    for alpha in args.alpha_grid:
        rule = f"power:{alpha:g}"
        base = LEGACY_ANCHORS[method] * (1152.0 ** alpha)
        grid = round_grid(base, points=args.alpha_points)
        for lr in grid:
            tag = f"alpha{alpha:g}_{method}_lr{lr:.4g}"
            key = canonical_tag(tag, epochs=args.tune_epochs)
            if key in state["jobs"]:
                continue
            cost = job_cost(sec, args.tune_epochs)
            if not budget.fits(cost) or _stop["requested"]:
                return _alpha_verdict(results, state)
            print(f"[{stamp()}] alpha={alpha:g} {method} (~{cost/60:.0f} min) "
                  f"| {budget.report()}")
            r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux, lr_scaling=rule,
                        method=method, epochs=args.tune_epochs, tag=tag)
            if r:
                results.setdefault(f"{alpha:g}", []).append(r)
            record(state, key, r)          # persists the append too

        # An exponent whose own optimum sits on the edge of its grid would lose the
        # comparison for the wrong reason: flag it rather than ranking it.
        best_a = best_of(results.get(f"{alpha:g}", []))
        if best_a is not None:
            warn = boundary_warning(best_a, grid)
            if warn:
                print(f"  alpha={alpha:g}{warn}")
                state["phases"].setdefault("alpha_boundary", {})[f"{alpha:g}"] = warn.strip()
                save_state(state)
    return _alpha_verdict(results, state)


def _alpha_verdict(results: Dict[str, List],
                   state: Optional[Dict] = None) -> Optional[float]:
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
    flagged = (state or {}).get("phases", {}).get("alpha_boundary", {})
    if flagged:
        print(f"    NOTE: grid boundary hit for alpha in {sorted(flagged)} -- those "
              f"rows are not comparable until their grids are widened.")
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
        entry = out.setdefault(key, {"runs": [], "grid": []})
        runs = entry["runs"]
        # A resumed run inherits the grid an earlier round had already widened to,
        # so the extension budget is not spent twice on the same method.
        grid = entry.get("grid") or round_grid(base, points=args.lr_points)
        entry["grid"] = grid
        for extension in range(args.lr_extend_rounds + 1):
            for lr in grid:
                tag = f"lr_{key.replace('+', '_')}_{rule.replace(':', '')}_{lr:.4g}"
                jkey = canonical_tag(tag, epochs=args.tune_epochs)
                if jkey in state["jobs"]:
                    continue
                cost = job_cost(sec, args.tune_epochs)
                if not budget.fits(cost) or _stop["requested"]:
                    return out
                print(f"[{stamp()}] lr/{key} (~{cost/60:.0f} min) | {budget.report()}")
                extra = ("--scale-baselines",) if scale_baselines else ()
                r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux,
                            lr_scaling=rule, method=method, epochs=args.tune_epochs,
                            tag=tag, extra=extra)
                record(state, jkey, r)
                if r:
                    runs.append(r)
            best = best_of(runs)
            if best is None:
                break
            entry["best"] = best
            warn = boundary_warning(best, grid)
            print(f"  BEST {key}: eta_0={best['lr']:.6g}  val {best['val_acc']:.2f}%")
            # An optimum on an endpoint is not an optimum: widen the grid and keep
            # going rather than reporting a rate the grid boundary chose for us.
            if not warn:
                entry.pop("boundary", None)
                break
            print(warn)
            entry["boundary"] = warn.strip()
            if extension == args.lr_extend_rounds:
                break
            grid = extend_grid(grid, low="LOW end" in warn,
                               points=args.lr_extend_points)
            entry["grid"] = grid
            print(f"  -> extending to [{min(grid):.4g}, {max(grid):.4g}]: "
                  f"{args.lr_extend_points} more points "
                  f"(round {extension + 1}/{args.lr_extend_rounds})")
            save_state(state)
    return out


def _lr_jobs(args):
    """``(method, scale_baselines)`` pairs.

    **Adam** is additionally run under the sign-family rule: its step is
    approximately a sign step (``|s_ij| ~ 1``) and muP does prescribe a per-layer
    rate for it, so reporting the better of the two removes any suspicion that the
    baseline was handicapped.

    **SGD deliberately is not.** Its step is ``eta * m``, whose Frobenius norm is
    data-dependent, so *no* static multiplier corresponds to the unit-gain criterion
    -- applying the sign rule to SGD would be an arbitrary rescaling dressed up as a
    parameterization. The honest comparator is SGD as practitioners run it.
    """
    for method in args.methods:
        yield method, False
        if method == "adam" and not args.skip_baseline_variants:
            yield method, True


def phase_verify(args, state, budget, sec, rule: str,
                 tuned: Dict[str, Dict]) -> Dict[str, Dict]:
    """Is the short-horizon learning-rate ranking horizon-stable?

    Tuning at ``--tune-epochs`` and reporting at ``--final-epochs`` assumes the
    ranking of learning rates does not depend on the horizon. That is an assumption,
    and it is checkable: re-run the top-``k`` rates of a couple of methods at the
    *final* horizon and see whether the winner moves.

    Deliberately still on the **tuning split**, so ``val_acc`` is the comparison
    metric at both horizons -- comparing short-horizon val against long-horizon test
    would confound the horizon with the split and the metric.

    If the winner holds, the proxy is validated and the paper can say so. If it
    moves, the short-horizon tuning is unreliable and must be redone longer; that is
    a result worth knowing before 12 final runs are built on it.
    """
    out: Dict[str, Dict] = state["phases"].setdefault("verify", {})
    for method in args.verify_methods:
        entry = tuned.get(method)
        if not entry or not entry.get("runs"):
            continue
        short = sorted((r for r in entry["runs"] if r and "val_acc" in r),
                       key=lambda r: -r["val_acc"])[:args.verify_top]
        if len(short) < 2:
            continue
        long_runs = out.setdefault(method, {"short": short, "long": []})["long"]
        for r in short:
            tag = f"verify_{method}_{rule.replace(':', '')}_{r['lr']:.4g}"
            key = canonical_tag(tag, epochs=args.final_epochs)
            if key in state["jobs"]:
                continue
            cost = job_cost(sec, args.final_epochs)
            if not budget.fits(cost) or _stop["requested"]:
                return out
            print(f"[{stamp()}] verify/{method} at {args.final_epochs} ep "
                  f"(~{cost/60:.0f} min) | {budget.report()}")
            res = run_one(_tune_args(args), lr=r["lr"], lr_aux=args.lr_aux,
                          lr_scaling=rule, method=method,
                          epochs=args.final_epochs, tag=tag)
            record(state, key, res)
            if res:
                long_runs.append(res)

    _verify_verdict(out, args)
    return out


def _verify_verdict(out: Dict[str, Dict], args) -> None:
    if not out:
        return
    print("\n  --- horizon-stability verdict ---")
    for method, d in out.items():
        short, long = d.get("short", []), d.get("long", [])
        if len(long) < 2:
            print(f"    {method}: not enough long-horizon runs to judge")
            continue
        s_order = [r["lr"] for r in sorted(short, key=lambda r: -r["val_acc"])]
        l_order = [r["lr"] for r in sorted(long, key=lambda r: -r["val_acc"])]
        same = abs(s_order[0] - l_order[0]) < 1e-12
        print(f"    {method}: best at {args.tune_epochs} ep = {s_order[0]:.4g}, "
              f"at {args.final_epochs} ep = {l_order[0]:.4g}  "
              f"-> {'STABLE' if same else 'MOVED'}")
        if not same:
            print(f"      the {args.tune_epochs}-epoch proxy picked the wrong rate "
                  f"for this method; prefer the {args.final_epochs}-epoch winner and "
                  f"treat the tuned table as provisional.")
        d["stable"] = bool(same)
        d["best_short"], d["best_long"] = s_order[0], l_order[0]


def phase_final(args, state, budget, sec, rule: str, tuned: Dict[str, Dict]) -> None:
    """Full-50k runs at the tuned eta_0, seed-major so a cut night still completes
    a whole 1-seed table."""
    for seed in args.final_seeds:
        for method, scaled in _lr_jobs(args):
            key = f"{method}{'+scaled' if scaled else ''}"
            best = tuned.get(key, {}).get("best")
            if not best:
                continue
            tag = f"{key.replace('+', '_')}_{rule.replace(':', '')}"
            jkey = canonical_tag(tag, epochs=args.final_epochs, split="full", seed=seed)
            if jkey in state["jobs"]:
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
            record(state, jkey, r)
            if r:
                state["phases"].setdefault("final", {})[jkey] = r
                save_state(state)
                refresh_report(args, state, budget, sec, rule, None, tuned)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _fit_gain_slope(job: Dict):
    """Least-squares slope of ``log(gain_median)`` against ``log(epoch)``.

    Returns ``(slope, r_squared, n_points)`` or ``None`` if the series is missing.
    Reads the run's own ``metrics.json`` rather than re-deriving anything, and skips
    epoch 0 where the accumulated update is identically zero.
    """
    path = job.get("metrics")
    if not path or not Path(path).exists():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    hist = payload.get("history", {})
    steps, vals = hist.get("steps") or [], hist.get("gain_median") or []
    pts = [(s, v) for s, v in zip(steps, vals)
           if s and v and isinstance(v, (int, float)) and v > 0]
    if len(pts) < 4:
        return None
    xs = [math.log(s) for s, _ in pts]
    ys = [math.log(v) for _, v in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2, n


def phase_wd(args, state, budget, sec, rule: str, tuned: Dict[str, Dict]) -> None:
    """Re-run the best few methods at the final horizon with decay switched on.

    The primary table uses ``--weight-decay 0``, so that the experiment and the
    theorems describe the same algorithm and no coupled/decoupled question arises.
    This phase supplies the other number: whether decay changes the *ordering*, and
    what the absolute accuracy is for readers who expect a regularized ResNet-18.
    The decay is decoupled -- the only well-posed choice for a scale-invariant step.
    """
    if args.wd_ablation <= 0 or not tuned:
        return
    ranked = sorted(((d["best"]["val_acc"], k, d["best"]["lr"])
                     for k, d in tuned.items() if d.get("best")), reverse=True)
    picks = ranked[:max(0, args.wd_ablation_top)]
    out = state["phases"].setdefault("wd", {})
    print(f"  decay ablation on {[k for _, k, _ in picks]} "
          f"at wd={args.wd_ablation:g} (decoupled)")
    for _, key, lr in picks:
        method = key.replace("+scaled", "")
        seed = args.final_seeds[0]
        tag = f"wd_{key.replace('+', '_')}_{rule.replace(':', '')}"
        jkey = canonical_tag(tag, epochs=args.final_epochs, split="full", seed=seed)
        if jkey in state["jobs"]:
            continue
        cost = job_cost(sec, args.final_epochs)
        if not budget.fits(cost) or _stop["requested"]:
            return
        print(f"[{stamp()}] wd/{key} (~{cost / 60:.0f} min) | {budget.report()}")
        # ``extra`` lands at the END of the child argv, so these override the values
        # the driver already put there.
        extra = (("--scale-baselines",) if key.endswith("+scaled") else ()) + (
            "--weight-decay", repr(args.wd_ablation),
            "--weight-decay-mode", "decoupled")
        r = run_one(_tune_args(args), lr=lr, lr_aux=args.lr_aux, lr_scaling=rule,
                    method=method, epochs=args.final_epochs, tag=tag, split="full",
                    seed=seed, extra=extra)
        record(state, jkey, r)
        if r:
            # The undecayed counterpart of this exact configuration, for the delta.
            ref_key = canonical_tag(f"{key.replace('+', '_')}_{rule.replace(':', '')}",
                                    epochs=args.final_epochs, split="full", seed=seed)
            ref = (state["phases"].get("final") or {}).get(ref_key) or {}
            out[key] = {"wd": args.wd_ablation, "lr": lr,
                        "test_acc": r.get("test_acc"),
                        "test_acc_no_decay": ref.get("test_acc")}
            save_state(state)


def build_report(args, state, budget, sec, rule, alpha, tuned) -> str:
    lines = [
        "# Overnight run report",
        "",
        f"* started `{state.get('started')}`, wall clock {budget.report()}",
        f"* device `{args.device}`, {args.model} / {args.dataset}, "
        f"measured **{sec:.1f} s/epoch**" if sec else "* timing unavailable",
        f"* scaling rule **`{rule}`**, `--head-adamw {args.head_adamw}`, "
        f"lr_aux = {args.lr_aux:g}, momentum {args.momentum:g}, "
        f"weight decay {args.weight_decay:g} ({args.weight_decay_mode})",
        f"* tuning: {args.tune_epochs} epochs on the 45k/5k split, selected on "
        f"**val_acc** (tail mean of {args.last_k}); the test set was not used for "
        f"any decision",
        "",
    ]

    gain = {k: v for k, v in state["jobs"].items() if k.startswith("gain_") and v}
    if gain:
        lines += ["## Gain diagnostic (measures the exponent directly)", "",
                  "Growth of the accumulated update's RMS gain "
                  "`||X_t - X_0||_F / sqrt(fan_out)` against the epoch count, at a "
                  "**constant** learning rate (with annealing the accumulation "
                  "saturates and the slope would measure the schedule instead).",
                  "",
                  "A slope near **0.5** means successive sign steps stay incoherent, "
                  "supporting `unit-gain` (alpha=1/2); near **1.0** means they align, "
                  "supporting `mup` (alpha=1).", "",
                  "| run | fitted slope | R^2 | epochs used | reading |",
                  "| :--- | ---: | ---: | ---: | :--- |"]
        for k, v in sorted(gain.items()):
            fit = _fit_gain_slope(v)
            if fit is None:
                lines.append(f"| `{k}` | - | - | - | series unavailable "
                             f"(see `{Path(v['log']).name}`) |")
                continue
            slope, r2, n = fit
            reading = ("incoherent -> alpha=1/2" if slope < 0.7 else
                       "aligned -> alpha=1" if slope > 0.85 else "between 1/2 and 1")
            lines.append(f"| `{k}` | {slope:.3f} | {r2:.3f} | {n} | {reading} |")
        lines.append("")

    # Read the verdict back from the state file rather than trusting the live
    # variable, so a resumed run or a --report-only rebuild still shows the sweep
    # it already paid for.
    if alpha is None:
        alpha = _alpha_verdict(state["phases"].get("alpha") or {})
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
                         f"{len(d.get('runs') or [])} | {d.get('boundary', '')} |")
        lines += ["", "A `BOUNDARY` note means the optimum sat on a grid endpoint: "
                      "extend the grid and re-run that method before reporting it.", ""]

    verify = state["phases"].get("verify", {})
    if verify:
        lines += ["## Horizon stability", "",
                  f"Top-{args.verify_top} rates re-run at {args.final_epochs} epochs "
                  f"(still on the tuning split, so `val_acc` is comparable):", "",
                  "| method | best @ tune | best @ final | verdict |",
                  "| :--- | ---: | ---: | :--- |"]
        for m, d in verify.items():
            if "best_short" not in d:
                continue
            lines.append(f"| `{m}` | {d['best_short']:.4g} | {d['best_long']:.4g} | "
                         f"{'stable' if d.get('stable') else '**MOVED**'} |")
        lines += ["", "If a winner moved, the short-horizon table is provisional for "
                      "that family and the tuning should be redone at a longer "
                      "horizon.", ""]

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

    wd = state["phases"].get("wd", {})
    if wd:
        lines += [f"## Weight-decay ablation (decoupled, wd = {args.wd_ablation:g})", "",
                  f"The primary table above is unregularized (`--weight-decay "
                  f"{args.weight_decay:g}`), which is the setting the theorems analyse "
                  f"and the one both reference implementations use. These runs repeat "
                  f"the best methods at seed {args.final_seeds[0]} with decoupled decay "
                  f"on the matrix parameters, at the *same* eta_0. What matters is "
                  f"whether the ordering moves, not the absolute gain -- eta_0 was not "
                  f"re-tuned under decay.", "",
                  "| method | eta_0 | with decay | no decay | delta |",
                  "| :--- | ---: | ---: | ---: | ---: |"]
        for k, v in sorted(wd.items()):
            got, ref = v.get("test_acc"), v.get("test_acc_no_decay")
            cells = [f"`{k}`", f"{v['lr']:.6g}",
                     "-" if got is None else f"{got:.2f}%",
                     "-" if ref is None else f"{ref:.2f}%",
                     "-" if (got is None or ref is None) else f"{got - ref:+.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    done = sum(1 for v in state["jobs"].values() if v)
    failed = sum(1 for v in state["jobs"].values() if not v)
    lines += ["## What this run did NOT establish", "",
              f"* **`lr_aux` was fixed at {args.lr_aux:g}**, not tuned, and not verified "
              f"to be method-independent. The auxiliary group is AdamW on the same "
              f"parameters for every method, so its optimum should not depend on the "
              f"matrix rule -- but that is an argument, not a measurement. Check it "
              f"with `python3 -m centralized.tune --stage aux`.",
              f"* **Momentum ({args.momentum:g}) and weight decay "
              f"({args.weight_decay:g}) were held fixed** for every method, so this is "
              f"a comparison at equal momentum, not at each method's own optimum.",
              f"* Weight decay is **{args.weight_decay_mode}**"
              + (" (`X *= 1 - lr*wd`, the LMO sees the true gradient). The coupled "
                 "convention -- `wd*X` added to the gradient, which is what "
                 "Mishra et al.'s Algorithm 1 and our own earlier numbers used -- is "
                 "*not* an alternative worth reporting as an equal: every step "
                 "direction here is scale-invariant, so coupled decay shrinks nothing "
                 "and only rotates the direction. `--weight-decay-mode coupled` "
                 "reproduces it for the appendix ablation."
                 if args.weight_decay_mode == "decoupled" else
                 " -- `wd*X` is folded into the gradient. Every step direction in this "
                 "code is scale-invariant, so this shrinks nothing and only rotates the "
                 "direction, by an amount set by the drifting, method-dependent ratio "
                 "`wd*||X||/||G||`. Prefer `--weight-decay-mode decoupled`."),
              "* **A gap smaller than the seed spread is not a result.** Add seeds with "
              "`--resume` and aggregate before claiming one.",
              "",
              "## Next steps", "",
              f"{done} jobs completed, {failed} failed. Resume with:", "",
              "```bash",
              f"python3 -m centralized.overnight --device {args.device} --resume",
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

    return "\n".join(lines)


def refresh_report(args, state, budget, sec, rule, alpha, tuned, *,
                   echo: bool = False) -> None:
    """Rewrite ``REPORT.md`` from the current state.

    Called after every phase and after every final run, so the report on disk is
    always current: with no deadline the run may last all day, and you should be
    able to read what has been found without stopping it.
    """
    text = build_report(args, state, budget, sec, rule, alpha, tuned)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    if echo:
        print("\n" + "=" * 78)
        print(text)
        print("=" * 78)
    print(f"[{stamp()}] report refreshed -> {REPORT_PATH}", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--budget-hours", type=float, default=0.0,
                   help="Wall-clock deadline; 0 (the default) means NO deadline -- every "
                        "phase runs to completion and only Ctrl-C stops it "
                        "(the report is written on the way out)")
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--download", action="store_true",
                   help="Download CIFAR-10 if missing (do this on the first run)")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Applied to the MATRIX parameters only (the auxiliary group is never decayed). Defaults to 0: our theorems analyse unregularized f, the nanoGPT record we build on uses 0.0 for every group, and all ten of Mishra et al.'s best CIFAR configurations select 0 -- so 0 is the setting under which theory and experiment describe the same algorithm. The overnight driver re-runs the top methods at 5e-4 as an ablation.")
    p.add_argument("--wd-ablation", type=float, default=5e-4,
                   help="Decay rate for the `wd` phase, which re-runs the "
                        "best --wd-ablation-top methods at the final horizon "
                        "with decay switched on. 0 disables the phase.")
    p.add_argument("--wd-ablation-top", type=int, default=3,
                   help="How many of the tuned methods to re-run with decay.")
    p.add_argument("--weight-decay-mode", type=str, default="decoupled",
                   choices=["decoupled", "coupled"],
                   help="decoupled (default): X *= 1 - lr*wd, leaving the LMO to see "
                        "the true gradient. coupled folds wd*X into the gradient, "
                        "which cannot shrink a scale-invariant step at all and only "
                        "rotates it -- kept for the appendix ablation.")
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
                   default=["gain", "lr", "final", "verify", "wd"],
                   choices=["gain", "alpha", "lr", "verify", "final", "wd"],
                   help="Which phases to run, in order. The lr_aux study is a "
                        "separate tool: python3 -m centralized.tune --stage aux")
    p.add_argument("--tune-epochs", type=int, default=15,
                   help="Proxy horizon for ranking learning rates. The 'verify' "
                        "phase checks that the ranking survives to --final-epochs; "
                        "note --last-k is capped at epochs//3 so the selection "
                        "metric is a genuine tail at any horizon")
    p.add_argument("--gain-epochs", type=int, default=20)
    p.add_argument("--gain-methods", nargs="*",
                   default=["signmuon", "muonsign", "signsgd", "muon"],
                   help="Methods to run the --log-gain diagnostic on. The exponent "
                        "is only open for the SIGN family -- for the LMO family "
                        "unit-gain and mup are the same multiplier -- so the sign "
                        "three are the measurement and muon is the reference.")
    p.add_argument("--final-epochs", type=int, default=75,
                   help="Epoch budget for the full-50k runs; 75 matches the paper")
    p.add_argument("--final-seeds", nargs="*", type=int, default=[0, 1, 2],
                   help="Seed-major: all methods at seed 0, then seed 1, ... "
                        "so an interrupted night still leaves a complete table")
    p.add_argument("--lr-points", type=int, default=5,
                   help="Lattice points per method: 3 per decade, so 5 spans "
                        "~1.3 decades and 7 spans ~2.")
    # Retired by the 1-2-5 lattice, which fixes the resolution at three points
    # per decade. Kept as an explicit error rather than deleted: a stale script
    # passing it would otherwise get a silently different grid than it asked for.
    p.add_argument("--lr-decades", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--alpha-grid", nargs="*", type=float, default=[0.0, 0.5, 1.0],
                   help="0 is the paper's current global LR (the control), "
                        "1/2 is unit-gain, 1 is mup")
    p.add_argument("--alpha-points", type=int, default=5)
    p.add_argument("--alpha-decades", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--alpha-method", type=str, default="signmuon")
    p.add_argument("--lr-extend-rounds", type=int, default=4,
                   help="When a method's optimum lands on a grid endpoint, widen the "
                        "grid in that direction and re-tune, up to this many times. "
                        "0 restores the old behaviour of reporting the endpoint with "
                        "a warning.")
    p.add_argument("--lr-extend-points", type=int, default=2,
                   help="Points added per extension round.")
    p.add_argument("--report-only", action="store_true",
                   help="Rebuild REPORT.md from state.json and exit: runs nothing, "
                        "needs no GPU, and is safe to call while a run is in flight.")
    p.add_argument("--verify-methods", nargs="*", default=["signmuon", "muon"],
                   help="Methods whose top learning rates are re-run at "
                        "--final-epochs to check horizon stability")
    p.add_argument("--verify-top", type=int, default=3,
                   help="How many of each method's best rates to re-run")
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
    args = p.parse_args()
    # Retired by the 1-2-5 lattice. Rejected rather than ignored: a night launched
    # from a stale script would otherwise search a grid it did not ask for, and the
    # only symptom would be a differently-tuned eta_0 twelve hours later.
    for dead, replacement in (("lr_decades", "--lr-points"),
                              ("alpha_decades", "--alpha-points")):
        if getattr(args, dead) is not None:
            p.error(f"--{dead.replace('_', '-')} was retired when the grid moved "
                    f"onto the 1-2-5 lattice, which fixes the resolution at three "
                    f"points per decade. Use {replacement} to set the span "
                    f"(5 points ~ 1.3 decades, 7 ~ 2).")
    return args


def print_schedule(args, sec: float, budget: Budget) -> None:
    n_alpha = len(args.alpha_grid) * args.alpha_points
    n_lr = sum(1 for _ in _lr_jobs(args)) * args.lr_points
    n_final = sum(1 for _ in _lr_jobs(args)) * len(args.final_seeds)
    n_verify = len(args.verify_methods) * args.verify_top
    plan = [
        ("gain", 2, args.gain_epochs),
        ("alpha", n_alpha, args.tune_epochs),
        ("lr", n_lr, args.tune_epochs),
        ("verify", n_verify, args.final_epochs),
        ("final", n_final, args.final_epochs),
        ("wd", 0 if args.wd_ablation <= 0 else args.wd_ablation_top,
         args.final_epochs),
    ]
    # The lr phase can widen a method's grid when its optimum lands on an endpoint,
    # which is extra work the plan cannot know about in advance.
    extra_lr = sum(1 for _ in _lr_jobs(args)) * args.lr_extend_rounds * args.lr_extend_points
    plan = [(n, j, e) for n, j, e in plan if j > 0 and n in args.phases]
    print("\n" + "=" * 78)
    budget_desc = ("no deadline -- runs until Ctrl-C" if budget.unlimited
                   else f"budget {args.budget_hours:g} h")
    print(f"SCHEDULE  ({budget_desc}, measured {sec:.1f} s/epoch)")
    print("=" * 78)
    last_col = "done by" if budget.unlimited else "fits?"
    print(f"  {'phase':<8}{'jobs':>6}{'epochs':>8}{'hours':>8}{'cumulative':>12}"
          f"   {last_col}")
    cum = 0.0
    for name, n, ep in plan:
        if name not in args.phases:
            print(f"  {name:<8}{'-':>6}{'-':>8}{'-':>8}{'-':>12}   skipped")
            continue
        hrs = n * job_cost(sec, ep) / 3600.0
        cum += hrs
        if budget.unlimited:
            eta = (datetime.now() + timedelta(hours=cum)).strftime("%a %H:%M")
            status = eta
        else:
            status = "yes" if cum <= args.budget_hours else "NO -- will be cut"
        print(f"  {name:<8}{n:>6}{ep:>8}{hrs:>8.1f}{cum:>12.1f}   {status}")
    print("=" * 78)
    if extra_lr and "lr" in args.phases:
        print(f"  Plus up to {extra_lr} more lr jobs "
              f"(+{extra_lr * job_cost(sec, args.tune_epochs) / 3600.0:.1f} h) if "
              f"optima land on grid endpoints:\n  the grid is then widened and that "
              f"method re-tuned, rather than reporting the endpoint.")
    if budget.unlimited:
        eta = (datetime.now() + timedelta(hours=cum)).strftime("%a %H:%M")
        print(f"  Everything runs to completion: {cum:.1f} h total, finishing around "
              f"{eta}.")
        print(f"  Ctrl-C at any point stops cleanly and writes the report. The report "
              f"on disk is\n  refreshed after every phase and after every final run, so "
              f"you can read it mid-run.")
        print(f"  Finals are seed-major: all {sum(1 for _ in _lr_jobs(args))} "
              f"parameterizations at seed {args.final_seeds[0]} first, then the next "
              f"seed --\n  so stopping early leaves complete tables, never fragments.")
        print(flush=True)
        return
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

    state = load_state() if args.resume or args.report_only else {"jobs": {}, "phases": {}}
    state.setdefault("jobs", {})
    state.setdefault("phases", {})
    state["started"] = state.get("started") or datetime.now().isoformat(timespec="seconds")
    budget = Budget(args.budget_hours)

    if args.report_only:
        # Read-only: rebuild the report from whatever is on disk and leave the
        # state file alone, so this is safe to run while a job is training.
        rule = state["phases"].get("chosen_rule") or args.lr_scaling
        print(build_report(args, state, budget, state.get("epoch_seconds") or 0.0,
                           rule, None, state["phases"].get("lr", {})))
        REPORT_PATH.write_text(
            build_report(args, state, budget, state.get("epoch_seconds") or 0.0,
                         rule, None, state["phases"].get("lr", {})),
            encoding="utf-8")
        print(f"\n[{stamp()}] rewrote {REPORT_PATH}")
        return

    if args.download:
        # Fetch once, here, rather than letting every child run pass --download:
        # a partially written archive shared by many runs is a bad failure mode.
        print(f"[{stamp()}] fetching {args.dataset} into {args.data} ...")
        from centralized.data import build_loaders
        build_loaders(args.dataset, args.data, batch_size=args.batch_size,
                      download=True, split="tune", num_workers=0)
        print(f"[{stamp()}] dataset ready")

    # Resuming across a change of convention would silently mix runs that are not
    # comparable: the cached jobs are keyed by method and rate only.
    fingerprint = {
        "weight_decay_mode": args.weight_decay_mode,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "head_adamw": args.head_adamw,
        "lr_aux": args.lr_aux,
        "dataset": args.dataset,
        "model": args.model,
    }
    old = state.get("fingerprint")
    if args.resume and old and old != fingerprint:
        differs = {k: (old.get(k), v) for k, v in fingerprint.items() if old.get(k) != v}
        print("REFUSING to resume: this run's settings differ from the recorded ones,")
        print("so the cached jobs are not comparable with the new ones.")
        for k, (was, now) in differs.items():
            print(f"  {k}: recorded {was!r} -> requested {now!r}")
        print(f"\nStart a fresh sweep (drop --resume, which overwrites "
              f"{STATE_PATH.name}), or move the old results aside first. Use "
              f"--report-only to read the existing report without running anything.")
        sys.exit(1)
    state["fingerprint"] = fingerprint

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
    if budget.unlimited:
        print(f"[{stamp()}] no deadline: every phase runs to completion. Ctrl-C stops "
              f"cleanly and writes the report; the report on disk is refreshed after "
              f"every phase, so you can read it at any time without stopping.")

    def refresh(echo: bool = False) -> None:
        refresh_report(args, state, budget, sec, rule, alpha,
                       tuned or state["phases"].get("lr", {}), echo=echo)

    def banner(name: str, note: str = "") -> None:
        print(f"\n{'=' * 78}\n[{stamp()}] PHASE {name}{note}\n{'=' * 78}", flush=True)

    # A second Ctrl-C raises out of the handler; the report must still be written,
    # so the phases run inside try/finally rather than being trusted to complete.
    try:
        if "gain" in args.phases:
            banner("gain")
            phase_gain(args, state, budget, sec)
            refresh()
        if "alpha" in args.phases and not _stop["requested"]:
            banner("alpha")
            alpha = phase_alpha(args, state, budget, sec)
            if alpha is not None:
                # Use the measured exponent for everything downstream.
                rule = {0.0: "legacy", 0.5: "unit-gain", 1.0: "mup"}.get(
                    round(alpha, 6), f"power:{alpha:g}")
                print(f"\n  -> using rule '{rule}' for the remaining phases")
                state["phases"]["chosen_rule"] = rule
                save_state(state)
            refresh()
        if "lr" in args.phases and not _stop["requested"]:
            banner("lr", f"  (rule '{rule}')")
            tuned = phase_lr(args, state, budget, sec, rule)
            refresh()
        if "verify" in args.phases and not _stop["requested"]:
            banner("verify", f"  (horizon stability, {args.final_epochs} epochs)")
            phase_verify(args, state, budget, sec, rule,
                         tuned or state["phases"].get("lr", {}))
            refresh()
        if "final" in args.phases and not _stop["requested"]:
            banner("final", f"  ({args.final_epochs} epochs, full 50k, "
                            f"seeds {args.final_seeds})")
            phase_final(args, state, budget, sec, rule,
                        tuned or state["phases"].get("lr", {}))
            refresh()
        if "wd" in args.phases and not _stop["requested"]:
            banner("wd", f"  (decay ablation, wd={args.wd_ablation:g} decoupled)")
            phase_wd(args, state, budget, sec, rule,
                     tuned or state["phases"].get("lr", {}))
    except KeyboardInterrupt:
        print(f"\n[{stamp()}] interrupted -- writing the report from what completed.")
    finally:
        save_state(state)
        refresh(echo=True)


if __name__ == "__main__":
    main()
