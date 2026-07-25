"""Equal-budget, validation-only learning-rate search.

    # stage 1: is the optimal auxiliary rate method-independent?
    python3 -m centralized.tune --stage aux --device cuda:0

    # stage 2: which per-layer scaling exponent?
    python3 -m centralized.tune --stage alpha --method signmuon --device cuda:0

    # stage 3: eta_0 per method, identical budget for all of them
    python3 -m centralized.tune --stage lr --lr-scaling unit-gain --device cuda:0

Three properties make this a defensible protocol rather than a sweep:

1. **Selection is on validation accuracy only** (``--split tune``, a fixed 45k/5k
   partition). The test set is never read during tuning.
2. **Equal budget.** Every method gets the same number of configurations, on a grid
   that is *multiplicatively* anchored -- a shared absolute grid would be unfair
   because the families have genuinely different natural scales.
3. **Boundary check.** If a method's winner sits at an endpoint of its grid, that
   is reported as a failure, not a result: the grid is extended and re-run. An
   optimum on the boundary is the second-most-common reviewer catch after
   test-set tuning.

The search is coarse (``sqrt(10)`` spacing) then fine (``2x`` around the winner) and
runs at a reduced epoch count; ``--verify-horizon`` re-runs the top-``k`` at the full
budget to confirm the ranking is horizon-stable, which is the assumption a short
proxy makes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from common.lr_scaling import RULES
from common.utils import results_root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # code/, the package root

#: All ten methods, in the paper's order.
ALL_METHODS = ["signmuon", "muonusign", "muonsign", "ef21signmuon",
               "ef21muonusign", "ef21muonsign", "muon", "signsgd", "sgd", "adam"]

#: Anchor for each method's grid, in the ``legacy`` parameterization. New methods
#: are anchored at their sibling: MuonUSign<->Muon (both take a full LMO step),
#: MuonSign<->SignMuon (both take a unit-sign step), EF21-SignMuon<->EF21-MuonUSign.
LEGACY_ANCHORS: Dict[str, float] = {
    "signmuon": 1e-3,
    "muonsign": 1e-3,
    "signsgd": 1e-3,
    "muon": 1.5e-2,
    "muonusign": 1.5e-2,
    "ef21muonusign": 8e-3,
    "ef21muonsign": 7e-3,
    "ef21signmuon": 8e-3,
    "sgd": 1.5e-2,
    "adam": 1e-3,
}

#: Under a scaling rule the sign family's eta_0 is a *base* rate, larger than the
#: legacy per-layer-constant rate by roughly the reciprocal of the rule's typical
#: multiplier. These are order-of-magnitude anchors only; the coarse grid spans
#: 1.5 decades either side, so being 3x off costs nothing.
SCALED_ANCHOR_BOOST: Dict[str, float] = {
    "none": 1.0,
    "legacy": 1.0,
    "unit-gain": 34.0,          # ~sqrt(fan_in) at the median ResNet-18 layer
    "mup": 1152.0,              # ~fan_in at the median layer
    "mishra-analysis": 384.0,   # ~sqrt(m n) at the median layer
}

AUX_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
ALPHA_GRID = [0.0, 0.5, 1.0]


def geom_grid(anchor: float, decades: float, points: int) -> List[float]:
    """``points`` values log-spaced symmetrically around ``anchor``."""
    if points < 2:
        return [anchor]
    lo = math.log10(anchor) - decades
    step = 2 * decades / (points - 1)
    return [10 ** (lo + i * step) for i in range(points)]


def refine_grid(best: float, factor: float = 2.0, points: int = 4) -> List[float]:
    """``points`` values geometrically around ``best``, spanning ``factor`` either way."""
    half = (points - 1) / 2
    ratio = factor ** (1.0 / max(half, 1))
    return [best * ratio ** (i - half) for i in range(points)]


# --------------------------------------------------------------------------
# Running one configuration
# --------------------------------------------------------------------------

_SUMMARY_RE = re.compile(r"val_acc\s+mean of last\s+\d+\s*:\s*([\d.]+)%")
_FINAL_RE = re.compile(r"test_acc mean of last\s+\d+\s*:\s*([\d.]+)%")


_EPOCH_TIME_RE = re.compile(r"median epoch time\s*:\s*([\d.]+)s")
_TARGET_RE = re.compile(r"epochs to [\d.]+% test acc\s*:\s*(\d+)")


def run_one(args, *, lr: float, lr_aux: float, lr_scaling: str,
            method: str, epochs: int, tag: str, split: str = "tune",
            seed: Optional[int] = None,
            extra: Sequence[str] = ()) -> Optional[Dict[str, float]]:
    """Launch one training run; return its metrics, or ``None`` on failure.

    Selection uses ``val_acc`` averaged over the last ``--last-k`` epochs -- the
    tail mean, not a single epoch, so the choice is not decided by one noisy
    evaluation. ``test_acc`` is parsed and recorded but is **never** used for
    selection; with ``split="tune"`` it is measured on 45k-trained models anyway.
    """
    cmd = [
        sys.executable, "-m", "centralized.main",
        "--dataset", args.dataset, "--model", args.model,
        "--optimizer", method, "--lr-scaling", lr_scaling,
        "--split", split, "--val-seed", str(args.val_seed),
        "--epochs", str(epochs), "--batch-size", str(args.batch_size),
        "--lr", repr(lr), "--lr-aux", repr(lr_aux),
        "--momentum", str(args.momentum), "--weight-decay", str(args.weight_decay),
        "--head-adamw", args.head_adamw, "--last-k", str(args.last_k),
        "--device", args.device,
        "--seed", str(args.seed if seed is None else seed),
        "--data", args.data, "--num-workers", str(args.num_workers),
        "--run-name", f"{'tune' if split == 'tune' else 'final'}_{tag}",
        *extra,
    ]
    if getattr(args, "nondeterministic", False):
        cmd.append("--nondeterministic")
    if getattr(args, "download", False):
        cmd.append("--download")

    log_dir = results_root() / "tuning_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"

    print(f"  lr={lr:<12.6g} lr_aux={lr_aux:<10.6g} -> ", end="", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, text=True)
    text = log_path.read_text(encoding="utf-8", errors="replace")

    if proc.returncode != 0:
        tail = "".join(text.splitlines(keepends=True)[-3:]).strip()
        print(f"FAILED (exit {proc.returncode}) {tail[:160]}")
        return None

    out: Dict[str, float] = {"lr": lr, "lr_aux": lr_aux, "log": str(log_path)}
    m_val = _SUMMARY_RE.search(text)
    m_test = _FINAL_RE.search(text)
    m_time = _EPOCH_TIME_RE.search(text)
    m_target = _TARGET_RE.search(text)
    if m_val:
        out["val_acc"] = float(m_val.group(1))
    if m_test:
        out["test_acc"] = float(m_test.group(1))     # recorded, NEVER selected on
    if m_time:
        out["epoch_seconds"] = float(m_time.group(1))
    if m_target:
        out["epochs_to_target"] = int(m_target.group(1))

    key = "val_acc" if "val_acc" in out else "test_acc"
    if key not in out:
        print(f"no metric found (see {log_path.name})")
        return None
    print(f"{key.replace('_', ' ')} {out[key]:.2f}%"
          + (f"  [{out['epoch_seconds']:.1f}s/ep]" if "epoch_seconds" in out else ""))
    return out


def best_of(results: Sequence[Optional[Dict[str, float]]],
            key: str = "val_acc") -> Optional[Dict[str, float]]:
    """Best run by ``key`` (validation accuracy by default)."""
    valid = [r for r in results if r is not None and key in r]
    return max(valid, key=lambda r: r[key]) if valid else None


def boundary_warning(best: Dict[str, float], grid: Sequence[float], key: str = "lr") -> str:
    """Flag an optimum sitting on an endpoint of the grid."""
    if not grid or best is None:
        return ""
    lo, hi = min(grid), max(grid)
    if math.isclose(best[key], lo, rel_tol=1e-9):
        return f"  !! winner is at the LOW end of the {key} grid -- extend downward and re-run"
    if math.isclose(best[key], hi, rel_tol=1e-9):
        return f"  !! winner is at the HIGH end of the {key} grid -- extend upward and re-run"
    return ""


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_aux(args) -> Dict:
    """Is the optimal auxiliary rate method-independent?

    The auxiliary group is AdamW on the same parameters for every method, so its
    optimum should not depend on the matrix rule. Verifying that on two anchor
    methods spanning an order of magnitude in ``eta_0`` earns the right to fix one
    ``lr_aux`` globally -- instead of a 2-D grid per method (10x the cost) or an
    unverified assertion.
    """
    anchors = args.aux_anchors or ["signmuon", "muon"]
    out: Dict[str, Dict] = {}
    for method in anchors:
        base = LEGACY_ANCHORS[method] * SCALED_ANCHOR_BOOST.get(args.lr_scaling, 1.0)
        lr_grid = geom_grid(base, decades=0.5, points=4)      # 4 x 4 = 16 configs
        print(f"\n=== aux stage: {method} (lr around {base:.4g}) ===")
        results = []
        for lr in lr_grid:
            for lr_aux in AUX_GRID:
                tag = f"aux_{method}_{args.lr_scaling}_lr{lr:.4g}_aux{lr_aux:.4g}"
                r = run_one(args, lr=lr, lr_aux=lr_aux, lr_scaling=args.lr_scaling,
                            method=method, epochs=args.epochs, tag=tag)
                if r:
                    results.append(r)
        best = best_of(results)
        if best is None:
            print(f"  no successful run for {method}")
            continue
        out[method] = {"best": best, "all": results}
        print(f"  best: lr={best['lr']:.4g}, lr_aux={best['lr_aux']:.4g}, "
              f"val {best['val_acc']:.2f}%")

    if len(out) >= 2:
        chosen = {m: d["best"]["lr_aux"] for m, d in out.items()}
        print("\n--- aux verdict ---")
        for m, v in chosen.items():
            print(f"  {m:<16} best lr_aux = {v:.4g}")
        if len(set(chosen.values())) == 1:
            v = next(iter(chosen.values()))
            print(f"  AGREE -> fix lr_aux = {v:.4g} for every method, and report that "
                  f"it was verified method-independent.")
        else:
            print("  DISAGREE -> lr_aux must be tuned per method; give every method "
                  "the same 2-D budget and say so.")
    return out


def stage_alpha(args) -> Dict:
    """Which per-layer scaling exponent for the sign family?

    Sweeps ``power:ALPHA`` jointly with ``eta_0`` on one representative sign-family
    method. ``alpha = 0`` is a global learning rate, ``1/2`` is the unit-gain rule,
    ``1`` is muP-with-alignment.

    Caveat worth reporting: ResNet-18 is a **weak instrument** for this. Twelve of
    its twenty conv layers have ``fan_in/fan_out = 9`` exactly, and those hold ~63%
    of the parameters, so ``alpha`` is only identified through the transition and
    1x1-downsample layers. Confirm on a second architecture (or read the
    ``--log-gain`` diagnostic, which measures the exponent directly) before
    treating a small val-accuracy gap as decisive.
    """
    method = args.method or "signmuon"
    out: Dict[str, Dict] = {}
    for alpha in (args.alpha_grid or ALPHA_GRID):
        rule = f"power:{alpha:g}"
        boost = SCALED_ANCHOR_BOOST["unit-gain"] ** (2 * alpha)   # ~fan_in^alpha
        base = LEGACY_ANCHORS[method] * boost
        grid = geom_grid(base, decades=1.0, points=5)
        print(f"\n=== alpha stage: {method}, alpha={alpha:g} (lr around {base:.4g}) ===")
        results = []
        for lr in grid:
            tag = f"alpha{alpha:g}_{method}_lr{lr:.4g}"
            r = run_one(args, lr=lr, lr_aux=args.lr_aux, lr_scaling=rule,
                        method=method, epochs=args.epochs, tag=tag)
            if r:
                results.append(r)
        best = best_of(results)
        if best is None:
            continue
        out[rule] = {"best": best, "all": results, "alpha": alpha}
        warn = boundary_warning(best, grid)
        print(f"  best: lr={best['lr']:.4g}, val {best['val_acc']:.2f}%")
        if warn:
            print(warn)

    if out:
        print("\n--- alpha verdict ---")
        ranked = sorted(out.items(), key=lambda kv: -kv[1]["best"]["val_acc"])
        for rule, d in ranked:
            print(f"  alpha={d['alpha']:<4g} val {d['best']['val_acc']:.2f}%  "
                  f"(at lr={d['best']['lr']:.4g})")
        top, second = ranked[0][1], ranked[1][1] if len(ranked) > 1 else None
        if second and abs(top["best"]["val_acc"] - second["best"]["val_acc"]) < 0.3:
            print("  gap < 0.3% -> NOT decisive on this architecture. Use the "
                  "--log-gain diagnostic, or a model with more shape diversity.")
        else:
            print(f"  -> alpha = {top['alpha']:g}")
    return out


def stage_lr(args) -> Dict:
    """Tune ``eta_0`` per method under a fixed scaling rule, equal budget for all."""
    methods = args.methods or ALL_METHODS
    boost = SCALED_ANCHOR_BOOST.get(args.lr_scaling, 1.0)
    out: Dict[str, Dict] = {}

    for method in methods:
        # Only the sign family's anchor moves with the scaling rule; the LMO family
        # keeps the aspect factor under every rule, so its eta_0 is unchanged.
        from centralized.train import LMO_FAMILY
        cls = LMO_FAMILY.get(method)
        is_sign = cls is not None and cls.family == "sign"
        base = LEGACY_ANCHORS[method] * (boost if is_sign else 1.0)

        coarse = geom_grid(base, decades=1.5, points=7)
        print(f"\n=== lr stage: {method} (coarse, 7 pts around {base:.4g}) ===")
        results = [run_one(args, lr=lr, lr_aux=args.lr_aux, lr_scaling=args.lr_scaling,
                           method=method, epochs=args.epochs,
                           tag=f"lr_{method}_{args.lr_scaling}_c{lr:.4g}")
                   for lr in coarse]
        best = best_of(results)
        if best is None:
            print(f"  every run failed for {method}")
            continue
        warn = boundary_warning(best, coarse)
        if warn:
            print(warn)

        fine = refine_grid(best["lr"], factor=2.0, points=4)
        print(f"  fine (4 pts around {best['lr']:.4g}):")
        results += [run_one(args, lr=lr, lr_aux=args.lr_aux, lr_scaling=args.lr_scaling,
                            method=method, epochs=args.epochs,
                            tag=f"lr_{method}_{args.lr_scaling}_f{lr:.4g}")
                    for lr in fine]
        best = best_of(results)
        out[method] = {"best": best, "all": [r for r in results if r],
                       "n_configs": len(coarse) + len(fine),
                       "boundary_warning": warn}
        print(f"  BEST: lr={best['lr']:.6g}, val {best['val_acc']:.2f}% "
              f"({len(coarse) + len(fine)} configs)")

    if out:
        print(f"\n--- eta_0 per method (rule '{args.lr_scaling}', "
              f"{next(iter(out.values()))['n_configs']} configs each) ---")
        for method, d in out.items():
            flag = "  [BOUNDARY]" if d["boundary_warning"] else ""
            print(f"  {method:<16} eta_0 = {d['best']['lr']:<12.6g} "
                  f"val {d['best']['val_acc']:.2f}%{flag}")
        _check_family_agreement(out)
    return out


def _check_family_agreement(out: Dict[str, Dict]) -> None:
    """The scaling rule's falsifiable prediction.

    Once the shape dependence lives in ``lambda``, ``eta_0`` is shape-free, so the
    tuned ``eta_0`` should agree *within* each family. Reporting this is a result,
    not a protocol note -- and if it fails, that is a real finding about where the
    families differ.
    """
    from centralized.train import LMO_FAMILY
    groups: Dict[str, List[Tuple[str, float]]] = {}
    for method, d in out.items():
        cls = LMO_FAMILY.get(method)
        if cls is None:
            continue
        groups.setdefault(cls.family, []).append((method, d["best"]["lr"]))

    print("\n--- family agreement (the scaling rule's prediction) ---")
    for family, entries in groups.items():
        if len(entries) < 2:
            continue
        vals = [v for _, v in entries]
        spread = max(vals) / min(vals)
        names = ", ".join(f"{m}={v:.4g}" for m, v in entries)
        verdict = ("AGREE (within 2x, i.e. one fine-grid step)" if spread <= 2.0
                   else f"DISAGREE by {spread:.1f}x")
        print(f"  {family:<6} {names}")
        print(f"  {'':<6} spread {spread:.2f}x -> {verdict}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["aux", "alpha", "lr"])
    p.add_argument("--methods", nargs="*", default=None,
                   help="lr stage: which methods (default: all ten)")
    p.add_argument("--method", type=str, default=None,
                   help="alpha stage: the representative sign-family method")
    p.add_argument("--aux-anchors", nargs="*", default=None,
                   help="aux stage: anchor methods (default: signmuon muon)")
    p.add_argument("--alpha-grid", nargs="*", type=float, default=None)
    p.add_argument("--lr-scaling", type=str, default="unit-gain",
                   help=", ".join(sorted(RULES)) + ", or power:ALPHA[,BETA]")

    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--epochs", type=int, default=20,
                   help="Short proxy horizon for tuning; verify with --verify-horizon")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-aux", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--head-adamw", type=str, default="always",
                   choices=["auto", "always", "never"])
    p.add_argument("--last-k", type=int, default=5)
    p.add_argument("--val-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--download", action="store_true", help="Download the dataset if missing")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--out", type=str, default=None,
                   help="Write the stage results to this JSON (default: "
                        "results/tuning/<stage>_<rule>.json)")
    return p.parse_args()


def main() -> None:
    args = get_args()
    stage = {"aux": stage_aux, "alpha": stage_alpha, "lr": stage_lr}[args.stage]
    out = stage(args)

    path = Path(args.out) if args.out else (
        results_root() / "tuning" / f"{args.stage}_{args.lr_scaling.replace(':', '')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stage": args.stage, "lr_scaling": args.lr_scaling,
                   "epochs": args.epochs, "selection_metric": "val_acc (last-k mean)",
                   "results": out}, f, indent=2)
    print(f"\nWritten to {path}")
    print("Selection used validation accuracy only; the test set was not read.")


if __name__ == "__main__":
    main()
