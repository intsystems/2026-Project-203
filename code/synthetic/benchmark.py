"""Synthetic smooth convex benchmark: F(X) = 1/2 <X, A X B> (Equation 9).

Replaces the two near-duplicate scripts ``synthetic_benchmark_grad.py`` (grid
search) and ``synthetic_benchmark_conclusion.py`` (final runs at the tuned
hyperparameters); they shared everything but their ``__main__`` block, had
different hardcoded GPU indices, and between them covered only 6 of the paper's
8 methods.

    python3 -m synthetic.benchmark --mode grid  --device cuda:0
    python3 -m synthetic.benchmark --mode final --device cuda:0
    python3 -m synthetic.benchmark --mode grid --methods signmuon muonusign

Problem
-------
``A in S_+^m``, ``B in S_+^n`` have eigenvalues drawn uniformly from ``(0, 1)``,
which bounds the Frobenius-norm smoothness constant by ``L <= 1``;
``X_0 ~ N(0, 0.01)`` entrywise. Methods are ranked by the number of iterations to
reach ``F(X) <= 1e-3`` within ``T_max = 5000``.

Two details that matter for the numbers
---------------------------------------
* ``--lmo-dtype float32`` runs the Newton-Schulz iteration in single precision.
  The default ``bfloat16`` matches the reference Muon implementation and is what
  the reported numbers used, but bfloat16 carries only ~3 decimal digits, so for
  the methods that *sign the LMO output* (SignMuon, MuonSign) entries of
  ``polar(M)`` near zero can flip -- on a deterministic 500x500 problem this is a
  real source of run-to-run variation in the iteration count.
* ``EF21MuonSign`` is scored on its **exact** model ``X``, not on the compressed
  broadcast model ``W`` held in ``p.data``. ``X`` is the iterate the convergence
  theory bounds. (The superseded scripts scored ``W``.)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from common.utils import results_root
from common.optimizers import (
    EF21MuonSign,
    EF21MuonUSign,
    EF21SignMuon,
    Muon,
    MuonSign,
    MuonUSign,
    SignMuon,
    SignSGD,
)

METHOD_CLASSES = {
    "signmuon": SignMuon,
    "ef21signmuon": EF21SignMuon,
    "muonusign": MuonUSign,
    "muonsign": MuonSign,
    "ef21muonusign": EF21MuonUSign,
    "ef21muonsign": EF21MuonSign,
    "muon": Muon,
    "signsgd": SignSGD,
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

DEFAULT_METHODS = list(METHOD_CLASSES)

# Learning-rate grids as "lo:hi:step" (inclusive of both endpoints).
#
# These reproduce the paper's hyperparameter-search table where that table is
# self-consistent. Two rows are NOT reproducible from the table as printed and
# are marked below; see the accompanying review notes. Override on the command
# line with ``--lr-grid METHOD=lo:hi:step``.
DEFAULT_LR_GRIDS: Dict[str, str] = {
    "signmuon":      "1e-4:1e-3:1e-4",     # paper table, contains the reported 2e-4
    "muonusign":     "1e-4:1e-3:1e-4",     # paper table (never actually run)
    "muonsign":      "1e-4:1e-3:1e-4",     # no paper row; mirrors its siblings
    "ef21signmuon":  "1e-4:1e-3:1e-4",     # no paper row; diverging method
    # The paper's table says [5e-3, 2e-2] step 1e-4 for both EF21 rows, but the
    # reported optima (3.3e-3 and 2.8e-3) lie BELOW that range. The grid below is
    # the union, coarsened to step 1e-4 only where it must be.
    "ef21muonusign": "1e-3:2e-2:1e-4",
    "ef21muonsign":  "1e-3:2e-2:1e-4",
    # The paper's table says step 1e-3 from 5e-3, but the reported optimum
    # (6.5e-3) is off that grid; step 5e-4 contains it.
    "muon":          "5e-3:2e-2:5e-4",
    "signsgd":       "5e-5:2e-4:1e-5",     # paper table, contains the reported 1.5e-4
    "sgd":           "1e-2:1e-1:1e-2",     # paper table, contains the reported 1e-1
    "adam":          "1e-2:1e-1:1e-2",     # paper table, contains the reported 7e-2
}

DEFAULT_MOMENTUM_GRID = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95"

# Tuned hyperparameters currently reported in the paper (Table "Tuned learning
# rates ..."). Used by ``--mode final``. ``None`` means "not yet tuned".
REPORTED_BEST: Dict[str, Optional[Tuple[float, float]]] = {
    "signmuon":      (2e-4, 0.2),
    "muonusign":     None,
    "muonsign":      None,
    "ef21signmuon":  None,
    "ef21muonusign": (3.3e-3, 0.1),
    "ef21muonsign":  (2.8e-3, 0.1),
    "muon":          (6.5e-3, 0.1),
    "signsgd":       (1.5e-4, 0.8),
    "sgd":           (1e-1, 0.95),
    "adam":          (7e-2, 0.0),
}


# --------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------


def generate_psd_matrix(dim: int, device, generator=None) -> torch.Tensor:
    """Symmetric PSD matrix with eigenvalues uniform in ``(0, 1)``."""
    eigenvalues = torch.rand(dim, device=device, generator=generator)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device, generator=generator))
    return Q @ torch.diag(eigenvalues) @ Q.T


class QuadraticMatrixProblem(nn.Module):
    """``F(X) = 1/2 <X, A X B>`` with ``X`` the (only) parameter."""

    def __init__(self, m: int = 500, n: int = 500, device="cpu", generator=None):
        super().__init__()
        init = torch.randn(m, n, device=device, generator=generator) * 0.1
        self.X = nn.Parameter(init)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(self.X * (A @ self.X @ B))


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------


def run_one(
    method: str,
    kwargs: Dict[str, float],
    A: torch.Tensor,
    B: torch.Tensor,
    m: int = 500,
    n: int = 500,
    target_loss: float = 1e-3,
    max_iters: int = 5000,
    device="cpu",
    init_seed: int = 42,
    lmo_dtype: str = "bfloat16",
) -> Dict:
    """Run one (method, hyperparameter) configuration to the target loss.

    ``init_seed`` fixes ``X_0``, so every configuration starts from the *same*
    point and the comparison is not confounded by initialization.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(init_seed))
    model = QuadraticMatrixProblem(m, n, device=device, generator=gen)

    cls = METHOD_CLASSES[method]
    opt_kwargs = dict(kwargs)
    if method in ("sgd", "adam"):
        opt_kwargs.pop("nesterov", None)
        if method == "adam":
            opt_kwargs.pop("momentum", None)
    else:
        opt_kwargs["lmo_dtype"] = getattr(torch, lmo_dtype)
    optimizer = cls(model.parameters(), **opt_kwargs)

    # EF21MuonSign holds the compressed broadcast model W in p.data; F is scored
    # on the exact model X that the theory bounds.
    using_exact = getattr(optimizer, "using_exact", None)

    def tracked_loss() -> float:
        if using_exact is None:
            with torch.no_grad():
                return float(model(A, B))
        with using_exact():
            with torch.no_grad():
                return float(model(A, B))

    start = time.time()
    iters_to_converge = max_iters
    loss_history: List[float] = []
    grad_norm_history: List[float] = []

    for i in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        model(A, B).backward()                      # gradient at p.data (= W)

        f_val = tracked_loss()                      # value at the tracked iterate
        loss_history.append(f_val)
        grad_norm_history.append(float(model.X.grad.norm()))

        if f_val <= target_loss:
            iters_to_converge = i + 1
            break
        if not (f_val == f_val) or f_val > 1e12:    # NaN or blow-up
            break

        optimizer.step()

    return {
        "method": method,
        "kwargs": {k: v for k, v in kwargs.items()},
        "iters_to_converge": iters_to_converge,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "time_seconds": time.time() - start,
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
    }


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------


def parse_lr_grid(spec: str) -> List[float]:
    """``"lo:hi:step"`` -> inclusive list of learning rates."""
    lo, hi, step = (float(x) for x in spec.split(":"))
    if step <= 0:
        raise ValueError(f"grid step must be positive: {spec!r}")
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 12) for i in range(n + 1)]


def parse_momentum_grid(spec: str) -> List[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def better(candidate: Dict, incumbent: Optional[Dict]) -> bool:
    """Fewer iterations wins; ties broken by the lower final loss."""
    if incumbent is None:
        return True
    if candidate["iters_to_converge"] != incumbent["iters_to_converge"]:
        return candidate["iters_to_converge"] < incumbent["iters_to_converge"]
    return candidate["final_loss"] < incumbent["final_loss"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["grid", "final"], default="grid")
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS,
                   choices=DEFAULT_METHODS)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--m", type=int, default=500)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--target-loss", type=float, default=1e-3)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--problem-seed", type=int, default=1337, help="Seed for A and B")
    p.add_argument("--init-seed", type=int, default=42, help="Seed for X_0")
    p.add_argument("--lmo-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--lr-grid", nargs="*", default=[],
                   metavar="METHOD=lo:hi:step", help="Override a method's LR grid")
    p.add_argument("--momentum-grid", type=str, default=DEFAULT_MOMENTUM_GRID)
    p.add_argument("--out", type=str, default=None,
                   help="Output directory (default: results/synthetic/, one "
                        "subdirectory per method)")
    p.add_argument("--save-histories", action="store_true",
                   help="Store the full loss/grad-norm curves (large)")
    return p.parse_args()


def main() -> None:
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}  (mode={args.mode}, lmo_dtype={args.lmo_dtype})\n")

    gen = torch.Generator(device=device)
    gen.manual_seed(args.problem_seed)
    A = generate_psd_matrix(args.m, device=device, generator=gen)
    B = generate_psd_matrix(args.n, device=device, generator=gen)

    lr_grids = dict(DEFAULT_LR_GRIDS)
    for override in args.lr_grid:
        key, _, spec = override.partition("=")
        if key not in METHOD_CLASSES:
            raise ValueError(f"--lr-grid: unknown method {key!r}")
        lr_grids[key] = spec
    momenta = parse_momentum_grid(args.momentum_grid)

    out_root = Path(args.out) if args.out else results_root() / "synthetic"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for method in args.methods:
        if args.mode == "final":
            best = REPORTED_BEST.get(method)
            if best is None:
                print(f"{method}: no tuned hyperparameters reported yet -- "
                      f"run --mode grid first. Skipping.")
                continue
            configs = [{"lr": best[0], "momentum": best[1]}]
        else:
            lrs = parse_lr_grid(lr_grids[method])
            configs = ([{"lr": lr} for lr in lrs] if method == "adam"
                       else [{"lr": lr, "momentum": mom} for lr in lrs for mom in momenta])

        print(f"--- {method}: {len(configs)} configuration(s) ---")
        best_metrics = None
        for cfg in configs:
            metrics = run_one(
                method, cfg, A, B, m=args.m, n=args.n,
                target_loss=args.target_loss, max_iters=args.max_iters,
                device=device, init_seed=args.init_seed, lmo_dtype=args.lmo_dtype,
            )
            iters = metrics["iters_to_converge"]
            reached = iters < args.max_iters
            print(f"  lr={cfg['lr']:<10.6g} mom={cfg.get('momentum', 0):<5} "
                  f"iters={iters if reached else '>' + str(args.max_iters):<8} "
                  f"loss={metrics['final_loss']:.6f}")
            if better(metrics, best_metrics):
                best_metrics = metrics

        if best_metrics is None:
            continue
        if not args.save_histories:
            best_metrics = {k: v for k, v in best_metrics.items()
                            if k not in ("loss_history", "grad_norm_history")}

        method_dir = out_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "problem": {"m": args.m, "n": args.n, "target_loss": args.target_loss,
                        "max_iters": args.max_iters,
                        "problem_seed": args.problem_seed, "init_seed": args.init_seed,
                        "lmo_dtype": args.lmo_dtype},
            "grid": lr_grids.get(method) if args.mode == "grid" else None,
            "momentum_grid": momenta if args.mode == "grid" else None,
            "best": best_metrics,
        }
        with open(method_dir / f"{args.mode}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        summary.append(best_metrics)

    print(f"\n{'method':<16}{'iters':>10}{'final loss':>14}   best hyperparameters")
    print("-" * 72)
    for m in summary:
        iters = m["iters_to_converge"]
        iters_str = str(iters) if iters < args.max_iters else f">{args.max_iters}"
        kw = ", ".join(f"{k}={v:g}" for k, v in m["kwargs"].items())
        print(f"{m['method']:<16}{iters_str:>10}{m['final_loss']:>14.6f}   {kw}")
    print(f"\nResults written to {out_root.resolve()}")


if __name__ == "__main__":
    main()
