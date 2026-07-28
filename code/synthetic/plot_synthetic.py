"""Figures for the synthetic convex problem, from the JSON ``run_gpu`` writes.

    python3 -m synthetic.plot_synthetic                      # results/synthetic/
    python3 -m synthetic.plot_synthetic --results results/synthetic_20x20
    python3 -m synthetic.plot_synthetic --figures loss gnorm

Replaces ``notebooks/plot_synthetic.ipynb``, which still read the pre-refactor
``saves_synthetic_001/`` layout and the retired method names. Everything here
reads ``results/synthetic*/<method>/<stage>.json`` and skips, with a message, any
figure whose stage has not been run -- so a partial run still produces the
figures it can.

Five figures, each named for the ``\\label`` it belongs to rather than for a
float number (the numbers move whenever a float is added elsewhere):

===========  =========================  ==================================
stage        file                       paper
===========  =========================  ==================================
``final``    ``loss``, ``GN``           ``fig:synthetic_results``
``floor``    ``floor``                  ``fig:synthetic_dynamics`` (left)
``horizon``  ``horizon``                ``fig:synthetic_dynamics`` (right)
``kappa``    ``kappa``                  ``fig:synthetic_kappa``
===========  =========================  ==================================

The last three are currently **commented out** in the paper -- the
``\\includegraphics`` lines exist but are disabled because the files were never
produced. This script produces them.

Output goes to ``<results>/figures/`` as both PDF and PNG. Nothing is written
into ``aaai_article/``: copy a figure over deliberately once you have looked at
it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from common.plotting import (INK_2, MUTED, color_of, label_of, legend,
                             order_methods, save_figure, style_axes)
from common.utils import results_root

#: ``stage -> the figures that need it``. Used to report what a partial run is
#: missing, instead of failing on the first absent file.
FIGURES = {
    "loss": "final",
    "gnorm": "final",
    "floor": "floor",
    "horizon": "horizon",
    "kappa": "kappa",
}


def load_stage(results: Path, stage: str) -> Dict[str, dict]:
    """``{method: result}`` for one stage, in the paper's method order."""
    found = {}
    for path in sorted(results.glob(f"*/{stage}.json")):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipping {path}: {exc}")
            continue
        found[path.parent.name] = payload
    return {m: found[m] for m in order_methods(found)}


def _problem_caption(payloads: Sequence[dict]) -> str:
    if not payloads:
        return ""
    p = next(iter(payloads))["problem"]
    return (f"{p['m']}x{p['n']}, spectrum {p['spectrum']}, "
            f"L/sigma = {p['condition_number']:.3g}, lmo_dtype {p['lmo_dtype']}")


# --------------------------------------------------------------------------
# fig:synthetic_results -- loss and gradient-norm trajectories
# --------------------------------------------------------------------------


def fig_trajectory(plt, data: Dict[str, dict], key: str, ylabel: str):
    """One curve per method, at that method's own tuned hyperparameters."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    style_axes(ax, logy=True)
    drawn = 0
    for method, payload in data.items():
        hist = payload["result"].get(key)
        if not hist:
            continue
        # A history is only kept when the stage ran on ONE problem instance;
        # _aggregate drops it otherwise, because averaging trajectories across
        # instances would smear the very oscillation these plots are about.
        ax.plot(range(len(hist)), hist, color=color_of(method), linewidth=1.6,
                label=label_of(method), zorder=3)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("iteration", color=INK_2, fontsize=8.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)
    ax.set_title(_problem_caption(list(data.values())), color=MUTED, fontsize=7.5,
                 loc="left", pad=8)
    legend(ax, outside=True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# fig:synthetic_dynamics (left) -- the accuracy floor of a constant step
# --------------------------------------------------------------------------


def fig_floor(plt, data: Dict[str, dict]):
    """``||grad F||_inf`` against ``eta``, with the fitted exponent per method.

    The theory says a `+-1` step of fixed length cannot converge at a constant
    rate: the gradient plateaus at a level linear in ``eta``. A slope of 1 on
    this log-log plot is that prediction; SGD has no floor and is expected to be
    absent or ragged.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    style_axes(ax, logx=True, logy=True)
    drawn = 0
    for method, payload in data.items():
        rec = payload["result"]
        rows = [r for r in rec["rows"] if r.get("settled")]
        if len(rows) < 2:
            continue
        slope = rec.get("slope_gnorm")
        ax.plot([r["lr"] for r in rows], [r["g_inf"] for r in rows],
                color=color_of(method), linewidth=1.5, marker="o", markersize=3.2,
                label=f"{label_of(method)}  (slope {slope:.2f})"
                      if slope is not None else label_of(method), zorder=3)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel(r"step size $\eta$", color=INK_2, fontsize=8.5)
    ax.set_ylabel(r"floor $\|\nabla F\|_\infty$", color=INK_2, fontsize=8.5)
    ax.set_title("slope 1 is the predicted linear floor", color=MUTED,
                 fontsize=7.5, loc="left", pad=8)
    legend(ax, outside=True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# fig:synthetic_dynamics (right) -- best error at each tuned budget
# --------------------------------------------------------------------------


def fig_horizon(plt, data: Dict[str, dict]):
    """Error against budget ``T``, with ``(eta, mu, schedule)`` retuned at each.

    ``err ~ T^-p``: the nonconvex bound the theorems prove is ``p = 1/2``, a
    strongly convex rate would be ``p = 1``. Retuning per budget is the point --
    imposing one schedule on every budget measures the schedule, not the method.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    style_axes(ax, logx=True, logy=True)
    drawn = 0
    for method, payload in data.items():
        rec = payload["result"]
        rows = rec.get("rows") or []
        if len(rows) < 2:
            continue
        p = rec.get("exponent_gnorm")
        ax.plot([r["T"] for r in rows], [r["best_gnorm"] for r in rows],
                color=color_of(method), linewidth=1.5, marker="o", markersize=3.2,
                label=f"{label_of(method)}  (p = {p:.2f})" if p is not None
                      else label_of(method), zorder=3)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("budget $T$ (iterations)", color=INK_2, fontsize=8.5)
    ax.set_ylabel(r"best $\|\nabla F\|$ within $T$", color=INK_2, fontsize=8.5)
    ax.set_title(r"$p = 1/2$ is the nonconvex prediction", color=MUTED,
                 fontsize=7.5, loc="left", pad=8)
    legend(ax, outside=True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# fig:synthetic_kappa -- the same, against the condition number
# --------------------------------------------------------------------------


def fig_kappa(plt, data: Dict[str, dict]):
    """Best gradient norm within the budget against ``L/sigma``, tuned at each."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    style_axes(ax, logx=True, logy=True)
    drawn = 0
    for method, payload in data.items():
        rec = payload["result"]
        rows = rec.get("rows") or []
        if len(rows) < 2:
            continue
        s = rec.get("exponent_kappa")
        ax.plot([r["kappa"] for r in rows], [r["best_gnorm"] for r in rows],
                color=color_of(method), linewidth=1.5, marker="o", markersize=3.2,
                label=f"{label_of(method)}  (slope {s:.2f})" if s is not None
                      else label_of(method), zorder=3)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel(r"condition number $L/\sigma$", color=INK_2, fontsize=8.5)
    ax.set_ylabel(r"best $\|\nabla F\|$ within budget", color=INK_2, fontsize=8.5)
    legend(ax, outside=True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=None,
                   help="Directory holding <method>/<stage>.json "
                        "(default: results/synthetic/). Point it at "
                        "results/synthetic_20x20/ to plot a small-size pass.")
    p.add_argument("--out", default=None,
                   help="Where to write the figures (default: <results>/figures/)")
    p.add_argument("--figures", nargs="+", default=sorted(FIGURES),
                   choices=sorted(FIGURES), metavar="NAME",
                   help=f"default: all of {sorted(FIGURES)}")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="default: pdf png")
    return p.parse_args()


def main() -> int:
    args = get_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install -r requirements.txt")
        return 1

    results = Path(args.results) if args.results else results_root() / "synthetic"
    if not results.is_dir():
        print(f"No results at {results.resolve()}.\n"
              f"Run `python3 -m synthetic.run_gpu` first (or `--m 20` for a "
              f"small pass, which writes results/synthetic_20x20/).")
        return 1
    out = Path(args.out) if args.out else results / "figures"

    builders = {
        "loss": lambda d: fig_trajectory(plt, d, "loss_history", "$F(X_t)$"),
        "gnorm": lambda d: fig_trajectory(plt, d, "grad_norm_history",
                                          r"$\|\nabla F(X_t)\|_F$"),
        "floor": lambda d: fig_floor(plt, d),
        "horizon": lambda d: fig_horizon(plt, d),
        "kappa": lambda d: fig_kappa(plt, d),
    }
    stems = {"loss": "loss", "gnorm": "GN", "floor": "floor",
             "horizon": "horizon", "kappa": "kappa"}

    cache: Dict[str, Dict[str, dict]] = {}
    written, skipped = [], []
    for name in args.figures:
        stage = FIGURES[name]
        if stage not in cache:
            cache[stage] = load_stage(results, stage)
        data = cache[stage]
        if not data:
            skipped.append(f"{name}: stage '{stage}' has no output under "
                           f"{results.name}/ -- run "
                           f"`python3 -m synthetic.run_gpu --stages {stage}`")
            continue
        fig = builders[name](data)
        if fig is None:
            note = (" (--mode final keeps trajectories only for a single problem "
                    "instance; re-run with --save-histories)"
                    if stage == "final" else " (too few usable points to plot)")
            skipped.append(f"{name}: '{stage}' ran but has nothing to draw{note}")
            continue
        written += save_figure(fig, out, stems[name], formats=args.formats)
        plt.close(fig)

    for line in skipped:
        print(f"  ~ {line}")
    if not written:
        print("Nothing plotted.")
        return 1
    print(f"\nWrote {len(written)} file(s) to {out.resolve()}:")
    for path in written:
        print(f"    {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
