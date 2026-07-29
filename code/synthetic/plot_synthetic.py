"""Figures for the synthetic convex problem, from the JSON ``run_gpu`` writes.

    python3 -m synthetic.plot_synthetic                      # results/synthetic/
    python3 -m synthetic.plot_synthetic --results results/synthetic_20x20
    python3 -m synthetic.plot_synthetic --figures loss gnorm

Reads ``results/synthetic*/<method>/<stage>.json`` and skips, with a message,
any figure whose stage has not been run -- so a partial run still produces the
figures it can.

Each figure is named for the ``\\label`` it belongs to rather than for a float
number (the numbers move whenever a float is added elsewhere):

=========================  =====================  =========================
figure                     stages it needs        paper
=========================  =====================  =========================
``synthetic_main``         ``final``              ``fig:synthetic_main``
``loss``, ``GN``           ``final``              ``fig:synthetic_results``
``diagnostics``            ``floor``,             ``fig:synthetic_dynamics``
                           ``horizon``, ``kappa``
=========================  =====================  =========================

``diagnostics`` is one row of three panels under a single legend, the layout
the counterexample and federated figures use, because the panels draw the same
ten methods. Drawn one per figure with a legend beside each, the legend was
wider than the axes it belonged to.

Output goes to ``<results>/figures/`` as both PDF and PNG. Nothing is written
into ``aaai_article/``: copy a figure over deliberately once you have looked at
it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

from common.plotting import (FS_ANNOT, FS_LABEL, FS_LEGEND, INK_2, MUTED,
                             TEXT_WIDTH, color_of, label_of, order_methods,
                             figure_legend, panel_legend, save_figure,
                             style_axes, use_paper_style)
from common.utils import results_root

use_paper_style()

#: Width of a half-page panel: the diagnostic figures below are authored at the
#: size a ``0.48\textwidth`` subfigure prints at. A figure drawn wider is scaled
#: down by LaTeX and takes its type with it, which is how 9 pt labels turn into
#: 5 pt ones. The paper's own figure is ``fig_trajectories``, at TEXT_WIDTH.
PANEL_WIDTH = 0.48 * TEXT_WIDTH

#: Most entries a shared legend fits across ``TEXT_WIDTH`` before it overruns the
#: figure. Set from the longest label set here (``EF21-MuonUSign`` and friends);
#: shorter labels would fit more, but the cost of being conservative is one line
#: of white space and the cost of being wrong is a silently clipped label.
MAX_LEGEND_COLS = 5

#: ``stage -> the figures that need it``. Used to report what a partial run is
#: missing, instead of failing on the first absent file.
FIGURES = {
    "trajectories": ("final",),
    "loss": ("final",),
    "gnorm": ("final",),
    "diagnostics": ("floor", "horizon", "kappa"),
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


def _draw_trajectory(ax, data: Dict[str, dict], key: str, ylabel: str) -> int:
    """One curve per method, at that method's own tuned hyperparameters."""
    drawn = 0
    for method, payload in data.items():
        hist = payload["result"].get(key)
        if not hist:
            continue
        # The curve is the geometric mean over the stage's problem instances,
        # elementwise, matching how every scalar in the tables is aggregated.
        # The plateau oscillation survives it -- averaging independent draws
        # damps its amplitude but not its level, which is what the panel is for.
        ax.plot(range(len(hist)), hist, color=color_of(method), linewidth=1.6,
                label=label_of(method), zorder=3)
        drawn += 1
    ax.set_xlabel("iteration", color=INK_2, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=FS_LABEL)
    return drawn


def fig_trajectory(plt, data: Dict[str, dict], key: str, ylabel: str):
    """A single trajectory panel -- the diagnostic form, one metric at a time."""
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, 2.4))
    style_axes(ax, logy=True)
    if not _draw_trajectory(ax, data, key, ylabel):
        plt.close(fig)
        return None
    ax.set_title(_problem_caption(list(data.values())), color=MUTED,
                 fontsize=FS_ANNOT - 1.5, loc="left", pad=6)
    # Inside, not in a gutter: every curve here decays, so the upper right is
    # empty, and a gutter wide enough for "EF21-MuonUSign" would take a third of
    # a 3.4-inch panel.
    panel_legend(ax, "upper right")
    fig.subplots_adjust(left=0.155, right=0.985, top=0.90, bottom=0.185)
    return fig


def fig_trajectories(plt, data: Dict[str, dict]):
    """The paper's figure: loss and gradient norm side by side, one legend.

    Two panels rather than two separate subfigures, for the same reason the CIFAR
    and counterexample figures are laid out this way: the panels draw the same
    methods, so a shared legend under both says once what two legends would say
    twice -- and at a two-column width neither panel has to give up a third of
    its area to hold it.

    The legend wraps past ``MAX_LEGEND_COLS``. In one row, ten entries of these
    lengths overrun ``TEXT_WIDTH`` at both ends, and matplotlib centres the
    overrun rather than reporting it: the outermost labels are simply clipped
    off the page, which in the first version of this figure cost SignMuon its
    name and Adam its entry.
    """
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), squeeze=False)
    specs = [("loss_history", "$F(X_t)$"),
             ("grad_norm_history", r"$\|\nabla F(X_t)\|_F$")]
    for ax, (key, ylabel) in zip(axes[0], specs):
        style_axes(ax, logy=True)
        if not _draw_trajectory(ax, data, key, ylabel):
            plt.close(fig)
            return None
    axes[0][0].set_title(_problem_caption(list(data.values())), color=MUTED,
                         fontsize=FS_ANNOT - 1.5, loc="left", pad=6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    rows = 1 + (len(labels) - 1) // MAX_LEGEND_COLS
    ncol = -(-len(labels) // rows)          # balance the rows, don't fill-then-spill
    figure_legend(fig, handles, labels, ncol=ncol)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90,
                        bottom=0.30 + 0.08 * (rows - 1), wspace=0.22)
    return fig


# --------------------------------------------------------------------------
# fig:synthetic_dynamics (left) -- the accuracy floor of a constant step
# --------------------------------------------------------------------------


def _curve(ax, method: str, xs, ys) -> None:
    """One method's series, labelled with its name and nothing else.

    The fitted exponents used to ride in the label. With ten methods that made
    the legend wider than the axes it belonged to, and the panels were squeezed
    into the strip left over. Every exponent is a column of the table these
    panels illustrate, so the label carries the name alone and the legend goes
    under the row, as in every other figure in the paper.
    """
    ax.plot(xs, ys, color=color_of(method), linewidth=1.5, marker="o",
            markersize=3.2, label=label_of(method), zorder=3)


def _panel_floor(ax, data: Dict[str, dict]) -> int:
    """``||grad F||_inf`` against ``eta``: the accuracy floor of a constant step.

    A normalized step of fixed length cannot converge at a constant rate; the
    gradient plateaus at a level linear in ``eta``, so slope 1 is the
    prediction. SGD has no floor and is absent by construction.
    """
    drawn = 0
    for method, payload in data.items():
        rows = [r for r in payload["result"]["rows"] if r.get("settled")]
        if len(rows) < 2:
            continue
        _curve(ax, method, [r["lr"] for r in rows], [r["g_inf"] for r in rows])
        drawn += 1
    ax.set_xlabel(r"step size $\eta$", color=INK_2, fontsize=FS_LABEL)
    ax.set_ylabel(r"floor $\|\nabla F\|_\infty$", color=INK_2, fontsize=FS_LABEL)
    return drawn


def _panel_horizon(ax, data: Dict[str, dict]) -> int:
    """Error against budget ``T``, with ``(eta, mu, schedule)`` retuned at each.

    Plots the quantity the theorems bound, ``min_t ||grad F(X_t)||_*^2`` in the
    norm dual to each method's LMO ball, so the exponent fitted on the curve is
    the ``p`` in the table. Retuning per budget is the point: imposing one
    schedule on every budget measures the schedule, not the method.
    """
    drawn = 0
    for method, payload in data.items():
        rec = payload["result"]
        rows = rec.get("rows") or []
        if len(rows) < 2:
            continue
        # Falls back to the Frobenius norm for a results tree written before the
        # dual norm was recorded.
        dual = all(r.get("best_dual") is not None for r in rows)
        ys = ([r["best_dual"] ** 2 for r in rows] if dual
              else [r["best_gnorm"] for r in rows])
        _curve(ax, method, [r["T"] for r in rows], ys)
        drawn += 1
    ax.set_xlabel("budget $T$ (iterations)", color=INK_2, fontsize=FS_LABEL)
    ax.set_ylabel(r"$\min_{t\leq T}\|\nabla F(\mathbf{X}_t)\|_*^2$",
                  color=INK_2, fontsize=FS_LABEL)
    return drawn


def _panel_kappa(ax, data: Dict[str, dict]) -> int:
    """Best gradient norm within the budget against ``L/sigma``, tuned at each."""
    drawn = 0
    for method, payload in data.items():
        rows = payload["result"].get("rows") or []
        if len(rows) < 2:
            continue
        _curve(ax, method, [r["kappa"] for r in rows],
               [r["best_gnorm"] for r in rows])
        drawn += 1
    ax.set_xlabel(r"condition number $L/\sigma$", color=INK_2, fontsize=FS_LABEL)
    ax.set_ylabel(r"best $\|\nabla F\|$ within budget", color=INK_2,
                  fontsize=FS_LABEL)
    return drawn


#: ``stage -> (panel drawer, subcaption)``, in printed order.
DIAGNOSTIC_PANELS = (
    ("floor", _panel_floor, "floor"),
    ("horizon", _panel_horizon, "budget"),
    ("kappa", _panel_kappa, "conditioning"),
)


def fig_diagnostics(plt, staged: Dict[str, Dict[str, dict]]):
    """The three log-log diagnostics as one row, under a single legend.

    Laid out like the counterexample and federated figures: full text width, one
    legend beneath the row because the panels draw the same ten methods, and no
    per-panel legend at all. A stage that has not been run drops its panel
    rather than the whole figure.
    """
    present = [(stage, draw) for stage, draw, _ in DIAGNOSTIC_PANELS
               if staged.get(stage)]
    if not present:
        return None
    fig, axes = plt.subplots(1, len(present), figsize=(TEXT_WIDTH, 2.35),
                             squeeze=False)
    for ax, (stage, draw) in zip(axes[0], present):
        style_axes(ax, logx=True, logy=True)
        if not draw(ax, staged[stage]):
            plt.close(fig)
            return None
    # Across every panel, not just the first: SGD has no floor and so never
    # appears in the left one, and a legend taken from there alone would omit a
    # method the other two panels draw.
    handles, labels = {}, []
    for ax in axes[0]:
        for handle, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab], _ = handle, labels.append(lab)
    rows = 1 + (len(labels) - 1) // MAX_LEGEND_COLS
    figure_legend(fig, [handles[lab] for lab in labels], labels,
                  ncol=-(-len(labels) // rows), fontsize=FS_LEGEND)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.97,
                        bottom=0.30 + 0.07 * (rows - 1), wspace=0.30)
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
        "trajectories": lambda s: fig_trajectories(plt, s["final"]),
        "loss": lambda s: fig_trajectory(plt, s["final"], "loss_history",
                                         "$F(X_t)$"),
        "gnorm": lambda s: fig_trajectory(plt, s["final"], "grad_norm_history",
                                          r"$\|\nabla F(X_t)\|_F$"),
        "diagnostics": lambda s: fig_diagnostics(plt, s),
    }
    stems = {"trajectories": "synthetic_main", "loss": "loss", "gnorm": "GN",
             "diagnostics": "diagnostics"}

    cache: Dict[str, Dict[str, dict]] = {}
    written, skipped = [], []
    for name in args.figures:
        stages = FIGURES[name]
        for stage in stages:
            if stage not in cache:
                cache[stage] = load_stage(results, stage)
        staged = {stage: cache[stage] for stage in stages}
        missing = [stage for stage in stages if not staged[stage]]
        if len(missing) == len(stages):
            skipped.append(
                f"{name}: no output for {' or '.join(missing)} under "
                f"{results.name}/ -- run "
                f"`python3 -m synthetic.run_gpu --stages {' '.join(missing)}`")
            continue
        if missing:
            # A multi-panel figure drops the panels it cannot draw rather than
            # the whole figure, but says which, so a partial run is never
            # mistaken for a complete one.
            skipped.append(f"{name}: drawn without the "
                           f"{', '.join(missing)} panel(s), which have no output")
        fig = builders[name](staged)
        if fig is None:
            note = (" (--mode final keeps trajectories only for a single problem "
                    "instance; re-run with --save-histories)"
                    if "final" in stages else " (too few usable points to plot)")
            skipped.append(f"{name}: ran but has nothing to draw{note}")
            continue
        # Every figure here is laid out at the width it prints at, so none of
        # them may be re-scaled by a tight bounding box.
        written += save_figure(fig, out, stems[name], formats=args.formats,
                               tight=False)
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
