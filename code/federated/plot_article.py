"""The federated figure as the article prints it: two panels, one shared legend.

    python3 -m federated.plot_article --root results/federated --n-parties 11

``plot_federated.py`` is the exploratory plotter -- one file per metric, legend
parked in the gutter, any metric you ask for. That layout does not survive
contact with the page: at ``0.48\\textwidth`` each panel is about 3.4 inches wide
and an eleven-entry legend beside the axes takes nearly half of it, leaving the
curves in a strip too narrow to read.

This writes the article figure instead: both panels in one full-width file, a
single legend underneath spanning both, and markers as a second channel so the
eleven methods stay distinguishable in grayscale and to a reader with a colour
vision deficiency. The output is one PDF, included with ``\\includegraphics``
rather than as two subfigures.

Deliberate departures from the exploratory plotter
--------------------------------------------------
* **Linear loss axis, clipped to the tail.** The log axis compressed the region
  where the methods actually differ into the top decade. Round 0 (loss 2.3) is
  off-scale by construction: it is the same value for every method and spending
  a third of the axis on it is what flattened everything else.
* **Markers, spaced apart per method** so that two curves within a line width of
  each other can still be told apart -- the same device Figure 1 uses.
* **No per-curve seed count in the legend.** Every curve here is five seeds; the
  caption says so once.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from matplotlib.ticker import MaxNLocator

from aggregate import aggregate_group, group_key, load_runs
from common.plotting import (FS_ANNOT, FS_LEGEND, MUTED, SURFACE, TEXT_WIDTH,
                             color_of, label_of, marker_of, order_methods,
                             save_figure, style_axes, use_paper_style)

use_paper_style()

#: (metric, axis label, clip the y-axis to the tail?)
PANELS = [
    ("test_loss", "test cross-entropy", True),
    ("test_acc", "test accuracy (%)", True),
]

#: Markers are drawn every ``MARK_EVERY`` points, offset per method so that they
#: do not stack into a vertical line at the same x.
MARK_EVERY = 4

#: Line style carries the method's *role*, so that hue is not the only channel
#: separating eleven curves. Solid is a one-bit method -- what the paper is about;
#: dashed is an uncompressed reference; dash-dot is the compressed baseline the
#: proposed methods are meant to beat. A reader can then find the comparison
#: without reading the legend at all.
REFERENCES = {"muon", "muonserver", "sgd", "adam"}
BASELINES = {"signsgd"}


def _style(alg: str) -> dict:
    if alg in REFERENCES:
        return {"linestyle": (0, (5, 2)), "linewidth": 1.2, "marker": None}
    if alg in BASELINES:
        return {"linestyle": (0, (6, 1.6, 1, 1.6)), "linewidth": 1.4}
    return {"linestyle": "-", "linewidth": 1.7}


def _tail_window(data, metric, frac: float = 0.45):
    """``(x0, x1, y0, y1)`` for an inset over the converged tail of the run.

    The tail is where every claim in the table lives and where the curves are
    within a line width of each other, so it is the only part worth magnifying.
    Bounds come from the data rather than being hard-coded, so the inset survives
    a change of horizon.
    """
    xs, ys = [], []
    for runs in data.values():
        agg = aggregate_group(runs, metric)
        if not agg or not agg["steps"]:
            continue
        cut = max(agg["steps"]) * (1.0 - frac)
        for s, m in zip(agg["steps"], agg["mean"]):
            if s >= cut:
                xs.append(s)
                ys.append(m)
    if not xs:
        return None
    pad = 0.10 * (max(ys) - min(ys)) or 0.01
    return min(xs), max(xs), min(ys) - pad, max(ys) + pad


def collect(root: Path, n_parties: int, rounds: int) -> Dict[str, List[dict]]:
    """``{algorithm: runs}``, keeping the largest same-config group per method."""
    runs = load_runs([root])
    kept: Dict[str, List[dict]] = {}
    for run in runs:
        cfg = run["config"]
        if n_parties and cfg.get("n_parties") != n_parties:
            continue
        if rounds and cfg.get("rounds") != rounds:
            continue
        if cfg.get("split") != "full":          # tuning runs never enter a figure
            continue
        alg = cfg.get("algorithm") or cfg.get("optimizer")
        if alg:
            kept.setdefault(alg, []).append(run)

    resolved = {}
    for alg, group in kept.items():
        by_cfg: Dict[tuple, List[dict]] = {}
        for run in group:
            by_cfg.setdefault(group_key(run["config"]), []).append(run)
        best = max(by_cfg.values(), key=len)
        if len(by_cfg) > 1:
            dropped = sum(len(v) for v in by_cfg.values()) - len(best)
            print(f"  ~ {alg}: {len(by_cfg)} configurations, keeping the "
                  f"{len(best)}-seed one and ignoring {dropped} run(s)")
        resolved[alg] = best
    return {a: resolved[a] for a in order_methods(resolved)}


def _draw_series(ax, data, metric, *, bands: bool, collect_handles: bool):
    """Draw every method onto ``ax``; return (handles, labels) if asked."""
    handles, labels = [], []
    for i, (alg, runs) in enumerate(data.items()):
        agg = aggregate_group(runs, metric)
        if agg is None:
            continue
        style, color = _style(alg), color_of(alg)
        mark = style.pop("marker", marker_of(alg))
        line, = ax.plot(agg["steps"], agg["mean"], color=color, zorder=3,
                        marker=mark or None, markersize=3.4,
                        markevery=(i % MARK_EVERY + 1, MARK_EVERY + 2),
                        markeredgewidth=0.0, **style)
        if bands:
            lo = [m - s for m, s in zip(agg["mean"], agg["std"])]
            hi = [m + s for m, s in zip(agg["mean"], agg["std"])]
            ax.fill_between(agg["steps"], lo, hi, color=color, alpha=0.13,
                            linewidth=0, zorder=1)
        if collect_handles:
            handles.append(line)
            labels.append(label_of(alg))
    return handles, labels


def draw(plt, data: Dict[str, List[dict]], out: Path, formats: List[str]) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.2))
    handles, labels = [], []

    for panel, (metric, ylabel, clip) in zip(axes, PANELS):
        style_axes(panel)
        h, l = _draw_series(panel, data, metric, bands=True,
                            collect_handles=metric == PANELS[0][0])
        if h:
            handles, labels = h, l
        panel.set_xlabel("communication round")
        panel.set_ylabel(ylabel)

        # The tail carries every number in the table and is where the curves sit
        # within a line width of each other, so it gets magnified rather than
        # described. Bands are omitted inside the inset: at this zoom they overlap
        # into a single wash and hide the very lines the inset exists to separate.
        win = _tail_window(data, metric)
        if win:
            x0, x1, y0, y1 = win
            # Park the inset in whichever corner the curves have vacated: a
            # decreasing loss empties the top right, a rising accuracy the bottom.
            loc = [0.40, 0.50, 0.57, 0.46] if metric.endswith("loss") \
                else [0.40, 0.11, 0.57, 0.46]
            inset = panel.inset_axes(loc, zorder=6)
            style_axes(inset)
            _draw_series(inset, data, metric, bands=False, collect_handles=False)
            inset.set_xlim(x0, x1)
            inset.set_ylim(y0, y1)
            # style_axes leaves the patch transparent, which is right for a full
            # panel and wrong for one floating over live curves.
            inset.set_facecolor(SURFACE)
            inset.patch.set_alpha(1.0)
            inset.tick_params(labelsize=FS_ANNOT - 2.0, length=2, pad=1.5)
            inset.set_xlabel("")
            inset.set_ylabel("")
            inset.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color(MUTED)
                spine.set_linewidth(0.6)
            panel.indicate_inset_zoom(inset, edgecolor=MUTED, linewidth=0.6,
                                      alpha=0.5)

        if clip:
            # Round 0 is the untrained model -- identical for every method, and
            # three times the final loss. Including it spends a third of the axis
            # on a point that carries no comparison and flattens the rest. The
            # range is taken over the BANDS, not the means, or the clip cuts the
            # very uncertainty it is there to show.
            lows, highs = [], []
            for runs in data.values():
                agg = aggregate_group(runs, metric)
                if not agg:
                    continue
                for s, m, sd in zip(agg["steps"], agg["mean"], agg["std"]):
                    if s > 0:
                        lows.append(m - sd)
                        highs.append(m + sd)
            if lows:
                lo_t, hi_t = min(lows), max(highs)
                pad = 0.06 * (hi_t - lo_t)
                panel.set_ylim(lo_t - pad, hi_t + pad)

    axes[0].set_title("band is $\\pm1$ s.d. over 5 seeds", color=MUTED,
                      fontsize=FS_ANNOT - 1.0, loc="left", pad=5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=FS_LEGEND, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.9, columnspacing=1.3, handletextpad=0.5)
    return save_figure(fig, out, "fig_federated_main", formats=formats)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/federated")
    p.add_argument("--out", default=None, help="default: <root>/figures/")
    p.add_argument("--n-parties", type=int, default=11)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    if not root.is_dir():
        print(f"No runs at {root.resolve()}.")
        return 1
    data = collect(root, args.n_parties, args.rounds)
    if not data:
        print(f"Nothing to plot under {root.resolve()} after filtering.")
        return 1
    print(f"Plotting {len(data)} method(s): {', '.join(data)}")
    written = draw(plt, data, Path(args.out) if args.out else root / "figures",
                   args.formats)
    for path in written:
        print(f"    {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
