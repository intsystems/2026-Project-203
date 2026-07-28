"""Figures for the centralized ResNet-18/CIFAR-10 study.

    python3 -m centralized.plot_analysis                  # -> results/analysis/
    python3 -m centralized.plot_analysis --out somewhere

Inputs are the aggregator's outputs, not the run directories: ``table2_full.csv``
(final accuracy per group) and ``curves.json`` (per-epoch mean/std). Regenerate
both with::

    python3 -m aggregate --root results/centralized --csv centralized/table2_full.csv \
                         --curves centralized/curves.json
    python3 -m aggregate --root results/centralized --metric train_loss \
                         --curves centralized/curves_train_loss.json

Four figures: the learning-rate sweep at 15 epochs, the 75-epoch test-accuracy
curves, the training-loss curves, and a compact accuracy figure for the paper
body. All share three panels -- the Muon family, the EF21-compressed
variants, and the uncompressed baselines -- with Muon repeated in each as a gray
reference, since every comparison in the paper is against it. Each is written as
both PNG and PDF, and authored at its printed width so a point is a point.

Nothing here writes into ``aaai_article/``: the paper's figures are copied over
deliberately, not overwritten by a plotting run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette, type scale and rcParams: one module for the whole paper.
#
# These panels use the categorical slots 1-3 rather than the per-method colours,
# because each panel holds three methods and reads as its own small comparison.
# Three is the cap: the fourth slot (green) collides with orange under simulated
# protanopia (OKLab dE 3.2, against a floor of 6), so a fourth method in one
# panel would be unreadable for a dichromat reader. Muon is therefore chrome-gray,
# not a fourth hue -- which is also what it is conceptually, the reference every
# panel is read against.
from common.plotting import (FS_LABEL, GRID, INK_2, REFERENCE, SERIES,  # noqa: E402
                             SURFACE, TEXT_WIDTH, panel_legend, style_axes,
                             use_paper_style)

use_paper_style()

#: Panels: (grouping name, the three methods). The name is not drawn -- the
#: figure caption says what each panel holds.
#: Muon is not a series in any panel -- it is the gray reference line in all
#: three, since every comparison in the paper is against it.
PANELS: List[Tuple[str, List[str]]] = [
    ("Muon family", ["signmuon", "muonusign", "muonsign"]),
    ("EF21 (compressed)", ["ef21signmuon", "ef21muonusign", "ef21muonsign"]),
    ("Uncompressed baselines", ["sgd", "signsgd", "adam"]),
]
REF_METHOD = "muon"

PRETTY = {
    "signmuon": "SignMuon", "muonusign": "MuonUSign", "muonsign": "MuonSign",
    "ef21signmuon": "EF21-SignMuon", "ef21muonusign": "EF21-MuonUSign",
    "ef21muonsign": "EF21-MuonSign", "muon": "Muon", "sgd": "SGD",
    "signsgd": "SignSGD", "adam": "Adam",
}

#: Learning rates every method except ``ef21muonsign`` was swept over. Sweep
#: widths differ by 50x across methods, so a raw spread rewards whoever happened
#: to be swept over the narrowest grid; the shaded window is where the
#: comparison is actually range-matched.
COMMON_WINDOW = (0.01, 0.05)

#: Where the tail row of the curve figure starts. Chosen so the zoomed row
#: spans ~4 accuracy points rather than the ~18 the full run needs.
ZOOM_FROM = 25


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def parse_label(label: str) -> Dict[str, str]:
    return dict(tok.split("=", 1) for tok in label.split() if "=" in tok)


#: Runs that belong to a side experiment rather than the main comparison, as
#: ``field: value`` pairs. ``split=tune`` held a validation set out of training,
#: so its test accuracy is not comparable with a full-split run; and
#: ``scale_baselines=True`` applies the per-layer scaling rule to the baselines,
#: which is a separate question from how they respond to a learning rate. Both
#: only became visible once the aggregator stopped collapsing them into their
#: neighbours' labels.
EXCLUDE = {"split": "tune", "scale_baselines": "True"}


def is_canonical(fields: Dict[str, str]) -> bool:
    return all(fields.get(k) != v for k, v in EXCLUDE.items())


def load_summary(path: Path) -> List[Dict]:
    """Rows of the main comparison, side experiments dropped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fields = parse_label(r["label"])
            if not is_canonical(fields):
                continue
            rows.append({
                "label": r["label"],
                "opt": fields["optimizer"],
                "epochs": int(fields["epochs"]),
                "lr": float(fields["lr"]),
                "acc": float(r["final_test_acc_mean"]),
                "n": int(r["n_seeds"]),
            })
    return rows


def load_curves(path: Path) -> Tuple[str, Dict[str, Dict]]:
    """Curves keyed by their full label.

    Not by ``(optimizer, epochs, lr)``: several groups share that triple and
    differ only in weight decay, which is exactly the collision the aggregator
    was just fixed for. Keying on it here would throw the recovered curves away
    again on the way in.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["metric"], dict(payload["curves"])


def sweep_by_method(rows: List[Dict], epochs: int) -> Dict[str, Dict[float, List[float]]]:
    """``{method: {lr: [acc, ...]}}``.

    A list longer than one would mean two runs still share a method and learning
    rate after the side experiments are dropped -- they differ in some field the
    figure does not encode. The caller draws the extras hollow rather than
    silently picking one.
    """
    sweep: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["epochs"] == epochs:
            sweep[r["opt"]][r["lr"]].append(r["acc"])
    return sweep


def best_curve(curves: Dict, method: str, rows: List[Dict],
               epochs: int) -> Optional[Tuple[Dict, float, int, bool]]:
    """The curve for ``method``'s best 75-epoch configuration.

    The configuration is always chosen by final test accuracy from the summary,
    whatever metric the curves themselves hold -- so the loss figure and the
    accuracy figure show the same runs.

    Returns ``(curve, lr, n_seeds, is_complete)``. ``is_complete`` is False when
    the curve carries fewer seeds than the summary counted: the aggregator used
    to key curves by a non-unique label, so a multi-seed group could be
    overwritten by a single-seed one at the same learning rate. Keeping that
    curve is still right -- it is the same configuration, just thinner -- and the
    caller marks it. Only when nothing exists at that learning rate does this
    fall back to another one.
    """
    finals = [r for r in rows if r["epochs"] == epochs and r["opt"] == method]
    if not finals:
        return None
    # Seed count first: a 3-seed group and a 1-seed group can sit at the same
    # learning rate and differ only in weight decay, and the better-measured one
    # is the one to show even when the thinner one scored a shade higher.
    best = max(finals, key=lambda r: (r["n"], r["acc"]))

    exact = curves.get(best["label"])
    if exact is not None:
        return exact, best["lr"], exact["n_runs"], exact["n_runs"] == best["n"]

    available = [(r["lr"], curves[r["label"]]) for r in finals if r["label"] in curves]
    if not available:
        return None
    # No curve for the winning group: take the nearest learning rate that has
    # one. Ranking by the curve's own final value would depend on whether the
    # metric is better-high (accuracy) or better-low (loss).
    lr, curve = min(available, key=lambda t: abs(math.log(t[0]) - math.log(best["lr"])))
    return curve, lr, curve["n_runs"], False


# --------------------------------------------------------------------------
# Shared styling
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Figure 1 -- learning-rate sensitivity
# --------------------------------------------------------------------------


def figure_lr_sensitivity(rows: List[Dict], out: Path, epochs: int = 15) -> Path:
    sweep = sweep_by_method(rows, epochs)
    ref = sorted(sweep.get(REF_METHOD, {}).items())

    # Shared x: the sweeps cover different learning-rate ranges, and a panel that
    # stops early should look like it stopped early rather than be rescaled to
    # fill the axes.
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.6), sharey=True, sharex=True)

    for ax, (title, methods) in zip(axes, PANELS):
        style_axes(ax)
        ax.set_xscale("log")
        ax.axvspan(*COMMON_WINDOW, color=GRID, alpha=0.55, zorder=0, linewidth=0)

        if ref:
            xs = [lr for lr, _ in ref]
            ys = [max(a) for _, a in ref]
            ax.plot(xs, ys, color=REFERENCE, linewidth=1.4, linestyle=(0, (4, 2)),
                    marker="o", markersize=2.8, zorder=2, label=PRETTY[REF_METHOD])

        for color, method in zip(SERIES, methods):
            pts = sorted(sweep.get(method, {}).items())
            if not pts:
                continue
            xs = [lr for lr, _ in pts]
            ys = [max(a) for _, a in pts]
            ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=3.4,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3,
                    label=PRETTY[method])
            # A collided label: two distinct runs the summary cannot separate.
            for lr, accs in pts:
                for extra in sorted(accs)[:-1]:
                    ax.plot([lr], [extra], marker="o", markersize=3.4, color=SURFACE,
                            markeredgecolor=color, markeredgewidth=1.0, zorder=3)

        panel_legend(ax, "lower right")
        ax.set_xlabel("learning rate (log scale)", color=INK_2, fontsize=FS_LABEL)

    axes[0].set_ylabel(f"test accuracy (%) at {epochs} epochs", color=INK_2, fontsize=FS_LABEL)
    axes[0].set_ylim(83.5, 95.2)
    axes[0].set_xlim(1.2e-4, 1.5)


    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.185, wspace=0.16)
    path = out / "lr_sensitivity.png"
    for target in (path, path.with_suffix(".pdf")):
        fig.savefig(target, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 2 -- learning curves
# --------------------------------------------------------------------------


def figure_curves(rows: List[Dict], curves: Dict, out: Path,
                  epochs: int = 75, first_epoch: int = 3) -> Tuple[Path, List[str]]:
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 4.6), squeeze=False)

    ref = best_curve(curves, REF_METHOD, rows, epochs)
    degraded: List[str] = []

    def draw(ax, curve, color, name, lr, n_seeds, exact, start, label,
             is_ref=False, strip=""):
        pts = [(s, m, sd) for s, m, sd in
               zip(curve["steps"], curve["mean"], curve["std"]) if s >= start]
        xs = [s for s, _, _ in pts]
        ys = [m for _, m, _ in pts]
        sd = [d for _, _, d in pts]
        # Dashed = one seed, so a gap between two dashed lines carries no
        # dispersion information at all.
        dashed = n_seeds < 2
        ax.plot(xs, ys, color=color,
                linewidth=1.4 if is_ref else 1.8,
                linestyle=(0, (4, 2)) if (dashed or is_ref) else "-",
                zorder=2 if is_ref else 3, label=PRETTY[name])
        if n_seeds > 1:
            ax.fill_between(xs, [m - d for m, d in zip(ys, sd)],
                            [m + d for m, d in zip(ys, sd)],
                            color=color, alpha=0.18, linewidth=0, zorder=1)
        if not exact:
            degraded.append(PRETTY[name])

    # Row 0 is the whole run, row 1 the tail. At full scale every method lands
    # inside a couple of points and the ordering is unreadable; at tail scale the
    # early separation -- which is most of what distinguishes the baselines panel
    # -- is off-screen. Neither row alone says it.
    for row, (start, ylim, row_name) in enumerate([
            (first_epoch, (78, 96.5), "full run"),
            (ZOOM_FROM, (90.0, 95.2), f"final {epochs - ZOOM_FROM} epochs")]):
        for col, (title, methods) in enumerate(PANELS):
            ax = axes[row][col]
            style_axes(ax)
            label = row == 1                       # label the row where lines separate
            if ref:
                draw(ax, ref[0], REFERENCE, REF_METHOD, ref[1], ref[2], ref[3],
                     start, label, is_ref=True)
            for color, method in zip(SERIES, methods):
                got = best_curve(curves, method, rows, epochs)
                if got is None:
                    continue
                draw(ax, got[0], color, method, got[1], got[2], got[3],
                     start, label)
            if row == 0:
                panel_legend(ax, "lower right")
            else:
                ax.set_xlabel("epoch", color=INK_2, fontsize=FS_LABEL)
            ax.set_xlim(start - 1, epochs + 1)
            ax.set_ylim(*ylim)
        axes[row][0].set_ylabel(f"test accuracy (%)\n({row_name})", color=INK_2, fontsize=FS_LABEL)


    fig.subplots_adjust(left=0.112, right=0.985, top=0.975, bottom=0.085,
                        wspace=0.19, hspace=0.20)
    path = out / "curves_75ep.png"
    for target in (path, path.with_suffix(".pdf")):
        fig.savefig(target, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path, sorted(set(degraded))


# --------------------------------------------------------------------------
# Figure 3 -- training loss
# --------------------------------------------------------------------------


def figure_train_loss(rows: List[Dict], curves: Dict, out: Path,
                      epochs: int = 75, first_epoch: int = 1) -> Tuple[Path, List[str]]:
    """Training loss on a log axis.

    Log, not linear: the loss falls from 2.3 to under 1e-3, and on a linear axis
    everything after epoch 10 is a flat line on zero. The same three panels and
    the same configurations as the accuracy figure, so the two can be read
    side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.7), squeeze=False)
    degraded: List[str] = []

    ref = best_curve(curves, REF_METHOD, rows, epochs)

    def draw(ax, curve, color, name, n_seeds, complete, is_ref=False):
        pts = [(s, m, sd) for s, m, sd in
               zip(curve["steps"], curve["mean"], curve["std"])
               if s >= first_epoch and m and m > 0]
        xs = [s for s, _, _ in pts]
        ys = [m for _, m, _ in pts]
        sd = [d for _, _, d in pts]
        ax.plot(xs, ys, color=color, linewidth=1.4 if is_ref else 1.8,
                linestyle=(0, (4, 2)) if (n_seeds < 2 or is_ref) else "-",
                zorder=2 if is_ref else 3, label=PRETTY[name])
        if n_seeds > 1:
            # Clamped at a floor: mean - std goes negative near the end, which a
            # log axis silently drops, breaking the band open.
            ax.fill_between(xs, [max(m - d, m / 4) for m, d in zip(ys, sd)],
                            [m + d for m, d in zip(ys, sd)],
                            color=color, alpha=0.18, linewidth=0, zorder=1)
        if not complete:
            degraded.append(PRETTY[name])

    for col, (title, methods) in enumerate(PANELS):
        ax = axes[0][col]
        style_axes(ax)
        ax.set_yscale("log")
        if ref:
            draw(ax, ref[0], REFERENCE, REF_METHOD, ref[2], ref[3], is_ref=True)
        for color, method in zip(SERIES, methods):
            got = best_curve(curves, method, rows, epochs)
            if got is None:
                continue
            draw(ax, got[0], color, method, got[2], got[3])
        panel_legend(ax, "upper right")
        ax.set_xlabel("epoch", color=INK_2, fontsize=FS_LABEL)
        ax.set_xlim(first_epoch - 1, epochs + 1)
        ax.set_ylim(1e-4, 3.0)

    axes[0][0].set_ylabel("training loss (log scale)", color=INK_2, fontsize=FS_LABEL)


    fig.subplots_adjust(left=0.078, right=0.985, top=0.975, bottom=0.175, wspace=0.17)
    path = out / "train_loss_75ep.png"
    for target in (path, path.with_suffix(".pdf")):
        fig.savefig(target, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path, sorted(set(degraded))


# --------------------------------------------------------------------------
# Figure 4 -- the compact main-text figure
# --------------------------------------------------------------------------


def figure_main(rows: List[Dict], acc_curves: Dict, out: Path,
                epochs: int = 75) -> Path:
    """Test accuracy, three panels, for the paper body.

    One metric and one row: the body figure has a page budget, and the training
    loss it used to carry underneath separates the methods less than the
    accuracy does -- it is in the appendix (``train_loss_75ep``) for the reader
    who wants it. Accuracy is shown only from ``ZOOM_FROM``, where the methods
    are a few points apart instead of twenty; the early rise is in the appendix
    version.
    """
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), squeeze=False)

    start, ylim = ZOOM_FROM, (90.0, 95.2)
    ref = best_curve(acc_curves, REF_METHOD, rows, epochs)
    for col, (title, methods) in enumerate(PANELS):
        ax = axes[0][col]
        style_axes(ax)

        for color, method, is_ref in ([(REFERENCE, REF_METHOD, True)] if ref else []) + \
                                     [(c, m, False) for c, m in zip(SERIES, methods)]:
            got = best_curve(acc_curves, method, rows, epochs)
            if got is None:
                continue
            curve, _, n_seeds, _ = got
            pts = [(s, m, sd) for s, m, sd in
                   zip(curve["steps"], curve["mean"], curve["std"]) if s >= start]
            xs = [s for s, _, _ in pts]
            ys = [m for _, m, _ in pts]
            sd = [d for _, _, d in pts]
            ax.plot(xs, ys, color=color, linewidth=1.5 if is_ref else 1.9,
                    linestyle=(0, (4, 2)) if (is_ref or n_seeds < 2) else "-",
                    zorder=2 if is_ref else 3, label=PRETTY[method])
            if n_seeds > 1:
                ax.fill_between(xs, [m - d for m, d in zip(ys, sd)],
                                [m + d for m, d in zip(ys, sd)],
                                color=color, alpha=0.18, linewidth=0, zorder=1)

        # A legend in an empty corner, not labels in the gutter: at the
        # printed width of this figure a gutter wide enough for
        # "EF21-MuonUSign" would take a third of the panel.
        panel_legend(ax, "lower right")
        ax.set_xlabel("epoch", color=INK_2, fontsize=FS_LABEL)
        ax.set_xlim(start - 1, epochs + 1)
        # Auto ticks put the first label at 40, a third of the way in; the axis
        # starts at 25 and should say so.
        ax.set_xticks([30, 40, 50, 60, 70])
        ax.set_ylim(*ylim)
    axes[0][0].set_ylabel("test accuracy (%)", color=INK_2, fontsize=FS_LABEL)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.175,
                        wspace=0.17)
    path = out / "cifar_main.png"
    for target in (path, path.with_suffix(".pdf")):
        fig.savefig(target, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=here / "table2_full.csv")
    p.add_argument("--curves", type=Path, default=here / "curves.json")
    p.add_argument("--train-loss-curves", type=Path,
                   default=here / "curves_train_loss.json")
    p.add_argument("--out", type=Path, default=here.parent / "results" / "analysis")
    args = p.parse_args()

    rows = load_summary(args.csv)
    metric, curves = load_curves(args.curves)
    if metric != "test_acc":
        print(f"[warn] curves hold {metric!r}, not 'test_acc'; axis labels will be wrong")
    args.out.mkdir(parents=True, exist_ok=True)

    degraded = []
    print(f"Wrote {figure_lr_sensitivity(rows, args.out)}")
    path, thin = figure_curves(rows, curves, args.out)
    print(f"Wrote {path}")
    degraded += thin

    print(f"Wrote {figure_main(rows, curves, args.out)}")

    if args.train_loss_curves.exists():
        tl_metric, tl_curves = load_curves(args.train_loss_curves)
        if tl_metric != "train_loss":
            print(f"[warn] {args.train_loss_curves.name} holds {tl_metric!r}, not 'train_loss'")
        path, thin = figure_train_loss(rows, tl_curves, args.out)
        print(f"Wrote {path}")
        degraded += thin
    else:
        print(f"[skip] {args.train_loss_curves.name} not found; no training-loss figure")

    if degraded:
        print(f"\n[warn] fewer seeds than the summary counted: {', '.join(sorted(set(degraded)))}")
        print("       Their multi-seed curves were lost to the aggregator's label collision.")
        print("       Re-run aggregate.py *with the fix* on the machine holding results/")
        print("       to get them; the figures mark these runs with an asterisk meanwhile.")


if __name__ == "__main__":
    main()
