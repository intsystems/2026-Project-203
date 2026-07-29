"""Figures for the centralized ResNet-18/CIFAR-10 study.

    python3 -m centralized.plot_analysis --bundle article_export   # the new protocol
    python3 -m centralized.plot_analysis --legacy                  # the submitted figures

Both write to ``results/analysis/``. Two input formats, one set of figure
functions:

* **``--bundle``** reads ``runs.csv`` and ``curves.csv`` from
  ``python3 -m centralized.export_article``. That is the point of the bundle: a
  few hundred KiB, so the figures can be redrawn on a laptop from the one archive
  brought off the GPU box, and every figure is a function of exactly the files the
  paper ships.
* **``--legacy``** reads `aggregate.py`'s ``table2_full.csv`` and ``curves*.json``
  from ``centralized/``. These are the files the *submitted* figures were drawn
  from, and the run tree behind them is not local, so this path stays until the
  re-run at the new protocol lands. It reproduces those figures as published --
  including the sweep panel's test accuracy at 15 epochs, which is what the old
  protocol selected on.

Three figures for the paper, one diagnostic:

| figure | paper |
| :--- | :--- |
| ``cifar_main`` | ``fig:cifar_results``, main text |
| ``curves_75ep`` | ``fig:cifar_curves_appendix`` |
| ``train_loss_75ep`` | ``fig:cifar_train_loss`` |
| ``lr_sensitivity`` | ``fig:cifar_lr`` |

All share three panels -- the Muon family, the EF21-compressed variants, and the
uncompressed baselines -- with Muon repeated in each as a gray reference, since
every comparison in the paper is against it. Each is written as both PNG and PDF,
and authored at its printed width so a point is a point.

Numbers quoted in the captions (the range-matched sweep spreads) are **printed**
at the end of a run rather than read off the plot, so a caption can be checked
against the data it describes.

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
from typing import Dict, List, Optional, Sequence, Tuple

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

#: Learning rates every method was swept over. Sweep widths differ across methods
#: -- the grid is widened for whoever's optimum lands on an endpoint -- so a raw
#: spread rewards whoever happened to be swept over the narrowest grid; the shaded
#: window is where the comparison is actually range-matched.
COMMON_WINDOW = (0.01, 0.05)

#: Where the tail row of the curve figure starts. Chosen so the zoomed row
#: spans ~4 accuracy points rather than the ~18 the full run needs.
ZOOM_FROM = 25


# --------------------------------------------------------------------------
# Loading the export bundle
# --------------------------------------------------------------------------


def _f(value: Optional[str]) -> Optional[float]:
    """A CSV cell as a float; empty and unparsable both become ``None``.

    Every derived column in ``runs.csv`` can legitimately be blank -- ``val_*`` on
    a full-split run, ``epochs_to_94`` for a method that never got there -- so a
    blank is data, not an error.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found.\nRun `python3 -m centralized.export_article` "
            f"on the machine holding results/, bring the .tar.gz over, unpack it, "
            f"and point --bundle at the unpacked directory.")
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class Bundle:
    """``runs.csv`` + ``curves.csv``, indexed the two ways the figures need."""

    #: The selection phase now ranks on validation accuracy; ``LegacyBundle``
    #: overrides this because the old protocol's sweep panel is test accuracy.
    sweep_metric = "validation"
    #: Autoscale the sweep panel: a rate two lattice steps off the optimum can sit
    #: anywhere, and a fixed window inherited from another protocol would clip it
    #: silently. ``LegacyBundle`` pins the published one instead.
    sweep_ylim: Optional[Tuple[float, float]] = None

    def __init__(self, root: Path, final_epochs: int = 75) -> None:
        self.root = root
        self.final_epochs = final_epochs
        self.runs = read_rows(root / "runs.csv")
        self.curve_rows = read_rows(root / "curves.csv")
        # Selection and reporting share a horizon under the current protocol, so
        # the sweep axis is labelled from the data rather than from a constant
        # that could disagree with it.
        self.sweep_epochs = next(
            (int(float(r["epochs"])) for r in self.runs if r["phase"] == "lr"),
            final_epochs)

    def reported(self) -> Dict[str, Dict]:
        """``{optimizer: {lr, n_seeds, acc}}`` for the reported (``final``) runs.

        A clean tree holds exactly one rate per method. If a re-tune left two, the
        better-*measured* group wins before the better-*scoring* one: a 3-seed group
        and a 1-seed group are not comparable on accuracy alone.
        """
        groups: Dict[Tuple[str, float], List[float]] = defaultdict(list)
        for r in self.runs:
            if r["phase"] != "final":
                continue
            acc, lr = _f(r["test_acc_tail"]), _f(r["lr"])
            if acc is not None and lr is not None:
                groups[(r["optimizer"], lr)].append(acc)
        out: Dict[str, Dict] = {}
        for (opt, lr), accs in groups.items():
            cand = {"lr": lr, "n_seeds": len(accs), "acc": sum(accs) / len(accs)}
            prev = out.get(opt)
            if prev is None or (cand["n_seeds"], cand["acc"]) > (prev["n_seeds"],
                                                                 prev["acc"]):
                out[opt] = cand
        return out

    def sweep(self) -> Dict[str, Dict[float, float]]:
        """``{optimizer: {eta_0: val_acc_tail}}`` from the selection phase.

        Validation accuracy, not test: these runs trained on 45k with 5k held out,
        and ``val_acc`` is the quantity selection actually ranked. Plotting their
        test accuracy instead would draw a figure of a number no decision used.
        """
        out: Dict[str, Dict[float, float]] = defaultdict(dict)
        for r in self.runs:
            if r["phase"] != "lr":
                continue
            lr, acc = _f(r["lr"]), _f(r["val_acc_tail"])
            if lr is None or acc is None:
                continue
            # A repeated (method, rate) can only be a re-run; keep the better one
            # rather than whichever the file happened to list last.
            out[r["optimizer"]][lr] = max(out[r["optimizer"]].get(lr, -1e9), acc)
        return out

    def curve(self, optimizer: str, lr: float,
              metric: str) -> Optional[Dict[str, List[float]]]:
        """Pointwise mean/std of ``metric`` over the seeds of one configuration.

        Averaged on the epochs the seeds have in common -- ``curves.csv`` carries
        the epoch index explicitly, so seeds that recorded different numbers of
        epochs line up rather than being zipped by position.
        """
        per_seed: Dict[str, Dict[int, float]] = defaultdict(dict)
        for r in self.curve_rows:
            if (r["phase"] != "final" or r["optimizer"] != optimizer
                    or _f(r["lr"]) != lr):
                continue
            val = _f(r.get(metric))
            if val is not None:
                per_seed[r["seed"]][int(float(r["epoch"]))] = val
        if not per_seed:
            return None
        common = set.intersection(*(set(d) for d in per_seed.values()))
        if not common:
            return None
        steps = sorted(common)
        mean, std = [], []
        for s in steps:
            vals = [d[s] for d in per_seed.values()]
            m = sum(vals) / len(vals)
            mean.append(m)
            std.append(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
                       if len(vals) > 1 else 0.0)
        return {"steps": steps, "mean": mean, "std": std, "n_seeds": len(per_seed)}


class LegacyBundle:
    """The pre-2026-07-29 inputs: `aggregate.py`'s summary CSV plus curve JSONs.

    The figures in the current submission were drawn from these files, and the run
    tree that produced them is not local, so this path has to keep working until
    the re-run at the new protocol lands. It is a read-only compatibility shim
    exposing the same three accessors as ``Bundle``, so there is still exactly one
    set of figure functions -- two of those would drift, and the published figures
    are the ones that must not move.

    Two differences from the bundle path are real and are surfaced rather than
    papered over: these files carry only seed-*aggregated* curves (per-seed data
    was never written out), and the sweep panel holds **test** accuracy at 15
    epochs, because that is what the old protocol selected on. The new protocol
    selects on validation accuracy at the reporting horizon; ``sweep_metric`` and
    ``sweep_epochs`` carry the difference into the axis label.
    """

    #: Runs belonging to a side experiment rather than the main comparison.
    #: ``split=tune`` held a validation set out of training, so its test accuracy
    #: is not comparable with a full-split run; ``scale_baselines=True`` applies
    #: the per-layer rule to SGD/Adam, which is a different question. Both fields
    #: appear in a label only when `aggregate.unique_labels` had to disambiguate.
    EXCLUDE = {"split": "tune", "scale_baselines": "True"}

    sweep_metric = "test"
    #: The window the submitted fig:cifar_lr was drawn in, pinned so this path
    #: reproduces it rather than merely resembling it.
    sweep_ylim = (83.5, 95.2)

    def __init__(self, root: Path, *, final_epochs: int = 75,
                 sweep_epochs: int = 15) -> None:
        self.root = root
        self.final_epochs = final_epochs
        self.sweep_epochs = sweep_epochs
        self.rows = [r for r in self._summary(root / "table2_full.csv")]
        self.curves = {
            "test_acc": self._curves(root / "curves.json", "test_acc"),
            "train_loss": self._curves(root / "curves_train_loss.json", "train_loss"),
        }

    @staticmethod
    def _fields(label: str) -> Dict[str, str]:
        return dict(tok.split("=", 1) for tok in label.split() if "=" in tok)

    def _summary(self, path: Path) -> List[Dict]:
        for r in read_rows(path):
            f = self._fields(r["label"])
            if any(f.get(k) == v for k, v in self.EXCLUDE.items()):
                continue
            # `n_seeds` was renamed to `n_runs` when the aggregator learned to
            # count runs rather than seeds; accept either, since these files
            # predate the rename and a fresh one would not.
            n = r.get("n_runs") or r.get("n_seeds") or "1"
            yield {"label": r["label"], "opt": f["optimizer"],
                   "epochs": int(f["epochs"]), "lr": float(f["lr"]),
                   "acc": float(r["final_test_acc_mean"]), "n": int(n)}

    @staticmethod
    def _curves(path: Path, want: str) -> Dict[str, Dict]:
        if not path.exists():
            print(f"[skip] {path.name} not found; figures needing {want} are omitted")
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("metric") != want:
            print(f"[warn] {path.name} holds {payload.get('metric')!r}, not {want!r}")
        return dict(payload.get("curves") or {})

    def reported(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for r in self.rows:
            if r["epochs"] != self.final_epochs:
                continue
            cand = {"lr": r["lr"], "n_seeds": r["n"], "acc": r["acc"],
                    "label": r["label"]}
            prev = out.get(r["opt"])
            # Seed count first: a 3-seed group and a 1-seed group can sit at the
            # same rate and differ only in weight decay, and the better-measured
            # one is the one to show even when the thinner one scored a shade higher.
            if prev is None or (cand["n_seeds"], cand["acc"]) > (prev["n_seeds"],
                                                                 prev["acc"]):
                out[r["opt"]] = cand
        return out

    def sweep(self) -> Dict[str, Dict[float, float]]:
        out: Dict[str, Dict[float, float]] = defaultdict(dict)
        for r in self.rows:
            if r["epochs"] == self.sweep_epochs:
                out[r["opt"]][r["lr"]] = max(out[r["opt"]].get(r["lr"], -1e9),
                                             r["acc"])
        return out

    def curve(self, optimizer: str, lr: float,
              metric: str) -> Optional[Dict[str, List[float]]]:
        entry = self.reported().get(optimizer)
        table = self.curves.get(metric) or {}
        got = table.get(entry["label"]) if entry else None
        if got is None:
            # No curve under the winning group's label. Fall back to the nearest
            # rate that has one rather than dropping the method: ranking by the
            # curve's own final value would depend on whether the metric is
            # better-high (accuracy) or better-low (loss).
            same = [(r["lr"], table[r["label"]]) for r in self.rows
                    if r["opt"] == optimizer and r["epochs"] == self.final_epochs
                    and r["label"] in table]
            if not same:
                return None
            _, got = min(same, key=lambda t: abs(math.log(t[0]) - math.log(lr)))
        return {"steps": got["steps"], "mean": got["mean"], "std": got["std"],
                "n_seeds": got.get("n_runs", got.get("n", 1))}


def best_curve(bundle, method: str, metric: str) -> Optional[Tuple[Dict, int]]:
    """``(curve, n_seeds)`` for ``method``'s reported configuration.

    The configuration is chosen from ``runs.csv`` by test accuracy, whatever metric
    the curve itself holds, so the loss figure and the accuracy figure show the
    same runs.
    """
    entry = bundle.reported().get(method)
    if entry is None:
        return None
    curve = bundle.curve(method, entry["lr"], metric)
    return (curve, curve["n_seeds"]) if curve else None


# --------------------------------------------------------------------------
# Figure 1 -- learning-rate sensitivity
# --------------------------------------------------------------------------


def window_spreads(sweep: Dict[str, Dict[float, float]]) -> Dict[str, float]:
    """Val-accuracy range of each method inside ``COMMON_WINDOW``.

    This is the quantity the caption of ``fig:cifar_lr`` quotes, and quoting it
    from the plot by eye is how a caption goes stale. Only rates every method was
    swept over count, or a method with a wider grid is penalized for the extra
    points at the ends.
    """
    lo, hi = COMMON_WINDOW
    out = {}
    for method, pts in sweep.items():
        vals = [a for lr, a in pts.items() if lo <= lr <= hi]
        if len(vals) >= 2:
            out[method] = max(vals) - min(vals)
    return out


def figure_lr_sensitivity(bundle, out: Path) -> Path:
    sweep = bundle.sweep()
    ref = sorted(sweep.get(REF_METHOD, {}).items())

    # Shared x: the sweeps cover different learning-rate ranges, and a panel that
    # stops early should look like it stopped early rather than be rescaled to
    # fill the axes.
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.6), sharey=True, sharex=True)

    for ax, (_, methods) in zip(axes, PANELS):
        style_axes(ax)
        ax.set_xscale("log")
        ax.axvspan(*COMMON_WINDOW, color=GRID, alpha=0.55, zorder=0, linewidth=0)

        if ref:
            ax.plot([lr for lr, _ in ref], [a for _, a in ref], color=REFERENCE,
                    linewidth=1.4, linestyle=(0, (4, 2)), marker="o", markersize=2.8,
                    zorder=2, label=PRETTY[REF_METHOD])

        for color, method in zip(SERIES, methods):
            pts = sorted(sweep.get(method, {}).items())
            if not pts:
                continue
            ax.plot([lr for lr, _ in pts], [a for _, a in pts], color=color,
                    linewidth=1.8, marker="o", markersize=3.4,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3,
                    label=PRETTY[method])

        panel_legend(ax, "lower right")
        ax.set_xlabel("learning rate (log scale)", color=INK_2, fontsize=FS_LABEL)

    axes[0].set_ylabel(f"{bundle.sweep_metric} accuracy (%) at "
                       f"{bundle.sweep_epochs} epochs",
                       color=INK_2, fontsize=FS_LABEL)
    axes[0].set_xlim(1.2e-4, 1.5)
    if bundle.sweep_ylim:
        axes[0].set_ylim(*bundle.sweep_ylim)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.185, wspace=0.16)
    return _save(fig, out / "lr_sensitivity.png")


# --------------------------------------------------------------------------
# Figures 2-4 -- learning curves
# --------------------------------------------------------------------------


def _draw(ax, curve: Dict, color, name: str, *, start: int, is_ref: bool = False,
          floor: Optional[float] = None, width: Tuple[float, float] = (1.4, 1.8)
          ) -> None:
    """One mean curve with its +/-1 sd band, from ``start`` onwards.

    Dashed means one seed, so a gap between two dashed lines carries no dispersion
    information at all and should not read as one. ``floor`` clamps the lower edge
    of the band, for the log-scale loss panel where ``mean - std`` goes negative
    near the end and a log axis silently drops the point, breaking the band open.
    """
    pts = [(s, m, d) for s, m, d in
           zip(curve["steps"], curve["mean"], curve["std"])
           if s >= start and (floor is None or m > 0)]
    if not pts:
        return
    xs = [s for s, _, _ in pts]
    ys = [m for _, m, _ in pts]
    sd = [d for _, _, d in pts]
    dashed = curve["n_seeds"] < 2 or is_ref
    ax.plot(xs, ys, color=color, linewidth=width[0] if is_ref else width[1],
            linestyle=(0, (4, 2)) if dashed else "-",
            zorder=2 if is_ref else 3, label=PRETTY[name])
    if curve["n_seeds"] > 1:
        low = [max(m - d, m / 4) if floor is not None else m - d
               for m, d in zip(ys, sd)]
        ax.fill_between(xs, low, [m + d for m, d in zip(ys, sd)],
                        color=color, alpha=0.18, linewidth=0, zorder=1)


def _panel_series(bundle, methods: Sequence[str], metric: str):
    """``(color, method, curve, is_ref)`` for one panel, reference line first."""
    ref = best_curve(bundle, REF_METHOD, metric)
    if ref:
        yield REFERENCE, REF_METHOD, ref[0], True
    for color, method in zip(SERIES, methods):
        got = best_curve(bundle, method, metric)
        if got:
            yield color, method, got[0], False


def figure_curves(bundle, out: Path, epochs: int, first_epoch: int = 3) -> Path:
    """Test accuracy at two scales.

    Row 0 is the whole run, row 1 the tail. At full scale every method lands
    inside a couple of points and the ordering is unreadable; at tail scale the
    early separation -- which is most of what distinguishes the baselines panel --
    is off-screen. Neither row alone says it.
    """
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 4.6), squeeze=False)

    for row, (start, ylim, row_name) in enumerate([
            (first_epoch, (78, 96.5), "full run"),
            (ZOOM_FROM, (90.0, 95.2), f"final {epochs - ZOOM_FROM} epochs")]):
        for col, (_, methods) in enumerate(PANELS):
            ax = axes[row][col]
            style_axes(ax)
            for color, method, curve, is_ref in _panel_series(bundle, methods,
                                                              "test_acc"):
                _draw(ax, curve, color, method, start=start, is_ref=is_ref)
            if row == 0:
                panel_legend(ax, "lower right")
            else:
                ax.set_xlabel("epoch", color=INK_2, fontsize=FS_LABEL)
            ax.set_xlim(start - 1, epochs + 1)
            ax.set_ylim(*ylim)
        axes[row][0].set_ylabel(f"test accuracy (%)\n({row_name})",
                                color=INK_2, fontsize=FS_LABEL)

    fig.subplots_adjust(left=0.112, right=0.985, top=0.975, bottom=0.085,
                        wspace=0.19, hspace=0.20)
    return _save(fig, out / "curves_75ep.png")


def figure_train_loss(bundle, out: Path, epochs: int,
                      first_epoch: int = 1) -> Path:
    """Training loss on a log axis.

    Log, not linear: the loss falls from 2.3 to under 1e-3, and on a linear axis
    everything after epoch 10 is a flat line on zero. The same three panels and
    the same configurations as the accuracy figure, so the two can be read side
    by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.7), squeeze=False)

    for col, (_, methods) in enumerate(PANELS):
        ax = axes[0][col]
        style_axes(ax)
        ax.set_yscale("log")
        for color, method, curve, is_ref in _panel_series(bundle, methods,
                                                          "train_loss"):
            _draw(ax, curve, color, method, start=first_epoch, is_ref=is_ref,
                  floor=0.0)
        panel_legend(ax, "upper right")
        ax.set_xlabel("epoch", color=INK_2, fontsize=FS_LABEL)
        ax.set_xlim(first_epoch - 1, epochs + 1)
        ax.set_ylim(1e-4, 3.0)

    axes[0][0].set_ylabel("training loss (log scale)", color=INK_2, fontsize=FS_LABEL)

    fig.subplots_adjust(left=0.078, right=0.985, top=0.975, bottom=0.175, wspace=0.17)
    return _save(fig, out / "train_loss_75ep.png")


def figure_main(bundle, out: Path, epochs: int) -> Path:
    """Test accuracy, three panels, for the paper body.

    One metric and one row: the body figure has a page budget, and the training
    loss it used to carry underneath separates the methods less than the accuracy
    does -- it is in the appendix (``train_loss_75ep``) for the reader who wants
    it. Accuracy is shown only from ``ZOOM_FROM``, where the methods are a few
    points apart instead of twenty; the early rise is in the appendix version.
    """
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), squeeze=False)

    for col, (_, methods) in enumerate(PANELS):
        ax = axes[0][col]
        style_axes(ax)
        for color, method, curve, is_ref in _panel_series(bundle, methods, "test_acc"):
            # A shade heavier than the appendix figures: this one is printed at
            # the same width but read at a glance, and it is the only figure whose
            # lines are the argument rather than the evidence.
            _draw(ax, curve, color, method, start=ZOOM_FROM, is_ref=is_ref,
                  width=(1.5, 1.9))
        # A legend in an empty corner, not labels in the gutter: at the printed
        # width of this figure a gutter wide enough for "EF21-MuonUSign" would
        # take a third of the panel.
        panel_legend(ax, "lower right")
        ax.set_xlabel("epoch", color=INK_2, fontsize=FS_LABEL)
        ax.set_xlim(ZOOM_FROM - 1, epochs + 1)
        # Auto ticks put the first label at 40, a third of the way in; the axis
        # starts at 25 and should say so.
        ax.set_xticks([30, 40, 50, 60, 70])
        ax.set_ylim(90.0, 95.2)
    axes[0][0].set_ylabel("test accuracy (%)", color=INK_2, fontsize=FS_LABEL)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.175, wspace=0.17)
    return _save(fig, out / "cifar_main.png")


def _save(fig, path: Path) -> Path:
    for target in (path, path.with_suffix(".pdf")):
        fig.savefig(target, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path,
                   default=here.parent / "results" / "article_export",
                   help="Directory written by centralized.export_article "
                        "(default: results/article_export)")
    p.add_argument("--legacy", type=Path, nargs="?", const=here, default=None,
                   metavar="DIR",
                   help="Draw from aggregate.py's table2_full.csv + curves*.json "
                        "instead (default DIR: centralized/). This is the path that "
                        "reproduces the submitted figures, whose run tree is not "
                        "local; it selects on test accuracy at --sweep-epochs, as "
                        "the old protocol did")
    p.add_argument("--out", type=Path, default=here.parent / "results" / "analysis")
    p.add_argument("--epochs", type=int, default=75,
                   help="Reporting horizon, for the axis limits")
    p.add_argument("--sweep-epochs", type=int, default=15,
                   help="--legacy only: the horizon its learning-rate sweep was run "
                        "at. The bundle path reads this off the data instead")
    args = p.parse_args()

    if args.legacy is not None:
        bundle = LegacyBundle(args.legacy, final_epochs=args.epochs,
                              sweep_epochs=args.sweep_epochs)
        source, n = args.legacy, len(bundle.rows)
    else:
        bundle = Bundle(args.bundle, final_epochs=args.epochs)
        source, n = args.bundle, len(bundle.runs)
    args.out.mkdir(parents=True, exist_ok=True)

    reported = bundle.reported()
    thin = sorted(m for m, d in reported.items() if d["n_seeds"] < 2)
    print(f"{n} rows in {source}; {len(reported)} methods reported")
    if thin:
        print(f"[warn] single seed, drawn dashed and with no band: {', '.join(thin)}")
    missing = [m for _, ms in PANELS for m in ms if m not in reported]
    if missing:
        print(f"[warn] no final runs for: {', '.join(missing)} -- panels drawn "
              f"without them")

    for path in (figure_main(bundle, args.out, args.epochs),
                 figure_curves(bundle, args.out, args.epochs),
                 figure_train_loss(bundle, args.out, args.epochs),
                 figure_lr_sensitivity(bundle, args.out)):
        print(f"Wrote {path} (+ .pdf)")

    # The caption of fig:cifar_lr quotes these; print them so it can be checked
    # rather than remembered.
    spreads = window_spreads(bundle.sweep())
    if spreads:
        lo, hi = COMMON_WINDOW
        print(f"\n{bundle.sweep_metric}-accuracy spread within the range-matched "
              f"window [{lo:g}, {hi:g}] -- quoted in the fig:cifar_lr caption:")
        for method, spread in sorted(spreads.items(), key=lambda kv: kv[1]):
            print(f"  {PRETTY.get(method, method):<16}{spread:6.2f} points")


if __name__ == "__main__":
    main()
