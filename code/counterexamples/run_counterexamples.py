"""
Run all eight optimizers on the two linear counterexamples and report which
methods diverge.

Usage
-----
Run from ``code/``, the package root:

    python3 -m counterexamples.run_counterexamples             # default mu=0.0
    python3 -m counterexamples.run_counterexamples --nesterov  # Nesterov momentum
    python3 -m counterexamples.run_counterexamples --mu 0.9 --T 80 --eta 2e-3

Outputs
-------
* a per-problem table of  f[0], f[T-1], the mean per-step descent inner product
  <G, d_t>, and a DIVERGES / descends verdict, and
* three figures -- ``signmuon_counterexample``, ``muonsign_counterexample`` and
  ``ef21_signmuon_counterexample`` -- written as PNG + PDF to ``figures/`` and as
  PDF to ``../aaai_article/images/counterexamples/``, which is what LaTeX
  includes.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")            # headless-safe; figures are written to disk
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from counterexamples.optimizers import OPTIMIZERS, PAPER_METHODS, REFERENCE_METHODS  # noqa: E402
from counterexamples.problems import (  # noqa: E402
    make_linear_problem,
    signmuon_counterexample,
    muonsign_counterexample,
    ef21_signmuon_counterexample,
)

# Per-method plot style (colours/markers consistent with the paper figures).
# Keys are the *internal* algorithm names used by ``optimizers.py``.
STYLES = {
    "SignMuon":        ("#1f77b4", "^", "-"),
    "EF21-SignMuon":   ("#8c564b", "D", (0, (5.5, 5.6))),
    "MuonUSign":       ("#2ca02c", "s", "-"),
    "MuonSign":      ("#e377c2", "v", (0, (3, 1, 1, 1))),
    "EF21-MuonUSign":  ("#4A3322", "o", (0, (4, 2))),
    "EF21-MuonSign": ("#10C5D5", "P", (0, (3, 1))),
    "SignSGD":         ("#9467bd", "*", "-"),
    "Muon":            ("#999999", None, (0, (1, 2))),
}

# Map the internal algorithm names to the paper's display names.  The code's
# "sign before *and* after the LMO" method (``MuonSign``) is the paper's
# ``MuonSign`` (Theorem 3), and its error-feedback counterpart ``EF21-MuonSign``
# is the paper's bidirectional ``EF21-MuonSign``.  Everything else is unchanged.
DISPLAY_NAMES = {
    "SignMuon":        "SignMuon",
    "EF21-SignMuon":   "EF21-SignMuon",
    "MuonUSign":       "MuonUSign",
    "MuonSign":      "MuonSign",
    "EF21-MuonUSign":  "EF21-MuonUSign",
    "EF21-MuonSign": "EF21-MuonSign",
    "SignSGD":         "SignSGD",
    "Muon":            "Muon",
}

# Divergence tests.  For a LINEAR objective f(W)=Tr(G^T W), f decreases iff the
# per-step descent inner product <G, d_t> is positive, so the exact, eta/T-free
# test is "mean <G, d_t> persistently negative" (verdict_mode="inner").  For the
# universal EF21-SignMuon instance (periodic ramps, not a quadratic) the ascent is
# second-order: <G, d_t> stays POSITIVE while f rises, so we instead test the tail
# slope of f (verdict_mode="slope").
#
# SLOPE_TOL is ABSOLUTE, and the instance's periodic term scales with A(mu) -- 199
# at mu=0.99.  The slope verdict is therefore only discriminating at small mu; the
# mu-independent check of the exact 49/480 rate is counterexamples.problems, which
# measures over whole periods.
DIVERGE_TOL = 1e-6
SLOPE_TOL = 1e-3

# Shared visual style (kept in sync with plot_ef21_momentum.py).
LABEL_FS, TICK_FS, LEG_FS = 18, 13, 12.5
# Iterations shown in the magnified panel: past this the bounded methods just
# repeat their period-2 cycle, so a short window keeps the panel readable.
MAGNIFIED_WINDOW = 60


def run(opt_cls, grad_fn, loss_fn, shape, T, eta, mu, nesterov):
    """Run one optimizer; return (losses, mean per-step <G, d_t>)."""
    opt = opt_cls(shape, eta=eta, mu=mu, nesterov=nesterov)
    losses = []
    inner = []
    for _ in range(T):
        prev = opt.track_point().copy()
        losses.append(loss_fn(opt.track_point()))
        G = grad_fn(opt.grad_point())
        opt.step(G)
        # d_t reconstructed from the tracked-model shift:  X_{t+1} = X_t - eta*d_t
        d_t = (prev - opt.track_point()) / eta
        inner.append(float(np.sum(G * d_t)))
    return np.asarray(losses), float(np.mean(inner))


def run_problem(title, grad_fn, loss_fn, shape, T, eta, mu, nesterov,
                verdict_mode="inner"):
    """Run every optimizer on one problem, print a table.

    ``verdict_mode`` selects the divergence test: ``"inner"`` (mean <G,d> < 0,
    for the linear objectives) or ``"slope"`` (positive tail slope of f, for the
    curvature-driven EF21-SignMuon instance).

    Returns ``(results, diverging)`` where ``results`` maps method -> loss
    trajectory and ``diverging`` lists the methods that diverge.
    """
    order = PAPER_METHODS + REFERENCE_METHODS

    results, diverging = {}, []
    print(f"\n{title}")
    print(f"  dim={shape}, eta={eta}, mu={mu}, "
          f"momentum={'Nesterov' if nesterov else 'standard'}, T={T}, "
          f"test={verdict_mode}")
    print(f"  {'method':<17}{'f[0]':>9}{'f[T-1]':>12}{'mean<G,d>':>12}"
          f"{'tail slope':>12}   verdict")
    print("  " + "-" * 76)
    for name in order:
        losses, mean_inner = run(OPTIMIZERS[name], grad_fn, loss_fn,
                                 shape, T, eta, mu, nesterov)
        results[name] = losses
        # Measure over a whole number of periods: the EF21-SignMuon trajectory
        # is period-two, so mismatched endpoint parity leaks half an oscillation
        # into the slope and hides the exact rate.
        half = T // 2
        half -= (T - 1 - half) % 2
        slope = float((losses[-1] - losses[half]) / max(1, (T - 1 - half))) / eta
        if verdict_mode == "slope":
            is_div = slope > SLOPE_TOL
        else:
            is_div = mean_inner < -DIVERGE_TOL
        if is_div:
            diverging.append(name)
        verdict = "DIVERGES (up)" if is_div else "descends"
        print(f"  {name:<17}{losses[0]:>9.3f}{losses[-1]:>12.3f}"
              f"{mean_inner:>12.3f}{slope:>12.4f}   {verdict}")
    return results, diverging


def _draw_curves(ax, results, diverging, T, markers=True):
    """Plot every trajectory on ``ax`` with the shared per-method style.

    ``markers=False`` draws lines only, which is much cleaner when several
    bounded curves are packed into a narrow band (the magnified panel)."""
    for name, losses in results.items():
        color, marker, ls = STYLES[name]
        emph = name in diverging
        ax.plot(
            losses, label=DISPLAY_NAMES[name], color=color, linestyle=ls, alpha=0.95,
            linewidth=4.4 if emph else 2.6,
            marker=marker if markers else None, markersize=12 if emph else 9,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.3,
            markevery=max(1, T // 12), solid_capstyle="round",
            zorder=6 if emph else 3,
        )
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("Iteration", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=6, width=1.3)
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.25, zorder=0)


def _save(fig, outfiles):
    """Write each ``(stem, want_png)`` target: PDF always, PNG only where asked.
    The local ``figures/`` dir keeps PNG+PDF (handy for quick viewing); the
    paper's image dir gets the PDF only (LaTeX embeds the vector version)."""
    msgs = []
    for stem, want_png in outfiles:
        fig.savefig(stem + ".pdf", bbox_inches="tight")
        if want_png:
            fig.savefig(stem + ".png", bbox_inches="tight", dpi=150)
        msgs.append(stem + (".{png,pdf}" if want_png else ".pdf"))
    plt.close(fig)
    print(f"  saved -> {', '.join(msgs)}")


def plot_problem(results, diverging, outfiles, T,
                 ylabel=r"$f(W) = \mathrm{Tr}(G^\top W)$"):
    """Single-panel line plot of every trajectory, emphasising the diverging
    method(s).  No plot title is drawn (the paper adds LaTeX captions); legends
    use the paper's display names.  Each diverging method gets a colour-matched
    arrow.  ``outfiles`` is a list of ``(stem, want_png)`` targets."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _draw_curves(ax, results, diverging, T)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    for i, name in enumerate(diverging):
        color = STYLES[name][0]
        k = int(T * (0.42 + 0.14 * i))
        ax.annotate(
            DISPLAY_NAMES[name] + " diverges",
            xy=(k, results[name][k]),
            xytext=(0.30, 0.94 - 0.075 * i), textcoords="axes fraction",
            color=color, fontstyle="italic", ha="left", fontsize=14,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
        )
    ax.legend(loc="lower left", fontsize=LEG_FS, frameon=True, framealpha=0.95,
              edgecolor="0.75", ncol=2)
    fig.tight_layout()
    _save(fig, outfiles)


def plot_two_scales(results, diverging, outfiles, T, ylabel, mag_top=None):
    """Two side-by-side panels for a *bounded* objective where one method
    dominates the axis: (left) full scale, so the diverging method is visible;
    (right) magnified so the bounded methods' decrease is visible while the
    diverging curve climbs out of the frame.  ``mag_top`` sets the magnified
    panel's upper y-limit: raising it above the bounded band (e.g. 1.2) keeps a
    stretch of the diverging curve's sustained ascent in view; ``None`` fits the
    bounded methods only.  A single legend sits below both; no panel titles are
    drawn (the paper's caption names left and right)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))
    _draw_curves(axL, results, diverging, T, markers=True)
    win = min(MAGNIFIED_WINDOW, T)
    _draw_curves(axR, results, diverging, win, markers=True)  # short window
    axL.set_ylabel(ylabel, fontsize=LABEL_FS)

    # left: annotate the diverging method on the full scale
    for i, name in enumerate(diverging):
        axL.annotate(
            DISPLAY_NAMES[name] + " diverges",
            xy=(int(T * 0.42), results[name][int(T * 0.42)]),
            xytext=(0.06, 0.88), textcoords="axes fraction",
            color=STYLES[name][0], fontstyle="italic", ha="left",
            fontsize=TICK_FS + 1,
            arrowprops=dict(arrowstyle="->", color=STYLES[name][0], lw=1.8),
        )

    # right: magnify to the bounded methods' band over the first `win` steps
    # (past this they just repeat the period-2 cycle).  If mag_top is given, the
    # ceiling is raised above that band so the diverging curve stays visible
    # climbing until it exits the top -- the sustained ascent, not just an
    # off-scale arrow.
    bounded = [v for n, v in results.items() if n not in diverging]
    lo = min(float(v[:win].min()) for v in bounded)
    hi = max(float(v[:win].max()) for v in bounded)
    pad = 0.12 * (hi - lo)
    top = hi + pad if mag_top is None else mag_top
    axR.set_xlim(-0.5, win - 0.5)
    axR.set_ylim(lo - pad, top)
    # clean round ticks (…, -0.5, 0, 0.5, 1.0) instead of matplotlib's auto 0.25 grid
    axR.yaxis.set_major_locator(MultipleLocator(0.5))
    for name in diverging:
        curve = results[name]
        # anchor the arrow on the still-visible, climbing part of the curve
        below = np.where(curve[:win] <= top)[0]
        k = int(below[-1]) if below.size else 1
        axR.annotate(
            DISPLAY_NAMES[name] + r" diverges $\uparrow$",
            xy=(k, min(float(curve[k]), top)), xytext=(0.30, 0.80),
            textcoords="axes fraction", color=STYLES[name][0],
            fontstyle="italic", ha="left", fontsize=TICK_FS + 1,
            arrowprops=dict(arrowstyle="->", color=STYLES[name][0], lw=1.6),
        )

    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=LEG_FS,
               frameon=True, framealpha=0.95, edgecolor="0.75",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, outfiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=None,
                        help="momentum coefficient; overrides every per-problem "
                             "default (default: use each problem's own mu)")
    parser.add_argument("--nesterov", action="store_true",
                        help="use Nesterov momentum instead of standard")
    parser.add_argument("--eta", type=float, default=None,
                        help="learning rate (overrides the per-problem default)")
    parser.add_argument("--T", type=int, default=None,
                        help="number of iterations (overrides the per-problem default)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)
    # The paper picks up the same PDFs from aaai_article/images/counterexamples/.
    images_dir = os.path.abspath(
        os.path.join(here, "..", "..", "aaai_article", "images", "counterexamples"))
    have_images = os.path.isdir(os.path.dirname(images_dir))
    if have_images:
        os.makedirs(images_dir, exist_ok=True)

    def stems(basename):
        # (stem, want_png): the local figures/ dir keeps PNG+PDF; the paper's
        # image dir gets the PDF only.
        out = [(os.path.join(figdir, basename), True)]
        if have_images:
            out.append((os.path.join(images_dir, basename), False))
        return out

    # Per-problem step size and horizon.  The two gradients live on very
    # different scales (problem 1 has sigma_1 = 1001; problem 2 has M = 100), so
    # each gets an eta that makes the divergence clearly visible without the
    # steep descenders dominating the axis.  --eta / --T override both.
    #
    # Momentum: the two LINEAR counterexamples are run momentum-free (mu = 0).
    # This is WITHOUT LOSS OF GENERALITY -- the objective has a constant
    # gradient G, so the momentum buffer is always a positive scalar multiple of
    # G, and the Muon LMO is scale-invariant; hence sign(polar(.)) /
    # polar(sign(.)) and the EF21 targets are identical for every mu in [0, 1)
    # and both momentum variants (cf. Proposition "reduction").  The mu = 0
    # trajectory shown is thus the trajectory for all mu.
    G1, _ = signmuon_counterexample()
    G2, _ = muonsign_counterexample(eps=1.0, M=100.0)
    grad1, loss1 = make_linear_problem(G1)
    grad2, loss2 = make_linear_problem(G2)
    lin_ylabel = r"$f(W) = \mathrm{Tr}(G^\top W)$"
    quad_ylabel = (r"$f(\mathbf{W})=-\gamma W_{22}"
                   r"+A\sum_i\Phi_i+\sum_k b_k$")
    problems = [
        dict(
            title="Counterexample 1 -- SignMuon (Theorem 1)",
            grad_fn=grad1, loss_fn=loss1, shape=G1.shape, eta=1e-3, T=60,
            mu=0.0, verdict_mode="inner", ylabel=lin_ylabel,
            outfiles=stems("signmuon_counterexample"),
        ),
        dict(
            title="Counterexample 2 -- MuonSign / MuonUSign (Theorems 2--3)",
            grad_fn=grad2, loss_fn=loss2, shape=G2.shape, eta=5e-3, T=60,
            mu=0.0, verdict_mode="inner", ylabel=lin_ylabel,
            outfiles=stems("muonsign_counterexample"),
        ),
        dict(
            title="Counterexample 3 -- EF21-SignMuon (Appendix theorem)",
            # The universal construction depends on (mu, variant); it is rebuilt
            # per run below.  It diverges for EVERY mu in [0,1) and both variants
            # (the iterate trajectory is identical), so any mu witnesses it.
            # The objective is bounded below, so the bounded methods stay near
            # the floor while EF21-SignMuon dominates the axis: use a two-scale
            # (full + magnified) figure.
            builder=ef21_signmuon_counterexample, eta=1.0, T=600,
            mu=0.0, verdict_mode="slope", ylabel=quad_ylabel, twoscale=True,
            # magnified panel: raise the ceiling to 1.2 so EF21-SignMuon's
            # sustained climb is visible (not just the bounded band).
            mag_top=1.2,
            outfiles=stems("ef21_signmuon_counterexample"),
        ),
    ]

    for p in problems:
        eta = args.eta if args.eta is not None else p["eta"]
        T = args.T if args.T is not None else p["T"]
        mu = args.mu if args.mu is not None else p["mu"]
        if "builder" in p:
            grad_fn, loss_fn, shape, _ = p["builder"](mu=mu, nesterov=args.nesterov)
        else:
            grad_fn, loss_fn, shape = p["grad_fn"], p["loss_fn"], p["shape"]
        results, diverging = run_problem(
            p["title"], grad_fn, loss_fn, shape, T, eta,
            mu, args.nesterov, verdict_mode=p["verdict_mode"])
        if p.get("twoscale"):
            plot_two_scales(results, diverging, p["outfiles"], T, p["ylabel"],
                            mag_top=p.get("mag_top"))
        else:
            plot_problem(results, diverging, p["outfiles"], T, ylabel=p["ylabel"])


if __name__ == "__main__":
    main()
