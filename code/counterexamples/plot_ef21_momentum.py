"""Paper figure: momentum validation for the EF21-SignMuon counterexample.

For each momentum coefficient mu and each momentum variant, the appendix builds
a (mu, variant)-specific L-smooth function on which EF21-SignMuon diverges
(Theorem~\\ref{th:ef_div}).  This figure runs the actual algorithm on each of
those functions and shows the objective diverging for every setting.

Two panels: (left) standard momentum, (right) Nesterov.  In each we plot
``f(X_t) - f(X_0)`` for ``mu in {0, 0.5, 0.9, 0.95, 0.99}``.  Subtracting f(X_0)
removes the only genuinely mu-dependent offset (the field constant A(mu)
multiplies a bounded periodic term), after which all curves share the common
line of slope 49/480: the divergence rate is the same for every mu and both
variants, as the reduction lemma predicts.  Output:
``figures/ef21_signmuon_momentum.{pdf,png}``.

Shares the visual style (fonts, grid, bottom legend) of the optimizer-comparison
figure in ``run_counterexamples.py``.
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from counterexamples.optimizers import EF21SignMuon                       # noqa: E402
from counterexamples.problems import ef21_signmuon_counterexample         # noqa: E402

# --- shared style (kept in sync with run_counterexamples.py) --------------
LABEL_FS, TICK_FS, LEG_FS = 18, 13, 12.5

T = 1000
MUS = [0.0, 0.5, 0.9, 0.95, 0.99]
RATE = 49 / 480
# mu is an ordered magnitude -> a single-hue blue ORDINAL ramp (light->dark as mu
# grows), not a rainbow.  Kept no lighter than the "step 250" blue so the pale end
# still reads on white; the curves bunch, so each also carries a distinct MARKER
# (secondary encoding) to stay separable where the lines overlap.
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
MARKERS = ["o", "s", "^", "D", "v"]


def trajectory(mu, nesterov):
    grad_fn, loss_fn, shape, _ = ef21_signmuon_counterexample(
        mu=mu, nesterov=nesterov)
    opt = EF21SignMuon(shape, eta=1.0, mu=mu, nesterov=nesterov)
    f = np.empty(T)
    for t in range(T):
        f[t] = loss_fn(opt.track_point())
        opt.step(grad_fn(opt.grad_point()))
    return f - f[0]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    tt = np.arange(T)
    for ax, nesterov in zip(axes, (False, True)):
        ax.plot(tt, RATE * tt, color="0.2", ls=(0, (1, 1.6)), lw=2.0, zorder=10,
                label=r"slope $\frac{49}{480}$")
        for i, (mu, c, mk) in enumerate(zip(MUS, COLORS, MARKERS)):
            ax.plot(tt, trajectory(mu, nesterov), color=c, lw=2.2,
                    marker=mk, markersize=6.5,
                    markevery=(int(i * 170 / len(MUS)), 170),
                    markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.9,
                    label=rf"$\mu={mu:g}$", zorder=5 + i, solid_capstyle="round")
        ax.axhline(0, color="black", lw=0.8, alpha=0.4, zorder=1)
        ax.set_xlabel("Iteration", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS, length=6, width=1.2)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.25, zorder=0)
        ax.margins(x=0.01)
    axes[0].set_ylabel(r"$f(\mathbf{X}_t)-f(\mathbf{X}_0)$", fontsize=LABEL_FS)

    handles, labels = axes[0].get_legend_handles_labels()
    # put the mu curves first, the reference line last
    order = list(range(1, len(labels))) + [0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=6, fontsize=LEG_FS,
               frameon=True, framealpha=0.95, edgecolor="0.75",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)
    # (stem, want_png): local figures/ keeps PNG+PDF; the paper image dir (if the
    # tree is present alongside the code) gets the PDF only.  For the standalone
    # supplement the paper branch simply does not fire.
    targets = [(os.path.join(figdir, "ef21_signmuon_momentum"), True)]
    images_dir = os.path.join(here, "..", "..", "aaai_article",
                              "images", "counterexamples")
    if os.path.isdir(os.path.dirname(images_dir)):
        os.makedirs(images_dir, exist_ok=True)
        targets.append((os.path.join(images_dir, "ef21_signmuon_momentum"), False))
    msgs = []
    for stem, want_png in targets:
        fig.savefig(stem + ".pdf", bbox_inches="tight")
        if want_png:
            fig.savefig(stem + ".png", dpi=160, bbox_inches="tight")
        msgs.append(os.path.abspath(stem) + (".{pdf,png}" if want_png else ".pdf"))
    plt.close(fig)
    print("saved", ", ".join(msgs))

    # numeric check printed alongside
    print(f"{'variant':<9}{'mu':>6}{'tail slope':>12}   (exact 49/480 = "
          f"{RATE:.6f})")
    for nesterov in (False, True):
        for mu in MUS:
            f = trajectory(mu, nesterov)
            s = (f[-1] - f[T // 2]) / (T - T // 2)
            print(f"{'Nesterov' if nesterov else 'standard':<9}{mu:>6}"
                  f"{s:>12.6f}")


if __name__ == "__main__":
    main()
