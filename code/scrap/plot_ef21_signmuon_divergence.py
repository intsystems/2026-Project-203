"""
Paper figure for the theorem "Divergence of EF21-SignMuon"
(aaai_article/ef21_signmuon_divergence.tex).

Construction (2x2, mu=0, W0=d0=0):
    f(W) = c1*eta*L*W00 + c2*eta*L*W11 + (L/2)(W01-a*eta)^2 + (L/2)(W10-b*eta)^2
    c1=1/5, c2=1/20, a=b=-2/5.

Panel (a): the SAME quadratic instance (L=eta=1), six/relevant methods. Only
           EF21-SignMuon ascends (f->+inf); tracking the gradient (EF21-MuonUSign)
           and full Muon descend.
Panel (b): scale invariance. For the eta-scaled family, f(X_t)/(L*eta^2) is the
           SAME trajectory for every (L,eta): all curves collapse onto one line of
           slope r = 0.07160, so EF21-SignMuon diverges for EVERY L>0, eta>0.

float64 is used; for this instance it agrees with 90-digit mpmath (the divergence
is precision-stable, not a rounding artifact).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C1, C2, A, B = 1.0/5, 1.0/20, -2.0/5, -2.0/5
RATE = 0.0715989997  # exact per-step ascent rate on the period-2 orbit


def lmo(M, tol=1e-9):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(S > tol * (S[0] if S.size else 1.0)))
    return U[:, :r] @ Vt[:r, :]


def ssign(Y):
    return np.mean(np.abs(Y)) * np.sign(Y)


def run(kind, L, eta, T):
    """Actual (un-normalized) EF21/Muon dynamics on the eta-scaled instance."""
    ae, be = A * eta, B * eta
    W = np.zeros((2, 2)); d = np.zeros((2, 2)); g = np.zeros((2, 2))
    def grad(W): return np.array([[C1*eta*L, L*(W[0,1]-ae)], [L*(W[1,0]-be), C2*eta*L]])
    def loss(W): return C1*eta*L*W[0,0]+C2*eta*L*W[1,1]+L/2*(W[0,1]-ae)**2+L/2*(W[1,0]-be)**2
    vals = np.empty(T)
    for t in range(T):
        vals[t] = loss(W)
        G = grad(W)
        if kind == "EF21-SignMuon":
            D = lmo(G); d = d + ssign(D - d); step = d
        elif kind == "EF21-MuonUSign":
            g = g + ssign(G - g); step = lmo(g)
        elif kind == "Muon":
            step = lmo(G)
        elif kind == "SignMuon":
            step = np.sign(lmo(G))
        W = W - eta * step
    return vals


# ---------------------------------------------------------------- figure
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0))

# Panel (a): one instance (L=eta=1), method comparison
T = 600
methods = [
    ("EF21-SignMuon",  "#d62728", "-",         4.2, "X", 6),
    ("EF21-MuonUSign", "#1f77b4", "-",         3.0, "o", 4),
    ("Muon",           "#7f7f7f", (0,(1,2)),   2.2, None, 2),
    ("SignMuon",       "#2ca02c", (0,(5,3)),   2.4, "^", 3),
]
for name, col, ls, lw, mk, z in methods:
    v = run(name, 1.0, 1.0, T)
    axA.plot(v, label=name, color=col, linestyle=ls, linewidth=lw, marker=mk,
             markersize=9, markevery=max(1, T//10),
             markerfacecolor=col, markeredgecolor="white", markeredgewidth=1.2, zorder=z)
axA.axhline(0, color="black", lw=1.0, alpha=0.5)
axA.set_xlabel("Iteration", fontsize=16)
axA.set_ylabel(r"$f(\mathbf{W}_t)$", fontsize=16)
axA.set_title(r"(a) one $L$-smooth instance ($L=\eta=1$)", fontsize=14)
axA.tick_params(labelsize=12)
axA.grid(True, ls="--", lw=0.8, alpha=0.25)
axA.annotate("EF21-SignMuon\nascends: " + r"$f\to+\infty$",
             xy=(int(T*0.62), run("EF21-SignMuon",1.0,1.0,T)[int(T*0.62)]),
             xytext=(0.30, 0.86), textcoords="axes fraction", color="#d62728",
             fontstyle="italic", fontsize=12, ha="left",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
axA.legend(loc="lower left", fontsize=11, frameon=True, framealpha=0.95, edgecolor="0.75")

# Panel (b): scale invariance -- f/(L eta^2) collapses for every (L,eta)
Tb = 800
pairs = [(1.0, 1.0), (1.0, 0.3), (3.0, 0.5), (10.0, 0.1)]
cols = ["#d62728", "#ff7f0e", "#9467bd", "#17becf"]
for (L, eta), col in zip(pairs, cols):
    v = run("EF21-SignMuon", L, eta, Tb) / (L * eta**2)
    axB.plot(v, color=col, lw=2.2, alpha=0.9,
             label=r"$L=%g,\ \eta=%g$" % (L, eta))
axB.plot(np.arange(Tb), RATE*np.arange(Tb), color="black", ls=":", lw=1.6,
         label=r"slope $r=0.0716$")
axB.set_xlabel("Iteration", fontsize=16)
axB.set_ylabel(r"$f(\mathbf{X}_t)\,/\,(L\eta^2)$", fontsize=16)
axB.set_title(r"(b) divergence for every $(L,\eta)$: curves collapse", fontsize=14)
axB.tick_params(labelsize=12)
axB.grid(True, ls="--", lw=0.8, alpha=0.25)
axB.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95, edgecolor="0.75")

fig.tight_layout()
outdir_paper = os.path.join("..", "aaai_article", "images", "counterexamples")
os.makedirs("im_dif", exist_ok=True)
fig.savefig("im_dif/ef21_signmuon_divergence.png", bbox_inches="tight", dpi=150)
fig.savefig("im_dif/ef21_signmuon_divergence.pdf", bbox_inches="tight")
if os.path.isdir(os.path.dirname(outdir_paper)):
    os.makedirs(outdir_paper, exist_ok=True)
    fig.savefig(os.path.join(outdir_paper, "ef21_signmuon_divergence.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir_paper, "ef21_signmuon_divergence.png"), bbox_inches="tight", dpi=150)
print("saved ef21_signmuon_divergence.{pdf,png} to im_dif/ and aaai_article/images/counterexamples/")
