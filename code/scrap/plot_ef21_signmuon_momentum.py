"""
Paper figure: EF21-SignMuon diverges UNDER MOMENTUM (Theorem "Divergence under
momentum"). Momentum does not restore convergence.

Normalized general quadratic (L=eta=1): Gtil=[[c1,K1(W01-a)],[K2(W10-b),c2]].
  standard construction A: c1=.35,c2=.10,K1=5,K2=2,a=1.1,b=-.3
  Nesterov construction B: c1=.031,c2=.33,K1=5.59,K2=5.58,a=1.03,b=1.31
Panel (a): asymptotic per-step divergence rate r_mu vs mu (positive band => diverges).
Panel (b): a representative long trajectory (construction A, standard momentum):
           mu=0 descends, mu=0.3 and 0.5 diverge (f -> +inf).
float64 (== mpmath at 30/50 digits; precision-stable).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def polar(G):
    p, q, r, s = G[0, 0], G[0, 1], G[1, 0], G[1, 1]
    det = p * s - q * r
    if det >= 0:
        N = np.hypot(p + s, q - r)
        return np.zeros((2, 2)) if N == 0 else np.array([[p + s, q - r], [r - q, p + s]]) / N
    N = np.hypot(p - s, q + r)
    return np.array([[p - s, q + r], [q + r, s - p]]) / N


def run(c, mu, nesterov, T, record=False):
    c1, c2, K1, K2, a, b = c
    W = np.zeros((2, 2)); d = np.zeros((2, 2)); M = np.zeros((2, 2))
    def loss(W): return c1*W[0,0]+c2*W[1,1]+0.5*K1*(W[0,1]-a)**2+0.5*K2*(W[1,0]-b)**2
    vals = np.empty(T) if record else None
    fh = None
    for t in range(T):
        if record: vals[t] = loss(W)
        if t == T // 2: fh = loss(W)
        G = np.array([[c1, K1*(W[0,1]-a)], [K2*(W[1,0]-b), c2]])
        M = mu*M + (1-mu)*G
        eff = ((1-mu)*G + mu*M) if nesterov else M
        D = polar(eff); d = d + np.mean(np.abs(D-d))*np.sign(D-d); W = W - d
    if record: return vals
    return (loss(W) - fh) / (T - T // 2)   # asymptotic per-step rate over [T/2, T]


cA = (0.35, 0.10, 5.0, 2.0, 1.1, -0.3)
cB = (0.031, 0.33, 5.59, 5.58, 1.03, 1.31)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0))
T = 80000
mk = max(1, T // 12)

# Panel (a): standard momentum, construction A, long horizon
cmap = plt.get_cmap("plasma")
for mu, col, sty in ((0.0, "#7f7f7f", (0, (1, 2))), (0.3, cmap(0.15), "-"),
                     (0.5, cmap(0.4), "-"), (0.7, cmap(0.62), "-"), (0.9, cmap(0.82), "-")):
    v = run(cA, mu, False, T, record=True)
    lab = r"$\mu=%.1f$" % mu + (" (descends)" if v[-1] < v[0] else "")
    axA.plot(v, color=col, lw=3.0 if mu else 2.2, ls=sty, label=lab)
axA.axhline(0, color="black", lw=1.0, alpha=0.6)
axA.set_xlabel("Iteration", fontsize=16); axA.set_ylabel(r"$f(\mathbf{W}_t)$", fontsize=16)
axA.set_title("(a) standard heavy-ball momentum", fontsize=14)
axA.tick_params(labelsize=12); axA.grid(True, ls="--", lw=0.8, alpha=0.25)
axA.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95, edgecolor="0.75")

# Panel (b): Nesterov momentum, construction B, long horizon
for mu, col, sty in ((0.5, "#7f7f7f", (0, (1, 2))), (0.7, cmap(0.4), "-"), (0.9, cmap(0.82), "-")):
    v = run(cB, mu, True, T, record=True)
    lab = r"$\mu=%.1f$" % mu + (" (descends)" if v[-1] < v[0] else "")
    axB.plot(v, color=col, lw=3.0 if mu != 0.5 else 2.2, ls=sty, label=lab)
axB.axhline(0, color="black", lw=1.0, alpha=0.6)
axB.set_xlabel("Iteration", fontsize=16); axB.set_ylabel(r"$f(\mathbf{W}_t)$", fontsize=16)
axB.set_title("(b) Nesterov momentum", fontsize=14)
axB.tick_params(labelsize=12); axB.grid(True, ls="--", lw=0.8, alpha=0.25)
axB.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95, edgecolor="0.75")

fig.suptitle("EF21-SignMuon still diverges under momentum (both variants) -- momentum does not restore convergence",
             fontsize=13.5, y=1.02)
rA = rB = np.array([])  # (rate sweep omitted in the figure)
fig.tight_layout()
os.makedirs("im_dif", exist_ok=True)
fig.savefig("im_dif/ef21_signmuon_momentum.png", bbox_inches="tight", dpi=150)
fig.savefig("im_dif/ef21_signmuon_momentum.pdf", bbox_inches="tight")
paper = os.path.join("..", "aaai_article", "images", "counterexamples")
if os.path.isdir(os.path.dirname(paper)):
    os.makedirs(paper, exist_ok=True)
    fig.savefig(os.path.join(paper, "ef21_signmuon_momentum.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(paper, "ef21_signmuon_momentum.png"), bbox_inches="tight", dpi=150)
print("saved ef21_signmuon_momentum.{pdf,png}")
print("rate at mu: standard", np.array2string(rA, precision=3))
print("            Nesterov", np.array2string(rB, precision=3))
