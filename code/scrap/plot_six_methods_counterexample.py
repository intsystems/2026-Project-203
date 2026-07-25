"""
Six-method comparison on the QUADRATIC-VALLEY counterexample on which
EF21-SignMuon provably ascends (Theorem "Divergence of EF21-SignMuon",
appendix file ef21_signmuon_divergence.tex), NOT the linear one.

    f(W) = c1*W_00 + c2*W_11 + (K/2)(W_01 - a)^2 + (K/2)(W_10 - b)^2 ,  W in R^{2x2}
    grad f(W) = [[ c1,          K(W_01 - a) ],
                 [ K(W_10 - b), c2          ]]

Verified divergent constants (precision-stable to >=90 digits, float64 == mpmath):

    c1 = 1/5,  c2 = 1/20,  a = b = -2/5,  K = 1,  eta = 1,  mu = 0,  W0 = 0.

Mechanism (see the theorem): the LMO discards gradient magnitude, so the
off-diagonal step stays O(1) and the exact polar factor sign-flips its
off-diagonal every step (a period-2 chatter, both states in the reflection
regime det<0). EF21-SignMuon tracks this discontinuous target; the shared
scaled-sign magnitude alpha_t = mean|Delta| = O(1) overshoots the small diagonal
targets D_00,D_11 = +/-(c1-c2)/N, so the diagonal estimator averages to the
WRONG sign and W_00,W_11 -> +inf. The ascent is second-order (curvature): the
first-order work <G,d_est> is positive, but 1/2||d_off||^2 exceeds it. Exact
ascent rate 0.07160 per step. The matrices are WELL-conditioned on the orbit
(cond<3.5) -- this is NOT rank-deficiency. The gradient-tracking variants
(EF21-MuonUSign) track the smooth Lipschitz gradient instead and descend.
f is L-smooth with L = K; it is unbounded below only in the W_00,W_11 -> -inf
direction, which the ascending trajectory never visits (W_00>=-7.07,W_11>=-0.95),
so a smooth floor there restores lower boundedness without touching the dynamics.

NOTE: for a FIXED f, divergence needs eta above a threshold (eta=1 here works,
eta=0.5 descends). The theorem's "diverges for ANY eta" uses the scale-invariant
family f_eta with offsets a*eta -- there the normalized trajectory, hence the
divergence, is independent of eta.

Methods (all share one heavy-ball momentum MU; state reset per run):
  SignMuon        sign(LMO(M))                      -> step -eta*sign(D)
  EF21-SignMuon   EF21 tracks D=LMO(M) (scaled-sign), step -eta*d_est   <-- diverges
  MuonSign        LMO(  sign(M) )                   -> step -eta*D
  MuonUSign       LMO( scaled-sign(M) )  == MuonSign (LMO scale-invariance)
  EF21-MuonSign   EF21 tracks M with PLAIN sign  (g += sign(Delta)), LMO after
  EF21-MuonUSign  EF21 tracks M with SCALED sign (g += mean|Delta|*sign(Delta)), LMO after
Muon (full LMO, no compression) is a faint gray reference.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless; comment out to use plt.show()
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Quadratic-valley counterexample (verified constants that make EF21-SignMuon
# ascend; see ef21_signmuon_divergence.tex). ASYMMETRIC diagonal c1 != c2 and
# EQUAL SAME-SIGN offsets a = b are both essential.
# ----------------------------------------------------------------------
C1_LIN = 1.0 / 5.0        # linear coeff on W_00
C2_LIN = 1.0 / 20.0       # linear coeff on W_11  (asymmetric: c1 != c2)
K_VAL  = 1.0              # curvature of the two quadratic valleys ( = smoothness L )
A_OFF  = -2.0 / 5.0       # valley center for W_01
B_OFF  = -2.0 / 5.0       # valley center for W_10  (same sign, equal to a)
DIM    = (2, 2)

def calc_grad(W):
    return np.array([[C1_LIN,                    K_VAL * (W[0, 1] - A_OFF)],
                     [K_VAL * (W[1, 0] - B_OFF), C2_LIN]])

def calc_loss(W):
    return (C1_LIN * W[0, 0] + C2_LIN * W[1, 1]
            + 0.5 * K_VAL * (W[0, 1] - A_OFF) ** 2
            + 0.5 * K_VAL * (W[1, 0] - B_OFF) ** 2)

# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def lmo(M, tol=1e-9):
    """Muon LMO direction U_r V_r^T using only rank-r NONZERO singular directions
    (paper's spectral/nuclear LMO). Full U@Vt would add arbitrary null-space
    directions on rank-deficient inputs. Scale-invariant: lmo(c*M)=lmo(M), c>0."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(S > tol * (S[0] if S.size else 1.0)))
    return U[:, :r] @ Vt[:r, :]

def scaled_sign(Y):
    """Contractive 1-bit compressor mean|Y| * sign(Y)  (the 'U'/USign compressor)."""
    return np.mean(np.abs(Y)) * np.sign(Y)

# ----------------------------------------------------------------------
# The six optimizers (closures with internal momentum + EF state).
# Each takes the raw gradient g and returns direction d; outer loop: W <- W - eta*d.
# ----------------------------------------------------------------------
def make_signmuon(mu):
    s = {"M": np.zeros(DIM)}
    def update(g):
        s["M"] = mu * s["M"] + g
        return np.sign(lmo(s["M"]))                       # sign AFTER lmo
    return update

def make_ef21_signmuon(mu):
    s = {"M": np.zeros(DIM), "d": np.zeros(DIM)}
    def update(g):
        s["M"] = mu * s["M"] + g
        D = lmo(s["M"])                                   # exact LMO first
        delta = D - s["d"]
        s["d"] = s["d"] + np.mean(np.abs(delta)) * np.sign(delta)   # scaled-sign EF21 on LMO dir
        return s["d"]
    return update

def make_muonsign(mu):
    s = {"M": np.zeros(DIM)}
    def update(g):
        s["M"] = mu * s["M"] + g
        return lmo(np.sign(s["M"]))                       # plain sign BEFORE lmo
    return update

def make_muon_usign(mu):
    s = {"M": np.zeros(DIM)}
    def update(g):
        s["M"] = mu * s["M"] + g
        return lmo(scaled_sign(s["M"]))                   # scaled sign BEFORE lmo (== plain, scale-inv)
    return update

def make_ef21_muonsign(mu):
    """EF21 tracking the gradient with the PLAIN sign compressor (no magnitude)."""
    s = {"M": np.zeros(DIM), "g": np.zeros(DIM)}
    def update(g):
        s["M"] = mu * s["M"] + g
        delta = s["M"] - s["g"]
        s["g"] = s["g"] + np.sign(delta)                  # plain sign, unit steps
        return lmo(s["g"])
    return update

def make_ef21_muon_usign(mu):
    """EF21 tracking the gradient with the SCALED sign compressor (paper's box 1237)."""
    s = {"M": np.zeros(DIM), "g": np.zeros(DIM)}
    def update(g):
        s["M"] = mu * s["M"] + g
        delta = s["M"] - s["g"]
        s["g"] = s["g"] + scaled_sign(delta)              # scaled sign (contractive)
        return lmo(s["g"])
    return update

def make_muon(mu):                                        # faint reference, no compression
    s = {"M": np.zeros(DIM)}
    def update(g):
        s["M"] = mu * s["M"] + g
        return lmo(s["M"])
    return update

# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
T   = 600
ETA = 1.0         # for the FIXED f, EF21-SignMuon ascends once eta exceeds ~0.8; eta=1 is clean.
MU  = 0.0         # momentum-free case of the theorem (mu>0 leaves the LMO direction unchanged)

factories = {
    "SignMuon":       make_signmuon,
    "MuonSign":       make_muonsign,
    "MuonUSign":      make_muon_usign,
    "EF21-SignMuon":  make_ef21_signmuon,
    "EF21-MuonSign":  make_ef21_muonsign,
    "EF21-MuonUSign": make_ef21_muon_usign,
    "Muon":           make_muon,
}

func_vals = {}
for name, factory in factories.items():
    fn = factory(MU)                     # fresh state per run
    W = np.zeros(DIM)
    vals = []
    for t in range(T):
        vals.append(float(calc_loss(W)))
        W = W - ETA * fn(calc_grad(W))
    func_vals[name] = vals

print(f"MU={MU}, ETA={ETA}, K={K_VAL}, a={A_OFF}, b={B_OFF}, T={T}")
print(f"{'method':<16}{'f[0]':>8}{'f[T-1]':>12}   trend")
for name, v in func_vals.items():
    print(f"{name:<16}{v[0]:>8.3f}{v[-1]:>12.3f}   {'DIVERGES (up)' if v[-1] > v[0] + 1.0 else 'descends'}")

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
# (color, marker, linestyle, linewidth, markersize, zorder)
styles = {
    "EF21-SignMuon":  ("#d62728", "X", "-",            4.4, 13, 6),  # the diverging one, emphasized
    "SignMuon":       ("#1f77b4", "^", "-",            3.0, 10, 4),
    "MuonSign":       ("#2ca02c", "s", "-",            3.0, 10, 4),
    "MuonUSign":      ("#10C5D5", "v", (0, (3, 3)),    2.4, 11, 5),
    "EF21-MuonSign":  ("#9467bd", "D", (0, (5, 3)),    2.8, 10, 4),
    "EF21-MuonUSign": ("#4A3322", "o", "-",            3.6, 10, 5),
    "Muon":           ("#999999", None, (0, (1, 2)),   2.0,  0, 2),
}

fig, ax = plt.subplots(figsize=(10, 6.5))
for name, (color, marker, ls, lw, ms, z) in styles.items():
    ax.plot(func_vals[name], label=name, color=color, linestyle=ls, alpha=0.9,
            linewidth=lw, marker=marker, markersize=ms, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=1.4,
            markevery=max(1, T // 10), solid_capstyle="round", zorder=z)

ax.set_xlabel("Iteration", fontsize=20)
ax.set_ylabel(r"$f(W) = c_1 W_{00} + c_2 W_{11} + \frac{K}{2}\sum(\cdot)^2$", fontsize=18)
ax.axhline(0, color="black", linewidth=1.0, alpha=0.5, zorder=1)
ax.tick_params(axis="both", labelsize=16, length=6, width=1.3)
ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.25, zorder=0)
ax.annotate("EF21-SignMuon diverges (only method that ascends)",
            xy=(int(T * 0.5), func_vals["EF21-SignMuon"][int(T * 0.5)]),
            xytext=(0.05, 0.965), textcoords="axes fraction", color="#d62728",
            fontstyle="italic", ha="left", fontsize=13,
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8))
ax.legend(loc="lower left", fontsize=13, frameon=True, framealpha=0.95,
          edgecolor="0.75", ncol=2)
fig.tight_layout()

os.makedirs("im_dif", exist_ok=True)
fig.savefig("im_dif/six_methods_counterexample.pdf", bbox_inches="tight")
fig.savefig("im_dif/six_methods_counterexample.png", bbox_inches="tight", dpi=150)
print("\nsaved -> im_dif/six_methods_counterexample.{pdf,png}")
