"""Synthetic smooth convex benchmark: F(X) = 1/2 <X-C, A(X-C)B> (Equation 9).

    python3 -m synthetic.benchmark --mode grid       --device cuda:0
    python3 -m synthetic.benchmark --mode final      --device cuda:0
    python3 -m synthetic.benchmark --mode alignment  --device cuda:0
    python3 -m synthetic.benchmark --mode horizon    --device cuda:0
    python3 -m synthetic.benchmark --mode floor      --device cuda:0
    python3 -m synthetic.benchmark --mode stability  --device cuda:0
    python3 -m synthetic.benchmark --mode kappa      --device cuda:0

Problem
-------
``A in S_++^m``, ``B in S_++^n`` are symmetric with a prescribed spectrum in a
prescribed eigenbasis, so the Hessian of ``F`` is the Kronecker product
``B (x) A`` with eigenvalues ``lambda_i(A) * lambda_j(B)``. Two consequences are
used throughout:

* the Frobenius smoothness constant is *exactly* ``L = max_ij lambda_i mu_j``
  and the strong-convexity constant is *exactly* ``sigma = min_ij lambda_i mu_j``,
  both known in closed form -- no estimation anywhere below;
* ``grad F(X) = A (X - C) B`` and ``F(X) = 1/2 <X - C, grad F(X)>``, so both are
  computed in closed form. There is no autograd graph. This matters for the
  bidirectional method, whose gradient must be taken at the broadcast model
  ``W`` while the metrics are scored at the exact model ``X``; with a closed-form
  gradient either point is one matmul away.

``--spectrum uniform`` (the default, and what the paper's table used) draws the
eigenvalues from ``U(0,1)``, which bounds ``L <= 1`` but leaves the condition
number ``L/sigma`` uncontrolled -- it comes out near ``1e4`` at ``m=n=100`` and
near ``2.5e5`` at ``m=n=500``, and the paper never reports it. ``--spectrum
logspace --kappa K`` fixes ``L = 1`` and ``L/sigma = K`` exactly instead, which
is what the ``kappa`` mode sweeps.

What each mode measures
-----------------------
``grid`` / ``final``
    The paper's protocol: fewest iterations to ``F(X) <= 1e-3``, learning rate
    and momentum tuned per method. Reproduces Tables 1 and 3.

    Read the resulting number for what it is. For every sign-family method the
    iteration count on this problem is *exactly* inversely proportional to eta
    (measured at ``m=n=100``: SignMuon needs 385 / 729 / 1827 / 3657 iterations
    at eta = 1e-3 / 5e-4 / 2e-4 / 1e-4) and there is a hard eta above which the
    method never reaches the target at all. That is the signature of a constant
    step size on a problem whose minimizer it cannot reach: a ``+-1`` step has
    fixed length ``eta*sqrt(mn)``, so the iterate settles into a ball of that
    radius and ``F`` plateaus. The tuner therefore returns the largest eta whose
    plateau still fits under the target, and "iterations to target" is
    ``const / eta_max`` -- a measurement of the *accuracy floor*, not of the
    descent rate. At matched eta the ranking can invert (SignSGD reaches 1e-3 in
    1553 iterations against SignMuon's 1827 at eta = 2e-4). The ``floor`` and
    ``horizon`` modes below separate the two effects.

``alignment``
    The descent lemma behind every guarantee in the paper is

        F(X_{t+1}) <= F(X_t) - eta <grad F(X_t), d_t> + (eta^2 L / 2) ||d_t||_F^2,

    so a method makes progress exactly insofar as its normalized alignment

        rho_t := <grad F(X_t), d_t> / (||grad F(X_t)||_F ||d_t||_F)

    stays positive. The divergence theorems construct instances where it does
    not. This mode measures the distribution of ``rho_t`` along the tuned
    trajectory on *random* instances -- the empirical counterpart of those
    theorems, and the one number here that is about the methods rather than
    about the tuning protocol. Closed-form references it is checked against:
    ``rho = 1`` for SGD; ``rho = ||G||_1 / (||G||_F sqrt(mn)) -> sqrt(2/pi)``
    for SignSGD on an incoherent gradient; ``rho = ||G||_nuc / (||G||_F sqrt(r))``
    for Muon. SignMuon, MuonUSign and MuonSign have no closed form -- that gap is
    the paper's subject.

``horizon``
    The theory's actual claim is not "reaches 1e-3 in N steps" but a rate: for a
    budget ``T``, tuning the step size gives
    ``eta*(T) = sqrt(2(F_0-F_*) / (L ||s||_F^2 T))`` and error ``O(T^{-1/2})``.
    Both exponents are predictions. This mode tunes ``(eta, momentum, schedule)``
    separately at each ``T`` and fits ``err ~ T^-p`` and ``eta* ~ T^-q``,
    reporting ``p`` and ``q`` against the predicted ``1/2``. The schedule is
    tuned per method rather than imposed, because nothing says the six methods
    want the same one.

``floor``
    The plateau ``F_inf(eta)`` and ``||grad F||_inf(eta)`` of a constant step,
    and the fitted exponent in ``eta``. The descent lemma predicts the gradient
    floor is linear in ``eta`` with slope ``L ||s||_F^2 / 2 rho``; since
    ``||s||_F^2 = mn`` for *both* SignMuon and SignSGD, any difference between
    them is attributable to ``rho`` alone, which is the cleanest available
    version of the paper's claim.

``stability``
    Largest stable ``eta`` per method, by bisection. Reported both raw and as a
    step length ``eta_max * ||s||_F``: if the operative trust region were the
    Frobenius ball the latter would be family-independent, so the spread across
    families measures how wrong the Frobenius bound is.

``kappa``
    The above at a controlled condition number, swept over decades. Conditioning
    is the only knob that governs quadratic dynamics, and the paper currently
    reports one uncontrolled draw.

Two details that carry over from the previous version
-----------------------------------------------------
* ``--lmo-dtype float32`` runs the Newton-Schulz iteration in single precision.
  The default ``bfloat16`` matches the reference Muon implementation and is what
  the paper's numbers used, but bfloat16 carries only ~3 decimal digits, so for
  the methods that *sign the LMO output* (SignMuon, MuonSign) entries of
  ``polar(M)`` near zero can flip -- a real source of run-to-run variation in the
  iteration count on a deterministic problem.
* ``EF21MuonSign`` is scored on its **exact** model ``X``, not on the compressed
  broadcast model ``W`` held in ``p.data``. ``X`` is the iterate the convergence
  theory bounds. (Scripts before the 2026-07 refactor scored ``W``.)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from common.utils import results_root
from common.optimizers import (
    EF21MuonSign,
    EF21MuonUSign,
    EF21SignMuon,
    Muon,
    MuonSign,
    MuonUSign,
    SignMuon,
    SignSGD,
)

METHOD_CLASSES = {
    "signmuon": SignMuon,
    "ef21signmuon": EF21SignMuon,
    "muonusign": MuonUSign,
    "muonsign": MuonSign,
    "ef21muonusign": EF21MuonUSign,
    "ef21muonsign": EF21MuonSign,
    "muon": Muon,
    "signsgd": SignSGD,
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

DEFAULT_METHODS = list(METHOD_CLASSES)

#: Methods whose step is a ``+-1`` matrix (``||s||_F = sqrt(mn)``). The rest take
#: a step of Frobenius norm ``sqrt(min(m,n))`` -- except ``sgd``/``adam``, whose
#: step length is not normalized at all.
SIGN_FAMILY = ("signmuon", "muonsign", "signsgd")
LMO_FAMILY = ("muon", "muonusign", "ef21muonusign", "ef21muonsign", "ef21signmuon")


# --------------------------------------------------------------------------
# Learning-rate / momentum grids
# --------------------------------------------------------------------------

# Grid syntax:
#   "lo:hi:step"  linear, inclusive of both endpoints (the paper's Table 3 form)
#   "lo:hi:xN"    logarithmic, N points per decade, inclusive of both endpoints
#
# The paper's Table 3 grids are linear and one decade wide, which was enough to
# leave two rows censored at a boundary and so not optima at all:
#
#   * SGD's reported (eta, mu) = (0.1, 0.95) is the *top* of both of its grids
#     ([1e-2, 1e-1] and {0.1 ... 0.95}), so "SGD: 972 iterations" is an upper
#     bound on SGD's iteration count, not its tuned value.
#   * The two EF21 rows report optima (3.3e-3, 2.8e-3) that lie *below* their
#     stated grid [5e-3, 2e-2] -- they came from a run whose grid is not in the
#     repository.
#
# Widening a one-decade linear grid to the four decades needed here would cost
# hundreds of points per method, so the defaults below are logarithmic. This is
# a deliberate departure from Table 3, which must be updated to match; the
# paper's exact linear grids are kept in ``PAPER_LR_GRIDS`` and selectable with
# ``--grid-preset paper``.
DEFAULT_LR_GRIDS: Dict[str, str] = {
    "signmuon":      "1e-5:1e-2:x6",
    "muonusign":     "1e-5:1e-2:x6",
    "muonsign":      "1e-5:1e-2:x6",
    "ef21signmuon":  "1e-5:1e-2:x6",
    "ef21muonusign": "1e-4:1e-1:x6",
    "ef21muonsign":  "1e-4:1e-1:x6",
    "muon":          "1e-4:1e-1:x6",
    "signsgd":       "1e-5:1e-2:x6",
    "sgd":           "1e-3:1e+1:x6",
    "adam":          "1e-3:1e+1:x6",
}

#: Table 3 exactly as printed, for ``--grid-preset paper``. Two rows cannot
#: reproduce their own reported optimum; see the note above.
PAPER_LR_GRIDS: Dict[str, str] = {
    "signmuon":      "1e-4:1e-3:1e-4",
    "muonusign":     "1e-4:1e-3:1e-4",
    "muonsign":      "1e-4:1e-3:1e-4",
    "ef21signmuon":  "1e-4:1e-3:1e-4",
    "ef21muonusign": "5e-3:2e-2:1e-4",
    "ef21muonsign":  "5e-3:2e-2:1e-4",
    "muon":          "5e-3:2e-2:1e-3",
    "signsgd":       "5e-5:2e-4:1e-5",
    "sgd":           "1e-2:1e-1:1e-2",
    "adam":          "1e-2:1e-1:1e-2",
}

DEFAULT_MOMENTUM_GRID = "0.0,0.5,0.9,0.95"
PAPER_MOMENTUM_GRID = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95"

# Tuned hyperparameters currently reported in the paper (Table 1). Used by
# ``--mode final``. ``None`` means "not yet tuned".
REPORTED_BEST: Dict[str, Optional[Tuple[float, float]]] = {
    "signmuon":      (2e-4, 0.2),
    "muonusign":     None,
    "muonsign":      None,
    "ef21signmuon":  None,
    "ef21muonusign": (3.3e-3, 0.1),
    "ef21muonsign":  (2.8e-3, 0.1),
    "muon":          (6.5e-3, 0.1),
    "signsgd":       (1.5e-4, 0.8),
    "sgd":           (1e-1, 0.95),
    "adam":          (7e-2, 0.0),
}


def parse_lr_grid(spec: str) -> List[float]:
    """``"lo:hi:step"`` (linear) or ``"lo:hi:xN"`` (log, N per decade)."""
    lo, hi, step = spec.split(":")
    lo, hi = float(lo), float(hi)
    if step.startswith("x"):
        per_decade = float(step[1:])
        if per_decade <= 0:
            raise ValueError(f"points per decade must be positive: {spec!r}")
        decades = math.log10(hi / lo)
        n = int(round(decades * per_decade))
        return [lo * 10.0 ** (i * decades / n) for i in range(n + 1)]
    step = float(step)
    if step <= 0:
        raise ValueError(f"grid step must be positive: {spec!r}")
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 12) for i in range(n + 1)]


def parse_momentum_grid(spec: str) -> List[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


# --------------------------------------------------------------------------
# Step-size schedules
# --------------------------------------------------------------------------

#: ``eta_t = eta * SCHEDULES[name](t, T)``. Tuned per method rather than fixed:
#: the constant-step bound and the decaying-step bound are different theorems
#: and there is no reason all six methods should prefer the same one.
SCHEDULES: Dict[str, Callable[[int, int], float]] = {
    "const":  lambda t, T: 1.0,
    "sqrt":   lambda t, T: 1.0 / math.sqrt(1.0 + t),
    "linear": lambda t, T: max(0.0, 1.0 - t / max(1, T)),
}


# --------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------


class Quadratic:
    """``F(X) = 1/2 <X-C, A(X-C)B>`` with everything about it known exactly.

    ``L`` and ``sigma`` are the extreme eigenvalues of the Hessian ``B (x) A``,
    i.e. the extreme products ``lambda_i(A) lambda_j(B)``, so no estimation is
    needed anywhere. ``F_star = 0`` at ``X = C``.
    """

    def __init__(self, m: int, n: int, device, seed: int,
                 spectrum: str = "uniform", kappa: float = 1e4,
                 basis: str = "haar", shift: float = 0.0):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        # A is drawn to completion before B, matching the draw order of the
        # pre-2026-07 generator so that ``--mode grid`` keeps generating the
        # instance the paper's table was tuned on.
        la = self._spectrum(m, spectrum, kappa, device, gen)
        self.A = self._embed(la, basis, device, gen)
        lb = self._spectrum(n, spectrum, kappa, device, gen)
        self.B = self._embed(lb, basis, device, gen)

        prod = torch.outer(la, lb)
        self.L = float(prod.max())
        self.sigma = float(prod.min())
        self.kappa = self.L / self.sigma

        self.C = (torch.randn(m, n, device=device, generator=gen) * shift
                  if shift else None)
        self.m, self.n = m, n

    @staticmethod
    def _spectrum(dim: int, kind: str, kappa: float, device, gen) -> Tensor:
        if kind == "uniform":
            # The paper's construction: eigenvalues ~ U(0,1), so L <= 1 but the
            # condition number is whatever the draw gives.
            return torch.rand(dim, device=device, generator=gen)
        if kind == "logspace":
            # Each factor gets condition number sqrt(kappa) and top eigenvalue 1,
            # so the Kronecker product has L = 1 and condition number exactly
            # kappa.
            lo = kappa ** -0.5
            return torch.logspace(math.log10(lo), 0.0, dim, device=device)
        raise ValueError(f"unknown spectrum {kind!r}")

    @staticmethod
    def _embed(eigenvalues: Tensor, basis: str, device, gen) -> Tensor:
        if basis == "diagonal":
            return torch.diag(eigenvalues)
        if basis == "haar":
            dim = eigenvalues.numel()
            Q, _ = torch.linalg.qr(
                torch.randn(dim, dim, device=device, generator=gen))
            return Q @ torch.diag(eigenvalues) @ Q.T
        raise ValueError(f"unknown basis {basis!r}")

    def grad(self, X: Tensor) -> Tensor:
        D = X if self.C is None else X - self.C
        return self.A @ D @ self.B

    def value_and_grad(self, X: Tensor) -> Tuple[float, Tensor]:
        D = X if self.C is None else X - self.C
        G = self.A @ D @ self.B
        return 0.5 * float(torch.sum(D * G)), G

    def initial_point(self, seed: int, scale: float = 0.1) -> Tensor:
        gen = torch.Generator(device=self.A.device)
        gen.manual_seed(int(seed))
        return torch.randn(self.m, self.n, device=self.A.device,
                           generator=gen) * scale


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------


def _build(method: str, X: Tensor, kwargs: Dict[str, float], lmo_dtype: str):
    cls = METHOD_CLASSES[method]
    opt_kwargs = dict(kwargs)
    opt_kwargs.pop("schedule", None)
    if method in ("sgd", "adam"):
        opt_kwargs.pop("nesterov", None)
        if method == "adam":
            opt_kwargs.pop("momentum", None)
    else:
        opt_kwargs["lmo_dtype"] = getattr(torch, lmo_dtype)
    return cls([X], **opt_kwargs)


def run_one(
    method: str,
    kwargs: Dict[str, float],
    problem: Quadratic,
    target_loss: float = 1e-3,
    max_iters: int = 5000,
    init_seed: int = 42,
    lmo_dtype: str = "bfloat16",
    schedule: str = "const",
    capture_alignment: bool = False,
    keep_history: bool = False,
    stop_at_target: bool = False,
) -> Dict:
    """Run one ``(method, hyperparameter)`` configuration for ``max_iters`` steps.

    ``iters_to_converge`` records the first iteration at which
    ``F <= target_loss`` (``max_iters`` if never). ``stop_at_target`` returns
    there, which is what the paper's protocol needs and all it needs; the sweeps
    leave it off because they read the plateau, which only exists after the
    target is passed.

    ``init_seed`` fixes ``X_0``, so every configuration starts from the same
    point and the comparison is not confounded by initialization.
    """
    X = torch.nn.Parameter(problem.initial_point(init_seed))
    optimizer = _build(method, X, kwargs, lmo_dtype)
    lr0 = kwargs["lr"]
    sched = SCHEDULES[schedule]

    if capture_alignment and hasattr(optimizer, "capture_direction"):
        optimizer.capture_direction = True

    def tracked() -> Tensor:
        """The iterate the convergence theory bounds.

        EF21MuonSign holds the compressed broadcast model ``W`` in ``p.data`` and
        the exact model ``X`` in its state; every other method has only one.
        """
        return optimizer.state.get(X, {}).get("exact_model", X.data)

    iters_to_converge = max_iters
    best_f = math.inf
    best_gnorm = math.inf
    f_hist: List[float] = []
    g_hist: List[float] = []
    rho_hist: List[float] = []
    diverged = False
    start = time.time()

    for t in range(max_iters):
        # Gradient at p.data -- which is W for the bidirectional method, exactly
        # as its algorithm requires -- and metrics at the tracked iterate.
        X.grad = problem.grad(X.data)
        f_val, g_tracked = problem.value_and_grad(tracked())
        g_norm = float(g_tracked.norm())

        f_hist.append(f_val)
        g_hist.append(g_norm)
        best_f = min(best_f, f_val)
        best_gnorm = min(best_gnorm, g_norm)
        if f_val <= target_loss and iters_to_converge == max_iters:
            iters_to_converge = t + 1
            if stop_at_target:
                break

        if not math.isfinite(f_val) or f_val > 1e12:
            diverged = True
            break

        for group in optimizer.param_groups:
            group["lr"] = lr0 * sched(t, max_iters)
        optimizer.step()

        if capture_alignment:
            d = optimizer.state[X].get("last_direction")
            if d is not None:
                denom = g_norm * float(d.norm())
                if denom > 0:
                    rho_hist.append(float(torch.sum(g_tracked * d)) / denom)

    out = {
        "method": method,
        "kwargs": {k: v for k, v in kwargs.items()},
        "schedule": schedule,
        "iters_to_converge": iters_to_converge,
        "reached_target": iters_to_converge < max_iters,
        "best_f": best_f,
        "best_gnorm": best_gnorm,
        "final_loss": f_hist[-1] if f_hist else float("nan"),
        "diverged": diverged,
        "time_seconds": time.time() - start,
    }
    if rho_hist:
        rho_hist.sort()
        k = len(rho_hist)
        out["rho"] = {
            "min": rho_hist[0],
            "p01": rho_hist[max(0, k // 100)],
            "median": rho_hist[k // 2],
            "mean": sum(rho_hist) / k,
            "max": rho_hist[-1],
            "frac_negative": sum(1 for r in rho_hist if r < 0) / k,
        }
    if keep_history:
        out["loss_history"] = f_hist
        out["grad_norm_history"] = g_hist
    return out


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------

#: ``objective -> (metric key, want-small)``. ``iters`` is the paper's protocol;
#: ``best_f`` / ``best_gnorm`` are what the theory's rate statements are about.
OBJECTIVES = {
    "iters": ("iters_to_converge", True),
    "best_f": ("best_f", True),
    "best_gnorm": ("best_gnorm", True),
}


def _score(metrics: Dict, objective: str) -> Tuple[float, float]:
    key, _ = OBJECTIVES[objective]
    value = metrics[key]
    if not math.isfinite(value):
        return (math.inf, math.inf)
    # Ties on the iteration count (very common -- it is an integer) are broken by
    # the loss actually reached.
    return (value, metrics["best_f"])


def tune(
    method: str,
    problems: Sequence[Quadratic],
    lrs: Sequence[float],
    momenta: Sequence[float],
    schedules: Sequence[str],
    objective: str = "iters",
    verbose: bool = True,
    **run_kwargs,
) -> Dict:
    """Grid-search ``(lr, momentum, schedule)``, averaging over problem instances.

    Averaging is geometric on the error metrics and arithmetic on the iteration
    count. Multiple instances are the point: one draw of ``A``, ``B`` is a single
    sample of a random problem, and the paper reports one.
    """
    best: Optional[Dict] = None
    boundary_lr = (lrs[0], lrs[-1])
    boundary_mom = (momenta[0], momenta[-1])

    for lr in lrs:
        for mom in (momenta if method != "adam" else [0.0]):
            for sch in schedules:
                cfg = {"lr": lr} if method == "adam" else {"lr": lr, "momentum": mom}
                runs = [run_one(method, cfg, prob, schedule=sch, **run_kwargs)
                        for prob in problems]
                agg = _aggregate(runs)
                if best is None or _score(agg, objective) < _score(best, objective):
                    best = agg
                if verbose:
                    it = agg["iters_to_converge"]
                    reached = agg["reached_target"]
                    print(f"    lr={lr:<10.4g} mom={mom:<5} sch={sch:<7}"
                          f" iters={(f'{it:.0f}' if reached else 'none'):<8}"
                          f" best_f={agg['best_f']:.3e}"
                          f" best_gn={agg['best_gnorm']:.3e}")

    assert best is not None
    edges = []
    if best["kwargs"]["lr"] in boundary_lr and len(lrs) > 1:
        edges.append("lr")
    if best["kwargs"].get("momentum") in boundary_mom and len(momenta) > 1:
        edges.append("momentum")
    best["on_grid_boundary"] = edges
    if edges and verbose:
        print(f"  !! {method}: tuned {', '.join(edges)} sits on the grid "
              f"boundary -- this is an upper bound, not an optimum. Widen with "
              f"--lr-grid / --momentum-grid.")
    return best


def _aggregate(runs: Sequence[Dict]) -> Dict:
    """Combine repeats over problem instances into one record."""
    if len(runs) == 1:
        return dict(runs[0])
    out = dict(runs[0])
    out["n_instances"] = len(runs)
    out["reached_target"] = all(r["reached_target"] for r in runs)
    out["diverged"] = any(r["diverged"] for r in runs)
    out["iters_to_converge"] = sum(r["iters_to_converge"] for r in runs) / len(runs)
    for key in ("best_f", "best_gnorm", "final_loss"):
        out[key] = _geomean([r[key] for r in runs])
    if "rho" in runs[0]:
        out["rho"] = {k: (min(r["rho"][k] for r in runs) if k in ("min", "p01")
                          else sum(r["rho"][k] for r in runs) / len(runs))
                      for k in runs[0]["rho"]}
    out.pop("loss_history", None)
    out.pop("grad_norm_history", None)
    return out


def _geomean(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v) and v > 0]
    if not vals:
        return math.inf
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def loglog_fit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Least-squares slope of ``log y`` on ``log x``, and its ``R^2``.

    Returned as ``(slope, r2)``; the exponent reported in the tables is
    ``-slope``. ``R^2`` is reported alongside because a clean exponent on a
    two-point fit means nothing.
    """
    pts = [(x, y) for x, y in zip(xs, ys)
           if x > 0 and y > 0 and math.isfinite(y)]
    if len(pts) < 2:
        return (float("nan"), float("nan"))
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    k = len(pts)
    mx, my = sum(lx) / k, sum(ly) / k
    sxx = sum((a - mx) ** 2 for a in lx)
    if sxx == 0:
        return (float("nan"), float("nan"))
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
    ss_res = sum((b - my - slope * (a - mx)) ** 2 for a, b in zip(lx, ly))
    ss_tot = sum((b - my) ** 2 for b in ly)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (slope, r2)


def step_norm(method: str, m: int, n: int) -> float:
    """``||s||_F`` of one unit step for the **exact** LMO (see ``common.lr_scaling``).

    The sign family realizes ``sqrt(mn)`` exactly -- a ``+-1`` matrix has no
    approximation in it. The LMO family does not: five Newton-Schulz steps leave
    the singular values in a band around 1 rather than at 1, so the realized step
    is measurably shorter than ``sqrt(min(m,n))`` -- 0.87x at ``64x64``, 0.94x at
    ``500x500``, and 0.78x at the most rectangular ResNet-18 shape (``64x576``).
    Anything comparing an LMO method's measured floor against ``L ||s||_F^2 / 2``
    should expect that much slack, and the same factor silently rescales the
    effective learning rate of every LMO-family run relative to the theory.
    """
    if method in SIGN_FAMILY:
        return math.sqrt(m * n)
    if method in LMO_FAMILY:
        return math.sqrt(min(m, n))
    return float("nan")            # sgd / adam: not a normalized step


# --------------------------------------------------------------------------
# Closed-form alignment references
# --------------------------------------------------------------------------


def rho_reference(method: str, G: Tensor) -> Optional[float]:
    """``rho`` predicted in closed form at gradient ``G``, where one exists.

    ``SGD``: ``d = G``, so ``rho = 1``.
    ``SignSGD``: ``d = sign(G)``, so ``rho = ||G||_1 / (||G||_F sqrt(mn))``,
    which tends to ``sqrt(2/pi) ~ 0.798`` as ``G`` becomes entrywise Gaussian.
    ``Muon``: ``d = polar(G)``, so ``rho = ||G||_nuc / (||G||_F sqrt(r))``.
    The three sign-around-the-LMO methods have no closed form; that is the point.
    """
    m, n = G.shape
    fro = float(G.norm())
    if fro == 0:
        return None
    if method == "sgd":
        return 1.0
    if method == "signsgd":
        return float(G.abs().sum()) / (fro * math.sqrt(m * n))
    if method == "muon":
        s = torch.linalg.svdvals(G.float())
        r = int((s > 1e-10 * s[0]).sum())
        return float(s.sum()) / (fro * math.sqrt(r))
    return None


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def tuned_hyperparameters(out_root: Path, method: str
                          ) -> Optional[Tuple[float, float, str, str]]:
    """``(lr, momentum, schedule, source)`` for ``--mode final``.

    A grid result written by this repository wins over ``REPORTED_BEST``, which
    is only the paper's printed table and goes stale the moment the grid is
    re-run. The source is returned so the run log says which was used.
    """
    path = out_root / method / "grid.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            result = json.load(f)["result"]
        return (result["kwargs"]["lr"], result["kwargs"].get("momentum", 0.0),
                result.get("schedule", "const"), str(path))
    best = REPORTED_BEST.get(method)
    return (best[0], best[1], "const", "paper table") if best else None


def output_slug(args) -> str:
    """Basename of the JSON a run writes, inside ``results/synthetic/<method>/``.

    The grid preset is part of it. ``--grid-preset paper`` re-runs the *grid*
    mode on Table 3's published ranges purely to document how the old table was
    produced; writing that into ``grid.json`` would overwrite the current tuned
    optimum with the superseded one -- and ``--mode final`` reads ``grid.json``,
    so the two would silently swap. They are different measurements and get
    different files.
    """
    if args.mode == "grid" and getattr(args, "grid_preset", "default") != "default":
        return f"grid-{args.grid_preset}"
    return args.mode


def mode_grid(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """The paper's protocol: fewest iterations to the target loss."""
    summary = []
    for method in args.methods:
        schedules = args.schedules
        if args.mode == "final":
            best = tuned_hyperparameters(out_root, method)
            if best is None:
                print(f"{method}: no tuned hyperparameters anywhere -- run "
                      f"--mode grid first. Skipping.")
                continue
            lrs, moms, schedules = [best[0]], [best[1]], [best[2]]
            print(f"--- {method}: lr={best[0]:g} momentum={best[1]:g} "
                  f"schedule={best[2]} (from {best[3]}) ---")
        else:
            lrs = parse_lr_grid(lr_grids[method])
            moms = momenta
            n_cfg = (len(lrs) * (1 if method == "adam" else len(moms))
                     * len(schedules))
            print(f"--- {method}: {n_cfg} configuration(s) x "
                  f"{len(problems)} instance(s) ---")
        best_metrics = tune(
            method, problems, lrs, moms, schedules, objective="iters",
            target_loss=args.target_loss, max_iters=args.max_iters,
            init_seed=args.init_seed, lmo_dtype=args.lmo_dtype,
            keep_history=args.save_histories and args.mode == "final",
            stop_at_target=True,
        )
        _write(out_root, method, output_slug(args), args, problems, best_metrics)
        summary.append(best_metrics)

    print(f"\n{'method':<16}{'iters':>10}{'best F':>14}{'||g||':>12}   hyperparameters")
    print("-" * 84)
    for m in summary:
        it = (f"{m['iters_to_converge']:.0f}" if m["reached_target"]
              else f">{args.max_iters}")
        kw = ", ".join(f"{k}={v:g}" for k, v in m["kwargs"].items())
        flag = "  [BOUNDARY]" if m.get("on_grid_boundary") else ""
        print(f"{m['method']:<16}{it:>10}{m['best_f']:>14.3e}"
              f"{m['best_gnorm']:>12.3e}   {kw}, {m['schedule']}{flag}")
    return summary


def mode_alignment(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Distribution of ``rho_t = <g, d> / (||g|| ||d||)`` along a tuned run."""
    prob = problems[0]
    G0 = prob.grad(prob.initial_point(args.init_seed))

    summary = []
    for method in args.methods:
        lrs = parse_lr_grid(lr_grids[method])
        print(f"--- {method} ---")
        best = tune(method, problems, lrs, momenta, args.schedules,
                    objective="best_gnorm", verbose=args.verbose,
                    target_loss=args.target_loss, max_iters=args.align_iters,
                    init_seed=args.init_seed, lmo_dtype=args.lmo_dtype,
                    capture_alignment=True)
        best["rho_reference_at_X0"] = rho_reference(method, G0)
        _write(out_root, method, "alignment", args, problems, best)
        summary.append(best)

    print(f"\n{'method':<16}{'min rho':>10}{'p01':>10}{'median':>10}{'mean':>10}"
          f"{'%neg':>8}{'ref@X0,mu=0':>13}{'tuned':>18}")
    print("-" * 97)
    for m in summary:
        r = m.get("rho")
        ref = m["rho_reference_at_X0"]
        ref_s = f"{ref:.4f}" if ref is not None else "--"
        cfg = (f"eta={m['kwargs']['lr']:.3g},mu="
               f"{m['kwargs'].get('momentum', 0):g}")
        if r is None:
            # torch.optim's SGD and Adam have no capture hook. SGD's rho is 1 by
            # definition; Adam's step is not one the paper's descent lemma covers.
            print(f"{m['method']:<16}" + f"{'--':>10}" * 4
                  + f"{'--':>8}{ref_s:>13}{cfg:>18}")
            continue
        print(f"{m['method']:<16}{r['min']:>10.4f}{r['p01']:>10.4f}"
              f"{r['median']:>10.4f}{r['mean']:>10.4f}"
              f"{100 * r['frac_negative']:>7.2f}%{ref_s:>13}{cfg:>18}")
    print("\n  rho > 0 throughout means the descent lemma bites on every step of "
          "this trajectory:\n  the divergence theorems' construction does not "
          "occur under random data.\n"
          "  The reference column is the closed form for d = compressor(grad F) at "
          "X_0 with no\n  momentum, so it is only comparable to a row whose tuned "
          "mu is 0; with momentum the\n  step is built from M_t rather than the "
          "current gradient and rho drops accordingly.")
    return summary


def mode_horizon(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Tune per budget ``T`` and fit ``err ~ T^-p``, ``eta* ~ T^-q``."""
    budgets = args.budgets
    summary = []
    for method in args.methods:
        lrs = parse_lr_grid(lr_grids[method])
        rows = []
        for T in budgets:
            print(f"--- {method}, T={T} ---")
            best = tune(method, problems, lrs, momenta, args.schedules,
                        objective=args.objective, verbose=args.verbose,
                        target_loss=0.0, max_iters=T,
                        init_seed=args.init_seed, lmo_dtype=args.lmo_dtype)
            rows.append({"T": T, "lr": best["kwargs"]["lr"],
                         "momentum": best["kwargs"].get("momentum"),
                         "schedule": best["schedule"],
                         "best_f": best["best_f"],
                         "best_gnorm": best["best_gnorm"],
                         "on_grid_boundary": best["on_grid_boundary"]})
            print(f"  -> lr*={rows[-1]['lr']:.4g} sch={rows[-1]['schedule']} "
                  f"best_f={rows[-1]['best_f']:.4e} "
                  f"best_gn={rows[-1]['best_gnorm']:.4e}")

        p_f, r2_f = loglog_fit(budgets, [r["best_f"] for r in rows])
        p_g, r2_g = loglog_fit(budgets, [r["best_gnorm"] for r in rows])
        q, r2_q = loglog_fit(budgets, [r["lr"] for r in rows])
        rec = {"method": method, "rows": rows,
               "exponent_f": -p_f, "r2_f": r2_f,
               "exponent_gnorm": -p_g, "r2_gnorm": r2_g,
               "exponent_lr": -q, "r2_lr": r2_q}
        _write(out_root, method, "horizon", args, problems, rec)
        summary.append(rec)

    print(f"\n{'method':<16}{'p (||g||)':>12}{'R2':>7}{'p (F)':>10}{'R2':>7}"
          f"{'q (eta*)':>11}{'R2':>7}")
    print("-" * 70)
    for r in summary:
        print(f"{r['method']:<16}{r['exponent_gnorm']:>12.3f}{r['r2_gnorm']:>7.3f}"
              f"{r['exponent_f']:>10.3f}{r['r2_f']:>7.3f}"
              f"{r['exponent_lr']:>11.3f}{r['r2_lr']:>7.3f}")
    print("\n  Two predictions, and the fit says which regime the problem is in."
          "\n    p = q = 1/2  the nonconvex L-smooth bound the paper's theorems "
          "prove: the rate\n                 term (F_0-F_*)/(eta T) balances the "
          "floor eta L ||s||_F^2 / 2.\n    p = q = 1    strongly convex, which "
          "this quadratic is (sigma > 0): the rate term\n                 "
          "contracts geometrically instead, so the budget-optimal eta is just "
          "the\n                 smallest one that finishes contracting within "
          "T, and the error is\n                 floor-limited at eta ~ 1/T."
          "\n  SGD converges linearly with no floor, so no power law fits it at "
          "all -- expect a\n  large p, and read its q as 'eta* is the "
          "stability-limited one at every budget'.")
    return summary


def mode_floor(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Plateau of a constant step, and its exponent in ``eta``."""
    mom = momenta[0]
    summary = []
    for method in args.methods:
        lrs = parse_lr_grid(lr_grids[method])
        lr_max = max(lrs)
        rows = []
        print(f"--- {method} (momentum={mom}) ---")
        for lr in lrs:
            # Time to reach the plateau scales like 1/eta, so a fixed budget
            # would leave the small-eta runs still descending and fake a floor
            # that is really just "how far it got". Budget is scaled to match,
            # capped so the sweep terminates.
            iters = min(args.floor_max_iters,
                        int(round(args.floor_iters * lr_max / lr)))
            cfg = {"lr": lr} if method == "adam" else {"lr": lr, "momentum": mom}
            runs = [run_one(method, cfg, p, schedule="const", target_loss=0.0,
                            max_iters=iters, init_seed=args.init_seed,
                            lmo_dtype=args.lmo_dtype, keep_history=True)
                    for p in problems]
            f_pairs = [_plateau(r["loss_history"]) for r in runs]
            g_pairs = [_plateau(r["grad_norm_history"]) for r in runs]
            f_inf = _geomean([v for v, _ in f_pairs])
            g_inf = _geomean([v for v, _ in g_pairs])
            stable = all(not r["diverged"] for r in runs)
            settled = stable and all(s for _, s in g_pairs)
            rows.append({"lr": lr, "iters": iters, "f_inf": f_inf, "g_inf": g_inf,
                         "stable": stable, "settled": settled})
            note = ("" if settled else
                    ("   DIVERGED" if not stable else "   still descending"))
            print(f"    lr={lr:<10.4g} T={iters:<7} F_inf={f_inf:.4e} "
                  f"|g|_inf={g_inf:.4e}{note}")

        good = [r for r in rows if r["settled"]]
        if len(good) < 2:
            print(f"  {method}: fewer than two settled points -- no floor to fit. "
                  f"For SGD that is the correct answer (its step vanishes with "
                  f"the gradient, so it converges linearly here); for a "
                  f"normalized step it means the budget was too short -- raise "
                  f"--floor-iters.")
        sf, r2f = loglog_fit([r["lr"] for r in good], [r["f_inf"] for r in good])
        sg, r2g = loglog_fit([r["lr"] for r in good], [r["g_inf"] for r in good])
        rec = {"method": method, "momentum": mom, "rows": rows,
               "n_settled": len(good),
               "slope_f": sf, "r2_f": r2f, "slope_gnorm": sg, "r2_gnorm": r2g,
               "step_norm": step_norm(method, args.m, args.n)}
        _write(out_root, method, "floor", args, problems, rec)
        summary.append(rec)

    L = problems[0].L
    print(f"\n{'method':<16}{'settled':>9}{'d log|g|/d log eta':>20}{'R2':>7}"
          f"{'d log F/d log eta':>19}{'R2':>7}{'L||s||^2/2':>13}")
    print("-" * 91)
    for r in summary:
        s = r["step_norm"]
        pred = L * s * s / 2 if math.isfinite(s) else float("nan")
        pred_s = f"{pred:.4g}" if math.isfinite(pred) else "--"
        n = f"{r['n_settled']}/{len(r['rows'])}"
        if r["n_settled"] < 2:
            print(f"{r['method']:<16}{n:>9}{'no floor':>20}{'':>7}{'':>19}"
                  f"{'':>7}{pred_s:>13}")
            continue
        print(f"{r['method']:<16}{n:>9}{r['slope_gnorm']:>20.3f}{r['r2_gnorm']:>7.3f}"
              f"{r['slope_f']:>19.3f}{r['r2_f']:>7.3f}{pred_s:>13}")
    print("\n  The descent lemma predicts a gradient floor linear in eta "
          "(slope 1) with\n  coefficient L||s||_F^2 / (2 rho). SignMuon and "
          "SignSGD share ||s||_F^2 = mn\n  exactly, so any gap between their "
          "floors is attributable to rho alone.")
    return summary


def mode_stability(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """Largest stable ``eta``, by bisection on ``log eta``."""
    mom = momenta[0]
    T = args.stability_iters
    summary = []

    def stable(method: str, lr: float) -> bool:
        cfg = {"lr": lr} if method == "adam" else {"lr": lr, "momentum": mom}
        for p in problems:
            r = run_one(method, cfg, p, schedule="const", target_loss=0.0,
                        max_iters=T, init_seed=args.init_seed,
                        lmo_dtype=args.lmo_dtype)
            f0, _ = p.value_and_grad(p.initial_point(args.init_seed))
            if r["diverged"] or not (r["best_f"] < f0):
                return False
        return True

    for method in args.methods:
        lo, hi = args.stability_lo, args.stability_hi
        if not stable(method, lo):
            print(f"--- {method}: unstable already at eta={lo:g}; widen "
                  f"--stability-lo ---")
            continue
        censored = stable(method, hi)
        if censored:
            # Not a measurement: everything below is a lower bound. Adam lands
            # here by construction -- its step is bounded by roughly lr whatever
            # the gradient does, so it oscillates rather than diverging and has
            # no stability edge of this kind.
            print(f"--- {method}: still stable at eta={hi:g}; this is a LOWER "
                  f"BOUND, widen --stability-hi to measure it ---")
            eta_max = hi
        else:
            for _ in range(args.stability_steps):
                mid = math.sqrt(lo * hi)
                if stable(method, mid):
                    lo = mid
                else:
                    hi = mid
            eta_max = lo
        s = step_norm(method, args.m, args.n)
        rec = {"method": method, "eta_max": eta_max, "step_norm": s,
               "step_length": eta_max * s, "momentum": mom, "iters": T,
               "censored": censored}
        print(f"--- {method}: eta_max = {eta_max:.4g}, "
              f"eta_max*||s||_F = {eta_max * s:.4g} ---")
        _write(out_root, method, "stability", args, problems, rec)
        summary.append(rec)

    L = problems[0].L
    print(f"\n{'method':<16}{'eta_max':>12}{'||s||_F':>10}"
          f"{'eta_max*||s||_F':>18}{'x (2/L)':>10}")
    print("-" * 66)
    for r in summary:
        eta = ("> " if r["censored"] else "") + f"{r['eta_max']:.4g}"
        if not math.isfinite(r["step_norm"]):
            # sgd / adam: the step is not normalized, so there is no ||s||_F.
            # SGD's eta_max is directly comparable to the textbook 2/L instead.
            print(f"{r['method']:<16}{eta:>12}{'--':>10}{'--':>18}"
                  f"{r['eta_max'] / (2 / L):>10.3f}")
            continue
        print(f"{r['method']:<16}{eta:>12}{r['step_norm']:>10.4g}"
              f"{r['step_length']:>18.4g}{r['step_length'] / (2 / L):>10.3f}")
    print(f"\n  L = {L:.4g}. SGD is the control: its eta_max must land on the "
          f"textbook 2/L.\n  If the operative trust region were the Frobenius "
          f"ball, eta_max*||s||_F\n  would be family-independent and near 2/L = "
          f"{2 / L:.4g}; the spread measures how far\n  off the Frobenius bound "
          f"is for each step geometry.")
    return summary


def mode_kappa(args, problems, lr_grids, momenta, out_root) -> List[Dict]:
    """The tuned comparison, swept over a controlled condition number."""
    summary = []
    probs = problems
    for method in args.methods:
        lrs = parse_lr_grid(lr_grids[method])
        rows = []
        for kappa in args.kappas:
            probs = [Quadratic(args.m, args.n, args.device, seed,
                               spectrum="logspace", kappa=kappa,
                               basis=args.basis, shift=args.shift)
                     for seed in args.problem_seeds]
            print(f"--- {method}, kappa={kappa:g} ---")
            best = tune(method, probs, lrs, momenta, args.schedules,
                        objective=args.objective, verbose=args.verbose,
                        target_loss=args.target_loss, max_iters=args.max_iters,
                        init_seed=args.init_seed, lmo_dtype=args.lmo_dtype)
            rows.append({"kappa": kappa, "lr": best["kwargs"]["lr"],
                         "iters": best["iters_to_converge"],
                         "reached": best["reached_target"],
                         "best_f": best["best_f"],
                         "best_gnorm": best["best_gnorm"]})
            print(f"  -> lr*={rows[-1]['lr']:.4g} iters={rows[-1]['iters']:.0f} "
                  f"best_gn={rows[-1]['best_gnorm']:.4e}")
        slope, r2 = loglog_fit([r["kappa"] for r in rows],
                               [r["best_gnorm"] for r in rows])
        rec = {"method": method, "rows": rows, "kappas": list(args.kappas),
               "exponent_kappa": slope, "r2_kappa": r2}
        # ``probs`` is the last kappa's instance list; the per-kappa L and sigma
        # are in ``rows``, this only records the shared shape/basis metadata.
        _write(out_root, method, "kappa", args, probs, rec)
        summary.append(rec)

    header = f"{'method':<16}" + "".join(f"{k:>12.0e}" for k in args.kappas)
    print(f"\nbest ||grad F|| after {args.max_iters} tuned iterations")
    print(header)
    print("-" * len(header))
    for r in summary:
        print(f"{r['method']:<16}"
              + "".join(f"{row['best_gnorm']:>12.3e}" for row in r["rows"]))
    print(f"\n{'method':<16}{'d log||g|| / d log kappa':>26}{'R2':>7}")
    print("-" * 49)
    for r in summary:
        print(f"{r['method']:<16}{r['exponent_kappa']:>26.3f}{r['r2_kappa']:>7.3f}")
    return summary


MODES = {
    "grid": mode_grid,
    "final": mode_grid,
    "alignment": mode_alignment,
    "horizon": mode_horizon,
    "floor": mode_floor,
    "stability": mode_stability,
    "kappa": mode_kappa,
}

#: ``grid``/``final`` keep the paper's 500x500 problem; the sweeps run many more
#: configurations and default to 100x100, which the tuned rankings are unchanged
#: by and which turns hours into minutes.
DEFAULT_SIZE = {"grid": 500, "final": 500}


def _plateau(history: Sequence[float], tol: float = 0.15) -> Tuple[float, bool]:
    """``(level, settled)`` for the tail of a constant-step trajectory.

    ``level`` is the mean over the last quarter. ``settled`` compares it against
    the third quarter: a run that is still descending has not reached its floor,
    and reading one off it would report "how far it got in the budget" instead.
    That distinction is the whole point of this mode, so non-settled points are
    excluded from the fit rather than silently averaged in.

    Every *normalized* step has a floor, which is all of them except SGD: an LMO
    step has fixed norm ``sqrt(r)`` and a sign step fixed norm ``sqrt(mn)``
    whatever the gradient does, so a constant ``eta`` cannot converge. Muon is
    not exempt; only plain SGD, whose step vanishes with the gradient, converges
    linearly here and never settles.
    """
    k = len(history)
    q3 = [v for v in history[k // 2: 3 * k // 4] if math.isfinite(v)]
    q4 = [v for v in history[3 * k // 4:] if math.isfinite(v)]
    if not q4 or not q3:
        return (float("inf"), False)
    m4 = sum(q4) / len(q4)
    m3 = sum(q3) / len(q3)
    if m4 <= 0:
        return (m4, False)
    return (m4, abs(m4 - m3) / m4 < tol)


def _write(out_root: Path, method: str, mode: str, args, problems, payload) -> None:
    method_dir = out_root / method
    method_dir.mkdir(parents=True, exist_ok=True)
    p0 = problems[0]
    body = {
        "problem": {
            "m": args.m, "n": args.n, "spectrum": args.spectrum,
            "basis": args.basis, "shift": args.shift,
            "L": p0.L, "sigma": p0.sigma, "condition_number": p0.kappa,
            "target_loss": args.target_loss, "max_iters": args.max_iters,
            "problem_seeds": args.problem_seeds, "init_seed": args.init_seed,
            "lmo_dtype": args.lmo_dtype,
        },
        "schedules": args.schedules,
        "result": payload,
    }
    with open(method_dir / f"{mode}.json", "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), default="grid")
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS,
                   choices=DEFAULT_METHODS)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--verbose", action="store_true",
                   help="print every configuration, not just the tuned one")

    g = p.add_argument_group("problem")
    g.add_argument("--m", type=int, default=None,
                   help="default: 500 for grid/final, 100 for the sweeps")
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--spectrum", choices=["uniform", "logspace"], default="uniform",
                   help="uniform: the paper's U(0,1) draw (L<=1, kappa uncontrolled); "
                        "logspace: L=1 and condition number exactly --kappa")
    g.add_argument("--kappa", type=float, default=1e4,
                   help="condition number for --spectrum logspace")
    g.add_argument("--basis", choices=["haar", "diagonal"], default="haar",
                   help="eigenbasis of A and B; Muon and SGD are equivariant "
                        "under it, the sign family is not")
    g.add_argument("--shift", type=float, default=0.0,
                   help="entrywise scale of a random minimizer C; 0 puts the "
                        "minimizer at the origin, which is a special point for "
                        "sign methods")
    g.add_argument("--problem-seeds", type=int, nargs="+", default=None,
                   help="one instance per seed, results averaged "
                        "(default: [1337] for grid/final, three seeds otherwise)")
    g.add_argument("--init-seed", type=int, default=42, help="seed for X_0")

    g = p.add_argument_group("run")
    g.add_argument("--target-loss", type=float, default=1e-3)
    g.add_argument("--max-iters", type=int, default=5000)
    g.add_argument("--lmo-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    g.add_argument("--objective", choices=list(OBJECTIVES), default="best_gnorm",
                   help="what the tuner minimizes in horizon/kappa mode "
                        "(grid/final always use the paper's iteration count)")

    g = p.add_argument_group("grids")
    g.add_argument("--grid-preset", choices=["default", "paper"], default="default",
                   help="'paper' restores Table 3's linear grids, two of whose "
                        "rows cannot reproduce their own reported optimum")
    g.add_argument("--lr-grid", nargs="*", default=[],
                   metavar="METHOD=lo:hi:step", help="override one method's grid; "
                   "'lo:hi:step' is linear, 'lo:hi:xN' is N points per decade")
    g.add_argument("--momentum-grid", type=str, default=None)
    g.add_argument("--schedules", nargs="+", default=None,
                   choices=list(SCHEDULES),
                   help="step-size schedules to tune over, separately per method "
                        "(default: 'const' everywhere except horizon mode, which "
                        "tunes over 'const sqrt' because nothing says the six "
                        "methods want the same schedule)")

    g = p.add_argument_group("mode-specific")
    g.add_argument("--budgets", type=int, nargs="+",
                   default=[125, 250, 500, 1000, 2000, 4000],
                   help="horizon mode: iteration budgets T")
    g.add_argument("--align-iters", type=int, default=2000,
                   help="alignment mode: trajectory length")
    g.add_argument("--floor-iters", type=int, default=3000,
                   help="floor mode: budget at the LARGEST eta in the grid; "
                        "smaller eta gets proportionally longer, since time to "
                        "plateau scales like 1/eta")
    g.add_argument("--floor-max-iters", type=int, default=60000,
                   help="floor mode: cap on that scaled budget")
    g.add_argument("--stability-iters", type=int, default=300)
    g.add_argument("--stability-lo", type=float, default=1e-6)
    g.add_argument("--stability-hi", type=float, default=1e2)
    g.add_argument("--stability-steps", type=int, default=20)
    g.add_argument("--kappas", type=float, nargs="+",
                   default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6])

    g = p.add_argument_group("output")
    g.add_argument("--out", type=str, default=None,
                   help="output directory (default: results/synthetic/)")
    g.add_argument("--save-histories", action="store_true",
                   help="store the full loss/grad-norm curves (large)")
    return p.parse_args()


def main() -> None:
    args = get_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    size = DEFAULT_SIZE.get(args.mode, 100)
    if args.m is None:
        args.m = size
    if args.n is None:
        args.n = size
    if args.problem_seeds is None:
        args.problem_seeds = ([1337] if args.mode in ("grid", "final")
                              else [1337, 1338, 1339])
    if args.momentum_grid is None:
        args.momentum_grid = (PAPER_MOMENTUM_GRID if args.grid_preset == "paper"
                              else DEFAULT_MOMENTUM_GRID)
    if args.schedules is None:
        args.schedules = ["const", "sqrt"] if args.mode == "horizon" else ["const"]

    lr_grids = dict(PAPER_LR_GRIDS if args.grid_preset == "paper"
                    else DEFAULT_LR_GRIDS)
    for override in args.lr_grid:
        key, _, spec = override.partition("=")
        if key not in METHOD_CLASSES:
            raise ValueError(f"--lr-grid: unknown method {key!r}")
        lr_grids[key] = spec
    momenta = parse_momentum_grid(args.momentum_grid)

    problems = [Quadratic(args.m, args.n, args.device, seed,
                          spectrum=args.spectrum, kappa=args.kappa,
                          basis=args.basis, shift=args.shift)
                for seed in args.problem_seeds]
    p0 = problems[0]

    print(f"mode={args.mode}  device={args.device}  {args.m}x{args.n}  "
          f"lmo_dtype={args.lmo_dtype}")
    print(f"problem: spectrum={args.spectrum} basis={args.basis} "
          f"shift={args.shift}  instances={len(problems)}")
    print(f"         L={p0.L:.6g}  sigma={p0.sigma:.6g}  "
          f"condition number={p0.kappa:.4g}\n")

    out_root = Path(args.out) if args.out else results_root() / "synthetic"
    out_root.mkdir(parents=True, exist_ok=True)

    MODES[args.mode](args, problems, lr_grids, momenta, out_root)
    print(f"\nResults written to {out_root.resolve()}")


if __name__ == "__main__":
    main()
