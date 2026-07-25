"""
Exact-SVD implementations of the eight optimizers studied in the paper
"SignMuon, MuonSign, and the Role of Error Feedback".

Every method follows the corresponding pseudocode box in the paper verbatim,
with two deliberate choices:

  * the Muon LMO is computed with an **exact SVD** (rank-truncated ``U V^T``)
    instead of the Newton-Schulz polynomial the paper uses in practice, and
  * momentum is the EMA / heavy-ball form of the centralized algorithm boxes
    ``M_t = mu*M_{t-1} + (1-mu)*G_t`` with an optional Nesterov look-ahead
    ``M_tilde = (1-mu)*G_t + mu*M_t``.

The momentum coefficient defaults to ``mu = 0.8`` and standard (non-Nesterov)
momentum, but both are parameters so the behaviour can be re-tested later.

Interface (used by ``run_counterexamples.py``)
----------------------------------------------
Each optimizer owns its model state.  The driver loop is simply::

    opt = SignMuon(shape, eta, mu, nesterov)
    for t in range(T):
        losses.append(loss_fn(opt.track_point()))   # record, then step
        G = grad_fn(opt.grad_point())
        opt.step(G)

``grad_point()`` is where the (stochastic) gradient is evaluated and
``track_point()`` is the model whose objective value we plot.  For every method
except EF21-MuonUDSign these coincide with the single model ``X``; the
bidirectional method evaluates the gradient at the *compressed* broadcast model
``W`` while the tracked/exact model is ``X``.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

# Newton-Schulz coefficients kept only for reference / documentation; the exact
# SVD path below is what the counterexamples use.
NS_COEFFS = (3.4445, 4.7750, 2.0315)


def muon_lmo(Y: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    r"""Muon LMO direction :math:`\mathbf D = \mathbf U \mathbf V^\top` via exact SVD.

    ``muon_lmo`` returns the (rank-truncated) orthogonal polar factor, i.e. the
    minimizer ``-A(Y)`` of the spectral-norm linear-minimization oracle.

    Only the singular directions with **nonzero** singular value are kept
    (``r = rank(Y)``).  The full ``U @ Vt`` would append an arbitrary orthonormal
    completion of the null space, which is *non-unique* on rank-deficient inputs
    (e.g. ``sign(G)`` is often low-rank) and would silently change the answer.
    ``muon_lmo`` is scale-invariant: ``muon_lmo(c*Y) == muon_lmo(Y)`` for ``c > 0``.
    """
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    if S.size == 0:
        return np.zeros_like(Y)
    r = int(np.count_nonzero(S > tol * S[0]))
    if r == 0:
        return np.zeros_like(Y)
    return U[:, :r] @ Vt[:r, :]


def scaled_sign(Y: np.ndarray) -> np.ndarray:
    r"""Contractive 1-bit compressor :math:`\mathrm{mean}|\mathbf Y|\,\operatorname{sign}(\mathbf Y)`.

    This is the ``USign`` / ``UDSign`` scaled-sign operator: it transmits one bit
    per entry (the sign) plus a single shared magnitude scalar per matrix.
    """
    return np.mean(np.abs(Y)) * np.sign(Y)


# --------------------------------------------------------------------------
# Base optimizer
# --------------------------------------------------------------------------


class Optimizer:
    """Common momentum bookkeeping shared by every method.

    Parameters
    ----------
    shape : tuple[int, int]
        Matrix parameter shape.
    eta : float
        Learning rate ``eta_t`` (constant here).
    mu : float
        Momentum coefficient ``mu``; the paper's experiments fix ``mu = 0.8``.
    nesterov : bool
        If ``True`` use the Nesterov look-ahead effective direction, otherwise
        the standard (default) momentum ``M_tilde = M_t``.
    """

    name = "Optimizer"

    def __init__(self, shape, eta: float, mu: float = 0.8, nesterov: bool = False):
        self.shape = tuple(shape)
        self.eta = float(eta)
        self.mu = float(mu)
        self.nesterov = bool(nesterov)
        self.X = np.zeros(self.shape)   # tracked / exact model
        self.M = np.zeros(self.shape)   # momentum buffer M_t

    # -- momentum ---------------------------------------------------------
    def _effective_direction(self, G: np.ndarray) -> np.ndarray:
        r"""Update ``M_t`` and return the effective direction ``\tilde M_t``.

        ``M_t     = mu * M_{t-1} + (1 - mu) * G_t``            (EMA momentum)
        ``M_tilde = M_t``                                      (default), or
        ``M_tilde = (1 - mu) * G_t + mu * M_t``                (Nesterov).
        """
        self.M = self.mu * self.M + (1.0 - self.mu) * G
        if self.nesterov:
            return (1.0 - self.mu) * G + self.mu * self.M
        return self.M

    # -- driver hooks -----------------------------------------------------
    def grad_point(self) -> np.ndarray:
        """Point at which the next gradient is evaluated."""
        return self.X

    def track_point(self) -> np.ndarray:
        """Model whose objective value is recorded / plotted."""
        return self.X

    def step(self, G: np.ndarray) -> None:                      # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# Reference methods
# --------------------------------------------------------------------------


class SignSGD(Optimizer):
    """Reference: sign compression of the (momentum) gradient itself."""

    name = "SignSGD"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        self.X = self.X - self.eta * np.sign(M_tilde)


class Muon(Optimizer):
    """Reference: full-precision Muon, ``X <- X - eta * LMO(M_tilde)``."""

    name = "Muon"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        self.X = self.X - self.eta * muon_lmo(M_tilde)


# --------------------------------------------------------------------------
# Sign-around-the-LMO methods (no error feedback)
# --------------------------------------------------------------------------


class SignMuon(Optimizer):
    """Algorithm ``central_alg``: sign **after** the LMO.

    ``D = LMO(M_tilde);  X <- X - eta * sign(D)``.
    """

    name = "SignMuon"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        D = muon_lmo(M_tilde)
        self.X = self.X - self.eta * np.sign(D)


class MuonUSign(Optimizer):
    """Algorithm ``alg:muon_usign`` (a.k.a. MuonSign): sign **before** the LMO.

    ``s = sign(M_tilde);  D = LMO(s);  X <- X - eta * D``.

    Because the LMO is scale-invariant, using the scaled sign instead of the
    plain sign gives the identical direction, so MuonUSign == MuonSign here.
    """

    name = "MuonUSign"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        s = np.sign(M_tilde)
        D = muon_lmo(s)
        self.X = self.X - self.eta * D


class MuonUDSign(Optimizer):
    """Algorithm ``alg:muon_udsign``: sign before *and* after the LMO.

    ``s_up = sign(M_tilde);  D = LMO(s_up);  X <- X - eta * sign(D)``.
    """

    name = "MuonUDSign"

    def step(self, G):
        M_tilde = self._effective_direction(G)
        s_up = np.sign(M_tilde)
        D = muon_lmo(s_up)
        s_down = np.sign(D)
        self.X = self.X - self.eta * s_down


# --------------------------------------------------------------------------
# Error-feedback (EF21) methods
# --------------------------------------------------------------------------


class EF21SignMuon(Optimizer):
    """Algorithm ``ef21_signmuon``: EF21 on the LMO *direction*.

    ``D = LMO(M_tilde)`` is tracked by a scaled-sign EF21 estimator
    ``d_est`` and the step is ``X <- X - eta * d_est``.
    """

    name = "EF21-SignMuon"

    def __init__(self, shape, eta, mu=0.8, nesterov=False):
        super().__init__(shape, eta, mu, nesterov)
        self.d_est = np.zeros(self.shape)   # EF21 estimator of the LMO direction

    def step(self, G):
        M_tilde = self._effective_direction(G)
        D = muon_lmo(M_tilde)
        delta = D - self.d_est
        alpha = np.mean(np.abs(delta))
        self.d_est = self.d_est + alpha * np.sign(delta)
        self.X = self.X - self.eta * self.d_est


class EF21MuonUSign(Optimizer):
    """Algorithm ``central_alg_ef``: EF21 on the momentum, LMO after.

    A scaled-sign EF21 estimator ``g_est`` tracks the effective momentum
    ``M_tilde``; the step uses the full Muon LMO of the reconstructed
    estimator: ``D = LMO(g_est);  X <- X - eta * D``.
    """

    name = "EF21-MuonUSign"

    def __init__(self, shape, eta, mu=0.8, nesterov=False):
        super().__init__(shape, eta, mu, nesterov)
        self.g_est = np.zeros(self.shape)   # EF21 estimator of the momentum

    def step(self, G):
        M_tilde = self._effective_direction(G)
        delta = M_tilde - self.g_est
        alpha = np.mean(np.abs(delta))
        self.g_est = self.g_est + alpha * np.sign(delta)
        D = muon_lmo(self.g_est)
        self.X = self.X - self.eta * D


class EF21MuonUDSign(Optimizer):
    """Algorithm ``central_alg_ud``: bidirectional EF21 (uplink + downlink).

    Uplink EF21 reconstructs the momentum estimator ``g_est`` (as in
    EF21-MuonUSign) and the server advances the **exact** model ``X`` with the
    Muon LMO step.  A second EF21-P loop compresses the model shift on the
    downlink into ``W``; the client evaluates its gradient at ``W`` while the
    objective is tracked on the exact server model ``X``.
    """

    name = "EF21-MuonUDSign"

    def __init__(self, shape, eta, mu=0.8, nesterov=False):
        super().__init__(shape, eta, mu, nesterov)
        self.g_est = np.zeros(self.shape)   # uplink EF21 estimator
        self.W = np.zeros(self.shape)       # compressed broadcast model

    def grad_point(self):
        # Gradient is evaluated at the compressed broadcast model.
        return self.W

    def track_point(self):
        # The exact server model carries the "true" progress.
        return self.X

    def step(self, G):
        M_tilde = self._effective_direction(G)
        # --- uplink EF21 ---
        delta_up = M_tilde - self.g_est
        alpha_up = np.mean(np.abs(delta_up))
        self.g_est = self.g_est + alpha_up * np.sign(delta_up)
        # --- exact server step ---
        D = muon_lmo(self.g_est)
        self.X = self.X - self.eta * D
        # --- downlink EF21-P (compress the model shift into W) ---
        delta_dn = self.X - self.W
        alpha_dn = np.mean(np.abs(delta_dn))
        self.W = self.W + alpha_dn * np.sign(delta_dn)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

# Insertion order defines the six paper algorithms first, then the two
# references, which is how the runner iterates and plots them.
OPTIMIZERS = {
    "SignMuon":        SignMuon,
    "EF21-SignMuon":   EF21SignMuon,
    "MuonUSign":       MuonUSign,
    "MuonUDSign":      MuonUDSign,
    "EF21-MuonUSign":  EF21MuonUSign,
    "EF21-MuonUDSign": EF21MuonUDSign,
    "SignSGD":         SignSGD,
    "Muon":            Muon,
}

PAPER_METHODS = ["SignMuon", "EF21-SignMuon", "MuonUSign", "MuonUDSign",
                 "EF21-MuonUSign", "EF21-MuonUDSign"]
REFERENCE_METHODS = ["SignSGD", "Muon"]
