"""Centralized (single-node) implementations of the eight methods in the paper
"SignMuon, MuonSign, and the Role of Error Feedback".

Naming follows the paper exactly:

===================  =========================================================
Method               Matrix-parameter update ``d_t`` (step is ``X -= eta*d_t``)
===================  =========================================================
``Muon``             ``polar(M)``                       (full precision)
``SignSGD``          ``sign(M)``                        (reference)
``SignMuon``         ``sign(polar(M))``                 sign AFTER  the LMO
``MuonUSign``        ``polar(sign(M))``                 sign BEFORE the LMO
``MuonSign``         ``sign(polar(sign(M)))``           sign on BOTH sides
``EF21SignMuon``     EF21 on ``polar(M)``               (diverges, Thm 4)
``EF21MuonUSign``    ``polar(g_est)``, ``g_est ~ M``    uplink EF21
``EF21MuonSign``     as above + downlink EF21-P on X    bidirectional EF21
===================  =========================================================

Here ``M`` is the effective momentum direction ``M_tilde`` and
``polar(Y) = -A(Y) = U V^T`` is the Muon LMO direction.

Momentum uses the EMA form of the paper's algorithm boxes,
``M_t = mu*M_{t-1} + (1-mu)*G_t``, with the Nesterov look-ahead
``M_tilde = (1-mu)*G_t + mu*M_t``. This is *exactly* equivalent to the
heavy-ball form ``mu*M_{t-1} + G_t`` used in the paper's main text -- the two
buffers differ by the constant factor ``(1-mu)`` and every method above is
positively homogeneous in ``M`` (``sign``, ``polar`` and the EF21 recursion all
commute with multiplication by a positive scalar), so the iterate trajectory is
identical. The EMA form is used here only to match the pseudocode boxes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from common.lr_scaling import FAMILY_LMO, FAMILY_SIGN

__all__ = [
    "zeropower_via_newtonschulz5",
    "muon_lmo",
    "muon_orthogonalized_update",
    "scaled_sign",
    "Muon",
    "SignSGD",
    "SignMuon",
    "MuonUSign",
    "MuonSign",
    "EF21SignMuon",
    "EF21MuonUSign",
    "EF21MuonSign",
    "OPTIMIZERS",
    "FAMILY_LMO",
    "FAMILY_SIGN",
]

# 5th-order Newton-Schulz coefficients of Algorithm 1 (Y <- a*Y - b*A*Y + c*A^2*Y).
NS_COEFFS = (3.4445, 4.7750, 2.0315)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def zeropower_via_newtonschulz5(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Tensor:
    """Approximate the orthogonal polar factor ``U V^T`` of ``G`` (Algorithm 1).

    ``dtype`` is the working precision of the iteration. The default
    ``bfloat16`` matches the reference Muon implementation and is what the
    reported experiments used; pass ``dtype=None`` to iterate in the input dtype
    (e.g. float32), which matters whenever the *sign pattern* of the result is
    the quantity of interest -- bfloat16 carries ~3 decimal digits, so entries of
    ``polar(G)`` close to zero can flip sign.

    Supports tensors with ``ndim >= 2``; batched leading dimensions are allowed.
    """
    assert G.ndim >= 2, "Muon orthogonalization expects at least 2D tensors."

    a, b, c = NS_COEFFS

    # ``.to()`` is a no-op returning ``G`` itself when the dtype already matches,
    # so clone before the in-place normalization below or we would scribble on
    # the caller's gradient buffer.
    X = G.to(dtype=dtype) if dtype is not None else G
    if X is G:
        X = X.clone()

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        A = X @ X.mT
        B = -b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(device=G.device, dtype=G.dtype)


def muon_lmo(
    grad: Tensor,
    ns_steps: int = 5,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    scale_aspect: bool = True,
) -> Tensor:
    """Muon LMO direction ``polar(grad) = -A(grad)`` for a parameter tensor.

    * 4D convolution filters are flattened to ``[out_channels, -1]``, orthogonalized,
      and reshaped back.
    * 2D matrices are used directly.
    * ``ndim < 2`` tensors are returned unchanged (no matrix structure to exploit).

    ``scale_aspect`` applies Muon's aspect-ratio factor ``sqrt(max(1, m/n))``.
    It is a practical heuristic, not part of the LMO itself: it rescales the step
    and is therefore invisible to any method that signs the LMO *output*
    (``SignMuon``, ``MuonSign``), but it does change the step of ``Muon``,
    ``MuonUSign`` and ``EF21MuonUSign``. Set ``False`` for the exact
    ``U V^T`` of the theory.
    """
    if grad.ndim < 2:
        return grad

    orig_shape = grad.shape if grad.ndim > 2 else None
    G = grad.reshape(grad.shape[0], -1) if orig_shape is not None else grad

    orth = zeropower_via_newtonschulz5(G, steps=ns_steps, dtype=dtype)
    if scale_aspect:
        orth = orth * max(1.0, orth.size(-2) / orth.size(-1)) ** 0.5

    return orth.reshape(orig_shape) if orig_shape is not None else orth


# Backwards-compatible alias (the pre-refactor name).
muon_orthogonalized_update = muon_lmo


def scaled_sign(Y: Tensor) -> Tensor:
    """Contractive 1-bit compressor ``mean|Y| * sign(Y)`` (the USign operator).

    One bit per entry plus a single shared magnitude scalar per tensor.
    """
    return Y.abs().mean() * torch.sign(Y)


# --------------------------------------------------------------------------
# Shared optimizer skeleton
# --------------------------------------------------------------------------


class _BaseMethod(Optimizer):
    """Momentum / weight-decay bookkeeping shared by every method.

    Subclasses implement :meth:`_direction`, which maps the effective momentum
    direction ``m_tilde`` to the step direction ``d_t`` (the update is always
    ``p <- p - lr * lambda_mult * d_t``), and may override :meth:`_post_step`.

    Weight decay
    ------------
    ``decoupled_weight_decay=False`` (the default) folds ``wd * p`` into the
    gradient *before* the momentum buffer, which is what the reported
    experiments used. ``True`` instead shrinks the parameter multiplicatively
    (``p *= 1 - lr*wd``) and leaves the gradient -- and hence the LMO geometry --
    untouched; this is the AdamW/Muon convention and what the federated driver
    uses.
    """

    method_name = "base"
    #: Step family, see ``common.lr_scaling``. Together with the parameter shape
    #: this fixes the per-layer learning-rate multiplier.
    family = FAMILY_LMO

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        lambda_mult: float = 1.0,
        ns_steps: int = 5,
        lmo_dtype: Optional[torch.dtype] = torch.bfloat16,
        scale_aspect: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must lie in [0, 1), got {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and momentum <= 0.0:
            raise ValueError("Nesterov momentum requires a positive momentum")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            lambda_mult=lambda_mult,
            ns_steps=ns_steps,
            lmo_dtype=lmo_dtype,
            scale_aspect=scale_aspect,
        )
        super().__init__(params, defaults)

    # -- hooks ------------------------------------------------------------
    def _direction(self, m_tilde: Tensor, state: dict, group: dict) -> Tensor:
        raise NotImplementedError

    def _post_step(self, p, state: dict, group: dict) -> None:
        """Called after ``p`` has been updated (used by the downlink EF loop)."""

    def _lmo(self, Y: Tensor, group: dict) -> Tensor:
        return muon_lmo(
            Y,
            ns_steps=group["ns_steps"],
            dtype=group["lmo_dtype"],
            scale_aspect=group["scale_aspect"],
        )

    # -- driver -----------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            decoupled = group["decoupled_weight_decay"]
            lambda_mult = group["lambda_mult"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(f"{type(self).__name__} does not support sparse gradients")

                state = self.state[p]
                g = p.grad

                # The bidirectional method differentiates the model the gradient
                # was taken at (W = p.data) from the exact model X it advances.
                ref = self._weight_decay_target(p, state)
                if wd != 0 and not decoupled:
                    g = g.add(ref, alpha=wd)

                # 1) momentum (EMA form of the paper's algorithm boxes)
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g, alpha=1.0 - mu)
                m_tilde = g.mul(1.0 - mu).add_(buf, alpha=mu) if nesterov else buf

                # 2) method-specific step direction
                d_t = self._direction(m_tilde, state, group)

                # 3) decoupled weight decay + parameter step
                target = self._step_target(p, state)
                if wd != 0 and decoupled:
                    target.mul_(1.0 - lr * wd)
                target.add_(d_t, alpha=-lr * lambda_mult)

                self._post_step(p, state, group)

        return loss

    # -- overridable notion of "the model" --------------------------------
    def _weight_decay_target(self, p, state: dict) -> Tensor:
        return p.data

    def _step_target(self, p, state: dict) -> Tensor:
        return p.data


class _EF21Mixin:
    """Scaled-sign EF21 estimator kept in the per-parameter state.

    ``g_{t} = g_{t-1} + alpha_t * sign(Delta_t)`` with
    ``Delta_t = target_t - g_{t-1}`` and ``alpha_t = mean|Delta_t|``: exactly the
    compressor of Equation (16), applied to whichever ``target`` the method
    tracks (the momentum for the MuonUSign family, the LMO output for
    EF21-SignMuon).
    """

    @staticmethod
    def _ef21_update(target: Tensor, state: dict, key: str) -> Tensor:
        if key not in state:
            state[key] = torch.zeros_like(target)
        est = state[key]
        delta = target - est
        est.add_(delta.abs().mean() * torch.sign(delta))
        return est


# --------------------------------------------------------------------------
# Reference methods
# --------------------------------------------------------------------------


class Muon(_BaseMethod):
    """Full-precision Muon: ``d_t = polar(M_tilde)``."""

    method_name = "muon"
    family = FAMILY_LMO

    def _direction(self, m_tilde, state, group):
        return self._lmo(m_tilde, group)


class SignSGD(_BaseMethod):
    """SignSGD with momentum: ``d_t = sign(M_tilde)``.

    Applies to tensors of any shape (no matrix structure is used).
    """

    method_name = "signsgd"
    family = FAMILY_SIGN

    def _direction(self, m_tilde, state, group):
        return torch.sign(m_tilde)


# --------------------------------------------------------------------------
# Sign around the LMO (Theorems 1-3: all three can diverge)
# --------------------------------------------------------------------------


class SignMuon(_BaseMethod):
    """SignMuon -- sign AFTER the LMO: ``d_t = sign(polar(M_tilde))`` (Thm 1)."""

    method_name = "signmuon"
    family = FAMILY_SIGN

    def _direction(self, m_tilde, state, group):
        return torch.sign(self._lmo(m_tilde, group))


class MuonUSign(_BaseMethod):
    """MuonUSign -- sign BEFORE the LMO: ``d_t = polar(sign(M_tilde))`` (Thm 2).

    The uplink carries one bit per parameter; the LMO runs on the (uncompressed)
    receiving side, so the step itself is full precision.
    """

    method_name = "muonusign"
    family = FAMILY_LMO

    def _direction(self, m_tilde, state, group):
        return self._lmo(torch.sign(m_tilde), group)


class MuonSign(_BaseMethod):
    """MuonSign -- sign on BOTH sides: ``d_t = sign(polar(sign(M_tilde)))`` (Thm 3).

    Both the uplink and the downlink cost one bit per parameter.
    """

    method_name = "muonsign"
    family = FAMILY_SIGN

    def _direction(self, m_tilde, state, group):
        return torch.sign(self._lmo(torch.sign(m_tilde), group))


# --------------------------------------------------------------------------
# Error-feedback methods
# --------------------------------------------------------------------------


class EF21SignMuon(_BaseMethod, _EF21Mixin):
    """EF21 on the LMO *output* -- the method that still diverges (Theorem 4).

    ``D_t = polar(M_tilde)`` is tracked by a scaled-sign EF21 estimator and the
    step is ``d_t = d_est``. Included because the paper needs it as the
    counterexample: a single shared magnitude ``alpha_t`` rescales all entries at
    once, so a large sign-flipping off-diagonal pins ``alpha_t`` and the
    estimate of a small target locks onto the wrong sign.
    """

    method_name = "ef21signmuon"
    family = FAMILY_LMO

    def _direction(self, m_tilde, state, group):
        return self._ef21_update(self._lmo(m_tilde, group), state, "dir_estimator")


class EF21MuonUSign(_BaseMethod, _EF21Mixin):
    """EF21-MuonUSign -- EF21 on the momentum, full Muon LMO after it.

    Single-node reduction of EF21-Muon (Gruntkowska et al., 2025) with the
    scaled-sign compressor on the uplink and the identity on the downlink.
    Compressing *before* the oracle is what restores the descent property:
    ``g_est -> M_tilde`` and hence ``d_t -> polar(M_tilde)``.
    """

    method_name = "ef21muonusign"
    family = FAMILY_LMO

    def _direction(self, m_tilde, state, group):
        g_est = self._ef21_update(m_tilde, state, "grad_estimator")
        return self._lmo(g_est, group)


class EF21MuonSign(_BaseMethod, _EF21Mixin):
    """EF21-MuonSign -- bidirectional EF21 (scaled sign on both channels).

    Two error-feedback loops:

    * uplink, on the gradient residual: ``g_est += mean|M-g_est| * sign(M-g_est)``
    * downlink, on the model increment: ``W += mean|X-W| * sign(X-W)``

    ``p.data`` holds the *broadcast* model ``W`` -- so ``p.grad`` is the gradient
    at ``W``, as the algorithm requires -- while the exact server model ``X``
    lives in the ``exact_model`` state buffer. Use :meth:`using_exact` (or
    :meth:`restore_exact`) to evaluate on ``X``, which is the iterate the
    convergence theory bounds.
    """

    method_name = "ef21muonsign"
    family = FAMILY_LMO

    def _weight_decay_target(self, p, state):
        return self._exact(p)

    def _step_target(self, p, state):
        return self._exact(p)

    def _exact(self, p) -> Tensor:
        state = self.state[p]
        if "exact_model" not in state:
            state["exact_model"] = p.data.clone()
        return state["exact_model"]

    def _direction(self, m_tilde, state, group):
        g_est = self._ef21_update(m_tilde, state, "grad_estimator")
        return self._lmo(g_est, group)

    def _post_step(self, p, state, group):
        # Downlink EF21-P: broadcast a scaled sign of the model shift X - W.
        delta = state["exact_model"] - p.data
        p.data.add_(delta.abs().mean() * torch.sign(delta))

    # -- exposing the exact model ----------------------------------------
    @torch.no_grad()
    def restore_exact(self, params=None) -> None:
        """Copy the exact server model ``X`` back into ``p.data``, permanently."""
        groups = self.param_groups if params is None else [{"params": params}]
        for group in groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "exact_model" in st:
                    p.data.copy_(st["exact_model"])

    @contextmanager
    def using_exact(self):
        """Temporarily expose ``X`` in ``p.data``, restoring ``W`` afterwards.

        Wrap evaluation in this so metrics are computed on the exact model while
        the invariant "``p.grad`` is the gradient at ``W``" survives for the next
        training step.
        """
        saved = {}
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "exact_model" in st:
                    saved[p] = p.data.clone()
                    p.data.copy_(st["exact_model"])
        try:
            yield
        finally:
            for p, w in saved.items():
                p.data.copy_(w)


# --------------------------------------------------------------------------
# Registry + deprecated aliases
# --------------------------------------------------------------------------

OPTIMIZERS = {
    "signmuon": SignMuon,
    "ef21signmuon": EF21SignMuon,
    "muonusign": MuonUSign,
    "muonsign": MuonSign,
    "ef21muonusign": EF21MuonUSign,
    "ef21muonsign": EF21MuonSign,
    "muon": Muon,
    "signsgd": SignSGD,
}

PAPER_METHODS = ["signmuon", "ef21signmuon", "muonusign", "muonsign",
                 "ef21muonusign", "ef21muonsign"]
REFERENCE_METHODS = ["muon", "signsgd"]

# Pre-refactor class names. ``MuonSign`` used to denote the sign-BEFORE method,
# which the paper now calls MuonUSign -- the alias below is deliberately NOT
# provided for that name, because silently resolving it to the new (both-sides)
# ``MuonSign`` would change the algorithm rather than just the label.
EF_USignMuon = EF21MuonUSign
EF_UDSignMuon = EF21MuonSign
