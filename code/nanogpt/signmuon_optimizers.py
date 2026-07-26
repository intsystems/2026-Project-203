"""
Distributed (data-parallel, NOT federated) implementations of the six paper
optimizers plus SignSGD and a reference Muon, packaged for the modded-nanogpt
speedrun.  This build targets **record #40** (2025-10-04, the last record before
NorMuon), whose hidden-matrix optimizer is still a *clean, separable* Muon:

    momentum (Nesterov EMA)  ->  Polar Express orthogonalization (LMO)  ->  step

so the paper's sign/EF21 variants inject exactly at the LMO, unchanged in spirit
from the earlier classic-record port.

    SignMuon         sign AFTER the LMO          X <- X - eta * sign(PE(M))
    EF21-SignMuon    EF21 on the LMO direction   X <- X - eta * d_est,  d_est ~ PE(M)
    MuonUSign        sign BEFORE the LMO         X <- X - eta * PE(sign(M))
    MuonSign         sign BEFORE and AFTER LMO   X <- X - eta * sign(PE(sign(M)))
    EF21-MuonUSign   EF21 on the momentum        X <- X - eta * PE(g_est), g_est ~ M
    EF21-MuonSign    bidirectional EF21          exact X step + sign-compressed broadcast W
    SignSGD          sign of the momentum        X <- X - eta * sign(M)
    Muon             reference (no compression)  X <- X - eta * PE(M)

Every method's learning rate is per-layer scaled so that one ``eta_0`` means the
same thing (the per-step RMS gain) for all eight -- see "Per-layer learning-rate
scaling" below.  For the LMO family that scaling IS record #40's own aspect
factor, so ``Muon`` here is the record verbatim.

Here ``M`` is the (Nesterov) heavy-ball momentum of the *averaged* gradient and
``PE`` is the Muon Polar-Express orthogonalization (approximate polar factor).
The math for every method is verbatim the centralized algorithm boxes of the
paper and their numpy reference in ``code/counterexamples/optimizers.py`` --
only the *systems* layer differs.

-----------------------------------------------------------------------------
Two things that are specific to the record-#40 model
-----------------------------------------------------------------------------
1. Polar Express, not Newton-Schulz.  Record #40 replaced the Newton-Schulz
   iteration (record <=#37) with Polar Express (record #38), a different quintic
   whose ``coeffs_list`` are baked in below.  We reproduce it in **pure torch**
   (no Triton) so it (a) runs identically on A100 and H100 and (b) is unit-
   testable on CPU.  The Triton kernels in the upstream record are a *speed*
   optimization only; the arithmetic here is the same up to bf16 rounding.

2. Merged ``qkvo_w`` attention weight.  Record #40 stores Q/K/V/O in a single
   ``[hdim, 4*dim]`` parameter (so it batches with the ``[dim, 4*dim]`` MLP
   weights for one reduce_scatter) but the model *always* accesses it through
   ``.view(4, hdim, dim)``.  Muon must therefore orthogonalize the FOUR
   ``[hdim, dim]`` sub-matrices independently -- NOT the merged matrix.  We
   detect this via the ``.module == "attn"`` tag the model attaches to the
   parameter and reshape only around the LMO call (every other op is elementwise
   and shape-agnostic).  Same-shaped MLP weights (``.module == "mlp"``) are
   orthogonalized as a single matrix, as the model uses them.

-----------------------------------------------------------------------------
Distributed vs. federated -- the part that actually changes
-----------------------------------------------------------------------------
In the FEDERATED code (``code/federated_algorithms.py``) each client keeps its
own momentum / EF21 estimator, compresses ITS OWN update, ships the compressed
message, and the server aggregates the *compressed* messages (majority vote, or
an average of scaled-signs). Compression sits on the wire between many clients.

In DISTRIBUTED data-parallel training (this file, following the modded-nanogpt
Muon) there is exactly one logical model. The ranks hold different data shards;
``dist.reduce_scatter(..., op=AVG)`` averages their gradients so that -- for each
parameter -- a single owning rank receives the *true global mean gradient*. That
owning rank then runs the ordinary CENTRALIZED update (momentum, LMO, sign, EF21
estimator, ...) and ``dist.all_gather`` puts the updated parameter back on every
rank.

Consequences that make this correct and simple:
  * There is ONE momentum buffer / EF21 estimator per parameter, living on the
    rank that owns that parameter (via ``self.state[p]``) -- never per-rank
    replicas that need reconciling. Parameters are sharded round-robin: rank r
    owns ``params[base_i + r]`` inside every ``world_size``-sized chunk.
  * The 1-bit "compression" here is a property of the update RULE (sign / EF21),
    not of the rank-to-rank transport (gradients cross the wire in full
    precision via reduce_scatter). This is the honest distributed analog of the
    centralized algorithms the paper analyzes -- see the README for the full
    argument.
  * ``EF21-MuonSign`` additionally keeps an *exact* server model ``X`` as
    optimizer state and lets the live parameters be the sign-compressed
    broadcast model ``W``; the gradient is naturally evaluated at ``W`` (the
    forward pass uses the live parameter) exactly as the downlink EF21-P scheme
    requires. ``swap_in_exact()`` / ``swap_out_exact()`` expose ``X`` for
    evaluation.

The reduce_scatter / all_gather transport is world-size agnostic (it runs on 1
GPU just as on 8): on a single process the collectives are no-ops and every
parameter is updated locally with the (grad-accumulated) mean gradient.  That is
exactly what makes the single-A100 build reproduce the 8xH100 optimizer
trajectory -- see train_gpt_a100.py.
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.distributed as dist

__all__ = [
    "polar_express",
    "zeropower_via_newtonschulz5",  # backward-compatible alias for polar_express
    "Muon",
    "SignSGD",
    "SignMuon",
    "MuonUSign",
    "MuonSign",
    "EF21SignMuon",
    "EF21MuonUSign",
    "EF21MuonSign",
    "OPTIMIZERS",
    "PAPER_METHODS",
    "FAMILY_LMO",
    "FAMILY_SIGN",
    "LR_SCALING_RULES",
    "lmo_shape",
    "layer_multiplier",
    "describe_lr_scaling",
]

# ---------------------------------------------------------------------------
# Polar Express orthogonalization (Muon LMO), record #40's ``coeffs_list``.
# Pure torch (no Triton) so it runs on any device and is CPU-testable.
# torch.compile can be disabled with SIGNMUON_NO_COMPILE=1 (the CPU tests do
# this and then monkeypatch this function with an exact-SVD polar factor).
# ---------------------------------------------------------------------------

# Computed for num_iters=5, safety_factor=2e-2, cushion=2  (verbatim from record #40)
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _polar_express_eager(G: Tensor, steps: int = 5) -> Tensor:
    """Polar Express iteration approximating the orthogonal polar factor
    ``U V^T`` of ``G = U S V^T`` (the Muon LMO), record #40's coefficients.

    Runs internally in bfloat16 (as the record does) and returns a tensor in
    ``G``'s original dtype.  Supports a leading batch dimension so the four
    stacked Q/K/V/O blocks of the merged attention weight are orthogonalized in
    a single call.  ``steps`` is accepted for signature compatibility with the
    old Newton-Schulz LMO but ignored -- the number of iterations is fixed by
    ``_POLAR_EXPRESS_COEFFS``.

    Pure-torch equivalent of the upstream Triton kernels; same arithmetic up to
    bf16 rounding, which only perturbs an LMO whose spectrum is deliberately
    non-unit (~Uniform(0.68, 1.13)) anyway.
    """
    assert G.ndim >= 2, "Muon orthogonalization expects at least 2D tensors."
    in_dtype = G.dtype
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    # Ensure spectral norm is at most 1 (record #40's exact normalization).
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(in_dtype)


if os.environ.get("SIGNMUON_NO_COMPILE", "0") == "1":
    polar_express = _polar_express_eager
else:  # pragma: no cover - exercised on the GPU cluster, not in CPU tests
    polar_express = torch.compile(_polar_express_eager)

# Backward-compatible alias: earlier code / tests referred to the LMO as
# ``zeropower_via_newtonschulz5``.  Record #40 uses Polar Express instead, but
# the *interface* (2D-or-batched tensor in, orthogonal factor out) is identical.
zeropower_via_newtonschulz5 = polar_express


# ---------------------------------------------------------------------------
# Per-layer learning-rate scaling (unit gain).
#
# Self-contained mirror of ``code/common/lr_scaling.py`` -- kept duplicated on
# purpose: every logged run prints this file verbatim, so a log must define its
# own learning rates without reference to the rest of the repo.
# ``test_signmuon_optimizers.py`` asserts the two implementations agree.
#
# The criterion (paper appendix "Per-Layer Step Sizes").  The RMS gain of a step
# matrix ``s in R^{m x n}`` acting on an isotropic input is exactly
# ``gamma(s) = ||s||_F / sqrt(m)``.  Requiring the same per-step gain on every
# layer gives one formula, ``lambda = sqrt(fan_out) / ||s||_F``, and the two
# families have exact Frobenius norms:
#
#     lmo  step ``U V^T``:  ||s||_F = sqrt(min(m, n))  ->  lambda = sqrt(max(1, m/n))
#     sign step ``+-1``:    ||s||_F = sqrt(m n)        ->  lambda = 1 / sqrt(n)
#
# The first line is *exactly* the aspect factor shipped in Keller Jordan's Muon
# and used verbatim by record #40 (`max(1, p.size(-2)/p.size(-1))**0.5`), so the
# reference ``Muon`` here is bit-identical to the record; the second line is the
# counterpart the sign family never had (record #40 has no sign family).  Because
# both lines equalize the same quantity, ONE learning rate ``eta_0`` -- the
# per-step RMS gain -- is directly comparable across all eight methods.
#
# Caveat, inherited from record #40: the rule reads ``(m, n) = (fan_out, fan_in)``
# off the stored tensor, and record #40 stores the MLP ``c_fc`` transposed
# (``[dim, hdim]``, used as ``x @ c_fc``) so it can share a reduce_scatter with
# the attention weight.  For ``c_fc`` the semantic fan_out/fan_in are therefore
# swapped, and both families get a 2x smaller multiplier than a
# ``[fan_out, fan_in]`` reading would give.  This is what record #40 itself does
# for Muon, so it is the default here; ``lr_scaling="semantic"`` corrects it
# (using the ``fan_out_sem`` / ``fan_in_sem`` tags the model attaches) as an
# ablation that changes the Muon baseline away from the record.
# ---------------------------------------------------------------------------

FAMILY_LMO = "lmo"      # final step is polar(.):   ||s||_F = sqrt(min(m, n))
FAMILY_SIGN = "sign"    # final step has +-1 entries: ||s||_F = sqrt(m n)


def lmo_shape(p: Tensor) -> tuple[int, int]:
    """``(m, n)`` of the matrix the LMO actually operates on.

    Record #40 stores Q/K/V/O merged as ``[hdim, 4*dim]`` but the model -- and
    therefore the LMO, see :meth:`_DistributedMatrixOptimizer._lmo` -- uses it as
    four ``[hdim, dim]`` blocks, so the per-layer multiplier must be computed on
    the block shape, not the merged one.
    """
    m, n = int(p.shape[-2]), int(p.shape[-1])
    if getattr(p, "module", None) == "attn":
        n //= 4
    return m, n


def semantic_shape(p: Tensor) -> tuple[int, int]:
    """``(fan_out, fan_in)`` of the linear map the parameter implements.

    Falls back to :func:`lmo_shape` for parameters the model did not tag.
    """
    m = getattr(p, "fan_out_sem", None)
    n = getattr(p, "fan_in_sem", None)
    if m is None or n is None:
        return lmo_shape(p)
    return int(m), int(n)


def _aspect(m: int, n: int) -> float:
    return math.sqrt(max(1.0, m / n))


def _unit_gain(family: str, m: int, n: int) -> float:
    return _aspect(m, n) if family == FAMILY_LMO else 1.0 / math.sqrt(n)


def _mup(family: str, m: int, n: int) -> float:
    return _aspect(m, n) if family == FAMILY_LMO else 1.0 / n


def _legacy(family: str, m: int, n: int) -> float:
    return _aspect(m, n) if family == FAMILY_LMO else 1.0


def _no_scaling(family: str, m: int, n: int) -> float:
    return 1.0


#: name -> (shape accessor, multiplier(family, m, n), one-line description)
LR_SCALING_RULES = {
    "unit-gain": (lmo_shape, _unit_gain,
                  "lmo: sqrt(max(1,m/n)) (== record #40 / Keller Jordan);  sign: 1/sqrt(fan_in)"),
    "semantic": (semantic_shape, _unit_gain,
                 "as unit-gain but on the SEMANTIC (fan_out, fan_in); changes Muon vs record #40"),
    "mup": (lmo_shape, _mup,
            "lmo: sqrt(max(1,m/n));  sign: 1/fan_in  (assumes accumulated sign steps align)"),
    "legacy": (lmo_shape, _legacy,
               "lmo: sqrt(max(1,m/n));  sign: 1  (one global rate for sign steps)"),
    "none": (lmo_shape, _no_scaling,
             "lambda = 1 everywhere (what Mishra et al., Algorithm 1 does)"),
}


def layer_multiplier(p: Tensor, family: str, rule: str = "unit-gain") -> float:
    """Per-layer multiplier ``lambda`` for one parameter (``eta_layer = eta_0 * lambda``)."""
    try:
        shape_of, mult, _ = LR_SCALING_RULES[rule]
    except KeyError:
        raise ValueError(
            f"unknown lr_scaling rule {rule!r}; choose from {sorted(LR_SCALING_RULES)}") from None
    m, n = shape_of(p)
    return mult(family, m, n)


def describe_lr_scaling(optimizer: "_DistributedMatrixOptimizer",
                        names: dict | None = None) -> str:
    """Table of the per-layer multipliers in force, for the run log.

    An unlogged per-layer learning rate is an unreproducible one, and the spread
    is the number a reader will want. ``names`` optionally maps ``id(param)`` to
    a human-readable parameter name.
    """
    import re

    names = names or {}
    rule = optimizer.lr_scaling
    family = optimizer.family
    lines = [f"per-layer LR scaling '{rule}' (family={family}): {LR_SCALING_RULES[rule][2]}",
             f"  {'parameter':<32}{'stored':>15}{'lmo m':>8}{'lmo n':>8}{'lambda':>13}{'count':>7}"]
    seen: dict[tuple, int] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            m, n = lmo_shape(p)
            lam = layer_multiplier(p, family, rule)
            # collapse the per-block index so 22 identical MLP matrices are one row
            name = re.sub(r"\.\d+\.", ".*.", names.get(id(p), getattr(p, "module", "?")))
            key = (name, tuple(p.shape), m, n, round(lam, 12))
            seen[key] = seen.get(key, 0) + 1
    mults = []
    for (name, stored, m, n, lam), count in seen.items():
        mults.append(lam)
        lines.append(f"  {name:<32}{str(list(stored)):>15}{m:>8}{n:>8}{lam:>13.6g}{count:>7}")
    if mults:
        lo, hi = min(mults), max(mults)
        lines.append(f"  spread {hi / lo:.2f}x   (min {lo:.6g}, max {hi:.6g})")
        lines.append("  eta_layer = lr * lambda * p.lr_mul; for the lmo family lambda is "
                     "record #40's own aspect factor")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared distributed base: reduce_scatter -> owning-rank update -> all_gather.
# ---------------------------------------------------------------------------


class _DistributedMatrixOptimizer(Optimizer):
    """Base class handling the modded-nanogpt sharded transport.

    Subclasses implement only :meth:`update_param`, the *centralized*
    per-parameter update, and never touch the collectives. All parameters in
    one param-group share a shape so ``reduce_scatter`` / ``all_gather`` (which
    require equal shapes) are valid.
    """

    #: Step family (see the per-layer LR scaling section above). ``FAMILY_LMO``
    #: for methods whose final step is an orthogonal (LMO) direction,
    #: ``FAMILY_SIGN`` for sign-terminated steps whose entries are already ~= 1.
    #: Together with the parameter shape this fixes the per-layer multiplier.
    family = FAMILY_LMO

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.95,
                 lr_scaling="unit-gain", **extra):
        params = list(params)
        if lr_scaling not in LR_SCALING_RULES:
            raise ValueError(f"unknown lr_scaling {lr_scaling!r}; "
                             f"choose from {sorted(LR_SCALING_RULES)}")
        self.lr_scaling = lr_scaling
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        defaults.update(extra)
        # one param-group per unique shape (equal-shape lists for the collectives)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sorted(sizes):  # sorted => identical group order on every rank
            param_groups.append(dict(params=[p for p in params if p.shape == size]))
        super().__init__(param_groups, defaults)
        # lambda is a pure function of (rule, family, shape); cache it so the
        # per-step cost is a dict lookup rather than a sqrt per parameter.
        self._lambda = {id(p): layer_multiplier(p, self.family, lr_scaling)
                        for g in self.param_groups for p in g["params"]}

    # ---- centralized math helpers (single parameter, no collectives) -------

    @staticmethod
    def _is_attn(p: Tensor) -> bool:
        """Record #40 tags the merged Q/K/V/O attention weight with
        ``p.module == "attn"``; such a ``[hdim, 4*dim]`` parameter must be
        orthogonalized as four independent ``[hdim, dim]`` blocks."""
        return getattr(p, "module", None) == "attn"

    def _lmo(self, m: Tensor, p: Tensor) -> Tensor:
        """Apply the Muon LMO (Polar Express) to ``m``.  For the merged
        attention weight, reshape ``[hdim, 4*dim] -> [4, hdim, dim]`` so the
        four heads' Q/K/V/O sub-matrices are orthogonalized independently (the
        model uses exactly this ``.view(4, hdim, dim)``), then reshape back.
        Every other operation in the update rules is elementwise, so the reshape
        is localized to just this call."""
        if self._is_attn(p):
            h, w = m.shape[-2], m.shape[-1]
            d = polar_express(m.reshape(4, h, w // 4))
            return d.reshape(m.shape)
        return polar_express(m)

    def lambda_of(self, p: Tensor) -> float:
        """Per-layer multiplier for ``p`` (see the LR-scaling section above)."""
        lam = self._lambda.get(id(p))
        if lam is None:  # parameter added after construction (not used by the scripts)
            lam = layer_multiplier(p, self.family, self.lr_scaling)
            self._lambda[id(p)] = lam
        return lam

    def _effective_grad(self, grad: Tensor, group: dict, state: dict) -> Tensor:
        """Nesterov heavy-ball momentum, identical to the upstream Muon:

            buf  <- momentum * buf + (1 - momentum) * grad      (EMA buffer)
            m    <- (1 - momentum) * grad + momentum * buf      (look-ahead)

        Returns ``m`` (aliases ``grad`` after an in-place lerp; callers must not
        mutate it in place).
        """
        momentum = group["momentum"]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - momentum)
        return grad.lerp_(buf, momentum)

    def _eff_lr(self, p: Tensor, group: dict) -> float:
        # eta_layer = eta_0 * lambda(family, shape) * p.lr_mul.  For the LMO
        # family lambda is Muon's shipped aspect factor, so `Muon` here steps
        # exactly as record #40 does; the sign family gets its 1/sqrt(fan_in)
        # counterpart, which makes eta_0 mean the same thing for all eight
        # methods (the per-step RMS gain).
        return group["lr"] * self.lambda_of(p) * getattr(p, "lr_mul", 1.0)

    def _decoupled_weight_decay(self, target: Tensor, p: Tensor, group: dict) -> None:
        """AdamW-style decoupled decay applied to ``target`` (usually ``p``, but
        the *exact* model ``X`` for EF21-MuonSign)."""
        eff_wd = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
        if eff_wd != 0:
            target.mul_(1 - eff_wd)

    def update_param(self, p: Tensor, group: dict) -> None:  # pragma: no cover
        """In-place centralized update of one owned parameter ``p`` (whose
        ``.grad`` already holds the global mean gradient). Override in
        subclasses."""
        raise NotImplementedError

    # ---- distributed transport (identical to upstream Muon) ----------------

    @torch.no_grad()
    def step(self):
        # Efficient sharded implementation from the modded-nanogpt record
        # (@YouJiacheng, @KonstantinWilleke, @alexrgilbert, @adricarda,
        # @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal). Each rank owns
        # params[base_i + rank] within every world_size chunk.  On a single
        # process (world_size == 1) the collectives are no-ops.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []

        # Phase 1: average each parameter's gradient onto its owning rank.
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            assert all(p.grad is not None for p in params), (
                f"{type(self).__name__}.step(): some parameters have no .grad; "
                "every parameter in a group must take part in the collective")
            grad_pad = [p.grad for p in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                else:
                    # This rank owns nothing in the (padded) final chunk, but it
                    # must still take part in the collective. It needs a FRESH
                    # scratch buffer: reusing the previous chunk's `grad` would
                    # alias a parameter this rank *does* own and overwrite that
                    # parameter's freshly averaged gradient with padding zeros
                    # before phase 2 ever reads it.
                    grad = torch.empty_like(params[-1])
                reduce_scatter_futures.append(
                    dist.reduce_scatter(
                        grad, grad_pad[base_i:base_i + world_size],
                        op=dist.ReduceOp.AVG, async_op=True,
                    ).get_future()
                )

        # Phase 2: owning rank runs the centralized update, then broadcast back.
        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    self.update_param(params[base_i + rank], group)
                idx += 1
                all_gather_futures.append(
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank], async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(all_gather_futures).wait()


# ---------------------------------------------------------------------------
# Reference methods
# ---------------------------------------------------------------------------


class Muon(_DistributedMatrixOptimizer):
    """Reference full-precision Muon: ``X <- X - eta * PE(M)``."""

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p.grad, group, state)
        v = self._lmo(m, p)
        p.add_(v, alpha=-self._eff_lr(p, group))


class SignSGD(_DistributedMatrixOptimizer):
    """SignSGD with momentum: ``X <- X - eta * sign(M)`` (no LMO)."""

    family = FAMILY_SIGN

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p.grad, group, state)
        p.add_(m.sign(), alpha=-self._eff_lr(p, group))


# ---------------------------------------------------------------------------
# Sign-around-the-LMO methods (no error feedback)
# ---------------------------------------------------------------------------


class SignMuon(_DistributedMatrixOptimizer):
    """SignMuon -- sign AFTER the LMO: ``D = PE(M);  X <- X - eta * sign(D)``.

    The Theorem-1 divergence counterexample lives here: sign-compressing the LMO
    direction can destroy the descent property.
    """

    family = FAMILY_SIGN

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p.grad, group, state)
        d = self._lmo(m, p)
        p.add_(d.sign(), alpha=-self._eff_lr(p, group))


class MuonUSign(_DistributedMatrixOptimizer):
    """MuonUSign -- sign BEFORE the LMO: ``s = sign(M);  X <- X - eta * PE(s)``.

    The LMO is positively homogeneous of degree zero, so compressing with the
    scaled sign ``mean|M| * sign(M)`` instead of the bare ``sign(M)`` gives the
    identical direction: the "U" (unscaled) and scaled variants of the *uplink*
    compressor coincide here.  This is NOT ``MuonSign``, which signs the LMO
    output as well (see below).
    """

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p.grad, group, state)
        d = self._lmo(m.sign(), p)
        p.add_(d, alpha=-self._eff_lr(p, group))


class MuonSign(_DistributedMatrixOptimizer):
    """MuonSign -- sign BEFORE *and* AFTER the LMO:
    ``s = sign(M);  D = PE(s);  X <- X - eta * sign(D)``.
    """

    family = FAMILY_SIGN

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p.grad, group, state)
        d = self._lmo(m.sign(), p)
        p.add_(d.sign(), alpha=-self._eff_lr(p, group))


# ---------------------------------------------------------------------------
# Error-feedback (EF21) methods
# ---------------------------------------------------------------------------


class EF21SignMuon(_DistributedMatrixOptimizer):
    """EF21-SignMuon -- EF21 error feedback on the LMO DIRECTION.

        D      = PE(M)
        delta  = D - d_est
        d_est <- d_est + mean|delta| * sign(delta)     (scaled-sign compressor)
        X     <- X - eta * d_est

    Tracks the (discontinuous) LMO direction with a contractive 1-bit estimator.
    """

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p.grad, group, state)
        d = self._lmo(m, p)
        if "d_est" not in state:
            state["d_est"] = torch.zeros_like(p)
        d_est = state["d_est"]
        delta = d - d_est
        alpha = delta.abs().mean()
        d_est.add_(delta.sign() * alpha)        # d_est += mean|delta| * sign(delta)
        p.add_(d_est, alpha=-self._eff_lr(p, group))


class EF21MuonUSign(_DistributedMatrixOptimizer):
    """EF21-MuonUSign -- EF21 error feedback on the MOMENTUM, full LMO after.

        delta  = M - g_est
        g_est <- g_est + mean|delta| * sign(delta)     (uplink, scaled-sign)
        D      = PE(g_est)
        X     <- X - eta * D

    Applying the LMO to the (asymptotically exact) gradient estimator rather than
    to a sign is what restores convergence on the Theorem-1 counterexample.
    """

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        self._decoupled_weight_decay(p, p, group)
        m = self._effective_grad(p.grad, group, state)
        if "g_est" not in state:
            state["g_est"] = torch.zeros_like(p)
        g_est = state["g_est"]
        delta = m - g_est
        alpha = delta.abs().mean()
        g_est.add_(delta.sign() * alpha)        # g_est += mean|delta| * sign(delta)
        d = self._lmo(g_est, p)
        p.add_(d, alpha=-self._eff_lr(p, group))


class EF21MuonSign(_DistributedMatrixOptimizer):
    """EF21-MuonSign -- bidirectional EF21 (uplink gradient + downlink model).

    Uplink (as EF21-MuonUSign) reconstructs the gradient estimator ``g_est`` and
    the server advances an EXACT model ``X`` (kept as optimizer state) with the
    Muon LMO step. A second EF21-P loop compresses the model increment onto the
    live broadcast model ``W`` (= the parameter tensor), so the next gradient is
    evaluated at ``W``:

        delta_up  = M - g_est
        g_est    <- g_est + mean|delta_up| * sign(delta_up)      (uplink)
        X        <- X - eta * PE(g_est)                          (exact step)
        delta_dn  = X - W
        W        <- W + mean|delta_dn| * sign(delta_dn)          (downlink EF21-P)

    Only ``W`` (one bit + one scalar per entry per round, conceptually) is ever
    broadcast; ``X`` never leaves its owning rank except via
    :meth:`swap_in_exact` for evaluation. Weight decay acts on the exact model.
    """

    family = FAMILY_LMO

    def update_param(self, p, group):
        state = self.state[p]
        if "exact_model" not in state:
            state["exact_model"] = p.detach().clone()   # X_0 = W_0 (broadcast model)
        X = state["exact_model"]
        self._decoupled_weight_decay(X, p, group)       # decay the exact server model

        m = self._effective_grad(p.grad, group, state)  # gradient was taken at W = p
        if "g_est" not in state:
            state["g_est"] = torch.zeros_like(p)
        g_est = state["g_est"]
        delta_up = m - g_est
        alpha_up = delta_up.abs().mean()
        g_est.add_(delta_up.sign() * alpha_up)          # uplink EF21 on the gradient

        d = self._lmo(g_est, p)
        X.add_(d, alpha=-self._eff_lr(p, group))        # exact server step: X <- X - eta * PE(g_est)

        delta_dn = X - p                                # downlink EF21-P on the model increment
        alpha_dn = delta_dn.abs().mean()
        p.add_(delta_dn.sign() * alpha_dn)              # W <- W + mean|X - W| * sign(X - W)

    # -- expose the exact model X for evaluation -----------------------------

    @torch.no_grad()
    def swap_in_exact(self):
        """Broadcast the exact server model ``X`` into the live parameters (for
        eval / checkpointing), stashing the compressed model ``W``. Uses the same
        sharded all-gather as :meth:`step`."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self._w_backup: dict[int, tuple[Tensor, Tensor]] = {}
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    st = self.state[p]
                    self._w_backup[id(p)] = (p, p.detach().clone())
                    if "exact_model" in st:
                        p.copy_(st["exact_model"])
                futures.append(
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank], async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(futures).wait()

    @torch.no_grad()
    def swap_out_exact(self):
        """Restore the compressed broadcast model ``W`` saved by
        :meth:`swap_in_exact`."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backup = getattr(self, "_w_backup", {})
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if id(p) in backup:
                        p.copy_(backup[id(p)][1])
                futures.append(
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank], async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(futures).wait()
        self._w_backup = {}


# ---------------------------------------------------------------------------
# Registry: paper name -> class. Used by the train scripts' --optimizer knob.
# ---------------------------------------------------------------------------

OPTIMIZERS = {
    "SignMuon":        SignMuon,
    "EF21-SignMuon":   EF21SignMuon,
    "MuonUSign":       MuonUSign,
    "MuonSign":        MuonSign,
    "EF21-MuonUSign":  EF21MuonUSign,
    "EF21-MuonSign":   EF21MuonSign,
    "SignSGD":         SignSGD,
    "Muon":            Muon,
}

#: the six methods introduced in the paper (the two above are references)
PAPER_METHODS = [
    "SignMuon", "EF21-SignMuon", "MuonUSign",
    "MuonSign", "EF21-MuonUSign", "EF21-MuonSign",
]
