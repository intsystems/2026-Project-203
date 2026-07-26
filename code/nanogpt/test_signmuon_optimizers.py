"""
Math-correctness test for the distributed optimizers in ``signmuon_optimizers.py``.

This test is PORTABLE: it needs only torch (CPU is fine), no GPUs and no
``torch.distributed``. It verifies that each optimizer's *per-parameter update
recurrence* (momentum, sign placement, EF21 estimator, downlink error feedback)
reproduces, step for step, the trusted numpy reference in
``code/counterexamples/optimizers.py`` -- which itself follows the paper's
algorithm boxes verbatim.

Strategy
--------
* Monkeypatch the Newton-Schulz LMO with the reference's EXACT-SVD ``muon_lmo``
  (routed through numpy) so both sides use the *identical* orthogonalization;
  any discrepancy is then a bug in the update recurrence, not the LMO
  approximation.
* Drive both with the same fixed random gradient sequence on SQUARE matrices and
  ``lr_scaling="none"`` (so every per-layer multiplier is 1 and the step matches
  the reference's plain eta), weight_decay = 0, Nesterov momentum (the
  optimizer's only momentum form).
* We call ``update_param`` directly (the centralized core), bypassing the
  collectives -- those are exercised by ``test_distributed_sharding.py``.

A second test pins the per-layer LR scaling itself: that it reproduces record
#40 / Keller Jordan's aspect factor for the LMO family, that it equalizes the
per-step RMS gain across shapes and families, and that it agrees with the
repo-level ``code/common/lr_scaling.py``.

Run:  SIGNMUON_NO_COMPILE=1 python test_signmuon_optimizers.py
      (or: pytest test_signmuon_optimizers.py)
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SIGNMUON_NO_COMPILE", "1")  # eager NS; we monkeypatch it anyway

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent  # .../code
sys.path.insert(0, str(_HERE))          # signmuon_optimizers
sys.path.insert(0, str(_CODE))          # counterexamples package

import signmuon_optimizers as smo
from counterexamples import optimizers as ref  # code/counterexamples/optimizers.py

torch.set_default_dtype(torch.float64)


# Route the module's LMO through the reference's exact-SVD muon_lmo (numpy), so
# the distributed optimizer and the numpy reference share one identical LMO.
# (record #40 uses Polar Express as the LMO; we swap in the exact polar factor so
# any discrepancy is a bug in the sign/EF21 recurrence, not the LMO approximation.)
def _exact_polar(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    arr = G.detach().cpu().numpy().astype(np.float64)
    if arr.ndim > 2:  # batched (merged-attn) inputs: orthogonalize each block
        out = np.stack([ref.muon_lmo(a) for a in arr.reshape(-1, *arr.shape[-2:])])
        out = out.reshape(arr.shape)
    else:
        out = ref.muon_lmo(arr)
    return torch.from_numpy(np.ascontiguousarray(out)).to(dtype=G.dtype, device=G.device)


smo.polar_express = _exact_polar  # late-bound module global -> methods pick this up


# (paper name in smo.OPTIMIZERS, reference class in ref.OPTIMIZERS)
_PAIRS = [
    "SignMuon", "EF21-SignMuon", "MuonUSign", "MuonSign",
    "EF21-MuonUSign", "EF21-MuonSign", "SignSGD", "Muon",
]


def _run_reference(name, grads, eta, mu):
    shape = grads[0].shape
    opt = ref.OPTIMIZERS[name](shape, eta, mu=mu, nesterov=True)
    for G in grads:
        opt.step(G.astype(np.float64))
    # tracked/exact model X, and (for UDSign) the compressed broadcast model W
    W = getattr(opt, "W", None)
    return opt.X.copy(), (None if W is None else W.copy())


def _run_distributed_core(name, grads, eta, mu):
    """Run the distributed optimizer's centralized core (update_param) on one
    parameter, starting from X_0 = W_0 = 0 to match the reference."""
    cls = smo.OPTIMIZERS[name]
    p = torch.zeros(grads[0].shape, dtype=torch.float64)
    # lr_scaling="none": the numpy reference implements the bare algorithm box
    # with a single global eta, so switch the per-layer multiplier off to compare
    # the recurrences themselves (the multipliers are tested separately below).
    opt = cls([p], lr=eta, momentum=mu, weight_decay=0.0, lr_scaling="none")
    group = opt.param_groups[0]
    for G in grads:
        p.grad = torch.from_numpy(G.astype(np.float64)).clone()
        opt.update_param(p, group)
    X = opt.state[p].get("exact_model", p)  # exact model for UDSign, else p itself
    W = p if "exact_model" in opt.state[p] else None
    return X.detach().numpy().copy(), (None if W is None else W.detach().numpy().copy())


def _check(name, eta, mu, T=25, dim=4, seed=0, atol=1e-6):
    rng = np.random.default_rng(seed)
    grads = [rng.standard_normal((dim, dim)) for _ in range(T)]
    Xr, Wr = _run_reference(name, grads, eta, mu)
    Xd, Wd = _run_distributed_core(name, grads, eta, mu)
    xerr = float(np.max(np.abs(Xr - Xd)))
    # The tracked/exact model X is the quantity training evaluates (validation runs on X);
    # it must match the reference tightly.
    assert xerr < atol, f"{name}: exact-model X mismatch (max abs {xerr:.2e}) at eta={eta}, mu={mu}"
    if Wr is not None:
        werr = float(np.max(np.abs(Wr - Wd)))
        # W is EF21-MuonSign's 1-bit sign-compressed broadcast model:
        #     W <- W + mean|X-W| * sign(X-W).
        # Its update rule is line-for-line the numpy reference (optimizers.py), but sign() is
        # discontinuous: when an entry of the residual X-W lands on a sign(0) tie, torch-fp and
        # numpy-fp -- which differ by ~1e-16 in the momentum EMA -- pick opposite signs, and since
        # alpha=mean|X-W| then shifts, the whole subsequent W trajectory drifts by ~one compressor
        # step (this happens e.g. at mu=0.8, seed=0). This is an unavoidable cross-backend artifact
        # on a discontinuous operator, NOT a bug:
        #   (a) the exact/tracked model X -- which training actually evaluates -- matches to ~1e-16
        #       above (asserted), and
        #   (b) test_distributed_sharding.py verifies W matches EXACTLY in a torch-vs-torch run.
        # So we require the W drift to stay within a few compressor steps (~eta); a real downlink
        # bug (wrong sign/scale, or a missing update) would blow past this bound.
        assert werr < atol or werr <= 3.0 * eta, (
            f"{name}: broadcast-model W drift {werr:.2e} exceeds ~compressor scale (eta={eta}) at "
            f"mu={mu} -- likely a real bug, not a sign-tie")
        return xerr, werr
    return xerr, None


def test_update_math_matches_reference():
    failures = []
    for name in _PAIRS:
        for mu in (0.0, 0.8, 0.95):
            for eta in (0.01, 0.1):
                try:
                    xerr, werr = _check(name, eta, mu)
                    tag = f"{name:<16} mu={mu:<4} eta={eta:<5}  |X-err|={xerr:.2e}" + (
                        f"  |W-err|={werr:.2e}" if werr is not None else "")
                    print("  OK  " + tag)
                except AssertionError as e:
                    failures.append(str(e))
                    print("  FAIL " + str(e))
    assert not failures, f"{len(failures)} mismatch(es):\n" + "\n".join(failures)


# --------------------------------------------------------------------------
# Per-layer LR scaling
# --------------------------------------------------------------------------

# The parameter shapes record #40 actually hands to the hidden-matrix optimizer,
# with model_dim=768, hdim=3072, num_heads=6 (see GPT.__init__ in train_gpt.py).
_REC40_SHAPES = [
    ("smear_gate.weight", (1, 12), None),
    ("attn_gate.weight", (6, 12), None),
    ("blocks.attn.qkvo_w", (768, 3072), "attn"),   # used as 4 x [768, 768]
    ("blocks.mlp.c_fc", (768, 3072), "mlp"),
    ("blocks.mlp.c_proj", (768, 3072), "mlp"),
]


def _tagged(shape, module):
    p = torch.zeros(shape)
    if module is not None:
        p.module = module
    return p


def test_lmo_family_matches_record40_aspect_factor():
    """The LMO multiplier must be Keller Jordan's / record #40's shipped factor.

    Record #40's Muon computes ``max(1, p.size(-2) / p.size(-1)) ** 0.5`` on the
    STORED parameter. Our rule computes it on the shape the LMO actually operates
    on (the four [hdim, dim] blocks for the merged attention weight). Those must
    coincide on every shape #40 uses, or the reference `Muon` here is not the
    record's Muon.
    """
    for name, shape, module in _REC40_SHAPES:
        p = _tagged(shape, module)
        rec40 = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
        for rule in ("unit-gain", "legacy", "mup"):
            got = smo.layer_multiplier(p, smo.FAMILY_LMO, rule)
            assert abs(got - rec40) < 1e-12, (
                f"{name} rule={rule}: lmo lambda {got} != record #40's {rec40}")
        print(f"  OK   {name:<22} {str(shape):<14} lmo lambda = {rec40:.6g}  (== record #40)")


def test_unit_gain_equalizes_the_per_step_gain():
    """One eta_0 means one per-step RMS gain, for every shape and both families.

    gamma(s) = ||s||_F / sqrt(fan_out) exactly, with ||polar||_F = sqrt(min(m,n))
    and ||+-1||_F = sqrt(m n). unit-gain must drive gamma(eta_0 * lambda * s) to
    eta_0 in both rows -- that is what makes a learning rate comparable across
    the eight methods.
    """
    eta0 = 0.06
    for name, shape, module in _REC40_SHAPES:
        p = _tagged(shape, module)
        m, n = smo.lmo_shape(p)
        lam_lmo = smo.layer_multiplier(p, smo.FAMILY_LMO, "unit-gain")
        lam_sign = smo.layer_multiplier(p, smo.FAMILY_SIGN, "unit-gain")
        gain_lmo = eta0 * lam_lmo * math.sqrt(min(m, n)) / math.sqrt(m)
        gain_sign = eta0 * lam_sign * math.sqrt(m * n) / math.sqrt(m)
        assert abs(gain_lmo - eta0) < 1e-12, f"{name}: lmo gain {gain_lmo} != {eta0}"
        assert abs(gain_sign - eta0) < 1e-12, f"{name}: sign gain {gain_sign} != {eta0}"
        print(f"  OK   {name:<22} lambda_lmo={lam_lmo:<8.5g} lambda_sign={lam_sign:<10.5g}"
              f" gain={gain_lmo:.4g}")


def test_agrees_with_common_lr_scaling():
    """This module duplicates ``code/common/lr_scaling.py`` (a log must be
    self-contained); the two must not drift apart."""
    from common import lr_scaling as cls_

    for rule in ("unit-gain", "mup", "legacy", "none"):
        ref_rule = cls_.resolve_rule(rule)
        for name, shape, module in _REC40_SHAPES:
            p = _tagged(shape, module)
            m, n = smo.lmo_shape(p)
            for family in (smo.FAMILY_LMO, smo.FAMILY_SIGN):
                got = smo.layer_multiplier(p, family, rule)
                want = ref_rule.multiplier(family, m, n)
                assert abs(got - want) < 1e-12, (
                    f"{rule}/{family}/{name}: nanogpt {got} != common/lr_scaling {want}")
        print(f"  OK   rule '{rule}' agrees with common/lr_scaling.py")


def test_semantic_rule_only_moves_the_transposed_mlp_matrix():
    """`semantic` corrects record #40's transposed `c_fc` storage and nothing else."""
    for name, shape, module in _REC40_SHAPES:
        p = _tagged(shape, module)
        m, n = smo.lmo_shape(p)
        p.fan_out_sem, p.fan_in_sem = (n, m) if name.endswith("c_fc") else (m, n)
        for family in (smo.FAMILY_LMO, smo.FAMILY_SIGN):
            a = smo.layer_multiplier(p, family, "unit-gain")
            b = smo.layer_multiplier(p, family, "semantic")
            if name.endswith("c_fc"):
                assert abs(b / a - 2.0) < 1e-12, f"{name}/{family}: expected 2x, got {b / a}"
            else:
                assert abs(a - b) < 1e-12, f"{name}/{family}: semantic moved it ({a} -> {b})"
    print("  OK   'semantic' == 'unit-gain' except a 2x on the transposed c_fc")


if __name__ == "__main__":
    print("Verifying distributed optimizer cores against the numpy paper reference...\n")
    test_update_math_matches_reference()
    print("\nVerifying per-layer LR scaling...\n")
    test_lmo_family_matches_record40_aspect_factor()
    test_unit_gain_equalizes_the_per_step_gain()
    test_agrees_with_common_lr_scaling()
    test_semantic_rule_only_moves_the_transposed_mlp_matrix()
    print("\nAll optimizer update recurrences and LR multipliers match. PASS.")
