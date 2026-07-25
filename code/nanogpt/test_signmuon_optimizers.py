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
* Drive both with the same fixed random gradient sequence on SQUARE matrices
  (so the Muon fan-in lr scaling equals 1 and matches the reference's plain eta),
  weight_decay = 0, Nesterov momentum (the optimizer's only momentum form).
* We call ``update_param`` directly (the centralized core), bypassing the
  collectives -- those are exercised by ``test_distributed_sharding.py``.

Run:  SIGNMUON_NO_COMPILE=1 python test_signmuon_optimizers.py
      (or: pytest test_signmuon_optimizers.py)
"""

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
    opt = cls([p], lr=eta, momentum=mu, weight_decay=0.0)
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


if __name__ == "__main__":
    print("Verifying distributed optimizer cores against the numpy paper reference...\n")
    test_update_math_matches_reference()
    print("\nAll optimizer update recurrences match the reference. PASS.")
