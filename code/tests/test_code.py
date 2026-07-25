"""CPU-only checks for the optimizers, the federated driver, and the plumbing.

    python3 -m tests.test_code     # run everything, print a report
    pytest tests/test_code.py      # or under pytest

Both from the ``code/`` directory.

No GPU, no dataset download, a few seconds. What it pins down:

* the Newton-Schulz helper does not scribble on its input (it used to, whenever
  the input was already bfloat16, because ``Tensor.to`` returns ``self``);
* it really does approximate the polar factor;
* every sign/LMO method is invariant to a positive rescaling of the gradient --
  the property that makes the paper's heavy-ball main text and its EMA algorithm
  boxes describe the same trajectories;
* the paper's Theorem 1-3 descent inner products, computed with torch;
* **the federated driver with one client reproduces the centralized optimizer
  exactly**, for all eight matrix-parameter rules. This is the load-bearing test:
  it ties ``federated_algorithms.run_federated`` to ``optimizers.py`` so the two
  cannot silently diverge;
* several clients holding identical data reproduce the single-client run;
* EF21-MuonSign keeps its exact model ``X`` distinct from the broadcast ``W``;
* the metrics schema and the multi-seed aggregation.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from federated.algorithms import METHODS, run_federated
from common.optimizers import (
    EF21MuonSign,
    EF21MuonUSign,
    EF21SignMuon,
    Muon,
    MuonSign,
    MuonUSign,
    SignMuon,
    SignSGD,
    muon_lmo,
    zeropower_via_newtonschulz5,
)
from common.utils import History, split_param_names

CENTRAL_CLASSES = {
    "signmuon": SignMuon,
    "ef21signmuon": EF21SignMuon,
    "muonusign": MuonUSign,
    "muonsign": MuonSign,
    "ef21muonusign": EF21MuonUSign,
    "ef21muonsign": EF21MuonSign,
    "muon": Muon,
    "signsgd": SignSGD,
}

TOL = 1e-5


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class TinyNet(nn.Module):
    """Exactly one matrix parameter plus a 2-tensor 'head', so the routing is
    unambiguous: ``fc1.weight`` is the matrix, everything else is auxiliary."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def tiny_data(n: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 6, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return x, y


def tiny_loader(n: int = 8, seed: int = 0):
    """One deterministic full batch per epoch (``shuffle=False``)."""
    x, y = tiny_data(n, seed)
    return DataLoader(TensorDataset(x, y), batch_size=n, shuffle=False)


def exact_polar(Y: torch.Tensor) -> torch.Tensor:
    """Rank-truncated ``U V^T`` via SVD, in float64."""
    U, S, Vh = torch.linalg.svd(Y.double(), full_matrices=False)
    r = int((S > 1e-9 * S[0]).sum())
    return U[:, :r] @ Vh[:r, :]


# --------------------------------------------------------------------------
# Newton-Schulz helper
# --------------------------------------------------------------------------


def test_newton_schulz_does_not_mutate_input():
    for dtype in (torch.float32, torch.bfloat16):
        G = torch.randn(5, 7, dtype=dtype)
        before = G.clone()
        zeropower_via_newtonschulz5(G, steps=5, dtype=dtype)
        assert torch.equal(G, before), f"input mutated for dtype={dtype}"


def test_newton_schulz_approximates_polar():
    torch.manual_seed(0)
    # Well conditioned: 5 steps of the Muon quintic land close to the polar factor.
    G = torch.randn(8, 8)
    D = muon_lmo(G, ns_steps=5, dtype=torch.float32, scale_aspect=False)
    P = exact_polar(G).float()
    assert torch.equal(torch.sign(D), torch.sign(P)), "sign pattern differs"
    assert (D - P).norm() / P.norm() < 0.15, f"relative error {(D - P).norm() / P.norm()}"


def test_muon_lmo_shapes():
    for shape in [(4, 6), (6, 4), (5, 5), (3, 2, 2, 2)]:
        G = torch.randn(*shape)
        assert muon_lmo(G, dtype=torch.float32).shape == G.shape
    v = torch.randn(7)                       # ndim < 2 passes straight through
    assert torch.equal(muon_lmo(v), v)


def test_aspect_ratio_scaling_is_invisible_to_output_signing():
    """SignMuon/MuonSign sign the LMO output, so Muon's sqrt(max(1,m/n)) factor
    cannot change their step; Muon/MuonUSign it does scale."""
    torch.manual_seed(1)
    G = torch.randn(10, 4)
    a = muon_lmo(G, dtype=torch.float32, scale_aspect=True)
    b = muon_lmo(G, dtype=torch.float32, scale_aspect=False)
    assert torch.equal(torch.sign(a), torch.sign(b))
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------
# Gradient-scale invariance (the momentum-convention equivalence)
# --------------------------------------------------------------------------


def _drive(cls, grads, lr=0.1, momentum=0.7, nesterov=False):
    p = nn.Parameter(torch.zeros(4, 5))
    opt = cls([p], lr=lr, momentum=momentum, nesterov=nesterov,
              lmo_dtype=torch.float32)
    traj = []
    for g in grads:
        p.grad = g.clone()
        opt.step()
        traj.append(p.detach().clone())
    return traj


def test_gradient_scale_invariance():
    """Every method's iterates are unchanged when G -> c*G for c > 0.

    This is exactly why the paper's main-text heavy-ball momentum
    (``mu*M + G``) and its algorithm boxes' EMA (``mu*M + (1-mu)*G``) give
    identical trajectories: the two buffers differ by the constant factor
    ``1 - mu``, and sign, polar and the EF21 recursion are all positively
    homogeneous.
    """
    torch.manual_seed(2)
    grads = [torch.randn(4, 5) for _ in range(6)]
    for name, cls in CENTRAL_CLASSES.items():
        for nesterov in (False, True):
            base = _drive(cls, grads, nesterov=nesterov)
            scaled = _drive(cls, [3.0 * g for g in grads], nesterov=nesterov)
            for t, (u, v) in enumerate(zip(base, scaled)):
                assert torch.allclose(u, v, atol=1e-4), \
                    f"{name} (nesterov={nesterov}) not scale invariant at step {t}"


# --------------------------------------------------------------------------
# The paper's counterexamples, in torch
# --------------------------------------------------------------------------


def _signmuon_instance(sigma1: float):
    O = torch.tensor([[101., 20., 2., -2.],
                      [-20., 97., 20., -20.],
                      [-2., 20., 2., 101.],
                      [-2., 20., -101., -2.]], dtype=torch.float64) / 103.0
    u1 = torch.tensor([10., -3., 10., 10.], dtype=torch.float64).reshape(-1, 1) / math.sqrt(309.)
    v1 = torch.tensor([10., 3., -10., 10.], dtype=torch.float64).reshape(-1, 1) / math.sqrt(309.)
    return sigma1 * (u1 @ v1.T) + O, O


def _muonsign_instance(M: float, eps: float = 1.0):
    S = torch.tensor([[-1., -1., 1., 1., 1.],
                      [-1., -1., 1., -1., -1.],
                      [1., -1., 1., 1., -1.],
                      [1., 1., -1., -1., 1.],
                      [1., 1., 1., -1., 1.]], dtype=torch.float64)
    G = eps * S.clone()
    G[3, 1] = eps * S[3, 1] + (M - eps)
    return G, S


def test_theorem1_exact_oracle():
    """<G, sign(polar(G))> = (-43*sigma1 + 532)/103."""
    for sigma1 in (100.0, 1000.0):
        G, O = _signmuon_instance(sigma1)
        P = exact_polar(G)
        assert (P - O).norm() < 1e-10, "polar(G) should equal O"
        got = float((G * torch.sign(P)).sum())
        want = (-43.0 * sigma1 + 532.0) / 103.0
        assert abs(got - want) < 1e-6, f"sigma1={sigma1}: {got} != {want}"
        assert got < 0, "SignMuon must ascend under the exact oracle"


def test_theorems23_exact_oracle():
    G, S = _muonsign_instance(100.0)
    assert torch.equal(torch.sign(G), S)
    D = exact_polar(S)
    assert abs(float(D[3, 1]) - (-0.2425)) < 1e-3, float(D[3, 1])
    assert abs(float((G * D).sum()) - (-13.888)) < 1e-2          # MuonUSign, Thm 2
    assert abs(float((G * torch.sign(D)).sum()) - (-76.0)) < 1e-9  # MuonSign, Thm 3


def test_practical_oracle_needs_the_larger_constants():
    """The counterexamples under the *implemented* Newton-Schulz oracle.

    Theorem 1 at the published sigma1=1000, and Theorem 2 at the published M=100,
    do NOT ascend with 5 Newton-Schulz steps; sigma1=100 and M=500 do. Theorem 3
    ascends either way. See ``counterexamples/verify_ns_oracle.py``.
    """
    def after(G):                       # SignMuon
        return float((G * torch.sign(muon_lmo(G.float(), 5, torch.float32, False))).sum())

    def before(G, S):                   # MuonUSign
        return float((G * muon_lmo(S.float(), 5, torch.float32, False).double()).sum())

    def both(G, S):                     # MuonSign
        return float((G * torch.sign(muon_lmo(S.float(), 5, torch.float32, False)).double()).sum())

    assert after(_signmuon_instance(1000.0)[0]) > 0, "expected the known failure at sigma1=1000"
    assert after(_signmuon_instance(100.0)[0]) < 0, "sigma1=100 should ascend at 5 NS steps"

    G100, S = _muonsign_instance(100.0)
    G500, _ = _muonsign_instance(500.0)
    assert before(G100, S) > 0, "expected the known failure at M=100"
    assert before(G500, S) < 0, "M=500 should ascend at 5 NS steps"
    assert both(G100, S) < 0 and both(G500, S) < 0, "Theorem 3 is oracle-robust"


# --------------------------------------------------------------------------
# Federated <-> centralized equivalence
# --------------------------------------------------------------------------


def _centralized_reference(method, rounds, lr, lr_aux, momentum, seed=0):
    """Drive the centralized optimizer by hand, one full-batch step per round."""
    torch.manual_seed(seed)
    model = TinyNet()
    matrix_names, aux_names = split_param_names(model, 2)
    named = dict(model.named_parameters())

    opt = CENTRAL_CLASSES[method]([named[n] for n in matrix_names],
                                  lr=lr, momentum=momentum, weight_decay=0.0,
                                  lmo_dtype=torch.float32)
    aux = torch.optim.AdamW([named[n] for n in aux_names], lr=lr_aux, weight_decay=0.0)

    loader = tiny_loader()
    criterion = nn.CrossEntropyLoss()
    for _ in range(rounds):
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            opt.step()
            aux.step()
    if hasattr(opt, "restore_exact"):
        opt.restore_exact()
    return model


def _federated_run(method, rounds, lr, lr_aux, momentum, n_clients=1, seed=0):
    torch.manual_seed(seed)
    model = TinyNet()
    loaders = [tiny_loader() for _ in range(n_clients)]
    run_federated(
        method, model, loaders, [tiny_loader()],
        rounds=rounds, n_steps=1, lr=lr, lr_aux=lr_aux, momentum=momentum,
        weight_decay=0.0, eval_freq=10 ** 9, device="cpu",
        cosine_schedule=False, lmo_dtype=torch.float32, verbose=False,
    )
    return model


def test_federated_one_client_equals_centralized():
    """The load-bearing test: N=1 federated == the centralized optimizer.

    ``sgd`` and ``adam`` are excluded: they are torch optimizers on the
    centralized side and server-side aggregation rules on the federated side, and
    ``sgd`` deliberately keeps the heavy-ball momentum convention.
    """
    failures = []
    for method in CENTRAL_CLASSES:
        ref = _centralized_reference(method, rounds=6, lr=0.05, lr_aux=0.01, momentum=0.8)
        fed = _federated_run(method, rounds=6, lr=0.05, lr_aux=0.01, momentum=0.8)
        for (n, a), (m, b) in zip(ref.named_parameters(), fed.named_parameters()):
            assert n == m
            if not torch.allclose(a, b, atol=TOL):
                failures.append(f"{method}/{n}: max|diff| = {(a - b).abs().max():.3e}")
    assert not failures, "federated != centralized:\n  " + "\n  ".join(failures)


def test_identical_clients_reduce_to_one():
    """N clients with identical data give the same model as N=1.

    Holds for every uplink: the majority vote over identical signs is that sign,
    and the average of identical EF21 payloads is that payload.
    """
    failures = []
    for method in METHODS:
        one = _federated_run(method, rounds=4, lr=0.05, lr_aux=0.01, momentum=0.8, n_clients=1)
        many = _federated_run(method, rounds=4, lr=0.05, lr_aux=0.01, momentum=0.8, n_clients=3)
        for (n, a), (_, b) in zip(one.named_parameters(), many.named_parameters()):
            if not torch.allclose(a, b, atol=TOL):
                failures.append(f"{method}/{n}: max|diff| = {(a - b).abs().max():.3e}")
    assert not failures, "N=3 identical clients != N=1:\n  " + "\n  ".join(failures)


def test_all_federated_methods_run():
    for method in METHODS:
        model = TinyNet()
        history = run_federated(
            method, model, [tiny_loader()], [tiny_loader()],
            rounds=3, n_steps=2, lr=0.01, lr_aux=0.01, momentum=0.9,
            weight_decay=1e-4, eval_freq=1, device="cpu",
            lmo_dtype=torch.float32, verbose=False,
        )
        assert history.steps == [0, 1, 2, 3], f"{method}: {history.steps}"
        for value in history.series["test_acc"]:
            assert value is not None and 0.0 <= value <= 100.0
        for p in model.parameters():
            assert torch.isfinite(p).all(), f"{method} produced non-finite parameters"


def test_eval_freq_records_only_evaluated_rounds():
    history = run_federated(
        "signmuon", TinyNet(), [tiny_loader()], [tiny_loader()],
        rounds=5, n_steps=1, lr=0.01, eval_freq=2, device="cpu",
        lmo_dtype=torch.float32, verbose=False,
    )
    assert history.steps == [0, 2, 4, 5], history.steps


# --------------------------------------------------------------------------
# EF21-MuonSign bookkeeping
# --------------------------------------------------------------------------


def test_ef21muonsign_separates_exact_and_broadcast_models():
    torch.manual_seed(3)
    p = nn.Parameter(torch.zeros(4, 4))
    opt = EF21MuonSign([p], lr=0.1, momentum=0.0, lmo_dtype=torch.float32)
    for _ in range(4):
        p.grad = torch.randn(4, 4)
        opt.step()

    W = p.detach().clone()
    X = opt.state[p]["exact_model"]
    assert not torch.allclose(W, X), "W and X should differ once the downlink compresses"

    with opt.using_exact():
        assert torch.allclose(p.data, X), "using_exact must expose X"
    assert torch.allclose(p.data, W), "using_exact must put W back"

    opt.restore_exact()
    assert torch.allclose(p.data, X), "restore_exact must install X"


def test_ef21_estimator_tracks_a_constant_target():
    """With a constant gradient the EF21 estimator converges to it, which is why
    compressing before the oracle restores the descent property."""
    p = nn.Parameter(torch.zeros(6, 6))
    opt = EF21MuonUSign([p], lr=0.0, momentum=0.0, lmo_dtype=torch.float32)
    target = torch.randn(6, 6)
    for _ in range(1000):
        p.grad = target.clone()
        opt.step()
    est = opt.state[p]["grad_estimator"]
    assert (est - target).norm() / target.norm() < 1e-3, (est - target).norm()


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_split_param_names():
    model = TinyNet()
    matrix, aux = split_param_names(model, n_head_tensors=2)
    assert matrix == ["fc1.weight"]
    assert aux == ["fc1.bias", "fc2.weight", "fc2.bias"]
    assert set(matrix) | set(aux) == {n for n, _ in model.named_parameters()}


def test_history_pads_missing_series():
    h = History()
    h.record(0, test_acc=1.0)
    h.record(5, test_acc=2.0, test_loss=0.5)
    assert h.to_dict() == {"steps": [0, 5], "test_acc": [1.0, 2.0],
                           "test_loss": [None, 0.5]}
    assert h.last("test_acc") == 2.0


def test_aggregate_groups_by_seed(tmp_path=None):
    import json
    import tempfile
    from pathlib import Path

    import aggregate

    root = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    for seed, acc in ((0, 80.0), (1, 82.0), (2, 84.0)):
        d = root / "run" / f"seed{seed}"
        d.mkdir(parents=True)
        with open(d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"config": {"algorithm": "signmuon", "lr": 0.01, "seed": seed,
                                  "device": f"cuda:{seed}"},
                       "history": {"steps": [0, 1], "test_acc": [10.0, acc]}}, f)

    runs = aggregate.load_runs([root])
    assert len(runs) == 3
    keys = {aggregate.group_key(r["config"]) for r in runs}
    assert len(keys) == 1, "seed and device must not split the group"

    agg = aggregate.aggregate_group(runs, "test_acc")
    assert agg["steps"] == [0, 1] and agg["n_runs"] == 3
    assert abs(agg["mean"][-1] - 82.0) < 1e-12
    assert abs(agg["std"][-1] - 2.0) < 1e-12          # sample std of 80, 82, 84


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, str(exc)))
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
