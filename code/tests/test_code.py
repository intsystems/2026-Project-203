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


def _controlled_spectrum(shape, kappa: float, seed: int) -> torch.Tensor:
    """``U diag(s) V^T`` with ``s`` log-spaced over ``[1/kappa, 1]``.

    Testing on Gaussian matrices conflates the implementation with the *input's*
    conditioning -- a Gaussian's condition number is heavy-tailed, and 5 Newton-Schulz
    steps genuinely cannot lift a singular value that starts too small. Controlling
    the spectrum separates the two.
    """
    g = torch.Generator().manual_seed(seed)
    m, n = shape
    r = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, r, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, r, generator=g))
    s = torch.logspace(math.log10(1.0 / kappa), 0.0, r)
    return U @ torch.diag(s) @ V.T


def test_newton_schulz_gets_the_direction_but_not_the_scale():
    """What 5 steps of the Muon quintic actually deliver.

    The quintic is *not* a convergent iteration: it drives the singular values into
    a band around 1 and oscillates inside it. Two consequences, both measured over
    80 (shape, seed) cases:

    * On an **already orthogonal** input the direction is exact -- ``NS(Q) = c*Q`` for
      a positive scalar ``c``, since ``A = QQ^T`` is a multiple of the identity and
      each step just rescales. So cosine is 1.0000 and every sign agrees. But the
      *relative Frobenius error reaches 0.31*, because ``c`` oscillates in
      ``[0.69, 1.07]``. Asserting a small Frobenius error would therefore be wrong
      even for a perfect implementation -- the quintic does not fix the scale, which
      is exactly why magnitude is handled separately by ``lambda`` / ``scale_aspect``.
    * On a conditioned input the sign pattern agrees only 84-91% of the time. So
      ~1 entry in 8 of ``sign(polar(G))`` flips under the practical oracle -- the
      reason the sign-family methods are sensitive to the LMO approximation
      (cf. ``--lmo-dtype`` and ``counterexamples/verify_ns_oracle.py``).

    The assertions below therefore test *direction* and *approximate orthogonality*,
    with margins taken from the 80-case sweep.
    """
    shapes = [(8, 8), (16, 16), (64, 64), (32, 96)]

    # (a) orthogonal input: direction must be essentially exact.
    for shape in shapes:
        for seed in (0, 1):
            G = _controlled_spectrum(shape, kappa=1.0, seed=seed)
            D = muon_lmo(G, ns_steps=5, dtype=torch.float32, scale_aspect=False)
            P = exact_polar(G).float()
            cos = float((D * P).sum() / (D.norm() * P.norm()))
            agree = float((torch.sign(D) == torch.sign(P)).float().mean())
            assert cos > 0.999, f"{shape}: cosine {cos:.5f} on an orthogonal input"
            assert agree > 0.98, f"{shape}: {agree:.1%} sign agreement on an " \
                                 f"orthogonal input (should be exact)"

    # (b) moderately conditioned input: direction close, scale within the band.
    for shape in shapes:
        for seed in (0, 1, 2):
            G = _controlled_spectrum(shape, kappa=4.0, seed=seed)
            D = muon_lmo(G, ns_steps=5, dtype=torch.float32, scale_aspect=False)
            P = exact_polar(G).float()
            cos = float((D * P).sum() / (D.norm() * P.norm()))
            agree = float((torch.sign(D) == torch.sign(P)).float().mean())
            sv = torch.linalg.svdvals(D.double())
            assert cos > 0.95, f"{shape}: cosine {cos:.4f} -- wrong direction"
            assert agree > 0.70, f"{shape}: only {agree:.1%} of signs agree"
            assert 0.4 < float(sv.min()) and float(sv.max()) < 1.5, \
                f"{shape}: singular values [{sv.min():.3f}, {sv.max():.3f}] -- not " \
                f"approximately orthogonal"


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


def test_exact_and_newton_schulz_oracles_differ_on_the_instances():
    """Regression test for a measured fact, not a defect.

    The theorems are stated for the exact LMO and the counterexample code runs the
    exact LMO; networks are trained with Newton-Schulz, as practitioners do. The
    two oracles are different maps, and on these instances they disagree: at the
    published constants (sigma1=1000, M=100) the 5-step oracle does not ascend for
    Theorems 1-2, while sigma1=100 / M=500 ascend under both. Theorem 3 is
    oracle-robust. Pinned here so the numbers cannot drift; see
    ``counterexamples/verify_ns_oracle.py``.
    """
    def after(G):                       # SignMuon
        return float((G * torch.sign(muon_lmo(G.float(), 5, torch.float32, False))).sum())

    def before(G, S):                   # MuonUSign
        return float((G * muon_lmo(S.float(), 5, torch.float32, False).double()).sum())

    def both(G, S):                     # MuonSign
        return float((G * torch.sign(muon_lmo(S.float(), 5, torch.float32, False)).double()).sum())

    assert after(_signmuon_instance(1000.0)[0]) > 0, "sigma1=1000 does not ascend at 5 NS steps"
    assert after(_signmuon_instance(100.0)[0]) < 0, "sigma1=100 ascends at 5 NS steps"

    G100, S = _muonsign_instance(100.0)
    G500, _ = _muonsign_instance(500.0)
    assert before(G100, S) > 0, "M=100 does not ascend at 5 NS steps"
    assert before(G500, S) < 0, "M=500 ascends at 5 NS steps"
    assert both(G100, S) < 0 and both(G500, S) < 0, "Theorem 3 is oracle-robust"


# --------------------------------------------------------------------------
# Federated <-> centralized equivalence
# --------------------------------------------------------------------------


def _centralized_reference(method, rounds, lr, lr_aux, momentum, seed=0,
                           weight_decay=0.0):
    """Drive the centralized optimizer by hand, one full-batch step per round."""
    torch.manual_seed(seed)
    model = TinyNet()
    matrix_names, aux_names = split_param_names(model, 2)
    named = dict(model.named_parameters())

    opt = CENTRAL_CLASSES[method]([named[n] for n in matrix_names],
                                  lr=lr, momentum=momentum,
                                  weight_decay=weight_decay,
                                  decoupled_weight_decay=True,
                                  lmo_dtype=torch.float32)
    aux = torch.optim.AdamW([named[n] for n in aux_names], lr=lr_aux,
                            weight_decay=weight_decay)

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


def _federated_run(method, rounds, lr, lr_aux, momentum, n_clients=1, seed=0,
                   weight_decay=0.0):
    torch.manual_seed(seed)
    model = TinyNet()
    loaders = [tiny_loader() for _ in range(n_clients)]
    run_federated(
        method, model, loaders, [tiny_loader()],
        rounds=rounds, n_steps=1, lr=lr, lr_aux=lr_aux, momentum=momentum,
        weight_decay=weight_decay, eval_freq=10 ** 9, device="cpu",
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


def test_the_two_drivers_agree_on_the_weight_decay_convention():
    """Same equivalence, now with weight decay switched on.

    The zero-decay test above passes under *either* convention, so it cannot see a
    coupled/decoupled mismatch between the two drivers -- which is exactly the
    discrepancy that existed. Decoupled is the only well-posed choice here: every
    step direction is positively homogeneous of degree zero, so folding ``wd * X``
    into the gradient leaves the step length untouched and merely rotates it.
    """
    failures = []
    for method in CENTRAL_CLASSES:
        kw = dict(rounds=6, lr=0.05, lr_aux=0.01, momentum=0.8, weight_decay=0.02)
        ref = _centralized_reference(method, **kw)
        fed = _federated_run(method, **kw)
        moved = max((p - q).abs().max().item()
                    for p, q in zip(_centralized_reference(method, rounds=6, lr=0.05,
                                                           lr_aux=0.01, momentum=0.8,
                                                           weight_decay=0.0).parameters(),
                                    ref.parameters()))
        # A decay too small to move the parameters would make this test vacuous.
        assert moved > 10 * TOL, f"{method}: wd=0.02 changed nothing ({moved:.2e})"
        for (n, a), (m, b) in zip(ref.named_parameters(), fed.named_parameters()):
            assert n == m
            if not torch.allclose(a, b, atol=TOL):
                failures.append(f"{method}/{n}: max|diff| = {(a - b).abs().max():.3e}")
    assert not failures, ("the drivers disagree once weight decay is nonzero:\n  "
                          + "\n  ".join(failures))


def test_coupled_decay_cannot_change_the_step_length():
    """Coupled decay through a scale-invariant step map shrinks nothing.

    ``sign`` and ``polar`` are positively homogeneous of degree zero, so adding
    ``wd * X`` to the gradient can only rotate the update. This is the reason the
    coupled convention is kept for an ablation and is not the default.
    """
    torch.manual_seed(0)
    for name, cls in CENTRAL_CLASSES.items():
        p_c = nn.Parameter(torch.randn(16, 12))
        p_d = nn.Parameter(p_c.detach().clone())
        g = torch.randn(16, 12) * 1e-3          # small gradient => large rho
        steps = {}
        for tag, param, coupled in (("coupled", p_c, True), ("plain", p_d, False)):
            opt = cls([param], lr=0.1, momentum=0.0, lmo_dtype=torch.float32,
                      weight_decay=0.5 if coupled else 0.0,
                      decoupled_weight_decay=False)
            before = param.detach().clone()
            param.grad = g.clone()
            opt.step()
            if hasattr(opt, "restore_exact"):
                opt.restore_exact()
            steps[tag] = (param.detach() - before)
        n_c, n_p = steps["coupled"].norm().item(), steps["plain"].norm().item()
        assert abs(n_c - n_p) <= 1e-5 * max(n_p, 1e-12), (
            f"{name}: coupled decay changed the step norm {n_p:.6g} -> {n_c:.6g}; "
            f"the step map is supposed to be scale-invariant")
        # ... and it does rotate it, so it is not simply a no-op either.
        cos = (steps["coupled"] * steps["plain"]).sum() / (n_c * n_p + 1e-30)
        assert cos < 0.999, f"{name}: coupled decay had no effect at all (cos={cos:.4f})"


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


# --------------------------------------------------------------------------
# Per-layer learning-rate scaling
# --------------------------------------------------------------------------


def test_unit_gain_reproduces_the_shipped_muon_factor():
    """The criterion's strongest validation.

    Applying the unit-gain criterion to an LMO step must reproduce
    ``sqrt(max(1, fan_out/fan_in))``, the aspect factor in the reference Muon
    implementation. If this failed, the criterion would be ad hoc.
    """
    from common.lr_scaling import FAMILY_LMO, layer_multiplier, resolve_rule

    ug, legacy = resolve_rule("unit-gain"), resolve_rule("legacy")
    for shape in [(64, 3, 3, 3), (512, 512, 3, 3), (512, 256, 1, 1), (10, 512)]:
        a = layer_multiplier(ug, FAMILY_LMO, shape)
        b = layer_multiplier(legacy, FAMILY_LMO, shape)
        m, n = shape[0], math.prod(shape[1:]) if len(shape) > 1 else 1
        assert abs(a - b) < 1e-12, (shape, a, b)
        assert abs(a - math.sqrt(max(1.0, m / n))) < 1e-12


def test_unit_gain_equalizes_the_per_step_gain_exactly():
    """The invariant the whole rule exists to enforce.

    ``unit-gain`` is exactly ``lambda = sqrt(fan_out)/||s||_F``, so the per-step RMS
    gain ``gamma(eta_0 * lambda * s) = eta_0 * lambda * ||s||_F / sqrt(fan_out)``
    equals ``eta_0`` on *every* layer and in *both* families. If this ever fails the
    rule has been mis-implemented.
    """
    from common.lr_scaling import (FAMILY_LMO, FAMILY_SIGN, RESNET18_SHAPES,
                                   resolve_rule)

    rule = resolve_rule("unit-gain")
    for _, m, n in RESNET18_SHAPES:
        for family, s_fro in ((FAMILY_LMO, math.sqrt(min(m, n))),
                              (FAMILY_SIGN, math.sqrt(m * n))):
            lam = rule.multiplier(family, m, n)
            assert abs(lam - math.sqrt(m) / s_fro) < 1e-12, (m, n, family)
            gain = lam * s_fro / math.sqrt(m)
            assert abs(gain - 1.0) < 1e-12, (m, n, family, gain)


def test_initialization_gain_is_shape_independent():
    """The rule's premise: a fan-in-scaled init has the same gain at every shape, so
    the constant of proportionality is absorbed into eta_0 rather than varying."""
    from common.lr_scaling import RESNET18_SHAPES

    for _, m, n in RESNET18_SHAPES:
        he = math.sqrt(2 / n) * math.sqrt(m * n) / math.sqrt(m)          # He normal
        torch_default = (1 / math.sqrt(3 * n)) * math.sqrt(m * n) / math.sqrt(m)
        assert abs(he - math.sqrt(2)) < 1e-12
        assert abs(torch_default - 1 / math.sqrt(3)) < 1e-12


def test_unit_gain_sign_family_is_inverse_sqrt_fan_in():
    """And it must be independent of fan-out, which is what the algebra says."""
    from common.lr_scaling import FAMILY_SIGN, layer_multiplier, resolve_rule

    rule = resolve_rule("unit-gain")
    for shape in [(64, 3, 3, 3), (512, 512, 3, 3), (128, 64, 1, 1)]:
        n = math.prod(shape[1:])
        assert abs(layer_multiplier(rule, FAMILY_SIGN, shape) - 1 / math.sqrt(n)) < 1e-12
    # same fan-in, different fan-out => same multiplier
    a = layer_multiplier(rule, FAMILY_SIGN, (64, 64, 3, 3))
    b = layer_multiplier(rule, FAMILY_SIGN, (512, 64, 3, 3))
    assert abs(a - b) < 1e-12


def test_power_presets_match_named_rules():
    from common.lr_scaling import FAMILY_SIGN, layer_multiplier, resolve_rule

    shape = (256, 128, 3, 3)
    for spec, named in [("power:0.5", "unit-gain"), ("power:1", "mup"),
                        ("power:0.5,0.5", "mishra-analysis"), ("power:0", "none")]:
        a = layer_multiplier(resolve_rule(spec), FAMILY_SIGN, shape)
        b = layer_multiplier(resolve_rule(named), FAMILY_SIGN, shape)
        assert abs(a - b) < 1e-12, (spec, named, a, b)


def test_rms_gain_identity():
    """``gamma(A) = ||A||_F / sqrt(m)`` really is the RMS gain, and the two families
    have the exact Frobenius norms the derivation assumes."""
    torch.manual_seed(7)
    m, n = 40, 90
    M = torch.randn(m, n)
    P = exact_polar(M).float()
    assert abs(float(P.norm()) - math.sqrt(min(m, n))) < 1e-4       # ||UV^T||_F = sqrt(r)
    S = torch.sign(P)
    assert abs(float(S.norm()) - math.sqrt(m * n)) < 1e-4           # +-1 entries

    # rms(A u) / rms(u) -> ||A||_F / sqrt(m) in expectation
    u = torch.randn(n, 2048)
    for A in (P, S):
        emp = float((A @ u).norm() / math.sqrt(m)) / float(u.norm() / math.sqrt(n))
        pred = float(A.norm()) / math.sqrt(m)
        assert abs(emp - pred) / pred < 0.05, (emp, pred)


def test_lambda_mult_equals_folding_the_factor_into_the_lmo():
    """Putting the aspect factor in ``lambda_mult`` (with ``scale_aspect=False``)
    is exactly equivalent to leaving it inside the LMO -- so the refactor that
    moved it out cannot have changed any Muon-family result."""
    torch.manual_seed(11)
    # Both regimes: fan_out > fan_in (factor sqrt(2) != 1, so the test bites) and
    # fan_out < fan_in (factor clipped to 1).
    for shape in [(16, 2, 2, 2), (20, 5), (64, 64, 3, 3)]:
        n = math.prod(shape[1:])
        lam = math.sqrt(max(1.0, shape[0] / n))
        grads = [torch.randn(*shape) for _ in range(4)]

        p_in = nn.Parameter(torch.zeros(*shape))
        opt_in = Muon([p_in], lr=0.1, momentum=0.5, lmo_dtype=torch.float32,
                      scale_aspect=True)
        p_out = nn.Parameter(torch.zeros(*shape))
        opt_out = Muon([{"params": [p_out], "lambda_mult": lam}], lr=0.1, momentum=0.5,
                       lmo_dtype=torch.float32, scale_aspect=False)

        for g in grads:
            p_in.grad, p_out.grad = g.clone(), g.clone()
            opt_in.step()
            opt_out.step()
            assert torch.allclose(p_in, p_out, atol=1e-6), \
                f"{shape} (lambda={lam:.4f}): {(p_in - p_out).abs().max()}"
        assert p_in.abs().sum() > 0, "the test must actually take steps"


def test_optimizer_families_are_tagged():
    from common.lr_scaling import FAMILY_LMO, FAMILY_SIGN

    expected = {
        SignMuon: FAMILY_SIGN, MuonSign: FAMILY_SIGN, SignSGD: FAMILY_SIGN,
        Muon: FAMILY_LMO, MuonUSign: FAMILY_LMO, EF21SignMuon: FAMILY_LMO,
        EF21MuonUSign: FAMILY_LMO, EF21MuonSign: FAMILY_LMO,
    }
    for cls, fam in expected.items():
        assert cls.family == fam, (cls.__name__, cls.family, fam)


def test_build_optimizers_assigns_one_group_per_matrix_param():
    import argparse

    from centralized.train import build_optimizers

    args = argparse.Namespace(
        optimizer="signmuon", lr=0.1, lr_aux=0.01, momentum=0.9, nesterov=False,
        weight_decay=0.0, ns_steps=5, lmo_dtype="float32", lr_scaling="unit-gain",
        head_adamw="always", n_head_tensors=2)
    model = TinyNet()
    opt_main, opt_aux, info = build_optimizers(model, args)

    assert len(opt_main.param_groups) == 1                  # TinyNet has one matrix
    group = opt_main.param_groups[0]
    assert group["name"] == "fc1.weight"
    assert abs(group["lambda_mult"] - 1 / math.sqrt(6)) < 1e-12    # fan_in = 6
    assert group["scale_aspect"] is False, "the factor must not be applied twice"
    assert opt_aux is not None and len(opt_aux.param_groups[0]["params"]) == 3
    assert info["rule"] == "unit-gain" and info["family"] == "sign"


# --------------------------------------------------------------------------
# Protocol: validation split and metric helpers
# --------------------------------------------------------------------------


def test_val_split_is_disjoint_and_seed_stable():
    from centralized.data import _split_indices

    tr1, va1 = _split_indices(50_000, 5_000, val_seed=12345)
    tr2, va2 = _split_indices(50_000, 5_000, val_seed=12345)
    assert (tr1 == tr2).all() and (va1 == va2).all(), "split must be deterministic"
    assert len(va1) == 5_000 and len(tr1) == 45_000
    assert not (set(tr1.tolist()) & set(va1.tolist())), "train and val must be disjoint"

    _, va3 = _split_indices(50_000, 5_000, val_seed=999)
    assert set(va1.tolist()) != set(va3.tolist()), "a different val_seed must differ"


def test_history_selection_helpers():
    h = History()
    for step, (va, te) in enumerate([(80.0, 79.0), (85.0, 84.0), (83.0, 90.0),
                                     (84.0, 86.0), (84.5, 87.0)]):
        h.record(step, val_acc=va, test_acc=te)

    assert h.argbest("val_acc", "max") == 1          # best val, not best test
    assert h.at("test_acc", 1) == 84.0               # the number you'd report
    assert abs(h.last_k_mean("test_acc", 3) - (90.0 + 86.0 + 87.0) / 3) < 1e-12
    assert h.steps_to_target("test_acc", 86.0) == 2
    assert h.steps_to_target("test_acc", 99.0) is None
    assert h.values("val_acc") == [80.0, 85.0, 83.0, 84.0, 84.5]


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
