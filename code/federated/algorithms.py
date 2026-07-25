"""Federated implementations of the paper's methods, as one parameterized driver.

The paper's Table "The six federated methods as instantiations of the two
templates" says every method is fixed by three choices: *where the Muon LMO is
evaluated*, the *uplink* compressor, and the *downlink* compressor. This module
encodes exactly that, so all ten algorithms share one training loop and cannot
drift apart in learning-rate schedule, parameter routing, evaluation, or
weight-decay handling:

=================  ========  ===============  ==========  ============================
Method             LMO       Uplink           Downlink    Paper reference
=================  ========  ===============  ==========  ============================
``signmuon``       worker    sign / MV        exact       Alg. "fed_workerlmo", row 1
``ef21signmuon``   worker    EF21            exact        Alg. "fed_workerlmo", row 2
``muonusign``      server    sign / MV        exact       Alg. "fed_serverlmo", row 3
``muonsign``       server    sign / MV        sign        Alg. "fed_serverlmo", row 4
``ef21muonusign``  server    EF21            exact        Alg. "fed_serverlmo", row 5
``ef21muonsign``   server    EF21            EF21-P       Alg. "fed_serverlmo", row 6
``muon``           worker    exact (average)  exact       full-precision reference
``signsgd``        none      sign / MV        exact       reference
``sgd``            none      exact (average)  exact       reference
``adam``           none      exact (average)  exact       reference (server-side Adam)
=================  ========  ===============  ==========  ============================

Conventions shared by every method (this uniformity is the point of the module)
-----------------------------------------------------------------------------
* **Parameter routing.** The LMO/sign rule applies only to matrix parameters
  (``ndim >= 2``, excluding the classification head). Biases, BatchNorm scales
  and the head go to a server-side AdamW fed with the plain averaged gradient,
  as the paper specifies -- and identically for every method, so the *only*
  difference between two runs is the matrix-parameter rule.
* **Learning rate.** One cosine schedule (``utils.cosine_lr``), applied to both
  the main step size and the AdamW auxiliary rate, for every method.
* **Weight decay** is applied exactly once, decoupled, on the server
  (``X *= 1 - lr*wd`` for matrix parameters; AdamW's own ``weight_decay`` for the
  auxiliary ones). Clients accumulate *pure* gradients, so the LMO always sees
  the gradient geometry rather than a shrinkage-perturbed version of it.
* **Momentum** is the EMA form of the paper's boxes,
  ``M = mu*M + (1-mu)*G``, which is trajectory-identical to the heavy-ball form
  for every sign/LMO method (all are positively homogeneous in ``M``). The one
  exception is ``sgd``, whose step *is* the momentum buffer and which therefore
  keeps the heavy-ball convention of ``torch.optim.SGD``.

BatchNorm
---------
``freeze_bn_stats=True`` (the default, and the behaviour of the original code)
runs every BatchNorm layer in inference mode during local gradient
accumulation. Because the local models are discarded each round, this means the
running statistics are *never updated from data*: they stay at their
initialization ``(mean 0, var 1)`` for the whole run, in training and in
evaluation alike. BatchNorm therefore acts as a fixed identity normalization
with learnable affine parameters. This is self-consistent (no train/test
statistics mismatch) and is what the reported federated numbers were produced
with, but it is worth stating explicitly in a reproducibility appendix.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from common.optimizers import muon_lmo
from common.utils import History, cosine_lr, resolve_device, split_param_names

__all__ = ["MethodSpec", "METHODS", "evaluate_model", "run_federated"]


# --------------------------------------------------------------------------
# Method specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """Where the LMO runs and how each channel is compressed.

    Attributes
    ----------
    lmo : {"none", "worker", "server"}
        ``"worker"``: each client orthogonalizes locally and transmits a
        compressed *direction*. ``"server"``: clients transmit a compressed
        *gradient* and the server applies a single LMO to the reconstruction.
    uplink : {"exact", "sign_mv", "ef21"}
        ``"sign_mv"`` sends one bit per entry and the server takes a majority
        vote; ``"ef21"`` sends the scaled sign of the EF21 residual plus one
        scalar; ``"exact"`` sends the full-precision tensor and the server
        averages.
    downlink : {"exact", "sign", "ef21p"}
        ``"sign"`` broadcasts ``sign(D)`` (both sides apply the same step, so the
        models stay identical); ``"ef21p"`` keeps the exact model ``X`` on the
        server and broadcasts a scaled sign of the model shift, so clients see a
        compressed model ``W`` and evaluate their gradients there.
    server : {"step", "adam"}
        ``"step"``: ``X <- X - lr * D``. ``"adam"``: the aggregate is used as a
        gradient for a server-side Adam.
    client_momentum : bool
        Whether clients maintain a momentum buffer at all.
    momentum_form : {"ema", "heavy_ball"}
        See the module docstring; only ``sgd`` needs ``"heavy_ball"``.
    """

    lmo: str = "none"
    uplink: str = "exact"
    downlink: str = "exact"
    server: str = "step"
    client_momentum: bool = True
    momentum_form: str = "ema"

    @property
    def needs_exact_model(self) -> bool:
        """True when the server must keep an exact ``X`` distinct from ``W``."""
        return self.downlink == "ef21p"


METHODS: Dict[str, MethodSpec] = {
    # --- the six paper methods -------------------------------------------
    "signmuon":      MethodSpec(lmo="worker", uplink="sign_mv", downlink="exact"),
    "ef21signmuon":  MethodSpec(lmo="worker", uplink="ef21",    downlink="exact"),
    "muonusign":     MethodSpec(lmo="server", uplink="sign_mv", downlink="exact"),
    "muonsign":      MethodSpec(lmo="server", uplink="sign_mv", downlink="sign"),
    "ef21muonusign": MethodSpec(lmo="server", uplink="ef21",    downlink="exact"),
    "ef21muonsign":  MethodSpec(lmo="server", uplink="ef21",    downlink="ef21p"),
    # --- references -------------------------------------------------------
    "muon":          MethodSpec(lmo="worker", uplink="exact",   downlink="exact"),
    "signsgd":       MethodSpec(lmo="none",   uplink="sign_mv", downlink="exact"),
    "sgd":           MethodSpec(lmo="none",   uplink="exact",   downlink="exact",
                                momentum_form="heavy_ball"),
    "adam":          MethodSpec(lmo="none",   uplink="exact",   downlink="exact",
                                server="adam", client_momentum=False),
}

# Legacy CLI spellings kept working so old commands and logs still resolve.
METHOD_ALIASES = {
    "signmuon_cl": "signmuon",
    "signmuon_ef_21": "ef21muonusign",
    "signmuon_ef_ud": "ef21muonsign",
    "ef_usignmuon": "ef21muonusign",
    "ef_udsignmuon": "ef21muonsign",
}


def resolve_method(name: str) -> tuple[str, MethodSpec]:
    key = name.strip().lower().replace("-", "").replace(" ", "")
    key = METHOD_ALIASES.get(name.strip().lower(), METHOD_ALIASES.get(key, key))
    if key not in METHODS:
        raise ValueError(
            f"Unknown federated method {name!r}. Available: {sorted(METHODS)}"
        )
    return key, METHODS[key]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@contextmanager
def disable_bn_running_stats(model: nn.Module):
    """Temporarily put every BatchNorm layer in inference mode.

    Prevents the running statistics from being polluted by the small local
    batches; see the module docstring for the consequence.
    """
    flags = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            flags[module] = module.training
            module.eval()
    try:
        yield
    finally:
        for module, was_training in flags.items():
            module.train(was_training)


@torch.no_grad()
def evaluate_model(model, test_loaders: Sequence, device=None, verbose: bool = False):
    """Aggregate accuracy/loss over the per-client test loaders.

    Returns ``(per_client_accuracies, total_accuracy_pct, mean_loss)``. The loss
    is a true sample mean (``reduction="sum"`` divided by the sample count), so
    it does not depend on the batch size or on unequal client shard sizes.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    was_training = model.training
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    client_accuracies: List[float] = []
    total_correct = total_samples = 0
    total_loss = 0.0

    for client_id, loader in enumerate(test_loaders):
        correct = count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[-1]
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
        acc = 100.0 * correct / count if count else 0.0
        client_accuracies.append(acc)
        if verbose:
            print(f"  client {client_id}: {acc:.2f}%")
        total_correct += correct
        total_samples += count

    model.train(was_training)
    total_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    mean_loss = total_loss / total_samples if total_samples else 0.0
    return client_accuracies, total_acc, mean_loss


class _ClientData:
    """A client's loader plus a *persistent* iterator across rounds.

    Re-creating the iterator every round (as the previous implementation did)
    restarts the epoch each time, so a run only ever sees the first
    ``n_steps * batch_size`` samples of each freshly shuffled permutation. Keeping
    the iterator alive sweeps the client's whole shard and makes the data order a
    deterministic function of the DataLoader's seeded generator.
    """

    def __init__(self, loader):
        self.loader = loader
        self._it = None

    def next_batch(self):
        if self._it is None:
            self._it = iter(self.loader)
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


def _accumulate_gradients(
    model: nn.Module,
    data: _ClientData,
    n_steps: int,
    device,
    names: Optional[Iterable[str]] = None,
    freeze_bn_stats: bool = True,
) -> Dict[str, torch.Tensor]:
    """Average the gradient of the local loss over ``n_steps`` mini-batches.

    This is the paper's gradient-accumulation step,
    ``G_t = (1/n_steps) * sum_i grad f_j(X; xi_i)``. Gradients are *pure*: weight
    decay is applied on the server instead (see the module docstring).
    """
    wanted = set(names) if names is not None else None
    criterion = nn.CrossEntropyLoss()
    accumulated: Dict[str, torch.Tensor] = {}

    ctx = disable_bn_running_stats(model) if freeze_bn_stats else _nullcontext()
    with ctx:
        for _ in range(n_steps):
            x, y = data.next_batch()
            x, y = x.to(device), y.to(device).long()
            model.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            for name, param in model.named_parameters():
                if param.grad is None or (wanted is not None and name not in wanted):
                    continue
                if name in accumulated:
                    accumulated[name] += param.grad
                else:
                    accumulated[name] = param.grad.clone()

    for name in accumulated:
        accumulated[name] /= n_steps
    return accumulated


@contextmanager
def _nullcontext():
    yield


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def run_federated(
    method: str,
    global_model: nn.Module,
    train_loaders: Sequence,
    test_loaders: Sequence,
    rounds: int,
    n_steps: int,
    lr: float,
    lr_aux: float = 1e-3,
    momentum: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
    eval_freq: int = 1,
    device=None,
    buffer_device: str = "cpu",
    cosine_schedule: bool = True,
    freeze_bn_stats: bool = True,
    n_head_tensors: int = 2,
    adam_eps: float = 1e-8,
    lmo_dtype: Optional[torch.dtype] = torch.bfloat16,
    verbose: bool = True,
) -> History:
    """Train ``global_model`` with one of the ten federated methods.

    Returns a :class:`utils.History` recording ``test_acc`` / ``test_loss`` at
    every evaluated round (round 0 = before training). Un-evaluated rounds are
    simply absent rather than forward-filled, so curves from different seeds can
    be averaged pointwise.

    ``buffer_device`` is where the per-client momentum / EF21 buffers live
    between rounds; ``"cpu"`` keeps VRAM proportional to one client instead of
    ``N``, at the cost of a host<->device copy per round per client.
    """
    name, spec = resolve_method(method)
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    num_clients = len(train_loaders)

    global_model.to(device)
    global_model.train()

    matrix_names, aux_names = split_param_names(global_model, n_head_tensors)
    params = dict(global_model.named_parameters())
    if verbose:
        print(f"Federated {name} on {device}: {num_clients} clients, {rounds} rounds, "
              f"{len(matrix_names)} matrix params (LMO={spec.lmo}, "
              f"up={spec.uplink}, down={spec.downlink}), {len(aux_names)} aux params")

    # -- server state ------------------------------------------------------
    # AdamW on the auxiliary parameters, uniformly for every method.
    adamw = torch.optim.AdamW(
        [params[n] for n in aux_names], lr=lr_aux, weight_decay=weight_decay, eps=adam_eps
    ) if aux_names else None

    # Server-side Adam over the matrix parameters (the ``adam`` baseline only).
    server_adam = torch.optim.Adam(
        [params[n] for n in matrix_names], lr=lr, weight_decay=weight_decay, eps=adam_eps
    ) if spec.server == "adam" and matrix_names else None

    # EF21 reconstruction G_t on the server (uplink == "ef21").
    server_estimator = {n: torch.zeros_like(params[n], device=device)
                        for n in matrix_names} if spec.uplink == "ef21" else {}

    # Exact model X, distinct from the broadcast W held in ``params`` (ef21p).
    server_exact = {n: params[n].detach().clone()
                    for n in matrix_names} if spec.needs_exact_model else {}

    # -- client state ------------------------------------------------------
    def _zeros():
        return {n: torch.zeros_like(params[n], device=buffer_device) for n in matrix_names}

    client_momentum = [_zeros() for _ in range(num_clients)] if spec.client_momentum else None
    client_estimator = [_zeros() for _ in range(num_clients)] if spec.uplink == "ef21" else None
    client_data = [_ClientData(dl) for dl in train_loaders]
    # One reusable local model: cheaper than deep-copying the global model per
    # client per round, and semantically identical because the broadcast weights
    # are reloaded before every client's local work.
    local_model = copy.deepcopy(global_model).to(device)

    history = History()

    @contextmanager
    def _exact_view():
        """Expose the exact model X in ``global_model`` for evaluation."""
        if not server_exact:
            yield
            return
        backup = {n: params[n].detach().clone() for n in server_exact}
        with torch.no_grad():
            for n, X in server_exact.items():
                params[n].data.copy_(X)
        try:
            yield
        finally:
            with torch.no_grad():
                for n, w in backup.items():
                    params[n].data.copy_(w)

    def _evaluate(step: int, elapsed: Optional[float] = None) -> None:
        with _exact_view():
            _, acc, loss = evaluate_model(global_model, test_loaders, device=device)
        history.record(step, test_acc=acc, test_loss=loss)
        if verbose:
            timing = f" | {elapsed:.2f}s" if elapsed is not None else ""
            print(f"\rRound {step}{timing} | Accuracy: {acc:.2f}%, Loss: {loss:.4f}", flush=True)

    _evaluate(0)

    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        eta = cosine_lr(1.0, r - 1, rounds) if cosine_schedule else 1.0
        current_lr, current_lr_aux = lr * eta, lr_aux * eta
        if adamw is not None:
            for g in adamw.param_groups:
                g["lr"] = current_lr_aux
        if server_adam is not None:
            for g in server_adam.param_groups:
                g["lr"] = current_lr

        # ---------------- clients ----------------------------------------
        # The broadcast model: W (compressed) under ef21p, X otherwise. Both are
        # what ``params`` currently holds, so a plain state_dict copy is right.
        # Clients only *accumulate gradients* (no local parameter steps), so with
        # frozen BatchNorm statistics the local model is identical for every
        # client and one copy per round suffices.
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()

        uplink_payloads: List[Dict[str, torch.Tensor]] = []
        uplink_scales: List[Dict[str, torch.Tensor]] = []
        aux_grads_sum: Dict[str, torch.Tensor] = {}

        for j in range(num_clients):
            if not freeze_bn_stats and j > 0:
                # Live BatchNorm statistics *would* drift from one client to the
                # next within a round, so re-broadcast before each client.
                local_model.load_state_dict(global_model.state_dict())
            grads = _accumulate_gradients(
                local_model, client_data[j], n_steps, device,
                names=None, freeze_bn_stats=freeze_bn_stats,
            )

            for n in aux_names:
                if n in grads:
                    aux_grads_sum[n] = aux_grads_sum.get(n, 0) + grads[n]

            payload, scales = {}, {}
            for n in matrix_names:
                G = grads[n]

                # 1) client momentum
                if spec.client_momentum:
                    buf = client_momentum[j][n].to(device)
                    if spec.momentum_form == "ema":
                        buf.mul_(momentum).add_(G, alpha=1.0 - momentum)
                        m_tilde = G.mul(1.0 - momentum).add_(buf, alpha=momentum) \
                            if nesterov else buf
                    else:
                        buf.mul_(momentum).add_(G)
                        m_tilde = G.add(buf, alpha=momentum) if nesterov else buf
                else:
                    m_tilde = G

                # 2) worker-side LMO, when the sign acts after the oracle
                target = muon_lmo(m_tilde, ns_steps=ns_steps, dtype=lmo_dtype) \
                    if spec.lmo == "worker" else m_tilde

                # 3) uplink compression
                if spec.uplink == "ef21":
                    est = client_estimator[j][n].to(device)
                    delta = target - est
                    alpha = delta.abs().mean()
                    sign_delta = torch.sign(delta)
                    est.add_(alpha * sign_delta)
                    payload[n] = sign_delta.to(buffer_device)
                    scales[n] = alpha.to(buffer_device)
                    client_estimator[j][n] = est.to(buffer_device)
                elif spec.uplink == "sign_mv":
                    payload[n] = torch.sign(target).to(buffer_device)
                else:
                    payload[n] = target.detach().clone().to(buffer_device)

                if spec.client_momentum:
                    # Persist the momentum *buffer*, not the (possibly Nesterov)
                    # look-ahead direction derived from it.
                    client_momentum[j][n] = buf.to(buffer_device)

            uplink_payloads.append(payload)
            uplink_scales.append(scales)

        # ---------------- server -----------------------------------------
        with torch.no_grad():
            for n in matrix_names:
                # 1) aggregate the uplink
                if spec.uplink == "ef21":
                    agg = sum(uplink_payloads[j][n].to(device) * uplink_scales[j][n].to(device)
                              for j in range(num_clients)) / num_clients
                    server_estimator[n].add_(agg)
                    aggregate = server_estimator[n]
                elif spec.uplink == "sign_mv":
                    aggregate = torch.sign(
                        sum(uplink_payloads[j][n].to(device) for j in range(num_clients)))
                else:
                    aggregate = sum(uplink_payloads[j][n].to(device)
                                    for j in range(num_clients)) / num_clients

                # 2) server-side LMO, when the sign acted before the oracle
                D = muon_lmo(aggregate, ns_steps=ns_steps, dtype=lmo_dtype) \
                    if spec.lmo == "server" else aggregate

                # 3) the step, on X (ef21p) or directly on the broadcast model
                if spec.server == "adam":
                    params[n].grad = D.clone()
                    continue

                target_tensor = server_exact[n] if spec.needs_exact_model else params[n].data
                if weight_decay != 0:
                    target_tensor.mul_(1.0 - current_lr * weight_decay)

                if spec.downlink == "sign":
                    # Both sides apply the same 1-bit step, so the models agree.
                    target_tensor.add_(torch.sign(D), alpha=-current_lr)
                else:
                    target_tensor.add_(D, alpha=-current_lr)

                # 4) downlink error feedback: broadcast a scaled sign of X - W
                if spec.downlink == "ef21p":
                    shift = server_exact[n] - params[n].data
                    params[n].data.add_(shift.abs().mean() * torch.sign(shift))

        if server_adam is not None:
            server_adam.step()
            server_adam.zero_grad(set_to_none=True)

        # ---------------- auxiliary parameters (AdamW, uncompressed) -------
        if adamw is not None:
            adamw.zero_grad(set_to_none=True)
            for n in aux_names:
                if n in aux_grads_sum:
                    params[n].grad = (aux_grads_sum[n] / num_clients).clone()
            adamw.step()

        elapsed = time.perf_counter() - t0
        if r % eval_freq == 0 or r == rounds:
            _evaluate(r, elapsed)
        elif verbose:
            print(f"\rRound {r} | {elapsed:.2f}s (no eval)", end="", flush=True)

    # Leave the caller holding the exact model: it is the iterate the theory
    # bounds, and the one whose accuracy is reported.
    if server_exact:
        with torch.no_grad():
            for n, X in server_exact.items():
                params[n].data.copy_(X)

    if verbose:
        print()
    return history
