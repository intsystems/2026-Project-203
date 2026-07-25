"""Centralized training for CIFAR-10 / MNIST, shared by every optimizer.

Consolidates the former ``train_cifar.py`` and ``train_mnist.py`` (which were
~90% identical but had drifted: only one of them had the cosine schedule, only
one excluded the classification head from the LMO rule, and one returned a
*string* on an unknown optimizer name, which then blew up on tuple unpacking).

Parameter routing follows the paper: matrix parameters (``ndim >= 2``, excluding
the classification head) get the LMO/sign rule; biases, BatchNorm scales and the
head get AdamW. See ``--head-adamw`` for how the non-LMO baselines are treated.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn

from centralized.data import cifar10_loaders, mnist_loaders
from common.models import CNN2, ResNet9, ResNet18
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
from common.utils import History, resolve_device, split_param_names

# Methods implemented in optimizers.py, i.e. everything that follows the paper's
# matrix-parameter template. The remaining choices (sgd, adam) come from torch.
LMO_FAMILY = {
    "signmuon": SignMuon,
    "ef21signmuon": EF21SignMuon,
    "muonusign": MuonUSign,
    "muonsign": MuonSign,
    "ef21muonusign": EF21MuonUSign,
    "ef21muonsign": EF21MuonSign,
    "muon": Muon,
    "signsgd": SignSGD,
}

OPTIMIZER_CHOICES = list(LMO_FAMILY) + ["sgd", "adam"]

# ``signsgd`` is in LMO_FAMILY (it is implemented here) but does not use the LMO,
# so under ``--head-adamw auto`` it is treated like the other baselines: one rule
# for every parameter, which is the algorithm as published.
MATRIX_RULE_METHODS = {"signmuon", "ef21signmuon", "muonusign", "muonsign",
                       "ef21muonusign", "ef21muonsign", "muon"}

# Legacy spellings.
ALIASES = {
    "ef_usignmuon": "ef21muonusign",
    "ef_udsignmuon": "ef21muonsign",
}


def resolve_optimizer_name(name: str) -> str:
    key = name.strip().lower().replace("-", "")
    key = ALIASES.get(key, key)
    if key not in OPTIMIZER_CHOICES:
        raise ValueError(f"Unknown optimizer {name!r}. Available: {OPTIMIZER_CHOICES}")
    return key


# --------------------------------------------------------------------------
# Epoch loops
# --------------------------------------------------------------------------


def train_epoch(model, loader, optimizers, device) -> Tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0.0

    opt_main, opt_aux = optimizers
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for opt in (opt_main, opt_aux):
            if opt is not None:
                opt.zero_grad(set_to_none=True)

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        for opt in (opt_main, opt_aux):
            if opt is not None:
                opt.step()

        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = correct = total = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item()
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, opt_main=None) -> Tuple[float, float]:
    """Evaluate, exposing the *exact* model for the bidirectional EF method.

    ``EF21MuonSign`` keeps the broadcast model ``W`` in ``p.data`` and the exact
    server model ``X`` in its state; ``X`` is the iterate the convergence theory
    bounds, so metrics are computed there.
    """
    using_exact = getattr(opt_main, "using_exact", None)
    if using_exact is None:
        return eval_epoch(model, loader, device)
    with using_exact():
        return eval_epoch(model, loader, device)


# --------------------------------------------------------------------------
# Optimizer construction
# --------------------------------------------------------------------------


def build_optimizers(model, args):
    """Return ``(opt_main, opt_aux)``.

    ``--head-adamw``:

    * ``auto`` (default) -- the LMO methods put matrix parameters under their own
      rule and biases/BatchNorm/head under AdamW; ``sgd``, ``adam`` and
      ``signsgd`` apply their single rule to every parameter, as published. This
      reproduces the numbers currently in the paper.
    * ``always`` -- every method, baselines included, gets the AdamW auxiliary
      group, so the *only* difference between runs is the matrix rule. This is
      the strictly apples-to-apples setting.
    * ``never`` -- no auxiliary group at all.
    """
    name = resolve_optimizer_name(args.optimizer)
    n_head = getattr(args, "n_head_tensors", 2)
    matrix_names, aux_names = split_param_names(model, n_head)

    mode = getattr(args, "head_adamw", "auto")
    if mode == "auto":
        split = name in MATRIX_RULE_METHODS
    elif mode == "always":
        split = True
    else:
        split = False

    named = dict(model.named_parameters())
    if split:
        main_params = [named[n] for n in matrix_names]
        aux_params = [named[n] for n in aux_names]
    else:
        main_params = [p for p in model.parameters() if p.requires_grad]
        aux_params = []

    common = dict(
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
        lambda_mult=getattr(args, "lambda_mult", 1.0),
        ns_steps=getattr(args, "ns_steps", 5),
        lmo_dtype=getattr(torch, getattr(args, "lmo_dtype", "bfloat16")),
    )

    if name in LMO_FAMILY:
        opt_main = LMO_FAMILY[name](main_params, **common)
    elif name == "sgd":
        opt_main = torch.optim.SGD(main_params, lr=args.lr, momentum=args.momentum,
                                   nesterov=args.nesterov, weight_decay=args.weight_decay)
    elif name == "adam":
        opt_main = torch.optim.Adam(main_params, lr=args.lr, weight_decay=args.weight_decay)
    else:                                                    # unreachable
        raise ValueError(f"Unhandled optimizer {name!r}")

    opt_aux = (torch.optim.AdamW(aux_params, lr=args.lr_aux, weight_decay=0.0)
               if aux_params else None)
    return opt_main, opt_aux


def build_model(dataset: str, model_name: str) -> nn.Module:
    if dataset == "cifar10":
        if model_name == "resnet18":
            return ResNet18(in_channels=3, num_classes=10)
        if model_name == "resnet9":
            return ResNet9(num_classes=10)
        return CNN2(in_channels=3, input_size=32, out_dim=10)
    if model_name in ("resnet9",):
        raise ValueError("ResNet9 is hardcoded for 3 input channels (CIFAR).")
    if model_name == "resnet18":
        return ResNet18(in_channels=1, num_classes=10)
    return CNN2(in_channels=1, input_size=28, n_kernels=32, out_dim=10)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train(args) -> Tuple[nn.Module, History]:
    """Train for ``args.epochs`` and return ``(model, History)``.

    The history is step-indexed by epoch, with epoch 0 recording the metrics of
    the untrained model.
    """
    device = resolve_device(args.device)
    seed = getattr(args, "seed", None)

    if args.dataset == "cifar10":
        train_dl, test_dl = cifar10_loaders(
            args.data, batch_size=args.batch_size, download=args.download, seed=seed)
    else:
        train_dl, test_dl = mnist_loaders(
            args.data, batch_size=args.batch_size, download=args.download, seed=seed)

    model = build_model(args.dataset, args.model).to(device)
    opt_main, opt_aux = build_optimizers(model, args)

    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=args.epochs)
    scheduler_aux = (torch.optim.lr_scheduler.CosineAnnealingLR(opt_aux, T_max=args.epochs)
                     if opt_aux is not None else None)

    # Accuracies are recorded in *percent*, matching the federated driver, so that
    # aggregate.py can compare and average the two without unit surprises.
    history = History()
    loss_tr, acc_tr = evaluate(model, train_dl, device, opt_main)
    loss_te, acc_te = evaluate(model, test_dl, device, opt_main)
    history.record(0, train_loss=loss_tr, train_acc=100 * acc_tr,
                   test_loss=loss_te, test_acc=100 * acc_te)
    print(f"Epoch 0/{args.epochs} | train loss {loss_tr:.4f}, acc {100 * acc_tr:.2f}% "
          f"| test loss {loss_te:.4f}, acc {100 * acc_te:.2f}%")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss_tr, acc_tr = train_epoch(model, train_dl, (opt_main, opt_aux), device)

        scheduler_main.step()
        if scheduler_aux is not None:
            scheduler_aux.step()

        loss_te, acc_te = evaluate(model, test_dl, device, opt_main)
        history.record(epoch, train_loss=loss_tr, train_acc=100 * acc_tr,
                       test_loss=loss_te, test_acc=100 * acc_te)
        print(f"Epoch {epoch}/{args.epochs} | {time.perf_counter() - t0:.2f}s "
              f"| train loss {loss_tr:.4f}, acc {100 * acc_tr:.2f}% "
              f"| test loss {loss_te:.4f}, acc {100 * acc_te:.2f}%")

    # Leave the caller holding the exact model, not the compressed broadcast one.
    if hasattr(opt_main, "restore_exact"):
        opt_main.restore_exact()

    return model, history
