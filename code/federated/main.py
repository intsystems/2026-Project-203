"""Entry point for the federated experiments.

    python3 -m federated_main --model cnn2 --dataset cifar10 \
        --algorithm signmuon --rounds 2000 --n_parties 10 --n_steps 3 \
        --batch_size 64 --device cuda:0 --eval_freq 100 --seed 0

Results go to ``results/federated/<run_name>/seed<seed>/metrics.json``. The seed is
part of the path, so a multi-seed sweep is just the same command with different
``--seed`` values and nothing gets overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from federated.algorithms import METHODS, resolve_method, run_federated
from federated.data import get_federated_loaders, load_full_datasets, partition_data
from common.models import CNN2, ResNet9, ResNet18
from common.utils import (resolve_device, results_root, run_dir, save_run,
                          seed_everything)

# The ten methods, in the paper's order: six proposed, then the references.
ALGORITHM_CHOICES = [
    "signmuon", "ef21signmuon", "muonusign", "muonsign",
    "ef21muonusign", "ef21muonsign",
    "muon", "signsgd", "sgd", "adam",
    # legacy spellings, accepted so older commands keep working
    "signmuon_cl", "signmuon_ef_21", "signmuon_ef_ud",
]


@dataclass
class FederatedConfig:
    dataset: str
    model: str
    algorithm: str
    rounds: int
    n_parties: int
    n_steps: int
    batch_size: int
    lr: float
    lr_aux: float
    momentum: float
    nesterov: bool
    partition: str
    beta: float
    ns_steps: int
    seed: int
    device: str
    datadir: str
    run_name: str
    eval_freq: int
    weight_decay: float
    eps: float
    cosine_schedule: bool
    freeze_bn_stats: bool
    lmo_dtype: str


def get_params() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Federated Learning Experiment")

    # --- general ---------------------------------------------------------
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--model", type=str, default="cnn2", choices=["resnet9", "cnn2", "resnet18"])
    p.add_argument("--algorithm", type=str, default="signmuon", choices=ALGORITHM_CHOICES)
    p.add_argument("--data", type=str, default="./data_federated")
    p.add_argument("--download", action="store_true", help="Download dataset if missing")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)

    # --- federated hyperparameters ---------------------------------------
    p.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    p.add_argument("--n_parties", type=int, default=10, help="Number of clients")
    p.add_argument("--n_steps", type=int, default=5, help="Local accumulation steps per round")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-aux", type=float, default=1e-3,
                   help="Learning rate of the auxiliary AdamW (biases, BatchNorm, head)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true", help="Nesterov look-ahead momentum")
    p.add_argument("--no-cosine", action="store_true",
                   help="Disable the cosine learning-rate schedule (on by default, "
                        "uniformly for every method)")

    # --- Muon / LMO ------------------------------------------------------
    p.add_argument("--ns_steps", type=int, default=5, help="Newton-Schulz steps for the LMO")
    p.add_argument("--lmo-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float32"],
                   help="Working precision of the Newton-Schulz iteration")

    # --- partitioning ----------------------------------------------------
    p.add_argument("--partition", type=str, default="homo", choices=["homo", "noniid-labeldir"])
    p.add_argument("--beta", type=float, default=0.5, help="Dirichlet concentration parameter")

    # --- system ----------------------------------------------------------
    p.add_argument("--eval_freq", type=int, default=1, help="Evaluate every N rounds")
    p.add_argument("--run_name", type=str, default="", help="Auto-generated if empty")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--eps", type=float, default=1e-8, help="Epsilon for Adam/AdamW")
    p.add_argument("--live-bn-stats", action="store_true",
                   help="Let BatchNorm running statistics update during local "
                        "gradient accumulation (they are frozen by default, which "
                        "is what the reported numbers used)")
    return p


def build_model(dataset: str, model_name: str) -> torch.nn.Module:
    out_dim = 10
    in_ch = 3 if dataset == "cifar10" else 1
    size = 32 if dataset == "cifar10" else 28

    if model_name == "resnet18":
        return ResNet18(in_channels=in_ch, num_classes=out_dim)
    if model_name == "resnet9":
        return ResNet9(num_classes=out_dim)
    return CNN2(in_channels=in_ch, input_size=size, out_dim=out_dim)


def main() -> None:
    args = get_params().parse_args()

    seed_everything(args.seed)

    if args.dataset == "mnist" and args.model == "resnet9":
        raise ValueError("ResNet9 is hardcoded for 3 input channels (CIFAR); use cnn2 for MNIST.")
    if args.dataset == "mnist":
        args.weight_decay = 0.0

    method, _ = resolve_method(args.algorithm)

    if not args.run_name:
        args.run_name = (f"fed_{args.dataset}_{method}_{args.partition}_{args.model}"
                         f"_r{args.rounds}_c{args.n_parties}_s{args.n_steps}")

    config = FederatedConfig(
        dataset=args.dataset,
        model=args.model,
        algorithm=method,               # store the canonical name, not the alias
        rounds=args.rounds,
        n_parties=args.n_parties,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_aux=args.lr_aux,
        momentum=args.momentum,
        nesterov=bool(args.nesterov),
        partition=args.partition,
        beta=args.beta,
        ns_steps=args.ns_steps,
        seed=args.seed,
        device=args.device,
        datadir=args.data,
        run_name=args.run_name,
        eval_freq=args.eval_freq,
        weight_decay=args.weight_decay,
        eps=args.eps,
        cosine_schedule=not args.no_cosine,
        freeze_bn_stats=not args.live_bn_stats,
        lmo_dtype=args.lmo_dtype,
    )

    # 1) data
    train_ds, test_ds, y_train, y_test = load_full_datasets(
        args.dataset, args.data, download=args.download)
    train_map, test_map = partition_data(
        y_train, y_test, args.partition, args.n_parties, beta=args.beta)
    train_loaders, test_loaders = get_federated_loaders(
        train_ds, test_ds, train_map, test_map, args.n_parties, args.batch_size,
        seed=args.seed)

    # 2) model
    global_model = build_model(args.dataset, args.model)

    # 3) train
    device = resolve_device(args.device)
    print(f"Starting {method} on {device} (seed {args.seed})...")
    history = run_federated(
        method,
        global_model,
        train_loaders,
        test_loaders,
        rounds=args.rounds,
        n_steps=args.n_steps,
        lr=args.lr,
        lr_aux=args.lr_aux,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
        ns_steps=args.ns_steps,
        eval_freq=args.eval_freq,
        device=device,
        cosine_schedule=config.cosine_schedule,
        freeze_bn_stats=config.freeze_bn_stats,
        adam_eps=args.eps,
        lmo_dtype=getattr(torch, args.lmo_dtype),
    )

    out = run_dir(results_root() / "federated", args.run_name, args.seed)
    save_run(out, config, history, model=global_model.to("cpu"))
    print(f"Results saved to: {out}")
    print(f"Final test accuracy: {history.last('test_acc'):.2f}%")


if __name__ == "__main__":
    main()
