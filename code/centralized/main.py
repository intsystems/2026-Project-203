"""Entry point for the centralized experiments.

    python3 -m main --dataset cifar10 --model resnet18 --optimizer signmuon \
        --data data --device cuda:0 --epochs 75 --seed 0

Results go to ``results/centralized/<run_name>/seed<seed>/metrics.json``. The seed is part of
the path, so a multi-seed sweep is the same command with different ``--seed``
values and nothing gets overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from centralized.train import OPTIMIZER_CHOICES, resolve_optimizer_name, train
from common.utils import results_root, run_dir, save_run, seed_everything


@dataclass
class RunConfig:
    dataset: str
    model: str
    optimizer: str
    epochs: int
    batch_size: int
    lr: float
    lr_aux: float
    momentum: float
    nesterov: bool
    lambda_mult: float
    ns_steps: int
    lmo_dtype: str
    head_adamw: str
    weight_decay: float
    seed: int
    device: str
    data: str
    download: bool
    run_name: str


def get_params() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    p.add_argument("--model", type=str, default="cnn2", choices=["cnn2", "resnet9", "resnet18"])
    p.add_argument("--optimizer", type=str, default="signmuon",
                   choices=OPTIMIZER_CHOICES + ["ef_usignmuon", "ef_udsignmuon"])
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--download", action="store_true", help="Download dataset if missing")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-aux", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--weight-decay", type=float, default=5e-4)

    p.add_argument("--lambda-mult", type=float, default=1.0, help="Step-size multiplier")
    p.add_argument("--ns-steps", type=int, default=5, help="Newton-Schulz iterations")
    p.add_argument("--lmo-dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"],
                   help="Working precision of the Newton-Schulz iteration")
    p.add_argument("--n-head-tensors", type=int, default=2,
                   help="How many trailing tensors count as the classification head")
    p.add_argument("--head-adamw", type=str, default="auto",
                   choices=["auto", "always", "never"],
                   help="auto: AdamW auxiliary group for the LMO methods only "
                        "(reproduces the paper's numbers); always: for every method, "
                        "including the SGD/Adam/SignSGD baselines (strictly "
                        "apples-to-apples); never: no auxiliary group")

    p.add_argument("--run-name", type=str, default="",
                   help="Folder under results/centralized/ ; auto-generated if empty")
    return p


def main() -> None:
    args = get_params().parse_args()
    seed_everything(args.seed)

    args.optimizer = resolve_optimizer_name(args.optimizer)
    if args.dataset == "mnist":
        args.weight_decay = 0.0

    if not args.run_name:
        args.run_name = f"{args.dataset}_{args.model}_{args.optimizer}"

    config = RunConfig(
        dataset=args.dataset,
        model=args.model,
        optimizer=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_aux=args.lr_aux,
        momentum=args.momentum,
        nesterov=bool(args.nesterov),
        lambda_mult=args.lambda_mult,
        ns_steps=args.ns_steps,
        lmo_dtype=args.lmo_dtype,
        head_adamw=args.head_adamw,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        data=args.data,
        download=bool(args.download),
        run_name=args.run_name,
    )

    print(f"Training {args.optimizer} on {args.dataset}/{args.model} (seed {args.seed})...")
    model, history = train(args)

    out = run_dir(results_root() / "centralized", args.run_name, args.seed)
    save_run(out, config, history, model=model.to("cpu"))
    print(f"Saved run to: {out}")
    print(f"Final test accuracy: {history.last('test_acc'):.2f}%")


if __name__ == "__main__":
    main()
