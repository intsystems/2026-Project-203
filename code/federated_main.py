import argparse
import torch
import numpy as np
import random
import json
import shutil
from pathlib import Path
from dataclasses import asdict, dataclass
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import ResNet18, ResNet9, CNN2
from federated_algorithms import (
    federated_sgd,
    federated_signsgd,
    federated_muon,
    # federated_signmuon,
    # federated_signmuon_ef,
    federated_ef21_muon,
    federated_ef_ud_signmuon,
    federated_signmuon_client,
    federated_adam,
)
from federated_dataloader import *


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
    lambda_mult: float
    norm_weight: bool

def get_params():
    p = argparse.ArgumentParser(description="Federated Learning Experiment")
    
    # --- Основные настройки ---
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--model", type=str, default="cnn2", choices=["resnet9", "cnn2", "resnet18"])
    p.add_argument("--algorithm", type=str, default="signmuon",
                    choices=["signmuon_ef_21", "signmuon_cl",  "muon", "signsgd", "sgd", "adam"])
    p.add_argument("--data", type=str, default="./data_federated")
    p.add_argument("--download", action="store_true", help="Download dataset if missing")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)

    # --- Гиперпараметры федеративки ---
    p.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    p.add_argument("--n_parties", type=int, default=10, help="Number of clients")
    p.add_argument("--n_steps", type=int, default=5, help="Steps per client per round")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-aux", type=float, default=1e-3, help="Learning rate for auxiliary AdamW optimizer")
    p.add_argument("--momentum", type=float, default=0.9)
    
    # --- Параметры для Muon / SignMuon ---
    p.add_argument("--ns_steps", type=int, default=5, help="Newton-Schulz steps for Muon")
    p.add_argument("--lambda_mult", type=float, default=1.0, help="Multiplier for Muon step size")
    p.add_argument("--no_norm_weight", action="store_true", help="Disable weight normalization in Muon/SignMuon")

    # --- Partitioning ---
    p.add_argument("--partition", type=str, default="homo", choices=["homo", "noniid-labeldir"])
    p.add_argument("--beta", type=float, default=0.5, help="Dirichlet concentration parameter")
    
    # --- Системные настройки ---
    p.add_argument("--eval_freq", type=int, default=1, help="Evaluate every N rounds")
    p.add_argument("--run_name", type=str, default="", help="Auto-generated if empty")

    # --- Параметры для Adam ---
    p.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for Adam")
    p.add_argument("--eps", type=float, default=1e-8, help="Epsilon for Adam")
    
    return p

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_federated_run(
        model, 
        history, 
        config: FederatedConfig
):
    saves_root = Path(__file__).resolve().parent / "saves_federated"
    run_dir = saves_root / config.run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем веса глобальной модели
    torch.save({
        "model_state_dict": model.state_dict(), 
        "config": asdict(config)}, run_dir / "model_global.pt"
    )

    # Сохраняем метрики
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "history": history}, f, indent=2)
    
    print(f"Results saved to: {run_dir}")
    return run_dir

def main() -> None:
    args = get_params().parse_args()

    seed_everything(args.seed)

    if args.dataset == "mnist" and args.model == "resnet9":
        raise ValueError("ResNet9 is hardcoded for 3 input channels (CIFAR). It cannot be used with MNIST.")
    
    if args.dataset == "mnist":
        args.weight_decay = 0.0 

    if not args.run_name:
        args.run_name = f"fed_{args.dataset}_{args.algorithm}_{args.partition}_{args.model}_r{args.rounds}_c{args.n_parties}_s{args.n_steps}"

    config = FederatedConfig(
        dataset=args.dataset, 
        model=args.model, 
        algorithm=args.algorithm,
        rounds=args.rounds, 
        n_parties=args.n_parties, 
        n_steps=args.n_steps,
        batch_size=args.batch_size, 
        lr=args.lr,
        lr_aux=args.lr_aux, 
        momentum=args.momentum,
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
        lambda_mult=args.lambda_mult,
        norm_weight=not args.no_norm_weight
    )

    # 1. Разбиение данных
    train_ds, test_ds, y_train, y_test = load_full_datasets(args.dataset, args.data, download=args.download)
    
    train_map, test_map = partition_data(
        y_train, 
        y_test, 
        args.partition, 
        args.n_parties, 
        beta=args.beta
    )

    # if args.dataset == "cifar10":
    #     train_ds.transform = get_cifar10_transforms(train=True)
    #     test_ds.transform = get_cifar10_transforms(train=False)
    # else:
    #     train_ds.transform = get_mnist_transform()
    #     test_ds.transform = get_mnist_transform()

    train_loaders, test_loaders = get_federated_loaders(
        train_ds, 
        test_ds, 
        train_map, 
        test_map, 
        args.n_parties, 
        args.batch_size
    )
    
    # 2. Инициализация модели
    out_dim = 10
    in_ch = 3 if args.dataset == "cifar10" else 1
    size = 32 if args.dataset == "cifar10" else 28

    if args.model == "resnet18":
        global_model = ResNet18(in_channels=in_ch, num_classes=out_dim)
        print("global_model = ResNet18")
    elif args.model == "resnet9":
        global_model = ResNet9(num_classes=out_dim) 
        print("global_model = ResNet9")
    else:
        global_model = CNN2(in_channels=in_ch, input_size=size, out_dim=out_dim)
    
    # 3. Запуск алгоритма
    print(f"Starting {args.algorithm.upper()} on {args.device}...")

    # if args.algorithm == 'signmuon_ef':
    #     accs, losses = federated_signmuon_ef(
    #         global_model, train_loaders, args.n_parties, args.rounds,
    #         args.n_steps, args.lr, args.lr_aux, test_loaders,
    #         ns_steps=args.ns_steps,
    #         eval_freq=args.eval_freq,
    #         momentum=args.momentum,
    #         device=args.device
    #     )
    
    if args.algorithm == 'signmuon_ef_21':
        accs, losses = federated_ef21_muon(
            global_model, train_loaders, args.n_parties, args.rounds,
            args.n_steps, args.lr, args.lr_aux, test_loaders,
            ns_steps=args.ns_steps,
            eval_freq=args.eval_freq,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            device=args.device
        )
    
    elif args.algorithm == 'signmuon_ef_ud':
        accs, losses = federated_ef_ud_muon(
            global_model, train_loaders, args.n_parties, args.rounds,
            args.n_steps, args.lr, args.lr_aux, test_loaders,
            ns_steps=args.ns_steps,
            eval_freq=args.eval_freq,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            device=args.device
        )

    # Federated SignMuon (majority-vote + client momentum, Algorithm 2).
    # 'signmuon' is the canonical name used in the README and tables;
    # 'signmuon_cl' is kept as a backward-compatible alias.
    elif args.algorithm in ('signmuon', 'signmuon_cl'):
        accs, losses = federated_signmuon_client(
            global_model, train_loaders, args.n_parties, args.rounds,
            args.n_steps, args.lr, args.lr_aux, test_loaders,
            ns_steps=args.ns_steps,
            eval_freq=args.eval_freq,
            momentum=args.momentum,
            # weight_decay=args.weight_decay,
            device=args.device
        )

    elif args.algorithm == 'muon':
        accs, losses = federated_muon(
            global_model, train_loaders, args.n_parties, args.rounds,
            args.n_steps, args.lr, args.lr_aux, test_loaders,
            ns_steps=args.ns_steps,
            eval_freq=args.eval_freq,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            device=args.device
        )
        
    elif args.algorithm == 'signsgd':
        accs, losses = federated_signsgd(
            global_model, train_loaders, args.n_parties, args.rounds, 
            args.n_steps, args.lr, test_loaders, 
            momentum=args.momentum, 
            eval_freq=args.eval_freq, 
            device=args.device
        )
        
    elif args.algorithm == 'adam':
        accs, losses = federated_adam(
            global_model, train_loaders, args.n_parties, args.rounds, 
            args.n_steps, args.lr, test_loaders, 
            weight_decay=args.weight_decay, 
            eps=args.eps, 
            eval_freq=args.eval_freq, 
            device=args.device
        )
        
    elif args.algorithm == 'sgd':
        accs, losses = federated_sgd(
            global_model, train_loaders, args.n_parties, args.rounds, 
            args.n_steps, args.lr, test_loaders, 
            weight_decay=args.weight_decay,
            momentum=args.momentum, 
            eval_freq=args.eval_freq, 
            device=args.device
        )
        
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    global_model = global_model.to("cpu")
    history = {"test_acc": accs, "test_loss": losses}
    save_federated_run(global_model, history, config)

if __name__ == "__main__":
    main()