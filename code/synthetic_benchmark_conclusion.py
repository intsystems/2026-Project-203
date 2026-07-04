
import torch
import torch.nn as nn
import time
import math
import os
import json
import numpy as np
import copy
from optimizers import Muon, SignMuon, SignSGD, EFSignMuon
from torch.optim import SGD, Adam


def generate_psd_matrix(dim, device='cpu'):
    """
    Генерирует симметричную положительно полуопределенную матрицу,
    спектр которой равномерно распределен в (0, 1).
    """
    eigenvalues = torch.rand(dim, device=device)
    random_matrix = torch.randn(dim, dim, device=device)
    Q, _ = torch.linalg.qr(random_matrix)
    # M = Q * Lambda * Q^T
    return Q @ torch.diag(eigenvalues) @ Q.T


class QuadraticMatrixProblem(nn.Module):
    def __init__(self, m=500, n=500, device='cpu'):
        super().__init__()
        self.m, self.n = int(m), int(n)
        self.X = nn.Parameter(torch.randn(self.m, self.n, device=device) * 0.1)

    def forward(self, M, N):
        # F(X) = 1/2 * <X, M X N>
        MXN = M @ self.X @ N
        loss = 0.5 * torch.sum(self.X * MXN)
        return loss


def run_optimizer(opt_name, opt_class, opt_kwargs, M, N, m=500, n=500, target_loss=0.001, max_iters=5000, device='cuda'):
    torch.manual_seed(42)
    model = QuadraticMatrixProblem(m=int(m), n=int(n), device=device)
    
    clean_kwargs = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in opt_kwargs.items()}
    
    optimizer = opt_class(model.parameters(), **clean_kwargs)
    
    start_time = time.time()
    iters_to_converge = max_iters
    
    loss_history = []
    grad_norm_history = []
    
    for i in range(max_iters):
        optimizer.zero_grad()
        loss = model(M, N)
        loss.backward()
        
        l_val = loss.item()
        # Норма Фробениуса полного градиента
        g_norm = model.X.grad.norm().item() 
        
        loss_history.append(l_val)
        grad_norm_history.append(g_norm)
        
        if l_val <= target_loss:
            iters_to_converge = i + 1
            break
            
        optimizer.step()

    elapsed_time = time.time() - start_time
    final_loss = loss_history[-1]

    safe_opt_name = opt_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_dir = os.path.join("project/saves_synthetic_001", safe_opt_name)
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = {
        "optimizer": opt_name,
        "kwargs": opt_kwargs,
        "iters_to_converge": iters_to_converge,
        "final_loss": final_loss,
        "time_seconds": elapsed_time,
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history
    }
    
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    return iters_to_converge, final_loss, elapsed_time

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    m, n = 500, 500
    target_loss = 0.001
    max_iters = 5000

    torch.manual_seed(1337)
    M = generate_psd_matrix(m, device=device)
    N = generate_psd_matrix(n, device=device)

    experiments = [
        # ("Muon", Muon, {"lr": 0.0065, "momentum": 0.1, "norm_weight": False}),
        # ("SignMuon", SignMuon, {"lr": 0.0002, "momentum": 0.2, "norm_weight": False}),
        # ("SignSGD", SignSGD, {"lr": 0.00015, "momentum": 0.8}),
        # ("SGD", SGD, {"lr": 0.1, "momentum": 0.95}),
        ("EFSignMuon", EFSignMuon, {"lr": 0.0033, "momentum": 0.1}),
        # ("Adam", Adam, {"lr": 0.07})
    ]
    
    # os.makedirs("project/saves_synthetic_001", exist_ok=True)
    print(f"{'Optimizer':<15} | {'Iters to 0.001':<18} | {'Final Loss':<12} | {'Time (s)':<10}")
    print("-" * 65)
    for name, opt_class, kwargs in experiments:
        iters, f_loss, t = run_optimizer(name, opt_class, kwargs, M, N, m, n, target_loss, max_iters, device)
        if iters < max_iters:
            print(f"{name:<15} | {iters:<18} | {f_loss:.6f}     | {t:.2f}s")
        else:
            print(f"{name:<15} | > {max_iters:<16} | {f_loss:.6f}     | {t:.2f}s")

    print(f"\nBenchmark finished! Results saved in './project/saves_synthetic_001/'")