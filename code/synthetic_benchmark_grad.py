
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
    
    metrics = {
        "kwargs": opt_kwargs,
        "iters_to_converge": iters_to_converge,
        "final_loss": final_loss,
        "time_seconds": elapsed_time,
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history
    }
    del model
    del optimizer
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    m, n = 500, 500
    target_loss = 0.001
    max_iters = 5000

    torch.manual_seed(1337)
    M = generate_psd_matrix(m, device=device)
    N = generate_psd_matrix(n, device=device)


    momentums = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    lr_range_muon = [float(x) for x in np.arange(0.005, 0.021, 0.001)]

    lr_range = [float(x) for x in np.arange(0.0001, 0.001, 0.0001)]

    lr_range_ef = [float(x) for x in np.arange(0.001, 0.005, 0.0005)]
    
    lr_range_ef = [float(x) for x in np.arange(0.003, 0.0035, 0.0001)]

    signsgd_lr_range = [float(x) for x in np.arange(5e-5, 2.1e-4, 1e-5)]

    sgd_lr_range = [float(x) for x in np.arange(0.01, 0.11, 0.01)]

    grids = {
            # "Muon": (Muon, [
            #     {"lr": float(lr), "momentum": float(mom), "norm_weight": False}
            #     for lr in lr_range_muon for mom in momentums
            # ]),
            "EFSignMuon": (EFSignMuon, [
                {"lr": float(lr), "momentum": float(mom)}
                for lr in lr_range_ef for mom in momentums
            ]),
            # "SignMuon": (SignMuon, [
            #     {"lr": float(lr), "momentum": float(mom), "norm_weight": False}
            #     for lr in lr_range for mom in momentums
            # ]),
            # "SignSGD": (SignSGD, [
            #     {"lr": float(lr), "momentum": float(mom)}
            #     for lr in signsgd_lr_range for mom in momentums
            # ]),
            # "SGD": (SGD, [
            #     {"lr": float(lr), "momentum": float(mom)}
            #     for lr in sgd_lr_range for mom in momentums
            # ]),
            # "Adam": (Adam, [
            #     {"lr": float(lr)}
            #     for lr in sgd_lr_range
            # ]),
        }


    os.makedirs("saves_synthetic_01", exist_ok=True)
    print(f"{'Optimizer':<15} | {'Best Iters':<12} | {'Best Loss':<12} | {'Best Params'}")
    print("-" * 80)

    for opt_name, (opt_class, param_grid) in grids.items():
        print(f"Starting Grid Search for {opt_name} ({len(param_grid)} configs)...")
        
        best_metrics = None
        
        for kwargs in param_grid:
            metrics = run_optimizer(opt_name, opt_class, kwargs, M, N, m, n, target_loss, max_iters, device)
            print(f"lr: {kwargs['lr']}, mom: {kwargs.get('momentum', 0)}, Iters: {metrics['iters_to_converge']}, Loss: {metrics['final_loss']:.6f}")
            
            is_better = False
            if best_metrics is None:
                is_better = True
            else:
                if metrics["iters_to_converge"] < best_metrics["iters_to_converge"]:
                    is_better = True
                elif metrics["iters_to_converge"] == best_metrics["iters_to_converge"]:
                    if metrics["final_loss"] < best_metrics["final_loss"]:
                        is_better = True
                        
            if is_better:
                best_metrics = metrics
                print(f"  [New Best] lr: {kwargs['lr']}, mom: {kwargs.get('momentum', 0)}, Iters: {metrics['iters_to_converge']}, Loss: {metrics['final_loss']:.6f}")

        if best_metrics is not None:
            safe_opt_name = opt_name.replace(" ", "_").replace("(", "").replace(")", "")
            save_dir = os.path.join("saves_synthetic_01", safe_opt_name)
            os.makedirs(save_dir, exist_ok=True)
            
            best_metrics["optimizer"] = opt_name

            save_dict = copy.deepcopy(best_metrics)
            if "loss_history" in save_dict:
                save_dict["loss_history"] = [float(x) for x in save_dict["loss_history"]]
            if "grad_norm_history" in save_dict:
                save_dict["grad_norm_history"] = [float(x) for x in save_dict["grad_norm_history"]]
            
            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(save_dict, f, indent=4)
            
            iters = best_metrics["iters_to_converge"]
            loss = best_metrics["final_loss"]
            lr_str = f"lr={best_metrics['kwargs']['lr']}"
            mom_str = f"mom={best_metrics['kwargs'].get('momentum', 0)}"
            
            iter_str = str(iters) if iters < max_iters else f"> {max_iters}"
            print(f"==> {opt_name:<11} | {iter_str:<12} | {loss:.6f}     | lr: {lr_str}, mom: {mom_str}\n")