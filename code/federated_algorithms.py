import torch
import torch.optim as optim
import torch.nn as nn
from torch import autograd
from tqdm import tqdm
import copy
from torch.utils.data import ConcatDataset, DataLoader
import os
import torchvision.utils as vutils
from pathlib import Path
from models import *
from data_loader import *
from torchvision import transforms
from sklearn.cluster import KMeans
from contextlib import contextmanager
from optimizers import *
import time
import math


@contextmanager
def disable_bn_running_stats(model):
    """Context manager to temporarily disable BatchNorm running statistics updates."""
    bn_training_flags = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_training_flags[module] = module.training
            module.training = False
            module.eval()
    try:
        yield
    finally:
        for module, was_training in bn_training_flags.items():
            module.training = was_training
            if was_training:
                module.train()

def save_client_images(train_loaders, test_loaders, output_dir='saves/client_images', save_all=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for client_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        print(f"Saving images for client {client_id}")
        
        client_dir = os.path.join(output_dir, f'client_{client_id}')
        train_dir = os.path.join(client_dir, 'train')
        test_dir = os.path.join(client_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for batch_idx, (images, labels) in enumerate(train_loader):
            if save_all:
                for img_idx, (image, label) in enumerate(zip(images, labels)):
                    train_filename = os.path.join(train_dir, 
                        f'batch_{batch_idx}_img_{img_idx}_label_{label}.png')
                    vutils.save_image(image, train_filename)
            else:
                train_filename = os.path.join(train_dir, 
                    f'batch_{batch_idx}_label_{labels[0]}.png')
                vutils.save_image(images[0], train_filename)

        for batch_idx, (images, labels) in enumerate(test_loader):
            if save_all:
                for img_idx, (image, label) in enumerate(zip(images, labels)):
                    test_filename = os.path.join(test_dir, 
                        f'batch_{batch_idx}_img_{img_idx}_label_{label}.png')
                    vutils.save_image(image, test_filename)
            else:
                test_filename = os.path.join(test_dir, 
                    f'batch_{batch_idx}_label_{labels[0]}.png')
                vutils.save_image(images[0], test_filename)

        if save_all:
            print(f"Saved {len(train_loader.dataset)} training images and {len(test_loader.dataset)} test images for client {client_id}")
        else:
            print(f"Saved {len(train_loader)} training batch samples and {len(test_loader)} test batch samples for client {client_id}")


def evaluate_model(global_model, test_loaders, args = None, verbose = False, device=None):
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model.eval()
    global_model.to(device)

    criterion = nn.CrossEntropyLoss(reduction = "sum")

    client_accuracies = []
    total_correct, total_samples = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for client_id, test_loader in enumerate(test_loaders):
            client_correct, client_total = 0, 0
            client_loss = 0.0

            if verbose:
                print(f"\nEvaluating Client {client_id}")

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = global_model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[-1]

                loss = criterion(outputs, labels)
                client_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                client_correct += (predicted == labels).sum().item()
                client_total += labels.size(0)

            client_accuracy = (client_correct / client_total) * 100 if client_total > 0 else 0
            client_accuracies.append(client_accuracy)

            if verbose:
                print(f"Client {client_id} Classification Accuracy: {client_accuracy:.2f}%")

            total_correct += client_correct
            total_samples += client_total
            total_loss += client_loss

    total_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    total_loss = total_loss / total_samples if total_samples > 0 else 0 

    print(f"\rRound Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}")
    
    return client_accuracies, total_accuracy, total_loss



# =========================== SGD (FedSGD) ==============================================
def local_train_sgd(global_model, train_loader, n_steps, momentum_buffer, 
                    weight_decay=1e-4, momentum=0.9, device=None):
    
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    criterion = nn.CrossEntropyLoss()
    accumulated_grad = {}

    train_iter = iter(train_loader)

    with disable_bn_running_stats(local_model):
        for _ in range(n_steps):
            try:
                x_train, y_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_train, y_train = next(train_iter)

            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()

            local_model.zero_grad()
            loss = criterion(local_model(x_train), y_train)
            loss.backward()

            # Накопление градиентов
            for name, param in local_model.named_parameters():
                if param.grad is not None:
                    g = param.grad

                    if weight_decay != 0:
                        g = g.add(param.data, alpha=weight_decay)
                        
                    if name not in accumulated_grad: 
                        accumulated_grad[name] = g.clone()
                    else: 
                        accumulated_grad[name] += g

    momentum_output = {}
    
    # Усредняем градиент и обновляем моментум
    for name, G in accumulated_grad.items():
        G = G / n_steps
        
        if name not in momentum_buffer:
            momentum_buffer[name] = torch.zeros_like(G, device=device)
        else:
            momentum_buffer[name] = momentum_buffer[name].to(device)
            
        # M_t = mu * M_{t-1} + G_t
        momentum_buffer[name] = momentum * momentum_buffer[name] + G
        
        # Возвращаем результат на CPU для экономии памяти
        momentum_output[name] = momentum_buffer[name].cpu()
        momentum_buffer[name] = momentum_buffer[name].cpu()
        
    return momentum_output

def federated_sgd(global_model, train_loaders, num_clients, rounds, n_steps, lr, 
                  test_loaders, eval_freq=1, momentum=0.9, weight_decay=1e-4, device = None):
    """
    Federated SGD.
    
    Args:
        eval_freq: Evaluate model every N rounds (default: 1, i.e., every round). Set to higher value to evaluate less frequently.
    """
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    global_model.train()

    
    client_momentums = [
        {name: torch.zeros_like(param, device='cpu') 
         for name, param in global_model.named_parameters() if param.requires_grad}
        for _ in range(num_clients)
    ]

    round_accuracies = []
    round_losses = []

    # Инициализация 0 эпохи
    _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
    round_accuracies.append(total_accuracy)
    round_losses.append(total_loss)
    print(f"\rRound 0 - Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f} ")

    for r in range(rounds):
        round_start = time.perf_counter() 
        
        # Cosine Annealing
        current_lr = lr * 0.5 * (1 + math.cos(math.pi * r / rounds))
        
        print(f"\rRound {r + 1} | LR: {current_lr:.5f}", end='', flush=True)

        client_momentums_updates = [
            local_train_sgd(global_model, train_loaders[i], n_steps, 
                            momentum_buffer=client_momentums[i], 
                            weight_decay=weight_decay,
                            momentum=momentum, device=device)
            for i in range(num_clients)
        ]

        avg_momentum = {
            name: sum(c[name].to(device) for c in client_momentums_updates) / num_clients
            for name in client_momentums_updates[0].keys()
        }

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if param.requires_grad and name in avg_momentum:
                    param -= current_lr * avg_momentum[name]

        round_elapsed = time.perf_counter() - round_start
        if (r + 1) % eval_freq == 0 or r == rounds - 1:
            _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
            print(f"\rRound {r+1} | {round_elapsed:.2f}s | Acc: {total_accuracy:.2f}%, Loss: {total_loss:.4f}", end='', flush=True)
        else:
            total_accuracy = round_accuracies[-1] if round_accuracies else None
            total_loss = round_losses[-1] if round_losses else None
            print(f"\rRound {r+1} - (skipped evaluation)| {round_elapsed:.2f}s ", end='', flush=True)
        
        round_accuracies.append(total_accuracy)
        round_losses.append(total_loss)

    return round_accuracies, round_losses



# ========================= Sign SGD + momentum (Bernstein 2018) ==========================================
def local_train_signsgd(global_model, train_loader, n_steps, momentum_buffer, momentum=0.9, device=None):
    local_model = copy.deepcopy(global_model)
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    local_model.train()

    criterion = nn.CrossEntropyLoss()

    accumulated_grad = {}
    
    # Use context manager to disable BatchNorm running stats updates during gradient computation
    # This prevents corruption of BatchNorm statistics when using small batches
    with disable_bn_running_stats(local_model):
        for step, (x_train, y_train) in enumerate(train_loader):
            if step >= n_steps: break
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()

            for param in local_model.parameters():
                if param.requires_grad: param.grad = None

            loss = criterion(local_model(x_train), y_train)
            loss.backward()

            for name, param in local_model.named_parameters():
                if param.grad is not None:
                    if name not in accumulated_grad:
                        accumulated_grad[name] = param.grad.clone()
                    else:
                        accumulated_grad[name] += param.grad

    # Normalize by n_steps
    for name in accumulated_grad:
        accumulated_grad[name] = accumulated_grad[name] / n_steps

    # Client-side Momentum: M = mu * M + grad
    # Client-side sign compression: s = sign(M)
    sign_dict = {}
    for name, G in accumulated_grad.items():
        if name not in momentum_buffer:
            momentum_buffer[name] = torch.zeros_like(G, device=device)
        
        momentum_buffer[name] = momentum * momentum_buffer[name] + G
        sign_dict[name] = torch.sign(momentum_buffer[name])
        
    return sign_dict

def federated_signsgd(global_model, train_loaders, num_clients, rounds, n_steps, lr, test_loaders, momentum=0.9, eval_freq=1, device = None):
    """
    Federated SignSGD with momentum.
    """
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    global_model.train()

    client_momentums = [
        {name: torch.zeros_like(param, device=device) 
         for name, param in global_model.named_parameters() if param.requires_grad}
        for _ in range(num_clients)
    ]

    round_accuracies = []
    round_losses = []

    # Init
    _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
    round_accuracies.append(total_accuracy)
    round_losses.append(total_loss)
    print(f"Round 0 - Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f} ")
    
    for r in range(rounds):
        round_start = time.perf_counter() 
        print(f"\rRound {r+1}", end='', flush=True)

        # Get signs from clients (Client computes its own momentum and sends sign)
        client_signs = [
            local_train_signsgd(global_model, train_loaders[i], n_steps, 
                                momentum_buffer=client_momentums[i], momentum=momentum, device=device)
            for i in range(num_clients)
        ]

        # Aggregate signs using majority vote
        majority_sign = {
            name: torch.sign(sum(client_signs[i][name].to(device) for i in range(num_clients)))
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }

        # Server update: just apply the aggregated sign
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if param.requires_grad:
                    param -= lr * majority_sign[name]

        # Evaluate only at specified frequency or on last round
        round_elapsed = time.perf_counter() - round_start
        if (r + 1) % eval_freq == 0 or r == rounds - 1:
            _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
            print(f"\rRound {r+1} | {round_elapsed:.2f}s | Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}", end='', flush=True)
        else:
            total_accuracy = round_accuracies[-1] if round_accuracies else None
            total_loss = round_losses[-1] if round_losses else None
            print(f"\rRound {r+1} - (skipped evaluation)| {round_elapsed:.2f}s ", end='', flush=True)
        
        round_accuracies.append(total_accuracy)
        round_losses.append(total_loss)
    
    print()  # Newline after all rounds complete

    return round_accuracies, round_losses



# =========================== Adam =============================================
def local_train_adam(global_model, train_loader, n_steps, device=None):
    """
    Local training for Adam
    """
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    criterion = nn.CrossEntropyLoss()

    accumulated_grad = {}

    with disable_bn_running_stats(local_model):
        for step, (x_train, y_train) in enumerate(train_loader):
            if step >= n_steps:
                break

            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()

            for param in local_model.parameters():
                if param.requires_grad:
                    param.grad = None

            loss = criterion(local_model(x_train), y_train)
            loss.backward()

            for name, param in local_model.named_parameters():
                if param.grad is not None:
                    if name not in accumulated_grad:
                        accumulated_grad[name] = param.grad.clone()
                    else:
                        accumulated_grad[name] += param.grad


    return {name: g / n_steps for name, g in accumulated_grad.items()}     

def federated_adam(
        global_model, train_loaders, num_clients, rounds, n_steps, lr, 
        test_loaders, weight_decay=5e-4, eps=1e-8, eval_freq=1, device=None):
    """
    Federated Adam Algorithm.
    """
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    global_model.to(device)
    global_model.train()

    optimizer = torch.optim.Adam(
        global_model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    round_accuracies = []
    round_losses = []

    # инициализация 0 эпохи
    _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
    round_accuracies.append(total_accuracy)
    round_losses.append(total_loss)
    print(f"\rRound 0 - Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f} ", end='', flush=True)

    for r in range(rounds):
        round_start = time.perf_counter() 
        print(f"\rRound {r+1}", end='', flush=True)

        client_grads = [
            local_train_adam(global_model, train_loaders[i], n_steps, device=device)
            for i in range(num_clients)
        ]

        avg_grad = {
            name: sum(client_grads[i][name].to(device) for i in range(num_clients)) / num_clients
            for name, param in global_model.named_parameters() if param.requires_grad
        }

        optimizer.zero_grad()
        for name, param in global_model.named_parameters():
            if param.requires_grad:
                param.grad = avg_grad[name].clone()
        optimizer.step()

        round_elapsed = time.perf_counter() - round_start
        if (r + 1) % eval_freq == 0 or r == rounds - 1:
            _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
            print(f"\rRound {r} | {round_elapsed:.2f}s | Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}", end='', flush=True)
        else:
            total_accuracy = round_accuracies[-1] if round_accuracies else None
            total_loss = round_losses[-1] if round_losses else None
            print(f"\rRound {r} - (skipped evaluation)| {round_elapsed:.2f}s ", end='', flush=True)

        round_accuracies.append(total_accuracy)
        round_losses.append(total_loss)

    return round_accuracies, round_losses



# =========================== Muon =============================================
def local_train_muon(global_model, train_loader, client_momentum_buffer, 
                     n_steps, last_layer_names, weight_decay=1e-4, 
                     momentum=0.9, ns_steps: int = 5, device=None):
    """
    Local training for Federated Muon.
    """
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()

    criterion = nn.CrossEntropyLoss()
    accumulated_grad = {}

    train_iter = iter(train_loader)

    with disable_bn_running_stats(local_model):
        for _ in range(n_steps):
            try:
                x_train, y_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_train, y_train = next(train_iter)

            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()
            
            local_model.zero_grad()
            loss = criterion(local_model(x_train), y_train)
            loss.backward()

            for name, param in local_model.named_parameters():
                if param.grad is not None:
                    g = param.grad
                    if weight_decay != 0:
                        g = g.add(param.data, alpha=weight_decay)
                    if name not in accumulated_grad:
                        accumulated_grad[name] = g.clone()
                    else:
                        accumulated_grad[name] += g

    # Усредняем градиент
    for name in accumulated_grad:
        accumulated_grad[name] = accumulated_grad[name] / n_steps

    muon_update_dict = {}

    for name, G in accumulated_grad.items():
        if name in last_layer_names:
            muon_update_dict[name] = G.cpu()
            continue

        if name not in client_momentum_buffer:
            client_momentum_buffer[name] = torch.zeros_like(G, device=device)
        else:
            client_momentum_buffer[name] = client_momentum_buffer[name].to(device)
        
        # Обновление моментума на клиенте: M_t = mu * M_{t-1} + G_t
        client_momentum_buffer[name] = momentum * client_momentum_buffer[name] + G
        
        # LMO ортогонализация
        orth_G = muon_orthogonalized_update(client_momentum_buffer[name], ns_steps=ns_steps)
        
        # Возвращаем результат на CPU
        muon_update_dict[name] = orth_G.cpu()
        
        # Возвращаем обновленный буфер обратно на CPU для экономии VRAM
        client_momentum_buffer[name] = client_momentum_buffer[name].cpu()

    return muon_update_dict

def federated_muon(
        global_model, train_loaders, num_clients, rounds, n_steps, lr, lr_aux,
        test_loaders, ns_steps: int = 5, eval_freq: int = 1, momentum=0.9, weight_decay=1e-4, device=None):
    """
    Federated Muon Algorithm (with client-side momentum).
    """
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global_model.to(device)
    global_model.train()

    last_layer_names = [
        name for name, _ in list(global_model.named_parameters())[-2:]
    ]

    adamw = torch.optim.AdamW(
        [param for name, param in global_model.named_parameters() if name in last_layer_names],
        lr=lr_aux
    )

    client_momentums = [
        {name: torch.zeros_like(param, device='cpu') 
         for name, param in global_model.named_parameters() 
         if param.requires_grad and name not in last_layer_names}
        for _ in range(num_clients)
    ]

    round_accuracies = []
    round_losses = []

    # инициализация 0 эпохи
    _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
    round_accuracies.append(total_accuracy)
    round_losses.append(total_loss)
    print(f"\rRound 0 - Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f} ", end='', flush=True)

    base_lr = lr
    base_lr_aux = lr_aux

    for r in range(rounds):
        round_start = time.perf_counter()
        print(f"\rRound {r+1}", end='', flush=True)
        
        # Расчет косинусного затухания (от начального до нуля к концу обучения)
        import math
        eta = 0.5 * (1 + math.cos(math.pi * r / rounds))
        current_lr = base_lr * eta
        current_lr_aux = base_lr_aux * eta
        
        # Обновляем LR для AdamW (голова)
        for param_group in adamw.param_groups:
            param_group['lr'] = current_lr_aux

        client_updates = []
        for i in range(num_clients):
            updates = local_train_muon(
                global_model=global_model, 
                train_loader=train_loaders[i], 
                client_momentum_buffer=client_momentums[i],
                n_steps=n_steps, 
                last_layer_names=last_layer_names,
                momentum=momentum, 
                weight_decay=weight_decay,
                ns_steps=ns_steps, 
                device=device
            )
            client_updates.append(updates)

        # Агрегация
        avg_update = {}
        avg_head_grad = {}
        
        for name in client_updates[0].keys():
            if name in last_layer_names:
                avg_head_grad[name] = sum(c[name].to(device) for c in client_updates) / num_clients
            else:
                avg_update[name] = sum(c[name].to(device) for c in client_updates) / num_clients

        # Обновление скрытых слоев
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if param.requires_grad and name not in last_layer_names:
                    
                    if weight_decay != 0:
                        param.mul_(1.0 - current_lr * weight_decay)
                        
                    param -= current_lr * avg_update[name]
        
        # Обновление головы (AdamW)
        adamw.zero_grad()
        for name, param in global_model.named_parameters():
            if name in last_layer_names:
                param.grad = avg_head_grad[name].clone()
        adamw.step()

        # Оценка
        round_elapsed = time.perf_counter() - round_start
        if (r + 1) % eval_freq == 0 or r == rounds - 1:
            _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
            print(f"\rRound {r+1} | {round_elapsed:.2f}s | Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}", end='', flush=True)
        else:
            total_accuracy = round_accuracies[-1] if round_accuracies else None
            total_loss = round_losses[-1] if round_losses else None
            print(f"\rRound {r+1} - (skipped evaluation)| {round_elapsed:.2f}s ", end='', flush=True)

        round_accuracies.append(total_accuracy)
        round_losses.append(total_loss)

    print()
    return round_accuracies, round_losses



# =========================== Sign Muon (Client Moment) =========================================
def local_train_signmuon_client(global_model, train_loader, client_momentum_buffer,
                                n_steps, last_layer_names, weight_decay=1e-4, 
                                momentum=0.9, ns_steps=5, device=None):
    """
    Local training for SignMuon.

    Same protocol as SignSGD, but instead of taking sign(grad),
    we take sign(UV^T) where USV^T is the SVD of the (possibly
    reshaped) gradient tensor, approximated via Muon's
    Newton–Schulz orthogonalization.
    """
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    accumulated_grad = {}

    train_iter = iter(train_loader)

    # Disable BN running stats, as in SignSGD
    with disable_bn_running_stats(local_model):
        for _ in range(n_steps):
            try:
                x_train, y_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_train, y_train = next(train_iter)

            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()

            # Zero gradients for this batch
            local_model.zero_grad()
            loss = criterion(local_model(x_train), y_train)
            loss.backward()

            # Accumulate gradients
            for name, param in local_model.named_parameters():
                if param.grad is not None:
                    g = param.grad
                    if weight_decay != 0:
                        g = g.add(param.data, alpha=weight_decay)
                    if name not in accumulated_grad:
                        accumulated_grad[name] = g.clone()
                    else:
                        accumulated_grad[name] += g

    # Normalize accumulated gradients by n_steps
    for name in accumulated_grad:
        accumulated_grad[name] = accumulated_grad[name] / n_steps

    # Apply Muon-style orthogonalization, then take elementwise sign
    sign_muon_dict = {}

    for name, G in accumulated_grad.items():
        if name in last_layer_names:
            sign_muon_dict[name] = G.cpu()
            continue

        if name not in client_momentum_buffer:
            client_momentum_buffer[name] = torch.zeros_like(G, device=device)
        else:
            client_momentum_buffer[name] = client_momentum_buffer[name].to(device)

        # Update local momentum buffer IN PLACE
        # M_t = \mu * M_{t-1} + G_t
        client_momentum_buffer[name] = momentum * client_momentum_buffer[name] + G
        
        # LMO и Sign
        orth_M = muon_orthogonalized_update(client_momentum_buffer[name], ns_steps=ns_steps)
        compressed_update = torch.sign(orth_M)
        
        # Сохраняем результат для отправки на сервер (на CPU!)
        sign_muon_dict[name] = compressed_update.cpu()
        
        # Убираем обновленный буфер обратно на CPU, чтобы освободить VRAM
        client_momentum_buffer[name] = client_momentum_buffer[name].cpu()

    return sign_muon_dict

def federated_signmuon_client(
        global_model, train_loaders, num_clients, rounds, n_steps, lr, lr_aux,
        test_loaders, ns_steps: int = 5, eval_freq: int = 1, momentum=0.9, device = None):
    """
    Federated SignMuon.

    Same outer structure as Federated SignSGD:
    - Each client returns sign(UV^T) of its local (averaged) gradients.
    - Server aggregates via majority vote on these signs.
    - Hidden layers are updated directly with the majority sign.
    - The last layer is updated with AdamW, using the majority sign
      as a surrogate gradient (mirroring federated_signsgd).
    """
    if device is not None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    global_model.to(device)
    global_model.train()

    last_layer_names = [name for name, _ in list(global_model.named_parameters())[-2:]]

    adamw = torch.optim.AdamW(
        [param for name, param in global_model.named_parameters()
         if name in last_layer_names],
        lr=lr_aux
    )

    # Initialize clients momentum buffers
    client_momentums = [
        {name: torch.zeros_like(param, device='cpu')
        for name, param in global_model.named_parameters() if param.requires_grad and name not in last_layer_names}
        for _ in range(num_clients)
    ]

    round_accuracies = []
    round_losses = []

    # инициализация 0 эпохи
    _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
    round_accuracies.append(total_accuracy)
    round_losses.append(total_loss)
    print(f"\rRound 0 - Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f} ", end='', flush=True)
  
    for r in range(rounds):
        round_start = time.perf_counter() 
        print(f"\rRound {r+1}", end='', flush=True)

        client_signs = []
        for i in range(num_clients):
            # Передаем конкретному клиенту ЕГО личный буфер моментума
            local_signs = local_train_signmuon_client(
                global_model=global_model, 
                train_loader=train_loaders[i], 
                client_momentum_buffer=client_momentums[i],
                n_steps=n_steps, 
                last_layer_names=last_layer_names,
                momentum=momentum,
                ns_steps=ns_steps, 
                device=device
            )
            client_signs.append(local_signs)

        # 2. Server Aggregation: Majority Vote
        majority_sign = {
            name: torch.sign(sum(client_signs[i][name].to(device) for i in range(num_clients)))
            for name, param in global_model.named_parameters()
            if param.requires_grad and name not in last_layer_names
        }

        avg_head_grad = {
            name: sum(client_signs[i][name].to(device) for i in range(num_clients)) / num_clients
            for name in last_layer_names
        }

        # 3. Server Update (NO SERVER MOMENTUM)
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if param.requires_grad and name not in last_layer_names:
                    param -= lr * majority_sign[name]

        # 4. Update last layer with AdamW on the majority sign "gradient"
        adamw.zero_grad()
        for name, param in global_model.named_parameters():
            if name in last_layer_names:
                param.grad = avg_head_grad[name].clone() 
        adamw.step()

        # Evaluation
        round_elapsed = time.perf_counter() - round_start
        if (r + 1) % eval_freq == 0 or r == rounds - 1:
            _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
            print(f"\rRound {r+1} | {round_elapsed:.2f}s | Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}", end='', flush=True)
        else:
            total_accuracy = round_accuracies[-1] if round_accuracies else None
            total_loss = round_losses[-1] if round_losses else None
            print(f"\rRound {r+1} - (skipped evaluation) | {round_elapsed:.2f}s", end='', flush=True)

        round_accuracies.append(total_accuracy)
        round_losses.append(total_loss)

    print()
    return round_accuracies, round_losses



# =========================== Sign Muon with Error Feedback =========================================
def local_train_ef21_muon(global_model, train_loader, n_steps, 
                          local_G_estimator, local_momentum, last_layer_names, 
                          momentum=0.9, device=None):
    """
    EF21 Local Step.
    Clients compress the Markov residual.
    Weight decay is strictly omitted here to preserve SVD geometry.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Sent whole model to the device   
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    
    # 1. Accumulate Pure Gradients
    accumulated_grad = {}
    train_iter = iter(train_loader)

    with disable_bn_running_stats(local_model):
        for _ in range(n_steps):
            try:
                x_train, y_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_train, y_train = next(train_iter)

            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()

            local_model.zero_grad()
            loss = criterion(local_model(x_train), y_train)
            loss.backward()

            for name, param in local_model.named_parameters():
                if param.grad is not None:
                    # PURE GRADIENT ONLY: No weight decay added here
                    if name not in accumulated_grad:
                        accumulated_grad[name] = param.grad.clone()
                    else:
                        accumulated_grad[name] += param.grad

    for name in accumulated_grad:
        accumulated_grad[name] = accumulated_grad[name] / n_steps

    compressed_residual_dict = {}
    alpha_dict = {}

    # 2. EF21 Logic (Compute residual and compress)
    for name, G in accumulated_grad.items():
        if name in last_layer_names:
            # Uncompressed raw gradient for the classification head
            compressed_residual_dict[name] = G.cpu()
            alpha_dict[name] = torch.tensor(1.0, device='cpu')
            continue

        # Lazy transfer buffers to GPU
        if name not in local_G_estimator:
            local_G_estimator[name] = torch.zeros_like(G, device=device)
            local_momentum[name] = torch.zeros_like(G, device=device)
        else:
            local_G_estimator[name] = local_G_estimator[name].to(device)
            local_momentum[name] = local_momentum[name].to(device)

        # Step A: Update client momentum: M = mu * M + G
        local_momentum[name] = momentum * local_momentum[name] + G

        # # Step A: Update client momentum (EMA formulation to bound quantization scale) - empirically a bad suggestion
        # local_momentum[name] = momentum * local_momentum[name] + (1.0 - momentum) * G

        # Step B: Compute Markov residual (Delta = M_new - G_old_estimator)
        delta = local_momentum[name] - local_G_estimator[name]

        # Step C: Compress (Scaled Sign)
        scale = delta.abs().mean()
        sign_delta = torch.sign(delta)

        # Step D: Update local estimator (G_new = G_old + C(Delta))
        local_G_estimator[name] = local_G_estimator[name] + (scale * sign_delta)

        # Prepare for transfer (and move buffers to CPU to save VRAM)
        compressed_residual_dict[name] = sign_delta.cpu()
        alpha_dict[name] = scale.cpu()
        
        local_G_estimator[name] = local_G_estimator[name].cpu()
        local_momentum[name] = local_momentum[name].cpu()

    del local_model
    return compressed_residual_dict, alpha_dict

def federated_ef21_muon(
        global_model, train_loaders, num_clients, rounds, n_steps, lr, lr_aux,
        test_loaders, ns_steps: int = 5, eval_freq: int = 1, momentum=0.9, weight_decay=1e-4, device=None):
    """
    EF21-Muon with 1-bit (Sign) compression.
    Server maintains global estimator, applies Decoupled Weight Decay, applies LMO (Newton-Schulz), and updates the model.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running EF21-Muon (Sign) on {device}")
    
    global_model.to(device)
    global_model.train()

    last_layer_names = [name for name, _ in list(global_model.named_parameters())[-2:]]

    # AdamW natively handles Decoupled Weight Decay for the head
    adamw = torch.optim.AdamW(
        [param for name, param in global_model.named_parameters() if name in last_layer_names],
        lr=lr_aux, weight_decay=weight_decay
    )

    # ================== STATE INITIALIZATION ==================
    # 1. Global estimator on the server (GPU)
    global_G_estimator = {
        name: torch.zeros_like(param, device=device)
        for name, param in global_model.named_parameters()
        if param.requires_grad and name not in last_layer_names
    }

    # 2. Local buffers for clients (CPU to protect against OOM)
    client_G_estimators = [{name: torch.zeros_like(param, device='cpu') 
                            for name, param in global_model.named_parameters() 
                            if param.requires_grad and name not in last_layer_names} 
                           for _ in range(num_clients)]
    
    client_momentums = [{name: torch.zeros_like(param, device='cpu') 
                         for name, param in global_model.named_parameters() 
                         if param.requires_grad and name not in last_layer_names} 
                        for _ in range(num_clients)]
    # ==========================================================

    round_accuracies = []
    round_losses = []

    _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
    round_accuracies.append(total_accuracy)
    round_losses.append(total_loss)
    print(f"\rRound 0 - Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f} ")

    base_lr = lr
    base_lr_aux = lr_aux

    for r in range(rounds):
        round_start = time.perf_counter()
        
        # 1. Cosine Annealing Learning Rate Scheduler
        eta = 0.5 * (1 + math.cos(math.pi * r / rounds))
        current_lr = base_lr * eta
        current_lr_aux = base_lr_aux * eta
        
        # Update AdamW LR
        for param_group in adamw.param_groups:
            param_group['lr'] = current_lr_aux

        print(f"\rRound {r+1} | LR: {current_lr:.5f}", end='', flush=True)

        client_residuals = []
        client_alphas = []
        
        # 2. Local Training (EF21 step)
        for i in range(num_clients):
            res_sign, alphas = local_train_ef21_muon(
                global_model=global_model, train_loader=train_loaders[i], n_steps=n_steps,
                local_G_estimator=client_G_estimators[i],
                local_momentum=client_momentums[i],
                last_layer_names=last_layer_names,
                momentum=momentum, device=device
            )
            client_residuals.append(res_sign)
            client_alphas.append(alphas)

        # 3. Server Aggregation
        avg_head_grad = {}
        
        with torch.no_grad():
            for name in client_residuals[0].keys():
                if name in last_layer_names:
                    avg_head_grad[name] = sum(client_residuals[i][name].to(device) for i in range(num_clients)) / num_clients
                else:
                    # Aggregate compressed residuals: (1/M) * sum(alpha_i * sign_i)
                    agg_residual = sum(client_residuals[i][name].to(device) * client_alphas[i][name].to(device) 
                                       for i in range(num_clients)) / num_clients
                    
                    # UPDATE GLOBAL ESTIMATOR: G_{k+1} = G_k + R_agg
                    global_G_estimator[name] = global_G_estimator[name] + agg_residual

            # 4. Server LMO (Muon) and Weight Decay
            for name, param in global_model.named_parameters():
                if param.requires_grad and name not in last_layer_names:
                    
                    # Decoupled Weight Decay applied strictly on the server
                    if weight_decay != 0:
                        param.mul_(1.0 - current_lr * weight_decay)

                    # Server executes Newton-Schulz LMO on the global dense estimator
                    orth_G = muon_orthogonalized_update(global_G_estimator[name], ns_steps=ns_steps)
                    
                    # Step 
                    param -= current_lr * orth_G

        # 5. Update Classification Head
        adamw.zero_grad()
        for name, param in global_model.named_parameters():
            if name in last_layer_names:
                param.grad = avg_head_grad[name].clone()
        adamw.step()

        # Evaluation
        round_elapsed = time.perf_counter() - round_start
        if (r + 1) % eval_freq == 0 or r == rounds - 1:
            _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
            print(f"\rRound {r+1} | {round_elapsed:.2f}s | Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}", end='', flush=True)
        else:
            total_accuracy = round_accuracies[-1] if round_accuracies else None
            total_loss = round_losses[-1] if round_losses else None
            print(f"\rRound {r+1} - (skipped evaluation) | {round_elapsed:.2f}s", end='', flush=True)

        round_accuracies.append(total_accuracy)
        round_losses.append(total_loss)

    print()
    return round_accuracies, round_losses





# # =========================== Sign Muon (Central Moment) =========================================
# def local_train_signmuon(global_model, train_loader, n_steps, ns_steps: int = 5, device = None):
#     """
#     Local training for SignMuon.

#     Same protocol as SignSGD, but instead of taking sign(grad),
#     we take sign(UV^T) where USV^T is the SVD of the (possibly
#     reshaped) gradient tensor, approximated via Muon's
#     Newton–Schulz orthogonalization.
#     """
#     local_model = copy.deepcopy(global_model)
#     if device is not None:
#         if not torch.cuda.is_available():
#             device = torch.device("cpu")
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     local_model.to(device)
#     local_model.train()

#     criterion = nn.CrossEntropyLoss()

#     accumulated_grad = {}

#     # Disable BN running stats, as in SignSGD
#     with disable_bn_running_stats(local_model):
#         for step, (x_train, y_train) in enumerate(train_loader):
#             if step >= n_steps:
#                 break

#             x_train, y_train = x_train.to(device), y_train.to(device)
#             y_train = y_train.long()

#             # Zero gradients for this batch
#             for param in local_model.parameters():
#                 if param.requires_grad:
#                     param.grad = None

#             loss = criterion(local_model(x_train), y_train)
#             loss.backward()

#             # Accumulate gradients
#             for name, param in local_model.named_parameters():
#                 if param.grad is not None:
#                     if name not in accumulated_grad:
#                         accumulated_grad[name] = param.grad.clone()
#                     else:
#                         accumulated_grad[name] += param.grad

#     # Normalize accumulated gradients by n_steps
#     for name in accumulated_grad:
#         accumulated_grad[name] = accumulated_grad[name] / n_steps

#     # Apply Muon-style orthogonalization, then take elementwise sign
#     sign_muon_dict = {}
#     for name, G in accumulated_grad.items():
#         orth_G = muon_orthogonalized_update(G, ns_steps=ns_steps)
#         sign_muon_dict[name] = torch.sign(orth_G)

#     return sign_muon_dict


# def federated_signmuon(
#         global_model, train_loaders, num_clients, rounds, n_steps, lr, 
#         test_loaders, ns_steps: int = 5, eval_freq: int = 1, momentum=0.9, device = None):
#     """
#     Federated SignMuon.

#     Same outer structure as Federated SignSGD:
#     - Each client returns sign(UV^T) of its local (averaged) gradients.
#     - Server aggregates via majority vote on these signs.
#     - Hidden layers are updated directly with the majority sign.
#     - The last layer is updated with AdamW, using the majority sign
#       as a surrogate gradient (mirroring federated_signsgd).
#     """
#     if device is not None:
#         if not torch.cuda.is_available():
#             device = torch.device("cpu")
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
#     global_model.to(device)
#     global_model.train()

#     last_layer_names = [
#         name for name, _ in list(global_model.named_parameters())[-2:]
#     ]

#     adamw = torch.optim.AdamW(
#         [param for name, param in global_model.named_parameters()
#          if name in last_layer_names],
#         lr=lr
#     )

#     # Initialize momentum buffers
#     momentum_buffers = {
#         name: torch.zeros_like(param, device=device)
#         for name, param in global_model.named_parameters()
#         if param.requires_grad
#     }

#     round_accuracies = []
#     round_losses = []

#     # инициализация 0 эпохи
#     _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
#     round_accuracies.append(total_accuracy)
#     round_losses.append(total_loss)
#     print(f"\rRound 0 - Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f} ", end='', flush=True)
  
#     for r in range(rounds):
#         round_start = time.perf_counter() 
#         print(f"\rRound {r+1}", end='', flush=True)

#         client_signs = [
#             local_train_signmuon(global_model, train_loaders[i], n_steps, ns_steps=ns_steps, device=device)
#             for i in range(num_clients)
#         ]

#         majority_sign = {
#             name: torch.sign(sum(client_signs[i][name].to(device) for i in range(num_clients)))
#             for name, param in global_model.named_parameters()
#             if param.requires_grad
#         }

#         with torch.no_grad():
#             for name, param in global_model.named_parameters():
#                 if param.requires_grad:
#                     # Update momentum buffers: m_t = β * m_{t-1} + (1 - β) * majority_sign_t
#                     # momentum_buffers[name] = momentum * momentum_buffers[name] + (1 - momentum) * majority_sign[name]

#                     # Update momentum buffers: m_t = β * m_{t-1} + majority_sign_t
#                     momentum_buffers[name] = momentum * momentum_buffers[name] + majority_sign[name]
#                     # Then update parameters: x_{t+1} = x_t - η * sign(m_t)
#                     param -= lr * torch.sign(momentum_buffers[name])

#         # Update last layer with AdamW on the majority sign "gradient"
#         adamw.zero_grad()
#         for name, param in global_model.named_parameters():
#             if name in last_layer_names:
#                 param.grad = majority_sign[name].clone()
#         adamw.step()

#         round_elapsed = time.perf_counter() - round_start
#         if (r + 1) % eval_freq == 0 or r == rounds - 1:
#             _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
#             print(f"\rRound {r+1} | {round_elapsed:.2f}s | Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}", end='', flush=True)
#         else:
#             total_accuracy = round_accuracies[-1] if round_accuracies else None
#             total_loss = round_losses[-1] if round_losses else None
#             print(f"\rRound {r+1} - (skipped evaluation) | {round_elapsed:.2f}s", end='', flush=True)

#         round_accuracies.append(total_accuracy)
#         round_losses.append(total_loss)

#     print()
#     return round_accuracies, round_losses



# =========================== Sign Muon with Error Feedback =========================================
#def local_train_signmuon_ef(global_model, train_loader, n_steps, error_buffer, 
#                             last_layer_names, weight_decay=1e-4, ns_steps: int = 5, device=None):
#     """
#     Local training for SignMuon with Error Feedback.
#     Optimized for VRAM efficiency and mathematical correctness.
#     """
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
#     local_model = copy.deepcopy(global_model).to(device)
#     local_model.train()

#     criterion = nn.CrossEntropyLoss()
#     accumulated_grad = {}

#     train_iter = iter(train_loader)

#     with disable_bn_running_stats(local_model):
#         for _ in range(n_steps):
#             try:
#                 x_train, y_train = next(train_iter)
#             except StopIteration:
#                 train_iter = iter(train_loader)
#                 x_train, y_train = next(train_iter)

#             x_train, y_train = x_train.to(device), y_train.to(device)
#             y_train = y_train.long()

#             local_model.zero_grad()
#             loss = criterion(local_model(x_train), y_train)
#             loss.backward()

#             # Накопление градиентов с L2-регуляризацией
#             for name, param in local_model.named_parameters():
#                 if param.grad is not None:
#                     g = param.grad
#                     if weight_decay != 0:
#                         g = g.add(param.data, alpha=weight_decay)
                    
#                     if name not in accumulated_grad:
#                         accumulated_grad[name] = g.clone()
#                     else:
#                         accumulated_grad[name] += g

#     for name in accumulated_grad:
#         accumulated_grad[name] = accumulated_grad[name] / n_steps

#     sign_muon_dict = {}
#     alpha_dict = {}  

#     for name, G in accumulated_grad.items():
#         if name in last_layer_names:
#             sign_muon_dict[name] = G.cpu()
#             alpha_dict[name] = torch.tensor(1.0, device='cpu')
#             continue

#         if name not in error_buffer:
#             current_error = torch.zeros_like(G, device=device)
#         else:
#             current_error = error_buffer[name].to(device)

#         # Шаг 1: Добавляем ошибку прошлого раунда
#         G_compensated = G + current_error

#         # Шаг 2: LMO ортогонализация (выделение структуры)
#         orth_G = muon_orthogonalized_update(G_compensated, ns_steps=ns_steps)

#         # Шаг 3: Вычисление масштаба от СЫРОГО градиента 
#         scale = G_compensated.abs().mean()
#         alpha_dict[name] = scale.cpu() 

#         # Шаг 4: Знаковая компрессия
#         sign_G = torch.sign(orth_G)
#         sign_muon_dict[name] = sign_G.cpu() 

#         # Шаг 5: Обновление буфера ошибки
#         new_error = G_compensated - (scale * sign_G)
#         error_buffer[name] = new_error.cpu() 

#     return sign_muon_dict, alpha_dict

# def federated_signmuon_ef(
#         global_model, train_loaders, num_clients, rounds, n_steps, lr, lr_aux,
#         test_loaders, ns_steps: int = 5, eval_freq: int = 1, momentum=0.9, weight_decay=1e-4, device=None):
#     """
#     Federated SignMuon with Error Feedback on Clients and Momentum on Server.
#     """
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Running Federated SignMuon-EF on {device}")
    
#     global_model.to(device)
#     global_model.train()

#     last_layer_names = [name for name, _ in list(global_model.named_parameters())[-2:]]

#     # AdamW только для последних слоев (использует lr_aux)
#     adamw = torch.optim.AdamW(
#         [param for name, param in global_model.named_parameters() if name in last_layer_names],
#         lr=lr_aux, weight_decay=weight_decay
#     )

#     momentum_buffers = {
#         name: torch.zeros_like(param, device=device)
#         for name, param in global_model.named_parameters()
#         if param.requires_grad and name not in last_layer_names
#     }

#     client_error_buffers = [
#         {name: torch.zeros_like(param, device='cpu')
#          for name, param in global_model.named_parameters()
#          if param.requires_grad and name not in last_layer_names}
#         for _ in range(num_clients)
#     ]

#     round_accuracies = []
#     round_losses = []

#     # Initial evaluation
#     _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
#     round_accuracies.append(total_accuracy)
#     round_losses.append(total_loss)
#     print(f"\rRound 0 - Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f} ")

#     for r in range(rounds):
#         round_start = time.perf_counter()
#         print(f"\rRound {r+1}", end='', flush=True)

#         client_signs = []
#         client_alphas = []
        
#         # 1. Локальное обучение клиентов
#         for i in range(num_clients):
#             signs, alphas = local_train_signmuon_ef(
#                 global_model=global_model, 
#                 train_loader=train_loaders[i], 
#                 n_steps=n_steps,
#                 error_buffer=client_error_buffers[i],
#                 last_layer_names=last_layer_names,
#                 weight_decay=weight_decay,
#                 ns_steps=ns_steps, 
#                 device=device
#             )
#             client_signs.append(signs)
#             client_alphas.append(alphas)

#         # 2. Агрегация на сервере
#         avg_head_grad = {}
#         aggregated_update = {}

#         for name in client_signs[0].keys():
#             if name in last_layer_names:
#                 avg_head_grad[name] = sum(client_signs[i][name].to(device) for i in range(num_clients)) / num_clients
#             else:
#                 # Взвешенное усреднение: Sign * Scale
#                 agg_tensor = sum(client_signs[i][name].to(device) * client_alphas[i][name].to(device) 
#                                  for i in range(num_clients)) / num_clients
#                 aggregated_update[name] = agg_tensor

#         # 3. Обновление скрытых параметров (SignMuon Step)
#         with torch.no_grad():
#             for name, param in global_model.named_parameters():
#                 if param.requires_grad and name not in last_layer_names:
#                     # Серверный моментум: M_t = \mu M_{t-1} + aggregated_update
#                     momentum_buffers[name] = momentum * momentum_buffers[name] + aggregated_update[name]
#                     # Делаем шаг. Сохраняем Sign-парадигму.
#                     param -= lr * torch.sign(momentum_buffers[name])

#         # 4. AdamW
#         adamw.zero_grad()
#         for name, param in global_model.named_parameters():
#             if name in last_layer_names:
#                 param.grad = avg_head_grad[name].clone()
#         adamw.step()

#         # 5. Оценка
#         round_elapsed = time.perf_counter() - round_start
#         if (r + 1) % eval_freq == 0 or r == rounds - 1:
#             _, total_accuracy, total_loss = evaluate_model(global_model, test_loaders, verbose=False, device=device)
#             print(f"\rRound {r+1} | {round_elapsed:.2f}s | Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}", end='', flush=True)
#         else:
#             total_accuracy = round_accuracies[-1] if round_accuracies else None
#             total_loss = round_losses[-1] if round_losses else None
#             print(f"\rRound {r+1} - (skipped evaluation) | {round_elapsed:.2f}s", end='', flush=True)

#         round_accuracies.append(total_accuracy)
#         round_losses.append(total_loss)

#     print()
#     return round_accuracies, round_losses

