import argparse
import torch
import torch.nn as nn

from data_loader import mnist_loaders
from models import CNN2
from optimizers import SignMuon, Muon, SignSGD

def train_epoch(model, loader, optimizers, device):
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    opt_main, opt_aux = optimizers

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if opt_main is not None:
            opt_main.zero_grad()
        if opt_aux is not None:
            opt_aux.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        if opt_main is not None:
            opt_main.step()
        if opt_aux is not None:
            opt_aux.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def build_optimizers(model, args, device):
    sign_params = []
    aux_params = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            sign_params.append(p)
        else:
            aux_params.append(p)

    if args.optimizer == "signmuon":
        main_opt = SignMuon(
            sign_params,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            lambda_mult=args.lambda_mult,
            ns_steps=args.ns_steps,
        )
        aux_opt = torch.optim.Adam(aux_params, lr=args.lr_aux) if aux_params else None
        return main_opt, aux_opt
    
    elif args.optimizer == "muon":
        main_opt = Muon(
            sign_params,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            lambda_mult=args.lambda_mult,
            ns_steps=args.ns_steps,
        )
        aux_opt = torch.optim.Adam(aux_params, lr=args.lr_aux) if aux_params else None
        return main_opt, aux_opt

    elif args.optimizer == "signsgd":
        opt = SignSGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
        )
        return opt, None
    
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
        return opt, None
    elif args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        return opt, None
    else:
        return "Incorrect optimizer name"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=str, default="./data", help="MNIST data dir")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="signmuon", choices=["signmuon", "muon", "signsgd", "sgd", "adam"])
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-aux", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--lambda-mult", type=float, default=1.0)
    parser.add_argument("--ns-steps", type=int, default=5)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dl, test_dl = mnist_loaders(args.data, batch_size=args.batch_size, download=args.download)

    model = CNN2(in_channels=1, input_size=28, n_kernels=32, out_dim=10)
    model.to(device)

    opt_main, opt_aux = build_optimizers(model, args, device)
    optimizers = (opt_main, opt_aux)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    # начальная инициализация
    with torch.no_grad():
        loss_tr_0, acc_tr_0 = eval_epoch(model, train_dl, device)
        loss_te_0, acc_te_0 = eval_epoch(model, test_dl, device)
    
    train_losses.append(loss_tr_0)
    train_accs.append(acc_tr_0)
    test_losses.append(loss_te_0)
    test_accs.append(acc_te_0)
    
    print(
        f"Epoch 0/{args.epochs}  "
        f"| train loss {loss_tr_0:.4f}, acc {acc_tr_0:.3f}  "
        f"| test loss {loss_te_0:.4f}, acc {acc_te_0:.3f} "
    )

    for epoch in range(1, args.epochs + 1):
        loss_tr, acc_tr = train_epoch(model, train_dl, optimizers, device)

        loss_te, acc_te = eval_epoch(model, test_dl, device)

        train_losses.append(loss_tr)
        train_accs.append(acc_tr)
        test_losses.append(loss_te)
        test_accs.append(acc_te)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| train loss {loss_tr:.4f}, acc {acc_tr:.3f} "
            f"| test loss {loss_te:.4f}, acc {acc_te:.3f}"
        )

    history = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "test_loss": test_losses,
        "test_acc": test_accs,
    }
    return model, history


if __name__ == "__main__":
    args = get_args()
    train(args)

