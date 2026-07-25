"""Learning-rate grid search for any federated method, by early-stopped subprocess.

    python3 -m federated.grid --algorithm ef21muonsign --device cuda:0
    python3 -m federated.grid --algorithm muonusign --eval-round 200 \
        --lr 0.001:0.009:0.001 --lr-aux 0.0001

Generalizes the former ``federated_efud_grid.py``, which hardcoded one method,
one GPU index, and one grid. Each configuration is launched as
``python -m federated.main ...``, the log is scanned for the accuracy at
``--eval-round``, and the child is terminated as soon as that line appears -- so a
2000-round budget costs only ``eval_round`` rounds per grid point.

Because it early-stops, this ranks configurations by *early* accuracy. Re-run the
winner to completion before reporting it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # code/federated
ROOT = HERE.parent                              # code/  (the package root)


def parse_range(spec: str) -> list[float]:
    """``"lo:hi:step"`` (inclusive) or a comma-separated list."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        n = int(round((hi - lo) / step))
        return [round(lo + i * step, 12) for i in range(n + 1)]
    return [float(x) for x in spec.split(",") if x.strip()]


def run_one(args, lr: float, lr_aux: float) -> float | None:
    """Launch one run, return the accuracy at ``args.eval_round`` (or ``None``)."""
    cmd = [
        sys.executable, "-m", "federated.main",
        "--model", args.model, "--dataset", args.dataset,
        "--algorithm", args.algorithm,
        "--rounds", str(args.rounds), "--n_steps", str(args.n_steps),
        "--n_parties", str(args.n_parties), "--batch_size", str(args.batch_size),
        "--momentum", str(args.momentum), "--device", args.device,
        "--eval_freq", str(args.eval_round), "--seed", str(args.seed),
        "--lr", f"{lr:g}", "--lr-aux", f"{lr_aux:g}",
        "--run_name", f"grid_{args.algorithm}_lr{lr:g}_lraux{lr_aux:g}",
    ]
    log_dir = ROOT / "results" / "grid_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.algorithm}_c{args.n_parties}_s{args.n_steps}_lr{lr:g}_lraux{lr_aux:g}.txt"
    print(f"\n>>> lr={lr:g}  lr_aux={lr_aux:g}   (log: {log_path.name})", flush=True)

    acc = None
    pattern = re.compile(rf"Round {args.eval_round}\b.*Accuracy:\s*([\d.]+)%")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                match = pattern.search(line)
                if match:
                    acc = float(match.group(1))
                    proc.terminate()
                    break
        finally:
            proc.wait()

    print(f"    accuracy@{args.eval_round} = "
          + (f"{acc}%" if acc is not None else "NOT FOUND (see log)"), flush=True)
    return acc


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algorithm", type=str, required=True)
    p.add_argument("--lr", type=str, default="0.005:0.05:0.005",
                   help='"lo:hi:step" or a comma-separated list')
    p.add_argument("--lr-aux", type=str, default="0.001",
                   help="Same format; every (lr, lr_aux) pair is tried")
    p.add_argument("--eval-round", type=int, default=100,
                   help="Round at which configurations are compared")
    p.add_argument("--model", type=str, default="cnn2")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--n_parties", type=int, default=10)
    p.add_argument("--n_steps", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = get_args()
    grid = [(lr, lr_aux) for lr_aux in parse_range(args.lr_aux)
            for lr in parse_range(args.lr)]

    results = []
    for idx, (lr, lr_aux) in enumerate(grid, 1):
        print(f"\n=== [{idx}/{len(grid)}] {args.algorithm} ===", flush=True)
        results.append((lr, lr_aux, run_one(args, lr, lr_aux)))

    print("\n" + "=" * 60)
    print(f"{'lr':>10} | {'lr_aux':>10} | {'acc@' + str(args.eval_round):>12}")
    print("-" * 60)
    for lr, lr_aux, acc in results:
        print(f"{lr:>10g} | {lr_aux:>10g} | {('%.2f' % acc) if acc is not None else 'N/A':>12}")
    print("=" * 60)

    valid = [r for r in results if r[2] is not None]
    if valid:
        lr, lr_aux, acc = max(valid, key=lambda r: r[2])
        print(f"BEST: lr={lr:g}, lr_aux={lr_aux:g}, accuracy@{args.eval_round}={acc}%")
        print("Re-run this configuration to completion before reporting it.")
    else:
        print("No valid runs (the accuracy line was never found).")


if __name__ == "__main__":
    main()
