"""
Grid search for Federated EF-UDSignMuon (signmuon_ef_ud).

Runs `python3 -m federated_main ... --algorithm signmuon_ef_ud ...` over a grid of
learning rates with two fixed lr-aux regimes, and picks the configuration with the
highest test accuracy at round 100 (eval_freq=100).

Grid:
  Regime 1: lr in [0.01, 0.05] step 0.005,  lr_aux fixed = 0.001
  Regime 2: lr in [0.001, 0.009] step 0.001, lr_aux fixed = 0.0001
"""
import re
import sys
import subprocess
from pathlib import Path

# Directory of this script = code/ (where federated_main.py lives).
HERE = Path(__file__).resolve().parent

# ----------------------------- Grid ---------------------------------------
GRID = []
# Regime 1: lr in [0.01, 0.05] with step 0.005, lr_aux fixed at 0.001
for i in range(2, 11):                       # 0.010, 0.015, ..., 0.050
    GRID.append((round(0.005 * i, 4), 0.001))
# Regime 2: lr in [0.001, 0.009] with step 0.001, lr_aux fixed at 0.0001
for i in range(1, 10):                       # 0.001, 0.002, ..., 0.009
    GRID.append((round(0.001 * i, 4), 0.0001))

EVAL_ROUND = 100                             # round at which accuracy is compared
DEVICE = "cuda:3"

BASE_ARGS = [
    "--model", "cnn2",
    "--dataset", "cifar10",
    "--algorithm", "signmuon_ef_ud",
    "--rounds", "2000",
    "--n_steps", "3",
    "--n_parties", "10",
    "--batch_size", "64",
    "--momentum", "0.9",
    "--device", DEVICE,
    "--eval_freq", "100",
]


def run_one(lr, lr_aux):
    cmd = [sys.executable, "-m", "federated_main", *BASE_ARGS,
           "--lr", f"{lr:g}", "--lr-aux", f"{lr_aux:g}"]
    tag = f"lr{lr:g}_lraux{lr_aux:g}"
    log_path = HERE / f"output_grid/output_new_efud_muon_10_3_{tag}.txt"
    print(f"\n>>> lr={lr:g}  lr_aux={lr_aux:g}   (log: {log_path.name})", flush=True)

    acc = None
    # Регулярное выражение для поиска строки вида: "Round 100 | 1.28s | Accuracy: 54.04%, Loss: 1.3088"
    pattern = re.compile(rf"Round {EVAL_ROUND}\b.*Accuracy:\s*([\d\.]+)%")

    with open(log_path, "w") as logf:
        # Запускаем процесс через Popen, чтобы читать вывод на лету
        process = subprocess.Popen(
            cmd, 
            cwd=HERE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1  # Построчная буферизация
        )

        # Читаем вывод процесса строка за строкой по мере появления
        for line in process.stdout:
            logf.write(line)
            logf.flush()  # Сразу сбрасываем в файл, чтобы лог обновлялся в реальном времени

            match = pattern.search(line)
            if match:
                acc = match.group(1) # Получаем число, например "54.04"
                
                process.terminate()
                break

        process.wait()

    if acc is not None:
        print(f"    accuracy@{EVAL_ROUND} = {acc}%", flush=True)
        return float(acc)
    else:
        print(f"    accuracy@{EVAL_ROUND} NOT FOUND in log!", flush=True)
        return 0.0


def main():
    results = []
    for idx, (lr, lr_aux) in enumerate(GRID, 1):
        print(f"\n=== [{idx}/{len(GRID)}] ===", flush=True)
        acc = run_one(lr, lr_aux)
        results.append((lr, lr_aux, acc))

    print("\n" + "=" * 60)
    print(f"{'lr':>8} | {'lr_aux':>10} | {'acc@' + str(EVAL_ROUND):>12}")
    print("-" * 60)
    for lr, lr_aux, acc in results:
        acc_str = f"{acc}" if acc is not None else "N/A"
        print(f"{lr:>8g} | {lr_aux:>10g} | {acc_str:>12}")
    print("=" * 60)

    valid = [r for r in results if r[2] is not None]
    if valid:
        best = max(valid, key=lambda x: x[2])
        print(f"BEST: lr={best[0]:g}, lr_aux={best[1]:g}, "
              f"accuracy@{EVAL_ROUND}={best[2]}%")
    else:
        print("No valid runs (accuracy@100 not found in any log).")


if __name__ == "__main__":
    main()
