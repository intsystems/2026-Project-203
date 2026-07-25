# Reproducing every number and figure in the paper

Exact commands for *SignMuon, MuonSign, and the Role of Error Feedback*
(`aaai_article/v2_SignMuon_AAAI.tex`). Every command is run **from this `code/`
directory**, which is the Python package root.

```bash
cd code
pip install -r requirements.txt
python3 -m tests.test_code      # sanity check: CPU only, ~30 s, no downloads
```

Add `--download` on the first run of any experiment that needs CIFAR-10 or MNIST.

**Contents**

| Paper artifact | Section here |
| :--- | :--- |
| Theorems 1–3, Figure 1 | [1. Counterexamples](#1-counterexamples-figure-1-theorems-13) |
| Theorem 4 (EF21-SignMuon), appendix figures | [2. EF21-SignMuon divergence](#2-ef21-signmuon-divergence-theorem-4) |
| Table 1, Table 3, Figure 4 | [3. Synthetic convex problem](#3-synthetic-convex-problem-tables-1-and-3-figure-4) |
| Table 2, Figure 2 | [4. Centralized CIFAR-10](#4-centralized-cifar-10-table-2-figure-2) |
| Tables 4–5, Figures 5–6 | [5. Federated CIFAR-10](#5-federated-cifar-10-tables-45-figures-56) |
| — | [6. Multi-seed runs](#6-multi-seed-runs) |

Runtimes below are for a single A100. Everything writes to `results/`.

---

## 1. Counterexamples (Figure 1, Theorems 1–3)

CPU, numpy only, **< 10 seconds**. No hyperparameters to tune.

```bash
python3 -m counterexamples.problems              # prints the theorem constants
python3 -m counterexamples.run_counterexamples   # writes the figures
```

The first command verifies the exact values quoted in the theorem statements:

```
<G, sign(LMO(G))>   = -412.311   (Theorem 1: -42468/103)
<G, LMO(sign(G))>   =  -13.888   (Theorem 2: -13.89)
<G, sign(LMO(sign(G))))> = -76   (Theorem 3: -76)
```

The second writes, to **both** `counterexamples/figures/` (PNG + PDF) and
`../aaai_article/images/counterexamples/` (PDF, which is what LaTeX includes):

| File | Paper |
| :--- | :--- |
| `signmuon_counterexample.pdf` | Figure 1(a) |
| `muonsign_counterexample.pdf` | Figure 1(b) |
| `ef21_signmuon_counterexample.pdf` | appendix, `ef21_signmuon_divergence.tex` |

The LMO here is an **exact rank-truncated SVD**, matching the theorem statements.
Deep-learning runs use the Newton–Schulz approximation, as practitioners do. To see
how the two differ on these instances:

```bash
python3 -m counterexamples.verify_ns_oracle --trajectories
```

Options: `--mu 0.9`, `--nesterov`, `--eta`, `--T`. Momentum provably cannot change
these trajectories (Proposition 1), and the runner confirms it for any `--mu`.

## 2. EF21-SignMuon divergence (Theorem 4)

```bash
python3 -m counterexamples.plot_ef21_momentum    # ~20 s, CPU
```

Writes `ef21_signmuon_momentum.pdf` to the same two directories. Verifies the exact
rate `49/480` per step for every `μ ∈ {0, 0.25, 0.5, 0.9, 0.99}` and both momentum
variants (also printed by `python3 -m counterexamples.problems`).

## 3. Synthetic convex problem (Tables 1 and 3, Figure 4)

`F(X) = ½⟨X, AXB⟩`, `m = n = 500`, target `F ≤ 1e-3`, `T_max = 5000`.

**Table 3** (`tab:grid_search`) is the search grid; **Table 1**
(`tab:synthetic_results`) is its result. Full grid, all ten methods, ~2 h:

```bash
python3 -m synthetic.benchmark --mode grid --device cuda:0
```

Per-method (the grid is large; this is how to split it):

```bash
python3 -m synthetic.benchmark --mode grid --device cuda:0 --methods signmuon
python3 -m synthetic.benchmark --mode grid --device cuda:0 --methods muonusign muonsign
python3 -m synthetic.benchmark --mode grid --device cuda:0 --methods ef21muonusign ef21muonsign
python3 -m synthetic.benchmark --mode grid --device cuda:0 --methods muon signsgd sgd adam
```

To re-run only the reported optima (~2 min, reproduces the Table 1 iteration
counts without the search):

```bash
python3 -m synthetic.benchmark --mode final --device cuda:0 --save-histories
```

Results land in `results/synthetic/<method>/{grid,final}.json`; `--save-histories`
adds the loss and gradient-norm curves that **Figure 4** plots
(`images/synt/loss.png`, `images/synt/GN.png`) — see
[notebooks/plot_synthetic.ipynb](notebooks/plot_synthetic.ipynb).

> **Two caveats.** (i) `--mode final` uses the optima currently in the paper, which
> for Muon (`6.5e-3`) and EF21-MuonUSign (`3.3e-3`) are **not on the grid Table 3
> states**; the grid defaults in `synthetic/benchmark.py` are annotated with exactly
> which rows are and are not reproducible from that table. (ii) The default
> `--lmo-dtype bfloat16` matches the reference Muon implementation; pass
> `--lmo-dtype float32` for a precise LMO. Both are reported in the output JSON.

## 4. Centralized CIFAR-10 (Table 2, Figure 2)

ResNet-18, 75 epochs, batch 128, momentum 0.9, cosine-annealed learning rate.

### 4a. Reproducing the published table

The hyperparameters below are exactly Table 2 (`tab:cifar_central`), under the
`legacy` per-layer rule and `--head-adamw auto`, which is what produced them.

```bash
COMMON="--dataset cifar10 --model resnet18 --epochs 75 --batch-size 128 \
        --momentum 0.9 --lr-scaling legacy --head-adamw auto \
        --data ./data --device cuda:0 --seed 0"

python3 -m centralized.main $COMMON --optimizer signmuon      --lr 0.001 --lr-aux 0.0001
python3 -m centralized.main $COMMON --optimizer muon          --lr 0.015 --lr-aux 0.001
python3 -m centralized.main $COMMON --optimizer sgd           --lr 0.015 --lr-aux 0.001
python3 -m centralized.main $COMMON --optimizer signsgd       --lr 0.001 --lr-aux 0.0001
python3 -m centralized.main $COMMON --optimizer adam          --lr 0.001 --lr-aux 0.0001
python3 -m centralized.main $COMMON --optimizer ef21muonusign --lr 0.008 --lr-aux 0.001
python3 -m centralized.main $COMMON --optimizer ef21muonsign  --lr 0.007 --lr-aux 0.001
```

### 4b. One command for the whole protocol

```bash
cd code
python3 -m centralized.overnight --device cuda:0 --budget-hours 0     --final-seeds 0 1 2 --download
```

`--budget-hours 0` means **no deadline**: every phase runs to completion and only
Ctrl-C stops it. Pass a positive number instead to have it stop by itself.

Watch the first ~6 minutes. It runs the CPU test suite, prints the per-layer
learning-rate table, times two real epochs **on your GPU**, and then prints a
schedule saying exactly which phases fit the budget — for example, on an RTX A4000
at ~30 s/epoch:

```
  phase     jobs  epochs   hours  cumulative   done by
  gain         2      20     0.2         0.2   Sun 02:12
  alpha       15      15     1.0         1.2   Sun 03:13
  lr          48      15     3.3         4.5   Sun 06:29
  verify       6      75     1.7         6.2   Sun 08:12
  final       36      75    10.3        16.5   Sun 18:30
```

Once the schedule appears you can leave it. In the morning read
`results/overnight/REPORT.md`. Properties that matter for an unattended run:

* **budget-aware** — every phase is costed from the *measured* epoch time and the
  deadline is checked before each job, so it stops cleanly instead of being killed;
* **crash-isolated** — each job is a subprocess, so one diverging learning rate
  cannot take down the night;
* **resumable** — state is written after every job; `--resume` continues where it
  stopped;
* **priority-ordered** — the α measurement first, then η₀ for all methods, then a
  horizon-stability check, then finals **seed-major** (all methods at seed 0 before
  seed 1), so stopping early leaves complete tables rather than fragments;
* **readable mid-run** — `REPORT.md` is rewritten after every phase and every final
  run, so you never have to stop the job to see what it found.

Useful variants: `--preflight-only` (just the checks and the schedule),
`--dry-run`, `--phases lr final` (skip the α study once it is settled),
`--phases gain alpha lr final` to skip the horizon check, `--deterministic` to
disable cuDNN autotuning, `--resume` to continue an interrupted run (interrupted
jobs are deliberately not recorded, so they are retried rather than retired).

### 4c. The rigorous protocol, stage by stage

Four properties, enforced by the code rather than by discipline:

1. **No test-set tuning.** All selection runs use `--split tune`, a fixed 45k/5k
   partition (`--val-seed`, independent of `--seed` so every method sees the same
   split), and select on `val_acc` averaged over the last `--last-k` epochs.
2. **Equal budget.** `centralized/tune.py` gives every method the same number of
   configurations on a *multiplicatively* anchored grid, and **flags an optimum that
   lands on a grid endpoint** instead of reporting it.
3. **Per-layer learning rates are derived, not tuned.** `--lr-scaling` sets
   `η_layer = η₀ · λ(family, shape)` analytically; only `η₀` is searched, and it is
   a shape-free quantity. See §4c.
4. **`--head-adamw always`** so the only difference between two rows is the matrix
   rule.

```bash
TUNE="--dataset cifar10 --model resnet18 --epochs 20 --head-adamw always \
      --device cuda:0 --data ./data"

# Stage 1 (~2 GPU-h): is the optimal auxiliary rate method-independent?
#   Two anchor methods spanning an order of magnitude in eta_0, 4x4 grid each.
#   AGREE -> fix one lr_aux globally and report that it was verified.
python3 -m centralized.tune --stage aux $TUNE --lr-scaling unit-gain

# Stage 2 (~2 GPU-h): which per-layer exponent? alpha in {0, 1/2, 1}
python3 -m centralized.tune --stage alpha $TUNE --method signmuon

# Stage 3 (~10 GPU-h): eta_0 per method, 11 configs each (7 coarse + 4 fine)
python3 -m centralized.tune --stage lr $TUNE --lr-scaling unit-gain --lr-aux <stage-1>

# Finals: 3 seeds, full 50k, fixed 75-epoch budget
for m in signmuon muonusign muonsign ef21signmuon ef21muonusign ef21muonsign \
         muon signsgd sgd adam; do
  for s in 0 1 2; do
    python3 -m centralized.main --dataset cifar10 --model resnet18 --epochs 75 \
      --optimizer $m --lr-scaling unit-gain --head-adamw always \
      --lr <tuned> --lr-aux <stage-1> --seed $s --device cuda:0 --data ./data
  done
done
python3 -m aggregate --root results/centralized --metric test_acc --csv table2.csv
```

Use the **same seed set for every method**, so seed *k* means the same
initialization and data order everywhere and the comparison is *paired* — far more
powerful than unpaired at n=3. Adopt the decision rule up front: **claim a gap only
when it exceeds the paired std**, otherwise write "indistinguishable".

`muonusign`, `muonsign` and `ef21signmuon` are **not in Table 2** — MuonUSign is the
`-` row, and the other two were not run. `ef21signmuon` is the method Theorem 4 says
diverges, so it belongs in the table as a predicted-failure baseline.

### 4d. Which `--lr-scaling`?

The rules and what they assume are documented in
[common/lr_scaling.py](common/lr_scaling.py); `python3 -m common.lr_scaling` lists
them and `--compare` prints their per-layer profiles.

| rule | sign family | LMO family | note |
| :--- | :--- | :--- | :--- |
| `legacy` | `η₀` | `η₀·√max(1,m/n)` | what the paper's numbers used |
| `none` | `η₀` | `η₀` | also what Mishra et al.'s Algorithm 1 runs |
| **`unit-gain`** | `η₀/√fan_in` | `η₀·√max(1,m/n)` | **derived; recommended** |
| `mup` | `η₀/fan_in` | `η₀·√max(1,m/n)` | assumes accumulated steps align |
| `mishra-analysis` | `η₀/√(mn)` | `η₀/√min(m,n)` | the normalization in their *proof* |

`unit-gain` follows from one criterion — the update's RMS gain
`γ(A)=‖A‖_F/√fan_out` should be a fixed fraction of the initialization's — and its
strongest validation is that the same criterion **derives** the `√max(1,m/n)` factor
already in the reference Muon implementation. `mup` applies the criterion to the
*accumulated* update assuming alignment; two measurements decide between them:

```bash
python3 -m common.lr_scaling --measure     # single step: incoherent, favours unit-gain
python3 -m centralized.main ... --log-gain # accumulated: sqrt(t) growth -> unit-gain,
                                           #              linear t     -> mup
```

> **Caveat to report.** ResNet-18 is a weak instrument for the exponent: 12 of its 20
> conv layers have `fan_in/fan_out = 9` exactly and hold ~63% of the parameters, so
> `α` is identified only through the transition and 1×1-downsample layers. Confirm on
> a second architecture, or rely on `--log-gain`, before treating a small val gap as
> decisive.

### 4e. Epochs and the reported metric

Fix the budget at 75 epochs — the cosine schedule anneals to zero there, so it is
self-consistent rather than an arbitrary cut — and report, per run:

| metric | role |
| :--- | :--- |
| test accuracy, mean of the last `--last-k` epochs | **primary** |
| test accuracy at the best-`val_acc` epoch | early stopping, done properly |
| train accuracy | underfitting diagnostic (SignSGD reaches only ~93% train) |
| test loss | report with the explanation below |
| epochs to `--target-acc` | separates *speed* from final quality |
| median epoch time, uncompressed-parameter fraction | cost of the method |

All of these are printed in a `--- summary ---` block at the end of every run, so a
log alone fills a table row.

**On the test-loss/accuracy divergence:** test cross-entropy rises after epoch ≈40
while test accuracy keeps improving. That is the standard overconfidence regime once
train accuracy saturates near 100% — the loss on misclassified points grows while the
decision boundary still improves. It is not a bug and not a reason to early-stop on
test. Designate test accuracy the primary metric *a priori*, say so in one sentence,
and report the loss anyway.

## 5. Federated CIFAR-10 (Tables 4–5, Figures 5–6)

CNN2, 2000 rounds, batch 64, momentum 0.9, homogeneous split. **~3–6 h per run.**

### Experiment 1 — N = 3 clients, 1 local step (Table 4, Figure 5)

```bash
FED3="--model cnn2 --dataset cifar10 --rounds 2000 --n_parties 3 --n_steps 1 \
      --batch_size 64 --momentum 0.9 --data ./data_federated \
      --device cuda:0 --eval_freq 100 --seed 0 --lr-aux 0.001"

python3 -m federated.main $FED3 --algorithm signmuon      --lr 0.001
python3 -m federated.main $FED3 --algorithm muonusign     --lr 0.001
python3 -m federated.main $FED3 --algorithm muonsign      --lr 0.001
python3 -m federated.main $FED3 --algorithm muon          --lr 0.05
python3 -m federated.main $FED3 --algorithm signsgd       --lr 0.001
python3 -m federated.main $FED3 --algorithm sgd           --lr 0.03
python3 -m federated.main $FED3 --algorithm adam          --lr 0.001
python3 -m federated.main $FED3 --algorithm ef21muonusign --lr 0.02
python3 -m federated.main $FED3 --algorithm ef21muonsign  --lr 0.022
python3 -m federated.main $FED3 --algorithm ef21signmuon  --lr 0.02
```

### Experiment 2 — N = 10 clients, 3 local steps (Table 5, Figure 6)

```bash
FED10="--model cnn2 --dataset cifar10 --rounds 2000 --n_parties 10 --n_steps 3 \
       --batch_size 64 --momentum 0.9 --data ./data_federated \
       --device cuda:0 --eval_freq 100 --seed 0 --lr-aux 0.001"

python3 -m federated.main $FED10 --algorithm signmuon      --lr 0.001
python3 -m federated.main $FED10 --algorithm muonusign     --lr 0.001
python3 -m federated.main $FED10 --algorithm muonsign      --lr 0.001
python3 -m federated.main $FED10 --algorithm muon          --lr 0.05
python3 -m federated.main $FED10 --algorithm signsgd       --lr 0.001
python3 -m federated.main $FED10 --algorithm sgd           --lr 0.03
python3 -m federated.main $FED10 --algorithm adam          --lr 0.001
python3 -m federated.main $FED10 --algorithm ef21muonusign --lr 0.02
python3 -m federated.main $FED10 --algorithm ef21muonsign  --lr 0.035
python3 -m federated.main $FED10 --algorithm ef21signmuon  --lr 0.02
```

Results: `results/federated/fed_cifar10_<algorithm>_homo_cnn2_r2000_c<N>_s<steps>/seed0/`.
**Figures 5–6** (`3_1_{loss,acc}.png`, `10_3_{loss,acc}.png`) come from
[notebooks/plot_federated.ipynb](notebooks/plot_federated.ipynb).

Non-IID instead of homogeneous: add `--partition noniid-labeldir --beta 0.5`.

> **Three things the paper does not pin down.** (i) `lr_aux` is absent from Tables 4
> and 5; `0.001` is used above and is what the grid search treated as the main
> regime. (ii) The federated tables predate the uniformity fixes described in
> `../REVIEW_NOTES.md` §4 (the learning-rate schedule and the routing of
> biases/BatchNorm were not the same for every method), so these commands will not
> reproduce the published numbers exactly — they reproduce the *corrected*
> comparison. (iii) BatchNorm running statistics are never updated from data in the
> federated setting; see `README.md`.

### Re-tuning a learning rate

```bash
python3 -m federated.grid --algorithm ef21muonsign --device cuda:0 \
    --n_parties 10 --n_steps 3 --lr 0.005:0.05:0.005 --lr-aux 0.001 --eval-round 100
```

Each grid point is early-stopped at `--eval-round`, so the sweep costs
`eval_round` rounds per configuration instead of 2000. It ranks by *early*
accuracy — re-run the winner to completion before reporting it.

## 6. Multi-seed runs

Any command above becomes a multi-seed sweep by varying `--seed`; the seed is part
of the output path, so nothing is overwritten:

```bash
for s in 0 1 2 3 4; do
  python3 -m federated.main $FED10 --algorithm ef21muonsign --lr 0.035 --seed $s
done

python3 -m aggregate --metric test_acc --csv summary.csv --curves curves.json
```

`aggregate.py` groups runs by configuration-minus-seed and reports mean ± sample
std; `curves.json` holds the pointwise mean/std curves for error-band plots. It
scans all of `results/` by default (`--root results/federated` for one family).

The paper currently reports single-seed numbers. Several of its conclusions rest on
gaps smaller than a plausible seed spread — e.g. SignMuon `84.57%` vs
EF21-MuonUSign `84.59%` in Table 5 — so ≥3 seeds is worth the compute for the
headline comparisons.

---

## Notes

* **Notebooks are the plotting step**, and they still read the pre-reorganization
  `saves/`, `saves_federated/` and `saves_synthetic*/` paths. Repoint them at
  `results/` before use.
* **`nanogpt/`** (the modded-nanogpt speedrun adaptation) is not part of the paper's
  tables and has its own [README](nanogpt/README.md).
* **Hardware/software.** The paper's reproducibility checklist has placeholders for
  the GPU, CPU, RAM, OS and PyTorch/CUDA versions; fill them from the machine that
  produced the final numbers.
