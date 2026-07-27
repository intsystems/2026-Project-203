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
| `tab:synthetic_alignment`, `tab:synthetic_dynamics`, `tab:synthetic_results`, `tab:grid_search`, `fig:synthetic_*` | [3. Synthetic convex problem](#3-synthetic-convex-problem) |
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

## 3. Synthetic convex problem

`F(X) = ½⟨X, AXB⟩`, `m = n = 500`, target `F ≤ 1e-3`, `T_max = 5000`.

Artifacts are named by LaTeX label, not by number — the numbers shift whenever a
float is added elsewhere in the paper. `tab:synthetic_alignment` and
`tab:synthetic_dynamics` plus `fig:synthetic_dynamics` and `fig:synthetic_kappa`
are the current measurements; `tab:synthetic_results`, `tab:grid_search` and
`fig:synthetic_results` are the superseded fixed-target comparison retained in
`app:synthetic_v1`.

### Everything in one command

```bash
python3 -m synthetic.run_gpu --list      # what the stages are and what they cost
python3 -m synthetic.run_gpu --quick     # ~5 min at tiny sizes — RUN THIS FIRST
python3 -m synthetic.run_gpu             # the real thing, ~5 h on one A100
```

`--quick` writes to `results/synthetic_quick/`, so a smoke test never lands where
the real results belong. The five new sweeps take ~1h45 together; the remaining
~3 h is the `grid` stage re-running Table 1 at 500×500. Split them if you would
rather not hold one job open that long:

```bash
python3 -m synthetic.run_gpu --stages stability alignment floor horizon kappa
python3 -m synthetic.run_gpu --stages grid final
```

Stages run cheapest-first as separate subprocesses, so one failure does not take
the rest down, and a stage is skipped if its output already exists (`--force` to
redo). Everything lands under `results/synthetic/`:

| Path | What it is |
| :--- | :--- |
| `SUMMARY.md` | every table in one file — **this is the one to read** |
| `MANIFEST.json` | commit, GPU, torch version, argv and wall time per stage |
| `logs/<stage>.log` | full console output of that stage |
| `<method>/<mode>.json` | machine-readable results |

It also writes `results/synthetic_results.zip` with all of the above, since
`results/` is gitignored and a remote GPU box otherwise leaves nothing single to
bring back. `--summarize-only` rebuilds `SUMMARY.md` and the bundle from the JSON
already on disk, without re-running anything.

### Or one stage at a time

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

To re-run only the tuned optima (~2 min) for the Figure 4 curves. This reads
`results/synthetic/<method>/grid.json` if the grid stage has run, and falls back
to the paper's printed table otherwise; the run log says which it used:

```bash
python3 -m synthetic.benchmark --mode final --device cuda:0 --save-histories
```

Results land in `results/synthetic/<method>/{grid,final}.json`; `--save-histories`
adds the loss and gradient-norm curves that **Figure 4** plots
(`images/synt/loss.png`, `images/synt/GN.png`) — see
[notebooks/plot_synthetic.ipynb](notebooks/plot_synthetic.ipynb).

> **Three caveats.** (i) `--mode final` prefers `results/synthetic/<method>/grid.json`
> and only falls back to the paper's printed optima, which for Muon (`6.5e-3`) and
> EF21-MuonUSign (`3.3e-3`) are **not on the grid `tab:grid_search` states**;
> `PAPER_LR_GRIDS` in `synthetic/benchmark.py` holds that table verbatim and
> `--grid-preset paper` selects it, so the discrepancy is checkable. (ii) The default
> `--lmo-dtype bfloat16` matches the reference Muon implementation; pass
> `--lmo-dtype float32` for a precise LMO. Both are reported in the output JSON.
> (iii) **The default grids are no longer Table 3's.** Table 3's grids are linear and
> one decade wide, which left SGD censored at its own boundary — its reported
> `(η, μ) = (0.1, 0.95)` is the *top* of both grids, and at `m=n=100` the tuned
> value is roughly `η = 1.6` reaching the target in ~164 iterations against 2691
> at `η = 0.1`. "SGD: 972" is therefore an upper bound, not a tuned number, and
> it is what makes Adam look dominant in Table 1. The defaults are now
> logarithmic and 3–4 decades wide; the tuner prints `[BOUNDARY]` and records
> `on_grid_boundary` in the JSON whenever an optimum still lands on an edge.
> **Tables 1 and 3 must both be re-run and re-typeset.**

### 3b. What the iteration count actually measures

On this problem every sign-family method has an accuracy floor: a `±1` step has
fixed length `η√(mn)`, so a constant step size cannot converge and `F` plateaus.
Measured at `m=n=100`, the plateau is `F∞ ∝ η²` (equivalently `‖∇F‖∞ ∝ η`) to
three digits, and "iterations to `1e-3`" is exactly `const/η`. The tuner
therefore returns the largest `η` whose plateau still fits under the target, so
Table 1 ranks methods by *floor*, not by rate. The two effects pull in opposite
directions and must be reported separately:

| at `m=n=100`, momentum 0.2 | SignMuon | SignSGD |
| :--- | ---: | ---: |
| floor `F∞` at `η = 2e-4` | `2.57e-5` | `1.72e-4` |
| iterations to `1e-3` at matched `η = 2e-4` | 1532 | **1190** |
| iterations to `1e-3` at each method's tuned `η` | **~1830** | 2036 |

SignMuon's advantage is a ~2× lower floor; it *loses* ~28% on per-step descent.
Four modes measure these directly (all default to `m=n=100`, ~minutes each):

```bash
python3 -m synthetic.benchmark --mode alignment --device cuda:0   # <g,d>/(|g||d|) along the trajectory
python3 -m synthetic.benchmark --mode floor     --device cuda:0   # F∞(η), ‖∇F‖∞(η) and their exponents
python3 -m synthetic.benchmark --mode horizon   --device cuda:0   # err ~ T^-p and η* ~ T^-q vs the predicted 1/2
python3 -m synthetic.benchmark --mode stability --device cuda:0   # largest stable η vs the Frobenius trust region
python3 -m synthetic.benchmark --mode kappa     --device cuda:0   # the same at controlled condition number
```

`alignment` is the one that is about the methods rather than the protocol: it
reports the distribution of `ρ_t = ⟨∇F(X_t), d_t⟩ / (‖∇F(X_t)‖_F‖d_t‖_F)`, the
quantity the descent lemma needs positive and the one the divergence theorems
drive negative, against the closed-form references `ρ = 1` (SGD),
`ρ = ‖G‖₁/(‖G‖_F√(mn)) → √(2/π)` (SignSGD) and `ρ = ‖G‖_*/(‖G‖_F√r)` (Muon).

`horizon` tunes `(η, μ, schedule)` *separately at each budget* `T` rather than
imposing one schedule on all methods, and `--schedules const sqrt` (its default)
lets each method pick.

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

CNN2, **11 clients**, 3 local steps, 2000 rounds, batch 64, momentum 0.9,
homogeneous split — one federation scale, all ten methods.

The federated path now follows the **same protocol as §4**: a held-out validation
split, per-layer learning rates derived from the shape, an equal-budget lattice
grid with a boundary check, and multi-seed finals. See §5d for what changed and
why the published Table 5 numbers do not survive it.

> **Why 11 clients and not the published 10?** The majority vote's alignment —
> `E[⟨truth, ŝ⟩]/(mn)`, the fraction of a full-strength descent step actually
> delivered — is monotone in `N`, and 11 beats 10 by a clear margin (0.668 vs
> 0.641 at a realistic disagreement level). At a strictly `±1` uplink there is a
> second reason: an even vote's extra voter is never decisive, so `N = 10`
> delivers exactly what `N = 9` does while tying on ~15% of coordinates. See §5e.

### 5a. One command for the whole protocol

```bash
cd code
python3 -m federated.overnight --device cuda:0 --budget-hours 12 --download
```

Watch the first few minutes. It runs the CPU test suite, prints the per-layer
multiplier table and the transported grid anchors, times a start-up run and a
40-round run **on your GPU**, and then prints a schedule. Because a federated
round costs very different amounts on different hardware, the plan is *fitted to
the budget* from that measurement rather than being cut halfway: seeds, then the
weight-decay ablation, then the tuning horizon give way in that order, and the
reverse when there is slack. `--budget-hours 0` disables the fitting and runs
everything.

```
  phase     jobs  rounds   hours  cumulative   done by
  lr          55     400     3.1         3.1   Sat 01:40
  verify       4    2000     1.1         4.2   Sat 02:48
  final       33    2000     9.3        13.5   Sat 12:05
  wd           3    2000     0.8        14.3   Sat 12:56
```

In the morning read `results/federated_overnight/REPORT.md`. Same properties as
the centralized driver: budget-aware, crash-isolated (each job is a subprocess),
resumable (`--resume`, and interrupted jobs are retried rather than retired),
priority-ordered, seed-major finals, and readable mid-run.

Useful variants: `--dry-run` (schedule only), `--preflight-only`,
`--phases lr final`, `--methods signmuon ef21muonsign`, `--report-only`,
`--partition noniid-labeldir --beta 0.5`.

### 5b. The stages by hand

```bash
FED="--dataset cifar10 --model cnn2 --n_parties 11 --n_steps 3 --batch_size 64 \
     --lr-scaling unit-gain --device cuda:0 --data ./data_federated"

# What grid will each method search, and why?
python3 -m federated.tune --stage anchors $FED

# Is the auxiliary rate method-independent?  (~30 configs)
python3 -m federated.tune --stage aux $FED --rounds 400

# eta_0 per method, equal budget, selected on val_acc only
python3 -m federated.tune --stage lr $FED --rounds 400 --lr-points 5
```

Then the finals, three seeds, on the full 50k:

```bash
FED11="--model cnn2 --dataset cifar10 --rounds 2000 --n_parties 11 --n_steps 3 \
       --batch_size 64 --momentum 0.9 --lr-scaling unit-gain --split full \
       --data ./data_federated --device cuda:0 --eval_freq 100 --lr-aux 0.001"

for m in signmuon muonusign muonsign ef21signmuon ef21muonusign ef21muonsign \
         muon muonserver signsgd sgd adam; do
  for s in 0 1 2; do
    python3 -m federated.main $FED11 --algorithm $m --lr <tuned> --seed $s
  done
done
python3 -m aggregate --root results/federated --metric test_acc --csv table5.csv
```

> **`muonserver` is new, and it is the row that makes the comparison honest.**
> There are two full-precision Muons, one per template. `muon` orthogonalizes on
> the worker and the server averages the polar factors; `muonserver` averages first
> and orthogonalizes once. The second is what MuonUSign, EF21-MuonUSign and
> EF21-MuonSign *become* when the compressor is the identity, so it is their
> uncompressed control — while `muon` is the control for SignMuon and
> EF21-SignMuon. Averaging near-orthogonal matrices shortens the step by up to
> 27% at N=11, and by more as heterogeneity grows, so comparing the server-LMO
> family against `muon` confounds the cost of the 1-bit uplink with the cost of
> the averaging. See `federated/README.md`.

Use the **same seed set for every method**, so seed *k* means the same
initialization, partition and data order everywhere and the comparison is
*paired*. Claim a gap only when it exceeds the paired std.

Results: `results/federated/fed_cifar10_<algorithm>_homo_cnn2_r2000_c<N>_s<steps>_unit-gain/seed<s>/`.
**Figures 5–6** come from [notebooks/plot_federated.ipynb](notebooks/plot_federated.ipynb).

### 5c. Reproducing the *published* Table 5 instead

The published rates were tuned under the `legacy` per-layer rule (one global rate
for the sign family, Muon's aspect factor for the LMO family) with weight decay
`5e-4`. `--lr-scaling legacy` reproduces that convention exactly — the driver
applies the shape factor outside the oracle now, and under `legacy` the two
conventions coincide bit for bit (`test_federated_legacy_rule_is_the_old_convention`).

```bash
LEG="--model cnn2 --dataset cifar10 --rounds 2000 --n_parties 10 --n_steps 3 \
     `# 10, not 11: this block reproduces the published setting verbatim` \
     --batch_size 64 --momentum 0.9 --lr-scaling legacy --weight_decay 5e-4 \
     --split full --data ./data_federated --device cuda:0 --eval_freq 100 \
     --seed 0 --lr-aux 0.001"

python3 -m federated.main $LEG --algorithm signmuon      --lr 0.001
python3 -m federated.main $LEG --algorithm muon          --lr 0.05
python3 -m federated.main $LEG --algorithm signsgd       --lr 0.001
python3 -m federated.main $LEG --algorithm sgd           --lr 0.03
python3 -m federated.main $LEG --algorithm adam          --lr 0.001
python3 -m federated.main $LEG --algorithm ef21muonusign --lr 0.02
python3 -m federated.main $LEG --algorithm ef21muonsign  --lr 0.035
```

These still will not match the printed numbers, because the pre-refactor code was
not uniform across methods (`../REVIEW_NOTES.md` §4: the cosine schedule and the
routing of biases/BatchNorm differed per method). They reproduce the *corrected*
comparison under the old parameterization.

### 5d. What changed, and why

| | before | now |
| :--- | :--- | :--- |
| **Selection data** | test accuracy, read from the training log by `federated/grid.py` | `--split tune`: 5k held out of the 50k *before* the client partition, and a tuning run never loads the test set |
| **Per-layer LR** | one global rate (`legacy`) | `--lr-scaling unit-gain`, as centrally |
| **Grid** | ad-hoc `arange` per method | 1–2–5 lattice, equal budget, boundary extension |
| **Weight decay** | `5e-4`, and the auxiliary AdamW was decayed too | `0` primary (matching §4), auxiliary group never decayed, `5e-4` as an ablation |
| **Seeds** | 1 | 3 (5 if the budget allows) |
| **Data pipeline** | 10 `DataLoader`s, `num_workers=0` | dataset resident on the GPU, augmentation as tensor ops |

The per-layer rule matters much more here than on ResNet-18. CNN2 has three
matrix parameters with `fan_in` 75, 1600 and 4608, so the sign family's
multiplier spans **7.8×** — and at a single global rate a SignMuon step on
`conv1` is 8.7× the corresponding Muon step while on `fc1` it is 63.7×. Run
`python3 -m federated.tune --stage anchors` to see the numbers.

> **Two things to state in the paper.** (i) `lr_aux` is absent from Tables 4 and
> 5; `0.001` is used throughout, and `--stage aux` is the check that it may be
> held fixed. (ii) BatchNorm running statistics are never updated from data in
> the federated setting — see `README.md`.

### 5e. The uplink is not literally one bit, and the vote can tie at any `N`

`sign(0) = 0`, so a client transmits a symbol from `{−1, 0, +1}`. That is not a
corner case: `polar(M)` has an exactly-zero column wherever `M` does, and `M` does
wherever a feature was zero across the whole local batch — which after ReLU and
MaxPool is common. Measured on CNN2, **8–17% of transmitted entries are zero every
round**, so the true uplink cost is ≈**1.37 bits per parameter**, not 1. Two
consequences:

* the "32× reduction" claim is really ~22×, and should either say so or force the
  zeros to `±1` (`--uplink-zeros random`, which costs nothing in expected descent);
* the aggregate `sign(Σⱼ sⱼ)` can be zero at **any** client count, not only even
  ones — a zero vote is not `±1`, so parity alone does not save it. The main text's
  "is equal to +1 or −1 in each component" is wrong as implemented.

Both are now measured, not assumed: every run records `uplink_zero_frac` and
`mv_tie_frac` and prints them in the summary. Alignment
`A = E[⟨truth, ŝ⟩]/(mn)` over 400k coordinates, each client correct with
probability 0.65 and silent with probability `q`:

| `N` | tie % (`q`=0) | `A` (`q`=0) | tie % (`q`=0.10) | `A` (`q`=0.10) |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.00 | 0.6573 | 8.38 | 0.6183 |
| 10 | 15.32 | 0.6558 | 9.23 | 0.6410 |
| **11** | **0.00** | **0.7029** | **7.22** | **0.6676** |
| 15 | 0.00 | 0.7747 | 5.54 | 0.7406 |

At `q = 0` the classical parity pathology is visible: `N = 10` buys nothing over
`N = 9`. At CNN2's real zero rate that washes out and alignment is simply monotone
in `N`. Either way 11 beats 10, which is why it is the default — but the reason is
"more voters", not parity.

The *tie rule* does not matter: a tie carries no information about the true sign,
so `--mv-ties zero` (abstain, the default) and `--mv-ties random` deliver the same
expected descent, 0.6580 vs 0.6581 at `N = 10`. Randomizing only restores
`‖s‖_F = √(mn)`, which the unit-gain multiplier assumes, by adding noise of the
matching size.

The EF21 uplink is deliberately left ternary: its payload `α·sign(Δ)` is zero
exactly where the estimator is already on target, and pushing it off by `α` is the
mechanism Theorem 4 exploits.

### 5f. What the compression actually saves, end to end

Every run now prints the accounting instead of quoting a headline. On CNN2 at the
measured 10% zero rate:

| method | uplink | downlink | **round trip** |
| :--- | ---: | ---: | ---: |
| `signmuon`, `muonusign`, `ef21signmuon`, `ef21muonusign` | 22× | 1× | **1.9×** |
| `muonsign`, `ef21muonsign` | 22× | 29× | **25×** |
| `muon`, `muonserver`, `sgd`, `adam` | 1× | 1× | 1× |

Three corrections are folded in. The ternary alphabet costs 1.37 bits rather than
1. The auxiliary group (biases, BatchNorm, head) is never compressed in either
direction, which alone puts a *perfect* 1-bit uplink at 1.087 bits/parameter
model-wide — that is the "+ ε" in "1 bit per parameter", and it would not be
negligible on a model with a larger head. And, dominating everything: **four of
the six methods broadcast a full-precision model every round**, so their
round-trip saving cannot exceed 2× however good the uplink is.

The paper currently writes "reducing the volume of transmitted data by a factor of
~32×" about SignMuon. That is an uplink-only figure for a method whose downlink is
uncompressed. **The round-trip number for SignMuon is 1.9×.**

This is not a problem for the paper's argument — it is the argument. The methods
that compress both directions reach 25×, and the paper's own framing ("compressing
the downlink is where the guarantee starts to cost something, and we measure
what") is exactly the right one. The fix is to quote the round trip and let
EF21-MuonSign carry the communication claim.

### 5g. Is the per-layer rule doing its job? (`gain_spread`)

The unit-gain rule exists to make the realized gain `γ = ‖λ·s‖_F/√fan_out` the same
on every layer. Every run checks that: the per-layer profile is printed at round 1,
`gain_spread` is recorded at every evaluation, and the driver warns above 1.15×.

It is not flat for the LMO family. The derivation assumes the oracle returns
`‖polar‖_F = √min(m,n)` exactly; Newton–Schulz does not, and its error is
shape-dependent. Measured on `muonserver`:

| dtype | `ns_steps` | gain spread | conv1 | conv2 | fc1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | **5** (default) | **1.34×** | 0.875 | 0.943 | 0.705 |
| bfloat16 | 7 | 1.11× | 0.888 | 0.940 | 0.846 |
| float32 | 5 | 1.35× | 0.873 | 0.945 | 0.700 |

`fc1` (120×4608, 72% of the matrix parameters) realizes only 0.70 of the assumed
gain. The sign family is flat by construction and sits near 1.08×.

**Keep `--ns_steps 5` for the run.** It is what reference Muon uses and what the
published numbers used, and the residual 1.34× is under half a learning-rate grid
step (the lattice is 2–2.5×), so it cannot flip a ranking. `--ns_steps 7` roughly
halves the excess if you want it tighter; it never reaches 1.0, because the quintic
oscillates in a band around 1 rather than converging.

## 6. Multi-seed runs

Any command above becomes a multi-seed sweep by varying `--seed`; the seed is part
of the output path, so nothing is overwritten:

```bash
for s in 0 1 2 3 4; do
  python3 -m federated.main $FED11 --algorithm ef21muonsign --lr 0.05 --seed $s
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
