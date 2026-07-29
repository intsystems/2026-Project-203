# Reproducing every number and figure in the paper

Exact commands for *SignMuon, MuonSign, and the Role of Error Feedback*
(`aaai_article/v2_SignMuon_AAAI.tex`). Every command is run **from this `code/`
directory**, which is the Python package root.

```bash
cd code
pip install -r requirements.txt
python3 -m tests.test_code      # sanity check: CPU only, ~1 min, no downloads
```

Commands are written `python3`. On Windows the interpreter is usually plain
`python` — substitute it throughout. What matters is that it is the interpreter
that has `torch` installed.

Add `--download` on the first run of any experiment that needs CIFAR-10 or MNIST.

**Contents**

Artifacts are referred to by **LaTeX label**, never by float number: the numbers
shift whenever a float is added elsewhere in the paper, and every stale
"Table 3" in this file was a bug. Find a label in the `.tex` with
`grep -n 'label{tab:grid_search}' ../aaai_article/v2_SignMuon_AAAI.tex`.

| Paper artifact | Section here |
| :--- | :--- |
| Theorems 1–3, `fig:divergence_plot` | [1. Counterexamples](#1-counterexamples-figdivergence_plot-theorems-13) |
| Theorem 4 (EF21-SignMuon), `fig:ef21_momentum` | [2. EF21-SignMuon divergence](#2-ef21-signmuon-divergence-theorem-4) |
| `tab:synthetic_tuned`, `tab:synthetic_alignment`, `tab:synthetic_dynamics`, `fig:synthetic_*` | [3. Synthetic convex problem](#3-synthetic-convex-problem) |
| `tab:cifar_main`, `tab:cifar_central`, `fig:cifar_results`, `fig:cifar_curves_appendix`, `fig:cifar_lr` | [4. Centralized CIFAR-10](#4-centralized-cifar-10-resnet-18) |
| `tab:fed_master`, `tab:exp_2`, `tab:exp_3`, `fig:exp_2`, `fig:exp_3` | [5. Federated CIFAR-10](#5-federated-cifar-10-tabexp_2-tabexp_3-figexp_2-figexp_3) |
| `tab:nanogpt`, `tab:nanogpt_diag`, `fig:nanogpt*` | [nanogpt/README.md](nanogpt/README.md) |
| — | [6. Multi-seed runs](#6-multi-seed-runs) |

Runtimes below are for a single A100. Everything writes to `results/`.

---

## 1. Counterexamples (`fig:divergence_plot`, Theorems 1–3)

CPU, numpy only, **< 10 seconds**. No hyperparameters to tune.

```bash
python3 -m counterexamples.problems              # prints the theorem constants
python3 -m counterexamples.run_counterexamples   # writes the figures
```

`counterexamples.problems` verifies the exact values quoted in Theorems 1-2 and
the Theorem 4 rate:

```
<G, sign(LMO(G))>   = -412.311   (Theorem 1: -42468/103)
<G, LMO(sign(G))>   =  -13.888   (Theorem 2: -13.89)
rate 49/480 = 0.102083 for every mu in {0, 0.25, 0.5, 0.9, 0.99}, both variants
```

Theorem 3's `<G, sign(LMO(sign(G)))> = -76` is printed by
`counterexamples.run_counterexamples`, in its Counterexample-2 table.

`run_counterexamples` also writes, to **both** `counterexamples/figures/` (PNG + PDF) and
`../aaai_article/images/counterexamples/` (PDF, which is what LaTeX includes):

| File | Paper |
| :--- | :--- |
| `counterexamples_main.pdf` | `fig:divergence_plot` — all three instances, one panel each (SignMuon 4×4, MuonUSign/MuonSign 5×5, EF21-SignMuon 2×2) |

The LMO here is an **exact rank-truncated SVD**, matching the theorem statements.
Deep-learning runs use the Newton–Schulz approximation, as practitioners do. To see
how the two differ on these instances:

```bash
python3 -m counterexamples.verify_ns_oracle --trajectories
```

`verify_ns_oracle` takes `--steps`, `--sigmas`, `--Ms`, `--trajectories`,
`--traj-steps`, `--traj-dtype`, `--eta`, `--T`. It has **no** `--mu`/`--nesterov`
— those belong to `run_counterexamples`, where momentum provably cannot change
the trajectory (Proposition 1) and the runner confirms it for any `--mu`.

## 2. EF21-SignMuon divergence (Theorem 4)

```bash
python3 -m counterexamples.plot_ef21_momentum    # ~20 s, CPU
```

Writes `ef21_signmuon_momentum.pdf` to the same two directories. Verifies the exact
rate `49/480` per step for every `μ ∈ {0, 0.5, 0.9, 0.95, 0.99}` and both momentum
variants. `python3 -m counterexamples.problems` prints the same check over a
slightly different set, `μ ∈ {0, 0.25, 0.5, 0.9, 0.99}`.

Both measure the slope over a whole number of periods. The trajectory is
period-two, so a window whose endpoints have opposite parity leaks half an
oscillation into the slope and never shows the exact rate — that was a real bug
here, fixed 2026-07-28.

## 3. Synthetic convex problem

`F(X) = ½⟨X, AXB⟩` on `100 × 100` matrices, three independent draws of `(A, B)`,
`X₀ ~ N(0, 0.01)` entrywise. `A` and `B` are symmetric with a prescribed spectrum
in a Haar-random eigenbasis, so `L` and `σ` are the extreme products
`λᵢ(A)λⱼ(B)` in closed form and nothing is estimated.

### Two commands

```bash
python3 -m synthetic.run_gpu --force      # every stage, ~1.6 h on one GPU
python3 -m synthetic.run_gpu --archive    # rebuild SUMMARY.md + the .zip
```

The first runs `tests/test_code.py` as a preflight and refuses to start if it
fails, then all seven stages, then writes the archive itself; the second only
rebuilds it from what is already on disk.

**`--force` is what makes it a rerun.** A stage whose `<method>/<mode>.json`
already exists is skipped, so on a box that has run before, plain
`run_gpu` skips all seven and exits in seconds looking like a success. Drop
`--force` only on the first run of a fresh tree, or to resume one that died
part-way. `--force` overwrites `results/synthetic/` and the `.zip` beside it;
move both aside first if the previous run still backs numbers in the paper.

Everything lands under `results/synthetic/`:

| Path | What it is |
| :--- | :--- |
| `SUMMARY.md` | every table in one file — **this is the one to read** |
| `MANIFEST.json` | commit, GPU, CPU, RAM, OS, torch/CUDA/driver, argv and wall time per stage |
| `logs/<stage>.log` | full console output, including the preflight |
| `<method>/<mode>.json` | machine-readable results |

**`results/synthetic_results.zip` is the one file to bring back.** In VS Code,
right-click it in the Explorer under `code/results/` and choose Download;
`results/` is gitignored, so nothing else leaves the box on its own.

Useful variants:

```bash
python3 -m synthetic.run_gpu --list                 # what each stage measures
python3 -m synthetic.run_gpu --quick                # ~2 min smoke test, own tree
python3 -m synthetic.run_gpu --stages floor horizon # split the run
```

`--quick` writes to `results/synthetic_quick/` and `--m N` to
`results/synthetic_NxN/`, so no smoke test can land where the reported numbers
belong. `--m` is not a cost knob: below ~200×200 a step is dominated by
kernel-launch latency rather than arithmetic, so `--m 20` and `--m 100` take
about the same time. What a sweep costs is (configurations × iterations).

### Checking a run came back clean

```bash
grep -c '†' results/synthetic/SUMMARY.md      # censored optima; expect 0
```

A `†` marks a tuned value sitting on the edge of its grid, which is an upper
bound rather than a measurement. Each learning-rate window now ends *past* the
largest stability edge measured for its family, so a surviving `†` means the
optimum is at the edge of stability — a fact about the method — rather than a
grid too narrow to hold it. Either way the row should not be quoted as a tuned
value without saying which of the two it is.

### The stages

| Stage | Measures | Feeds |
| :--- | :--- | :--- |
| `stability` | Largest stable `η`, and the step length `η_max‖S‖_F`. SGD is the control: it must return `2/L` | `tab:synthetic_dynamics` |
| `alignment` | Distribution of `ρ_t = ⟨∇F, D_t⟩/(‖∇F‖_F‖D_t‖_F)` along the tuned trajectory | `tab:synthetic_alignment` |
| `floor` | Plateau `F∞(η)`, `‖∇F‖∞(η)` of a constant step and their exponents in `η` | `tab:synthetic_dynamics`, `fig:synthetic_dynamics` (left) |
| `horizon` | `err ~ T^-p`, `η* ~ T^-q` tuned separately at each budget | `tab:synthetic_dynamics`, `fig:synthetic_dynamics` (centre) |
| `kappa` | The tuned comparison swept over condition number at `L = 1` | `fig:synthetic_dynamics` (right) |
| `grid` | Fewest iterations to `F ≤ 1e-3` within `T_max = 5000`, `η` and `μ` tuned per method | `tab:synthetic_tuned` |
| `final` | Re-runs those optima with saved curves | `fig:synthetic_main` |

`tab:synthetic_results`, `tab:grid_search` and `fig:synthetic_results` are the
*superseded* run at `m = n = 500` and are not regenerated by any stage above;
the current-protocol counterparts are `tab:synthetic_tuned` and
`fig:synthetic_main`.

Three points where the code has to match the theory exactly, and does:

* **`horizon` fits `p` on the dual norm.** `err := min_t ‖∇F(X_t)‖²_*` is the
  squared *dual* norm the theorems bound, and which norm that is depends on the
  ball the LMO minimizes over: `ℓ1` for the sign family (`ℓ∞` ball), nuclear for
  the LMO family (spectral ball). `DUAL_NORM` in `synthetic/benchmark.py` is the
  map. `SUMMARY.md` reports that exponent and carries the Frobenius one beside
  it so rows stay comparable across families. Reporting the norm rather than its
  square halves `p` and leaves `q` unchanged.
* **`floor` predicts slope 1.** Balancing the two terms of the descent lemma
  gives a gradient floor linear in `η` with coefficient `L‖S‖_F/2ρ`. SignMuon and
  SignSGD share `‖S‖_F = √(mn)` exactly, so any gap between their floors is
  attributable to `ρ` alone.
* **EF21-MuonSign is scored on `X`,** the exact model the guarantee bounds, while
  its gradient is taken at the broadcast `W` — which the closed-form gradient
  makes free.

### Figures

```bash
python3 -m synthetic.plot_synthetic       # reads results/synthetic/
```

Writes `synthetic_main`, `loss`, `GN` and `diagnostics` as PDF + PNG into
`results/synthetic/figures/`, and reports which stages have not been run rather
than failing. `diagnostics` is one full-width row — floor, budget, conditioning
— under a single legend, the layout the counterexample and federated figures
use, since all three panels draw the same ten methods; it drops a panel whose
stage is missing rather than the whole figure. Nothing is copied into
`aaai_article/` automatically; do that deliberately.

### What is pinned, and what is not

Re-running a stage on the same machine reproduces it exactly, and `grid` and
`final` agree digit for digit on the run they share. Three things make that true:
`--init-seed` fixes `X₀`, `--problem-seeds` fixes `(A, B)`, and the tie-break RNG
— `sign(0)` draws a random `±1`, and exact zeros do occur in bfloat16 `polar`
output — is forked and re-seeded per run, so a configuration does not depend on
what was run before it. `grid` also re-runs its winner alone before reporting it,
since in bfloat16 a matmul of a different batch width can round differently.

Across machines it is *not* bit-exact, and cannot be: a different GPU or BLAS
perturbs a gradient at the last bit, and `sign` is discontinuous, so an entry
within rounding of zero flips and the step changes by `O(1)`. That is the
instability this paper is about. What survives is everything the tables report —
plateau levels, fitted slopes, `ρ` distributions — because each is a statistic
over a trajectory rather than a single iterate. Measured between two float32
reduction orders at `100 × 100` over 800 steps, those agree to `3e-3` at worst
and `1e-5` on the alignment statistics.

### Conventions that move numbers

* `--lmo-dtype bfloat16` (default) matches the reference Muon implementation.
  It carries ~3 decimal digits, so for the methods that sign the LMO *output* an
  entry of `polar(M)` near zero can flip; `--lmo-dtype float32` removes that. The
  value used is recorded in every output JSON.
* Learning-rate grids are logarithmic and 3–4 decades wide, because the optimal
  `η` spans that much across these methods. An optimum landing on a grid edge is
  an upper bound rather than a tuned value; the tuner prints `[BOUNDARY]` and
  records `on_grid_boundary` in the JSON, and such a row needs its grid widened
  before it is reported.
* Sweeps run batched — the whole `(η, μ, schedule)` grid advances as one
  `[B, m, n]` trajectory (`synthetic/batched.py`). `--runner sequential` is the
  reference one-at-a-time loop the batched one is tested against, and is hours
  slower.

## 4. Centralized CIFAR-10 (ResNet-18)

ResNet-18, 75 epochs, batch 128, momentum 0.9, cosine-annealed learning rate.

### 4a. Two commands, start to finish

On the GPU box:

```bash
cd code
python3 -m centralized.overnight --device cuda:0 --download   # the whole protocol
python3 -m centralized.export_article                         # pack it up
```

The second prints the path of a single `results/article_export.tar.gz` (~1 MB).
That is the file to bring home; the run tree stays put, because a 33-job ResNet-18
sweep is ~1.5 GB of `model.pt`. Unpack the archive next to `code/` and:

```bash
python3 -m centralized.plot_analysis --bundle article_export  # -> results/analysis/
```

which writes `fig:cifar_results` (`cifar_main`), `fig:cifar_curves_appendix`
(`curves_75ep`), `fig:cifar_train_loss` (`train_loss_75ep`) and `fig:cifar_lr`
(`lr_sensitivity`), as PNG and PDF, and prints the range-matched sweep spreads the
`fig:cifar_lr` caption quotes. Nothing is copied into `aaai_article/`
automatically; do that deliberately.

> **Redrawing the *submitted* figures, before the re-run lands.** The figures in
> the current submission came from the 2026-07-27 run, whose tree is on the GPU box
> rather than in this repository. `centralized/table2_full.csv` and
> `curves*.json` — `aggregate.py`'s outputs from that run — are committed, and
>
> ```bash
> python3 -m centralized.plot_analysis --legacy
> ```
>
> reproduces all four figures from them **byte for byte** (pinned by
> `test_the_submitted_figures_can_still_be_redrawn`). Its sweep panel is test
> accuracy at 15 epochs, because that is what the old protocol selected on; the
> `--bundle` path is validation accuracy at 75, per §4b. Delete those files once
> the re-run has replaced the figures, not before.

The archive contains:

| File | What it is |
| :--- | :--- |
| `table_cifar.csv` | **`tab:cifar_main` / `tab:cifar_central`**, aggregated over seeds exactly as the paper defines them — quote from here rather than retyping |
| `runs.csv` | one row per run: config plus every derived summary metric |
| `curves.csv` | tidy per-epoch series; the figures are a function of this and `runs.csv` alone |
| `gain.csv`, `gain_fits.csv` | the `--log-gain` series and its fitted exponent |
| `environment.json`, `hardware.tex` | **exact GPU, driver, CUDA, Python, PyTorch, commit** — §4f |
| `configs.json` | the full config of every run, nothing dropped |
| `overnight/` | `REPORT.md` and `state.json` from the driver |
| `MANIFEST.md` | what each file is, and how many seeds each configuration carries |

`--budget-hours` defaults to 0, meaning **no deadline**: every phase runs to
completion and only Ctrl-C stops it (cleanly, writing the report). Pass a positive
number to have it stop by itself.

Watch the first ~6 minutes. It runs the CPU test suite, prints the per-layer
learning-rate table, records the machine, times two real epochs **on your GPU**,
and then prints a schedule — for example, at the 12.9 s/epoch measured on an
RTX A4000:

```
  phase     jobs  epochs   hours  cumulative   done by
  gain         4      20     0.3         0.3   Sun 02:20
  aux         30      15     1.9         2.2   Sun 04:12
  lr          50      75    15.0        17.2   Sun 19:12
  final       30      75     9.0        26.2   Mon 04:12
  wd           3      75     0.9        27.1   Mon 05:04
```

Once the schedule appears you can leave it. In the morning — and the morning
after — read `results/overnight/REPORT.md`. Properties that matter for an
unattended run:

* **budget-aware** — every phase is costed from the *measured* epoch time and the
  deadline is checked before each job, so it stops cleanly instead of being killed;
* **crash-isolated** — each job is a subprocess, so one diverging learning rate
  cannot take down the night;
* **resumable** — state is written after every job; `--resume` continues where it
  stopped (interrupted jobs are deliberately not recorded, so they are retried
  rather than retired);
* **priority-ordered** — diagnostics, then `η₀`, then the headline table, then the
  ablation; finals are **seed-major** (all methods at seed 0 before seed 1), so
  stopping early leaves complete tables rather than fragments;
* **readable mid-run** — `REPORT.md` is rewritten after every phase and every final
  run, so you never have to stop the job to see what it found.

Useful variants: `--preflight-only` (just the checks and the schedule),
`--dry-run`, `--phases lr final` (skip the diagnostics once they are settled),
`--report-only` (rebuild `REPORT.md` from `state.json`; safe while a job trains),
`--deterministic` to disable cuDNN autotuning, `--lr-points 3` to halve the `lr`
phase.

### 4b. Selection happens at the reporting horizon

An earlier version of this protocol ranked learning rates at 15 epochs and
reported at 75. Re-running the top three rates at 75 **reversed** the ranking for
both methods checked:

| method | best @ 15 ep | best @ 75 ep |
| :--- | ---: | ---: |
| SignMuon | `0.02` (93.68) | `0.2` (94.54) |
| Muon | `0.05` (93.46) | `0.01` (94.60) |

That is a systematic artefact, not seed noise: each run anneals cosinally to zero
over *its own* horizon, so a 15-epoch run spends nearly its whole budget at a
decayed rate and a 75-epoch run does not. The bias is not even in a consistent
direction — SignMuon's optimum moved up three lattice steps, Muon's down two — so
there is no correction to apply, only a proxy to retire.

The `lr` phase therefore runs at `--final-epochs`. Selection is still `val_acc`
from the 45k/5k split, so the test set enters no decision; it simply costs 15 GPU-h
instead of 3.7 to make the reported rate the rate that won at the reported horizon.

### 4c. The stages by hand

```bash
# The horizon matters: --stage lr must run at the horizon the table reports,
# while aux and alpha compare two arms at one shared horizon and can be short.
TUNE="--dataset cifar10 --model resnet18 --head-adamw always \
      --device cuda:0 --data ./data"

# Is the auxiliary rate method-independent?  (~2 GPU-h)
#   Two anchor methods spanning an order of magnitude in eta_0, over one lr_aux
#   grid. AGREE -> fix one lr_aux globally and report that it was verified.
python3 -m centralized.tune --stage aux $TUNE --epochs 15 --lr-scaling unit-gain

# eta_0 per method, 5 lattice points each, at the reporting horizon (~15 GPU-h)
python3 -m centralized.tune --stage lr $TUNE --epochs 75 --lr-scaling unit-gain \
                            --lr-aux 0.001

# Finals: 3 seeds, full 50k, fixed 75-epoch budget
for m in signmuon muonusign muonsign ef21signmuon ef21muonusign ef21muonsign \
         muon signsgd sgd adam; do
  for s in 0 1 2; do
    python3 -m centralized.main --dataset cifar10 --model resnet18 --epochs 75 \
      --optimizer $m --lr-scaling unit-gain --head-adamw always \
      --lr <tuned> --lr-aux 0.001 --seed $s --device cuda:0 --data ./data
  done
done
python3 -m centralized.export_article
```

The per-layer exponent sweep (`--stage alpha`, `α ∈ {0, ½, 1}`) is available but
is **not** needed for any paper number: the `gain` phase measures the exponent
directly, and at a single width `α` is largely absorbed into `η₀` — the sweep came
out flat to within 0.3%. Run it with `--stage alpha --epochs 15` if you want it.

Use the **same seed set for every method**, so seed *k* means the same
initialization and data order everywhere and the comparison is *paired* — far more
powerful than unpaired at n=3. Adopt the decision rule up front: **claim a gap only
when it exceeds the paired std**, otherwise write "indistinguishable".

All ten methods are in `tab:cifar_central`, `ef21signmuon` included — the method
Theorem 4 says can be made to diverge, present as a predicted-failure baseline that
in fact tops the table. The theorem is a statement about worst cases, not about
ResNet-18.

### 4d. Four properties, enforced by the code rather than by discipline

1. **No test-set tuning.** All selection runs use `--split tune`, a fixed 45k/5k
   partition (`--val-seed`, independent of `--seed` so every method sees the same
   split), and select on `val_acc` averaged over the last `--last-k` epochs.
2. **Equal budget.** Every method gets the same number of configurations on a
   multiplicatively anchored 1–2–5 lattice (`tune.anchor_for`), and **an optimum
   that lands on a grid endpoint is widened and re-run**, not reported.
3. **Per-layer learning rates are derived, not tuned.** `--lr-scaling` sets
   `η_layer = η₀ · λ(family, shape)` analytically; only `η₀` is searched, and it is
   a shape-free quantity. See §4e.
4. **`--head-adamw always`** so the only difference between two rows is the matrix
   rule.

### 4e. Which `--lr-scaling`?

The rules and what they assume are documented in
[common/lr_scaling.py](common/lr_scaling.py); `python3 -m common.lr_scaling` lists
them and `--compare` prints their per-layer profiles.

| rule | sign family | LMO family | note |
| :--- | :--- | :--- | :--- |
| `legacy` | `η₀` | `η₀·√max(1,m/n)` | one global rate for the sign family |
| `none` | `η₀` | `η₀` | also what Mishra et al.'s Algorithm 1 runs |
| **`unit-gain`** | `η₀/√fan_in` | `η₀·√max(1,m/n)` | **derived; the default** |
| `mup` | `η₀/fan_in` | `η₀·√max(1,m/n)` | assumes accumulated steps align |
| `mishra-analysis` | `η₀/√(mn)` | `η₀/√min(m,n)` | the normalization in their *proof* |

`unit-gain` follows from one criterion — the update's RMS gain
`γ(A)=‖A‖_F/√fan_out` should be a fixed fraction of the initialization's — and its
strongest validation is that the same criterion **derives** the `√max(1,m/n)` factor
already in the reference Muon implementation. `mup` applies the criterion to the
*accumulated* update assuming alignment; two measurements decide between them:

```bash
python3 -m common.lr_scaling --measure     # single step: incoherent, favours unit-gain
python3 -m centralized.main ... --log-gain --constant-lr
                                           # accumulated: sqrt(t) growth -> unit-gain,
                                           #              linear t     -> mup
```

The second is the driver's `gain` phase, and the constant rate is not optional:
under a decaying schedule the accumulation saturates and the fit reports the
schedule rather than the alignment.

> **Caveat to report.** ResNet-18 is a weak instrument for the exponent: **13 of its
> 20** conv weight tensors have `fan_in/fan_out = 9` exactly and hold **84.5%** of all
> parameters (84.6% of conv parameters) — and a *single* shape, `(512, 4608)`,
> appearing three times, is 63% of the model on its own. So `α` is identified only
> through the transition and 1×1-downsample layers. Confirm on a second architecture,
> or rely on the `gain` phase, before treating a small val gap as decisive.

### 4f. Hardware and software, for the reproducibility appendix

Every run stamps the machine into its own `metrics.json` at the moment it runs
(`common.utils.save_run` → `common.hardware.describe`): GPU model and memory,
driver, CUDA, cuDNN, CPU, RAM, OS, **Python version, PyTorch version**, and the git
commit plus whether the tree was dirty. Nothing identifying is collected — no
hostname, no username, no absolute paths — so the record is safe in a double-blind
submission and cannot trip `anonymize.py`.

The export bundle carries the distinct values in `environment.json` and, ready to
paste into the appendix, `hardware.tex`. To read them off a tree directly:

```bash
python3 -m common.hardware                     # this machine, one prose sentence
python3 -m common.hardware --latex             # this machine, as a LaTeX row
python3 -m common.hardware --scan results      # every machine used, by experiment
```

Fill the checklist from these, not from whichever machine compiles the LaTeX: the
experiments in this paper ran on different GPUs, so a single "computing
infrastructure" sentence would be wrong for most of the table.

### 4g. Epochs and the reported metric

Fix the budget at 75 epochs — the cosine schedule anneals to zero there, so it is
self-consistent rather than an arbitrary cut — and report, per run:

| metric | role |
| :--- | :--- |
| test accuracy, mean of the last `--last-k` epochs | **primary** |
| train accuracy | underfitting diagnostic — though on ResNet-18/CIFAR-10 it separates nothing: every method in `tab:cifar_central` reaches 99.95–100.00% |
| test loss | report with the explanation below |
| epochs to `--target-acc` | separates *speed* from final quality |
| median epoch time | cost of the method |
| uncompressed-parameter fraction | the "+ ε" in "1 bit per parameter" |

Everything except the last is printed in a `--- summary ---` block at the end of
every run *and* aggregated across seeds in `table_cifar.csv`. The
uncompressed-parameter fraction is a property of the model rather than of the run,
so it is printed once at startup, on the `Parameters: … matrix (compressed) + …
auxiliary` line.

**On the test-loss/accuracy divergence:** test cross-entropy rises after epoch ≈40
while test accuracy keeps improving. That is the standard overconfidence regime once
train accuracy saturates near 100% — the loss on misclassified points grows while the
decision boundary still improves. It is not a bug and not a reason to early-stop on
test. Designate test accuracy the primary metric *a priori*, say so in one sentence,
and report the loss anyway.

## 5. Federated CIFAR-10 (`tab:exp_2`, `tab:exp_3`, `fig:exp_2`, `fig:exp_3`)

CNN2, **11 clients**, 3 local steps, 2000 rounds, batch 64, momentum 0.9,
homogeneous split — one federation scale, all eleven methods.

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
`fig:exp_2` and `fig:exp_3` come from:

```bash
python3 -m federated.plot_federated                 # -> results/federated/figures/
python3 -m federated.plot_federated --n-parties 11 --metrics test_acc test_loss
```

One curve per method with a ±1 sample-std band over seeds, grouped by
configuration-minus-seed. Nothing is copied into `aaai_article/` automatically.

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

### 5e. Exact zeros are randomized, so the channel is a strict one bit

`sign(0) = 0` would make a client transmit from `{−1, 0, +1}`, and that is not a
corner case: `polar(M)` has an exactly-zero column wherever `M` does, and `M` does
wherever a feature was zero across the whole local batch — which after ReLU and
MaxPool is common. Measured on CNN2, **8–17% of raw sign entries are zero every
round**.

The paper's convention, and the default, maps each zero to an independent random
`±1` (`common.optimizers.sign_pm1`) on every sign channel. So the uplink symbol
is a genuine 1 bit, the "32×" uplink figure is exact, and with `±1` client
messages an odd `N` cannot tie. `--uplink-zeros keep` restores the ternary channel
for the alphabet diagnostic.

Both are now measured, not assumed: every run records `uplink_zero_frac` and
`mv_tie_frac` and prints them in the summary. The alignment table below is
reproduced by `python3 -m federated.tune --stage votes` (Monte Carlo over 400k
coordinates, so the last digit moves between runs). Alignment
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

The *tie rule* does not matter for descent: a tie carries no information about the
true sign, so abstaining and coin-flipping deliver the same expected descent
(0.6580 vs 0.6581 at `N = 10`). Randomizing restores `‖s‖_F = √(mn)`, which the
unit-gain multiplier assumes, and is the default — though under the randomized
uplink it is reachable only at an even `N`.

The EF21 residual channels are randomized too. This is safe for Theorem 4: its
construction has entrywise-nonzero residuals at every step (verified), so the
convention never fires there and the divergence rate is still exactly 49/480.

### 5f. What the compression actually saves, end to end

Every run now prints the accounting instead of quoting a headline. On CNN2 at the
measured 10% zero rate:

| method | uplink | downlink | **round trip** |
| :--- | ---: | ---: | ---: |
| `signmuon`, `muonusign`, `ef21signmuon`, `ef21muonusign` | 32× | 1× | **~2×** |
| `muonsign`, `ef21muonsign` | 32× | 32× | **~29×** |
| `muon`, `muonserver`, `sgd`, `adam` | 1× | 1× | 1× |

Two corrections are folded in (the alphabet itself is a genuine 1 bit under the
randomized-zero convention). The auxiliary group (biases, BatchNorm, head) is never compressed in either
direction, which alone puts a *perfect* 1-bit uplink at 1.087 bits/parameter
model-wide — that is the "+ ε" in "1 bit per parameter", and it would not be
negligible on a model with a larger head. And, dominating everything: **four of
the six methods broadcast a full-precision model every round**, so their
round-trip saving cannot exceed 2× however good the uplink is.

The paper writes "reducing the volume of transmitted data by a factor of 32×"
about SignMuon. That is an uplink-only figure for a method whose downlink is
uncompressed, and the text now says so. **The round-trip number for SignMuon is
~2×**; recompute it with `communication_bits` after any re-run.

This is not a problem for the paper's argument — it is the argument. The methods
that compress both directions reach ~29×, and the paper's own framing ("compressing
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
It is the **federated** path's aggregator; the centralized one goes through
`centralized.export_article`, whose `table_cifar.csv` is the paper's table itself
rather than a generic summary.

Both arms now report mean ± sample std over seeds: three for `tab:cifar_main` and
`tab:cifar_central`, five for `tab:exp_3`. A single seed measures no dispersion at
all, so the tools print a blank rather than `± 0.00` for it — do not read a blank
as agreement. Claim a gap only when it exceeds the seed spread.

---

## Notes

* **Plotting is scripts, not notebooks.** The four notebooks under `notebooks/`
  read the pre-reorganization `saves/`, `saves_federated/` and `saves_synthetic*/`
  paths and were removed on 2026-07-28. Each has a replacement that reads
  `results/`:

  | figure | script |
  | :--- | :--- |
  | `fig:divergence_plot` | `counterexamples.run_counterexamples` |
  | `fig:ef21_momentum` | `counterexamples.plot_ef21_momentum` |
  | `fig:synthetic_*` | `synthetic.plot_synthetic` |
  | `fig:cifar_*` | `centralized.plot_analysis --bundle …` (reads the export bundle, §4a) |
  | `fig:exp_2`, `fig:exp_3` | `federated.plot_federated` |
  | `fig:nanogpt*` | `nanogpt/plot_article.py` |

  Only the two counterexample scripts write into `aaai_article/`; the rest write
  to `results/` and you copy over deliberately.
* **One style, in [`common/plotting.py`](common/README.md#plottingpy).** Every
  script above calls `use_paper_style()` and takes its colours from `color_of`,
  so a method keeps its colour across figures and every figure matches
  `aaai2027.sty` — Times text, STIX math, and TrueType outlines rather than the
  Type 3 fonts matplotlib emits by default and AAAI forbids. Figures are authored
  at their printed width (`TEXT_WIDTH` / `COLUMN_WIDTH`) and included at
  `width=\textwidth` / `\columnwidth`, so a 9 pt label is 9 pt on the page. If
  you add a figure, do the same rather than styling it locally.
* **`nanogpt/`** (the modded-nanogpt speedrun adaptation) has its own
  [README](nanogpt/README.md). It *is* in the paper — `tab:nanogpt` and
  `fig:nanogpt` are main text, `tab:nanogpt_diag`, `fig:nanogpt_appendix` and
  `fig:nanogpt_diag` are appendix — but it needs 8×H100 and is not reproducible
  from this file.
* **Hardware/software.** Do not fill the reproducibility checklist by hand. Every
  run stamps its own machine — GPU, memory, driver, CUDA, cuDNN, CPU, RAM, OS,
  Python, PyTorch, git commit — into its `metrics.json`, and

  ```bash
  python3 -m common.hardware --scan results            # prose, per experiment
  python3 -m common.hardware --scan results --latex    # the appendix table
  ```

  emits one row per (experiment, machine) pair over whatever is on disk. The
  centralized bundle also carries `environment.json` and `hardware.tex`. This
  matters because the experiments here ran on different GPUs, so one "computing
  infrastructure" sentence would be wrong for most of the table. Nothing
  identifying is collected — no hostname, no username, no absolute paths. See §4f.
