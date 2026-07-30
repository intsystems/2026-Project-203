# Reproducing every number and figure in the paper

The command for each artifact of *SignMuon, MuonSign, and the Role of Error
Feedback* (`aaai_article/signmuon_body.tex` / `signmuon_appendix.tex`). Every
command runs **from this `code/` directory**, the Python package root.

```bash
cd code
pip install -r requirements.txt
python3 -m tests.test_code      # sanity check: CPU only, ~1 min, no downloads
```

Commands are written `python3`; on Windows substitute `python`. What matters is that
it is the interpreter with `torch` installed. Add `--download` on the first run of
anything that needs CIFAR-10 or MNIST. Runtimes are for a single A100; everything
writes to `results/`.

This file is the command reference. *Why* each protocol is shaped the way it is
lives in the package READMEs, linked per section — they are the place to read before
changing a setting, and this file does not repeat them.

Artifacts are referred to by **LaTeX label**, never by float number: the numbers
shift whenever a float is added elsewhere, and every stale "Table 3" in this file
was a bug. Find a label with `grep -n 'label{tab:exp_3}' ../aaai_article/*.tex`.

| Paper artifact | Section |
| :--- | :--- |
| Theorems 1–3, `fig:divergence_plot` | [1. Counterexamples](#1-counterexamples-figdivergence_plot-theorems-13) |
| Theorem 4 (EF21-SignMuon), `fig:ef21_momentum` | [2. EF21-SignMuon divergence](#2-ef21-signmuon-divergence-theorem-4) |
| `tab:synthetic_tuned`, `tab:synthetic_alignment`, `tab:synthetic_dynamics`, `fig:synthetic_*` | [3. Synthetic convex problem](#3-synthetic-convex-problem) |
| `tab:cifar_main`, `tab:cifar_central`, `fig:cifar_results`, `fig:cifar_curves_appendix`, `fig:cifar_lr` | [4. Centralized CIFAR-10](#4-centralized-cifar-10-resnet-18) |
| `tab:fed_master`, `tab:exp_3`, `tab:commacct`, `fig:exp_3` | [5. Federated CIFAR-10](#5-federated-cifar-10-tabexp_3-figexp_3) |
| `tab:nanogpt`, `tab:nanogpt_diag`, `fig:nanogpt*` | [6. NanoGPT speedrun](#6-language-modelling-the-nanogpt-speedrun-tabnanogpt-fignanogpt) |
| — | [7. Multi-seed runs](#7-multi-seed-runs) |

Sections 3, 4 and 5 all follow the same shape: **compute on the GPU box → one
archive to download → unpack and plot anywhere.** The archive is
`results/synthetic_results.zip`, `results/article_export.tar.gz` and
`results/federated_export_results.zip` respectively, and each is written by the
command that computes, so there is nothing extra to remember. Section 6 (nanoGPT)
needs no archive: its logs are small enough to live in the repository, under
`results/nanogpt/`, which is the one part of `results/` that git tracks.

## The two overnight drivers

Sections 4 and 5 are each one command:

```bash
python3 -m centralized.overnight --device cuda:0 --download
python3 -m federated.overnight   --device cuda:0 --budget-hours 24 --download
```

Both are **self-checking** (they run `tests/test_code.py` first and refuse to start
if it fails; `--force` overrides), **budget-aware** (they time two real jobs on your
GPU, then print a schedule with a finish time per phase and check the deadline
before each job, so a run stops cleanly instead of being killed —
`--budget-hours 0` means no deadline, the centralized default),
**crash-isolated** (each job is a subprocess), **resumable** (`--resume`;
interrupted jobs are retried rather than retired), **priority-ordered** (diagnostics,
then `η₀`, then the headline table, then the ablation, with finals **seed-major** so
stopping early leaves a complete 1-seed table rather than a fragmentary 5-seed one),
and **readable mid-run** (`REPORT.md` is rewritten after every phase; Ctrl-C stops
cleanly and writes it). Each ends by packing its results into the one archive to
download — `centralized.export_article` and `federated.export_article` respectively,
both of which can also be run by hand at any time.

Useful for both: `--dry-run` (schedule only), `--preflight-only`,
`--phases lr final`, `--methods signmuon ef21muonsign`, `--lr-points 3`,
`--report-only` (rebuild `REPORT.md` from `state.json`; safe while a job trains).
Determinism is on by default in the federated driver and off in the centralized one
(`--deterministic` / `--nondeterministic` are the knobs; the centralized sweep
reports seed spread rather than a bitwise trajectory).

Only the federated driver *rebalances*: because a federated round costs very
different amounts on different hardware, the plan is fitted to `--budget-hours`
rather than cut halfway — seeds, then the weight-decay ablation, then the tuning
horizon give way in that order, and the reverse when there is slack. Whatever it
gives up is printed and recorded in `REPORT.md`.

---

## 1. Counterexamples (`fig:divergence_plot`, Theorems 1–3)

CPU, numpy only, **under a minute in total**. Nothing to tune, nothing to download.
Why the instances are built as they are, and what each verdict column means, is in
[`counterexamples/README.md`](counterexamples/README.md).

```bash
python3 -m counterexamples.problems               # the theorem constants
python3 -m counterexamples.run_counterexamples    # fig:divergence_plot
python3 -m counterexamples.enumerate_minimality   # the Thm 2-3 minimality claims
python3 -m counterexamples.verify_ns_oracle --trajectories   # exact vs Newton-Schulz
```

`counterexamples.problems` prints the exact values quoted in Theorems 1–2 and the
Theorem 4 rate:

```
<G, sign(LMO(G))>   = -412.311   (Theorem 1: -42468/103)
<G, LMO(sign(G))>   =  -13.888   (Theorem 2: -13.89)
rate 49/480 = 0.102083 for every mu in {0, 0.25, 0.5, 0.9, 0.99}, both variants
```

Theorem 3's `<G, sign(LMO(sign(G)))> = -76` is printed by `run_counterexamples`, in
its Counterexample-2 table. That script and `plot_ef21_momentum` below are the only
two in the repository that write into `aaai_article/`: each figure goes to
`counterexamples/figures/` as PNG + PDF and to
`../aaai_article/images/counterexamples/` as the PDF LaTeX includes.

`verify_ns_oracle` re-evaluates the instances under the *implemented* Newton–Schulz
oracle instead of the exact SVD the theorems are stated for, sweeping `--steps`,
`--sigmas` and `--Ms`. It deliberately has **no** `--mu`/`--nesterov` — those belong
to `run_counterexamples`, where momentum provably cannot change the trajectory
(Proposition 1).

## 2. EF21-SignMuon divergence (Theorem 4)

```bash
python3 -m counterexamples.plot_ef21_momentum    # fig:ef21_momentum, ~20 s, CPU
```

Confirms the exact rate `49/480` per step for every `μ ∈ {0, 0.5, 0.9, 0.95, 0.99}`
and both momentum variants; `counterexamples.problems` prints the same check over
`μ ∈ {0, 0.25, 0.5, 0.9, 0.99}`. Both measure the rise over a whole number of
periods, because the trajectory is period-two and a window whose endpoints have
opposite parity leaks half an oscillation into the slope — which is also why
`run_counterexamples` reads its verdict off the *period-two increment* rather than
off a tail slope against an absolute tolerance
([`counterexamples/README.md`](counterexamples/README.md)).

## 3. Synthetic convex problem

`F(X) = ½⟨X, AXB⟩` on `100 × 100` matrices, three independent draws of `(A, B)`,
`X₀ ~ N(0, 0.01)` entrywise, with `L` and `σ` known in closed form. What each stage
measures, why the problem is built this way, how the grids are set and what the
batched runner does are all in [`synthetic/README.md`](synthetic/README.md).

```bash
python3 -m synthetic.run_gpu --force      # every stage, ~1.6 h on one GPU
python3 -m synthetic.run_gpu --archive    # rebuild SUMMARY.md + the .zip
python3 -m synthetic.plot_synthetic       # figures, from results/synthetic/
```

The first runs `tests/test_code.py` as a preflight, then all seven stages, then
writes the archive; the second only rebuilds it from what is on disk. Useful
variants:

```bash
python3 -m synthetic.run_gpu --list                 # what each stage measures
python3 -m synthetic.run_gpu --quick                # ~2 min smoke test, own tree
python3 -m synthetic.run_gpu --stages floor horizon # split the run
grep '†' results/synthetic/SUMMARY.md | grep '^|'   # the censored cells, if any
```

That last command lists table rows only, not the three legend lines that explain the
dagger, so a plain `grep -c` is not the check. On the 2026-07-29 run it prints five
rows, all of them a *momentum* of `0.99` at the top of its grid, for SGD and SignSGD
at the largest condition numbers. **No learning rate is censored**, which is the
condition for reporting the tuned rates as measurements.

**`--force` is what makes the first command a rerun.** A stage is considered done as
soon as one `<method>/<mode>.json` exists, so on a box that has run before, plain
`run_gpu` skips all seven and exits in seconds looking like a success. `--force`
re-runs them and overwrites those JSON files in place — it does not clear the
directory, so if the earlier run covered a different `--methods` set its files
survive. Move `results/synthetic/` and the `.zip` beside it aside first if the
previous run still backs numbers in the paper. `--quick` writes to
`results/synthetic_quick/` and `--m N` to `results/synthetic_NxN/`, so no smoke test
can land where the reported numbers belong.

| Stage | Feeds |
| :--- | :--- |
| `stability` | `tab:synthetic_dynamics` — SGD is the control and must return `2/L`. **Do not skip it**: it is the only end-to-end check that the harness measures what it claims |
| `alignment` | `tab:synthetic_alignment` |
| `floor` | `tab:synthetic_dynamics`, `fig:synthetic_dynamics` (left) |
| `horizon` | `tab:synthetic_dynamics`, `fig:synthetic_dynamics` (centre) |
| `kappa` | `fig:synthetic_dynamics` (right) |
| `grid` | `tab:synthetic_tuned` |
| `final` | `fig:synthetic_main` |

| Path under `results/synthetic/` | What it is |
| :--- | :--- |
| `SUMMARY.md` | every table in one file — **this is the one to read** |
| `MANIFEST.json` | commit, GPU, CPU, RAM, OS, torch/CUDA/driver, argv and wall time per stage |
| `logs/<stage>.log` | full console output, including the preflight |
| `<method>/<mode>.json` | machine-readable results |

**`results/synthetic_results.zip` is the one file to bring back.** In VS Code,
right-click it under `code/results/` and choose Download; `results/` is gitignored,
so nothing else leaves the box on its own.

Three things to know when reading the output. A `†` marks a tuned value on the edge
of its grid, which is an upper bound rather than a measurement — each learning-rate
window ends *past* the largest stability edge measured for its family, so a surviving
`†` means the optimum sits at the edge of stability, a fact about the method, rather
than a grid too narrow to hold it. Expect `p` and `q` to disagree with each other:
the step sizes come out tuned as the nonconvex bound prescribes (`q ≈ 1/2`) while the
error attained falls at `p ≈ 2`, twice what strong convexity would give, because a
quadratic is easier than the worst case of its smoothness class — so neither `p ≈ 1/2`
nor `p ≈ 1` is the value to expect, and a `p` near 2 is not a bug. And EF21-MuonSign
is scored on `X`, the exact model the guarantee bounds, while its gradient is taken at
the broadcast `W` — which the closed-form gradient makes free.

### What is pinned, and what is not

On one machine a stage reproduces exactly, and `grid` and `final` agree digit for
digit on the run they share: `--init-seed` fixes `X₀`, `--problem-seeds` fixes
`(A, B)`, and the tie-break RNG is forked and re-seeded per run so a configuration
does not depend on what ran before it. `grid` also re-runs its winner alone before
reporting it, since in bfloat16 a matmul of a different batch width can round
differently.

Across machines it is *not* bit-exact and cannot be: a different GPU or BLAS perturbs
a gradient at the last bit, `sign` is discontinuous, and an entry within rounding of
zero flips the step by `O(1)` — which is the instability this paper is about. What
survives is everything the tables report, each being a statistic over a trajectory
rather than a single iterate; `synthetic/README.md` gives the measured per-method
agreement, and the `--lmo-dtype bfloat16` default that makes the effect visible.

## 4. Centralized CIFAR-10 (ResNet-18)

ResNet-18, 75 epochs, batch 128, momentum 0.9, cosine-annealed, three seeds. The
protocol and the reasons for it are in
[`centralized/README.md`](centralized/README.md) — in particular why `η₀` is
selected at 75 epochs rather than at a proxy, which `--lr-scaling` to use, and what
the reported metrics mean.

### 4a. Three commands, start to finish

```bash
# 1. on the GPU box
python3 -m centralized.overnight --device cuda:0 --download

# 2. download the file it prints: results/article_export.tar.gz  (~1 MB)

# 3. unpack it next to code/, then
python3 -m centralized.plot_analysis --bundle article_export   # -> results/analysis/
```

The driver calls `centralized.export_article` when it finishes, so the archive is
written for you; running it by hand rebuilds it at any time. That one file is what
to bring home — the run tree stays put, because a 36-job ResNet-18 sweep is ~1.5 GB
of `model.pt`.

Step 3 writes `fig:cifar_results` (`cifar_main`), `fig:cifar_curves_appendix`
(`curves_75ep`), `fig:cifar_train_loss` (`train_loss_75ep`) and `fig:cifar_lr`
(`lr_sensitivity`) as PNG and PDF, and prints the range-matched sweep spreads the
`fig:cifar_lr` caption quotes. Nothing is copied into `aaai_article/`
automatically; do that deliberately.

At the 12.9 s/epoch measured on an RTX A4000 the schedule reads:

```
  phase     jobs  epochs   hours  cumulative   done by
  gain         4      20     0.3         0.3   Sun 02:20
  aux         30      15     1.9         2.2   Sun 04:12
  lr          50      75    15.0        17.2   Sun 19:12
  final       30      75     9.0        26.2   Mon 04:12
  wd           3      75     0.9        27.1   Mon 05:04
```

The archive contains:

| File | What it is |
| :--- | :--- |
| `table_cifar.csv` | **`tab:cifar_main` / `tab:cifar_central`**, aggregated over seeds exactly as the paper defines them — quote from here rather than retyping |
| `runs.csv` | one row per run: config plus every derived summary metric |
| `curves.csv` | tidy per-epoch series; the figures are a function of this and `runs.csv` alone |
| `gain.csv`, `gain_fits.csv` | the `--log-gain` series and its fitted exponent |
| `environment.json`, `hardware.tex` | **exact GPU, driver, CUDA, Python, PyTorch, commit** — §4d |
| `configs.json` | the full config of every run, nothing dropped |
| `overnight/` | `REPORT.md` and `state.json` from the driver |
| `MANIFEST.md` | what each file is, and how many seeds each configuration carries |

> **Redrawing the *submitted* figures, before the re-run lands.** The figures in the
> current submission came from the 2026-07-27 run, whose tree is on the GPU box
> rather than in this repository. `centralized/table2_full.csv` and `curves*.json` —
> `aggregate.py`'s outputs from that run — are committed, and
> `python3 -m centralized.plot_analysis --legacy` reproduces all four figures from
> them **byte for byte** (pinned by
> `test_the_submitted_figures_can_still_be_redrawn`). Its sweep panel is test
> accuracy at 15 epochs, because that is what the old protocol selected on; the
> `--bundle` path is validation accuracy at 75. Delete those files once the re-run
> has replaced the figures, not before.

### 4b. Selection happens at the reporting horizon

The `lr` phase runs at `--final-epochs`, not at a short proxy. Re-running the top
three rates at 75 epochs once **reversed** the 15-epoch ranking for both methods
probed — SignMuon's optimum moved up three lattice steps, Muon's down two — because
each run anneals cosinally to zero over *its own* horizon, so the proxy measures a
different schedule rather than a noisier version of the same one. That costs 15
GPU-h instead of 3.7 and buys the one thing a proxy cannot: the rate in the table is
the rate that won at the horizon the table reports. Selection is still `val_acc` on
the 45k/5k split. The numbers and the argument are in
[`centralized/README.md`](centralized/README.md).

### 4c. The stages by hand

```bash
# The horizon matters: --stage lr must run at the horizon the table reports,
# while aux and alpha compare two arms at one shared horizon and can be short.
TUNE="--dataset cifar10 --model resnet18 --head-adamw always \
      --device cuda:0 --data ./data"

# Is the auxiliary rate method-independent?  (~2 GPU-h)
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

Two diagnostics the paper's appendix quotes but no table depends on:

```bash
python3 -m common.lr_scaling --measure     # single step: incoherent, favours unit-gain
python3 -m centralized.main ... --log-gain --constant-lr    # the driver's `gain` phase
```

The constant rate is not optional there: under a decaying schedule the accumulation
saturates and the fit reports the schedule rather than the alignment. The per-layer
exponent sweep (`--stage alpha`) is available but feeds no paper number — the `gain`
phase measures the exponent directly, and at a single width the exponent is largely
absorbed into `η₀` (the sweep came out flat to within 0.3%).

Use the **same seed set for every method**, so seed *k* means the same
initialization and data order everywhere and the comparison is *paired*. Claim a gap
only when it exceeds the paired std, otherwise write "indistinguishable". All ten
methods are in `tab:cifar_central`, `ef21signmuon` included — the method Theorem 4
says can be made to diverge, present as a predicted-failure baseline that in fact
tops the table.

### 4d. Hardware and software, for the reproducibility appendix

Every run stamps the machine into its own `metrics.json` as it runs
(`common.utils.save_run` → `common.hardware.describe`): GPU model and memory,
driver, CUDA, cuDNN, CPU, RAM, OS, Python and PyTorch versions, and the git commit
plus whether the tree was dirty. Nothing identifying is collected — no hostname, no
username, no absolute paths — so the record is safe in a double-blind submission.

```bash
python3 -m common.hardware                     # this machine, one prose sentence
python3 -m common.hardware --latex             # this machine, as a LaTeX row
python3 -m common.hardware --scan results      # every machine used, by experiment
```

Fill the checklist from these, not from whichever machine compiles the LaTeX: the
experiments ran on different GPUs, so a single "computing infrastructure" sentence
would be wrong for most of the table.

## 5. Federated CIFAR-10 (`tab:exp_3`, `fig:exp_3`)

CNN2, **11 clients**, 3 local accumulation steps, 2000 rounds, batch 64 per step
(= a gradient at batch 192), momentum 0.9, weight decay 0, homogeneous split,
**five seeds** — one federation scale, all eleven methods.

The protocol is the same as §4: a held-out validation split, per-layer rates derived
from the shape, an equal-budget lattice with a boundary check, multi-seed finals.
[`federated/README.md`](federated/README.md) covers what is specific to federation —
the eleven methods and their two uncompressed controls, why exact zeros are
randomized, how `tab:commacct` is computed and which methods actually compress the
downlink, the realized per-layer gain, frozen BatchNorm, and why `N = 11`. **Read
its compression section before quoting any ratio.**

### 5a. Three commands, start to finish

```bash
# 1. on the GPU box
python3 -m federated.overnight --device cuda:0 --budget-hours 24 --download

# 2. download the file it prints: results/federated_export_results.zip  (~1 MB)

# 3. anywhere
python3 -m federated.plot_article --bundle results/federated_export_results.zip
```

The driver calls `federated.export_article` when it finishes, so the archive is
written for you; `python3 -m federated.export_article` rebuilds it at any time from
what is on disk, without retraining. `--bundle` unpacks the `.zip` itself and writes
`fig:exp_3` as `fig_federated_main.pdf` (+ PNG) into `<bundle>/figures/`; copy it
into `aaai_article/images/federated_images/` deliberately. The run tree stays on the
GPU box — 2.9 MB of `model.pt` per job, ~370 MB for a night, none of it needed.

`SUMMARY.md` inside the bundle is the file to read: `tab:exp_3`, the communication
accounting and the per-run diagnostics, in one place. Beside it are
`table_federated.csv` (the table, aggregated over seeds as the paper defines the
columns — quote from here rather than retyping), `communication.csv`, `runs.csv`,
`curves.csv`, `configs.json`, `environment.json`, `hardware.tex`, `MANIFEST.json`,
and `runs/`, which holds each run's `metrics.json` in the original tree shape so
that `--bundle` is just `--root` on the unpacked copy.

Phases: `lr` (η₀ per method on the lattice, 400-round proxy) → `verify` (are the top
rates horizon-stable at 2000?) → `final` (full 50k, seed-major) → `wd` (the decay
ablation at `5e-4`). At the ~0.48 s/round of an A100:

```
  phase     jobs  rounds   hours  cumulative   done by
  lr          60     400     3.9         3.9   Sat 02:30
  verify       4    2000     1.1         5.0   Sat 03:36
  final       60    2000    16.7        21.7   Sat 20:18
  wd           3    2000     0.8        22.5   Sat 21:06
```

Sixty jobs per phase, not fifty-five: Adam is additionally tuned under the sign rule
(`--skip-baseline-variants` turns that off), so twelve tracks cover eleven methods.
The five-seed protocol is ~22 h — give it `--budget-hours 24`, or two nights with
`--resume`. At a smaller budget the driver drops seeds from the end and says which;
a 3-seed table is not what `tab:exp_3` reports. Federated-only variants:
`--partition noniid-labeldir --beta 0.5`, `--final-seeds 0 1 2`, `--wd-ablation 0`.

### 5b. The stages by hand

```bash
FED="--dataset cifar10 --model cnn2 --n_parties 11 --n_steps 3 --batch_size 64 \
     --lr-scaling unit-gain --device cuda:0 --data ./data_federated"

# What grid will each method search, and why?  Run this first.
python3 -m federated.tune --stage anchors $FED
python3 -m federated.tune --stage votes            # the majority-vote alignment table

# Is the auxiliary rate method-independent?  (~30 configs)
python3 -m federated.tune --stage aux $FED --rounds 400

# eta_0 per method, equal budget, selected on val_acc only
python3 -m federated.tune --stage lr $FED --rounds 400 --lr-points 5
```

`--stage anchors` is the sanity check that matters: it prints `λ` per layer per
family and the transported grid anchors, and every tuned rate downstream inherits
them. Then the finals, five seeds, on the full 50k:

```bash
FED11="--model cnn2 --dataset cifar10 --rounds 2000 --n_parties 11 --n_steps 3 \
       --batch_size 64 --momentum 0.9 --lr-scaling unit-gain --split full \
       --data ./data_federated --device cuda:0 --eval_freq 100 --lr-aux 0.001"

for m in signmuon muonusign muonsign ef21signmuon ef21muonusign ef21muonsign \
         muon muonserver signsgd sgd adam; do
  for s in 0 1 2 3 4; do
    python3 -m federated.main $FED11 --algorithm $m --lr <tuned> --seed $s
  done
done
python3 -m aggregate --root results/federated --metric test_acc --csv fed_table.csv
```

Runs land in
`results/federated/fed_cifar10_<algorithm>_homo_cnn2_r2000_c11_s3_unit-gain/seed<s>/`.
Use the same seed set for every method, so the comparison is *paired*, and claim a
gap only when it exceeds the paired std. Then bundle and plot as in §5a, or read a
live tree directly:

```bash
python3 -m federated.export_article                                  # -> the .zip
python3 -m federated.plot_article --root results/federated --n-parties 11
python3 -m federated.plot_federated --metrics gain_spread mv_tie_frac
```

`federated.plot_federated` is the exploratory plotter — one file per metric, any
recorded series you ask for — and is not what the paper prints. Both plotters take
either `--root` (a live tree) or `--bundle` (an archive, unpacked or still zipped).

> **`muonserver` is the row that makes the comparison honest.** There are two
> full-precision Muons, one per template: `muon` orthogonalizes on the worker and
> the server averages the polar factors, while `muonserver` averages first and
> orthogonalizes once — which is what MuonUSign, EF21-MuonUSign and EF21-MuonSign
> *become* when the compressor is the identity. Averaging near-orthogonal matrices
> shortens the step by up to 27% at N=11, so comparing the server-LMO family against
> `muon` alone confounds the cost of the 1-bit uplink with the cost of the
> averaging. It appears in `fig:exp_3` and not in `tab:exp_3`.

### 5c. Reproducing the *previously published* federated table instead

The rates published before this protocol (`icomp_article/`, `paper/main_ru.tex`)
were tuned under the `legacy` per-layer rule — one global rate for the sign family,
Muon's aspect factor for the LMO family — at `N = 10` with weight decay `5e-4`.
`--lr-scaling legacy` reproduces that convention exactly: the driver applies the
shape factor outside the oracle now, and under `legacy` the two conventions coincide
bit for bit (`test_federated_legacy_rule_is_the_old_convention`).

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
| **Selection data** | test accuracy, read from the training log by `federated/grid.py` | `--split tune`: 5k held out of the 50k *before* the client partition, and a tuning run never scores a test image |
| **Per-layer LR** | one global rate (`legacy`) | `--lr-scaling unit-gain`, as centrally |
| **Grid** | ad-hoc `arange` per method | 1–2–5 lattice, equal budget, boundary extension |
| **Weight decay** | `5e-4`, and the auxiliary AdamW was decayed too | `0` primary (matching §4), auxiliary group never decayed, `5e-4` as an ablation |
| **Clients** | 10 (even: the vote ties) | 11 |
| **Seeds** | 1 | 5 |
| **Sign alphabet** | ternary (`sign(0) = 0`) | randomized `±1`, a strict one bit |
| **Data pipeline** | 10 `DataLoader`s, `num_workers=0` | dataset resident on the GPU, augmentation as tensor ops |

Two things to keep stated in the paper: `lr_aux = 0.001` is used throughout and is
not tuned per method (`--stage aux` is the check that it may be held fixed), and
BatchNorm running statistics are never updated from data in the federated setting.

## 6. Language modelling: the nanoGPT speedrun (`tab:nanogpt`, `fig:nanogpt*`)

The one arm that needs rented hardware — **8×H100 for ~25 minutes of training per
method, eight methods** — and the one whose *analysis* needs nothing at all. The
released logs are in the repository, so every number and figure in the paper can be
rebuilt on a laptop without a GPU, without torch, and without downloading FineWeb.
Read [`nanogpt/README.md`](nanogpt/README.md) before changing a setting.

### 6a. Rebuild every nanoGPT number and figure, no GPU (~5 seconds)

```bash
cd code/nanogpt
python parse_logs.py     # ../results/nanogpt/logs -> runs.csv, steps.csv, diagnostics.csv
python make_tables.py    # -> SUMMARY.md, MANIFEST.json, tab_nanogpt.tex, tab_nanogpt_diag.tex
python plot_article.py   # -> ../results/nanogpt/figures/fig_nanogpt_{main,appendix,diag}.pdf
```

`parse_logs.py` and `make_tables.py` are standard library only; `plot_article.py`
needs matplotlib. Then:

```bash
cat ../results/nanogpt/SUMMARY.md               # both tables, the control check, provenance
diff <(sed -n '/tabular/,/tabular/p' ../../aaai_article/signmuon_body.tex) \
     ../results/nanogpt/tab_nanogpt.tex         # the paper's table vs the derived one
```

`SUMMARY.md` is the file to read. It carries `tab:nanogpt`, `tab:nanogpt_diag`, the
record-#40 control check (does the `Muon` arm land on the published curve?), the
wall-clock spread the appendix quotes, and a provenance table saying which build of
`signmuon_optimizers.py` produced each log — the eight runs are **not** all of one
vintage, and `nanogpt/README.md` §"Provenance" says why that is sound.

### 6b. The runs themselves (8×H100)

```bash
cd code/nanogpt
bash setup_env.sh && source .venv/bin/activate   # venv + a real Flash-Attention-3 fetch
python data/cached_fineweb10B.py 9               # ~900M train tokens + the val chunk
NANOGPT_ITERS=200 bash run_all.sh                # smoke pass first: do this once
bash run_all.sh                                  # the eight hero runs (~25 min each)
```

`run_all.sh` preflights the environment once rather than eight times, runs `Muon`
first (it *is* record #40's optimizer, so it must land on record #40's published
curve — `reference_record40.csv` is the pass/fail line), and writes logs into
`../results/nanogpt/logs/`. Then rerun §6a. No FA3 on your node, or only one GPU?
`NPROC=8 SCRIPT=train_gpt_a100.py bash run_all.sh` swaps FA3 → FlexAttention and
FP8 → bf16, both outside the optimizer.

Before renting anything, run the two CPU tests (they need torch, nothing else):

```bash
SIGNMUON_NO_COMPILE=1 python test_signmuon_optimizers.py   # update rules == the paper's
python test_distributed_sharding.py                        # sharded step() == centralized
```

### 6c. What is fixed a priori, and what is not

Nothing here is tuned by us. Every hyperparameter outside the matrix optimizer is
record #40's own; the five LMO-family methods run at the record's own `η₀ = 0.06`,
and the three sign-family methods at `0.03`, the spectral discount derived in
`app:lrscale` and defended three independent ways in `train_gpt.py`'s
`OPTIMIZER_CONFIG` comment. So every contrast in `tab:nanogpt` is
matched-hyperparameter. The one number with real uncertainty is that `0.03`;
`SIGN_PROBE_LR=0.01 bash run_all.sh` adds one downside probe per sign method.

Unlike §4 and §5, this arm is **single-run**: the table quotes record #40's own
five-seed spread (`± 0.0009`) as the noise scale instead, so differences below
~0.003 are not results. The runs in the repository predate the `SIGNMUON_SEED` knob
and are unseeded; see `nanogpt/README.md` §"Provenance".

## 7. Multi-seed runs

Any command above becomes a multi-seed sweep by varying `--seed`; the seed is part
of the output path, so nothing is overwritten:

```bash
for s in 0 1 2 3 4; do
  python3 -m federated.main $FED11 --algorithm ef21muonsign --lr 0.05 --seed $s
done

python3 -m aggregate --metric test_acc --csv summary.csv --curves curves.json
```

`aggregate.py` groups runs by configuration-minus-seed (and minus device/data-path
fields) and reports mean ± sample std; `curves.json` holds the pointwise mean/std
curves for error-band plots. It scans all of `results/` by default
(`--root results/federated` for one family). It is the **federated** path's
aggregator; the centralized one goes through `centralized.export_article`, whose
`table_cifar.csv` is the paper's table itself rather than a generic summary.

Both arms report mean ± sample std over seeds: three for `tab:cifar_main` and
`tab:cifar_central`, five for `tab:exp_3`. A single seed measures no dispersion at
all, so the tools print a blank rather than `± 0.00` — do not read a blank as
agreement. Claim a gap only when it exceeds the seed spread.

---

## Notes

* **Plotting is scripts, not notebooks.** The four notebooks under `notebooks/` read
  the pre-reorganization `saves*` paths and were removed on 2026-07-28. Each has a
  replacement that reads `results/`:

  | figure | script |
  | :--- | :--- |
  | `fig:divergence_plot` | `counterexamples.run_counterexamples` |
  | `fig:ef21_momentum` | `counterexamples.plot_ef21_momentum` |
  | `fig:synthetic_*` | `synthetic.plot_synthetic` |
  | `fig:cifar_*` | `centralized.plot_analysis --bundle …` (§4a) |
  | `fig:exp_3` | `federated.plot_article` |
  | `fig:nanogpt*` | `nanogpt/plot_article.py` |

* **One style, in [`common/plotting.py`](common/README.md#plottingpy).** Every script
  above calls `use_paper_style()` and takes its colours from `color_of`, so a method
  keeps its colour across figures and every figure matches `aaai2027.sty` — Times
  text, STIX math, and TrueType outlines rather than the Type 3 fonts matplotlib
  emits by default and AAAI forbids. Figures are authored at their printed width
  (`TEXT_WIDTH` / `COLUMN_WIDTH`) and included at `width=\textwidth` /
  `\columnwidth`, so a 9 pt label is 9 pt on the page. Do the same for a new one
  rather than styling it locally.
* **`nanogpt/`** has its own [README](nanogpt/README.md), which is the place to read
  before changing a setting; §6 above is the command reference. Training needs 8×H100,
  but the released logs are in the repository and every table and figure rebuilds from
  them on a laptop.
* **Disk.** `SIGNMUON_RESULTS` relocates the whole `results/` tree and every script
  follows it; `--data` is separate and also worth pointing off the system disk. One
  `model.pt` is 2.9 MB for CNN2 and 42.7 MB for ResNet-18, so the ~127-job federated
  night is ~370 MB and a centralized sweep ~1.5 GB. See [README.md](README.md).
