# `federated/` — all eleven methods, one driver

CNN2 on CIFAR-10, **11 clients**, 3 local accumulation steps, 2000 rounds, batch
64 per step, homogeneous split, five seeds. The experiment behind `tab:exp_3`,
`tab:commacct` and `fig:exp_3`.

## Three commands

```bash
# 1. on the GPU box: compute everything, then bundle it
cd code
python3 -m federated.overnight --device cuda:0 --budget-hours 24 --download

# 2. download the one file it prints at the end:
#    results/federated_export_results.zip   (~1 MB)

# 3. anywhere: unpack it and draw the figure
python3 -m federated.plot_article --bundle results/federated_export_results.zip
```

Step 1 runs `lr` → `final` → `wd`: rates tuned at the horizon the table reports,
full-50k finals over five seeds, then the weight-decay ablation. (`verify` is in the
default phase list but is vacuous while the two horizons agree, and says so.) Add
`--phases lr final rules wd` for the per-layer rule ablation, ~4 h more. It calls
`federated.export_article` at the end, and step 3 unpacks the `.zip` itself. The
run tree stays on the GPU box: one 2.9 MB `model.pt` per job, ~370 MB a night,
and the paper needs no weights.

Under `--budget-hours` the driver fits the plan to the deadline, shedding the
ablations, then seeds, then the tuning horizon, and **naming each concession** in
`REPORT.md`. A shortened horizon puts the proxy back, so it re-enables the
horizon-stability check and warns; `--budget-hours 0` means no deadline.

Read `SUMMARY.md` in the bundle first: `tab:exp_3`, the communication accounting
and the diagnostics in one file.

## Two heterogeneous experiments for the rebuttal

Copy the updated `code/` directory to the GPU machine. From that directory,
run one installment at a time: transfer, priority tuning, then remaining tuning.
Send each ZIP back for analysis before starting the next installment.

```bash
# 1. All ten article algorithms, homogeneous hyperparameters (50 finals).
python -m federated.heterogeneous --experiment transfer --device cuda:0 --download

# 2. Priority algorithms: LR tuning and five-seed finals.
python -m federated.heterogeneous --experiment tuned --tune-part priority --device cuda:0 --download

# 3. Remaining algorithms: the same LR tuning and five-seed finals.
python -m federated.heterogeneous --experiment tuned --tune-part remaining --device cuda:0 --download
```

At the end of **each** command, download the file printed after `Download:` and
send it back. The cumulative `rebuttal_results.zip` includes all completed work
from the three commands; the summary identifies unfinished methods. All three
commands share the same protocol directory and automatically resume.

Both use **CNN2/CIFAR-10, Dirichlet label skew with concentration 0.1**, 11 clients,
full participation, 2000 rounds, and 3 x 64 gradient accumulation at the same
broadcast model. Client objectives have equal weight. All other training choices
match the homogeneous experiment: cosine schedule, unit-gain scaling,
momentum 0.9 with each method's existing convention, zero weight decay, and
auxiliary AdamW LR 0.001. **Momentum is fixed throughout; only LR is tuned.**

- **Transfer:** keep the homogeneous validation-selected LR and momentum 0.9.
  All ten methods run together, without splitting transfer into parts
  (50 final runs). This measures robustness to heterogeneity without retuning.
- **Tuned, priority:** `signmuon`, `ef21muonusign`, `ef21muonsign`,
  `ef21signmuon`, and `muon` (25 initial validation trials plus up to 25 finals).
  These compare the main practical method, both convergent EF21 placements,
  divergent EF21-SignMuon, and full-precision Muon.
- **Tuned, remaining:** `muonusign`, `muonsign`, `signsgd`, `sgd`, and `adam`
  (25 initial validation trials plus up to 25 finals).

Both tuning parts use **the homogeneous 1-2-5 LR lattice and initial CNN2 search
centers**. Each method starts with five configurations. A boundary winner adds
two points beyond that endpoint, with at most three extensions (at most eleven
LRs per method), matching `federated.overnight`. Each method gets the same
search rule, one tuning seed, and the full 2000-round horizon. Select
by the last-five validation mean on the existing 45k/5k split, then retrain on
all 50k images. Auxiliary LR stays fixed. All ten algorithms are the ten rows
of the article's main federated table.

Use **five paired final seeds**. A seed determines
initialization and client partition, shared across methods and experiments.
The transfer LRs, taken from the saved homogeneous validation search, are:

| Method | LR | Tuning part |
| :-- | --: | :-- |
| `signmuon` | 0.1 | priority |
| `ef21muonusign` | 0.01 | priority |
| `ef21muonsign` | 0.05 | priority |
| `ef21signmuon` | 0.02 | priority |
| `muon` | 0.1 | priority |
| `muonusign` | 0.05 | remaining |
| `muonsign` | 0.02 | remaining |
| `signsgd` | 0.01 | remaining |
| `sgd` | 0.05 | remaining |
| `adam` | 0.001 | remaining |

Transfer answers whether the homogeneous settings remain effective; tuning checks
whether adapting LR changes the comparison. A winner still at a search endpoint
after the extension budget is flagged in the summary. `--lr-points`,
`--lr-extend-rounds`, and `--lr-extend-points` control the search; `--tune-seeds`
controls replication. `--methods` can restrict the family for a smoke run;
`--tune-part` affects only tuning and never reduces the transfer method list.

`--dry-run` prints the complete plan without training, downloads, or writes.
Rerunning either command automatically resumes completed jobs. Identical finals
are reused if tuning selects the transfer setting. `--retry-failed` retries
recorded failures; `--report-only` rebuilds outputs without training. `--rounds`
sets **both** horizons; a shorter smoke run is a separate protocol.

Outputs are in `results/federated_heterogeneous/<protocol-id>/`: read `SUMMARY.md`
or `table.csv` for mean +/- sample SD of per-seed last-five test accuracy, and
`tuning.csv` for all validation trials. Download `rebuttal_results.zip` for the
tables, state, full configs, and curves; weights stay on the training machine.
Per-job logs are included for diagnosing failed trials from the same ZIP.
The raw runs are isolated from the homogeneous article export. Changing training
settings creates a separate protocol directory. Search settings are recorded
separately and must stay fixed once tuning starts; switching between transfer,
priority tuning, and remaining tuning keeps the same directory.
`SIGNMUON_RESULTS` relocates the whole tree.

The original transfer cache is adopted automatically when upgrading from the
first fixed-grid driver. Completed 2000-round runs are retained; the revised
search protocol and any boundary extensions are recorded in `state.json`.
Keep the original results directory on the GPU machine to reuse those runs.
The ZIP is an analysis bundle; resuming uses that machine's original run tree.

### Continue a boundary search for MuonSign

After the three installments, MuonSign selected the lowest tested LR, `5e-5`.
From the same `code/` directory on the GPU machine, run:

```bash
python -m federated.heterogeneous --experiment tuned --tune-part remaining --refine-methods muonsign --device cuda:0 --download
```

`--refine-methods` selects work inside the existing ten-method cache. Keep the
original results tree and default `--methods` list. The command resumes MuonSign's
validation grid, first adding `1e-5` and `2e-5`, and continues with two lattice
points per extension until the validation winner has tested LRs on both sides.
The original three-extension cap does not apply to this explicit continuation.
Momentum, seeds, validation split, 2000-round horizon, and training settings stay
fixed. Test results do not enter selection. The selected LR gets five final
seeds; matching completed runs are reused.

The previous selected LR and continuation history are recorded in `state.json`
and `SUMMARY.md`. All other results remain in the same cumulative
`rebuttal_results.zip`; send that ZIP back when the command finishes. Rerunning
resumes an interrupted continuation, and `--retry-failed` retries failed jobs.
`--dry-run` shows the cached grid and next rates without training or writes.

On the uploaded hardware, each new validation LR takes about 3.4 minutes, and
five new final seeds take about 17 minutes. One extension plus new finals costs
about 24 minutes; each further extension adds about 7 minutes.

Rebuttal setup sentence: "CNN2/CIFAR-10, 11 clients, Dirichlet(0.1) label skew;
the homogeneous protocol and five paired seeds are retained."

| File | What it does |
| :--- | :--- |
| [`algorithms.py`](algorithms.py) | `run_federated`, *the* federated implementation; re-exports everything in `methods.py` |
| [`methods.py`](methods.py) | The `MethodSpec` registry and the bit accounting. No torch, so the export runs anywhere |
| [`main.py`](main.py) | Entry point: one run, one config, one `metrics.json` |
| [`data.py`](data.py) | Partitioning, the 45k/5k split, device-resident shards |
| [`tune.py`](tune.py) | Equal-budget, validation-only learning-rate search |
| [`overnight.py`](overnight.py) | **Step 1**: the whole protocol, resumable |
| [`heterogeneous.py`](heterogeneous.py) | Homogeneous-parameter transfer and validation-tuned label-skew experiments |
| [`export_article.py`](export_article.py) | **Step 2**: pack the results into one `.zip` |
| [`plot_article.py`](plot_article.py) | **Step 3**: `fig_federated_main.pdf`, the figure the paper prints |
| [`plot_federated.py`](plot_federated.py) | Exploratory plotter: one file per metric, also takes `--bundle` |
| [`grid.py`](grid.py) | Retired: it selected on the test set; points at `tune.py` |

Other useful commands:

```bash
python3 -m federated.overnight --dry-run      # the schedule, and nothing else
python3 -m federated.export_article           # rebuild the bundle without retraining
python3 -m federated.tune --stage anchors     # per-layer multipliers and grid anchors
python3 -m federated.tune --stage votes       # the majority-vote table at the bottom of this file
```

Running the stages by hand is in [`../REPRODUCE.md`
§5](../REPRODUCE.md#5-federated-cifar-10-tabexp_3-figexp_3).
[`../README.md`](../README.md) has the conventions shared with the centralized
experiment: parameter routing, cosine schedule, decoupled weight decay, EMA
momentum, derived per-layer rate. This file covers only what federation adds.

### What is in the bundle

| File | What it is |
| :--- | :--- |
| `SUMMARY.md` | every table in one file; **this is the one to read** |
| `table_federated.csv` | `tab:exp_3`, aggregated over seeds as the paper defines the columns |
| `communication.csv` | `tab:commacct`, from each run's own alphabet |
| `runs.csv` | one row per run: config plus every derived summary metric |
| `curves.csv` | tidy per-round series; the figures are a function of this alone |
| `configs.json` | the full config of every run, nothing dropped |
| `environment.json`, `hardware.tex` | every machine that contributed, for the reproducibility appendix |
| `MANIFEST.json` | commit, argv, run counts, and what was left out and why |
| `runs/` | each run's `metrics.json`, minus the weights |

`runs/` keeps the directory shape of the results tree, so `--bundle` is just
`--root` on the unpacked copy and no figure code need know bundles exist.

## One driver, eleven methods

Every method is one row of `tab:fed_master`: where the LMO is evaluated and what
compresses each channel. Adding one is a single dict entry, so the eleven cannot
drift apart in schedule, routing, evaluation or weight decay.

| Method | LMO | Uplink | Downlink | Family | Uncompressed control |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `signmuon` | worker | sign / MV | exact † | `sign` | `muon` |
| `ef21signmuon` | worker | EF21 | exact | `lmo` | `muon` |
| `muonusign` | server | sign / MV | exact | `lmo` | `muonserver` |
| `muonsign` | server | sign / MV | sign | `sign` | `muonserver` |
| `ef21muonusign` | server | EF21 | exact | `lmo` | `muonserver` |
| `ef21muonsign` | server | EF21 | EF21-P | `lmo` | `muonserver` |
| `muon` | worker | exact (average) | exact | `lmo` | — |
| `muonserver` | server | exact (average) | exact | `lmo` | — |
| `signsgd` | none | sign / MV | exact † | `sign` | `sgd` |
| `sgd`, `adam` | none | exact (average) | exact | — | — |

† These fields say which compressor the loop *applies*. SignMuon and SignSGD
apply none on the downlink yet are one bit each way, because the server
distributes the majority vote itself; see the compression section.

`sgd` and `adam` have no per-layer family on purpose: their step norm is
data-dependent, so no static multiplier implements the unit-gain criterion.
`--scale-baselines` gives Adam the sign rule for the ablation.

**The load-bearing test** is `test_federated_one_client_equals_centralized`: at
`N = 1` this driver must reproduce `common/optimizers.py` exactly, for all eight
matrix rules, under both the `legacy` and `unit-gain` conventions, on a square
and a tall matrix. Nothing else stops the two implementations drifting apart.

### Two full-precision Muons, and why one is not enough

`muon` orthogonalizes on the **worker** and the server averages the polar
factors; `muonserver` sends the momentum uncompressed and applies **one** LMO on
the server. Each is the uncompressed control for a different template:
`muonserver` is what MuonUSign, EF21-MuonUSign and EF21-MuonSign become under an
identity compressor. They are not interchangeable, since averaging
near-orthogonal matrices *shortens* the step. On a 120×4608 matrix, clients
sharing a signal plus independent noise of relative size `q`, in units of
`‖·‖_F/√min(m,n)`: at `q = 1` the average of polar factors falls from 0.769 at
`N = 1` to 0.559 at `N = 11` and 0.548 at `N = 21`; at `q = 3`, to 0.324 and
0.286. `polar(mean)` is flat to 0.3% throughout. So `muon`'s effective step
depends on the client count *and* on the gradient-to-noise ratio, which drifts
over training; a tuned `η₀` absorbs a constant handicap, not a drifting one, and
comparing the server-LMO family against `muon` would confound the 1-bit uplink's
cost with the averaging's.
`test_averaging_polar_factors_shrinks_the_step_but_polar_of_the_average_does_not`
pins the direction of both effects.

## What the compression actually saves

Every run prints the accounting rather than a headline (`communication_bits`,
the source of `tab:commacct`). On CNN2, 762,560 matrix parameters, 2,146
auxiliary, 3 matrix layers:

| method | uplink | downlink | **round trip** |
| :--- | ---: | ---: | ---: |
| `signmuon`, `signsgd`, `muonsign`, `ef21muonsign` | 29.4× | 29.4× | **29.4×** |
| `muonusign`, `ef21signmuon`, `ef21muonusign` | 29.4× | 1× | **1.9×** |
| `muon`, `muonserver`, `sgd`, `adam` | 1× | 1× | 1× |

Two corrections to a nominal "32×" are folded in: the auxiliary group rides
along uncompressed both ways, which alone puts a perfect 1-bit uplink at 1.087
bits/parameter model-wide (29.4×, not 32×); and each EF21 channel carries one
full-precision scale per matrix layer, four decimal places here but not on a
deeper model.

The split between the two compressed rows is **not** `spec.downlink != "exact"`;
that reading is the bug `compresses_downlink` prevents. The criterion is whether
the object the server distributes is *already* `±1`-valued. For SignMuon and
SignSGD it is: the vote `sign(Σⱼ sⱼ)` is the broadcast, each client applies
`X ← X − ηλ·s_agg` to its own replica, and replicas from a common `X₀` receive
identical updates and never drift. Dense server-side quantities fail: `polar(·)`
of the aggregate (MuonUSign, EF21-MuonUSign) and scaled averages of signs
(EF21-SignMuon) force a full-precision broadcast, capping those three at a 2×
round trip however good the uplink. That argues *for* the bidirectional methods,
not against the paper.

## Three things the numbers depend on

### Exact zeros are randomized, so every sign channel is a strict 1 bit

`sign(0) = 0` would make a client transmit from `{−1, 0, +1}`, and that is no
corner case: `polar(M)` has an exactly-zero column wherever `M` does, and `M`
does wherever a feature was zero across the whole local batch, routine after
ReLU and MaxPool. Measured on CNN2 over the five-seed runs at **0.1–3.0% of raw
sign entries per round**, and strongly method-dependent: SignSGD and MuonSign
1.4–3.0%, MuonUSign 0.1–1.4%, SignMuon 0.19–0.20%. Signing the LMO *output*
produces far fewer exact zeros than signing the momentum does.

The default maps each zero to an independent random `±1` (`sign_pm1`) on
**every** sign channel: the majority-vote uplink, both EF21 residual channels,
the MuonSign downlink. The channel is then one bit per parameter *whatever the
zero rate*, so `uplink_zero_frac` and `mv_tie_frac` are diagnostics only, and
`‖s‖_F = √(mn)` and the contraction identity `‖C(Y)−Y‖²_F = ‖Y‖²_F − ‖Y‖²₁/d`
hold exactly, not approximately. With `±1` messages an odd `N` cannot tie.
`--uplink-zeros keep` restores the ternary channel, the only setting under which
a zero rate costs bits.

### Verifying the per-layer rule (`gain_spread`)

The unit-gain rule equalizes the realized gain `γ = ‖λ·s‖_F/√fan_out` across
layers, so `η₀` means one thing everywhere. Every run checks it: the profile
prints at round 1, `gain_spread` is recorded at every evaluation, and the driver
warns above 1.15×. The sign family is exactly 1.00×, its steps ±1 with no zeros;
the LMO family is not, because the derivation assumes the oracle returns
`‖polar‖_F = √min(m,n)` and Newton–Schulz returns less, by a **shape-dependent**
amount. Measured on `muonserver`:

| dtype | `ns_steps` | gain spread | conv1 | conv2 | fc1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | **5** | **1.34×** | 0.875 | 0.943 | 0.705 |
| bfloat16 | 7 | 1.11× | 0.888 | 0.940 | 0.846 |
| float32 | 5 | 1.35× | 0.873 | 0.945 | 0.700 |

Live runs land in 1.2–1.35× at `ns_steps=5`, always on `fc1` (120×4608, aspect
38:1), which holds 72% of CNN2's matrix parameters and realizes ~0.70–0.78 of
the assumed gain; a tuned `η₀` is one constant and cannot absorb a per-layer
factor. **The default stays `--ns_steps 5`**, what reference Muon uses and what
the reported numbers used, because 1.34× is under half a lattice step (2–2.5×)
and cannot flip a ranking. `--ns_steps 7` roughly halves the excess; nothing
reaches 1.0, since the quintic oscillates in a band around 1 rather than
converging. The log prints **two** "spread" lines: `describe_rule`'s is over the
*multipliers*; the one that matters is labelled `gain spread`.

### BatchNorm statistics never update

Local models are discarded each round and BatchNorm runs in inference mode
during accumulation, so the running statistics stay at their initialization
`(0, 1)` for the whole run, in training and evaluation alike. BatchNorm is
therefore a fixed normalization with learnable affine parameters:
self-consistent, applied identically to every method, and what the reported
numbers used. It is also one reason a channel can be inactive across a whole
local batch, contributing an exactly zero row to the momentum.

`--live-bn-stats` changes that, and changes the numbers. What it does *not* do:
the local model is rebuilt from the global one every round and never written
back, so its statistics are discarded and evaluation still uses `(0, 1)`. The
flag *introduces* a train/eval mismatch rather than removing one.

## Per-layer rates and grid anchors

CNN2's three matrix parameters have `fan_in` 75, 1600 and 4608, so the sign
family's multiplier spans **7.8×**: at one global rate a SignMuon step is 8.7×
the corresponding Muon step on `conv1` and **67.9×** on `fc1`. That spread is
wider than ResNet-18's, with no single shape dominating the parameter count,
making CNN2 the better instrument for the rule. Grid anchors are *transported*,
not re-guessed:
`anchor(rule) = anchor_legacy · geomean(λ_legacy)/geomean(λ_rule)` on the
model's own shapes, which leaves the LMO family untouched and multiplies the
sign family by 28.65. `--stage anchors` prints all of it.

## Why 11 clients

Alignment `A = E[⟨truth, ŝ⟩]/(mn)`, the fraction of a full-strength descent step
actually delivered, over 400k coordinates with each client correct with
probability 0.65 and silent with probability `q` (`--stage votes`):

| `N` | tie % (`q`=0) | `A` (`q`=0) | tie % (`q`=0.10) | `A` (`q`=0.10) |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.00 | 0.6573 | 8.38 | 0.6183 |
| 10 | 15.32 | 0.6558 | 9.23 | 0.6410 |
| **11** | **0.00** | **0.7029** | **7.22** | **0.6676** |
| 15 | 0.00 | 0.7747 | 5.54 | 0.7406 |

At `q = 0`, the operating regime under the randomized convention, the classical
parity pathology shows: `N = 10` gains nothing over `N = 9`, since an even
vote's extra voter is never decisive and only creates ties. `N = 11` is both
better aligned than 10 and unable to tie. The table is Monte Carlo, so
differences below ~0.002 are noise, as the `N = 9` against `N = 10` pair itself
shows.

## Speed

`--eval_freq` defaults to 100 (2000 evaluations of 10k images used to dominate a
run), and `--loader gpu`, the default on CUDA, uploads the dataset once as
`uint8` and does crop, flip and normalization as tensor ops on the device
instead of eleven `DataLoader`s doing PIL decode on the main thread. The
augmentation is torchvision's, and
`test_gpu_crop_and_flip_match_torchvision_distributionally` compares the per-row
padding probability against `RandomCrop` over 2000 draws rather than checking
shapes.
