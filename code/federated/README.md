# `federated/` — all eleven methods, one driver

CNN2 on CIFAR-10, **11 clients**, 3 local accumulation steps, 2000 rounds, batch 64
per step, homogeneous split, five seeds. The arm behind `tab:exp_3`, `tab:commacct`
and `fig:exp_3`.

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

Step 1 runs the whole protocol — tune, verify, finals, ablation — and calls
`federated.export_article` when it finishes, so the archive is written for you.
Step 3 unpacks the `.zip` itself; you do not have to. The run tree stays on the
GPU box: it is one 2.9 MB `model.pt` per job, ~370 MB for a night, and nothing in
the paper needs the weights.

Read `SUMMARY.md` inside the bundle first — it holds `tab:exp_3`, the
communication accounting and the diagnostics in one file.

| File | What it does |
| :--- | :--- |
| [`algorithms.py`](algorithms.py) | `run_federated` + the `MethodSpec` registry — *the* federated implementation |
| [`main.py`](main.py) | Entry point: one run, one config, one `metrics.json` |
| [`data.py`](data.py) | Partitioning, the 45k/5k split, device-resident shards |
| [`tune.py`](tune.py) | Equal-budget, validation-only learning-rate search |
| [`overnight.py`](overnight.py) | **Step 1**: the whole protocol, resumable |
| [`export_article.py`](export_article.py) | **Step 2**: pack the results into one `.zip` |
| [`plot_article.py`](plot_article.py) | **Step 3**: `fig_federated_main.pdf`, the figure the paper prints |
| [`plot_federated.py`](plot_federated.py) | Exploratory plotter: one file per metric, also takes `--bundle` |
| [`grid.py`](grid.py) | Retired — it selected on the test set; points at `tune.py` |

Other useful commands:

```bash
python3 -m federated.overnight --dry-run      # the schedule, and nothing else
python3 -m federated.export_article           # rebuild the bundle without retraining
python3 -m federated.tune --stage anchors     # per-layer multipliers and grid anchors
python3 -m federated.tune --stage votes       # the majority-vote table at the bottom of this file
```

Running the stages by hand is in
[`../REPRODUCE.md` §5](../REPRODUCE.md#5-federated-cifar-10-tabexp_3-figexp_3).
Conventions shared with the centralized arm — parameter routing, the cosine
schedule, decoupled weight decay, EMA momentum, the derived per-layer rate — are in
[`../README.md`](../README.md); this file covers what is specific to federation.

### What is in the bundle

| File | What it is |
| :--- | :--- |
| `SUMMARY.md` | every table in one file — **this is the one to read** |
| `table_federated.csv` | `tab:exp_3`, aggregated over seeds as the paper defines the columns |
| `communication.csv` | `tab:commacct`, from each run's own alphabet |
| `runs.csv` | one row per run: config plus every derived summary metric |
| `curves.csv` | tidy per-round series; the figures are a function of this alone |
| `configs.json` | the full config of every run, nothing dropped |
| `environment.json`, `hardware.tex` | every machine that contributed, for the reproducibility appendix |
| `MANIFEST.json` | commit, argv, run counts, and what was left out and why |
| `runs/` | each run's `metrics.json`, minus the weights |

`runs/` keeps the directory shape of the results tree, which is why `--bundle` is
just `--root` on the unpacked copy and no figure code has to know that bundles
exist.

## One driver, eleven methods

Every method is one row of `tab:fed_master`: where the LMO is evaluated, and what
compresses each channel. Adding a method is one dict entry, which is why the eleven
cannot drift apart in schedule, routing, evaluation or weight decay.

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

† These fields say which compressor the loop *applies*. SignMuon and SignSGD apply
none on the downlink and are still one bit each way, because the object the server
distributes is the majority vote itself — see the compression section below.

`sgd` and `adam` have no per-layer family on purpose: their step norm is
data-dependent, so no static multiplier implements the unit-gain criterion.
`--scale-baselines` gives Adam the sign rule for the ablation.

**The load-bearing test** is `test_federated_one_client_equals_centralized`: this
driver at `N = 1` must reproduce `common/optimizers.py` exactly, for all eight
matrix rules, under both the `legacy` and `unit-gain` conventions, on a square and
on a tall matrix. Nothing else stops the two implementations drifting apart.

### Two full-precision Muons, and why one is not enough

`muon` orthogonalizes on the **worker** and the server averages the polar factors;
`muonserver` sends the momentum uncompressed and applies **one** LMO on the server.
Each is the uncompressed control for a different template — `muonserver` is what
MuonUSign, EF21-MuonUSign and EF21-MuonSign *become* when the compressor is the
identity — and they are not interchangeable, because averaging near-orthogonal
matrices *shortens* the step. Measured on a 120×4608 matrix with clients sharing a
signal plus independent noise of relative size `q`, as `‖·‖_F/√min(m,n)`: at `q = 1`
the average of polar factors falls from 0.769 at `N = 1` to 0.559 at `N = 11` and
0.548 at `N = 21`; at `q = 3`, to 0.324 and 0.286. `polar(mean)` is flat to 0.3%
throughout. So `muon`'s effective step depends on the client count *and* on the
gradient-to-noise ratio, which drifts over training — a tuned `η₀` absorbs a
constant handicap, not that one — and comparing the server-LMO family against
`muon` would confound the cost of the 1-bit uplink with the cost of the averaging.
`test_averaging_polar_factors_shrinks_the_step_but_polar_of_the_average_does_not`
pins the direction of both effects.

## What the compression actually saves

Every run prints the accounting rather than quoting a headline
(`communication_bits`, the source of `tab:commacct`). On CNN2 — 762,560 matrix
parameters, 2,146 auxiliary, 3 matrix layers:

| method | uplink | downlink | **round trip** |
| :--- | ---: | ---: | ---: |
| `signmuon`, `signsgd`, `muonsign`, `ef21muonsign` | 29.4× | 29.4× | **29.4×** |
| `muonusign`, `ef21signmuon`, `ef21muonusign` | 29.4× | 1× | **1.9×** |
| `muon`, `muonserver`, `sgd`, `adam` | 1× | 1× | 1× |

Two corrections to a nominal "32×" are folded in: the auxiliary group rides along
uncompressed in both directions, which alone puts a perfect 1-bit uplink at 1.087
bits/parameter model-wide (29.4×, not 32×); and each EF21 channel carries one
full-precision scale per matrix layer, four decimal places in here but not on a
deeper model.

The split between the two compressed rows is **not** `spec.downlink != "exact"` —
that reading is the bug `compresses_downlink` exists to prevent. The criterion is
whether the object the server must distribute is *already* `±1`-valued, and for
SignMuon and SignSGD it is: the vote `sign(Σⱼ sⱼ)` is itself the broadcast, each
client applies `X ← X − ηλ·s_agg` to its own replica, and replicas from a common
`X₀` receive identical updates and never drift. What fails is a dense server-side
quantity — `polar(·)` of the aggregate (MuonUSign, EF21-MuonUSign) or a scaled
average of signs (EF21-SignMuon) — and those three must broadcast a full-precision
model, so their round trip cannot exceed 2× however good the uplink is. That is the
argument *for* the bidirectional methods, not against the paper.

## Three things the numbers depend on

### Exact zeros are randomized, so every sign channel is a strict 1 bit

`sign(0) = 0` would make a client transmit from `{−1, 0, +1}`, and that is not a
corner case: `polar(M)` has an exactly-zero column wherever `M` does, and `M` does
wherever a feature was zero across the whole local batch — after ReLU and MaxPool,
routinely. Measured at **8–17% of raw sign entries per round** on SignMuon's uplink
on CNN2.

The default maps each zero to an independent random `±1` (`sign_pm1`) on **every**
sign channel — the majority-vote uplink, both EF21 residual channels, and the
MuonSign downlink. The channel is then one bit per parameter *whatever the zero
rate*, which is why `uplink_zero_frac` and `mv_tie_frac` are now diagnostics that
feed no accounting, and why `‖s‖_F = √(mn)` and the contraction identity
`‖C(Y)−Y‖²_F = ‖Y‖²_F − ‖Y‖²₁/d` hold exactly rather than approximately. With `±1`
messages an odd `N` cannot tie at all. `--uplink-zeros keep` restores the ternary
channel and is the only setting under which a zero rate costs bits.

### Is the per-layer rule doing its job? (`gain_spread`)

The unit-gain rule exists to make the realized gain `γ = ‖λ·s‖_F/√fan_out` the same
on every layer, so that `η₀` means one thing everywhere. Every run checks it: the
profile is printed at round 1, `gain_spread` is recorded at every evaluation, and
the driver warns above 1.15×. The sign family is exactly 1.00× — its steps are ±1
with no zeros — but the LMO family is not, because the derivation assumes the oracle
returns `‖polar‖_F = √min(m,n)` and Newton–Schulz returns less, by a
**shape-dependent** amount. Measured on `muonserver`:

| dtype | `ns_steps` | gain spread | conv1 | conv2 | fc1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | **5** | **1.34×** | 0.875 | 0.943 | 0.705 |
| bfloat16 | 7 | 1.11× | 0.888 | 0.940 | 0.846 |
| float32 | 5 | 1.35× | 0.873 | 0.945 | 0.700 |

Live runs land in 1.2–1.35× at `ns_steps=5`, and the offender is always `fc1`
(120×4608, aspect 38:1), which holds 72% of CNN2's matrix parameters and realizes
~0.70–0.78 of the assumed gain; a tuned `η₀` is one constant and cannot absorb a
per-layer factor. **The default stays `--ns_steps 5`** — what reference Muon uses
and what the reported numbers used — because 1.34× is under half a lattice step
(2–2.5×) and so cannot flip a ranking; `--ns_steps 7` roughly halves the excess, and
nothing reaches 1.0, since the quintic oscillates in a band around 1 rather than
converging. Note the log has **two** lines saying "spread": `describe_rule`'s is the
spread of the *multipliers*; the one that matters is labelled `gain spread`.

### BatchNorm statistics never update

Local models are discarded each round and BatchNorm runs in inference mode during
accumulation, so the running statistics stay at their initialization `(0, 1)` for
the whole run, in training and evaluation alike. BatchNorm is therefore a fixed
normalization with learnable affine parameters — self-consistent, applied
identically to every method, and what the reported numbers used. It is also one
reason a channel can be inactive across a whole local batch and contribute an
exactly zero row to the momentum.

`--live-bn-stats` changes it, and changes the numbers. Note what it does *not* do:
the local model is rebuilt from the global one every round and never written back,
so the statistics it accumulates are discarded and evaluation still uses `(0, 1)` —
the flag *introduces* a train/eval mismatch rather than removing one.

## Per-layer rates and grid anchors

CNN2's three matrix parameters have `fan_in` 75, 1600 and 4608, so the sign family's
multiplier spans **7.8×**, and at one global rate a SignMuon step is 8.7× the
corresponding Muon step on `conv1` and **67.9×** on `fc1`. That is a wider spread
than ResNet-18 offers with no single shape dominating the parameter count, which
makes CNN2 the better instrument for the rule. Grid anchors are *transported* rather
than re-guessed — `anchor(rule) = anchor_legacy · geomean(λ_legacy)/geomean(λ_rule)`
on the model's own shapes, which leaves the LMO family untouched and multiplies the
sign family by 28.65. `--stage anchors` prints all of it.

## Why 11 clients

Alignment `A = E[⟨truth, ŝ⟩]/(mn)` — the fraction of a full-strength descent step
actually delivered — over 400k coordinates, each client correct with probability
0.65 and silent with probability `q` (`--stage votes`):

| `N` | tie % (`q`=0) | `A` (`q`=0) | tie % (`q`=0.10) | `A` (`q`=0.10) |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.00 | 0.6573 | 8.38 | 0.6183 |
| 10 | 15.32 | 0.6558 | 9.23 | 0.6410 |
| **11** | **0.00** | **0.7029** | **7.22** | **0.6676** |
| 15 | 0.00 | 0.7747 | 5.54 | 0.7406 |

At `q = 0` — the operating regime under the randomized convention — the classical
parity pathology is visible: `N = 10` buys nothing over `N = 9`, because an even
vote's extra voter is never decisive, it only creates ties. So 11 both beats 10 on
alignment and cannot tie. The table is Monte Carlo, so differences below ~0.002 are
noise, which is itself the point of the `N = 9` vs `N = 10` comparison.

## Speed

`--eval_freq` defaults to 100 (2000 evaluations of 10k images used to dominate a
run), and `--loader gpu`, the default on CUDA, uploads the dataset once as `uint8`
and does the crop, flip and normalization as tensor ops on the device instead of
eleven `DataLoader`s doing PIL decode on the main thread. The augmentation is
torchvision's, and `test_gpu_crop_and_flip_match_torchvision_distributionally`
compares the per-row padding probability against `RandomCrop` itself over 2000
draws rather than checking shapes.
