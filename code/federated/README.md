# `federated/` — all eleven methods, one driver

CNN2 on CIFAR-10, **11 clients**, 3 local steps, 2000 rounds, homogeneous split.

| File | What it does |
| :--- | :--- |
| [`algorithms.py`](algorithms.py) | `run_federated` + the `MethodSpec` registry — *the* federated implementation |
| [`main.py`](main.py) | Entry point: one run, one config, one `metrics.json` |
| [`data.py`](data.py) | Partitioning, the 45k/5k split, device-resident shards |
| [`tune.py`](tune.py) | Equal-budget, validation-only learning-rate search |
| [`overnight.py`](overnight.py) | The whole protocol as one resumable command |
| [`grid.py`](grid.py) | Retired — it selected on the test set; points at `tune.py` |

## Start here

```bash
cd code
python3 -m federated.overnight --device cuda:0 --budget-hours 12 --download
```

The federated twin of `centralized/overnight.py`, with one addition: because a
federated round costs wildly different amounts on different hardware, the plan is
**fitted to the budget** from a measured round time rather than being cut halfway.
Seeds, then the weight-decay ablation, then the tuning horizon give way in that
order, and the reverse when there is slack. `--budget-hours 0` disables the fitting.

`--dry-run` prints the schedule and exits, which is the cheap way to find out what
your GPU can afford.

## One driver, eleven methods

Every method is one row of the paper's master table — where the LMO is evaluated,
and what compresses each channel. Adding a method is one dict entry, which is why
the eleven cannot drift apart in schedule, routing, evaluation or weight decay.

| Method | LMO | Uplink | Downlink | Family | Uncompressed control |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `signmuon` | worker | sign / MV | exact | `sign` | `muon` |
| `ef21signmuon` | worker | EF21 | exact | `lmo` | `muon` |
| `muonusign` | server | sign / MV | exact | `lmo` | `muonserver` |
| `muonsign` | server | sign / MV | sign | `sign` | `muonserver` |
| `ef21muonusign` | server | EF21 | exact | `lmo` | `muonserver` |
| `ef21muonsign` | server | EF21 | EF21-P | `lmo` | `muonserver` |
| `muon` | worker | exact (average) | exact | `lmo` | — |
| `muonserver` | server | exact (average) | exact | `lmo` | — |
| `signsgd` | none | sign / MV | exact | `sign` | `sgd` |
| `sgd`, `adam` | none | exact (average) | exact | — | — |

### Two full-precision Muons, and why one is not enough

`muon` orthogonalizes on the **worker** and the server averages the resulting polar
factors. `muonserver` sends the momentum uncompressed and applies **one** LMO on
the server. They are different algorithms, and each is the uncompressed control for
a different template — `muonserver` is literally what MuonUSign, EF21-MuonUSign and
EF21-MuonSign become when the compressor is the identity.

They are not interchangeable, because averaging near-orthogonal matrices *shortens*
the step. Measured on a 120×4608 matrix, clients sharing a signal plus independent
noise of relative size `q`, as `‖·‖_F/√min(m,n)`:

| `q` | N=1 | N=3 | N=11 | N=21 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.767 | 0.767 | 0.767 | 0.767 |
| 0.3 | 0.768 | 0.744 | 0.735 | 0.734 |
| 1.0 | 0.769 | 0.622 | **0.559** | 0.548 |
| 3.0 | 0.768 | 0.483 | 0.324 | 0.286 |

`polar(mean)` by contrast is flat to 0.3% from N=1 to N=21 at every noise level.
So `muon`'s effective step size depends on the client count *and* on the
gradient-to-noise ratio, which drifts over training — a tuned `η₀` absorbs a
constant handicap, not that one. Comparing the server-LMO family against `muon`
would confound "what does the 1-bit uplink cost?" with "what does averaging
orthogonal matrices cost?", and it would flatter the compressed methods.

`sgd` and `adam` have no per-layer family on purpose: their step norm is
data-dependent, so no static multiplier implements the unit-gain criterion, and
applying one anyway would be an arbitrary rescaling dressed up as a
parameterization. `--scale-baselines` gives Adam the sign rule for the ablation.

**The load-bearing test** is `test_federated_one_client_equals_centralized`: this
driver with `N = 1` must reproduce `common/optimizers.py` exactly, for all eight
matrix rules and under both the `legacy` and `unit-gain` conventions. That is what
stops the two implementations diverging.

## Three things the numbers depend on

### The uplink alphabet is ternary, so "1 bit" is an approximation

`sign(0) = 0`, so a client transmits from `{−1, 0, +1}`. Not a corner case:
`polar(M)` has an exactly-zero column wherever `M` does, and `M` does wherever a
feature was zero across the whole local batch — after ReLU and MaxPool, routinely.
Measured on CNN2: **8–17% of transmitted entries are zero, every round**, i.e.
≈1.37 bits per parameter rather than 1.

Two knock-ons, both now recorded rather than assumed. `uplink_zero_frac` measures
the bit cost; `mv_tie_frac` measures how often the majority vote comes out zero —
which, because a zero vote is not `±1`, can happen at **any** client count, not only
even ones. `--uplink-zeros random` makes the channel a genuine one bit;
`--mv-ties random` makes the server step a genuine sign matrix. Neither changes
expected descent, because a tie carries no information about the true sign, so both
default to the published behaviour.

The EF21 uplink is deliberately left ternary: `α·sign(Δ)` is zero exactly where the
estimator is already on target, and forcing it off by `α` is the mechanism the
EF21-SignMuon divergence theorem exploits.

### What the compression actually saves

Every run prints the accounting rather than quoting a headline number
(`communication_bits` in `algorithms.py`). On CNN2 at the measured 10% zero rate:

| method | uplink | downlink | **round trip** |
| :--- | ---: | ---: | ---: |
| `signmuon`, `muonusign`, `ef21signmuon`, `ef21muonusign` | 22× | 1× | **1.9×** |
| `muonsign`, `ef21muonsign` | 22× | 29× | **25×** |
| `muon`, `muonserver`, `sgd`, `adam` | 1× | 1× | 1× |

Three corrections to "32×" are folded in: the ternary alphabet (1.37 bits, not 1),
the auxiliary group riding along uncompressed in both directions (which alone puts
a perfect 1-bit uplink at 1.087 bits/parameter model-wide), and — the big one — the
fact that four of the six methods **broadcast a full-precision model every round**,
so their round-trip saving cannot exceed 2× however good the uplink is.

That is not an argument against the paper's methods. It is the argument *for* the
bidirectional ones, and it is what makes "we measure what the downlink guarantee
costs" the interesting sentence. It is only an argument against quoting 32× for a
uplink-only method.

### Is the per-layer rule doing its job? (`gain_spread`)

The unit-gain rule exists to make the realized gain `γ = ‖λ·s‖_F/√fan_out` the
**same on every layer**, so `η₀` means one thing everywhere. That is checkable, and
every run checks it: the per-layer profile is printed at round 1 and `gain_spread`
is recorded at every evaluation. A flat profile is 1.00×.

It is not flat for the LMO family. The derivation assumes the oracle returns
`‖polar‖_F = √min(m,n)` exactly; Newton–Schulz does not, and its error is
**shape-dependent**. Measured on `muonserver`, realized gain per layer:

| dtype | `ns_steps` | gain spread | conv1 | conv2 | fc1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | **5** | **1.34×** | 0.875 | 0.943 | 0.705 |
| bfloat16 | 7 | 1.11× | 0.888 | 0.940 | 0.846 |
| bfloat16 | 12 | 1.15× | 0.951 | 0.893 | 0.829 |
| float32 | 5 | 1.35× | 0.873 | 0.945 | 0.700 |
| float32 | 12 | 1.28× | 0.947 | 0.893 | 0.742 |

The exact value is data-dependent — live runs land in 1.2–1.35× at `ns_steps=5` —
but the shape of the profile is not: the offender is always `fc1` (120×4608,
aspect 38:1), which holds 72% of CNN2's matrix parameters and realizes ~0.70–0.78
of the gain the rule assumes. A
tuned `η₀` is one constant and cannot absorb a per-layer factor. The sign family is
flat by construction — its steps are exactly ±1 up to the zero entries — which is
why it sits near 1.08×.

**The default stays `--ns_steps 5`**, because that is what reference Muon uses and
what the published numbers used, and because the residual 1.34× is under half a
learning-rate grid step (the lattice is 2–2.5× per step) so it cannot flip a
ranking. `--ns_steps 7` roughly halves the excess in bfloat16 if you want it
tighter. It never reaches 1.0: the quintic oscillates in a band around 1 rather
than converging, which is the same fact `tests/` pins for the norm.

The driver warns above 1.15×. Note that the log contains **two** lines with the
word "spread" — `describe_rule`'s is the spread of the *multipliers* (1.0× for the
LMO family on CNN2, by construction); the one that matters is labelled
`gain spread`.

### Why 11 clients

Alignment `A = E[⟨truth, ŝ⟩]/(mn)` — the fraction of a full-strength descent step
actually delivered — over 400k coordinates, each client correct with probability
0.65 and silent with probability `q`:

| `N` | tie % (`q`=0) | `A` (`q`=0) | tie % (`q`=0.10) | `A` (`q`=0.10) |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.00 | 0.6573 | 8.38 | 0.6183 |
| 10 | 15.32 | 0.6558 | 9.23 | 0.6410 |
| **11** | **0.00** | **0.7029** | **7.22** | **0.6676** |
| 15 | 0.00 | 0.7747 | 5.54 | 0.7406 |

At `q = 0` the classical parity pathology is visible: `N = 10` buys nothing over
`N = 9`, because an even vote's extra voter is never decisive — it only creates
ties. At CNN2's real zero rate that washes out and alignment is simply monotone in
`N`. Either way 11 beats 10, which is why it is the default; the reason is "more
voters", not parity.

### BatchNorm statistics never update

Local models are discarded each round and BatchNorm runs in inference mode during
accumulation, so the running statistics stay at their initialization `(0, 1)` for
the entire run, in training and evaluation alike. BatchNorm is therefore a fixed
normalization with learnable affine parameters. This is self-consistent — no
train/test mismatch — and it is what the reported federated numbers used.
`--live-bn-stats` changes it, and changes the numbers.

## Per-layer learning rates matter more here than centrally

CNN2 has three matrix parameters, with `fan_in` 75, 1600 and 4608, so the sign
family's multiplier spans **7.8×**. At one global rate, a SignMuon step is 8.7× the
corresponding Muon step on `conv1` and **67.9×** on `fc1` (`√(mn)/√min(m,n)`, both
printed by `--stage anchors`).

```bash
python3 -m federated.tune --stage anchors    # per-layer multipliers, spreads, step ratios, grid anchors
python3 -m federated.tune --stage votes      # the majority-vote alignment table below
```

Grid anchors are *transported* rather than re-guessed: the published rates were
tuned under `legacy`, and

```
anchor(rule) = anchor_legacy · geomean(λ_legacy) / geomean(λ_rule)
```

evaluated on the model's own shapes. On CNN2 that leaves the LMO family untouched
and multiplies the sign family by 28.65.

## Speed

Two things used to dominate a run: `--eval_freq 1` (2000 evaluations of 10k images)
and ten `DataLoader`s with `num_workers=0` doing PIL decode on the main thread.
`--eval_freq` now defaults to 100, and `--loader gpu` (the default on CUDA) uploads
the dataset once as `uint8` and does the crop, flip and normalization as tensor ops
on the device. The augmentation is torchvision's, and
`test_gpu_crop_and_flip_match_torchvision_distributionally` compares the per-row
padding probability against `RandomCrop` itself over 2000 draws rather than checking
shapes.

See [`../REPRODUCE.md`](../REPRODUCE.md) §5 for the exact commands, including §5c for
reproducing the published table under the old convention.
