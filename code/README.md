# Code

Source for *SignMuon, MuonSign, and the Role of Error Feedback*.

> [!IMPORTANT]
> Every command runs from this `code/` directory, which is the Python package root:
> `python3 -m federated.main …`, not `python3 federated/main.py`.

```bash
cd code
pip install -r requirements.txt
python3 -m tests.test_code          # CPU only, no downloads, ~1 min
```

## Where to go

| I want to… | Go to |
| :--- | :--- |
| **reproduce a specific table or figure** | [REPRODUCE.md](REPRODUCE.md) — one command each |
| understand the optimizers themselves | [`common/`](common/README.md) |
| run the CIFAR-10 ResNet-18 study | [`centralized/`](centralized/README.md) |
| run the federated study | [`federated/`](federated/README.md) |
| check a divergence theorem | [`counterexamples/`](counterexamples/README.md) |
| see the convex-benchmark modes | [`synthetic/`](synthetic/README.md) |
| run the language-modelling arm | [`nanogpt/`](nanogpt/README.md) |
| know what the tests actually pin | [`tests/`](tests/README.md) |
| redraw a figure | `counterexamples.run_counterexamples`, `synthetic.plot_synthetic`, `centralized.plot_analysis`, `federated.plot_federated` — see [REPRODUCE.md](REPRODUCE.md) |
| package this for double-blind review | [ANONYMIZATION.md](ANONYMIZATION.md) |

```
code/
├── REPRODUCE.md          exact command per paper table/figure
├── ANONYMIZATION.md      preparing the anonymous supplement
├── aggregate.py          multi-seed mean ± std
├── anonymize.py          the anonymity scan and bundler
├── common/               optimizers, per-layer LR rules, models, plotting style, utilities
├── centralized/          ResNet-18 / CIFAR-10        (main, train, tune, overnight, data)
├── federated/            all eleven methods, one driver (algorithms, main, tune, overnight, plot_federated, data)
├── synthetic/            F(X) = ½⟨X, AXB⟩, seven modes (benchmark, run_gpu, plot_synthetic)
├── counterexamples/      Theorems 1–4, exact LMO
├── nanogpt/              modded-nanogpt speedrun port
├── tests/test_code.py    the CPU test suite
└── results/              all output (created on first run)
```

Two entry points do the whole protocol in one resumable, budget-aware command:

```bash
python3 -m centralized.overnight --device cuda:0 --download
python3 -m federated.overnight   --device cuda:0 --budget-hours 12 --download
```

Both self-check, record the exact GPU / CUDA / Python / PyTorch, time your GPU,
print a schedule with a finish time per phase, and rewrite a `REPORT.md` after
every phase so you can read it mid-run. Ctrl-C stops cleanly and writes it.
`--dry-run` prints the schedule and exits. The centralized driver has no deadline
by default; the federated one takes `--budget-hours 0` for the same.

When a centralized run finishes, one more command packs everything the paper needs
into a single ~1 MB archive to download, leaving the ~1.5 GB of checkpoints behind:

```bash
python3 -m centralized.export_article                         # -> results/article_export.tar.gz
python3 -m centralized.plot_analysis --bundle article_export  # unpack it here, redraw the figures
```

## Method names

| Name | Matrix step | Uplink | Downlink |
| :--- | :--- | :---: | :---: |
| `signmuon` | `sign(polar(M))` | 1 bit | exact |
| `muonusign` | `polar(sign(M))` | 1 bit | exact |
| `muonsign` | `sign(polar(sign(M)))` | 1 bit | 1 bit |
| `ef21signmuon` | EF21 on `polar(M)` — **diverges**, Thm 4 | 1 bit | exact |
| `ef21muonusign` | `polar(g_est)`, `g_est ≈ M` | 1 bit | exact |
| `ef21muonsign` | as above + downlink EF21-P | 1 bit | 1 bit |
| `muon` | `polar(M)` averaged over clients (worker LMO) | 32 bit | exact |
| `muonserver` | `polar(mean M)` (server LMO) | 32 bit | exact |
| `signsgd`, `sgd`, `adam` | references | | |

Older spellings still resolve, but the two drivers accept different sets — the
centralized one never used the `signmuon_*` names:

| driver | accepted aliases |
| :--- | :--- |
| `centralized.main --optimizer` | `ef_usignmuon`, `ef_udsignmuon` |
| `federated.main --algorithm` | `signmuon_cl`, `signmuon_ef_21`, `signmuon_ef_ud`, `muon_server` |

**`MuonSign` changed meaning**: it used to name the sign-*before* method, which the
paper now calls `MuonUSign`. There is deliberately no alias for the old spelling,
because resolving it silently would swap the algorithm rather than the label.

The LMO is an **exact rank-truncated SVD** in `counterexamples/` (matching the
theorem statements) and a **Newton–Schulz approximation** everywhere a network is
trained (matching practice).

## Conventions that apply to every method

Enforced in one place, so that two runs differ *only* in the matrix-parameter rule.

* **Parameter routing.** The LMO/sign rule applies to matrix parameters (`ndim ≥ 2`,
  excluding the classification head). Biases, BatchNorm scales and the head go to
  AdamW, which is **never weight-decayed** in either setting. Federated, this holds
  for every method including the references; centrally, `--head-adamw auto`
  reproduces the published split and `--head-adamw always` makes it uniform.
* **Learning rate.** One cosine schedule for both the main and auxiliary rate.
  `--lr-scaling` sets a *derived* per-layer multiplier `η_layer = η₀·λ(family, shape)`
  in both settings, so only the shape-free `η₀` is ever tuned. The multiplier is a
  deterministic function of the layer shape, known to server and clients alike, so
  federating it costs nothing in communication and does not affect the 1-bit claim.
* **Weight decay** applied exactly once, **decoupled** (`X *= 1 − lr·wd`, uniform
  across layers, *not* scaled by the per-layer multiplier). Not a style preference:
  every step direction here is positively homogeneous of degree *zero*, so folding
  `wd·X` into the gradient cannot change the step length at all — it only rotates
  the direction, by an amount set by the drifting, method-dependent ratio
  `wd‖X‖_F/‖G‖_F`. `--weight-decay-mode coupled` reproduces that convention for the
  appendix ablation, in **both** drivers (it was centralized-only until 2026-07-28;
  the federated flag existed as a function argument no CLI ever set).
* **Momentum** is the EMA form `M = μM + (1−μ)G` of the algorithm boxes,
  trajectory-identical to the heavy-ball form of the main text. `sgd` keeps
  heavy-ball, since its step *is* the buffer.

### Per-layer learning rates

The two families produce step matrices with different norms, so one global rate
cannot be right for both families and across layers at once. The rule is *derived*,
from the criterion that an update's RMS gain `γ(A)=‖A‖_F/√fan_out` should be a fixed
fraction of the initialization's:

| family | methods | `‖s‖_F` | multiplier |
| :--- | :--- | :--- | :--- |
| `lmo` | Muon, MuonUSign, EF21-Muon{USign,Sign}, EF21-SignMuon | `√min(m,n)` | `√max(1, m/n)` |
| `sign` | SignMuon, MuonSign, SignSGD | `√(mn)` | `1/√fan_in` |

The first row is the aspect factor already in the reference Muon implementation —
the criterion *derives* it, which is the main evidence that it is the right
criterion. The second row is its missing counterpart. `python3 -m common.lr_scaling`
lists the alternatives (`mup`, `mishra-analysis`, `power:α,β`) and what each assumes.

On CNN2 the sign family's multiplier spans **7.8×** (`fan_in` 75 → 4608), a wider
spread than ResNet-18 offers, so the federated setting is the more sensitive test of
the rule. `python3 -m federated.tune --stage anchors` prints the per-layer
multipliers for both families, their spread, the grid anchor each method inherits,
and the per-layer step-length ratio a single global rate would have to absorb
(8.7× on `conv1`, 67.9× on `fc1`).

## Three caveats worth knowing

* **Exact zeros are randomized, so every sign channel is a strict 1 bit.**
  `sign(0) = 0` would make the alphabet ternary, and that is not a corner case:
  `polar(M)` has an exactly-zero column wherever `M` does, which after ReLU and
  MaxPool is common (a feature zero across the whole local batch gives an
  exactly-zero gradient column) — **8–17% of raw sign entries on CNN2**. The
  paper's convention maps each zero to an independent random `±1`
  (`common.optimizers.sign_pm1`), on every sign channel. Runs still record
  `uplink_zero_frac` (the raw rate, before mapping); `--uplink-zeros keep`
  restores the ternary channel for diagnostics.
  See [`federated/README.md`](federated/README.md) for why the client count is 11.
* **BatchNorm in the federated setting.** Local models are discarded each round and
  BatchNorm runs in inference mode during gradient accumulation, so the running
  statistics are never updated from data — they stay at `(0, 1)` for the whole run.
  BatchNorm is therefore a fixed normalization with learnable affine parameters.
  Self-consistent, and what the reported numbers used; `--live-bn-stats` changes it.
* **`bfloat16` LMO.** The default matches the reference Muon implementation, but
  bfloat16 carries ~3 decimal digits, so for methods that sign the LMO *output* an
  entry of `polar(M)` near zero can flip. `--lmo-dtype float32` when the sign
  pattern is the object of study.

## Results and multi-seed runs

A run writes to `results/{centralized,federated}/<run_name>/seed<seed>/`:

* `metrics.json` — `{"config": {...}, "history": {"steps": [...], "test_acc": [...], ...}}`
* `model.pt` — final weights plus the config

> **Put this somewhere roomy before a long sweep.** `SIGNMUON_RESULTS` relocates
> the whole tree; `aggregate.py` and every plotting script follow it
> automatically. One `model.pt` per job adds up — **2.9 MB** for CNN2, **42.7 MB**
> for ResNet-18 — so the 95-job federated night is ~280 MB and a 36-job
> centralized sweep is ~1.5 GB, before the dataset. Running the system disk dry at
> 04:00 costs the night.
>
> ```bash
> export SIGNMUON_RESULTS=/mnt/scratch/signmuon-results   # or D:\...\results
> python3 -m federated.overnight --device cuda:0 --data /mnt/scratch/data_federated
> ```
>
> `--data` is separate and also worth pointing off the system disk: CIFAR-10 is
> ~180 MB per copy and the two settings use different directories by default.

`history` records the **x-axis explicitly** and stores only evaluated points, so
curves from different seeds (or different `--eval_freq`) align pointwise. Nothing is
deleted or forward-filled.

```bash
for s in 0 1 2 3 4; do python3 -m federated.main --seed $s ... ; done
python3 -m aggregate --metric test_acc --csv summary.csv
```

`aggregate.py` groups runs by configuration *minus* the seed (and minus
device/data-path fields) and reports mean ± sample std, flagging unequal seed counts
and single-seed groups instead of printing a misleading `± 0`.

## Tests

```bash
python3 -m tests.test_code                    # torch, CPU only, ~1 min
python3 -m counterexamples.problems           # the Theorem 1-4 constants
python3 -m counterexamples.verify_ns_oracle   # exact vs Newton-Schulz LMO
python3 -m common.lr_scaling --measure        # the sign-step operator norm
python3 -m anonymize --check                  # no identifying material
```

The central check is that the federated driver with one client reproduces the
corresponding centralized optimizer exactly, for all eight matrix rules and under
both scaling conventions — which is what keeps `federated/algorithms.py` and
`common/optimizers.py` from drifting apart. See [`tests/`](tests/README.md).
