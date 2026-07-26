# Code

Source for the paper *SignMuon, MuonSign, and the Role of Error Feedback*.

**→ [REPRODUCE.md](REPRODUCE.md) has the exact command for every table and figure
in the paper.** Start there.

> [!IMPORTANT]
> Every command runs from this `code/` directory, which is the Python package root:
> `python3 -m federated.main ...`, not `python3 federated/main.py`.

## Layout

```
code/
├── REPRODUCE.md          exact commands per paper table/figure
├── aggregate.py          collapses multi-seed runs into mean ± std
├── common/               shared library
│   ├── optimizers.py       the eight methods as torch.optim.Optimizer subclasses
│   ├── lr_scaling.py       derived per-layer learning-rate rules
│   ├── models.py           CNN2, ResNet9, ResNet18
│   └── utils.py            seeding, run dirs, metrics schema, parameter routing
├── centralized/          single-node CIFAR-10 / MNIST
│   ├── main.py             entry point
│   ├── train.py            training loop, optimizer construction
│   ├── tune.py             equal-budget, validation-only LR search
│   └── data.py             loaders, incl. the 45k/5k tuning split
├── federated/            all ten federated methods, one driver
│   ├── main.py             entry point
│   ├── algorithms.py       run_federated + the MethodSpec registry
│   ├── data.py             partitioning and per-client loaders
│   └── grid.py             early-stopped learning-rate search
├── synthetic/
│   └── benchmark.py        F(X) = ½⟨X, AXB⟩, grid + final modes
├── counterexamples/      exact-LMO reproduction of Theorems 1–4 and the figures
├── nanogpt/              modded-nanogpt speedrun adaptation (own README)
├── tests/test_code.py    CPU-only test suite
├── notebooks/            plotting only
└── results/              all output (created on first run)
```

## Method names

The code uses the paper's names throughout:

| Name | Matrix-parameter step | Uplink | Downlink |
| :--- | :--- | :--- | :--- |
| `signmuon` | `sign(polar(M))` | 1 bit | exact |
| `muonusign` | `polar(sign(M))` | 1 bit | exact |
| `muonsign` | `sign(polar(sign(M)))` | 1 bit | 1 bit |
| `ef21signmuon` | EF21 on `polar(M)` — **diverges**, Theorem 4 | 1 bit | exact |
| `ef21muonusign` | `polar(g_est)`, `g_est ≈ M` | 1 bit | exact |
| `ef21muonsign` | as above + downlink EF21-P | 1 bit | 1 bit |
| `muon`, `signsgd`, `sgd`, `adam` | references | | |

Older spellings (`signmuon_cl`, `signmuon_ef_21`, `signmuon_ef_ud`, `ef_usignmuon`,
`ef_udsignmuon`) still resolve. **`MuonSign` changed meaning**: it used to name the
sign-*before* method, which the paper now calls `MuonUSign`. There is deliberately
no alias for the old spelling, because resolving it silently would swap the
algorithm rather than just the label.

The LMO is an **exact rank-truncated SVD** in `counterexamples/` (matching the
theorem statements) and a **Newton–Schulz approximation** everywhere a network is
trained (matching practice).

## Conventions that apply to every method

Enforced in one place, so that two runs differ *only* in the matrix-parameter rule.

* **Parameter routing.** The LMO/sign rule applies to matrix parameters
  (`ndim ≥ 2`, excluding the classification head). Biases, BatchNorm scales and the
  head go to AdamW. In the federated driver this holds for every method, references
  included. Centralized, `--head-adamw auto` gives the LMO methods this split while
  `sgd`/`adam`/`signsgd` apply their single rule to all parameters (as published,
  and what the paper's numbers used); `--head-adamw always` makes it uniform.
* **Learning rate.** One cosine schedule for both the main and auxiliary rate. In
  the centralized setting `--lr-scaling` additionally sets a *derived* per-layer
  multiplier, `η_layer = η₀·λ(family, shape)`, so that only the shape-free `η₀` is
  ever tuned — see below.
* **Weight decay** applied exactly once, **decoupled** in both drivers
  (`X *= 1 − lr·wd`, uniform across layers — *not* scaled by the per-layer
  multiplier), so the LMO sees the true gradient geometry. This is not a style
  preference: every step direction here is positively homogeneous of degree *zero*
  (`sign(cM) = sign(M)`, `polar(cM) = polar(M)`), so folding `wd·X` into the
  gradient cannot change the step length at all — it only rotates the direction, by
  an amount set by the drifting, method-dependent ratio `wd·‖X‖_F/‖G‖_F`.
  `--weight-decay-mode coupled` reproduces that convention (which is what
  Mishra et al. and our own pre-2026-07-26 numbers used) for the appendix ablation.
* **Momentum** is the EMA form `M = μM + (1−μ)G` of the paper's algorithm boxes,
  trajectory-identical to the heavy-ball form of the main text (the two differ by a
  constant factor and every method is positively homogeneous in `M`). `sgd` keeps
  heavy-ball, since its step *is* the buffer.

### Per-layer learning rates

The six methods produce step matrices from two families with different norms, so one
global rate cannot be right for both families and across layers at once. The rule is
*derived*, from the criterion that an update's RMS gain `γ(A)=‖A‖_F/√fan_out` should
be a fixed fraction of the initialization's:

| family | methods | `‖s‖_F` | multiplier |
| :--- | :--- | :--- | :--- |
| `lmo` | Muon, MuonUSign, EF21-Muon{USign,Sign}, EF21-SignMuon | `√min(m,n)` | `√max(1, m/n)` |
| `sign` | SignMuon, MuonSign, SignSGD | `√(mn)` | `1/√fan_in` |

The first row is the aspect factor already in the reference Muon implementation — the
criterion *derives* it, which is the main evidence that it is the right criterion. The
second row is its missing counterpart. `python3 -m common.lr_scaling` lists the
alternatives (`mup`, `mishra-analysis`, `power:α,β`) and what each assumes;
`--measure` and `--compare` are the supporting diagnostics.

### Two caveats worth knowing

* **BatchNorm in the federated setting.** Local models are discarded each round and
  BatchNorm runs in inference mode during gradient accumulation, so the running
  statistics are never updated from data — they stay at `(0, 1)` for the whole run.
  BatchNorm therefore acts as a fixed normalization with learnable affine
  parameters. Self-consistent, and what the reported numbers used;
  `--live-bn-stats` changes it.
* **`bfloat16` LMO.** The default matches the reference Muon implementation, but
  bfloat16 carries ~3 decimal digits, so for methods that sign the LMO *output* an
  entry of `polar(M)` near zero can flip. `--lmo-dtype float32` when the sign
  pattern is the object of study.

## Results and multi-seed runs

A run writes to `results/{centralized,federated}/<run_name>/seed<seed>/`:

* `metrics.json` — `{"config": {...}, "history": {"steps": [...], "test_acc": [...], ...}}`
* `model.pt` — final weights plus the config

`history` records the **x-axis explicitly** and stores only evaluated points, so
curves from different seeds (or different `--eval_freq`) align pointwise. Nothing is
deleted or forward-filled. Synthetic results go to
`results/synthetic/<method>/{grid,final}.json`.

```bash
for s in 0 1 2 3 4; do python3 -m federated.main --seed $s ... ; done
python3 -m aggregate --metric test_acc --csv summary.csv
```

`aggregate.py` groups runs by configuration *minus* the seed (and minus
device/data-path fields) and reports mean ± sample std.

## Tests

```bash
python3 -m tests.test_code                    # torch, CPU only, seconds
python3 -m counterexamples.problems           # Theorem 1-4 constants
python3 -m counterexamples.verify_ns_oracle   # exact vs Newton-Schulz LMO
python3 -m common.lr_scaling --measure        # the sign-step operator norm
```

`tests/test_code.py`'s central check is that the federated driver with one client
reproduces the corresponding centralized optimizer exactly, for all eight
matrix-parameter rules — which is what keeps `federated/algorithms.py` and
`common/optimizers.py` from drifting apart.
