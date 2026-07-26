# Overnight run report

* started `2026-07-26T11:39:52`, wall clock 11:07:42 elapsed, no deadline (Ctrl-C to stop)
* device `cuda:0`, resnet18 / cifar10, measured **12.9 s/epoch**
* scaling rule **`mup`**, `--head-adamw always`, lr_aux = 0.001, momentum 0.9, weight decay 0 (decoupled)
* tuning: 15 epochs on the 45k/5k split, selected on **val_acc** (tail mean of 5); the test set was not used for any decision

## Gain diagnostic (measures the exponent directly)

Growth of the accumulated update's RMS gain `||X_t - X_0||_F / sqrt(fan_out)` against the epoch count, at a **constant** learning rate (with annealing the accumulation saturates and the slope would measure the schedule instead).

A slope near **0.5** means successive sign steps stay incoherent, supporting `unit-gain` (alpha=1/2); near **1.0** means they align, supporting `mup` (alpha=1).

| run | fitted slope | R^2 | epochs used | reading |
| :--- | ---: | ---: | ---: | :--- |
| `gain_muon_e20_t` | 0.513 | 1.000 | 20 | incoherent -> alpha=1/2 |
| `gain_signmuon_e20_t` | 0.516 | 1.000 | 20 | incoherent -> alpha=1/2 |

## Alpha sweep

Best exponent on `signmuon`: **alpha = 1**.

* alpha=0: val 93.14% at eta_0=0.003162
* alpha=0.5: val 93.22% at eta_0=0.3394
* alpha=1: val 93.42% at eta_0=1.152

## Tuned eta_0 (validation)

| method | eta_0 | val acc | configs | note |
| :--- | ---: | ---: | ---: | :--- |
| `signmuon` | 0.534711 | 93.45% | 4 |  |
| `muonusign` | 0.15 | 92.70% | 6 |  |
| `muonsign` | 0.1152 | 90.95% | 6 |  |
| `ef21signmuon` | 0.0172355 | 93.17% | 4 |  |
| `ef21muonusign` | 0.0172355 | 92.60% | 4 |  |
| `ef21muonsign` | 0.015081 | 92.79% | 4 |  |
| `muon` | 0.0323165 | 93.50% | 4 |  |
| `signsgd` | 0.1152 | 89.16% | 6 |  |
| `sgd` | 0.0323165 | 88.93% | 4 |  |
| `adam` | 0.000464159 | 89.90% | 4 |  |
| `adam+scaled` | 0.534711 | 89.40% | 4 |  |

A `BOUNDARY` note means the optimum sat on a grid endpoint: extend the grid and re-run that method before reporting it.

## Horizon stability

Top-3 rates re-run at 75 epochs (still on the tuning split, so `val_acc` is comparable):

| method | best @ tune | best @ final | verdict |
| :--- | ---: | ---: | :--- |
| `signmuon` | 0.5347 | 2.482 | **MOVED** |
| `muon` | 0.03232 | 0.03232 | stable |

If a winner moved, the short-horizon table is provisional for that family and the tuning should be redone at a longer horizon.

## Final runs (full 50k, test set)

| run | test acc (tail mean) | epochs to target |
| :--- | ---: | ---: |
| `adam_mup_e75_fs0` | 93.38% | 20 |
| `adam_scaled_mup_e75_fs0` | 93.11% | 23 |
| `ef21muonsign_mup_e75_fs0` | 94.26% | 10 |
| `ef21muonusign_mup_e75_fs0` | 94.05% | 10 |
| `ef21signmuon_mup_e75_fs0` | 94.54% | 10 |
| `muon_mup_e75_fs0` | 94.46% | 10 |
| `muonsign_mup_e75_fs0` | 93.34% | 16 |
| `muonusign_mup_e75_fs0` | 93.94% | 10 |
| `sgd_mup_e75_fs0` | 93.11% | 22 |
| `signmuon_mup_e75_fs0` | 94.27% | 7 |
| `signsgd_mup_e75_fs0` | 92.93% | 23 |

Aggregate across seeds with `python3 -m aggregate --root results/centralized`.

## Weight-decay ablation (decoupled, wd = 0.0005)

The primary table above is unregularized (`--weight-decay 0`), which is the setting the theorems analyse and the one both reference implementations use. These runs repeat the best methods at seed 0 with decoupled decay on the matrix parameters, at the *same* eta_0. What matters is whether the ordering moves, not the absolute gain -- eta_0 was not re-tuned under decay.

| method | eta_0 | with decay | no decay | delta |
| :--- | ---: | ---: | ---: | ---: |
| `ef21signmuon` | 0.0172355 | 94.62% | 94.54% | +0.08 |
| `muon` | 0.0323165 | 94.40% | 94.46% | -0.06 |
| `signmuon` | 0.534711 | 94.86% | 94.27% | +0.59 |

## What this run did NOT establish

* **`lr_aux` was fixed at 0.001**, not tuned, and not verified to be method-independent. The auxiliary group is AdamW on the same parameters for every method, so its optimum should not depend on the matrix rule -- but that is an argument, not a measurement. Check it with `python3 -m centralized.tune --stage aux`.
* **Momentum (0.9) and weight decay (0) were held fixed** for every method, so this is a comparison at equal momentum, not at each method's own optimum.
* Weight decay is **decoupled** (`X *= 1 - lr*wd`, the LMO sees the true gradient). The coupled convention -- `wd*X` added to the gradient, which is what Mishra et al.'s Algorithm 1 and our own earlier numbers used -- is *not* an alternative worth reporting as an equal: every step direction here is scale-invariant, so coupled decay shrinks nothing and only rotates the direction. `--weight-decay-mode coupled` reproduces it for the appendix ablation.
* **A gap smaller than the seed spread is not a result.** Add seeds with `--resume` and aggregate before claiming one.

## Next steps

87 jobs completed, 0 failed. Resume with:

```bash
python3 -m centralized.overnight --device cuda:0 --resume
```
