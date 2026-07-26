# Overnight run report

- started `2026-07-26T02:18:50`, wall clock 9:07:48 elapsed, no deadline (Ctrl-C to stop)
- device `cuda:0`, resnet18 / cifar10, measured **13.4 s/epoch**
- scaling rule `unit-gain`, `--head-adamw always`, lr_aux = 0.001, momentum 0.9, weight decay 0.0005
- tuning: 15 epochs on the 45k/5k split, selected on **val_acc** (tail mean of 5); the test set was not used for any decision

## Gain diagnostic (measures the exponent directly)

Growth of the accumulated update's RMS gain `||X_t - X_0||_F / sqrt(fan_out)` against the epoch count, at a **constant** learning rate (with annealing the accumulation saturates and the slope would measure the schedule instead).

A slope near **0.5** means successive sign steps stay incoherent, supporting `unit-gain` (alpha=1/2); near **1.0** means they align, supporting `mup` (alpha=1).


| run                   | fitted slope | R^2   | epochs used | reading                 |
| --------------------- | ------------ | ----- | ----------- | ----------------------- |
| `gain_muon_e20_t`     | 0.278        | 0.946 | 20          | incoherent -> alpha=1/2 |
| `gain_signmuon_e20_t` | 0.123        | 0.867 | 20          | incoherent -> alpha=1/2 |




## Tuned eta_0 (validation)


| method          | eta_0       | val acc | configs | note                                                                     |
| --------------- | ----------- | ------- | ------- | ------------------------------------------------------------------------ |
| `signmuon`      | 0.0034      | 92.14%  | 4       | !! winner is at the LOW end of the lr grid -- extend downward and re-run |
| `muonusign`     | 0.00696238  | 92.13%  | 4       |                                                                          |
| `muonsign`      | 0.0034      | 91.05%  | 4       | !! winner is at the LOW end of the lr grid -- extend downward and re-run |
| `ef21signmuon`  | 0.00371327  | 92.56%  | 4       |                                                                          |
| `ef21muonusign` | 0.00371327  | 92.08%  | 4       |                                                                          |
| `ef21muonsign`  | 0.00324911  | 91.79%  | 4       |                                                                          |
| `muon`          | 0.00696238  | 92.41%  | 4       |                                                                          |
| `signsgd`       | 0.0034      | 90.16%  | 4       | !! winner is at the LOW end of the lr grid -- extend downward and re-run |
| `sgd`           | 0.0323165   | 89.67%  | 4       |                                                                          |
| `adam`          | 0.000464159 | 89.59%  | 4       |                                                                          |
| `adam+scaled`   | 0.0157814   | 89.98%  | 4       |                                                                          |


A `BOUNDARY` note means the optimum sat on a grid endpoint: extend the grid and re-run that method before reporting it.

## Horizon stability

Top-3 rates re-run at 75 epochs (still on the tuning split, so `val_acc` is comparable):


| method     | best @ tune | best @ final | verdict |
| ---------- | ----------- | ------------ | ------- |
| `signmuon` | 0.0034      | 0.0034       | stable  |
| `muon`     | 0.006962    | 0.006962     | stable  |


If a winner moved, the short-horizon table is provisional for that family and the tuning should be redone at a longer horizon.

## Final runs (full 50k, test set)


| run                               | test acc (tail mean) | epochs to target |
| --------------------------------- | -------------------- | ---------------- |
| `ef21muonsign_unit-gain_e75_fs0`  | 93.76%               | 15               |
| `ef21muonusign_unit-gain_e75_fs0` | 93.72%               | 12               |
| `ef21signmuon_unit-gain_e75_fs0`  | 93.50%               | 16               |
| `muon_unit-gain_e75_fs0`          | 93.83%               | 16               |
| `muonsign_unit-gain_e75_fs0`      | 93.90%               | 20               |
| `muonusign_unit-gain_e75_fs0`     | 93.68%               | 15               |
| `sgd_unit-gain_e75_fs0`           | 94.06%               | 21               |
| `signmuon_unit-gain_e75_fs0`      | 93.92%               | 10               |
| `signsgd_unit-gain_e75_fs0`       | 93.80%               | 24               |


Aggregate across seeds with `python3 -m aggregate --root results/centralized`.

## What this run did NOT establish

- `lr_aux` **was fixed at 0.001**, not tuned, and not verified to be method-independent. The auxiliary group is AdamW on the same parameters for every method, so its optimum should not depend on the matrix rule -- but that is an argument, not a measurement. Check it with `python3 -m centralized.tune --stage aux`.
- **Momentum (0.9) and weight decay (0.0005) were held fixed** for every method, so this is a comparison at equal momentum, not at each method's own optimum.
- **Weight decay is coupled** by default (added to the gradient, so it passes through the LMO). The federated driver uses *decoupled* decay; `--weight-decay-mode decoupled` makes the two consistent, at the cost of changing these numbers.
- **A gap smaller than the seed spread is not a result.** Add seeds with `--resume` and aggregate before claiming one.



## Next steps

76 jobs completed, 0 failed. Resume with:

```bash
python3 -m centralized.overnight --device cuda:0 --resume
```

