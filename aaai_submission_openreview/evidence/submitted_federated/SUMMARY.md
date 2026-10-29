# Federated results

175 runs: 60 final, 92 lr, 2 preflight, 18 rules, 3 wd.

## The reported table (`tab:exp_3`)

Mean +/- sample std over seeds of the per-seed tail mean. A blank std means
one seed, which measures no dispersion -- do not read it as agreement.

| method | eta_0 | seeds | acc (%) | rounds to 80% | up (bits) | down (bits) | round trip |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| muon | 0.1 | 5 | 85.98 +/- 0.26 | 260 | 32.00 | 32.00 | 1.0x |
| muonserver | 0.05 | 5 | 85.74 +/- 0.11 | 400 | 32.00 | 32.00 | 1.0x |
| signmuon | 0.1 | 5 | 85.72 +/- 0.24 | 540 | 1.09 | 1.09 | 29.4x |
| ef21signmuon | 0.02 | 5 | 84.71 +/- 0.15 | 640 | 1.09 | 32.00 | 1.9x |
| muonusign | 0.05 | 5 | 84.56 +/- 0.35 | 500 | 1.09 | 32.00 | 1.9x |
| ef21muonsign | 0.05 | 5 | 83.99 +/- 0.28 | 980 | 1.09 | 1.09 | 29.4x |
| ef21muonusign | 0.01 | 5 | 83.56 +/- 0.41 | 900 | 1.09 | 32.00 | 1.9x |
| signsgd | 0.01 | 5 | 81.44 +/- 0.15 | 1140 | 1.09 | 1.09 | 29.4x |
| muonsign | 0.02 | 5 | 82.94 +/- 0.19 | 860 | 1.09 | 1.09 | 29.4x |
| sgd | 0.05 | 5 | 81.69 +/- 0.37 | 1240 | 32.00 | 32.00 | 1.0x |
| adam | 0.001 | 5 | 77.12 +/- 1.04 | -- (0/5) | 32.00 | 32.00 | 1.0x |
| adam+scaled | 0.05 | 5 | 78.23 +/- 1.13 | -- (0/5) | 32.00 | 32.00 | 1.0x |

`round trip` is the reduction against full precision with the uncompressed
auxiliary group counted in, computed from each run's own sign alphabet. It is
the number to quote, not the uplink-only one.

## Per-layer rule ablation

Each (method, rule) pair is tuned from scratch under that rule, so a
rule is not penalized for prescribing a different `eta_0` -- it should.
What would signify is the *ordering* of the methods changing.

| method | rule | eta_0 | seeds | acc (%) |
| :--- | :--- | ---: | ---: | ---: |
| signmuon | **unit-gain** | 0.1 | 5 | 85.72 +/- 0.24 |
| signmuon | mup | 2.0 | 3 | 85.52 +/- 0.01 |
| signmuon | none | 0.002 | 3 | 85.68 +/- 0.17 |
| signsgd | **unit-gain** | 0.01 | 5 | 81.44 +/- 0.15 |
| signsgd | mup | 0.5 | 3 | 81.47 +/- 0.25 |
| signsgd | none | 0.0005 | 3 | 81.37 +/- 0.21 |
| muonsign | **unit-gain** | 0.02 | 5 | 82.94 +/- 0.19 |
| muonsign | mup | 0.5 | 3 | 82.37 +/- 0.27 |
| muonsign | none | 0.001 | 3 | 82.26 +/- 0.40 |

Ordering under each rule: `unit-gain` signmuon > muonsign > signsgd; `mup` signmuon > muonsign > signsgd; `none` signmuon > muonsign > signsgd

**The ordering is the same under every rule.**

## Diagnostics

`gain spread` is max/min over layers of the realized `||lambda*s||_F/sqrt(fan_out)`;
the per-layer rule targets 1.00x, and the sign family is 1.00x by construction.
`MV ties` and `uplink zeros` are RAW rates, counted before the randomized-pm1
mapping, so they are diagnostics and feed no accounting.

| method | seed | gain spread (round 1 -> final) | MV ties | uplink zeros |
| :--- | ---: | ---: | ---: | ---: |
| muon | 0 | 1.08x -> 1.08x | -- | -- |
| muon | 1 | 1.10x -> 1.11x | -- | -- |
| muon | 2 | 1.06x -> 1.09x | -- | -- |
| muon | 3 | 1.05x -> 1.10x | -- | -- |
| muon | 4 | 1.05x -> 1.11x | -- | -- |
| muonserver | 0 | 1.07x -> 1.13x | -- | -- |
| muonserver | 1 | 1.11x -> 1.12x | -- | -- |
| muonserver | 2 | 1.13x -> 1.16x | -- | -- |
| muonserver | 3 | 1.13x -> 1.10x | -- | -- |
| muonserver | 4 | 1.09x -> 1.11x | -- | -- |
| signmuon | 0 | 1.00x -> 1.00x | 0.0000 | 0.0019 |
| signmuon | 1 | 1.00x -> 1.00x | 0.0000 | 0.0019 |
| signmuon | 2 | 1.00x -> 1.00x | 0.0000 | 0.0019 |
| signmuon | 3 | 1.00x -> 1.00x | 0.0000 | 0.0046 |
| signmuon | 4 | 1.00x -> 1.00x | 0.0000 | 0.0019 |
| ef21signmuon | 0 | 1.24x -> 1.08x | -- | 0.0000 |
| ef21signmuon | 1 | 1.26x -> 1.09x | -- | 0.0000 |
| ef21signmuon | 2 | 1.37x -> 1.09x | -- | 0.0000 |
| ef21signmuon | 3 | 1.29x -> 1.09x | -- | 0.0000 |
| ef21signmuon | 4 | 1.39x -> 1.11x | -- | 0.0000 |
| muonusign | 0 | 1.16x -> 1.05x | 0.0000 | 0.0054 |
| muonusign | 1 | 1.26x -> 1.06x | 0.0000 | 0.0168 |
| muonusign | 2 | 1.20x -> 1.09x | 0.0000 | 0.0054 |
| muonusign | 3 | 1.19x -> 1.07x | 0.0000 | 0.0012 |
| muonusign | 4 | 1.09x -> 1.08x | 0.0000 | 0.0050 |
| ef21muonsign | 0 | 1.07x -> 1.01x | -- | 0.0000 |
| ef21muonsign | 1 | 1.10x -> 1.03x | -- | 0.0000 |
| ef21muonsign | 2 | 1.05x -> 1.02x | -- | 0.0000 |
| ef21muonsign | 3 | 1.08x -> 1.05x | -- | 0.0000 |
| ef21muonsign | 4 | 1.06x -> 1.02x | -- | 0.0000 |
| ef21muonusign | 0 | 1.15x -> 1.06x | -- | 0.0000 |
| ef21muonusign | 1 | 1.12x -> 1.06x | -- | 0.0000 |
| ef21muonusign | 2 | 1.08x -> 1.09x | -- | 0.0000 |
| ef21muonusign | 3 | 1.13x -> 1.07x | -- | 0.0000 |
| ef21muonusign | 4 | 1.05x -> 1.02x | -- | 0.0000 |
| signsgd | 0 | 1.00x -> 1.00x | 0.0000 | 0.0121 |
| signsgd | 1 | 1.00x -> 1.00x | 0.0000 | 0.0313 |
| signsgd | 2 | 1.00x -> 1.00x | 0.0000 | 0.0123 |
| signsgd | 3 | 1.00x -> 1.00x | 0.0000 | 0.0167 |
| signsgd | 4 | 1.00x -> 1.00x | 0.0000 | 0.0144 |
| muonsign | 0 | 1.00x -> 1.00x | 0.0000 | 0.0100 |
| muonsign | 1 | 1.00x -> 1.00x | 0.0000 | 0.0373 |
| muonsign | 2 | 1.00x -> 1.00x | 0.0000 | 0.0154 |
| muonsign | 3 | 1.00x -> 1.00x | 0.0000 | 0.0199 |
| muonsign | 4 | 1.00x -> 1.00x | 0.0000 | 0.0173 |
| sgd | 0 | 1.11x -> 1.49x | -- | -- |
| sgd | 1 | 1.20x -> 1.30x | -- | -- |
| sgd | 2 | 1.35x -> 1.59x | -- | -- |
| sgd | 3 | 1.27x -> 1.39x | -- | -- |
| sgd | 4 | 1.16x -> 1.27x | -- | -- |
| adam | 0 | --x -> --x | -- | -- |
| adam | 1 | --x -> --x | -- | -- |
| adam | 2 | --x -> --x | -- | -- |
| adam | 3 | --x -> --x | -- | -- |
| adam | 4 | --x -> --x | -- | -- |
| adam+scaled | 0 | --x -> --x | -- | -- |
| adam+scaled | 1 | --x -> --x | -- | -- |
| adam+scaled | 2 | --x -> --x | -- | -- |
| adam+scaled | 3 | --x -> --x | -- | -- |
| adam+scaled | 4 | --x -> --x | -- | -- |

## The learning-rate search

Selected on validation accuracy only; no tuning run scores a test image.

| method | eta_0 | val acc (%) | rounds |
| :--- | ---: | ---: | ---: |
| muon | 0.01 | 82.26 | 2000 |
| muon | 0.02 | 83.63 | 2000 |
| muon | 0.05 | 84.84 | 2000 |
| muon | 0.1 | 85.51 | 2000 |
| muon | 0.2 | 84.71 | 2000 |
| muonserver | 0.01 | 84.84 | 2000 |
| muonserver | 0.02 | 84.80 | 2000 |
| muonserver | 0.05 | 85.46 | 2000 |
| muonserver | 0.1 | 85.08 | 2000 |
| muonserver | 0.2 | 83.52 | 2000 |
| signmuon | 0.005 | 82.30 | 2000 |
| signmuon | 0.01 | 83.70 | 2000 |
| signmuon | 0.02 | 84.24 | 2000 |
| signmuon | 0.05 | 84.70 | 2000 |
| signmuon | 0.1 | 84.72 | 2000 |
| signmuon | 0.2 | 83.44 | 2000 |
| signmuon | 0.5 | 8.78 | 2000 |
| signmuon+mup | 0.2 | 82.95 | 2000 |
| signmuon+mup | 0.5 | 84.30 | 2000 |
| signmuon+mup | 1.0 | 84.80 | 2000 |
| signmuon+mup | 2.0 | 85.02 | 2000 |
| signmuon+mup | 5.0 | 83.66 | 2000 |
| signmuon+none | 0.0002 | 82.18 | 2000 |
| signmuon+none | 0.0005 | 83.91 | 2000 |
| signmuon+none | 0.001 | 84.13 | 2000 |
| signmuon+none | 0.002 | 85.42 | 2000 |
| signmuon+none | 0.005 | 85.21 | 2000 |
| ef21signmuon | 0.005 | 78.50 | 2000 |
| ef21signmuon | 0.01 | 82.09 | 2000 |
| ef21signmuon | 0.02 | 84.62 | 2000 |
| ef21signmuon | 0.05 | 84.21 | 2000 |
| ef21signmuon | 0.1 | 83.94 | 2000 |
| muonusign | 0.01 | 80.81 | 2000 |
| muonusign | 0.02 | 83.20 | 2000 |
| muonusign | 0.05 | 84.25 | 2000 |
| muonusign | 0.1 | 83.76 | 2000 |
| muonusign | 0.2 | 80.79 | 2000 |
| ef21muonsign | 0.01 | 83.08 | 2000 |
| ef21muonsign | 0.02 | 83.57 | 2000 |
| ef21muonsign | 0.05 | 83.80 | 2000 |
| ef21muonsign | 0.1 | 61.55 | 2000 |
| ef21muonsign | 0.2 | 8.78 | 2000 |
| ef21muonusign | 0.005 | 80.66 | 2000 |
| ef21muonusign | 0.01 | 83.14 | 2000 |
| ef21muonusign | 0.02 | 82.99 | 2000 |
| ef21muonusign | 0.05 | 82.83 | 2000 |
| ef21muonusign | 0.1 | 73.00 | 2000 |
| signsgd | 0.005 | 77.75 | 2000 |
| signsgd | 0.01 | 80.45 | 2000 |
| signsgd | 0.02 | 80.18 | 2000 |
| signsgd | 0.05 | 68.93 | 2000 |
| signsgd | 0.1 | 44.58 | 2000 |
| signsgd+mup | 0.2 | 79.42 | 2000 |
| signsgd+mup | 0.5 | 80.92 | 2000 |
| signsgd+mup | 1.0 | 76.68 | 2000 |
| signsgd+mup | 2.0 | 54.09 | 2000 |
| signsgd+mup | 5.0 | 8.78 | 2000 |
| signsgd+none | 0.0002 | 79.66 | 2000 |
| signsgd+none | 0.0005 | 81.42 | 2000 |
| signsgd+none | 0.001 | 78.26 | 2000 |
| signsgd+none | 0.002 | 62.73 | 2000 |
| signsgd+none | 0.005 | 8.78 | 2000 |
| muonsign | 0.005 | 77.22 | 2000 |
| muonsign | 0.01 | 80.40 | 2000 |
| muonsign | 0.02 | 81.47 | 2000 |
| muonsign | 0.05 | 80.09 | 2000 |
| muonsign | 0.1 | 66.03 | 2000 |
| muonsign+mup | 0.2 | 78.75 | 2000 |
| muonsign+mup | 0.5 | 81.94 | 2000 |
| muonsign+mup | 1.0 | 81.80 | 2000 |
| muonsign+mup | 2.0 | 77.72 | 2000 |
| muonsign+mup | 5.0 | 8.78 | 2000 |
| muonsign+none | 0.0002 | 78.37 | 2000 |
| muonsign+none | 0.0005 | 81.65 | 2000 |
| muonsign+none | 0.001 | 81.78 | 2000 |
| muonsign+none | 0.002 | 78.34 | 2000 |
| muonsign+none | 0.005 | 8.78 | 2000 |
| sgd | 0.005 | 72.24 | 2000 |
| sgd | 0.01 | 75.32 | 2000 |
| sgd | 0.02 | 78.27 | 2000 |
| sgd | 0.05 | 80.94 | 2000 |
| sgd | 0.1 | 80.23 | 2000 |
| adam | 0.0002 | 71.51 | 2000 |
| adam | 0.0005 | 75.61 | 2000 |
| adam | 0.001 | 76.66 | 2000 |
| adam | 0.002 | 67.30 | 2000 |
| adam | 0.005 | 41.35 | 2000 |
| adam+scaled | 0.005 | 69.75 | 2000 |
| adam+scaled | 0.01 | 73.19 | 2000 |
| adam+scaled | 0.02 | 77.00 | 2000 |
| adam+scaled | 0.05 | 77.46 | 2000 |
| adam+scaled | 0.1 | 67.94 | 2000 |

## Not included

* `results/federated/tune_lr_adam_scaled_unit-gain_0.005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_scaled_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_scaled_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_scaled_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_scaled_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_unit-gain_0.0002_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_unit-gain_0.0005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_unit-gain_0.001_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_unit-gain_0.002_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_adam_unit-gain_0.005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonsign_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonsign_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonsign_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonsign_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonsign_unit-gain_0.2_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonusign_unit-gain_0.005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonusign_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonusign_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonusign_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21muonusign_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21signmuon_unit-gain_0.005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21signmuon_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21signmuon_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21signmuon_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_ef21signmuon_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muon_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muon_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muon_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muon_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muon_unit-gain_0.2_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonserver_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonserver_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonserver_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonserver_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonserver_unit-gain_0.2_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonsign_unit-gain_0.005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonsign_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonsign_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonsign_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonsign_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonusign_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonusign_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonusign_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonusign_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_muonusign_unit-gain_0.2_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_sgd_unit-gain_0.005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_sgd_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_sgd_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_sgd_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_sgd_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_sgd_unit-gain_0.2_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_sgd_unit-gain_0.5_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signmuon_unit-gain_0.005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signmuon_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signmuon_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signmuon_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signmuon_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signsgd_unit-gain_0.005_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signsgd_unit-gain_0.01_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signsgd_unit-gain_0.02_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signsgd_unit-gain_0.05_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_lr_signsgd_unit-gain_0.1_e400_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_verify_ef21muonsign_unit-gain_0.02_e2000_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_verify_ef21muonsign_unit-gain_0.05_e2000_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_verify_signmuon_unit-gain_0.05_e2000_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
* `results/federated/tune_verify_signmuon_unit-gain_0.1_e2000_t/seed0` -- sign convention uplink_zeros=keep, mv_ties=zero; the reported runs used random/random
