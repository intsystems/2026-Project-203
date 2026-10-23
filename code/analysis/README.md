# Experimental schedules and EF21 diagnostics

Run these commands from `code/`. Offline reports use only Python's standard
library. Calibration and live diagnostics require PyTorch. The analysis assumes
ordinary Frobenius smoothness and uniform compression and direction-quality
bounds.

## Inspect the existing nanoGPT measurements

```sh
python -m analysis.ef21_report --trace results/nanogpt/diagnostics.csv --schedule stable-linear --steps 2330 --schedule-horizon 2290
```

The schedule is an explicitly generated example, not recovered from the CSV.
The separate fixed-momentum filter gains do not model nanoGPT's momentum
warmup/cooldown. Historical CSVs contain sampled group medians: they support
descriptive comparisons, not a uniform contraction assumption or a complete
tracking recurrence. The report never labels them stationarity certificates.

## Record a new downlink trace

Add these options to a centralized or federated **EF21-MuonSign** run:

```text
--compression-log traces/downlink.csv --compression-every 1
```

`--compression-layers REGEX` selects parameter names. Each row includes the
actual learning rate and momentum, pre-gap energy, server-step energy, their
inner product, residual energy, post-gap energy, effective support, and the
observed tracking multiplier. Zero residuals have an empty alpha, not alpha=0.
Centralized logs also include uplink target/estimator lag. Federated logs leave
that lag empty because the uncompressed client targets are not retained.

`--compression-oracle` adds exact SVD measurements of the estimator's nuclear
norm, the implemented direction's operator norm, and its pairing defect. It
audits the direction actually used; it does not change the optimizer's LMO.
SVD is expensive on large blocks, so sample it sparingly.

The sidecar must be a new file. Logging synchronizes with the host and affects
timing; use it for diagnosis, not reported speed comparisons. It consumes no
random numbers and does not change updates. Sampling with `--compression-every`
larger than one saves work but prevents complete recurrence reconstruction.

### nanoGPT

The full repository's `nanogpt/train_gpt.py` accepts these environment variables:

```sh
export SIGNMUON_OPT=EF21-MuonSign
export SIGNMUON_COMPRESSION_LOG=traces/cproj
export SIGNMUON_COMPRESSION_LAYERS=c_proj
export SIGNMUON_COMPRESSION_EVERY=1
```

This writes `traces/cproj.rank0.csv`, etc., one file per owning rank, after
kernel warmup and state restoration. Set `SIGNMUON_COMPRESSION_ORACLE=1` for
the optional SVD audit. Use the normal launch command from
[nanoGPT's README](../nanogpt/README.md). Q/K/V/O are recorded as four actual
compressor blocks, and identically named layers keep their numerical indices.
The standalone optimizer can instead accept a caller-supplied recorder via
`set_compression_recorder(recorder, parameter_names)`.

Analyze a raw trace with:

```sh
python -m analysis.ef21_report --trace traces/cproj.rank0.csv --output traces/report.json
```

The analyzer checks step continuity and the continuity of gap energies. For a
complete segment it reconstructs a retrospective norm envelope using observed
compression multipliers. This is not a bound for unseen rounds or population
gradients. Keep the `generated_schedule_example` separate from the recorded
per-row learning rates.

## Calibrate against a known population gradient

```sh
python -m analysis.ef21_calibrate --steps 40 --output calibration.json
python -m analysis.ef21_calibrate --steps 40 --trace trace.csv --output traced-case.json
python -m analysis.ef21_report --trace trace.csv --steps 40 --lr 0.03
```

The first command runs 16 small controlled cases: constant/cosine schedule,
EMA/Nesterov, exact/sign downlink, and zero/random initialization. Each uses the
actual shared optimizer on a known quadratic with stochastic gradients. The
script checks the finite-trajectory, learning-rate-weighted nuclear-gradient
bound at both the server and broadcast models. It reports the initial gap,
tracking/oracle error, curvature, and query-transfer terms separately.
These are numerical calibrations of a pathwise inequality, not new stochastic
rates or evidence about a neural network's initialization.

## Closed fixed-momentum bounds

`analysis.ef21_fixed_momentum.fixed_momentum_bound` evaluates the explicit
stochastic budget in [ef21_fixed_momentum.py](ef21_fixed_momentum.py).
Its inputs must be uniform mathematical bounds. Sample medians of alpha and
sampled maxima of direction size do not provide those bounds for an NN run.
The function returns initialization, momentum lag, filtered noise, broadcast
lag, uplink tracking, and roundoff separately. `momentum_roundoff` and
`target_roundoff` allow bounded biased errors in the momentum/filter arithmetic;
server-step rounding is outside the result. `oracle_gain`
allows a certified multiplicative pairing guarantee for a numerical LMO.

```sh
python -m analysis.ef21_fixed_calibrate --steps 5 --output closed-bound-check.json
python -m unittest tests.test_fixed_momentum -v
```

The calibration enumerates the entire small noise law, including all adaptive
query histories. Direction bounds in that calibration cover the enumerated
support only. It exercises the shared Newton-Schulz LMO and both momenta,
downlinks, and schedules.

## Tests

```sh
python -m unittest tests.test_experiment_diagnostics tests.test_fixed_momentum -v
python nanogpt/test_signmuon_optimizers.py
```

The tests include logging/nonlogging trajectory and RNG equality, the actual
federated driver, Q/K/V/O block separation, undefined zero-residual alpha,
missing-round handling, schedule endpoints, filter gains, and known-gradient
stationarity calibration.
