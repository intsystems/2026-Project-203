# Machinery for experimental EF21 updates

The runnable tools and usage examples are in [code/analysis](../../code/analysis/README.md).
The schedule inequalities are in
[Experiments/Schedules.lean](../../lean_proofs/EF21Muon/Experiments/Schedules.lean).
Complete multilayer stochastic tracking and stationarity theorems, including
random compressor ties, are now in
[RandomizedBlocks.lean](../../lean_proofs/EF21Muon/Experiments/RandomizedBlocks.lean).
Their scope and correspondence with the paper are recorded in the
[proof audit](../../lean_proofs/PAPER_AUDIT.md).

[FIXED_MOMENTUM.md](FIXED_MOMENTUM.md) now derives an explicit stochastic
tracking-error budget for both momentum filters and both downlinks under
ordinary Frobenius smoothness and uniform contraction/oracle bounds. It includes
compression roundoff, worker averaging, initialization, and a numerical LMO
with multiplicative pairing quality. The scope of the numerical bound calculator
is stated there; the complete Lean trajectory results are linked above.

## 1. Use the update that the experiments actually take

The smooth squared-gradient proof uses the sharp step
`X_next = X - gamma * ||G||_* * polar(G)`. The experimental matrix optimizer uses
`X_next = X - eta_t * lambda_layer * D`, with D computed by Newton–Schulz or
Polar Express. Thus schedule substitution in the sharp-step bound is
insufficient even with fixed momentum. A fixed-radius step naturally gives a
first-moment nuclear-gradient bound, with an explicit direction error.

The CIFAR experiments use fixed momentum and constant/cosine learning rates.
`nanogpt/train_gpt.py` additionally warms momentum from 0.85 to its target over
300 steps and cools it over the last 50; its learning rate is stable then
linearly decaying, followed by an iteration extension. The supplied historical
report uses the current 2290-step decay horizon and 2330 total steps. Its
fixed-0.95 filter example is separate from that varying momentum schedule.

## 2. A schedule bound with an explicit residual

For one matrix block, write `h_t` for the actual estimator supplied to the LMO,
`g_t = grad f(X_t)` for the population gradient at the server, and D_t for the
implemented direction. Suppose

```text
||D_t||_op <= r_t,
||h_t||_* - <h_t,D_t> <= delta_t.
```

`radius_direction_bound` proves

```text
<g_t,D_t> >= ||g_t||_* - (1+r_t)||g_t-h_t||_* - delta_t.
```

This accommodates an approximate oracle; assuming operator norm one for the
actual Newton–Schulz output would skip a required check. Delta may be replaced
by its nonnegative part. With Frobenius L-smoothness and the fixed-radius step,
the descent remainder includes `L*eta_t^2*||D_t||_F^2/2`.

More generally, if the actual descent inequality is

```text
F_(t+1) <= F_t - eta_t*S_t + eta_t*e_t + C*eta_t^2/2,
```

and the final objective is bounded below by fmin, Lean's
`scheduled_stationarity` gives, for positive total learning-rate mass,

```text
sum_t eta_t*S_t / sum_t eta_t
  <= (F_0-fmin)/sum_t eta_t
     + sum_t eta_t*e_t / sum_t eta_t
     + (C/2)*sum_t eta_t^2 / sum_t eta_t.
```

Here S_t is the nuclear gradient norm, not its square. No monotonicity of the
schedule is required; zero steps are allowed. One may use sequences of
expectations once the one-step expected inequality and integrability have been
established separately. These algebraic lemmas do not supply those premises.

`scheduled_residual` proves the corresponding bound with error term epsilon
when each e_t is bounded above by epsilon. Constant eta leaves a curvature
term proportional to eta. A cosine schedule with the same positive peak eta
also has `sum eta^2 / sum eta` approaching `3*eta/4`; annealing within the run
does not alone make this upper-bound contribution disappear as the horizon
grows. Shrinking the peak removes that contribution, but does not automatically
control the tracking error. A residual in this upper bound is not a lower bound
on the algorithm's attainable error.

For constant momentum mu and fresh centered noise at a frozen query, the EMA's
limiting variance gain is `(1-mu)/(1+mu)`. For the implemented Nesterov filter
`M_t=mu*M_(t-1)+(1-mu)*g_t`, `H_t=mu*M_t+(1-mu)*g_t`, it is

```text
(1-mu)^2*(1+mu)^2 + (1-mu)*mu^4/(1+mu).
```

The calculator checks these gains against their impulse responses. They concern
the noise filter, not the whole optimization error: changing queries, uplink
compression, and downlink lag add other terms. They do not model nanoGPT's
momentum warmup/cooldown.

Auxiliary AdamW steps, decoupled weight decay, and normalization/data-dependent
training objectives require their own descent accounting. We do not claim the
matrix-only theorem proves convergence of that entire pipeline.

## 3. What alpha(R) says

For a nonzero compressor block of d entries,

```text
alpha(R) = ||R||_1^2 / (d*||R||_F^2),
||R-C(R)||_F^2 = (1-alpha(R))*||R||_F^2,
effective_support = d*alpha(R).
```

The scaled-sign identity is already verified in
[ScaledSign.lean](../../lean_proofs/EF21Muon/Compression/ScaledSign.lean).
Alpha measures concentration of magnitudes. A constant, all-positive dense
block has alpha=1; a one-coordinate block has alpha=1/d. Low rank, common sign,
and directional coherence alone do not imply small alpha. The ratio is
invariant to uniform nonzero rescaling of a *given* residual. Changing the
learning rate can change the residual's future shape, so this invariance does
not prove that learning-rate tuning cannot help.

Alpha is undefined at R=0; the new CSV leaves it empty and records zero residual
energy. Sampling and taking medians can hide a bad block. Historical nanoGPT
Q/K/V/O summaries also averaged the four per-block ratios. Such summaries cannot
be inserted as a uniform compressor constant in the existing theorem.

## 4. Track the driven residual rather than alpha alone

Let `e_t=X_t-W_t`, `u_t=X_(t+1)-X_t`, and `R_t=e_t+u_t`. Then

```text
e_(t+1) = R_t - C(R_t),
||e_(t+1)||_F <= q_t * (||e_t||_F + ||u_t||_F),
q_t = sqrt(1-alpha(R_t))             [ideal scaled sign, nonzero residual].
```

The measured norm ratio can replace q_t to include implementation rounding.
`tracking_step` and `tracking_envelope` prove the step inequality and its
finite-horizon recursion with arbitrary nonnegative multipliers. There is no
assumption of an unobserved future uniform alpha. With e_0=0 the envelope is
the sum of past movements, each multiplied by all subsequent q factors.

The CSV records the exact decomposition of residual energy,
`||R||^2=||e||^2+||u||^2+2<e,u>`, actual post-gap energy, and the deviation from
the ideal scaled-sign energy identity. A complete contiguous trace reconstructs
a retrospective envelope; a sampled trace does not. A current large alpha does
not imply a small accumulated gap, and a small alpha need not cause a large gap
if the incoming movement is sufficiently small.

When the relevant population gradient has Lipschitz bound
`||grad f(W)-grad f(X)||_* <= Lgap*||W-X||_F`,
`query_gradient_transfer` proves
`||grad f(W)||_* <= ||grad f(X)||_* + Lgap*||W-X||_F`.
The complete model gap may be needed for coupled layers. This is why a server
stationarity guarantee cannot simply be assigned to the broadcast model.

## 5. What the existing nanoGPT data establish

[nanogpt_report.json](nanogpt_report.json) is regenerated from the shipped
[diagnostics.csv](../../code/results/nanogpt/diagnostics.csv). On the
EF21-MuonSign run, final sampled group medians are:

| Layer type | Downlink alpha | Relative server/broadcast gap |
| --- | ---: | ---: |
| attention Q/K/V/O | 0.3803 | 0.001121 |
| MLP c_fc | 0.2529 | 0.001339 |
| MLP c_proj | 0.0001233 | 0.08274 |

This establishes an association between poor compression and a persistent gap.
It does not establish that zero initialization caused concentration, that this
layer alone caused the loss offset, or that a particular alternative compressor
is the only repair. Those causal claims need controlled interventions.

The next useful neural-network comparisons hold seeds, minibatches, schedules,
and evaluation points fixed while varying (a) the downlink on c_proj,
(b) its compressor granularity, and (c) its initialization. Evaluate both X and
W on the same held-out batches. First capture a complete c_proj trace; then use
sampled SVD probes to separate poor compression from an inaccurate LMO. A changed
granularity or exact downlink changes communication accounting and must be
reported as an ablation.

## 6. Reproducible calibration and verification

[calibration.json](calibration.json) contains 16 CPU cases using the actual
shared optimizer on a known quadratic, with both momenta and both schedules,
both downlinks and both initializations. The population gradient is known, so
each finite-trajectory stationarity inequality can be evaluated. The terms are
retrospective and may be loose; this is not an estimated neural-network rate.

[calibration_trace.csv](calibration_trace.csv),
[its bound](calibration_trace_result.json), and
[its tracking report](calibration_trace_report.json) give one complete example.
Reproduction commands are in [code/analysis/README.md](../../code/analysis/README.md).

New diagnostic CPU tests and the nanoGPT optimizer recurrence tests pass.
The broader CPU suite passed 86/96 tests; ten tests require `torchvision`, which
is absent from this environment. No large GPU training run was launched.
Lean build, axiom, and Comparator verification are recorded in
[lean_proofs](../../lean_proofs/README.md); the experimental targets certify the
descent/schedule machinery and the full adapted N-worker fixed-momentum
trajectory theorem in
[FreshTrajectory.lean](../../lean_proofs/EF21Muon/Experiments/FreshTrajectory.lean).
The latter includes closed tracking and stationarity bounds for one smooth
matrix block, both downlinks and a certified approximate direction.
[BlockTrajectory.lean](../../lean_proofs/EF21Muon/Experiments/BlockTrajectory.lean)
extends this to simultaneous blocks under the earlier paper's arbitrary-pair
layerwise smoothness assumptions, with one filtration and a single global
objective gap. Those assumptions force block independence of the population
gradients. The extension does not certify the full neural-network training
pipeline.
