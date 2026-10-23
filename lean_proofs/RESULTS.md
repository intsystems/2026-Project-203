# Mathematical results and scope

See the [README](README.md) for the paper correspondence, portable build instructions, and dependency versions. This document records the detailed statements, assumptions, constants, and limitations of each result.

## Verified deterministic baseline

The deterministic baseline covers **both EF21-MuonUSign and
EF21-MuonSign**, with one worker, one matrix layer, deterministic gradients, EMA
momentum, and LongTails' exact matrix polar oracle. No convexity is assumed.

| Module | Result |
| --- | --- |
| `EF21Muon/Compression/ScaledSign.lean` | Exact compression-error identity and squared Frobenius contraction with parameter `1 / (m n)` |
| `EF21Muon/Algorithm.lean` | Explicit momentum, estimator, server, and broadcast states |
| `EF21Muon/Tracking.lean` | Momentum, uplink, downlink, and combined tracking bounds derived from the updates |
| `EF21Muon/Geometry.lean` | Norm comparisons using LongTails' spectral/nuclear geometry |
| `EF21Muon/Convergence.lean` | Descent, Lyapunov decrease, and finite-time stationarity bound |
| `EF21Muon/Tuning.lean` | Explicit valid step sizes and separate statements for the two methods |
| `EF21Muon/Budget.lean` | Rounded iteration budget for a target squared-gradient accuracy |
| `EF21Muon/Protocol.lean`, `Execution.lean` | LongTails oracle protocol and proof that its execution produces the stated trajectory |
| `EF21Muon/Setups/Deterministic.lean` | Exact-gradient assumptions and an explicit restriction to the EF21 protocol |
| `EF21Muon/Theorems/DeterministicMuonUSign.lean`, `DeterministicMuonSign.lean` | Separate `OracleTemplate.Convergence` exports |

### Algorithm and assumptions

Write `beta = 1 - mu`, with `0 < beta <= 1`. Initialization is
`X = W = X0`, `M = G = 0`. Each round performs:

```text
X' = X - gamma * nuclearNorm(G) * polar(G)
W' = W + CP(X' - W)
M' = (1 - beta) * M + beta * grad f(W')
G' = G + CU(M' - G)
```

`CU` is the scaled sign. `CP` is the identity for MuonUSign and the scaled sign
for MuonSign. The first server step is zero under this initialization. The
proved trajectory cuts a round in the same place as the reference EF21 analysis.

The objective is differentiable, bounded below by `fmin`, and has a globally
`L`-Lipschitz gradient in the **Frobenius norm**. These are the assumptions of this
baseline; its `L` should not be silently identified with a differently defined
constant in the paper.

The trajectory fixes `sign(0) = +1`. The compressor lemmas separately allow any
compatible choice of -1 or +1 at zero, including each realization of random ties.
This baseline uses that fixed convention. [Random-tie extensions](RANDOM_TIES.md)
cover the stochastic convergence trajectories below.

### Explicit bound and tuning

Let `d = m n > 0`, `qU = sqrt(1 - 1/d)`, and take `qP = 0` for MuonUSign or
`qP = qU` for MuonSign. The following choices are proved valid:

```text
kappa = 1 + operatorNorm(operatorEquiv(m,n).inverse)
delta = min(beta, 1 - qU, 1 - qP) / 2
B     = 6 * beta * L / (1 - qP)
C     = 2 * L + qP * (B + 3 * beta * L)
gamma = delta / (4 * kappa^2 * (C + L + 1))
```

Here `operatorEquiv` is LongTails' map from flattened matrices with the Frobenius
norm to linear operators with the spectral norm. The proof establishes both
`normF(polar(G)) <= kappa` and `nuclearNorm(G) <= kappa * normF(G)`.

For every positive integer `T`, Lean proves

```text
(1/T) sum[t < T] nuclearNorm(grad f(X_t))^2
  <= 8 * (f(X0) - fmin) / (gamma * T)
     + 64 * kappa^2 * normF(grad f(X0))^2 / (delta * T).
```

The initialization term is retained. If `Delta >= f(X0) - fmin`,
`G0 >= normF(grad f(X0))`, and `epsilon > 0`, the explicit budget is

```text
T = max(1, ceil((8 * Delta / gamma + 64 * kappa^2 * G0^2 / delta) / epsilon^2)).
```

`Protocol` makes one exact-gradient call per round and returns the list
`[X0, ..., X_(T-1)]`; its success criterion is the average above, bounded by
`epsilon^2`. It does not evaluate extra gradients to select a best iterate.
The `allowedAlgorithm` field fixes this exact EF21 protocol in both exports.

### Verification and limits

`lake build` followed by `lake env lean Audit.lean` builds the project and
enforces the axiom allowlist for every declaration defined in the imported
`EF21Muon` modules, including private declarations.
`python scripts/check.py` runs these commands and checks that the dependency
pins remain unchanged. The audit also prints the
axioms used by selected compressor, tracking, convergence, budget, execution,
and divergence results. This is separate from the upstream Comparator's
independent check.

The axiom audit includes the convergence and divergence results listed below.
The optional Comparator audit covers the 45 frozen statements described in
[the LongTails section](#longtails-audit-and-accuracy-tuning).

The allowed axioms are `propext`, `Classical.choice`, and `Quot.sound`.
Unexpected dependencies, including `sorryAx`, cause the audit to fail.
The individual reports also include the general-distribution, distributed,
and divergence results below.

Local verification on 2026-10-23 with the pinned Windows toolchain: the full
library build and enforced axiom audit passed. All selected result reports used
only the allowed axioms.

This is a **deterministic baseline**, not a formalization of the paper's stochastic
theorem and not a tighter-constants claim. Its norm comparisons and scalar tracking
weights are conservative. The `O(1/T)` squared-gradient rate here uses exact
gradients; it cannot be compared directly with the stochastic `O(1/sqrt(T))` rate.
The step is the nuclear-norm-scaled schedule, not the fixed or cosine learning
rate used by the training runs. The later developments cover Nesterov momentum,
multiple workers/layers, fixed learning-rate schedules, and random ties. A finite
Newton-Schulz implementation is outside this baseline.

## Verified divergence results

Import `EF21Muon.Divergence.Theorems` for the divergence results, also exported
by the main `EF21Muon` module. Names below are relative to
`EF21Muon.Divergence`.

### Linear counterexamples (Theorems 1–3)

| Article result | Lean declarations | Exact descent pairing |
| --- | --- | --- |
| `th:1`, SignMuon, 4 by 4 | `SignMuon.polar_gradient`, `pairing`, `ascent`, `divergence` | `-42468 / 103` |
| `th:2`, MuonUSign, 5 by 5 | `MuonSign.polar_pattern`, `uplink_pairing`, `ascent`, `divergence` with `.muonUSign` | `6 - 82 * sqrt(17) / 17 < 0` |
| `th:3`, MuonSign, same 5 by 5 instance | `MuonSign.bidirectional_pairing`, `ascent`, `divergence` with `.muonSign` | `-76` |

The polar factors are proved from exact orthogonal and positive-definite
factorizations. The 5 by 5 certificate uses arithmetic in `Q(sqrt(17))`.
`LinearObjective` proves that `f(X) = inner(G,X)` is infinitely differentiable,
has gradient `G`, and satisfies spectral/nuclear smoothness for every `L >= 0`.
The linear instances are unbounded below.

`Linear.lean` formalizes Proposition `prop:reduction`: for both standard and
Nesterov momentum, every `0 <= mu < 1`, and arbitrary initialization, the
direction is constant and

```text
f(X_(t+1)) - f(X_t) = -eta_t * inner(G, direction(G)).
```

Positive learning rates give strict ascent. Divergence to positive infinity
requires that the partial sums of the learning rates tend to infinity; the
constant positive step case is included. A positive summable schedule is
claimed only to ascend.

### EF21-SignMuon (Theorem 4)

`EF21SignMuon.divergence` proves the complete statement of `th:ef_div` for
every `L > 0`, `eta > 0`, `0 <= mu < 1`, and both momentum variants. It
constructs an infinitely differentiable, globally nonnegative objective on
2 by 2 matrices satisfying

```text
nuclearNorm(grad F(X) - grad F(Y)) <= L * spectralNorm(X - Y).
```

For the actual algorithm `EF21SignMuon.run`, with zero position, momentum,
and estimate initially, it proves that some `c > 0` satisfies

```text
F(X_(t+2)) - F(X_t) = c * L * eta^2       for every t >= 3,
F(X_t) -> positive infinity.
```

| Article lemma | Lean declarations under `EF21SignMuon` |
| --- | --- |
| `lem:efsm_reduction`, scale reduction | `scale_reduction`, `run_scale` |
| `lem:efsm_twocycles`, alternating targets without the preamble | `harmless_cycle`, `harmless_average` |
| `lem:efsm_cycle`, wrong-sign cycle | `wrong_sign_cycle`, `wrong_sign_average`, `position_tail` |
| `lem:efsm_realize`, momentum and objective realization | `plateau_realization`, `prescribed_polar`, `normalized_run` |
| `rem:efsm_bdd`, boundedness below | `normalized_objective` and the general lemmas in `Clipping.lean` |

The normalized trajectory has objective increment `49/240` over each tail
period, and estimate entry `(2,2)` averaging `7/40` despite a constant target
entry `-7/25`. The realization includes the rank-one polar convention for
the second preamble target and the symbolic Nesterov preamble for all
`0 < mu < 1`; `mu = 0` is handled explicitly.

The formalization uses the article's periodic plateau ramps, three local
smooth corrections, and smooth scalar clipping to make the objective
nonnegative. `Plateaus.lean` constructs the ramps explicitly as the difference
of two translated periodic bumps. It proves smoothness, values in `[-1,1]`,
plateaus of half-width `1/100`, zero mean, periodic antiderivatives, and a
globally Lipschitz derivative. `EF21Field.lean` specializes these facts to the
paper's two periods and four residues. `plateau_realization` proves the exact
position, momentum, estimate, and objective increments on the raw objective.

The formal bounded-below extension clips the whole objective; the article
clips its diagonal term. Both preserve the gradients and objective increments
on the constructed trajectory. All smoothness,
gradient realization, and lower-bound obligations are proved. With `K` a
Frobenius Lipschitz constant for the normalized clipped gradient, the final
rescaling uses `L0 = kappa^2 * (K+1)` and the proved positive rate
`c = 49 / (240*L0)`.

These results use the exact LongTails polar oracle and the fixed convention
`sign(0) = +1`. The EF21-SignMuon update is defined separately in
`Divergence/EF21Algorithm.lean`, including its Nesterov effective momentum.

## Verified stochastic finite-sampling bound

`EF21Muon/Stochastic/Methods.lean` proves bounds for both methods with **one
worker, one matrix layer, and fresh independent samples from an arbitrary finite
distribution**. This includes uniform finite-dataset sampling with replacement.
The stochastic results are finite sample-path theorems; the separate deterministic
results above also have `OracleTemplate.Convergence` exports.

The stochastic assumptions use the paper's **spectral/nuclear smoothness**:

```text
nuclearNorm(grad f(X) - grad f(Y)) <= L * spectralNorm(X - Y).
```

The oracle is unbiased at every query and has Frobenius variance at most
`sigma^2`. `Oracle.ofNuclearVariance` proves that the paper's nuclear variance
assumption suffices. The proof derives the squared momentum and compression
recurrences from `replyStep`, cancels the unbiased-noise cross term, and proves
the matrix comparisons with `rho = sqrt(r)`, `r = min(m,n)`. It makes no convexity
or bounded-gradient assumption.

Let `a = 1/(m n)` and retain zero initialization `M0 = G0 = 0`, `W0 = X0`. Define

```text
W = 6*r/beta + 72*r*beta/a^2
N = 6*r*beta * (1 + (1-a)*beta/a + 12*beta^2/a^2).
```

For `T > 0`, the two method theorems establish

```text
(1/T) E[sum[t<T] nuclearNorm(grad f(X_t))^2]
  <= 4*(f(X0)-fmin)/(gamma*T)
     + 4*W*normF(grad f(X0))^2/T + 4*N*sigma^2.
```

`Sampling.paths` gives each finite sample path its product probability.
`expectedTotal_eq_paths` proves that the recursive expectation in the Lyapunov
argument equals the expected sum along those actual algorithm paths. The first
iterate is `X0`, and each round queries the stochastic oracle at the updated
broadcast point.

### Published constants

`Parameters.zeta_eq_paper` verifies algebraically the Theorem 19 / Remark 23
coefficient for a Frobenius compressor on the downlink:

```text
zeta = r*L^2 * (12/beta^2 + 24*r*(beta+2)/aP^2
             + 36*(beta^2+4)/a^2
             + 144*r*beta^2*(2*beta+5)/(aP^2*a^2)).
```

The certified step restriction is `gamma*(2*L + 2*sqrt(zeta)) <= 1`, equivalent
to the paper's reciprocal threshold when its denominator is positive. For
MuonSign, `aP = a`, so this is exactly its published one-worker, one-layer
coefficient. `muonUSign_stationarity` uses this same conservative Euclidean
downlink treatment with `aP = 1`.

`Parameters.zetaWith_one_eq_paper` also verifies the coefficient printed in
Theorem 19 for a spectral compressor: it has the same expression with the two
extra `r` factors inside the parentheses removed.

### Smaller sufficient step restriction for the exact downlink

`muonUSign_tighter_stationarity` uses the identity downlink directly. Since
`W' = X'`, its broadcast error vanishes exactly and the same bound holds with

```text
zeta_exact = r*L^2 * (12/beta^2 + 36*(beta^2+4)/a^2)
gamma*(2*L + 2*sqrt(zeta_exact)) <= 1.
```

The comparison `zeta_exact <= zeta_Theorem19` is proved in
`Parameters.zetaWith_zero_le_one`; it is strict for `L > 0` and `r > 0`, as
proved by `zetaWith_zero_lt_one`. Thus the exact downlink admits a larger
certified step size while preserving the noise and initialization terms.
This is a constants refinement in the finite-sampling, single-worker setting;
it is not a faster asymptotic rate or a novelty claim. No corresponding
improvement has been proved here for MuonSign's compressed downlink.

Evaluate these formulas with the standard-library-only calculator:

```sh
python scripts/compare_bounds.py --rows 32 --cols 32 --beta 0.1
```

For this example with `L = 1`, the MuonUSign threshold increases from
`7.00459e-6` to `7.18396e-6`, about 2.56%. At `beta = 1` it increases by a factor
of about 2.57. These are comparisons of sufficient theoretical thresholds;
the calculator reports the initialization and noise terms as well. It does
not run an optimizer or measure training performance.

The general-distribution and distributed extension below supersedes these sampling,
worker-count, and layer-count restrictions. The fixed-beta bound retains its noise
floor; the new accuracy-dependent tuning controls it explicitly.

## General noise, multiple workers and layers

`EF21Muon/Distributed/Methods.lean` instantiates the proof for both methods:
`matrix_stationarity false` is EF21-MuonUSign, and `matrix_stationarity true` is
EF21-MuonSign. It covers any positive number `N` of workers and any finite number
of layers with different matrix dimensions, smoothness constants, momentum
parameters, and step sizes. Each worker keeps its own momentum and compressed
estimator. The server applies each layer's polar oracle to the **average of the
worker estimators**. It broadcasts the new point before querying the workers.

The noise space is an arbitrary measurable space with a probability measure;
there is no finite-support restriction. Noise is centered at every query, has
a finite Frobenius second moment bounded by `sigma_i^2`, and is independent
across workers at each query and layer. Different layers may have correlated
noise. Each round uses a fresh sample. `Probability.pathMeasure` is the finite
product law, and the theorem integrates the sum along the actual `run` trajectory.
Integrability of the compressed errors, potential, and accumulated stationarity
is proved, rather than assumed as an extra trajectory condition.

`nuclear_variance_suffices` derives the required Frobenius moment and integrability
from a finite nuclear second moment, matching the paper's variance assumption.
`Probability.independent_average_variance` proves the `1/N` variance reduction;
it is not supplied as an assumption of the convergence theorem.

The objective is bounded below and differentiable with the stated gradient
blocks. Those blocks are the average of the local gradient blocks. The global
and local smoothness assumptions are exactly the arbitrary-pair, layerwise
spectral/nuclear inequalities in Assumption 2. In particular, this is the paper's
strong assumption that changing other layers alone does not change a gradient
block. The proof does not assume convexity, bounded gradients, equal worker
objectives, or independence across layers.

### Distributed bound and paper constants

For layer `i`, write `r_i = min(m_i,n_i)`, `a_i = 1/(m_i*n_i)`, and
`ell2_i = (1/N) sum_j L_(i,j)^2`. Define

```text
A_i = 6*r_i/beta_i
B_i = 72*r_i*beta_i/a_i^2
C_i = 6*r_i/a_i
I_i = A_i*beta_i^2*(1+2/beta_i)*L_i^2
      + B_i*beta_i^2*(1+2/beta_i)*ell2_i
      + C_i*6*beta_i^2*ell2_i/a_i
D_i = I_i*(2/aP_i-1)

noise_i = 6*r_i*beta_i * (1/N + (1-a_i)*beta_i/a_i
                         + 12*beta_i^2/a_i^2)

zeta_i = r_i * (12*L_i^2/beta_i^2
                + 24*tau_i*(beta_i+2)*L_i^2/aP_i^2
                + 36*(beta_i^2+4)*ell2_i/a_i^2
                + 144*tau_i*beta_i^2*(2*beta_i+5)*ell2_i/(aP_i^2*a_i^2))
```

`Distributed.zeta_eq_paper` proves this identity with distinct global and local
smoothness constants. The generic analysis also supports other measurable
contractive compressors. For the scaled-sign downlink, `aP_i = a_i` and
`tau_i = r_i`, giving the paper's Frobenius-compressor coefficient. For the
identity downlink, `aP_i = 1` and **`tau_i = 0`**, so all broadcast terms vanish.
`exact_downlink_le` proves this coefficient is no larger than any nonnegative
downlink factor. Thus the USign constants refinement extends to the general
noise, multiworker, multilayer setting.

For positive `gammaBar`, if every layer satisfies
`gamma_i*(2*L_i + 2*sqrt(zeta_i)) <= 1`, the result is

```text
(1/T) E[sum[t<T] sum_i (gamma_i/gammaBar)*nuclearNorm(grad_i f(X_t))^2]
  <= 4*Psi0/(gammaBar*T) + (4/gammaBar)*sum_i gamma_i*noise_i*sigma_i^2.
```

Set `gammaBar = (1/p) sum_i gamma_i` for the paper's weighted normalization.
For common steps, `uniform_cost` identifies the left side with the unweighted
sum of squared nuclear gradient norms. Zero initialization gives

```text
Psi0 = f(X0)-fmin
       + sum_i gamma_i * (A_i*normF(grad_i f(X0))^2
          + B_i*(1/N)*sum_j normF(grad_i f_j(X0))^2).
```

The initialization terms are retained. Only the averaged-momentum contribution
to `noise_i` receives `1/N`; compression and local tracking still incur local
noise. The theorem does not claim that the entire stochastic term scales as `1/N`.

### Explicit stochastic tuning and accuracy budget

The following conservative, accuracy-dependent choices are proved valid. They
are not a formalization of the paper's optimized minimum-of-root tuning.
For `epsilon > 0`, set

```text
eta_i = 6*r_i * (1/N + (1-a_i)/a_i + 12/a_i^2)
S     = sum_i eta_i*sigma_i^2
beta  = min(1, epsilon^2/(8*(S+1)))                  # common momentum
K_i   = 2*L_i + 2*sqrt(zeta_i)                     # computed with this beta
gamma = 1/(1 + sum_i K_i)                          # common step
T     = max(1, ceil(8*Psi0/(gamma*epsilon^2)))
```

`explicitControls` constructs the parameters in this order and proves their
positivity. `explicitControls_beta` and `explicitControls_gamma` verify the
formulas; the construction discards the original beta and gamma fields of its
base controls. `Problem.stationarity_at_budget` proves

```text
(1/T) E[sum[t<T] sum_i nuclearNorm(grad_i f(X_t))^2] <= epsilon^2.
```

The proof bounds the stochastic term by `epsilon^2/2` and the initialization/gap
term by `epsilon^2/2`. The budget depends on the initial potential; any known
upper bound on that potential can also be used by the finite-time bound. There
is one oracle query per worker per round, hence `T` parallel rounds and `N*T`
worker oracle calls.

### Smooth asymptotic rate with explicit horizon tuning

`Distributed/Rates.lean` closes the common-step asymptotic gap. For each layer,
the momentum-independent envelope

```text
Z_i = rho_i^2 * (12*L_i^2 + 72*tau_i*L_i^2/aP_i^2
      + 180*ell2_i/aU_i^2 + 1008*tau_i*ell2_i/(aP_i^2*aU_i^2))
```

satisfies `zeta_i <= Z_i/beta^2` for `0 < beta <= 1`.
Here `tau_i = 0` for the exact downlink and `tau_i = r_i` for the scaled-sign
downlink. Set

```text
g     = 1/(1 + sum_i (2*L_i + 2*sqrt(Z_i)))
beta  = 1/sqrt(T)
gamma = g*beta
B     = f(x0)-fmin + g*sum_i (6*rho_i^2*||grad_i f(x0)||_F^2
          + 72*rho_i^2/aU_i^2 * mean_j ||grad_i f_j(x0)||_F^2)
S     = sum_i noiseScale_i*sigma_i^2
```

For every positive horizon, `Problem.smooth_rate` proves

```text
stationarity(T) <= (4*B/g + 4*S)/sqrt(T).
```

The step restriction is proved from these choices. The initial potential and
noise coefficients are bounded by constants independent of the horizon.
`Problem.retune` preserves the objective, gradients, local smoothness constants,
oracle and noise law. `horizonControls` changes only momentum and the common
step; it uses `max(1,T)` so that the controls are also defined at horizon zero.
`Problem.horizon_rate_isBigO` and `matrix_horizon_rate_isBigO` prove the actual
Lean `Asymptotics.IsBigO` statement, instantiated for both EF21 methods.

`Distributed/WeightedRates.lean` proves the unequal-step statement as well.
Choose fixed `g_i > 0` with `g_i*(2*L_i + 2*sqrt(Z_i)) <= 1`; the explicit
choice `layerRateGamma_i = 1/(1 + 2*L_i + 2*sqrt(Z_i))` is certified. Set
`gamma_i(T) = g_i/sqrt(T)` and `gbar = mean_i g_i > 0`. Then

```text
B_g = f(x0)-fmin + sum_i g_i*(6*rho_i^2*||grad_i f(x0)||_F^2
        + 72*rho_i^2/aU_i^2 * mean_j ||grad_i f_j(x0)||_F^2)
S_g = sum_i g_i*noiseScale_i*sigma_i^2
weighted_stationarity(T) <= (4*B_g/gbar + 4*S_g/gbar)/sqrt(T).
```

`layer_horizon_cost` proves that its weights are exactly `g_i/gbar`, hence the
article's `gamma_i/mean_l gamma_l`, with no dependence on the horizon.
`Problem.weighted_horizon_rate_isBigO` transports one fixed objective and noise
law unchanged. `matrix_weighted_horizon_rate_isBigO` instantiates the actual
scaled-sign uplink and exact/scaled-sign downlinks of both methods. This proves
the explicit sufficient tuning now stated in Corollary 1; it does not claim to
optimize the upstream constants. The generalized-smoothness, constant-radius
rate of Theorem 5(ii)/Corollary 2 is proved separately below.

### Exact federated algorithm correspondence

`Distributed/Correspondence.lean` formalizes Proposition 2 by specifying both
round orders. `paperRun` performs gradient/uplink followed by step/downlink;
`frameworkRun` performs step/downlink followed by gradient/uplink. Both start
with zero worker momentum and estimates, and with the same server/client model.

`exact_instance` relates the full states after the client phase. `models_exact`
proves `X^(t+1) = X_t` and `W^(t+1) = W_t`. `aggregate_update` proves that the
server's incremental reconstruction equals the mean of the worker estimates.
`observable_sum` identifies the extra initial-model term; `average_transfer`
gives the corresponding stationarity-average inequality.

The statements allow arbitrary worker counts, heterogeneous layers, step
schedules, compressor realizations and gradient replies. They hold for each
fixed stochastic realization. `ofScheme_round` connects the round definition to
the `next` function used in the stochastic convergence proofs. Both concrete
downlink modes satisfy the required zero-preservation property. The abstract
server step also permits any fixed LMO selection that takes a zero step at a
zero estimate.

### Scaled-sign refinements

`Compression.scaledSign_realized_identity` proves the exact residual-dependent
contraction formula for nonzero inputs. `one_sparse_realizedContraction` proves
that every nonzero one-sparse input attains `1/d`.
`constant_magnitude_exact` proves exact reconstruction of equal-magnitude inputs.
`Compression/Asymptotics.lean` proves almost-sure convergence for pairwise
independent, identically distributed samples with integrable absolute and
squared values and nonzero second moment:

```text
alpha_n -> (E |X|)^2 / E[X^2]    almost surely.
```

`Compression/Gaussian.lean` derives the absolute and second moments from
Mathlib's actual centered Gaussian measure, with any fixed positive variance,
and specializes this theorem to `alpha_n -> 2/pi` almost surely. The Gaussian
moments are proved, not supplied as hypotheses.

`Compression/Density.lean` disproves the former claim that full support alone
implies dimension-independent contraction. In dimension `n+1`, `denseVector n`
is `(1,1/(n+1),...,1/(n+1))`: every entry is positive, but its contraction is at
most `4/(n+1)` and tends to zero. `effective_support_contraction` proves the
correct quantitative sufficient condition.

### Proof modules

| Module | Role |
| --- | --- |
| `Probability/Moments.lean` | L2 identities, worker averaging, independent-worker variance |
| `Probability/Updates.lean` | General-measure momentum and compression recurrences |
| `Probability/Paths.lean` | Product probability on paths and integrated Lyapunov induction |
| `Distributed/Layer.lean`, `Energy.lean` | Separate worker states and four tracking errors |
| `Distributed/Geometry.lean` | Simultaneous layerwise smoothness, polar descent, compressor measurability |
| `Distributed/Constants.lean` | Distributed paper coefficients and exact-downlink comparison |
| `Distributed/Setup.lean` | Objective/oracle assumptions and measurable synchronous updates |
| `Distributed/Convergence.lean`, `Bounds.lean` | Integrability, finite-time and weighted stationarity bounds |
| `Distributed/Tuning.lean`, `Methods.lean` | Explicit accuracy budget and both matrix method instantiations |
| `Distributed/Rates.lean`, `WeightedRates.lean` | Common and unequal layer schedules, finite-horizon rates, exact weights, and fixed-problem Big-O theorems |
| `Compression/Asymptotics.lean`, `Gaussian.lean`, `Density.lean` | Strong-law contraction limit, Gaussian specialization, and full-support counterexample |
| `Probability/MomentCounterexample.lean` | Probability measures with vanishing first moment and constant second moment |
| `Distributed/GeneralizedSteps.lean` | Squared constant-radius schedule and three sufficient generalized-smooth exponent conditions; no convergence claim |
| `Distributed/GeneralizedGeometry.lean` | Generalized-smooth simultaneous upper bound, constant-radius descent, and weighted gradient self-bounds; no constant global Lipschitz assumption |
| `Probability/FirstMoment.lean`, `Distributed/RadiusTracking.lean` | Worker and compression first moments, centered-noise EMA variance, and debiased global tracking |
| `Distributed/RadiusEnergy.lean`, `RadiusEnvelopes.lean` | First-moment Lyapunov drift and horizon-independent coefficient envelopes |
| `Probability/GeometricDrift.lean`, `PathStates.lean` | Integrated multiplicative drift, individual iterate integrability, and the minimum of iterate expectations |
| `Distributed/GeneralizedSetup.lean`, `GeneralizedDrift.lean` | Separate generalized-smooth problem, measurable radius updates, local objective self-bounds, and stochastic finite-time convergence |
| `Distributed/GeneralizedTuning.lean`, `GeneralizedRates.lean` | Article's radius/momentum schedule, explicit horizon-independent bound, unequal weights, and `O(T^(-1/4))` |
| `Distributed/GeneralizedCorrespondence.lean` | Erasure of the analysis-only noise EMA and connection to the paper's federated phases |
| `Distributed/GeneralizedPaper.lean` | Paper's round order, integrability at every paper iterate, sum of weighted expectations, and minimum-rate transfer through the index shift |
| `Divergence/Plateaus.lean` | Explicit smooth periodic ramps with the article's plateau widths, zero mean, and periodic potentials |
| `Distributed/Correspondence.lean` | Federated round rotation, full worker states, aggregate reconstruction and index shift |

The stochastic paper results are sample-path expectation theorems. Smooth
accuracy certificates now also use LongTails execution and trace semantics and
are included in the Comparator audit. Stochastic `OracleTemplate.Convergence`
packaging against a separate `Setup` remains further work. Fixed training
schedules are covered by the experimental theorems below;
[random-tie trajectories](RANDOM_TIES.md) are now covered separately.

## LongTails audit and accuracy tuning

### What is certified

Local verification on 2026-10-23 (Windows, pinned toolchain):
`python scripts/check.py --comparator` passed the full build, the axiom audit of
3,346 project declarations, dependency-pin preservation, and all 45 Comparator
targets, including Lean kernel replay. The
[saved Comparator record](comparator/VERIFICATION.json) includes the reference
and dependency-pin hashes. Adapter controls accepted a valid proof and rejected
both a mismatching statement and an unapproved axiom. Linux and macOS CI is
configured; it was not run locally.

The [target catalog](comparator/targets.json) contains 45 fixed statements.
The original eleven cover two deterministic `Convergence` exports, the smooth
EMA and Nesterov bounds, the two generalized-smooth first-moment bounds, two
stochastic execution certificates, and three budget/zero-noise identities.
Thirteen additions freeze three sharper polynomial bounds, five scalar
tuning/budget statements, three optimized method certificates, and two
comparisons with the previous method budgets. Eight further statements freeze
the recovered Nesterov worker averaging: full stochastic convergence, the
trajectory and paper polynomial bounds, coefficient dominance, the exact
leading variance coefficient, budget dominance, execution and paper accuracy.
Four experimental targets freeze fixed-radius descent with explicit estimator
and oracle errors, arbitrary-schedule stationarity, its bounded-error residual,
and a pathwise tracking envelope. Their conditional scope is described in
[the experiment notes](../research/experiments/README.md) and the source
[Schedules.lean](EF21Muon/Experiments/Schedules.lean); they do not discharge
the stochastic or full-training-pipeline premises.
Four fixed-momentum additions freeze the weighted scalar estimator-error
closure, its composition with schedule descent, the exact stochastic
EMA/Nesterov noise-filter bound, and the relative-pairing oracle inequality.
Three trajectory targets freeze the closed N-worker tracking bound, the
expected trajectory stationarity bound, and its adapted version under
conditional mean-zero innovations. The latter includes the filtration
measurability premises for communication errors and conclusion for the full
state. The assumptions and constants are described in
[FIXED_MOMENTUM.md](../research/experiments/FIXED_MOMENTUM.md).
Two multiple-block targets freeze per-block tracking and the complete adapted
stationarity theorem with the single global objective gap, explicit block
curvature terms, and the mean-layer normalization.
The [reference](comparator/Reference.lean.in) records their explicit types;
the [solutions](EF21Muon/ComparatorTargets.lean) prove those types. The checker
does not regenerate the reference from the solutions.

`python scripts/check.py --comparator` runs the ordinary library/axiom audit,
then the pinned upstream Comparator's comparison and axiom checks and its Lean
kernel replay on lean4export output. Only `propext`, `Classical.choice`, and
`Quot.sound` are permitted. Comparator also checks kernel primitives and the
quotient declarations. Challenge proof holes are permitted only in the isolated
reference, not in the solutions. Mathematical support modules and definitions
are shared with the reference. This checks statement preservation and proof
validity; it does not independently establish that a LaTeX claim was transcribed
correctly. The correspondence table and mathematical review remain necessary.

The native runner builds trusted sources directly. It does not use the upstream
Linux sandbox or protect against malicious build/elaboration code. No additional
kernel implementation is used. The frozen statements are certificate targets,
not an independent stochastic `Setup` for unrestricted algorithm submissions.

### Executed stochastic oracle and cost

[Execution.lean](EF21Muon/LongTails/Execution.lean) realizes the existing fresh
sample path as a LongTails algorithm. Its seed has the product law
`pathMeasure μ T` for an arbitrary probability measure `μ`; it consumes one new
sample per round. A query supplies all worker points and that round's sample to
the deterministic realization oracle `sampledOracle`. The reply contains the
`N` worker stochastic gradients, each over all model layers. This is the
prescribed sampled-gradient protocol, not a claim about arbitrary adaptive
oracle access. Worker independence and all moment, unbiasedness, smoothness,
and compressor assumptions are those of `Distributed.Problem`.

[Protocols.lean](EF21Muon/LongTails/Protocols.lean) proves that `execute` returns
exactly the trajectory used in the convergence proofs. The
[EMA](EF21Muon/Theorems/StochasticEMA.lean) and
[Nesterov](EF21Muon/Theorems/StochasticNesterov.lean) certificates establish both
its expected score and `executeWithTrace` length. There are exactly `T` batch
queries, charged as **`N T` worker full-gradient evaluations**. Polar-oracle
evaluation, communication, and matrix arithmetic are not included in this
gradient-query cost. Layers are components of a full worker gradient, not extra
independent queries. The output is the list of pre-update server points in the
rotated round convention used by the analysis; the existing paper-correspondence
proofs describe its index shift. No extra gradients select a best iterate.

### Sharper accuracy-dependent tuning

[Tuning.lean](EF21Muon/LongTails/Tuning.lean) proves a tuning improvement for
both EMA and Nesterov and both downlink modes in the smooth setting. The
stationarity criterion remains

```text
E[(1/T) sum_{t<T} sum_i nuclearNorm(grad_i f(X_t))^2] <= epsilon^2.
```

Let `g > 0` satisfy, for every layer,

```text
g * (2 L_i + 2 sqrt(rateEnvelope_i)) <= 1.
```

Use the corresponding EMA or Nesterov `rateEnvelope`. Both envelopes are
independent of momentum and step size. For example,
`g = 1 / (1 + sum_i (2 L_i + 2 sqrt(rateEnvelope_i)))` is sufficient.
Set `B` to the corresponding `rateInitial` at `g, X0` and `S` to the
corresponding `noiseScaleTotal` / `noiseTotalScale`. These explicit constants
include the initial local gradient errors and compressor dependence; **`B` is
not just the objective gap**. Their definitions are in
[EMA Rates.lean](EF21Muon/Distributed/Rates.lean) and
[Nesterov SmoothRates.lean](EF21Muon/Nesterov/SmoothRates.lean), with EMA noise
coefficients in [Distributed/Tuning.lean](EF21Muon/Distributed/Tuning.lean).
For `epsilon > 0`, choose

```text
beta    = epsilon^2 / (epsilon^2 + 8 S),
gamma_i = g beta,
T       = max(1, ceil(8 B / (g epsilon^2) + 64 B S / (g epsilon^4))).
```

The proof bounds the transient term by `epsilon^2/2` and the noise term by
`4 beta S <= epsilon^2/2`. It derives these bounds from the existing finite-time
Lyapunov inequalities and proves the execution score satisfies the criterion.
No convergence bound is assumed as a hypothesis.

The earlier accuracy tuning used `min(1, epsilon^2 / (8 (S + 1)))`.
The new formula removes its artificial positive noise offset. At `S = 0`,
`beta = 1` and the budget becomes `max(1, ceil(8 B / (g epsilon^2)))`.
For fixed problem and compressor constants, the new bound separates the
`epsilon^(-2)` deterministic term and the `S epsilon^(-4)` noise term. This
improves the low-noise dependence of that earlier certified tuning; it is not
a claim of pointwise dominance for every accuracy. Deterministic
`epsilon^(-2)` complexity was already available from the separate baseline.

This first tuning improvement gives **no new stochastic time exponent** and
does not establish an improvement over the strongest optimized bounds in
Gruntkowska et al. The following refinement optimizes a sharper certificate
from our own Lyapunov proof. The generalized-smooth first-moment results are
unchanged; the accuracy/query refinements below concern smooth objectives.
The shared article appendix now includes the polynomial bound and a sufficient
closed-form accuracy rule (`prop:polynomial_rate`, `cor:accuracy_budget`).
The numerical optimization and full oracle-certificate details remain here.

### Retaining the momentum powers

[Profiles.lean](EF21Muon/LongTails/Profiles.lean) proves the stronger finite-time
bound, for either momentum rule and the same safe fixed `g`,

```text
Q_T <= F_T(beta)
    := 4 [a/(beta T) + h beta/T + s1 beta + s2 beta^2 + s3 beta^3],
0 < beta <= 1,  gamma_i = g beta.
```

Here `Q_T` is the expected trajectory average of the sum of squared nuclear
gradient norms. All five coefficients are nonnegative and independent of
`beta,T` when the objective, noise law, compressors, initialization, and `g`
are fixed. The older certificate replaced `h beta/T` by `h/(beta T)` and
both higher noise powers by linear terms. We now retain these powers exactly.

Write `Delta=f(X0)-fmin`, `G_i=||grad_i f(X0)||_F^2`, and
`H_i=mean_j ||grad_i f_j(X0)||_F^2`. With `alpha_i=alphaU_i`, the EMA profile is

```text
a  = Delta/g + sum_i 6 rho_i^2 G_i,
h  = sum_i (72 rho_i^2/alpha_i^2) H_i,
s1 = sum_i (6 rho_i^2/N) sigma_i^2,
s2 = sum_i [6 rho_i^2 (1-alpha_i)/alpha_i] sigma_i^2,
s3 = sum_i (72 rho_i^2/alpha_i^2) sigma_i^2.
```

The earlier Nesterov profile uses the existing weights in
[SmoothEnergy.lean](EF21Muon/Nesterov/SmoothEnergy.lean):

```text
d_i = 8 rho_i^2,
u_i = 2 d_i/alpha_i,
j_i = d_i + 4 u_i/alpha_i,
r_i = 16 u_i/alpha_i + j_i,

a  = Delta/g + sum_i 2 d_i H_i,
h  = sum_i 6 r_i H_i,
s1 = sum_i 2 d_i sigma_i^2,
s2 = sum_i r_i sigma_i^2,
s3 = sum_i 6 r_i sigma_i^2.
```

The leading EMA noise coefficient retains its `1/N` worker averaging factor.
The earlier Nesterov profile has no such factor in `s1`. The new split-error
proof below recovers it and improves all five coefficients. The coefficients
include the compressor dependence and differ between the two methods.

For the actual Nesterov paper round order, `nesterov_paper_polynomial_bound`
proves the same expression with `s1,s2,s3` each doubled. It uses the existing
pathwise index-shift proof and assumes the downlink maps zero to zero, as do
both instantiated downlinks. The initial terms `a,h` are unchanged. Its score
is the existing `paperSmoothAverage`; it is not silently identified with the
rotated trajectory's score.

### Explicit budget with separated noise terms

[Polynomial.lean](EF21Muon/LongTails/Polynomial.lean) gives a closed choice:

```text
R1 = 24 s1/epsilon^2,
R2 = sqrt(24 s2/epsilon^2),
R3 = (24 s3/epsilon^2)^(1/3),

beta_bal = 1/(1+R1+R2+R3),
T_bal    = max(1, ceil[8(a+h)/epsilon^2 * (1+R1+R2+R3)]).
```

`Profile.bound_at_balancedBudget` proves `F_T_bal(beta_bal)<=epsilon^2`
for every `epsilon>0`, including zero coefficients and `a=0`. The three noise
terms together contribute at most `epsilon^2/2`; the transient terms contribute
at most the other half. [Optimized.lean](EF21Muon/Theorems/Optimized.lean)
instantiates this rule for EMA, Nesterov, and the Nesterov paper score.

Apart from the integer ceiling and the minimum of one round, the resulting
dependence is

```text
(a+h) [epsilon^(-2) + s1 epsilon^(-4)
                    + sqrt(s2) epsilon^(-3)
                    + s3^(1/3) epsilon^(-8/3)].
```

Previously all three noise coefficients multiplied `epsilon^(-4)`. Their
separation can reduce the budget when the leading coefficient is small and
the higher coefficients dominate. These are separate terms in one bound:
with fixed positive `s1`, the asymptotic stochastic accuracy exponent remains
four. Positive finite-worker noise gives positive `s1` for both profiles.
No general improvement over published rates is established here.

### Best constant momentum for this certificate

For a prescribed positive horizon, rewrite `F_T` as

```text
A/beta + B beta + C beta^2 + D beta^3,
A=4a/T, B=4(h/T+s1), C=4s2, D=4s3.
```

For `a>0`, `Profile.optimalBeta` globally minimizes this expression over
`0<beta<=1`. If `A>=B+2C+3D`, the minimizer is `beta=1`. Otherwise it is a
positive root in `(0,1)` of

```text
B beta^2 + 2 C beta^3 + 3 D beta^4 = A.
```

The proof establishes root existence by continuity and proves global
optimality by a nonnegative algebraic factorization; it does not rely on a
numerical optimizer. `Profile.optimalBudget` is the smallest positive integer
horizon at which this optimized certificate is at most `epsilon^2`.
`optimalBudget_minimal` proves that any other constant momentum satisfying
the same certificate requires at least that many rounds. Existence follows
from the explicit balanced budget above.

[Comparisons.lean](EF21Muon/LongTails/Comparisons.lean) proves exact identities
with the constants in the previous tuning pass:

```text
B_old = g(a+h),  S_old = s1+s2+s3,

T_opt <= max(1, ceil[8 B_old/(g epsilon^2)
                     + 64 B_old S_old/(g epsilon^4)]),
T_opt <= T_bal.
```

These inequalities are proved for both concrete profiles, not just their
orders of growth. For the Nesterov paper score, the first inequality uses
`S_old=2(s1+s2+s3)` to account for its transfer factor. It does not compare
the paper score against the smaller budget for a different trajectory score.

`Theorems.Optimized.ema` and `.nesterov` certify the optimized LongTails
execution's expected score and its exact `T_opt` batch queries, charged as
`N T_opt` worker gradient evaluations. `.nesterov_paper` separately proves the
actual paper score bound. These remain direct execution certificates under
the prescribed sampled-gradient protocol, rather than independent stochastic
`Setup`/`Convergence` exports.

Optimality is **within this scalar upper-bound certificate, for constant
momentum and fixed safe `g`**. It is not a lower bound on the actual algorithm,
nor an optimization over `g`, varying schedules, or other algorithms. The
optimal policy uses classical real-root selection and a least-integer
definition; executable numerical root finding is not verified here. The
explicit balanced policy also covers the degenerate case `a=0`, for which
the optimality theorems are not asserted.

### What retuning alone cannot prove

`Profile.certificate_barrier` proves that any positive `T,beta` with
`F_T(beta)<=epsilon^2` must satisfy

```text
64 a s1 <= epsilon^4 T.
```

For positive `a,s1`, an `epsilon^(-4)` term is therefore unavoidable **for
this certificate**, regardless of constant-momentum retuning. This follows
from its two nonnegative terms `4a/(beta T)` and `4s1 beta`. It is not a lower
bound on the algorithm's true stationarity or on oracle complexity for the
problem class. Improving that exponent requires additional mathematical
input beyond optimizing this bound.

### Recovering worker averaging for Nesterov

The new [averaged profile](EF21Muon/LongTails/AveragedNesterov.lean) and
[execution certificates](EF21Muon/Theorems/WorkerAveraged.lean) sharpen the
smooth Nesterov result for **both EF21-MuonUSign and EF21-MuonSign**. They use
the same `Problem`, `Scheme`, stochastic gradients, actual Nesterov updates,
stationarity score, and sufficient step envelope as the earlier theorem.
Worker noise is unbiased with bounded variance and independent across workers
at each frozen query; samples are fresh across rounds. Layers may have
correlated noise. No gradient bound or identical local objectives is added.

With the notation above, let `R_i = r_i-d_i = 40 d_i/alpha_i^2`. The new
profile has exactly the same polynomial form, with

```text
a_new  = Delta/g + sum_i 2 d_i G_i,
h_new  = sum_i (6 d_i G_i + 6 R_i H_i),
s1_new = sum_i (2 d_i/N) sigma_i^2 = s1_old/N,
s2_new = sum_i (d_i/N + R_i) sigma_i^2,
s3_new = sum_i (6 d_i/N + 6 R_i) sigma_i^2.
```

Thus the leading variance term in the trajectory bound is
`(64 beta/N) sum_i rho_i^2 sigma_i^2`, instead of
`64 beta sum_i rho_i^2 sigma_i^2`. This is a change to the convergence proof,
beyond retuning the earlier bound. The actual paper-round score satisfies the
same formula with all three noise coefficients doubled, as in the previous
transfer theorem, and with the unchanged initial terms.

#### Why averaging survives

[AveragedTracking.lean](EF21Muon/Nesterov/AveragedTracking.lean) separates
`E=||mean_j M_j-grad f||_F^2` from
`F=mean_j ||M_j-grad f_j||_F^2`. It also separates the averaged increment
`K=||mean_j(M_j-Mprev_j)||_F^2` from the local increment
`J=mean_j ||M_j-Mprev_j||_F^2`. Independent centered fresh noise contributes
`beta^2 sigma^2/N` to both global recurrences. The direction error uses
`E`, the compression error, and `K`; it need not be replaced by local errors.

[AveragedEnergy.lean](EF21Muon/Nesterov/AveragedEnergy.lean) gives `E,F,K,J`
the respective weights

```text
2 d/beta + 6 beta d,   6 beta R,   d,   4 u/alpha.
```

Their global/local sums equal the previous weights, so the previous
smoothness and broadcast restrictions still suffice. The compression and
broadcast error weights are unchanged. These identities are proved in
`weight_sums`; `layer_descent` proves the full expected Lyapunov drift.
[AveragedConvergence.lean](EF21Muon/Nesterov/AveragedConvergence.lean) proves
measurability, integrability, finite-horizon convergence and the paper-order
transfer. No stationarity estimate or trajectory integrability is assumed.

#### Comparison, tuning and costs

`averaged_nesterov_coefficients_le` proves all five new coefficients are no
larger than their earlier counterparts. It uses Jensen's inequality
`G_i <= H_i` and `N >= 1`. Consequently `averaged_nesterov_bound_le` holds
for every nonnegative `beta` and every horizon. For positive `a_new` and
accuracy, `averaged_nesterov_optimalBudget_le` proves the new best constant
momentum budget is no larger than the old best constant momentum budget, at
the same fixed safe step scale. In the single-worker case the profiles agree.

The existing explicit balancing rule applies to the new profile without a
positive-`a` restriction. `averaged_nesterov_balanced_accuracy` and
`averaged_nesterov_paper_balanced_accuracy` prove the corresponding actual
algorithm guarantees. `Theorems.WorkerAveraged.nesterov` additionally certifies
the expected LongTails execution score at the least horizon supported by this
profile, with exactly `T` batch queries and `N*T` worker gradient calls.
`.nesterov_paper` certifies the optimized paper-round score.

Only the leading noise coefficient scales entirely as `1/N`. Higher-order
compression terms remain, so this is not an all-regime linear-speedup claim.
For fixed positive worker count and positive leading noise, the scalar
certificate still has an `epsilon^(-4)` asymptotic term. Its leading stochastic
round cost has the recovered `1/N` factor; charging all worker calls multiplies
rounds by `N`. The improvement here concerns ordinary smoothness; generalized-smooth
Nesterov convergence is covered by its separate existing proof.

## Experimental fixed-radius error budgets

[Experiments/FixedMomentum.lean](EF21Muon/Experiments/FixedMomentum.lean)
proves update identities, the exact centered-noise filter variance, geometric
initialization bounds, and a closed weighted tracking budget from primitive
recurrences. It supports arbitrary nonnegative deterministic schedules with a
known peak, without requiring monotonicity. Relative oracle pairing quality
can preserve the rate exponent, with its constant explicitly charged.

The full analytical derivation is in
[FIXED_MOMENTUM.md](../research/experiments/FIXED_MOMENTUM.md). For one matrix
block with globally Lipschitz local gradients, it includes both momentum
filters, both downlinks, worker averaging, initialization, compression errors,
and optional momentum/target roundoff. A calculator and finite-noise-law
enumeration are provided in [code/analysis](../code/analysis/README.md).
Uniform contraction and direction-quality bounds are explicit assumptions;
sampled alpha medians cannot certify them. Server-step rounding and auxiliary
optimizer steps are outside this result.

[Experiments/Trajectory.lean](EF21Muon/Experiments/Trajectory.lean) now proves
the closed bound for the actual random N-worker updates, deriving the
compression recurrence, estimator error, objective drift and finite-horizon
integrability. The theorem `expected_stationarity` integrates the weighted
nuclear-gradient score along one complete trajectory on a common probability
space. Its right-hand side is the formula in the note, including the exact
`sigma*sqrt(filterGain(beta,theta)/N)` contribution.

[Experiments/FreshTrajectory.lean](EF21Muon/Experiments/FreshTrajectory.lean),
theorem `fresh_stationarity`, additionally derives cross-time orthogonality
from conditional mean-zero innovations and proves that all server, broadcast,
momentum and compressed-memory states are adapted to the filtration. Same-round
conditional worker cross moments must vanish; conditional independence with
centered noise is sufficient. No tracking or objective-drift recurrence,
trajectory integrability, or stationarity estimate is assumed. Setting
`theta=0` gives EMA and `theta=beta` gives the implemented Nesterov blend.
Both compressors have uniform Frobenius contraction contracts; arbitrary
nonnegative deterministic schedules are allowed. The objective's gradient is
the average of the local gradient maps, which share a global Frobenius
Lipschitz bound. Bounded, causal communication roundoff and a numerical
direction with uniform norm/pairing certificates are included. Momentum,
target blending and server arithmetic are exact; the note's optional
momentum/target-roundoff extension is analytical. This is a single matrix block
under ordinary smoothness, not the entire neural-network training pipeline.

For fixed momentum and fixed learning-rate peak, the analytical result gives
an O(1/T) transient plus the displayed noise, movement and numerical residuals.
With exact arithmetic, a fixed positive oracle-pairing gain, beta proportional
to T^(-1/2), and peak radius proportional to T^(-3/4), it gives O(T^(-1/4))
weighted nuclear first-moment stationarity. This is not a squared-gradient or
last-iterate theorem, and does not change the existing stochastic exponent.

### Multiple matrix blocks with the earlier smoothness assumptions

[Experiments/BlockTrajectory.lean](EF21Muon/Experiments/BlockTrajectory.lean)
extends the fixed-momentum trajectory to simultaneous, differently sized matrix
blocks. `Problem.ofDistributed` directly reuses the earlier ordinary-smooth
problem's objective, derivative, global lower bound, mean-gradient identity,
and arbitrary-pair spectral/nuclear smoothness assumptions. The local constants
may vary across workers; `workerL` takes their maximum in each block.

`local_eq` derives the absence of cross-block population-gradient dependence
from the written smoothness assumption. `advance` nevertheless uses the full
shared broadcast model to query every worker; `advance_block` and `run_block`
prove equality with the block trajectories used by the existing tracking
analysis. There is one probability space and one filtration for all blocks.
Different blocks' noise may be correlated. Conditional centering and vanishing
worker cross moments are imposed only within each block.

`tracking` supplies the closed weighted estimator-error budget for each block.
`objective_step` sums layerwise descent before integrating and telescoping the
global objective. Consequently `expected_stationarity_sum` charges
`f(X0)-fmin` once, rather than once per block. It has the form

```text
E sum_(t<T) sum_i eta_it*gain_i*nuclearNorm(grad_i f(X_t))
 <= f(X0)-fmin + sum_i [(gain_i+r_i)*rho_i*trackingBudget_i
                       + sum_(t<T) eta_it*delta_it
                       + (L_i*r_i^2/2)*sum_(t<T) eta_it^2].
```

`expected_stationarity` divides by the mean-layer weight
`Sbar=(sum_i gain_i*sum_(t<T) eta_it)/p`, assumed positive.
With equal schedules and unit gains, this gives a learning-rate-weighted sum
of all block nuclear-gradient norms, consistent with the paper's normalization.
`fresh_stationarity` combines this bound with adaptation of every server and
broadcast block and every worker's memories.

Momentum, schedules, compressors, communication-error bounds and direction
quality may differ between blocks. Each block supports EMA (`theta_i=0`) or
Nesterov (`theta_i=beta_i`), either downlink, and arbitrary nonnegative bounded
deterministic schedules, including constant/cosine schedules. Momentum,
target blending and server arithmetic remain exact. Tracking, objective drift,
integrability and the stationarity bound are derived from the updates.
The complete constants are in
[the multiple-block formulas](../research/experiments/FIXED_MOMENTUM.md#multiple-blocks-under-the-earlier-paper-assumptions).

This extension retains the earlier restrictive smoothness assumption; it does
not establish a theorem for general interacting neural-network layers. It
concerns ordinary smoothness and fixed-radius experimental steps, and leaves
the separate generalized-smooth paper theorems unchanged.

## Further work

1. Package the general-distribution theorem as stochastic
   `OracleTemplate.Convergence` exports.
2. Investigate whether the remaining higher-order compression terms can also
   benefit from worker averaging, and compare with published bounds under
   matching assumptions, costs, and stationarity measures. The leading Nesterov
   variance coefficient and comparison with the old optimized budget are now
   proved above.

Keep a bound on the expected gradient norm distinct from a bound on its expected
squared norm. The fixed-radius stochastic trajectory and scalar closure are
now composed above; applying the bound to training still requires valid uniform
smoothness, contraction and numerical-direction certificates.


## Article corrections and proof correspondence

The review uses the shared sources of the full AAAI article, not older drafts:
[ef21_musign_reduction.tex](../aaai_article/ef21_musign_reduction.tex),
[signmuon_body.tex](../aaai_article/signmuon_body.tex), and
[ef21_signmuon_divergence.tex](../aaai_article/ef21_signmuon_divergence.tex).

| Issue found | Correction in the article | Lean evidence / scope |
| --- | --- | --- |
| Full-support inputs were claimed to have `alpha = Theta(1)` | Require a uniform effective-support bound; distinguish measurements from guarantees | `denseVector_pos`, `denseVector_contraction_tendsto`, `effective_support_contraction` |
| The Gaussian limit lacked a mode of convergence and centering | State centered iid entries, fixed positive variance, dimension tending to infinity, almost surely | `realizedContraction_gaussian_ae` |
| The norm rate was described as implying the expected squared-norm rate | Remove that implication and give a rare-event counterexample | `rareLaw` is a probability measure with `E|Z|=1/a`, `E[Z^2]=1`; first moment tends to zero, second does not |
| A rate constant within a run was treated as covering practical fixed/cosine schedules | State both horizon-dependent learning rate and momentum; qualify experimental coverage | The formalized controls explicitly vary with `T`; experimental schedules are not covered |
| Stochastic assumptions omitted sampling/worker independence used in the analysis | State fresh sampling and conditional worker independence | `Problem.worker_independence` and product path measures |
| Generalized-smooth corollary omitted lower bounds, noise assumptions, initialization, and positivity | Add Assumptions 1 and 3, zero initialization, and fixed positive layer coefficients | `matrix_generalized_paper_rate` proves the corrected rate in the paper's round order; `generalized_exponent_conditions` separately checks the relaxed upstream step restriction |
| Smooth tuning referred to a single-layer optimized rule without an explicit joint choice | State a sufficient common momentum and positive layer steps using the proved `Z_i` envelope | `weighted_smooth_rate`, `layer_horizon_cost`, `matrix_weighted_horizon_rate_isBigO` |
| Proposition 2 described a round order different from its pseudocode and called the whole first round vacuous | Rotate gradient/uplink and step/downlink phases; only the initial step/downlink is vacuous | `exact_instance`, `models_exact`, `observable_sum`, `average_transfer` |
| Nesterov heuristic treated the post-update server model as an unbiased query | Replace the heuristic by a full proof at the actual broadcast query, tracking the filtered target and momentum increments | `Nesterov.smooth_paper_rate`, `paper_horizon_rate_isBigO`, `matrix_generalized_paper_rate`; theorem `thm:nesterov_conv` in the shared article appendix |
| The earlier Lean existence proof used cosine potentials instead of the article's ramps | Replace the formalized field with explicit plateau ramps and give their formula in the appendix; retain an explicit scalar-clipping extension | `plateauDerivative_plus`, `plateauDerivative_minus`, `plateauDerivative_mean_zero`, `plateau_realization`, `normalized_objective`, `divergence` |

The original smooth exports use `sign(0)=+1` and the partial polar selection with
zero step at zero. The [random-tie extensions](RANDOM_TIES.md) add the paper's
fresh independent binary signs to the complete stochastic trajectories and
their paper-order convergence bounds, with the same constants.


### Generalized-smooth step conditions checked separately

`Distributed/GeneralizedSteps.lean` defines the radius as
`eta/(sqrt(S)*sqrt(sqrt(S)))` and proves its square is
`eta^2/(S*sqrt(S))`. With `beta=1/sqrt(S)` and `q=sqrt(1-alphaD)`, it checks
that the three sufficient conditions in the corrected Corollary 2 bound the
three exponent terms in the upstream proof by `1/3` each. The second condition
relaxes as `sqrt(S)` grows; the extra `S` in the denominator of the printed
upstream sufficient bound is not needed for that particular inequality.
The article now uses the explicit third bound
`24*eta_i^2*sqrt(r_i)*(L1_i_max)^2 <= 1`, in place of an unspecified `O(1)`.
This module certifies step algebra only. The stochastic tracking and
stationarity rate are proved independently in the modules described below.

### Generalized-smooth geometry

`Distributed/GeneralizedGeometry.lean` uses exactly the article's arbitrary-pair
condition, with the coefficient evaluated at the first argument:

```text
nuclearNorm(grad_i f(x) - grad_i f(y))
  <= (L0_i + L1_i*nuclearNorm(grad_i f(x))) * spectralNorm(x_i-y_i).
```

`generalized_smooth_upper` derives the simultaneous layer upper bound with
these coefficients frozen at `x`; `generalized_gradient_continuous` also
establishes gradient continuity from this condition.
`generalized_radius_descent` then proves,
for `y_i=x_i-eta_i*polar(G_i)` and nonnegative radii,

```text
f(y) <= f(x) + sum_i [ -eta_i*nuclearNorm(grad_i f(x))
  + 2*eta_i*nuclearNorm(G_i-grad_i f(x))
  + eta_i^2/2 * (L0_i + L1_i*nuclearNorm(grad_i f(x))) ].
```

`generalized_tracking_descent` bounds the error term by the sum of the
Frobenius momentum and compression errors, with the layer's nuclear/Frobenius
comparison constant `rho_i`.

The gradient self-bound is proved directly by applying this step with
`G=grad f(x)`. For nonnegative weights `w_i` and any positive `M` satisfying
`w_i*L1_i <= M`, `generalized_weighted_gradient_self_bound` gives

```text
sum_i w_i*nuclearNorm(grad_i f(x))
  <= 2*M*(f(x)-fmin) + (sum_i w_i^2*L0_i)/M.
```

This form also works when every `L1_i` is zero: `M` can be any positive upper
bound. These deterministic ingredients feed the stochastic proof below.

### Generalized-smooth stochastic convergence

`GeneralizedProblem` fixes one objective, its local worker objectives, and one
noise law independently of the step controls. It assumes the article's
arbitrary-pair generalized smoothness for the global and local gradients,
local lower bounds, the mean objective/gradient identities, centered fresh
noise with bounded variance, and independence across workers. It assumes no
constant global Lipschitz bound and no bounded-gradient condition.
The nuclear noise-variance assumption in the article supplies the Frobenius
moment assumptions by `nuclear_variance_suffices`.

The exact-downlink update is

```text
x'_i = x_i - gamma_i * polar(mean_j G_ij)
w'_i = x'_i
M'_ij = (1-beta_i)*M_ij + beta_i*reply_j(x')_i
G'_ij = G_ij + CU_i(M'_ij-G_ij).
```

`radius_run_erases` and `radius_pathState_erases` show that the proof's
additional noise EMA changes none of these states. `ofRadiusScheme_round`
identifies the update with the framework round in Proposition 2. Thus
`models_exact` supplies the paper's one-step model index shift and
`observable_sum` accounts for the duplicated initial model.

For one layer, write `r=1-beta`, `q=sqrt(1-alphaU)` and let `Z` be the EMA of
the averaged centered worker noise. The proof tracks

```text
D = ||mean_j M_j - grad f(x) - Z||_F
L = mean_j ||M_j-grad f_j(x)||_F
H = mean_j ||M_j-G_j||_F.
```

`RadiusTracking.lean` proves directly from the stochastic updates

```text
E D' <= r*D + r*||grad f(x)-grad f(x')||_F
E L' <= r*L + r*mean_j ||grad f_j(x)-grad f_j(x')||_F + beta*sigma
E H' <= q*H + q*beta*L
        + q*beta*mean_j ||grad f_j(x)-grad f_j(x')||_F + q*beta*sigma
E ||Z'||_F^2 <= r*||Z||_F^2 + beta^2*sigma^2/N.
```

With weights `aD=2*gamma*rho/beta`, `aH=2*gamma*rho/(1-q)`,
`aL=q*aH`, and `aZ=gamma*rho/(beta*sqrt(beta))`, the Lyapunov function is

```text
V = f(x) - mean_j fmin_j + sum_i [aD_i*D_i + aL_i*L_i
                                  + aH_i*H_i + aZ_i*||Z_i||_F^2].
```

The squared noise EMA balances the first-moment direction error using
`2*||Z|| <= ||Z||^2/sqrt(beta)+sqrt(beta)`. The local lower bounds are used
through the proved gradient self-bounds. `radius_drift` establishes

```text
E V' + sum_i gamma_i*nuclearNorm(grad_i f(x)) <= (1+kappa)*V + b,
```

with nonnegative, explicitly defined `kappa` and `b`. The product-measure
induction proves integrability together with the finite-time bound; it does
not assume trajectory integrability or any stationarity estimate as a premise.

`generalizedHorizonControls_paper` proves that the controls are exactly
`S=T+2`, `beta_i=1/sqrt(S)`, `gamma_i=eta_i/S^(3/4)`. For fixed positive
`eta_i`, the checked envelopes give `kappa <= K/S`, `b <= B/S`, and
`V(initial) <= I`, where `K`, `B`, and `I` are independent of `T`.
`generalized_total_cost_for` then bounds the total gradient cost by
`exp(K)*(I+B)` for up to `S` rounds. In particular,

```text
mean_{0<=t<=T} E sum_i (eta_i/etaBar)*nuclearNorm(grad_i f(x_t))
  <= 2*exp(K)*(I+B)/(etaBar*(T+2)^(1/4)).
```

Here the iterates are the phase-rotated framework iterates. The minimum over
their expectations is bounded by this average. `GeneralizedPaper.lean`
defines the paper's gradient/uplink then step/downlink round and identifies
it with the phases in Proposition 2 through `radiusPaperNext_eq_phases`.
`radiusPaper_pathState` proves its relation to the framework on the same full
sample path. `paperRadiusMinimum_le` then transfers the minimum bound directly,
including the initial duplicate and the final paper iterate.
`integral_weightedNuclear` verifies the
identity between the expectation of the sum and the article's sum of weighted
expectations. The minimum is taken after expectation, not inside it.
`generalized_paper_iterate_integrable` proves integrability at every paper
iterate, including the last one, and `generalized_paper_expectation_eq` gives
the article's sum-of-expectations identity for these very iterates.

`matrix_generalized_paper_rate` instantiates the scaled-sign uplink,
identity downlink, and the exact weights `eta_i/mean_l eta_l`, and proves
`=O[atTop] (fun T => (T:Real)^(-1/4))`. Equal coefficients give the unweighted
Theorem 5(ii) through `matrix_generalized_paper_unweighted_rate`.

This is an independent Lyapunov proof of the article's rate, with different
constants from the upstream proof. Its exponential constant permits any fixed
positive layer coefficients, so it covers the sufficient restrictions in
Corollary 2 without using them as additional assumptions. The upstream
step-condition algebra remains separately checked. Neither this result nor
the article asserts this generalized rate for the compressed downlink or an
approximate polar oracle. The separate Nesterov proof is described below.

### Nesterov stochastic convergence

The complete development is in [EF21Muon/Nesterov/](EF21Muon/Nesterov/).
It verifies the paper's actual filter, with `a=1-beta`:

```text
M' = a*M + beta*reply
V' = a*M' + beta*reply
G' = G + CU(V'-G).
```

The state carries the previous momentum buffer so that
`V=(2-beta)*M-a*Mprev` and
`V'-V=(2-beta)*(M'-M)-a*(M-Mprev)` are exact identities.
`Tracking.lean` and `FirstTracking.lean` establish the squared and first-moment
recurrences for this target, including the compression and increment errors.
All noise laws may have unbounded support and finite variance; workers are
independent conditional on the query, and layers may have correlated noise.

`SmoothEnergy.lean` constructs a nonnegative Lyapunov function from the local
buffer error, target-compression error, buffer increment and broadcast error.
`SmoothConvergence.lean` proves its expected drift, trajectory integrability and
the full finite-horizon bound. `SmoothRates.lean` proves the envelope and
explicit `beta=1/sqrt(T)`, `gamma=g*beta` tuning. `SmoothHorizon.lean` supplies
the concrete scaled-sign schemes for either downlink and a fixed-problem Big-O
statement. The constants and Lyapunov function are exactly those stated in
the [shared article proof](../aaai_article/ef21_nesterov_convergence.tex).

`SmoothPaper.lean` defines the paper's gradient/uplink then step/downlink
update. `sharpPaper_pathState` proves its pathwise correspondence with the
phase-rotated process. `sharpPaperSum_eq` proves the exact stationarity-sum
identity, and `smooth_paper_average` proves integrability and the transferred
bound. In particular, `smooth_paper_rate` checks the explicit article bound

```text
paper average of sum_i E nuclearNorm(grad_i f)^2
  <= (4*rateInitial/g + 8*noiseTotalScale)/sqrt(T).
```

`paperSmoothAverage_eq` identifies the formal cost with this gradient average.
`paper_horizon_rate_isBigO` fixes the objective, initial point, oracle and
geometry while retuning only momentum and the common sharp step.

For generalized smoothness, `RadiusEnergy.lean` additionally tracks the
first-moment buffer increment and a squared noise EMA. The local-error weight
is `gamma*(2*rho+12*rho*q/(1-q))`, with `q=sqrt(1-alphaU)`; this differs from
the EMA weight. `GeneralizedDrift.lean` proves the full expected drift and
finite-path bound. `RadiusEnvelopes.lean` and `GeneralizedTuning.lean` prove
`kappa <= K/(T+2)`, `bias <= B/(T+2)` and a horizon-independent initial
potential for any fixed positive layer radii. The proof uses the already
checked local-objective gradient self-bounds, with no bounded-gradient premise.

`GeneralizedRates.lean` proves the concrete `O(T^(-1/4))` result.
`GeneralizedPaper.lean` transfers it to the exact paper updates, including
integrability at every iterate and the sum-of-weighted-expectations identity.
The final theorems are `matrix_generalized_paper_rate` and
`matrix_generalized_paper_unweighted_rate`; the minimum is taken after
expectation, over paper iterates `0,...,T`.

All modules are imported by the library root and included in the universal
checker and the audit of every declaration, including private declarations.
No `sorry`, extra axiom or native computation trust shortcut is used.
Both momentum variants therefore have complete stochastic proofs for the
rates and tuning now stated in the papers. The smooth Nesterov theorem
currently uses a common sharp step; unequal-step smooth EMA results remain
separate. Experimental fixed momentum and deterministic learning-rate schedules
are covered in the experimental development. [Random ties](RANDOM_TIES.md) are
covered by the trajectory extensions. A finite Newton-Schulz implementation is
outside these formal statements. Generalized-smooth convergence requires the
exact downlink.
