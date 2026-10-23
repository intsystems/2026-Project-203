# Paper–Lean correspondence audit

Reviewed on 2026-10-23 against the current working-tree sources. This is a
manual semantic audit, accompanied by the library's executable checks.

## Findings

The main convergence rates, norms, momentum formulas, worker factors, and
initializations agree with the Lean developments. The linear reduction's missing
condition has been added to the shared paper source, and fresh random sign ties
are now included in complete stochastic trajectories. The remaining coverage
differences are described below. No change to the advertised rate exponents was
identified.

### 1. Corrected: the linear reduction's condition on zero entries

The papers draw a fresh independent sign at zero
([AAAI definition](../aaai_article/signmuon_body.tex),
[OPT definition](../opt_article/opt_body.tex)). The previous wording of
Proposition `prop:reduction` claimed a constant direction for every nonzero
constant gradient. [Linear.lean](EF21Muon/Divergence/Linear.lean) proved that
claim for the deterministic convention `Compression.signBit 0 = 1`.

A counterexample to the paper's general wording is the row gradient
`G = (1, 0)`. Its polar factor is `(1, 0)`, and SignMuon's direction is
`(1, epsilon_t)`, where `epsilon_t` is a fresh random sign. The direction is
not constant. MuonUSign similarly produces `(1, epsilon_t)/sqrt(2)`.

The [shared appendix](../aaai_article/signmuon_appendix.tex) now requires every
argument of `sign` in the chosen placement, evaluated at `G`, to have no zero
entries. Only the added sentence is colored in the review versions. In
[Divergence/RandomTies.lean](EF21Muon/Divergence/RandomTies.lean),
`Divergence.RandomTies.constant_direction` proves the corrected statement for any tie
sequence and both momentum variants. The explicit witnesses satisfy its
`NoTies` condition. Their ascent conclusions and exact pairings remain valid:

| Method | Exact pairing checked in Lean |
| --- | --- |
| SignMuon | `-42468/103` |
| MuonUSign | `6 - 82/sqrt(17)` |
| MuonSign | `-76` |

The EF21-SignMuon cycle also has nonzero compressor residual entries, including
the preamble. The zero entries of its rank-one second **target** do not cause
ties: the compressor receives the target minus the previous estimate.
`Divergence.EF21SignMuon.residual_no_zeros` and `positionWith_eq` check this for the
whole EF21 cycle, so its trajectory is unchanged by any tie sequence.

### 2. Closed: random ties in the stochastic trajectories

[Compression/RandomTies.lean](EF21Muon/Compression/RandomTies.lean) constructs
independent fair bits for every round, layer, worker, matrix entry, and link.
The round law is combined with the gradient-sampling law by a product measure;
the infinite stream law is also formalized. Joint measurability, fairness,
independence, and the exact compressor error identity are proved. Random ties
leave the compression error norm unchanged, including at a zero input.

Complete random-compressor trajectories and bounds are exported in:

- [Convergence.lean](EF21Muon/RandomTies/Convergence.lean): ordinary-smooth EMA
  and worker-averaged Nesterov, with either downlink.
- [Paper.lean](EF21Muon/RandomTies/Paper.lean) and
  [NesterovPaper.lean](EF21Muon/RandomTies/NesterovPaper.lean): the literal
  query/uplink/server/downlink order and its expected squared-gradient average.
  The extra initial term and doubled noise coefficients are proved.
- [Generalized.lean](EF21Muon/RandomTies/Generalized.lean),
  [GeneralizedRates.lean](EF21Muon/RandomTies/GeneralizedRates.lean), and
  [GeneralizedPaper.lean](EF21Muon/RandomTies/GeneralizedPaper.lean): generalized
  smoothness with the exact downlink, horizon tuning, and the paper-order
  first-moment minimum bound of order `(T+2)^(-1/4)` for both momentum variants.
- [RandomizedBlocks.lean](EF21Muon/Experiments/RandomizedBlocks.lean): simultaneous
  matrix blocks, fixed momentum, deterministic schedules, conditional fresh
  gradient noise, and random uplink/downlink compression. Adaptedness and
  integrability are proved. [Experimental.lean](EF21Muon/RandomTies/Experimental.lean)
  supplies independent fair scaled-sign ties and proves equality of the explicit
  tracking budgets with the original constants.

These results derive tracking and convergence from the updates. They do not
assume the desired tracking bound. They retain the earlier constants and worker
factors; the printed tuning and schedule specializations in finding 4 still
have the stated composition limits.

### 3. The bounded-below construction differs

The plateau periods, plateau neighborhoods, prescribed gradients, three local
corrections, actual iterates, and increase `49/240` are proved in
`Divergence.EF21SignMuon.plateau_realization` in
[Theorems.lean](EF21Muon/Divergence/Theorems.lean).

For the final bounded-below objective, the paper modifies the coordinate term
`-(7/12) W_22` outside the visited region
([boundedness remark](../aaai_article/ef21_signmuon_divergence.tex)). Lean instead
smoothly clips the **value of the complete objective** below a threshold;
see `normalized_objective` in
[EF21Realization.lean](EF21Muon/Divergence/EF21Realization.lean).
It proves that gradients on the whole trajectory and all objective increments
are preserved. The final theorem proves global nonnegativity, smoothness,
actual algorithmic divergence, and the stated scaling for every `L, eta > 0`.

The existential conclusion of Theorem 4 is therefore supported. The particular
coordinate modification printed in the paper has not itself been formalized.
The README's earlier assertion that the paper explicitly described the scalar
alternative was stale and has been corrected.

### 4. Some paper specializations are still manual compositions

- **Polynomial profiles.** `RandomTies.ema_paper_average` and
  `nest_paper_average` now integrate the literal paper-order average, including
  the extra initial term and doubled noise coefficients. Substituting the
  existing polynomial profiles and accuracy budgets into these random-tie
  averages is still a separate algebraic composition. The earlier deterministic
  Nesterov export `averaged_nesterov_paper_polynomial_bound` already packages
  that composition.
- **Experimental schedules.** The closed tracking and stationarity theorems
  derive the error from the simultaneous random worker updates. The paper's
  `I`, `D`, and `R` coefficients agree with expanding their budgets at exact
  arithmetic, `direction = polar`, `B = rho = sqrt(min(m,n))`, `r = gain = 1`,
  and `delta = 0`. These parameter substitutions are not packaged as a separate
  paper-facing declaration. The constant/cosine sums and asymptotic
  substitutions in `cor:radius_schedules` are also manual; there is no Lean
  declaration of the cosine sum `S = eta_peak*(T+1)/2` in this library.
- **Accuracy/query corollary.** The constants `24` and `8`, balanced momentum,
  ceiling, and squared-accuracy conclusion are proved by
  `Profile.bound_at_balancedBudget`. Nesterov's worker-averaged paper version is
  composed in [WorkerAveraged.lean](EF21Muon/Theorems/WorkerAveraged.lean).
  The LongTails execution certificates still use the earlier fixed-sign
  trajectories. Random-tie execution certificates have not been exported;
  the random-tie convergence bounds themselves are proved.

These are gaps in the exported formal correspondence, rather than discrepancies
found in the displayed algebra. They should be closed before claiming that
every printed corollary is available as one complete Lean theorem.

## Results checked

Labels refer to the current shared LaTeX sources; theorem numbers can vary by
edition. The findings above qualify the coverage in this table.

| Paper result | Lean source and check |
| --- | --- |
| `prop:reduction`, `th:1`–`th:3` | [Linear.lean](EF21Muon/Divergence/Linear.lean), [SignMuon.lean](EF21Muon/Divergence/SignMuon.lean), [MuonSign.lean](EF21Muon/Divergence/MuonSign.lean): positive momentum multiples for both variants, exact polar certificates, exact pairings, arbitrary positive steps, and divergence for divergent cumulative steps. General proposition has finding 1. |
| `lem:efsm_reduction`, `lem:efsm_twocycles`, `lem:efsm_cycle` | [Scaling.lean](EF21Muon/Divergence/Scaling.lean), [HarmlessCycle.lean](EF21Muon/Divergence/HarmlessCycle.lean), [EF21Cycle.lean](EF21Muon/Divergence/EF21Cycle.lean): the scale law, parameterized harmless cycle, preamble, rational wrong-sign cycle, and drift agree. |
| `lem:efsm_realize`, `th:ef_div` | [Theorems.lean](EF21Muon/Divergence/Theorems.lean): realization is proved from an actual smooth objective; prescribed gradients are not a premise of the final theorem. Boundedness uses finding 3. |
| `prop:instance` | [Correspondence.lean](EF21Muon/Distributed/Correspondence.lean): full states, worker estimator averaging, zero initialization, and the exact index shift `X_rot(t+1) = X_paper(t)`. |
| `lem:signcontr` and contraction discussion | [ScaledSign.lean](EF21Muon/Compression/ScaledSign.lean), [Density.lean](EF21Muon/Compression/Density.lean), [Gaussian.lean](EF21Muon/Compression/Gaussian.lean): compatible-sign identity, `alpha = 1/d`, sparse sharpness, and the almost-sure `2/pi` limit for centered iid Gaussian entries of fixed positive variance. Gaussian assumptions concern the input distribution, not optimizer residuals generally. |
| `thm:conv(i)`, `cor:smooth` | [WeightedRates.lean](EF21Muon/Distributed/WeightedRates.lean): same `Z_i`, fixed `g_i`, `beta = T^(-1/2)`, `gamma_i = g_i beta`, and weights `g_i / average(g)`. The rate controls the sum of expected **squared nuclear norms**. Objective and noise law are fixed across horizons. See findings 2 and 4. |
| `thm:conv(ii)`, `cor:l0l1` | [GeneralizedPaper.lean](EF21Muon/Distributed/GeneralizedPaper.lean): exact downlink, `S = T+2`, `beta = S^(-1/2)`, radius `eta_i S^(-3/4)`, same weights, and minimum over expectations of the nuclear **first moment**. Lean allows any fixed positive `eta_i`, covering the paper's more restrictive sufficient choices. Finding 2 applies. |
| `lem:nest_smooth_tracking`, `thm:nesterov_conv(i)` | [AveragedEnergy.lean](EF21Muon/Nesterov/AveragedEnergy.lean), [SmoothRates.lean](EF21Muon/Nesterov/SmoothRates.lean), [SmoothPaper.lean](EF21Muon/Nesterov/SmoothPaper.lean): actual Nesterov target `V = (1-beta)M + beta G`, previous-momentum state, matching Lyapunov weight sums, safe-step envelope, and paper-order squared-gradient rate. Finding 2 applies. |
| `lem:nest_general_tracking`, `thm:nesterov_conv(ii)` | [RadiusEnergy.lean](EF21Muon/Nesterov/RadiusEnergy.lean), [RadiusEnvelopes.lean](EF21Muon/Nesterov/RadiusEnvelopes.lean), [GeneralizedTuning.lean](EF21Muon/Nesterov/GeneralizedTuning.lean), [GeneralizedPaper.lean](EF21Muon/Nesterov/GeneralizedPaper.lean): same noise-EMA separation, first-moment tracking, objective-gap control, horizon-independent envelopes, and minimum of expectations. Finding 2 applies. |
| `prop:polynomial_rate`, `cor:accuracy_budget` | [Profiles.lean](EF21Muon/LongTails/Profiles.lean), [AveragedNesterov.lean](EF21Muon/LongTails/AveragedNesterov.lean), [Polynomial.lean](EF21Muon/LongTails/Polynomial.lean): both coefficient columns, including Nesterov's recovered `1/N`, the double-noise adjustment, and balanced accuracy budget agree. Findings 2 and 4 apply. |
| `lem:radius_tracking`, `thm:fixed_radius`, `cor:radius_schedules` | [Trajectory.lean](EF21Muon/Experiments/Trajectory.lean), [FreshTrajectory.lean](EF21Muon/Experiments/FreshTrajectory.lean), [BlockTrajectory.lean](EF21Muon/Experiments/BlockTrajectory.lean): actual simultaneous updates, common filtration, derived tracking, and one global objective gap. Findings 2 and 4 apply. |

## Assumptions and proof obligations inspected

- Lean's layerwise ordinary and generalized smoothness inequalities quantify
  over arbitrary pairs of complete models, exactly as the paper does. Block
  independence is derived from that assumption; it is not silently imposed as
  an additional premise. These are the strong layerwise assumptions already
  explained in the paper's preliminaries.
- Smooth results require a lower bound for the global objective. Generalized
  results also use lower bounds for the individual worker objectives, as stated
  in Assumption 1 and the convergence appendix.
- Lean bounds gradient noise in Frobenius norm. The paper's nuclear second
  moment bound implies this. Fresh rounds are represented by a finite product
  measure; worker independence is assumed at each query. The fixed-schedule
  development instead uses conditional centering and worker orthogonality with
  a common filtration. Cross-layer independence is not required.
- Joint measurability of the stochastic oracle and measurability of the
  compressors are explicit Lean premises. They are standard probability
  regularity assumptions implicit in the paper's expectations.
- Integrability is proved for the actual trajectories. The final convergence
  and tracking results do not assume the stationarity bound or the tracking
  recurrence they are meant to establish.
- The formal polar oracle is the minimum-Frobenius-norm maximizer over the
  spectral ball, including `polar 0 = 0`. Its support pairing, spectral bound,
  and Frobenius bound are proved. The explicit divergence matrices have exact
  polar certificates, including the rank-one preamble. The paper's broader
  remark that other measurable LMO selections also suffice is not an exported
  general-selection theorem for the smooth/generalized trajectories.
- EMA generalized convergence uses a different Lyapunov proof from the cited
  proof summarized in the paper. Its conclusion covers the paper's tuning;
  kernel verification does not check each line of the cited proof. The new
  Nesterov proofs follow the displayed tracking quantities and coefficient
  envelopes.

## Supporting prose outside this formalization

The numbered results above are the audit's mathematical core. The Mishra and
S-Muon comparisons, the Gluon extension remark, unit-gain initialization
calculations, communication accounting, and empirical interpretations are not
all supplied as Lean theorems. In particular, the Mishra section's pairing
`-42468/103` is checked, but its conditional sign-error decomposition and its
bounded-below extension of the linear example are not composed as a Lean
result. Its cross-reference to the EF21-specific coordinate modification
should be replaced by a construction for the linear example if retained.

## Executable verification

`python scripts/check.py --comparator` completed successfully on this working
tree (exit code 0):

- Library build passed.
- The transitive axiom audit passed for 3,816 library declarations, including
  private declarations. Allowed axioms are `propext`, `Classical.choice`, and
  `Quot.sound`.
- Dependency pins were unchanged.
- All 45 frozen Comparator statements matched, and their exported proofs were
  accepted by Lean's default kernel.

The generated Comparator reference contains statement placeholders with
`sorry`. Those placeholders are not proofs accepted for the library: Comparator
matches the solution statements, checks their dependencies, and replays the
solution exports. The audit of the actual library rejects `sorryAx`.

These checks establish the validity of the Lean declarations. The semantic
comparison with the paper, documented above, is separate from the executable
checks.
