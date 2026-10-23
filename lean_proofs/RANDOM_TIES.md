# Random signs at zero

The paper draws a fresh fair sign at each zero entry. The Lean extension uses
that convention in the complete EMA and Nesterov trajectories. No change to the
convergence constants is required.

## Probability model

[Compression/RandomTies.lean](EF21Muon/Compression/RandomTies.lean) defines the
binary sign and the scaled compressor

\[
C_b(Y)=\frac{\|Y\|_1}{d}\operatorname{sign}_b(Y),\qquad d=mn.
\]

For every bit assignment, including zero inputs, Lean proves

\[
\|C_b(Y)-Y\|_F^2
=\|Y\|_F^2-\frac{\|Y\|_1^2}{d}
=\|C_+(Y)-Y\|_F^2.
\]

`coinLaw` supplies independent fair bits for each worker, layer, entry, and
link in a round. `pathMeasure (coinLaw.prod μ)` supplies fresh round seeds,
independent of the gradient oracle's seed. `streamLaw` gives the corresponding
infinite stream. Fairness, independence, joint measurability, and contraction
are proved.

## Complete trajectory theorems

Names in this table are relative to `EF21Muon.RandomTies`.

| Setting | Export | Meaning |
| --- | --- | --- |
| Ordinary smoothness, EMA | [ema_paper_average](EF21Muon/RandomTies/Paper.lean) | Expected squared-gradient average for the literal paper update order, including random ties on both links |
| Ordinary smoothness, Nesterov | [nest_paper_average](EF21Muon/RandomTies/NesterovPaper.lean) | Same update order; retains the recovered leading worker factor `1/N` |
| Generalized smoothness, EMA | [ema_generalized_paper_minimum_rate](EF21Muon/RandomTies/GeneralizedPaper.lean) | Paper-order minimum of expected weighted nuclear norms, bounded by a horizon-independent constant times `(T+2)^(-1/4)` |
| Generalized smoothness, Nesterov | [nest_generalized_paper_minimum_rate](EF21Muon/RandomTies/GeneralizedPaper.lean) | Same first-moment rate with the existing Nesterov constants |
| Fixed momentum and deterministic experimental schedules | [RandomizedBlocks.fresh_stationarity](EF21Muon/Experiments/RandomizedBlocks.lean) | Full simultaneous N-worker, multilayer updates, including adaptedness, integrability, derived tracking, and weighted stationarity |

The generalized results use the exact downlink, as in the paper. For the
experimental theorem, [Experimental.lean](EF21Muon/RandomTies/Experimental.lean)
instantiates both random scaled-sign links with the independent stream.
`experimental_trackingBudget_eq` proves that every explicit transient and
residual coefficient equals its original fixed-sign counterpart.

The smooth paper-order estimates have the existing form

\[
\frac{4}{\bar\gamma T}\mathbb E\sum_{t<T}\mathrm{cost}(s_t)
\le \frac{4\Phi(s_0)}{\bar\gamma T}
+\frac{8\,\mathrm{noiseTotal}}{\bar\gamma},
\]

with the appropriate EMA or Nesterov potential and noise term. The generalized
proofs retain the same horizon-independent envelopes `K`, `I`, and `B`, giving

\[
\min_{0\le t\le T}\sum_i\frac{\eta_i}{\bar\eta}
  \mathbb E\|\nabla_i f(X_t)\|_*
\le \frac{2e^K(I+B)}{\bar\eta\,(T+2)^{1/4}}.
\]

These theorems derive the estimates from the random updates. Equality of the
one-step error norms transfers the Lyapunov drift; it does not assert equality
of stochastic trajectories with different tie outcomes.

## Linear reduction and divergence

The shared paper proposition now requires nonzero entries at every sign
argument evaluated at its constant gradient. This condition is necessary:
the row gradient `(1,0)` gives SignMuon direction `(1,±1)`.

[Divergence/RandomTies.lean](EF21Muon/Divergence/RandomTies.lean) proves the
corrected constant-direction claim and checks the condition for all three
linear witnesses. It also proves that every EF21 cycle residual has nonzero
entries, including the preamble. All these divergence trajectories agree with
the earlier ones for every sequence of tie bits.

## Verification and remaining compositions

All modules are imported by the library root and checked by
`python scripts/check.py --comparator`. The transitive audit includes their
private declarations and rejects unapproved axioms and proof holes.

The 45 frozen Comparator targets still concern the earlier exports. New
random-tie theorems are kernel checked and axiom audited; they have not been
added to that frozen target set. Polynomial tuning and query-budget execution
certificates have not been repackaged for random-tie executions. The exact
constant/cosine schedule sums and the paper's exact-polar substitutions remain
separate specializations, as recorded in [PAPER_AUDIT.md](PAPER_AUDIT.md).
