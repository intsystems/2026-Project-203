# Lean formalization of SignMuon and the EF21 methods

This Lean 4 library accompanies **SignMuon, MuonSign, and the Role of Error
Feedback**. It formalizes divergence counterexamples for SignMuon, MuonUSign,
MuonSign, and EF21-SignMuon, scaled-sign compression, and convergence results
for EF21-MuonUSign and EF21-MuonSign.

The sources form the `EF21Muon` Lean library, including both EMA and Nesterov
momentum. This folder contains the formal proofs and their checker; the paper's
experimental code is documented separately in the [repository README](../README.md).

The paper sources are in [aaai_article/](../aaai_article/). Detailed statements,
assumptions, constants, and intermediate results are in [RESULTS.md](RESULTS.md).

## Role of LongTails

We use [LongTails](https://github.com/k3nfalt/longtails), by Tyurin et al., as a
pinned dependency alongside Mathlib. Its role in this library is:

| Part of the formalization | Use of LongTails |
| --- | --- |
| Matrix geometry | Reuses the matrix space, exact polar oracle, spectral/nuclear norms, and supporting geometry and smoothness lemmas |
| Deterministic EMA baseline | Uses the oracle protocol and exports `OracleTemplate.Convergence` theorems with explicit gradient-query budgets |
| Paper's stochastic EMA and Nesterov results | Uses the matrix geometry; the probability, tracking, convergence, and paper-correspondence proofs are developed here |
| Stochastic accuracy certificates | Uses `execute` and `executeWithTrace` to certify the expected trajectory score and the exact batch-query count for EMA and Nesterov |
| Optimized smooth bounds | Retains three different momentum powers in the noise term, proves the best constant momentum and smallest certified horizon at fixed safe step scale, and compares with the previous budgets |
| Experimental schedule tools | Full adapted N-worker, multiple-block fixed-radius trajectory under the paper's layerwise smoothness assumptions; closed tracking and stationarity bounds, EMA/Nesterov noise filtering, both downlinks, and approximate-oracle quality |
| Statement audit | Uses the pinned Comparator to compare 45 frozen statements, enforce its axiom allowlist, and replay exported proofs in Lean's kernel |

Run the complete audit with `python scripts/check.py --comparator`.
The local Windows run on 2026-10-23 passed all 45 targets; see the
[Comparator audit record](comparator/VERIFICATION.json).
The [frozen reference](comparator/Reference.lean.in) is kept separately from the
[proved declarations](EF21Muon/ComparatorTargets.lean); checking does not
regenerate it. The stochastic certificates are direct execution theorems,
not `OracleTemplate.Convergence` exports against an independent stochastic
`Setup`. This is an audit of trusted sources, without an OS sandbox for hostile
proof submissions. See [scope and tuning](RESULTS.md#longtails-audit-and-accuracy-tuning).

The sharper [polynomial bound](EF21Muon/LongTails/Profiles.lean) separates the
noise coefficients multiplying `beta`, `beta^2`, and `beta^3`. Its
[optimal tuning](EF21Muon/LongTails/Polynomial.lean) and
[execution certificates](EF21Muon/Theorems/Optimized.lean) apply to EMA and
Nesterov; a separate theorem covers Nesterov's actual paper round order.
The optimized budget is proved no larger than the previous certified budget.
Optimality is within this scalar bound for constant momentum and fixed safe
step scale. Positive leading noise still forces an `epsilon^(-4)` term in
this certificate; no new general stochastic exponent is claimed. See
[the coefficients, explicit budget, and limitations](RESULTS.md#retaining-the-momentum-powers).

The new [Nesterov averaged profile](EF21Muon/LongTails/AveragedNesterov.lean)
recovers `s1_new = s1_old/N` under the same assumptions, updates, score and step
restriction. All five coefficients improve or stay equal, and its optimized
horizon is proved no larger than the old optimized horizon. The
[worker-averaged certificates](EF21Muon/Theorems/WorkerAveraged.lean) cover both
smooth downlink modes and the paper round order. Higher-order compression
terms remain, so this does not establish linear speedup in every regime or
change the fixed-worker stochastic exponent. See
[the proof and exact coefficients](RESULTS.md#recovering-worker-averaging-for-nesterov).

## Correspondence with the paper

Paper labels below are stable identifiers from the LaTeX sources. Lean names
are relative to the `EF21Muon` namespace; follow each link to its source file.

The [paper–Lean semantic audit](PAPER_AUDIT.md) records the current coverage and
remaining differences. The linear reduction's condition on zero entries is now
in the paper, and [random ties](RANDOM_TIES.md) are formalized throughout the
stochastic convergence trajectories.

| Paper result | Lean declarations | Coverage |
| --- | --- | --- |
| Lemma 5, scaled-sign contractivity, `lem:signcontr` | [`Compression.scaledSign_error_identity`, `scaledSign_contractivity`, `scaledSign_realized_identity`, `one_sparse_realizedContraction`](EF21Muon/Compression/ScaledSign.lean) | Exact error identity, realized contraction, sharpness at nonzero one-sparse inputs, and almost-sure centered Gaussian limit; see [Gaussian.lean](EF21Muon/Compression/Gaussian.lean) |
| Linear-objective reduction, `prop:reduction` | [`Divergence.linear_objective_increment`, `linear_objective_strictAscent`](EF21Muon/Divergence/Linear.lean) | Standard and Nesterov momentum; positive steps give ascent; nonsummable steps give divergence |
| Theorem 1, SignMuon, `th:1` | [`Divergence.SignMuon.pairing`, `ascent`, `divergence`](EF21Muon/Divergence/SignMuon.lean) | Exact 4 by 4 linear counterexample |
| Theorem 2, MuonUSign, `th:2` | [`Divergence.MuonSign.uplink_pairing`, `ascent`, `divergence`](EF21Muon/Divergence/MuonSign.lean) with `.muonUSign` | Exact 5 by 5 linear counterexample |
| Theorem 3, MuonSign, `th:3` | [`Divergence.MuonSign.bidirectional_pairing`, `ascent`, `divergence`](EF21Muon/Divergence/MuonSign.lean) with `.muonSign` | Exact 5 by 5 linear counterexample |
| Theorem 4, EF21-SignMuon, `th:ef_div` | [`Divergence.EF21SignMuon.divergence`](EF21Muon/Divergence/Theorems.lean) | Smooth, bounded-below plateau-ramp counterexample for both momentum variants |
| Lemma 1, scale reduction, `lem:efsm_reduction` | [`Divergence.EF21SignMuon.scale_reduction`](EF21Muon/Divergence/Scaling.lean) | Smoothness, iterates, and objective values under rescaling |
| Lemma 2, harmless cycle, `lem:efsm_twocycles` | [`Divergence.EF21SignMuon.harmless_cycle`, `harmless_average`](EF21Muon/Divergence/HarmlessCycle.lean) | Exact parameterized two-cycle and its average |
| Lemma 3, wrong-sign cycle, `lem:efsm_cycle` | [`Divergence.EF21SignMuon.wrong_sign_cycle`, `wrong_sign_average`](EF21Muon/Divergence/EF21Cycle.lean) | Exact rational cycle and wrong-sign average |
| Lemma 4, realization, `lem:efsm_realize` | [`Divergence.EF21SignMuon.plateau_realization`](EF21Muon/Divergence/Theorems.lean), [`Plateaus.lean`](EF21Muon/Divergence/Plateaus.lean) | The article's plateau widths, periods, zero mean, three corrections, exact iterates, and objective increase |
| Proposition 2, exact instance, `prop:instance` | [`Distributed.Correspondence.exact_instance`, `models_exact`, `aggregate_update`, `ofScheme_round`](EF21Muon/Distributed/Correspondence.lean) | Pathwise federated round rotation, full worker states, server aggregation, and one-step model index shift; arbitrary schedules and fixed compressor outcomes |
| Theorem 5(i), smooth convergence, `thm:conv` | [`Distributed.Problem.smooth_rate`, `matrix_horizon_rate_isBigO`](EF21Muon/Distributed/Rates.lean) | Explicit `O(T^(-1/2))` rate for both methods, general noise, multiple workers/layers, and horizon-dependent tuning; unequal steps are covered below |
| Corollary 1, smooth case, `cor:smooth` | [`Distributed.Problem.average_stationarity`](EF21Muon/Distributed/Bounds.lean), [`matrix_weighted_horizon_rate_isBigO`](EF21Muon/Distributed/WeightedRates.lean) | Weighted finite-time and asymptotic bounds, with the exact layer weights and explicit sufficient tuning now stated in the article |
| Theorem 5(ii) and Corollary 2, `cor:l0l1` | [`Distributed.matrix_generalized_paper_unweighted_rate`, `matrix_generalized_paper_rate`](EF21Muon/Distributed/GeneralizedPaper.lean) | Stochastic `O(T^(-1/4))` nuclear first-moment rate for the paper's round order, scaled-sign uplink and exact downlink, fixed unequal layer weights, and the article's horizon schedule |
| Nesterov convergence, `thm:nesterov_conv`, smooth case | [`Nesterov.smooth_paper_rate`, `paper_horizon_rate_isBigO`](EF21Muon/Nesterov/SmoothPaper.lean) | Full stochastic `O(T^(-1/2))` squared-gradient rate, both downlink modes, common sharp step, and the paper's exact round order |
| Nesterov convergence, generalized case | [`Nesterov.matrix_generalized_paper_rate`, `matrix_generalized_paper_unweighted_rate`](EF21Muon/Nesterov/GeneralizedPaper.lean) | Full stochastic `O(T^(-1/4))` nuclear first-moment rate, exact downlink, fixed unequal layer weights, and the paper's exact round order |
| Worker-averaged Nesterov proof and finite-time bound, `prop:polynomial_rate` | [`averaged_nesterov_paper_polynomial_bound`](EF21Muon/LongTails/AveragedNesterov.lean), [EMA profile](EF21Muon/LongTails/Profiles.lean) | Leading variance divided by N, with separate higher-order compression terms; EMA uses the same round rotation |
| Accuracy and query budget, `cor:accuracy_budget` | [`Profile.bound_at_balancedBudget`](EF21Muon/LongTails/Polynomial.lean) | Explicit sufficient momentum and horizon for a target squared-gradient accuracy |
| Prescribed-radius convergence, `thm:fixed_radius`, and schedule corollary | [`tracking`, `expected_stationarity_sum`, `expected_stationarity`](EF21Muon/Experiments/BlockTrajectory.lean) | Exact-oracle specialization of the full N-worker multiple-block trajectory bound; constant/cosine conclusions follow by substituting their step sums |

Additional results include a deterministic baseline with separate
[`DeterministicMuonUSign.result`](EF21Muon/Theorems/DeterministicMuonUSign.lean)
and [`DeterministicMuonSign.result`](EF21Muon/Theorems/DeterministicMuonSign.lean)
LongTails exports, plus [finite-sampling stochastic bounds](EF21Muon/Stochastic/Methods.lean).

### Scope and differences

- The convergence theorems use the exact matrix polar oracle, EMA or Nesterov momentum,
  and zero momentum/estimator initialization. Smooth convergence uses the
  nuclear-norm-scaled step; generalized convergence uses a constant radius
  within each run. Both fixed and fresh independent random signs at zero are
  covered; see [the random-tie theorem map](RANDOM_TIES.md).
- The general stochastic result assumes fresh noise each round, independent
  workers, and the paper's arbitrary-pair layerwise spectral/nuclear smoothness.
  The detailed assumptions are in [RESULTS.md](RESULTS.md#general-noise-multiple-workers-and-layers).
- Smooth convergence includes an explicit horizon schedule `beta = 1/sqrt(T)`,
  `gamma_i = g_i/sqrt(T)` and a proved `O(T^(-1/2))` statement for a fixed objective
  and noise law. Positive layer coefficients `g_i` are chosen from a
  momentum-independent envelope of the paper's step restriction. This is the
  explicit sufficient tuning in the corrected Corollary 1.
  The Nesterov theorem uses the same exponents with a common positive `g`,
  chosen from its own proved envelope; see [SmoothRates.lean](EF21Muon/Nesterov/SmoothRates.lean).
- Generalized convergence fixes the objective and noise law and uses
  `S=T+2`, `beta=1/sqrt(S)`, and radius `eta_i/S^(3/4)`. It proves both an
  average bound and the minimum over iterate expectations. The proof uses
  the article's local objective lower bounds and arbitrary-pair `(L0,L1)`
  smoothness, without a bounded-gradient assumption. Its Lyapunov proof has
  different constants from the cited upstream proof; the article's sufficient
  restrictions are covered. See [the proof details](RESULTS.md#generalized-smooth-stochastic-convergence).
- The federated correspondence is pathwise. It proves both model index shifts
  and the duplicated initial term in stationarity sums, and connects its round
  definition to the stochastic scheme used by the convergence proofs.
- The Nesterov development tracks the actual filtered target
  `V=(1-beta)*M+beta*gradient`, its compression residual, and momentum increments.
  It proves integrability and convergence for the actual finite-product stochastic
  paths, rather than assuming a tracking or stationarity estimate.
  The paper proof and sufficient constants are in the shared
  [Nesterov appendix](../aaai_article/ef21_nesterov_convergence.tex).
- [Experiments/Schedules.lean](EF21Muon/Experiments/Schedules.lean)
  provides conditional bounds for arbitrary learning-rate schedules, approximate
  directions, and measured downlink tracking. It keeps their error terms explicit;
  see [the experiment notes](../research/experiments/README.md).
  [FixedMomentum.lean](EF21Muon/Experiments/FixedMomentum.lean) additionally
  proves the weighted error closure from primitive tracking recurrences and
  the exact EMA/Nesterov noise-filter bound.
  [Trajectory.lean](EF21Muon/Experiments/Trajectory.lean) derives these
  recurrences and integrability from the actual N-worker updates.
  [FreshTrajectory.lean](EF21Muon/Experiments/FreshTrajectory.lean), theorem
  `Trajectory.fresh_stationarity`, combines conditional mean-zero noise,
  causal updates and the closed expected stationarity bound. It covers one
  smooth matrix block with fixed momentum, deterministic constant/cosine or
  other nonnegative schedules, both downlinks, bounded communication roundoff,
  and a uniformly certified approximate direction. Momentum, target blending
  and server arithmetic are exact in this theorem.
  [BlockTrajectory.lean](EF21Muon/Experiments/BlockTrajectory.lean) extends this
  to simultaneous matrix blocks under the earlier arbitrary-pair layerwise
  spectral/nuclear smoothness assumptions. `BlockTrajectory.fresh_stationarity`
  uses one filtration for all blocks, permits correlated noise between blocks,
  and allows separate schedules, momentum and compression parameters per block.
  Its bound charges the global objective gap once and uses the paper's mean-layer
  normalization. `Problem.ofDistributed` reuses the earlier problem assumptions
  directly. Block independence follows from these strong smoothness assumptions;
  this extension does not cover general interactions between network layers. See
  [the formulas and assumptions](../research/experiments/FIXED_MOMENTUM.md).
- EF21-SignMuon's realization uses the article's plateau-ramp field and three
  localized corrections. Its divergence theorem makes this objective globally
  nonnegative by smooth scalar clipping, an alternative to the coordinate
  modification in the article's boundedness remark. Smoothness, exact iterates,
  and boundedness below are proved for the scalar modification; the specific
  coordinate modification in the article is not formalized.

### Corrections found while formalizing

The article sources have been updated to match the proved statements. The
review found that full support does not guarantee constant-order contraction,
that first-moment rates do not imply second-moment rates, and that constancy
within a run does not certify the experiments' horizon-independent or annealed
schedules. Explicit counterexamples are kernel checked in
[Density.lean](EF21Muon/Compression/Density.lean) and
[MomentCounterexample.lean](EF21Muon/Probability/MomentCounterexample.lean).
See [the detailed review](RESULTS.md#article-corrections-and-proof-correspondence)
for the assumptions, corrected tuning, and scope limitations.

## Build and verify

Install [Elan](https://github.com/leanprover/elan#installation), Git, and Python 3,
and make sure `lake` is on PATH. Elan selects and installs the pinned Lean version
from [lean-toolchain](lean-toolchain). An internet connection is needed for the
initial toolchain and dependency downloads.

From a fresh clone, run the same checker on Linux, macOS, or Windows:

```sh
git clone https://github.com/intsystems/2026-Project-203.git
cd 2026-Project-203/lean_proofs
python scripts/check.py --fetch-cache
```

Use `python3` if that is the name of your Python 3 interpreter. The checker finds
the project from its own location, so it can also be run from another directory:

```sh
python lean_proofs/scripts/check.py
```

To include the LongTails Comparator statement and kernel audit:

```sh
python lean_proofs/scripts/check.py --comparator
```

The checker always runs `lake build` and `lake env lean Audit.lean`, stops on
failure, and rejects changes to the pinned dependency files. `--fetch-cache`
first runs `lake exe cache get` to download Mathlib's compiled cache. It is
optional for subsequent checks. The checker uses Python's standard library and
the installed Lake command; it inherits your normal toolchain and cache settings.

Keep [lake-manifest.json](lake-manifest.json): it locks all dependency revisions.
The reproduction commands use this lock; `lake update` is for intentional
dependency changes.

The underlying verification commands can also be run directly from this folder:

```sh
lake build
lake env lean Audit.lean
```

### Axiom audit and continuous integration

[Audit.lean](Audit.lean) checks every declaration defined in the imported
`EF21Muon` modules, including private declarations and definitions outside the
`EF21Muon` namespace. It checks transitive axiom dependencies and exits with an
error if it finds an axiom other than:

```text
propext
Classical.choice
Quot.sound
```

This rejects `sorry`/`admit` placeholders (`sorryAx`), extra axiom declarations,
and dependencies on `Lean.ofReduceBool` from `native_decide`. The audit also
prints individual axiom reports for selected results. The optional `--comparator`
step additionally compares frozen statements and replays their exported proofs.
The reference challenge deliberately contains proof holes; it is excluded from
the library and this axiom audit. The proved solutions must pass both audits.

The [GitHub Actions workflow](../.github/workflows/lean.yml) runs the same checker
on Linux, macOS, and Windows for pushes and pull requests affecting the
formalization, plus Comparator on Linux. Run this checker locally before
preparing a paper release.

### Pinned dependencies

| Component | Version or revision |
| --- | --- |
| Lean | `leanprover/lean4:v4.35.0-rc2` |
| LongTails | [`f9079117032f800d5b836550291edf33c706ae19`](https://github.com/k3nfalt/longtails/tree/f9079117032f800d5b836550291edf33c706ae19) |
| Comparator | `32bd61da1d68fbaa310234964e9b820b03a0f82f` |
| lean4export | `6cea97789dc088ea47fcea15692db85685aedac5` |
| Mathlib and other transitive dependencies | Exact revisions in [lake-manifest.json](lake-manifest.json) |

## Library layout

| Location | Contents |
| --- | --- |
| [EF21Muon.lean](EF21Muon.lean) | Entry point importing the library |
| [Compression/](EF21Muon/Compression/) | Scaled-sign identity and contractivity |
| [Theorems/](EF21Muon/Theorems/) | Deterministic LongTails convergence exports and stochastic accuracy/query certificates |
| [LongTails/](EF21Muon/LongTails/) | Stochastic execution bridge, polynomial profiles, optimal constant tuning, and comparisons with previous budgets |
| [comparator/](comparator/) | Frozen statement reference, target catalog, and audit record |
| [Stochastic/](EF21Muon/Stochastic/) | Finite-sampling stochastic analysis |
| [Probability/](EF21Muon/Probability/), [Distributed/](EF21Muon/Distributed/) | General measures, multiple workers/layers, constants, and tuning |
| [Nesterov/](EF21Muon/Nesterov/) | Stochastic Nesterov tracking, smooth and generalized convergence, and correspondence with the paper |
| [Experiments/](EF21Muon/Experiments/) | Full adapted N-worker fixed-radius trajectory, closed tracking/stationarity budgets, arbitrary deterministic schedules, noise filtering, and numerical oracle quality |
| [Divergence/](EF21Muon/Divergence/) | Exact counterexamples and the smooth EF21-SignMuon witness |
| [Audit.lean](Audit.lean) | Enforced axiom allowlist and selected theorem reports |
| [RESULTS.md](RESULTS.md) | Detailed mathematical statements and limitations |
| [scripts/check.py](scripts/check.py) | Portable build, axiom audit, and dependency-pin check |
| [scripts/compare_bounds.py](scripts/compare_bounds.py) | Optional bound calculator |

The optional calculator requires Python 3 and uses only its standard library:

```sh
python scripts/compare_bounds.py --rows 32 --cols 32 --beta 0.1
```

## License

The formalization is distributed under the repository's [MIT license](LICENSE).
Dependencies retain their respective licenses.
