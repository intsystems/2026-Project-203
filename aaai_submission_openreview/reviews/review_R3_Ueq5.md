# Human review 3 — Ueq5

Note ID: `xHT5WIiVi7`. Source: `signmuon_openreview_page.json`.

## Brief summary

The reviewer finds the placement study useful (6, confidence 2), but identifies a ranking-changing table error, an experimental/theorem mismatch, separable smoothness, and an invalid first-to-second-moment inference. The “N/A” feedback field is not treated as a concern.

## Every concern / request

1. **R3.2 — Reviewer identifies the reversed MuonSign versus SignSGD ranking.** Correct the main table and every affected comparison.

2. **R3.1 — Reviewer asks which learning-rate rule proves the squared-gradient rate.** Do not transfer the sharp-stepsize result to cosine schedules.

3. **R3.3 — Reviewer identifies the separability forced by layer-wise smoothness.** Replace the restriction with an explicit whole-model formulation.

4. **R3.4 — Reviewer rejects squaring an expected first-moment bound.** Remove the unsupported inference and state two distinct results.

5. **R3.5 — Reviewer asks to distinguish ideal EF21 from the implementations.** State the remaining oracle and optimizer qualifications.

## Response priorities

Own the two errors directly. Give the independently reconstructed values first, then separate the sharp second-moment result, new jointly smooth first-moment bound, and remaining experimental gaps. No full-run coverage claim.

## Original review

### title

Useful counterexamples and practical comparisons, but convergence claims and reported results need clarification

### review

This paper compares sign compression before, after, and on both sides of Muon’s orthogonalization. It gives counterexamples showing failure, including for output-level error feedback, and establishes conditional guarantees for gradient-level variants. Experiments on CIFAR-10 and nanoGPT favor signing after orthogonalization, highlighting a gap between worst-case theory and practical performance.

### strengths

- Sections 4.2–4.3 show that combining sign compression with Muon can turn an update into an ascent direction. Theorem 4 goes further by showing that the specific output-level error-feedback method can fail even on a smooth, lower-bounded objective. These results give a concrete explanation of why the placement of compression matters.
- Section 5 compares the different placements on synthetic problems, centralized and federated CIFAR-10, and language modelling. The paper openly reports that methods without a general guarantee can perform better in practice. This is more useful than presenting only the best-performing variant.

### weaknesses

- Table 2 reports MuonSign below SignSGD, at 80.68% versus 81.83%, whereas Table A8 reports 82.94% versus 81.44% under the unit-gain rule. Its caption says these are the five-seed results from Table 2. This may be a fixable reporting issue, but it reverses an explicitly discussed ranking. The authors should identify the correct results and revise the affected comparisons.
- Appendix A.10 requires a gradient-dependent stepsize for the squared-gradient rate and explicitly excludes the tested EF21-MuonSign schedule. Appendix A.1 also acknowledges that the layer-wise smoothness assumption forces separability across layers. These limitations do not invalidate the counterexamples, but they substantially restrict the positive results. The main text should distinguish the analyzed variants from the experimental implementations more clearly.
- The text treats an $O(T^{-1/4})$ bound on the expected gradient norm as corresponding to an $O(T^{-1/2})$ bound on its expected square. This does not follow by squaring the first-moment bound. This appears fixable by narrowing the statement or supplying a separate second-moment argument; it does not by itself invalidate the separately stated sharp-stepsize result.

### justification

I lean toward acceptance mainly because the explicit counterexamples and the distinction between compression targets are useful contributions in their own right. The inconsistent experimental results and the scope of the positive guarantees prevent a stronger recommendation. Reconciling the tables and correcting the rate interpretation would increase my confidence in the paper.

### feedback

N/A

### rating

6

### confidence

2
