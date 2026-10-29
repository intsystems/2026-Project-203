# AI review — AI_Reviewer

Note ID: `LKOb5GFqWI`. Source: `signmuon_openreview_page.json`.

## Brief summary

Imported review. Scientific summaries and answers have not been prepared.

## Every concern / request

1. **AI.1 — Summary needed for imported review passage 1.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

2. **AI.2 — Summary needed for imported review passage 2.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

3. **AI.3 — Summary needed for imported review passage 3.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

4. **AI.4 — Summary needed for imported review passage 4.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

5. **AI.5 — Summary needed for imported review passage 5.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

6. **AI.6 — Summary needed for imported review passage 6.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

7. **AI.7 — Summary needed for imported review passage 7.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

8. **AI.8 — Summary needed for imported review passage 8.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

9. **AI.9 — Summary needed for imported review passage 9.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

10. **AI.10 — Summary needed for imported review passage 10.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

11. **AI.11 — Summary needed for imported review passage 11.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

12. **AI.12 — Summary needed for imported review passage 12.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

13. **AI.13 — Summary needed for imported review passage 13.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

14. **AI.14 — Summary needed for imported review passage 14.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

15. **AI.15 — Summary needed for imported review passage 15.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

16. **AI.16 — Summary needed for imported review passage 16.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

17. **AI.17 — Summary needed for imported review passage 17.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

18. **AI.18 — Summary needed for imported review passage 18.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

19. **AI.19 — Summary needed for imported review passage 19.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

20. **AI.20 — Summary needed for imported review passage 20.** An agent must write a concrete summary, split distinct concerns, and link all matching review passages.

## Response priorities

Read `AGENT_START.md` and prepare the evidence map before drafting.

## Original review

### title

AI Review

### review

**Title:** SignMuon, MuonSign, and the Role of Error Feedback

**Synopsis of the paper**  
The paper studies where to place one-bit sign compression around Muon’s matrix-valued linear minimization oracle. It gives counterexamples showing that all direct placements can ascend and that error feedback on the oracle output can still diverge, while gradient-side error feedback inherits nonconvex convergence guarantees. Experiments show that the theoretically unsupported sign-after placement performs best among the compressed variants on the tested workloads.

**Summary of Review**  
The principal novel contribution is the systematic negative analysis of sign placement around Muon’s oracle, including explicit counterexamples and a bounded-below divergence construction for direct oracle-output error feedback. The experiments substantiate a theory–practice discrepancy and show substantial gains over plain SignSGD. However, the assumptions stated in the main paper do not support the positive convergence claim in the stochastic, coupled multilayer setting being advertised. The bidirectionally compressed method also exhibits a large discrepancy between the iterate covered by theory and the model that trains successfully.

**Strengths**

- **The negative results sharply isolate a failure caused by composing sign compression with the LMO.** Theorems 1–3 provide explicit low-dimensional ascent instances for every positive step size and momentum, while Theorem 4 extends the failure to a smooth, bounded-below objective for direct oracle-output error feedback. Full-precision Muon descends on the linear instances, making the role of compression concrete.

- **The paper directly investigates rather than conceals the mismatch between worst-case theory and observed optimization.** Figure 1 demonstrates the constructed failures, while Section 5 measures gradient–step alignment on random quadratics and evaluates all placements across centralized vision, distributed vision, and language modeling. This makes the scope of the counterexamples and the unresolved empirical behavior assessable.

- **The failed and successful error-feedback placements are connected by a useful mechanistic diagnosis.** Section 4.3 attributes the failure to tracking a polar target that can change by constant order independently of the step size, whereas Section 4.4 tracks the gradient-side quantity whose variation is step-size-dependent. This provides insight beyond simply listing negative and positive results.

**Weaknesses**

- **The stated assumptions do not support the positive convergence claim in the stochastic multilayer setting.** Equation (3), asserted for arbitrary $X,Y$, makes $\nabla\_i g$ invariant to every other block when $X\_i=Y\_i$, while Equations (1) and (5) introduce stochastic gradients but the claimed rate under Assumptions 1–2 states no unbiasedness or noise condition. Unless smoothness is intended only for block-coordinate changes and the theory uses exact gradients, the advertised scope does not follow.

- **EF21-MuonSign controls a different iterate from the model that trains successfully.** Equation (13) evaluates gradients at broadcast model $W\_t$, while the guarantee bounds server model $X\_t$; Table 3 reports validation losses of $3.3213$ for $W$ and $5.5198$ for $X$, with every other reported optimizer between $3.2785$ and $3.4049$. Thus the practically useful model in the largest-scale experiment is not the theoretically controlled output.

- **The positive error-feedback contribution is primarily a framework instantiation rather than a new convergence analysis.** Section 4.4 states that EF21-MuonUSign and EF21-MuonSign are exact instances inheriting the existing EF21-Muon guarantee; gradient-side error feedback before a structured matrix transform also appears for full-matrix preconditioners in Modoranu et al. (2024). The novelty is therefore concentrated in the Muon/sign construction and LMO-specific negative diagnosis, which should frame the contribution.

- **Plain SignSGD is insufficient to attribute the empirical gains exclusively to matrix geometry.** Jiang et al. (2024), absent from Section 2, the references, and Tables 1–3, develop variance-reduced sign estimators and evaluate ResNet-18 on CIFAR-10 as well as distributed majority-vote optimization. Comparing against such a modern sign method is necessary to distinguish geometry from improved gradient tracking.

- **The implemented positive methods are not shown to satisfy the exact-oracle guarantees.** Section 4.4 formulates the theory through the polar LMO, whereas Section 4.1 states that every experiment approximates it with a fifth-order Newton–Schulz iteration. Without an inexact-LMO condition or approximation-error analysis, the experimental EF21 variants are not established as the algorithms covered by the convergence result.

- **The federated evidence covers a narrow distributed regime.** Section 5.2 uses a homogeneous split across eleven clients with no local steps, thereby avoiding the heterogeneity and client drift central to federated optimization. This experiment supports synchronous distributed learning claims, but broader federated conclusions require non-IID partitions and local updating.

**Suggestions for Improvement**

- **A compact method matrix would make the six similarly named variants directly comparable.** Placing it after Equation (6), with columns for sign location, error-feedback target, LMO location, transmitted objects, downlink precision, and guaranteed iterate, would eliminate repeated reconstruction across Sections 4.3–4.5.

- **The random-quadratic alignment observation could motivate an intermediate structural condition under which sign-after remains reliable.** Section 5 reports consistently positive alignment despite Theorems 1–3; studying sign stability, spectral structure, or high-dimensional noise effects could connect these findings to mechanistic analyses of sign optimization such as Xiao et al. (2025).

- **The theory would read more linearly if Section 4.1’s implementation and step-scaling material began the experimental section.** The current placement interrupts the progression from method definitions to counterexamples, failed oracle-output feedback, and successful gradient-side feedback.

- **Impossibility language should be scoped to the analyzed repair.** Theorem 4 rules out the specified scaled-sign error-feedback recursion on the LMO output, not every possible repair of sign-after; phrases such as “prove unrepairable” state a broader conclusion than the theorem.

**Additional Details and Minor Issues**

- A selection for $A(0)$ or $\operatorname{polar}(0)$ should be defined.
- In Table 3, the $0.0021$ gap between EF21-SignMuon and SignMuon is below the paper’s stated approximate noise level and comes from single-seed runs.
- The centralized claim of an approximately twofold wall-clock margin is not accompanied by elapsed-time measurements.
- “One bit” should consistently acknowledge the additional per-layer scalar; Table 2 reports $1.09$ bits per parameter.
- Table 2’s rounds-to-threshold values for MuonSign and SGD are conditional means over only the seeds that reached the threshold.

**References**

Jiang, W., Yang, S., Yang, W., & Zhang, L. (2024). Efficient sign-based optimization: Accelerating convergence via variance reduction. *Advances in Neural Information Processing Systems, 37*, 33891–33932. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b7db05c0e383518d789b6e93131f1f0-Abstract-Conference.html))

Modoranu, I.-V., Kalinov, A., Kurtic, E., Frantar, E., & Alistarh, D. (2024). Error feedback can accurately compress preconditioners. In *Proceedings of the 41st International Conference on Machine Learning* (pp. 35910–35933). Proceedings of Machine Learning Research, 235. ([proceedings.mlr.press](https://proceedings.mlr.press/v235/modoranu24a.html))

Xiao, K. L., Marshall, N., Agarwala, A., & Paquette, E. (2025). Exact risk curves of SignSGD in high dimensions: Quantifying preconditioning and noise-compression effects. In *Proceedings of the 42nd International Conference on Machine Learning* (pp. 68391–68439). Proceedings of Machine Learning Research, 267. ([proceedings.mlr.press](https://proceedings.mlr.press/v267/xiao25c.html))
