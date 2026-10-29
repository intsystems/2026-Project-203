# Human review 4 — pqje

Note ID: `kkv2ydIYff`. Source: `signmuon_openreview_page.json`.

## Brief summary

The reviewer is borderline (5, confidence 4). Questions 1 and 2 prioritize theorem coverage and inconsistent tables. Further concerns ask for positive SignMuon conditions, honest prior-work credit, explicit nanoGPT transport, and a clearer main text.

## Every concern / request

1. **R4.1 — Reviewer asks whether the experimental schedules have the proved rate.** Identify the distinction between squared-gradient steps and deterministic schedules.

2. **R4.2 — Reviewer asks whether nanoGPT Nesterov momentum is covered.** Separate a fixed Nesterov coefficient from momentum warmup.

3. **R4.3 — Reviewer identifies inconsistent main and supplementary results.** Reconstruct the five runs and correct the ranking.

4. **R4.4 — Reviewer asks for useful sufficient conditions for SignMuon.** Supply a structural condition and a conditional result.

5. **R4.15 — Reviewer asks why the favorable quadratic alignment extends to neural networks.** Distinguish a sufficient theorem from evidence along actual training trajectories.

6. **R4.6 — Reviewer questions novelty relative to EF21 and concurrent SignMuon.** Credit existing frameworks and the latest concurrent theory.

7. **R4.5 — Reviewer says nanoGPT does not test the proposed distributed compression.** Make the full-precision transport explicit in the main text.

8. **R4.7 — Reviewer asks for definitions before theorem references.** Introduce the operations before the introduction cites alignment results.

9. **R4.8 — Reviewer objects to broad impossibility wording.** Replace “unrepairable” with the specified fixed-parameter output-EF statement.

10. **R4.9 — Reviewer rejects presenting empirical ordering as a reversed theoretical prediction.** The existence of failure examples does not predict method ranking.

11. **R4.10 — Reviewer asks for a direct conclusion without repetitive caveats.** Remove repeated “document rather than resolve” language.

12. **R4.11 — Reviewer asks which of X and W is updated and analyzed.** Explain exact server X, compressed broadcast W and their roles.

13. **R4.12 — Reviewer asks to explain target iteration counts before using them.** Distinguish finite-target success and stationary error floors from asymptotic rate.

14. **R4.13 — Reviewer asks for a clearer statement of the Section 5.2 results.** Lead with measured accuracy instead of theorem qualifications.

15. **R4.14 — Reviewer asks to move essential explanations into the main paper.** Prepare a coverage table and shorten captions.

## Response priorities

Answer explicit Questions 1–2 first. Do not claim the new theory certifies finite polar solvers or changing nanoGPT momentum. State positive geometry and prior-work credit, then centralized transport. Concrete replacement prose remains available internally for all writing criticisms.

## Original review

### title

Review for Submission 48060

### review

The paper studies one-bit sign compression of Muon updates, with the sign applied after orthogonalization (SignMuon), before it (MuonUSign), or both (MuonSign). It constructs explicit examples where each method produces an ascent direction. It also shows that, for every choice of step size and momentum, there is a smooth, lower-bounded objective on which error feedback applied to the polar output fails. Applying error feedback before orthogonalization instead fits the existing EF21-Muon framework and gives convergence guarantees under specific learning-rate and momentum choices.

### strengths

**1. The paper addresses an interesting question.** 

Both the elementwise sign and the polar operator produce descent directions when applied directly to the gradient. Studying whether this property is preserved when the two are combined is a natural question, especially in the context of compressing Muon updates.

**2. The counterexamples clearly illustrate the limitations.** 

The authors construct small matrices showing that all three placements of the sign operator can produce ascent directions, even with exact gradients. These examples are easy to verify and show that the issue comes from combining the two operators.

**3. The analysis of error-feedback placement is useful.**

 The paper explains why applying error feedback to the momentum is different from applying it to the polar output, which can change sharply even for small changes in its input. Theorem 4 provides a concrete divergence example for EF21-SignMuon, while the other placements retain existing convergence guarantees. This helps explain why the placement of error feedback matters.

### weaknesses

**1. Scope.**

The most interesting finding is that sign-after methods consistently perform best, despite being the ones targeted by the negative theoretical results. Even the random quadratic experiments show positive alignment between SignMuon's update and the gradient. This raises a more important question than whether the method can fail on carefully constructed examples: under what reasonable assumptions does sign compression after orthogonalization preserve a useful descent direction, and why are the theoretical failure cases not observed in practice? Appendix A.7 already identifies the gradient-weighted sign disagreement term that could serve as a starting point for such an analysis, but the paper does not pursue it. Instead, the theory focuses on worst-case counterexamples, while the positive convergence results largely follow from existing work. Moreover, SignMuon and its practical effectiveness have already been reported in the concurrent work cited by the authors, so these cannot be considered new contributions.

**2. Experiments.**

The nanoGPT experiments do not actually evaluate the proposed distributed compression methods. As explained in Appendix A.16, the gradients are first averaged in full precision using reduce-scatter, and the compressed update is then applied locally. In contrast, Algorithms A8–A9 compress each worker's updates before aggregation. This is an important difference, since averaging the signs of local polar factors is generally not equivalent to taking the sign of the polar factor after averaging. Although Appendix A.16 describes this implementation, Section 5.3 presents the results as evidence that the one-bit methods work for language modeling without clearly explaining this distinction. The experiments therefore only support the centralized variants and do not demonstrate the performance or communication benefits of the proposed distributed methods at this scale.

**3. Writing and Presentation**

The paper is considerably harder to read than its content requires. This is clear from the introduction, which refers to results before explaining them. For example, the following phrases are very difficult to understand at that point in the paper:

   - *the gradient/step alignment that Theorems 1–3 drive negative*
   - *exactly the placement we prove unrepairable*
   - *the cost of the guarantee at the placement rather than at the downlink*

   The conclusion has a similar problem, with sentences such as:

   > *Experiment orders the methods the other way round, which we document rather than resolve: what we preclude is a guarantee holding everywhere, not one under a realistic condition.*

   The phrase "*document rather than resolve*" also appears repeatedly throughout the paper. The authors spend too much time explaining what their results do not establish instead of clearly stating what they do.

   The technical sections are even harder to follow. Section 4.4 introduces the feature that distinguishes the two convergent methods, EF21-MuonSign's two models, in a single, unnecessarily complicated sentence:

   > *The server model X takes the exact LMO step and never leaves the server; what crosses the downlink is the one-bit refresh of W, the only model the rest of the method observes, since gradients, momentum and the uplink residual are all computed there, whereas the guarantee bounds the gradient at X.*

   It is difficult to understand from this description which model is updated, communicated, or covered by the theoretical guarantee. Similarly, Section 5 opens with several technical statements and the phrase "*a fixed-target iteration count ranks accuracy floors, not rates*", which is only explained in Appendix A.12. Section 5.2 presents the main experimental finding through a series of qualifications rather than explaining the result directly, starting with:

   > *Sign-after is what Theorem 1 makes diverge and error feedback repairs for no (L, η, μ) (Theorem 4), yet ...*

   Throughout the paper, important explanations are deferred to appendix sections (A.7, A.10, A.12, A.16, A.17), and several figure captions are unnecessarily long. Overall, the paper needs a much simpler and more direct presentation, particularly in the sections explaining the main results, ideally with a table stating which implementation choices each theorem covers.

### justification

The main weakness is that the paper does not address its most interesting question, why sign compression after orthogonalization works well in practice despite the theoretical counterexamples. The remaining contributions are limited, as the counterexamples show expected failures, the convergence guarantees largely follow from existing work, and the main practical finding has already been reported in concurrent work.

### feedback

#### **Questions**

**1. Are any experiments covered by the convergence guarantees?** 

Appendix A.10 states that the squared-gradient guarantee requires the learning-rate rule $\eta_{t,i}=\gamma_i\|g_{t,i}\|_*$, which is not used in the experiments. It also states that the experimental EF21-MuonSign schedule is not covered by either corollary, and that the Nesterov variant used in nanoGPT is outside the theoretical framework. Could the authors clarify which guarantees, if any, apply to each EF21 experiment in Tables 1–3? Currently, the paper compares methods with theoretical guarantees, but the actual experimental configurations do not necessarily satisfy their assumptions. This should be made clear in the main text.

**2. Inconsistencies between the main and supplementary tables.** 

I noticed some values differ between tables that appear to describe the same experiments. Federated MuonSign achieves $80.68\pm1.88$ in Table 2 but $82.94\pm0.19$ under the unit-gain rule in Table A8, whose caption refers to the same five seeds as Table 2. With the Table A8 value, MuonSign is above SignSGD rather than below it, which changes the ordering stated in Section 5.2. Could the authors explain these values and state which runs support the main conclusions?

### rating

5

### confidence

4
