# Human review 5 — Tx34

Note ID: `EDZy4k8BK7`. Source: `signmuon_openreview_page.json`.

## Brief summary

The reviewer rates the paper 4 (confidence 3) and asks two direct questions: what is new beyond EF21, and what Theorem 4 actually rules out. They also correctly reject an inferred performance ranking and treating sufficient compressor conditions as necessary.

## Every concern / request

1. **R5.1 — Reviewer asks what is new beyond the EF21-Muon framework.** State the new proofs and credit inherited positive rates.

2. **R5.3 — Reviewer asks what optimizer-design lesson the counterexamples provide.** Explain what is tracked and the consequences of its variation.

3. **R5.2 — Reviewer asks whether Theorem 4 includes decreasing learning rates.** Write the quantifiers and identify their limit.

4. **R5.4 — Reviewer says parameter-dependent divergence need not preclude finite-time rates.** Acknowledge the dependence of the constructed objective and its constants.

5. **R5.5 — Reviewer rejects a theoretical prediction of the empirical ranking.** Separate worst-case statements from comparative accuracy.

6. **R5.6 — Reviewer objects to presenting sufficient compressor assumptions as necessary.** Retract claims that only changing compressors can help.

## Response priorities

Prioritize the two explicit questions. Defend the concrete output-tracking failure as the submitted novelty. Agree with the precise logical objections and avoid using the new plain-method softplus result to overstate the EF result.

## Original review

### title

review

### review

The paper studies three ways to combine sign compression with Muon: SignMuon, MuonUSign, and MuonSign.

It gives examples where these methods fail to descend and shows that error feedback does not fix every version. 

It also gives convergence results for two EF21 variants and compares the methods on CIFAR-10 and nanoGPT.

### strengths

S1. The explicit counterexamples are useful and show why the position of sign compression matters.

S2. The paper compares several related methods (Algorithm 1-Algorithm 9) and examines both their theoretical limits and practical performance.

### weaknesses

1. **The main contribution is not clear enough.**  The paper presents several methods and failure cases but gives limited guidance on when and why users should choose each method. The convergence results mainly follow from the existing EF21-Muon framework. While the counterexamples are useful, the paper should explain more clearly what new lesson they provide for optimizer design.

2. **Fixed-step divergence does not rule out a finite-time convergence rate.**   Theorem 4 shows that, for any fixed learning rate and momentum, there exists a problem on which EF21-SignMuon diverges. However, that problem can depend on the chosen parameters. This result alone does not rule out a finite-time small-gradient guarantee when the parameters depend on the iteration budget. Claims that this placement cannot be repaired should therefore be stated more carefully. 

3. **The theory does not predict the experimental ranking.**   A failure example for method A and a convergence guarantee for method B do not imply that B should perform better on CIFAR-10 or nanoGPT. Thus, A performing better is not a reversal of a theoretical prediction. 

4. **Sufficient conditions are treated as necessary conditions.**  The paper argues that only changing the compressor can remove certain limitations. However, showing that a compressor fails a condition used in the current proof does not show that other analyses or parameter choices cannot work.

### justification

I recommend rejection in the current form because the main contribution beyond existing work is not clear enough. The convergence results mainly rely on the existing EF21-Muon framework. The new counterexamples are useful, but the paper gives limited guidance.

### feedback

1. What are the main new contributions beyond the existing EF21-Muon framework? 

2. What exactly does the counterexample in Theorem 4 rule out? Does it show that any fixed and decresing learning rate and momentum can fail on a suitable problem?

### rating

4

### confidence

3
