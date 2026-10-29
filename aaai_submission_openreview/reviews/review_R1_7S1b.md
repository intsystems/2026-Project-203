# Human review 1 — 7S1b

Note ID: `smHroyG4g5`. Source: `signmuon_openreview_page.json`.

## Brief summary

The reviewer values the explicit negative examples, the distinction between EF targets, and the experimental breadth (6, confidence 4). The five questions concern a positive explanation for SignMuon, federation beyond the tested setting, attribution of EF21 results, practical costs, and language-model uncertainty.

## Every concern / request

1. **R1.1 — Reviewer asks for gradient structure that permits SignMuon descent.** Explain a sufficient condition beyond the random-quadratic observation.

2. **R1.6 — Reviewer requests neural-network alignment and spectral measurements.** Measure the proposed condition along actual network trajectories.

3. **R1.2 — Reviewer asks whether federation changes preserve the ordering.** State what the 11-client experiment supports.

4. **R1.7 — Reviewer requests heterogeneous and local-training experiments.** Broaden the federated evaluation beyond synchronized homogeneous clients.

5. **R1.3 — Reviewer asks which positive results are inherited from EF21.** Credit the existing arbitrary-norm framework and isolate extensions.

6. **R1.9 — Reviewer asks what technical contribution is new.** Distinguish the negative construction from positive framework instantiation.

7. **R1.4 — Reviewer asks how to choose a communication variant.** Recommend choices with explicit evidence limits.

8. **R1.10 — Reviewer requests scales and memory in a comparison table.** Account for payloads, precision, estimators and polar location.

9. **R1.5 — Reviewer asks which nanoGPT differences are reliable.** Separate descriptive gaps from statistical evidence.

10. **R1.8 — Reviewer requests additional language-model seeds.** Repeat training rather than repeated validation of one run.

## Response priorities

Follow Feedback 1–5. State the structural condition first, separate inherited rates from new negative proofs, and reserve requested new experiments as visible unfinished fragments. The internal table supplies complete costs; the export compresses the practical advice.

## Original review

### title

1

### review

This paper studies how to combine **Muon’s matrix-aware optimization geometry with extreme one-bit communication**, focusing on where element-wise sign compression should be applied relative to Muon’s Linear Minimization Oracle (LMO). The authors consider three natural variants: **SignMuon**, which applies the sign after the LMO; **MuonUSign**, which signs the momentum/gradient before the LMO; and **MuonSign**, which applies sign compression on both sides. These variants aim to retain Muon’s matrix-aware structure while reducing communication to approximately one bit per parameter.

The main theoretical finding is that **none of these three sign placements is guaranteed to be a descent method in general**. The paper constructs explicit small-dimensional linear objectives on which SignMuon, MuonUSign, and MuonSign produce ascent directions for any positive step size and any momentum coefficient. The authors further show that applying standard error feedback to the LMO output does not repair SignMuon: the resulting EF21-SignMuon can still diverge on smooth, lower-bounded objectives.

To obtain convergence guarantees, the paper instead applies error feedback **before the LMO, at the gradient/momentum level**. This leads to **EF21-MuonUSign**, which provides a one-bit uplink, and **EF21-MuonSign**, which additionally compresses the downlink through a second error-feedback loop. Under smooth non-convex objectives, both variants achieve the standard (O(T^{-1/2})) convergence rate for the squared gradient norm, showing that one-bit compression changes constants rather than the asymptotic rate.

The paper evaluates the proposed variants against Muon, SignSGD, SGD, and Adam in centralized CIFAR-10 training, federated CIFAR-10, synthetic quadratic problems, and a nanoGPT language-modeling speedrun. Interestingly, the empirical results reveal a clear **theory–practice tension**: SignMuon, despite lacking a general convergence guarantee and admitting explicit divergence examples, is consistently among the strongest compressed methods. In federated CIFAR-10, for example, SignMuon approximately matches full-precision Muon while using about one bit per parameter in both communication directions and substantially outperforms SignSGD at the same communication budget.

Overall, the paper concludes that composing sign compression with Muon’s spectral LMO can destroy the worst-case descent property, and that convergence can be restored by applying error feedback to the gradient rather than to the LMO output. At the same time, the experiments suggest that the theoretically problematic sign-after-LMO strategy can be more effective in realistic neural-network training than the provably convergent alternatives, exposing an unresolved gap between worst-case optimization theory and empirical behavior.

### strengths

1. **The paper identifies a clear and practically relevant problem.** The work targets the tension between Muon’s matrix-aware optimization geometry and the high communication cost of transmitting dense full-precision updates. Studying whether Muon can retain its advantages under an approximately one-bit-per-parameter communication budget is timely and particularly relevant to distributed and federated training.

2. **The theoretical analysis is unusually concrete and provides meaningful negative results.** Rather than only establishing sufficient conditions for convergence, the paper constructs explicit low-dimensional counterexamples showing that all three natural sign placements—SignMuon, MuonUSign, and MuonSign—can become ascent methods even on linear objectives, independently of the step size and momentum choice. These results clearly expose a fundamental difficulty in naively composing sign compression with an LMO.

3. **The error-feedback analysis provides a useful distinction between where compression should be corrected.** The paper shows that applying error feedback to the LMO output does not generally repair SignMuon, while applying it at the gradient/momentum level leads to EF21-MuonUSign and EF21-MuonSign with convergence guarantees. This distinction provides a useful conceptual insight into why error feedback interacts differently with gradient compression and LMO-output compression.

4. **The paper combines theoretical results with experiments across diverse settings.** The empirical evaluation includes synthetic quadratic problems, centralized CIFAR-10 training, federated CIFAR-10, and a nanoGPT language-modeling speedrun. This breadth is valuable because it evaluates both the optimization behavior targeted by the theory and the practical performance of the proposed compression strategies at substantially different model scales.

5. **The theory–practice discrepancy is presented transparently rather than hidden.** A particularly interesting aspect of the paper is that the theoretically problematic sign-after-LMO strategy performs very well empirically. For example, SignMuon approximately matches full-precision Muon in the federated CIFAR-10 experiment while using about one bit per parameter in both directions, despite the paper proving that this placement can diverge in the worst case. The authors explicitly acknowledge this discrepancy and frame identifying realistic conditions under which SignMuon behaves well as an open problem. This makes the empirical findings informative even though they do not directly follow from the worst-case theory.

### weaknesses

1. **The central theory–practice gap remains largely unresolved.** The paper’s strongest theoretical results show that all three sign placements can fail even on simple objectives, yet the experiments consistently favor SignMuon—the variant for which the paper provides particularly strong negative results. The authors acknowledge this tension explicitly, but currently provide limited characterization of *why* the pathological cases do not arise in practical neural-network training. This is not necessarily a fatal flaw, but it weakens the connection between the theoretical contribution and the empirical conclusions. It would strengthen the paper to empirically characterize quantities related to the counterexamples, such as gradient–update alignment, singular-value structure, or conditions under which (\langle G,s(G)\rangle) approaches or crosses zero during realistic training.

2. **The federated-learning evaluation is relatively idealized.** Section 5.2 considers only (N=11) clients with a homogeneous CIFAR-10 partition and no local steps. This removes two central difficulties of practical federated learning: statistical heterogeneity and client drift caused by multiple local updates. Consequently, it is unclear whether the reported advantage of SignMuon persists under non-IID partitions, multiple local epochs/steps, partial participation, or larger client populations. This is a fixable experimental limitation; evaluations under heterogeneous partitions and standard FedAvg-style local training would substantially strengthen the federated claims.

3. **The empirical evidence at language-model scale is limited by the single-seed evaluation.** The nanoGPT experiment is an important addition because it tests substantially larger matrix dimensions, but Table 3 and Figure 3 report only a single seed for the compressed methods. Since several methods have relatively small differences in validation loss, it is difficult to determine how robust the reported ordering is. The paper itself notes that differences below approximately (0.003) are comparable to the seed variation of the reference Muon experiment. Running at least a small number of additional seeds for the strongest methods would make the large-scale empirical conclusion considerably more convincing.

4. **Some of the proposed convergent methods rely heavily on an existing error-feedback framework.** EF21-MuonUSign and EF21-MuonSign are presented as instances/extensions of the EF21-Muon framework, with the main technical step being to instantiate the compressor with scaled sign and verify the corresponding contractivity assumptions. The resulting convergence guarantees are useful, but the methodological novelty of these positive results appears more incremental than the negative counterexamples. The paper would benefit from more explicitly separating what follows directly from the existing EF21-Muon framework from what requires genuinely new analysis, and from emphasizing the latter.

5. **The practical communication story should account more explicitly for auxiliary costs and memory.** The headline description emphasizes one-bit-per-parameter communication, but the error-feedback variants additionally transmit per-layer scaling information and maintain residual/estimator buffers, while some variants retain full-precision downlink communication. The paper does discuss these details, including the additional buffer requirements, but the practical trade-off could be presented more systematically. A table summarizing uplink/downlink bits, auxiliary scalars, additional client/server memory, LMO location, and theoretical guarantee for each variant would make the comparison substantially clearer and prevent the “one-bit” characterization from obscuring meaningful implementation differences.

### justification

I lean positive on this paper because it addresses a practically relevant problem and provides a technically interesting analysis of one-bit compression for Muon. In particular, the explicit counterexamples showing that all three natural sign placements can produce ascent directions are insightful, and the distinction between applying error feedback to the LMO output versus the gradient provides a useful theoretical message. The extensive evaluation across centralized learning, federated learning, and language modeling further strengthens the work.

My main reservation is the unresolved gap between the theoretical and empirical findings. The strongest empirical method, SignMuon, is precisely the variant for which the paper establishes strong worst-case negative results, while the theoretically convergent variants generally perform worse. Although I view this tension as interesting rather than fatal, a clearer characterization of when SignMuon behaves well in realistic optimization would substantially strengthen the overall contribution. I am also somewhat concerned that the federated experiments use a relatively idealized homogeneous setting without local steps, and that the language-modeling results are mostly based on a single seed.

Overall, I find the problem important, the negative theoretical results nontrivial, and the empirical observations interesting enough to support a positive recommendation. I would rate the paper higher if the authors could better bridge the theory–practice gap and provide stronger empirical validation of the main conclusions under more realistic federated settings and additional large-scale runs.

### feedback

1. **Can the authors provide more intuition for the theory–practice discrepancy?** The theoretical results show that SignMuon can produce ascent directions and cannot generally be repaired by error feedback applied after the LMO, whereas SignMuon is among the strongest methods in the practical experiments. Can the authors clarify what properties of realistic neural-network gradients may prevent the counterexamples from arising, beyond the empirical observation that gradient–step alignment remains positive on the tested random quadratics?

2. **How broadly should the federated-learning conclusions be interpreted?** The current federated experiment uses 11 clients, homogeneous data, and no local steps. Could the authors clarify whether there are theoretical or empirical reasons to expect the relative ordering of SignMuon, MuonUSign, and MuonSign to remain similar under non-IID data or multiple local updates?

3. **What parts of the convergence analysis for EF21-MuonUSign and EF21-MuonSign are genuinely new relative to the existing EF21-Muon framework?** The paper states that these methods instantiate the existing framework using scaled-sign compressors. It would be helpful if the rebuttal could clearly distinguish results inherited directly from prior EF21-Muon theory from the new technical arguments required for the proposed methods.

4. **How should one choose among the proposed variants in practice?** The results reveal a meaningful trade-off: SignMuon provides strong empirical performance and one-bit bidirectional communication but lacks a general convergence guarantee, while EF21-MuonUSign/MuonSign provide guarantees but can perform worse empirically and introduce additional state or communication considerations. Could the authors provide a concise practical recommendation for different communication regimes (e.g., uplink-only versus bidirectional constraints)?

5. **How confident are the authors in the ordering observed in the nanoGPT experiment?** Since the compressed methods are evaluated with a single seed and some differences in validation loss are relatively small, could the authors clarify which comparisons they consider sufficiently large to support robust conclusions and which should be interpreted only as preliminary evidence?

### rating

6

### confidence

4
