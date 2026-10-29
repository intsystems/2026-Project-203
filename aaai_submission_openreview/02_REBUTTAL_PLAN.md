# Rebuttal plan and concern-to-evidence ledger

## Strategy

Lead with defensible results. R3/R4's table discrepancy is a real reporting error settled by **already submitted** five-seed logs. R5's quantifier and necessity objections are mathematically correct; narrow the claims while defending the LMO-specific negative constructions. R1/R4's request for a positive SignMuon explanation can now receive a sufficient structural condition and conditional theorem. None of these changes establishes an end-to-end guarantee for the actual network runs.

The updated article is an author resource. Every active reply states its answer directly; it does not ask reviewers to consult an inaccessible revision. Text promises below have concrete proposed wording/tables in this file. New training is never claimed.

## Evidence register

| Tag | Evidence | What it establishes | Limit / remaining action |
| --- | --- | --- | --- |
| E1 | Submitted A8; original nested `code/results/federated_export_results.zip`; `evidence/verified_claims.json` | Independently recomputed five-seed tail means, sample SDs and threshold crossings; all 175 original/current training histories identical | Fix stale main Table 2 and associated ranking. No new training or significance test. |
| E2 | Submitted A.11/A.14/A.15/A.18; federated `algorithms.py`, `main.py`, `methods.py` | CNN2,11 homogeneous fully participating clients; three fixed-weight minibatches of 64 accumulated per client | Zero local parameter steps. No extrapolated ordering under label skew/local training/partial participation. Live heterogeneous runner is not a result. |
| E3 | Submitted Table3/A.16/A.17; `code/results/nanogpt/` | Exact losses, X/W gap, one run/method, full-precision reduce-scatter then local optimizer/all-gather | No variant-specific variance; no measured one-bit network savings; changing momentum and auxiliary Adam. |
| E4 | Submitted A5 and `article_export.tar.gz` → `table_cifar.csv`, `runs.csv` | Updated central accuracy values and existing seconds/epoch | Three final seeds; old generic2x timing claim should be replaced with measured scope. |
| C1 | Submitted communication appendix, federated `methods.py` and estimator allocation | Logical bits, FP32 scales/auxiliaries, polar location and persistent buffers | Single-GPU simulation; no packed-network throughput/latency measurement. |
| N1 | Submitted Theorems1–4; full proofs; rational certificates in `verified_claims.json` | Each plain placement can ascend; specified output-EF can diverge on smooth lower-bounded objectives | Exact polar; Thm4 has fixed parameters and parameter-dependent objective. |
| N2 | Existing author `signmuon_addendum_body.tex` | Softplus strengthens the three plain witnesses to one fixed smooth lower-bounded objective for all positive step and EMA/Nesterov schedules; EF gradient floor scales as c_mu L eta | No arbitrary-schedule impossibility for output EF; its vanishing floor does not refute horizon-tuned rates. |
| G1 | Existing author addendum; OLion geometry | Structural sign/polar alignment criterion and exact-polar conditional SignMuon theorem, including stochastic fixed EMA/Nesterov | Unmeasured uniform geometric margin in our networks; distributed final-vote condition is separate; no performance ranking. |
| T1 | Submitted A.10; updated Nesterov/fixed-radius/rate TeX | Separate sharp second-moment proof; new fixed-momentum deterministic-schedule bounds | Never square a first-moment bound; the squared result needs gradient-scaled radius and horizon tuning. |
| J1 | New `evidence/EF21_JOINT_SMOOTHNESS.md`, Sections1–3 | Whole-tuple norm reduction removes separability. Explicit bound(J) gives first-moment rates/residuals for either downlink, fixed EMA/Nesterov and arbitrary deterministic LR | Exact polar, global joint smoothness, no auxiliary Adam/decoupled decay; not inserted in article. Constants may be poor. |
| P1 | EF21-Muon, Modoranu2024, Mishra2026v2 | Correct prior-work attribution, including concurrent actual-vote conditional stochastic theory | Replace current main/v1 comparison. Do not market the general framework or practical SignMuon as new. |
| P2 | Jiang2024SSVR; Xiao2025; existing SignSGD implementation | Relevant modern sign baselines and theory; existing SignSGD already has momentum | No SSVR comparison completed; no exclusive geometry attribution; SignSGD risk curves do not prove SignMuon properties. |
| W1 | Concrete proposed text and tables below | Reviewable presentation corrections and scope statements | Proposed article changes, not silently applied to author TeX. |

## Precise new positive statements

For (M=USigma V^	op), (Q=UV^	op), (S=mathrm{sign}(Q)), write (c_j=[U^	op S V]_{jj}). Then

[
 langle M,S
angle=sum_jsigma_j c_j.
]

Thus **every (c_jgedelta>0)** is an easily stated sufficient condition for (langle M,S
anglegedelta|M|_*). More generally, let (a=sum_jsigma_j/r), (b=|Q|_1/r), (
ho=|Sigma-aI|_F/(asqrt r)), (arepsilon=|c-bmathbf1|_2/(bsqrt r)). Cauchy–Schwarz gives

[
 langle M,S
anglege a|Q|_1(1-arepsilon
ho).
]

The mean identity is (sum_j c_j=|Q|_1). A uniformly positive **product margin**, not small correlation dispersion alone, is needed. This development is motivated by OLion's conditional geometry and must be credited. Uniform alignment along the *actual momentum signals*, joint smoothness, unbiased finite-variance noise, exact polar, (eta=eta_0T^{-3/4}), and (mu=1-T^{-1/2}) fixed within a run give an average expected first-moment nuclear-gradient rate (O(T^{-1/4})) for EMA/Nesterov. Full proof: the existing author addendum. Do not claim the actual networks satisfy its premise.

For EF21, analyze all layers as one block with norm (max_i|X_i|_{
m op}/lambda_i), dual (mathcal N_lambda(G)=sum_ilambda_i|G_i|_*), and (B^2=sum_ilambda_i^2min(m_i,n_i)). Joint Frobenius smoothness implies the arbitrary-norm framework's smoothness with constant (B^2L); separate scaled-sign compression of layers is still Frobenius contractive. This removes the original block-separability restriction **within the prior framework**, without claiming a new general framework.

The explicit bound(J) in `EF21_JOINT_SMOOTHNESS.md` further proves an LR-weighted expected first-moment guarantee at server X for deterministic schedules, either downlink, and fixed EMA/Nesterov. Fixed peak LR/momentum leave a residual. Constant/cosine schedules with (T^{-3/4}) peak and (1-mu=T^{-1/2}) yield (O(T^{-1/4})). It is not a squared-gradient rate. Joint coupling is handled by the full model displacement in every gradient-drift term. The proof is analytic; the recorded numerical norm checks are only sanity checks, not the proof.

## What each guarantee covers

| Statement | Oracle / update | LR / momentum | Objective/noise | What it controls |
| --- | --- | --- | --- | --- |
| Submitted Thms1–3 | Exact polar with three plain sign compositions | Any positive steps; linear-case EMA/Nesterov | Linear local descent witnesses, unbounded below | Negative gradient–step pairing, not a network ranking |
| New bounded-below plain witnesses | Same exact compositions | Any positive step schedule, any EMA/Nesterov coefficients in[0,1) | One fixed softplus objective per placement, smooth convex/lower-bounded | Gradient norm bounded away from zero for all t; objective diverges if sum of steps diverges |
| Submitted Thm4, output EF | Specified scaled-sign estimate of exact polar outputs | Every fixed eta>0, fixed mu<1; objective may depend on them | Smooth/lower-bounded construction | Divergence; no universal constant-step rule based only on L,mu |
| Submitted/updated sharp EF21 rate | Exact gradient/momentum LMO with scaled radius | Gradient-dependent radius, prescribed horizon-tuned momentum | Explicit smoothness and stochastic variance assumptions | Separate expected **squared** dual-gradient rate O(T^-1/2) at X |
| New J1 bound(J) | Exact polar, gradient-side EF, identity or EF model downlink | Deterministic constant/cosine/etc; fixed EMA/Nesterov | Joint Frobenius smoothness, conditionally centered bounded-variance noise | Weighted **first** moment at X; residual for fixed parameters, tuned O(T^-1/4) |
| New conditional plain SignMuon | Exact polar-sign update | Horizon-tuned steps and fixed EMA/Nesterov | Joint smoothness/noise plus uniform positive alignment | First moment; condition not measured in the networks |
| CIFAR actual implementations | Finite five-iteration polynomial polar; auxiliary Adam group | Tuned cosine; fixed momentum | Real networks; global smoothness not verified | Empirical results, no complete end-to-end guarantee |
| nanoGPT actual implementation | Five Polar Express stages, full-precision reduction; auxiliary Adam | Momentum warmup/cooldown, flat/linear LR | Single training run/method | Central optimizer evidence, no full-run theorem/network benefit |

Finite polynomial stages may not preserve exact-polar signs. The addendum's sign-mismatch penalty is a conditional extension; its smallness is not established for the actual runs. Failure of spectral contractivity is not proof that no other analysis, LR, initialization or compressor repair can work.

## Complete method / systems comparison

Define (P(H)=operatorname{polar}(H)), (s(H)=operatorname{sign}(H)), and (C(R)=operatorname{mean}|R|,s(R)). The descent update subtracts the listed direction. The following compares distributed semantics, not nanoGPT's full-precision transport. Matrix parameters are compressed; auxiliary vector/head parameters use FP32. Raw sign votes are nonlinear and cannot be moved through averaging or the polar operation.

| Method | Sign placement / EF target | Polar location | Matrix uplink | Matrix downlink | Extra persistent matrix-size state beyond common model/momentum | Guarantee |
| --- | --- | --- | --- | --- | --- | --- |
| SignMuon | (s(P(M_j))), worker votes; no EF | Clients | 1 bit/entry | 1 bit/entry | None | Plain-method counterexample; new conditional central alignment result |
| MuonUSign | Server polar after client sign vote; no EF | Server | 1 bit/entry | 32 bits/entry | None | Plain-method counterexample |
| MuonSign | Server polar after client sign vote, sign output; no EF | Server | 1 bit/entry | 1 bit/entry | None | Plain-method counterexample |
| EF21-SignMuon | EF tracks client polar output | Clients | 1 bit/entry + FP32 scale/block | 32 bits/entry | One estimator/client + one server aggregate | Output-EF counterexample already at N=1 |
| EF21-MuonUSign | EF tracks gradient/momentum input to polar | Server | 1 bit/entry + FP32 scale/block | 32 bits/entry | One estimator/client + one server aggregate | Exact-polar EF21 at server X=W, with stated steps/noise |
| EF21-MuonSign | Input EF plus model-downlink EF | Server | 1 bit/entry + FP32 scale/block | 1 bit/entry + FP32 scale/block | One estimator/client + one server aggregate + exact X separate from broadcast W | Exact-polar EF21 at server X; W is queried/communicated |

CNN2: (P_{
m mat}=762560), (P_{
m aux}=2146), 3 matrix blocks. Per-client/per-round per-channel counts:

- Plain compressed: ((P_{
m mat}+32P_{
m aux})/(P_{
m mat}+P_{
m aux})=1.0870) bits/parameter.
- EF compressed: add (32\times3) bits, giving 1.0871 bits/parameter.
- Bidirectionally compressed payload reduction versus FP32: about 29.4x; uplink-only **round-trip** reduction: about 1.93x.
- Each extra FP32 matrix-size buffer has 3,050,240 bytes (about 2.91 MiB). Scratch polar workspaces and common momentum/Adam state are excluded; actual peak memory is not measured by this table.

Practical recommendation: if uplink is limiting and the analyzed repair matters, EF21-MuonUSign keeps the downlink exact; if both channels are constrained, EF21-MuonSign compresses both, with additional state and an X/W gap to monitor. SignMuon is the strongest observed vision-accuracy candidate, with a worst-case failure and only conditional positive theory. No universal ordering or guarantee for the finite solvers is implied.

## Concrete proposed article corrections

### Reporting and empirical claims

> Table 2 in the submission was stale. The submitted five independent 2000-round runs agree with Table A8: MuonSign 82.94±0.19%, SignSGD 81.44±0.15%, SignMuon 85.72±0.24%, and Muon 85.98±0.26% (sample SD). MuonSign therefore exceeds SignSGD by 1.51 percentage points. These values replace the main table and reverse the affected comparison. The central main table is similarly synchronized to submitted Table A5. No additional training was required for these corrections.

> The federated experiment is synchronous distributed CNN2 training with 11 homogeneous fully participating clients. Each client accumulates three minibatches of 64 at the same weights; clients take no local parameter steps. The result does not establish an ordering under statistical heterogeneity, local training or partial participation.

> The nanoGPT experiment averages gradients in full precision by reduce-scatter, then applies a centralized optimizer on owning ranks and gathers parameters. It tests optimizer behavior at language-model scale; it does not test compress-before-aggregation worker voting or demonstrate one-bit network savings. Each compressed method has one independent training run. The 0.0021 loss difference between EF21-SignMuon and SignMuon is inconclusive; no compressed-method variance estimate is available. Upstream Muon variation is not a variance estimate for the other methods.

> EF21-MuonSign's exact server X has validation loss 5.5198 and its broadcast W has 3.3213 in the same run. Gradients are queried at W; the ideal theorem bounds stationarity at X. Successful validation at W does not validate the guarantee at X. Layerwise downlink concentration is observed, but does not prove its cause or that changing compressors is the only remedy.

### Definitions and guarantees

> The sign direction is itself an LMO: (-operatorname{sign}(G)) minimizes (langle G,D
angle) over the unit entrywise infinity-norm ball. Composing it with the polar LMO can nevertheless produce an ascent direction. Plain signs carry no scale; scaled-sign EF transmits one mean-absolute residual scale per block in addition to bits. Auxiliary FP32 parameters are counted separately.

> Centralized placement formulas correspond to a single worker. Distributed variants apply signs or polar factors locally before aggregation, as specified in their algorithms; these operations generally do not commute with averaging. We retain the existing method names to keep code and tables traceable and add an explicit composition-order legend.

> Theorem 4 states: for every fixed (L>0,eta>0,muin[0,1)), there exists an (L)-smooth lower-bounded objective on which the specified output-EF recursion diverges. The objective can depend on the parameters. The result does not establish failure of decreasing schedules or exclude horizon-tuned finite-time rates, other EF repairs or different analyses. Replace “unrepairable”, “unconditional guarantee” and “only changing the compressor” with the precise relevant statement.

> Our convergence statements require explicit conditional unbiasedness and finite variance as well as smoothness and lower boundedness. The squared-gradient rate uses a separate second-moment argument with gradient-scaled steps. A first-moment bound cannot be squared to obtain an expected-square bound. The new whole-tuple formulation permits coupled layers, and the deterministic-schedule theorem gives first-moment residuals/rates for ideal fixed-momentum EF21. Finite polynomial polar, changing nanoGPT momentum and auxiliary Adam remain outside the full-run guarantee.

> The linear examples establish a failure of descent outside the lower-bounded assumption; (f	o-infty) for uncompressed Muon is literal on those linear objectives. New softplus examples preserve each plain placement's failure while satisfying smoothness and lower boundedness. Set the theoretical thin-SVD selection (operatorname{polar}(0)=0); distinguish this from randomized strict-binary sign ties in the implementation.

### Presentation and related work

- State operations and the meaning of gradient–step alignment before referring to numbered theorems. Put the method and coverage tables near the definitions.
- Move practical polar approximation and per-layer scaling from the middle of the theory to experimental setup. Define X as the exact server iterate and W as its compressed broadcast in two short sentences.
- Lead results paragraphs with measured outcomes. Explain target-crossing success counts before reporting target rounds; a fixed target can be affected by residual floors and does not alone identify an asymptotic rate. Corrected MuonSign and SGD both reach 80% in 5/5 submitted runs.
- Replace repeated “document rather than resolve” wording with the actual result. Shorten captions and bring the essential qualifications from A.7/A.10/A.12/A.16/A.17 into the main text.
- Credit prior EF21-Muon for the general positive framework, and Modoranu2024 for compression before full-matrix transforms. Credit Mishra v2's conditional actual-vote theory and OLion's geometric analysis. Do not repeat the current manuscript's v1-only dismissal of concurrent SignMuon.
- Cite Jiang2024SSVR and narrow geometry-only attribution. The existing SignSGD baseline already uses momentum, but no SSVR training comparison was performed. Xiao2025 analyzes SignSGD, not our composed SignMuon update.
- Withdraw the broad 2x wall-clock assertion. Submitted A5 does contain seconds/epoch (SignMuon 16.6, Muon 14.4, SignSGD 12.5, SGD 12.2, Adam 12.7); those are not end-to-end compressed-network timing measurements.

## Primary external sources checked on 29 October 2026

These links are internal evidence only and are excluded from every exported rebuttal. They were used to check attribution and the most recent concurrent theory, not to invent new empirical support.

- [Gruntkowska et al., Error Feedback for Muon and Friends](https://arxiv.org/html/2510.00643v1): arbitrary-norm distributed EF21 framework and its smooth stochastic theorems.
- [Wang, Shen and Zhang, OLion v2](https://arxiv.org/html/2602.01105v2): conditional diagonal-correlation geometry and spectral-spread interaction. Our own network margin is not measured.
- [Mishra et al., SignMuon v2](https://arxiv.org/html/2605.16311v2): conditional actual-vote theory, stochastic momentum and joint-block extension. The favorable condition is not automatic; our negative examples remain compatible with it.
- [Modoranu et al., ICML2024](https://proceedings.mlr.press/v235/modoranu24a.html): error feedback for compressed structured preconditioners.
- [Jiang et al., NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b7db05c0e383518d789b6e93131f1f0-Abstract-Conference.html): variance-reduced sign methods; no new comparison is available here.
- [Xiao et al., ICML2025](https://proceedings.mlr.press/v267/xiao25c.html): SignSGD risk analysis, not a guarantee for signing a polar factor.

## Every concern: source, finding, limit, concise answer

Exact reviewer quotes live in `annotations.json` and the immutable original-review blocks. The following ledger is generated from the same 64 selectable concerns. “Pending” concerns remain excluded from the active draft and visibly flagged in the editor. A completed limitation statement can be checked even when a separate request for new experiments remains open.

### R1.1 — Reviewer asks for gradient structure that permits SignMuon descent

**Status:** CHECKED — selected.

**Reviewer request:** Explain a sufficient condition beyond the random-quadratic observation.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The diagonal-correlation condition implies alignment by weighting each diagonal entry by its nonnegative singular value. Uniform alignment, joint smoothness, centered finite-variance noise, exact polar and horizon-tuned fixed EMA/Nesterov yield an expected first-moment nuclear-gradient rate O(T^-1/4). It is a conditional result; actual trajectory verification and distributed-vote alignment remain open.

**Concise candidate:** For momentum $M=U\Sigma V^\top$, if each diagonal entry of $U^\top\mathrm{sign}(UV^\top)V$ is at least $\delta>0$, then $\langle M,\mathrm{sign}(UV^\top)\rangle\ge\delta\|M\|_*$. A uniform margin, joint smoothness and unbiased finite-variance noise yield a conditional SignMuon rate; the margin is unmeasured in our networks.

### R1.6 — Reviewer requests neural-network alignment and spectral measurements

**Status:** PENDING — excluded; needs evidence.

**Reviewer request:** Measure the proposed condition along actual network trajectories.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No neural-network measurement of the full geometric margin is available. Proposed diagnostics must record momentum/gradient alignment, diagonal correlations and spectral spread for the actual approximate solver and vote.

**Concise candidate:** No neural-network measurement of the full geometric margin is available. Proposed diagnostics must record momentum/gradient alignment, diagonal correlations and spectral spread for the actual approximate solver and vote.

### R1.2 — Reviewer asks whether federation changes preserve the ordering

**Status:** CHECKED — selected.

**Reviewer request:** State what the 11-client experiment supports.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. New heterogeneous runner exists in live repo, but no completed new result was available.

**Finding, meaning and limits:** Neither theory nor the existing experiments predict an ordering under label skew, local updates, partial participation or larger client counts. Three accumulated mini-batches are not local parameter steps.

**Concise candidate:** The 11-client IID experiment takes no local updates. We claim no ordering under non-IID data, local training or partial participation.

### R1.7 — Reviewer requests heterogeneous and local-training experiments

**Status:** PENDING — excluded; needs evidence.

**Reviewer request:** Broaden the federated evaluation beyond synchronized homogeneous clients.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. New heterogeneous runner exists in live repo, but no completed new result was available.

**Finding, meaning and limits:** No completed non-IID/local-training/partial-participation result is available in the audited evidence. A newly added heterogeneous runner is preparation, not experimental evidence.

**Concise candidate:** No completed non-IID/local-training/partial-participation result is available in the audited evidence. A newly added heterogeneous runner is preparation, not experimental evidence.

### R1.3 — Reviewer asks which positive results are inherited from EF21

**Status:** CHECKED — selected.

**Reviewer request:** Credit the existing arbitrary-norm framework and isolate extensions.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit. T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment.

**Finding, meaning and limits:** Scaled-sign contractivity and the base positive convergence guarantee instantiate prior EF21-Muon. The new rebuttal analysis treats all layers as one arbitrary-norm block and proves joint-smooth deterministic-schedule first-moment bounds for fixed EMA/Nesterov and either downlink.

**Concise candidate:** The base positive rates instantiate EF21-Muon. Our added analysis removes layer separability and treats fixed EMA/Nesterov with deterministic schedules; it does not certify finite polar approximations.

### R1.9 — Reviewer asks what technical contribution is new

**Status:** CHECKED — selected.

**Reviewer request:** Distinguish the negative construction from positive framework instantiation.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.

**Finding, meaning and limits:** The placement-specific ascent examples and smooth bounded-below output-EF limit cycle are the principal submitted new analysis. The new conditional SignMuon and whole-tuple EF21 derivations are follow-up results; the general EF21 framework is not ours.

**Concise candidate:** The principal new submitted proofs are the placement-specific ascent examples and bounded-below output-EF divergence.

### R1.4 — Reviewer asks how to choose a communication variant

**Status:** CHECKED — selected.

**Reviewer request:** Recommend choices with explicit evidence limits.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput. E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** Under an uplink constraint EF21-MuonUSign retains an exact/full-precision downlink; under a bidirectional constraint EF21-MuonSign compresses both sides. SignMuon is the empirical candidate if conditional rather than general guarantees are acceptable. No universal ranking is predicted.

**Concise candidate:** Uplink-only: EF21-MuonUSign; both links: EF21-MuonSign for the analyzed repair, or SignMuon for the strongest observed accuracy.

### R1.10 — Reviewer requests scales and memory in a comparison table

**Status:** CHECKED — selected.

**Reviewer request:** Account for payloads, precision, estimators and polar location.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** The internal method table is prepared in 02_REBUTTAL_PLAN.md. A compressed CNN2 channel costs 1.0870 bits/parameter for plain signs, 1.0871 for EF with FP32 auxiliary parameters and one FP32 scale per matrix. Each gradient EF method adds one client estimator and one global server estimator; two-sided EF additionally stores exact X separately from W.

**Concise candidate:** CNN2 compressed payloads are 1.087 bits/parameter including FP32 auxiliaries; EF adds one scale/matrix, client/server estimators, and a separate server model for two-sided EF. Polar: clients for SignMuon, server for gradient EF.

### R1.5 — Reviewer asks which nanoGPT differences are reliable

**Status:** CHECKED — selected.

**Reviewer request:** Separate descriptive gaps from statistical evidence.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** EF21-SignMuon vs SignMuon is 3.2860 vs 3.2881, gap .0021. The approximate .003 upstream variation concerns Muon, not these variants. SignMuon vs SignSGD gap is .1168 nats, descriptive only; no variant-specific variance estimate or robust ranking.

**Concise candidate:** nanoGPT: the 0.0021 gap is inconclusive. All compressed-method rankings are preliminary single-run observations; even larger gaps lack a variant-specific variance estimate.

### R1.8 — Reviewer requests additional language-model seeds

**Status:** PENDING — excluded; needs evidence.

**Reviewer request:** Repeat training rather than repeated validation of one run.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No additional independent compressed-method training runs are available. Repeated evaluations and upstream Muon runs cannot substitute for them.

**Concise candidate:** No additional independent compressed-method training runs are available. Repeated evaluations and upstream Muon runs cannot substitute for them.

### R2.1 — Reviewer asks to identify sign as an LMO earlier

**Status:** CHECKED — selected.

**Reviewer request:** Define the geometry in Section 2.

**Sources:** W1: submitted Section 2 and current signmuon_body.tex Section 4.1.

**Finding, meaning and limits:** We will state in Section 2 that $-\mathrm{sign}(G)$ minimizes $\langle G,D\rangle$ over $\|D\|_\infty\le1$.

**Concise candidate:** We will state in Section 2 that $-\mathrm{sign}(G)$ minimizes $\langle G,D\rangle$ over $\|D\|_\infty\le1$.

### R2.4 — Reviewer asks to distinguish raw signs from scaled EF messages

**Status:** CHECKED — selected.

**Reviewer request:** Explain the EF scale when the variants are first introduced.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** Plain signs send bits only; EF sends the residual signs plus its mean absolute value per matrix. FP32 auxiliary parameters are also counted.

**Concise candidate:** Plain signs send bits only; EF sends the residual signs plus its mean absolute value per matrix. FP32 auxiliary parameters are also counted.

### R2.5 — Reviewer asks to distinguish centralized formulas from distributed algorithms

**Status:** CHECKED — selected.

**Reviewer request:** Explain the N=1 correspondence and nonlinear aggregation.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. New heterogeneous runner exists in live repo, but no completed new result was available. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** Algorithms A2/A4/A5 define centralized placements. At N=1 the distributed definitions reduce to them; at N>1 local sign/LMO and majority voting define different algorithms. We will label both settings explicitly.

**Concise candidate:** Algorithms A2/A4/A5 define centralized placements. At N=1 the distributed definitions reduce to them; at N>1 local sign/LMO and majority voting define different algorithms. We will label both settings explicitly.

### R2.6 — Reviewer questions the linear objective under lower boundedness

**Status:** CHECKED — selected.

**Reviewer request:** Explain why the linear tests do not satisfy Assumption 1.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.

**Finding, meaning and limits:** The linear examples test descent outside Assumption 1; $f\to-\infty$ is literal there. We also proved lower-bounded replacements: $f(X)=\log(1+e^{\langle G,X\rangle})$ retains the plain placements’ ascent directions for every positive step schedule and EMA/Nesterov momentum.

**Concise candidate:** The linear examples test descent outside Assumption 1; $f\to-\infty$ is literal there. We also proved lower-bounded replacements: $f(X)=\log(1+e^{\langle G,X\rangle})$ retains the plain placements’ ascent directions for every positive step schedule and EMA/Nesterov momentum.

### R2.7 — Reviewer suggests consistent names for sign placements

**Status:** CHECKED — selected.

**Reviewer request:** Make the composition order readable without changing identities.

**Sources:** W1: current signmuon_body.tex equations (5)–(6); method table in 02_REBUTTAL_PLAN.md.

**Finding, meaning and limits:** We will retain labels to keep tables/code traceable and add an operation-order legend: SignMuon = sign(polar(M)); MuonUSign = polar(sign(M)); MuonSign = sign(polar(sign(M))).

**Concise candidate:** We will retain labels to keep tables/code traceable and add an operation-order legend: SignMuon = sign(polar(M)); MuonUSign = polar(sign(M)); MuonSign = sign(polar(sign(M))).

### R2.2 — Reviewer suggests larger centralized and federated experiments

**Status:** PENDING — excluded; needs evidence.

**Reviewer request:** Additional models and truly distributed language-model training would strengthen the evidence.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. New heterogeneous runner exists in live repo, but no completed new result was available. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No new large-model or distributed-language-model run was completed. nanoGPT evaluates the centralized optimizer after full-precision averaging, so it cannot fill this request.

**Concise candidate:** No new large-model or distributed-language-model run was completed. nanoGPT evaluates the centralized optimizer after full-precision averaging, so it cannot fill this request.

### R2.3 — Reviewer asks for progress on the theory-practice explanation

**Status:** CHECKED — selected.

**Reviewer request:** Give a positive condition without claiming a complete explanation.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** With joint smoothness, unbiased finite-variance noise, exact polar and horizon-tuned steps/momentum, we prove SignMuon convergence if $\langle M,\mathrm{sign}(\mathrm{polar}(M))\rangle\ge\delta\|M\|_*$ uniformly. It suffices that all diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ exceed $\delta>0$ for $M=U\Sigma V^\top$. Verification in our networks remains open.

**Concise candidate:** With joint smoothness, unbiased finite-variance noise, exact polar and horizon-tuned steps/momentum, we prove SignMuon convergence if $\langle M,\mathrm{sign}(\mathrm{polar}(M))\rangle\ge\delta\|M\|_*$ uniformly. It suffices that all diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ exceed $\delta>0$ for $M=U\Sigma V^\top$. Verification in our networks remains open.

### R2.8 — Reviewer wants a realistic explanation of observed success

**Status:** PENDING — excluded; needs evidence.

**Reviewer request:** Check the proposed sufficient condition on trained networks.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The conditional proof supplies a mathematical intermediate result. It does not establish that network momentum, approximate polar factors or distributed votes satisfy it.

**Concise candidate:** The conditional proof supplies a mathematical intermediate result. It does not establish that network momentum, approximate polar factors or distributed votes satisfy it.

### R3.2 — Reviewer identifies the reversed MuonSign versus SignSGD ranking

**Status:** CHECKED — selected.

**Reviewer request:** Correct the main table and every affected comparison.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** Table 2 is stale. Five independent submitted runs, each averaged over the final five evaluations at the 2000-round horizon, give MuonSign 82.9424 ± 0.1868 and SignSGD 81.4356 ± 0.1466 sample SD. These agree with A8, reversing the main-table sentence. SignMuon 85.7156 ± .2444 versus Muon 85.9836 ± .2608; no significance claim.

**Concise candidate:** Table 2 is stale. Recomputing the five submitted runs gives MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8: MuonSign is higher by 1.51 percentage points. SignMuon is 85.72±0.24% and Muon 85.98±0.26%. We will replace the main table and reverse the affected sentence; no new training was needed.

### R3.1 — Reviewer asks which learning-rate rule proves the squared-gradient rate

**Status:** CHECKED — selected.

**Reviewer request:** Do not transfer the sharp-stepsize result to cosine schedules.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment.

**Finding, meaning and limits:** The squared dual-gradient rate O(T^-1/2) has a separate second-moment argument with gradient-scaled radii and horizon-tuned momentum. It does not follow for the cosine settings used in the experiments.

**Concise candidate:** The squared-gradient $O(T^{-1/2})$ result uses gradient-scaled steps and a separate second-moment proof; it does not certify the experimental cosine schedule.

### R3.3 — Reviewer identifies the separability forced by layer-wise smoothness

**Status:** CHECKED — selected.

**Reviewer request:** Replace the restriction with an explicit whole-model formulation.

**Sources:** J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit.

**Finding, meaning and limits:** We removed separability by treating the whole model as one block with norm $\max_i\|X_i\|_{\rm op}/\lambda_i$ and dual $\sum_i\lambda_i\|G_i\|_*$, for fixed layer weights $\lambda_i>0$. Joint Frobenius smoothness suffices. The squared-rate formulation then uses one common radius scaled by this dual norm, changing the per-layer rule.

**Concise candidate:** We removed separability by treating the whole model as one block with norm $\max_i\|X_i\|_{\rm op}/\lambda_i$ and dual $\sum_i\lambda_i\|G_i\|_*$, for fixed layer weights $\lambda_i>0$. Joint Frobenius smoothness suffices. The squared-rate formulation then uses one common radius scaled by this dual norm, changing the per-layer rule.

### R3.4 — Reviewer rejects squaring an expected first-moment bound

**Status:** CHECKED — selected.

**Reviewer request:** Remove the unsupported inference and state two distinct results.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit.

**Finding, meaning and limits:** You are right: squaring $\mathbb E\|\nabla f\|$ does not bound $\mathbb E\|\nabla f\|^2$. We will delete that inference. For deterministic constant/cosine radii, our new joint-smooth bound gives a weighted first moment $O(T^{-1/4})$ when peak LR scales as $T^{-3/4}$ and $1-\mu=T^{-1/2}$. Fixed LR/momentum leave a residual bound.

**Concise candidate:** You are right: squaring $\mathbb E\|\nabla f\|$ does not bound $\mathbb E\|\nabla f\|^2$. We will delete that inference. For deterministic constant/cosine radii, our new joint-smooth bound gives a weighted first moment $O(T^{-1/4})$ when peak LR scales as $T^{-3/4}$ and $1-\mu=T^{-1/2}$. Fixed LR/momentum leave a residual bound.

### R3.5 — Reviewer asks to distinguish ideal EF21 from the implementations

**Status:** CHECKED — selected.

**Reviewer request:** State the remaining oracle and optimizer qualifications.

**Sources:** J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The new deterministic-schedule bound allows fixed EMA/Nesterov and either downlink but assumes exact polar, no decoupled weight decay and all jointly analyzed updates of the specified form. Finite polynomial polar, momentum warmup and auxiliary Adam are not covered.

**Concise candidate:** These statements concern exact polar factors, fixed-within-run momentum and the server iterate. Finite polar solvers, nanoGPT momentum warmup and auxiliary Adam remain outside the guarantees.

### R4.1 — Reviewer asks whether the experimental schedules have the proved rate

**Status:** CHECKED — selected.

**Reviewer request:** Identify the distinction between squared-gradient steps and deterministic schedules.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit.

**Finding, meaning and limits:** The original sharp-stepsize squared guarantee does not cover Tables 1–3. New joint-smooth deterministic-schedule proof (J) gives a residual or a horizon-tuned first-moment rate, not the advertised squared rate.

**Concise candidate:** No complete network run is certified: all use finite polar solvers; nanoGPT also warms momentum and uses auxiliary Adam. The squared-gradient rate needs gradient-scaled steps. We now prove, under joint smoothness and unbiased bounded-variance noise, a constant/cosine first-moment bound for exact-polar EF21 with either downlink and fixed EMA/Nesterov. Fixed peak LR/momentum leave a residual; $T^{-3/4}$ peak LR and $1-\mu=T^{-1/2}$ give $O(T^{-1/4})$.

### R4.2 — Reviewer asks whether nanoGPT Nesterov momentum is covered

**Status:** CHECKED — selected.

**Reviewer request:** Separate a fixed Nesterov coefficient from momentum warmup.

**Sources:** J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** Bound (J) permits fixed Nesterov; nanoGPT changes momentum during the run and is not covered.

**Concise candidate:** No complete network run is certified: all use finite polar solvers; nanoGPT also warms momentum and uses auxiliary Adam. The squared-gradient rate needs gradient-scaled steps. We now prove, under joint smoothness and unbiased bounded-variance noise, a constant/cosine first-moment bound for exact-polar EF21 with either downlink and fixed EMA/Nesterov. Fixed peak LR/momentum leave a residual; $T^{-3/4}$ peak LR and $1-\mu=T^{-1/2}$ give $O(T^{-1/4})$.

### R4.3 — Reviewer identifies inconsistent main and supplementary results

**Status:** CHECKED — selected.

**Reviewer request:** Reconstruct the five runs and correct the ranking.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** The original submitted five-seed archive agrees with A8. Table 2 is stale; final-five-evaluation means give 82.9424 ± .1868 vs 81.4356 ± .1466. The main-table comparison must be reversed.

**Concise candidate:** The submitted five-seed logs agree with A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%; Table 2 is stale. We will replace it and reverse the ranking sentence. SignMuon remains 85.72±0.24%, versus Muon 85.98±0.26%.

### R4.4 — Reviewer asks for useful sufficient conditions for SignMuon

**Status:** CHECKED — selected.

**Reviewer request:** Supply a structural condition and a conditional result.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The full addendum gives the weaker covariance/spread criterion epsilon*rho<1 and a complete stochastic fixed-momentum first-moment proof. The simpler diagonal lower bound in the export is sufficient and self-contained.

**Concise candidate:** For $M=U\Sigma V^\top$, positive diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ bounded below by $\delta$ imply alignment at least $\delta\|M\|_*$. This yields conditional SignMuon convergence; its margin is unverified in our networks.

### R4.15 — Reviewer asks why the favorable quadratic alignment extends to neural networks

**Status:** PENDING — excluded; needs evidence.

**Reviewer request:** Distinguish a sufficient theorem from evidence along actual training trajectories.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The sufficient margin has not been checked on these neural-network trajectories, approximate solvers or distributed votes. The favorable quadratic result does not establish it.

**Concise candidate:** The sufficient margin has not been checked on these neural-network trajectories, approximate solvers or distributed votes. The favorable quadratic result does not establish it.

### R4.6 — Reviewer questions novelty relative to EF21 and concurrent SignMuon

**Status:** CHECKED — selected.

**Reviewer request:** Credit existing frameworks and the latest concurrent theory.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows.

**Finding, meaning and limits:** The new submitted contribution is the negative sign-composition and output-EF analysis. The base EF21 theory and practical SignMuon precede this response. Mishra v2 now has conditional actual-vote stochastic momentum results, so our current v1 comparison must be replaced.

**Concise candidate:** We credit EF21 and concurrent SignMuon, including Mishra v2’s conditional theory; our main novelty is the negative placement analysis.

### R4.5 — Reviewer says nanoGPT does not test the proposed distributed compression

**Status:** CHECKED — selected.

**Reviewer request:** Make the full-precision transport explicit in the main text.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The reviewer is correct. Full-precision gradient averaging precedes the owning-rank optimizer; parameter all-gather follows. Local compress-then-vote semantics and actual one-bit communication gains are not tested by this experiment.

**Concise candidate:** nanoGPT applies these optimizers after full-precision reduce-scatter. It tests centralized updates, not compressed worker votes or network savings.

### R4.7 — Reviewer asks for definitions before theorem references

**Status:** CHECKED — selected.

**Reviewer request:** Introduce the operations before the introduction cites alignment results.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Introduce the operations before the introduction cites alignment results. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Concise candidate:** We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.

### R4.8 — Reviewer objects to broad impossibility wording

**Status:** CHECKED — selected.

**Reviewer request:** Replace “unrepairable” with the specified fixed-parameter output-EF statement.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Replace “unrepairable” with the specified fixed-parameter output-EF statement. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Concise candidate:** We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.

### R4.9 — Reviewer rejects presenting empirical ordering as a reversed theoretical prediction

**Status:** CHECKED — selected.

**Reviewer request:** The existence of failure examples does not predict method ranking.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** The existence of failure examples does not predict method ranking. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Concise candidate:** We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.

### R4.10 — Reviewer asks for a direct conclusion without repetitive caveats

**Status:** CHECKED — selected.

**Reviewer request:** Remove repeated “document rather than resolve” language.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Remove repeated “document rather than resolve” language. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Concise candidate:** We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.

### R4.11 — Reviewer asks which of X and W is updated and analyzed

**Status:** CHECKED — selected.

**Reviewer request:** Explain exact server X, compressed broadcast W and their roles.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Explain exact server X, compressed broadcast W and their roles. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Concise candidate:** We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.

### R4.12 — Reviewer asks to explain target iteration counts before using them

**Status:** CHECKED — selected.

**Reviewer request:** Distinguish finite-target success and stationary error floors from asymptotic rate.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Distinguish finite-target success and stationary error floors from asymptotic rate. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Concise candidate:** We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.

### R4.13 — Reviewer asks for a clearer statement of the Section 5.2 results

**Status:** CHECKED — selected.

**Reviewer request:** Lead with measured accuracy instead of theorem qualifications.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Lead with measured accuracy instead of theorem qualifications. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Concise candidate:** We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.

### R4.14 — Reviewer asks to move essential explanations into the main paper

**Status:** CHECKED — selected.

**Reviewer request:** Prepare a coverage table and shorten captions.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Prepare a coverage table and shorten captions. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Concise candidate:** We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.

### R5.1 — Reviewer asks what is new beyond the EF21-Muon framework

**Status:** CHECKED — selected.

**Reviewer request:** State the new proofs and credit inherited positive rates.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced.

**Finding, meaning and limits:** The submitted novelty is the negative sign-placement diagnosis and smooth bounded-below output-EF cycle. The proof separates the discontinuous polar target from the smooth gradient-side target; this is Muon/LMO-specific and not a new arbitrary-norm EF framework.

**Concise candidate:** The new submitted analysis is the explicit ascent constructions and the smooth, lower-bounded output-EF limit cycle. It shows why tracking a polar direction can fail although gradient-side EF succeeds: the polar target can jump by constant order independently of the step size. The base positive rates instantiate EF21-Muon; we will credit that directly.

### R5.3 — Reviewer asks what optimizer-design lesson the counterexamples provide

**Status:** CHECKED — selected.

**Reviewer request:** Explain what is tracked and the consequences of its variation.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** The example proves that sharing one residual scale while tracking a rapidly changing polar output can create a persistent wrong-sign component. Gradient-level EF avoids this particular tracking obstacle under smoothness; the guarantee still requires its stated assumptions.

**Concise candidate:** The new submitted analysis is the explicit ascent constructions and the smooth, lower-bounded output-EF limit cycle. It shows why tracking a polar direction can fail although gradient-side EF succeeds: the polar target can jump by constant order independently of the step size. The base positive rates instantiate EF21-Muon; we will credit that directly.

### R5.2 — Reviewer asks whether Theorem 4 includes decreasing learning rates

**Status:** CHECKED — selected.

**Reviewer request:** Write the quantifiers and identify their limit.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.

**Finding, meaning and limits:** For each fixed L, eta and mu there is an objective and either momentum variant diverges. The smooth objective may depend on those parameters. The scaled gradient floor is Omega(c_mu L eta), not a horizon-independent positive constant. No decreasing-schedule result is established for output EF.

**Concise candidate:** Theorem 4 says $\forall L,\eta>0,\mu\in[0,1),\ \exists f_{L,\eta,\mu}$ causing divergence for the specified output-EF recursion with fixed parameters. It rules out universal constant-step rules based only on $L,\mu$. It does not cover decreasing schedules or rule out horizon-tuned finite-time rates: the objective depends on the parameters, and its gradient floor scales as $L\eta$.

### R5.4 — Reviewer says parameter-dependent divergence need not preclude finite-time rates

**Status:** CHECKED — selected.

**Reviewer request:** Acknowledge the dependence of the constructed objective and its constants.

**Sources:** N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.

**Finding, meaning and limits:** The new floor calculation makes the limitation explicit. It shrinks with eta and does not defeat horizon-tuned convergence theorems whose objective gap and constants are fixed.

**Concise candidate:** Theorem 4 says $\forall L,\eta>0,\mu\in[0,1),\ \exists f_{L,\eta,\mu}$ causing divergence for the specified output-EF recursion with fixed parameters. It rules out universal constant-step rules based only on $L,\mu$. It does not cover decreasing schedules or rule out horizon-tuned finite-time rates: the objective depends on the parameters, and its gradient floor scales as $L\eta$.

### R5.5 — Reviewer rejects a theoretical prediction of the empirical ranking

**Status:** CHECKED — selected.

**Reviewer request:** Separate worst-case statements from comparative accuracy.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.

**Finding, meaning and limits:** The theory does not predict which optimizer wins on a task. The positive SignMuon theorem is conditional and compatible with the counterexamples. A new fixed softplus objective proves arbitrary-schedule nonstationarity only for each plain placement; this does not strengthen the output-EF quantifiers.

**Concise candidate:** The theory predicts no empirical ranking. Our added SignMuon analysis gives a first-moment rate under a uniform positive alignment margin, not a predicted winner. Separately, new fixed softplus witnesses prove failure for every positive step schedule and EMA/Nesterov momentum for the three plain placements; this stronger statement does not extend to output EF.

### R5.6 — Reviewer objects to presenting sufficient compressor assumptions as necessary

**Status:** CHECKED — selected.

**Reviewer request:** Retract claims that only changing compressors can help.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** Failure of spectral contractivity does not prove nonconvergence under every analysis or parameter choice. c_proj concentration is observational. No initialization/step/compressor ablation proves causation or uniqueness of a remedy.

**Concise candidate:** We will remove “unrepairable” and “only changing the compressor”: failing a sufficient condition excludes neither other analyses nor other parameter choices.

### AI.1 — Reviewer asks for the missing stochastic assumptions

**Status:** CHECKED — selected.

**Reviewer request:** State the noise assumptions where the theorem is stated.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit.

**Finding, meaning and limits:** Unbiased bounded nuclear-variance assumptions are in submitted A.10 but missing from the advertised main statement. The new proof uses conditional zero mean and a joint Frobenius variance bound.

**Concise candidate:** A.10 states unbiased bounded-variance noise; we will add it to the main statement. One max-spectral norm on all layers removes separability. Our new deterministic-schedule bound allows fixed EMA/Nesterov and either EF21 downlink, but not finite polar solvers, changing momentum or auxiliary Adam.

### AI.2 — Reviewer says the layer smoothness forces separability

**Status:** CHECKED — selected.

**Reviewer request:** Use jointly coupled smoothness instead of the original restrictive assumption.

**Sources:** J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit.

**Finding, meaning and limits:** The whole-tuple max-spectral norm and sum-nuclear dual permit arbitrary inter-layer coupling under ordinary joint Frobenius smoothness. This reduction may worsen dimension constants.

**Concise candidate:** A.10 states unbiased bounded-variance noise; we will add it to the main statement. One max-spectral norm on all layers removes separability. Our new deterministic-schedule bound allows fixed EMA/Nesterov and either EF21 downlink, but not finite polar solvers, changing momentum or auxiliary Adam.

### AI.7 — Reviewer asks for an inexact polar-oracle guarantee

**Status:** CHECKED — selected.

**Reviewer request:** Do not identify the finite polynomial solver with the exact polar theorem.

**Sources:** J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit. G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** There is no trajectory-wide oracle-error bound for the five-step solvers. The new addendum provides a conditional weighted sign-mismatch penalty, not a certificate that it vanishes in the actual experiments. Full-run theory coverage remains open.

**Concise candidate:** A.10 states unbiased bounded-variance noise; we will add it to the main statement. One max-spectral norm on all layers removes separability. Our new deterministic-schedule bound allows fixed EMA/Nesterov and either EF21 downlink, but not finite polar solvers, changing momentum or auxiliary Adam.

### AI.3 — Reviewer says the guaranteed iterate differs from the queried model

**Status:** CHECKED — selected.

**Reviewer request:** Explain X versus W without transferring the guarantee.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit.

**Finding, meaning and limits:** X takes the exact server update; W tracks it through compressed downlink; workers evaluate gradients at W. The guarantee is stationarity at X, not W.

**Concise candidate:** EF21-MuonSign losses are 5.5198 at server X and 3.3213 at broadcast W. The guarantee concerns X; success at W does not validate it.

### AI.4 — Reviewer identifies the large nanoGPT X/W loss gap

**Status:** CHECKED — selected.

**Reviewer request:** Acknowledge that the practically successful model is not the guaranteed one.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The two reported losses are evaluations of the same EF21-MuonSign run, not two independent runs. The gap is 2.1985 nats; c_proj residual concentration is an observation, not a causal initialization/compressor ablation.

**Concise candidate:** EF21-MuonSign losses are 5.5198 at server X and 3.3213 at broadcast W. The guarantee concerns X; success at W does not validate it.

### AI.5 — Reviewer asks to frame positive EF21 results as an instantiation

**Status:** CHECKED — selected.

**Reviewer request:** Identify the paper-specific contribution.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows.

**Finding, meaning and limits:** The arbitrary-norm framework and general idea of gradient-side EF before a matrix transform are prior work. The new submitted proofs concern sign/polar ascent and output-EF divergence.

**Concise candidate:** We credit EF21-Muon and Modoranu et al.; the submitted novelty is the negative sign-placement/output-EF analysis.

### AI.18 — Reviewer cites gradient-side EF before full-matrix preconditioners

**Status:** CHECKED — selected.

**Reviewer request:** Acknowledge Modoranu et al. explicitly.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced.

**Finding, meaning and limits:** Modoranu et al. 2024 establishes a precedent for EF compression before a structured preconditioner. It does not supply the sign/polar negative construction; cite both facts accurately.

**Concise candidate:** We credit EF21-Muon and Modoranu et al.; the submitted novelty is the negative sign-placement/output-EF analysis.

### AI.6 — Reviewer requests the modern SSVR sign baseline

**Status:** PENDING — excluded; needs evidence.

**Reviewer request:** Compare with Jiang et al. rather than claiming all gains arise from geometry.

**Sources:** P2: Jiang et al., NeurIPS 2024, Efficient Sign-based Optimization; code/common/optimizers.py SignSGD baseline.

**Finding, meaning and limits:** No SSVR run has been completed. Our SignSGD already uses momentum, but this does not substitute for a variance-reduced baseline; add the citation and remove exclusive causal attribution.

**Concise candidate:** No SSVR run has been completed. Our SignSGD already uses momentum, but this does not substitute for a variance-reduced baseline; add the citation and remove exclusive causal attribution.

### AI.19 — Reviewer rejects attributing gains exclusively to matrix geometry

**Status:** CHECKED — selected.

**Reviewer request:** Explain the existing momentum baseline and the missing comparison.

**Sources:** P2: code/common/optimizers.py and code/federated/algorithms.py; Jiang et al. 2024.

**Finding, meaning and limits:** The existing SignSGD baseline uses the same EMA momentum coefficient. Thus the comparison is not against memoryless gradient signs, but it does not isolate geometry against all variance-reduction choices.

**Concise candidate:** SignSGD uses momentum; SSVR is untested. We will cite Jiang et al. and remove exclusive attribution to geometry.

### AI.8 — Reviewer asks to limit broad federated claims

**Status:** CHECKED — selected.

**Reviewer request:** State the actual data and update regime.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. New heterogeneous runner exists in live repo, but no completed new result was available. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No non-IID/local-update result is completed; nanoGPT uses full-precision gradient transport and centralized optimizer semantics.

**Concise candidate:** Federation uses 11 IID clients, no local updates. nanoGPT averages full-precision gradients; no distributed one-bit gains are demonstrated.

### AI.9 — Reviewer requests a method matrix

**Status:** CHECKED — selected.

**Reviewer request:** Prepare the matrix of operations, payloads, states and guaranteed iterate.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput. W1: 02_REBUTTAL_PLAN.md method matrix.

**Finding, meaning and limits:** A concrete table is prepared internally with all six methods, EF targets, polar location, payloads, memory and theoretical iterate. It should replace the current scattered explanations.

**Concise candidate:** We will add the method/coverage table and move implementation details to the experiments.

### AI.11 — Reviewer asks to move Section 4.1 implementation details

**Status:** CHECKED — selected.

**Reviewer request:** Restore a clear sequence of definitions and theoretical results.

**Sources:** W1: 02_REBUTTAL_PLAN.md proposed presentation changes.

**Finding, meaning and limits:** Move solver and layer-scaling details from Section 4.1 to the experimental setup. Keep theory definitions before counterexamples and EF results.

**Concise candidate:** We will add the method/coverage table and move implementation details to the experiments.

### AI.10 — Reviewer asks for a structural condition explaining positive alignment

**Status:** CHECKED — selected.

**Reviewer request:** Give a sufficient condition with explicit scope.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The diagonal lower bound implies alignment by a singular-value-weighted sum. The full covariance criterion and stochastic first-moment theorem are in the author addendum. Credit OLion v2; it is not an empirical explanation for our networks.

**Concise candidate:** A sufficient SignMuon condition is $[U^\top\mathrm{sign}(UV^\top)V]_{jj}\ge\delta>0$ for $M=U\Sigma V^\top$; it implies alignment $\ge\delta\|M\|_*$.

### AI.20 — Reviewer asks to connect the structural condition to realistic training

**Status:** PENDING — excluded; needs evidence.

**Reviewer request:** Measure the margin for the actual polar approximation and vote.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The geometric criterion has not been measured on the actual trajectories. Xiao et al. 2025 concerns SignSGD and does not by itself justify the SignMuon condition.

**Concise candidate:** The geometric criterion has not been measured on the actual trajectories. Xiao et al. 2025 concerns SignSGD and does not by itself justify the SignMuon condition.

### AI.13 — Reviewer asks for the zero-input oracle selection

**Status:** CHECKED — selected.

**Reviewer request:** Define the exact thin-SVD selection and distinguish sign ties.

**Sources:** W1: thin-SVD convention in aaai_article/ef21_signmuon_divergence.tex; code/common/optimizers.py.

**Finding, meaning and limits:** Use polar(0)=0 in the exact theory. For plain strict-binary sign ties at zero, the implementation can randomize; this is distinct from the polar selection.

**Concise candidate:** We will define polar(0)=0. The 0.0021 gap is inconclusive. A5 reports seconds/epoch; we drop the broad 2x claim. Payloads include FP32 auxiliaries/EF scales. Corrected MuonSign/SGD runs reach 80% in all 5 seeds.

### AI.14 — Reviewer questions the single-seed 0.0021 nanoGPT ordering

**Status:** CHECKED — selected.

**Reviewer request:** Do not infer a reliable ranking from that difference.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The gap EF21-SignMuon 3.2860 vs SignMuon 3.2881 is below an upstream Muon noise reference, not a variant-specific noise estimate. All compressed-method ranks are preliminary.

**Concise candidate:** We will define polar(0)=0. The 0.0021 gap is inconclusive. A5 reports seconds/epoch; we drop the broad 2x claim. Payloads include FP32 auxiliaries/EF scales. Corrected MuonSign/SGD runs reach 80% in all 5 seeds.

### AI.15 — Reviewer asks for timing evidence behind the wall-clock claim

**Status:** CHECKED — selected.

**Reviewer request:** Use submitted timing data with the correct scope.

**Sources:** E4: submitted supplement Table A5, seconds/epoch (SignMuon16.6, SGD12.2, SignSGD12.5, Adam12.7); centralized export table_cifar.csv.

**Finding, meaning and limits:** Elapsed seconds/epoch are already in A5, so timing data are not entirely absent. The old main-table threshold numbers are stale. Withdraw the generic approximately 2x end-to-end speed claim and report per-epoch timing/threshold separately; compilation/validation and network transport differ across settings.

**Concise candidate:** We will define polar(0)=0. The 0.0021 gap is inconclusive. A5 reports seconds/epoch; we drop the broad 2x claim. Payloads include FP32 auxiliaries/EF scales. Corrected MuonSign/SGD runs reach 80% in all 5 seeds.

### AI.16 — Reviewer asks to count scales in one-bit claims

**Status:** CHECKED — selected.

**Reviewer request:** Include one scale per matrix and full-precision auxiliary parameters.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** CNN2 has 762560 matrix and 2146 auxiliary parameters and 3 compressed blocks. Bits/channel are 1.0870 plain and 1.0871 EF, not exactly one per total parameter; uplink-only round-trip gain is 1.93x, not 29.4x.

**Concise candidate:** We will define polar(0)=0. The 0.0021 gap is inconclusive. A5 reports seconds/epoch; we drop the broad 2x claim. Payloads include FP32 auxiliaries/EF scales. Corrected MuonSign/SGD runs reach 80% in all 5 seeds.

### AI.17 — Reviewer flags rounds-to-target conditioned on successful seeds

**Status:** CHECKED — selected.

**Reviewer request:** Correct the stale main table and report successes.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** Corrected submitted archive: MuonSign rounds80 860, SGD1240, both 5/5 successes; SignSGD1140 and SignMuon540, also5/5. Do not use the stale main-table 4/5 and3/5 denominators.

**Concise candidate:** We will define polar(0)=0. The 0.0021 gap is inconclusive. A5 reports seconds/epoch; we drop the broad 2x claim. Payloads include FP32 auxiliaries/EF scales. Corrected MuonSign/SGD runs reach 80% in all 5 seeds.

### AI.12 — Reviewer asks to scope impossibility to the analyzed EF repair

**Status:** CHECKED — selected.

**Reviewer request:** Avoid ruling out all repairs and decreasing schedules.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.

**Finding, meaning and limits:** Theorem4 is forall fixed L,eta,mu exists parameter-dependent f for the specified scaled-sign output estimator. It does not rule out horizon-tuned finite-time rates, other EF schemes or different analyses.

**Concise candidate:** “Unrepairable” will be limited to the specified output-EF counterexample; decreasing schedules and other repairs are not ruled out.
