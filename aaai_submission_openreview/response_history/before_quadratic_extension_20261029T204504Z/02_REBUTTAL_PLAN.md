# Rebuttal plan and concern-to-evidence ledger — post-submission additions revision 4

## Contribution and response logic

Lead with the supported algorithmic result: SignMuon retains Muon-level quality at a sign budget and leads the tested sign methods in the submitted vision experiments. The contribution is the controlled comparison of three sign placements and their EF variants, with theory explaining the conditions and limits of composing sign and polar operations. Use reply space for the measured contribution and completed additions. Do not claim priority for the SignMuon idea or superiority to an untested modern baseline.

Each reply answers the reviewer’s question, gives evidence, and explains its relevance. All human replies contain an active conditional higher-score request. All displayed “Re” headings are numbered. Reviewers cannot access the working article; the exported text states the answers directly. No training was run by this preparation. Newly supplied transfer results are identified separately from submitted evidence.

## Proof audit: separability

**EF21 still assumes separability in the revised manuscript.** The condition
$\|\nabla_i f(X)-\nabla_i f(Y)\|_*\le L_i\|X_i-Y_i\|_{op}$ for all tuples forces the left side to vanish whenever $X_i=Y_i$, even when other blocks differ. Hence each gradient block depends only on its own block.

`aaai_article/ef21_fixed_radius.tex` explicitly invokes this fact in the tracking proof. The Nesterov theorem retains the same assumptions. `signmuon_addendum_body.tex` labels an EF21 extension to joint Frobenius smoothness an **open target**.

**The added conditional plain SignMuon theorem allows interacting layers.** It assumes joint Frobenius smoothness, lower boundedness, conditionally unbiased finite-variance stochastic gradients and a uniform positive alignment margin. It is distinct from gradient-side EF21.

`evidence/EF21_JOINT_SMOOTHNESS.md` is a separate candidate calculation prepared by the agent. It is not a manuscript result and is not used by the revised public replies. Author mathematical review/incorporation remains outstanding. Revision 1 conflated that note with the paper; revision 2 corrects the claim everywhere in the active workspace.

## Evidence register

| Tag | Source | Supported result and limits |
| --- | --- | --- |
| E1 | Submitted A8 and original federated result archive; `verified_claims.json` | Five independent seeds, tail-five evaluation means and across-seed sample SDs. All 175 original/current histories match. Synchronize main Table 2. |
| E2 | Submitted A.11/A.14–15; federated code | 11 IID fully participating CNN2 clients, 3 fixed-weight minibatches of 64, no local parameter updates. No extrapolated ordering for non-IID/local training/partial participation. |
| E3 | Submitted Table 3/A.16–17; `code/results/nanogpt/` | One training run per method, full-precision reduce-scatter before the optimizer, parameter all-gather after. Central optimizer evidence, no one-bit LLM network measurement. |
| E4 | Submitted A5 and original `article_export.tar.gz` | Correct centralized means/SDs over 3 seeds and existing seconds/epoch. Synchronize main Table 1. |
| C1 | Communication appendix and federated code | Logical payload includes FP32 auxiliaries and EF scales; no measured packed-network throughput. |
| N1 | Submitted Theorems 1–4 and verified exact certificates | Plain placements can ascend; specified output EF can diverge on a smooth lower-bounded objective. Theorem 4 has fixed parameters and a parameter-dependent objective. |
| N2 | Author addendum | Fixed softplus witnesses for plain placements under arbitrary positive schedules; not an arbitrary-schedule EF21 impossibility theorem. |
| G1 | Author addendum and OLion geometry | Conditional plain SignMuon convergence with joint smoothness and uniform alignment. Network margin not measured; local voting is a separate algorithm. |
| T1 | Submitted A.10; revised Nesterov/fixed-radius/rate TeX | Separate second-moment sharp-step proof; fixed EMA/Nesterov deterministic-schedule first-moment bounds retain separability and exact polar. |
| E5 | Supplied transfer archive and all 50 raw histories | Dirichlet(0.1), unchanged IID rates, 11 clients/5 seeds; LR retuning pending. |
| J1 | Separate internal candidate note | Not relied on. Do not claim the manuscript removed EF21 separability. |
| P1 | EF21-Muon, Modoranu 2024, Mishra v2 | Credit general EF theory, compression before transforms and concurrent conditional SignMuon theory. Update the working manuscript’s v1-only comparison. |
| P2 | Jiang SSVR, Xiao SignSGD, current baseline code | SignSGD already uses EMA. SSVR is applicable but needs a matched gradient-query budget and implementation; no run is available. Narrow exclusive geometry attribution. |
| F1 | Existing Lean audit and paper-correspondence record | The core EF21 tracking/convergence proofs and original failure results are formally verified; the printed cosine specialization is composed manually. Conditional plain SignMuon is a separate analytic result. |
| W1 | Concrete revisions below | Proposed article wording; these edits have not been applied to author TeX by this rebuttal revision. |

## Numerical basis for the practical contribution

| Setting | SignMuon | Full-precision Muon | EMA SignSGD | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Central CIFAR-10, 3 seeds (A5) | 94.60±0.15% | 94.35±0.27% | 93.37±0.26% | 1.23-point gain over SignSGD; no claim of separation from Muon. |
| Federated CNN2, 5 seeds (A8) | 85.72±0.24% | 85.98±0.26% | 81.44±0.15% | 4.28-point gain over SignSGD at the same payload. |
| nanoGPT, 1 run/method | 3.2881 | 3.2785 | 3.4049 | 0.1168-nat observed gain over SignSGD; 0.0096 above Muon. |

The submitted A5/A8 and original logs already contain these values. Main tables are to be synchronized; the corrected MuonSign federated result is **82.94±0.19%**, above SignSGD by **1.51 points**. The correction changes that comparison, while preserving SignMuon’s leading compressed vision result. All corrected MuonSign and SGD runs reach 80% (5/5).

EF21-SignMuon’s nanoGPT loss is 3.2860, 0.0021 below SignMuon. There is no compressed-method variance estimate; no ranking is based on this difference. EF21-MuonSign’s broadcast W and server X have losses 3.3213 and 5.5198 in the same run. Gradients are queried at W; the ideal theorem concerns X. Layerwise observations do not prove the cause or a unique remedy.

## New label-skew transfer evidence supplied during the audit

The author’s addendum and `code/results/rebuttal_results.zip` were updated during the final audit. A frozen byte-identical copy is `evidence/heterogeneous_transfer_results.zip`. `verify_heterogeneous_transfer.py` independently recomputes all 50 complete 2000-round histories, verifies method/config/seed coverage and reproduces every table mean/sample SD. This is new rebuttal evidence, not part of the original submission.

All methods reuse their IID-selected learning rates and momentum 0.9. CNN2 uses 11 fully participating clients, Dirichlet(0.1) labels and no local parameter steps. The 5 paired seeds change both initialization and partition. Learning-rate retuning is pending; these results compare transfer robustness, not retuned optima.

| Method | Transfer accuracy, mean ± sample SD |
| --- | ---: |
| EF21-MuonSign | 83.23±0.36% |
| EF21-MuonUSign | 82.99±0.46% |
| SGD | 81.03±0.62% |
| Adam | 71.17±7.16% |
| Muon | 46.02±8.55% |
| EF21-SignMuon | 45.46±7.66% |
| SignMuon | 35.92±8.49% |
| SignSGD | 26.42±3.23% |
| MuonUSign | 26.21±2.75% |
| MuonSign | 25.63±3.98% |

The IID ordering changes in this transfer test: gradient-side EF is robust here while the unretuned SignMuon and Muon settings degrade. This supports a conditional practical recommendation, not a universal ranking or a claim about the pending retuned comparison. Local training and partial participation remain untested.

## Conditional SignMuon statement to explain directly

For an oracle input $H=U\Sigma V^\top$, set $Q=UV^\top$, $S=\mathrm{sign}(Q)$ and $c=\mathrm{diag}(U^\top S V)$. Then
$\langle H,S\rangle=\sum_j\sigma_j c_j$.
If every $c_j\ge\delta>0$, alignment is at least $\delta\|H\|_*$. This is a sufficient condition, not a necessary characterization. The author addendum also gives the less restrictive OLion-inspired product margin $\varepsilon(H)\rho(H)\le1-\delta$; both require a uniform margin along the actual oracle inputs, including the Nesterov combination rather than only the raw gradient.

Under lower boundedness, joint Frobenius smoothness, conditional unbiasedness/finite variance, exact polar and the uniform margin, $\eta=\eta_0T^{-3/4}$ and $1-\mu=T^{-1/2}$ give an average expected sum-of-nuclear-gradient-norm bound $O(T^{-1/4})$ for EMA or Nesterov. Parameters are fixed within a run and tuned with its horizon. With exact gradients/no momentum, $\eta\propto T^{-1/2}$ gives $O(T^{-1/2})$. This does not establish that network trajectories, approximate polar signs or local-worker votes satisfy the premise. Credit OLion’s geometry and concurrent conditional SignMuon theory.

## Theorem and implementation coverage

| Result | Steps / momentum | Smoothness / oracle | Quantity and scope |
| --- | --- | --- | --- |
| Submitted sharp EF21 rate | Gradient-scaled radius; horizon-tuned momentum | Original separable blocks; exact polar; stated unbiased noise/variance | Separate expected squared-gradient $O(T^{-1/2})$ result; not an experimental-cosine guarantee. |
| Revised ideal EF21 schedule bound | Constant/cosine; fixed EMA/Nesterov | Original separable blocks; exact polar | Weighted first moment, $O(T^{-1})$ transient plus residuals for fixed parameters; $O(T^{-1/4})$ with $\eta_{peak}\propto T^{-3/4}$ and $1-\mu=T^{-1/2}$. |
| Added conditional plain SignMuon | Horizon-tuned EMA/Nesterov | Joint smoothness; exact polar; uniform margin | Average expected gradient norm $O(T^{-1/4})$; permits interacting layers. |
| Actual nanoGPT implementation | Changing momentum, fixed practical schedule, auxiliary Adam | Finite polar approximation, full-precision collective communication | Measures optimizer quality. No complete implementation guarantee or one-bit network measurement. |

A first-moment bound cannot be squared into an expected-square bound. Fixed schedule residuals are not a squared-gradient rate or a certificate for every network update. The paper’s EF21 guarantees concern server X, including with a compressed downlink.

## Complete method / systems comparison

| Method | Sign/EF target | Polar location | Matrix uplink/downlink | Extra persistent matrix state |
| --- | --- | --- | --- | --- |
| SignMuon | Sign after polar; worker votes | Clients | 1 / 1 bit per entry | None beyond common momentum/model |
| MuonUSign | Sign before polar; server vote | Server | 1 / 32 | None |
| MuonSign | Sign before and after polar | Server | 1 / 1 | None |
| EF21-SignMuon | EF tracks polar output | Clients | 1+scale / 32 | Client estimator and server aggregate |
| EF21-MuonUSign | EF tracks gradient/momentum input | Server | 1+scale / 32 | Client estimator and server aggregate |
| EF21-MuonSign | Input EF and model-downlink EF | Server | 1+scale / 1+scale | Estimators plus exact X separate from broadcast W |

CNN2 has 762560 matrix and 2146 auxiliary parameters in 3 matrix blocks. FP32 auxiliaries yield 1.0870 bits/total parameter per compressed channel; EF adds one FP32 scale/block, giving 1.0871. Bidirectional payload reduction is about 29.4× versus FP32; uplink-only round-trip reduction is about 1.93×. A matrix-size FP32 buffer is 3,050,240 bytes (2.91 MiB); common momentum/Adam and polar scratch space are excluded. These are logical simulation payloads, not measured transport throughput.

Practical recommendation: choose SignMuon for its leading observed compressed accuracy in the submitted IID vision experiments. Under label-skew transfer, the gradient-EF variants are stronger with the unchanged IID rates; retuning is pending. If an analyzed gradient-side repair is required, choose EF21-MuonUSign for an uplink constraint or EF21-MuonSign for both links; monitor X/W and account for estimator memory. The same mathematical and implementation qualifications in the coverage table apply.

## Concrete article revisions

- Lead the introduction and results with measured SignMuon quality and the comparison being made. State the controlled comparative contribution and credit prior/concurrent method ideas.
- Synchronize both main vision tables to submitted A5/A8 and the original logs. Reverse only the affected MuonSign-versus-SignSGD comparison. Define mean/sample SD and independent seeds.
- State sign as the entrywise infinity-ball LMO in Section 2. Distinguish plain sign bits from scaled-sign EF messages and FP32 auxiliaries when first introduced.
- Add the three operation-order formulas/legend and centralized/distributed labels. A2/A4/A5 correspond to one worker; A8–A9 put signs/polar factors before aggregation. Voting does not commute with averaging or polar.
- Put the coverage table in the main text. Define X as the exact server model and W as the compressed broadcast model in separate sentences. State where gradients are queried and where the theorem measures stationarity.
- Replace the invalid first-to-second-moment inference with the separate arguments and parameter choices. Explicitly retain EF21 separability; distinguish the joint-smooth conditional SignMuon theorem.
- State Theorem 4 as $\forall L,\eta>0,\mu\in[0,1),\exists f_{L,\eta,\mu}$ with fixed parameters. It does not cover decreasing schedules or exclude horizon-tuned finite-time rates. Replace blanket “unrepairable” and “only changing the compressor” claims with this exact scope and sufficient-condition statements.
- Explain that the linear ascent witnesses test descent outside lower boundedness; $f\to-\infty$ is literal there. Added softplus witnesses establish smooth lower-bounded failures for the three plain placements under arbitrary positive schedules, separately from output EF.
- Explain nanoGPT’s full-precision reduce-scatter and all-gather in the main text. One run/method supports the stated observed loss comparisons; additional compressed-method seeds remain outstanding.
- Shorten the introduction, results and captions, define gradient/update alignment before theorem references, move solver details to setup, and explain target-crossing counts without calling them asymptotic rates.
- Cite A5’s measured epoch times (SignMuon 16.6 s, Muon 14.4 s, SignSGD 12.5 s), and replace the broad 2× claim with its measured scope. This is not a compressed-network timing experiment.
- Credit EF21-Muon, Modoranu, OLion and the current Mishra v2 theory accurately. Explain that EMA SignSGD is a matched baseline; a fair SSVR experiment remains outstanding, so avoid universal sign-method superiority or exclusive geometry attribution.

## How the rewritten answers explain the theory

Each answer names the method, states what was proved, and explains why it answers the concern. “Our contribution” replaces bare headings. The replies no longer refer vaguely to “new proofs.”

- **Failure results:** signing before, after or on both sides of orthogonalization can produce an ascent direction. Error feedback on the orthogonalized output can still diverge on a smooth, lower-bounded objective. These results explain why the placement of correction matters.
- **Generalized EF21 guarantees:** EMA and Nesterov momentum with deterministic learning rates are covered by tracking/convergence bounds. These expose stepsize, gradient-noise and numerical-direction errors. The core arguments are verified in Lean. Fixed momentum and layer separability remain assumptions; an actual numerical implementation needs an error or quality bound.
- **Conditional SignMuon guarantee:** signing must retain a fixed positive fraction of the original update’s alignment with momentum. Under the stated smoothness/noise assumptions and suitable parameters, the expected gradient norm vanishes. The theorem permits interacting layers. It is not claimed Lean-verified and its alignment condition is not yet measured in the network experiments.

The 2026-10-23 audit passed 45 frozen statement comparisons and kernel replay. The pinned dependency and reference hashes still match. `PAPER_AUDIT.md` distinguishes the general formalized arguments from printed cosine sums and other manually composed specializations. We did not rerun the Lean build during this prose edit.

## Post-submission additions and shared results table

The replies explicitly distinguish the submitted results from completed additions made after submission or in response to the reviews. The submitted sign-placement failure theorems remain identified as submitted results. The EF21 momentum/schedule extensions and conditional plain SignMuon theorem are described as additions. Lean claims retain the scope documented in `evidence/revision_3_readability.md`; the existing 45-statement audit was not rerun for this writing edit.

The complete heterogeneous comparison appears once, in **7S1b, Question 2**. It reports all 10 methods from the same frozen 50-run archive, including full-precision Muon, SGD and Adam. Every other review and the AC comment refer to that exact reviewer/question, without external links. The headline is that gradient-side EF leads the tested heterogeneous transfer regime, while SignMuon is strong on the submitted IID/centralized tasks. This is practical guidance on compression placement; IID-selected rates, no local updates and pending retuning are explicit. The table does not rank separately retuned heterogeneous optima.

Credit-only sentences are omitted from the exports as requested by the author. The source record and references remain intact, and the replies make no priority claim for the SignMuon idea or the general EF21 framework.

## Per-concern ledger

### R1.1 — Reviewer asks for gradient structure that permits SignMuon descent

**Status:** Checked statement; selected.

**Reviewer request:** Explain a sufficient condition beyond the random-quadratic observation.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The diagonal-correlation condition implies alignment by weighting each diagonal entry by its nonnegative singular value. Uniform alignment, joint smoothness, centered finite-variance noise, exact polar and horizon-tuned fixed EMA/Nesterov yield an expected first-moment nuclear-gradient rate O(T^-1/4). It is a conditional result; actual trajectory verification and distributed-vote alignment remain open.

**Current answer:** Since submission, we proved SignMuon converges under smoothness, unbiased bounded-variance noise and sufficient alignment of signed updates with momentum. It allows interacting layers; network tests remain open.

### R1.6 — Reviewer requests neural-network alignment and spectral measurements

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Measure the proposed condition along actual network trajectories.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No neural-network measurement of the full geometric margin is available. Proposed diagnostics must record momentum/gradient alignment, diagonal correlations and spectral spread for the actual approximate solver and vote.

**Current answer:** No neural-network measurement of the full geometric margin is available. Proposed diagnostics must record momentum/gradient alignment, diagonal correlations and spectral spread for the actual approximate solver and vote.

### R1.2 — Reviewer asks whether federation changes preserve the ordering

**Status:** Checked statement; selected.

**Reviewer request:** State what the 11-client experiment supports.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR-retuned comparison, local updates and partial participation remain outstanding.

**Finding, meaning and limits:** New supplied transfer evidence changes the empirical ordering under label skew: EF21-MuonSign 83.23±0.36%, EF21-MuonUSign 82.99±0.46%, SignMuon 35.92±8.49% with each method’s IID learning rate (five seeds). This tests hyperparameter transfer, not retuned best performance. No local updates or partial participation were tested. Revision 4 exports all 10 transfer results in 7S1b Q2, ordered by mean accuracy, with sample SD, 5 seeds, 11 clients, Dirichlet(0.1), 2000 rounds, no local updates and IID-selected rates. Gradient-side EF leads this transfer comparison. Retuned optima and local-update behavior are not established.

**Current answer:** Following your review, we tested CNN2/CIFAR-10: 11 clients, Dirichlet(0.1), 2000 rounds, no local updates, IID-selected rates. Mean±SD, 5 seeds:

|Method|Accuracy (%)|
|---|---:|
|EF21-MuonSign|83.23±0.36|
|EF21-MuonUSign|82.99±0.46|
|SGD|81.03±0.62|
|Adam|71.17±7.16|
|Muon|46.02±8.55|
|EF21-SignMuon|45.46±7.66|
|SignMuon|35.92±8.49|
|SignSGD|26.42±3.23|
|MuonUSign|26.21±2.75|
|MuonSign|25.63±3.98|

Gradient-side error feedback leads. Retuning is pending.

### R1.7 — Reviewer requests heterogeneous and local-training experiments

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Broaden the federated evaluation beyond synchronized homogeneous clients.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR-retuned comparison, local updates and partial participation remain outstanding.

**Finding, meaning and limits:** Five-seed Dirichlet(0.1) transfer is complete and independently recomputed. LR retuning, local model updates, partial participation and larger client populations remain outstanding. A partial answer does not complete the entire broader-federation request. Revision 4 exports all 10 transfer results in 7S1b Q2, ordered by mean accuracy, with sample SD, 5 seeds, 11 clients, Dirichlet(0.1), 2000 rounds, no local updates and IID-selected rates. Gradient-side EF leads this transfer comparison. Retuned optima and local-update behavior are not established.

**Current answer:** Five-seed non-IID transfer is complete. LR retuning, local updates, partial participation and larger client populations remain outstanding.

### R1.3 — Reviewer asks which positive results are inherited from EF21

**Status:** Checked statement; selected.

**Reviewer request:** Credit the existing arbitrary-norm framework and isolate extensions.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The base positive EF21 rates instantiate prior work. The revised manuscript adds fixed EMA/Nesterov and deterministic constant/cosine bounds under its original block-separable assumptions. It does not remove EF21 layer separability.

**Current answer:** In response to your review, we extended EF21 to fixed EMA/Nesterov and deterministic rates, retaining separable layers. Its core proofs are verified in Lean.

### R1.9 — Reviewer asks what technical contribution is new

**Status:** Checked statement; selected.

**Reviewer request:** Distinguish the negative construction from positive framework instantiation.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The contribution is the controlled comparison of three sign placements and their EF variants, demonstrating Muon-level quality at a sign budget, plus placement-specific ascent and output-EF divergence proofs. The author addendum adds conditional SignMuon analysis with joint smoothness; the general EF21 framework and practical SignMuon idea are credited to prior/concurrent work.

**Current answer:** Our submitted proofs show 3 sign placements can ascend and output EF can diverge.

### R1.4 — Reviewer asks how to choose a communication variant

**Status:** Checked statement; selected.

**Reviewer request:** Recommend choices with explicit evidence limits.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput. E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** Under an uplink constraint EF21-MuonUSign retains an exact/full-precision downlink; under a bidirectional constraint EF21-MuonSign compresses both sides. SignMuon is the empirical candidate if conditional rather than general guarantees are acceptable. No universal ranking is predicted.

**Current answer:** Use SignMuon for IID data (85.72% vs Muon 85.98%); under heterogeneity, use EF21-MuonUSign for uplink-only compression or EF21-MuonSign for both links.

### R1.10 — Reviewer requests scales and memory in a comparison table

**Status:** Checked statement; selected.

**Reviewer request:** Account for payloads, precision, estimators and polar location.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** The internal method table is prepared in 02_REBUTTAL_PLAN.md. A compressed CNN2 channel costs 1.0870 bits/parameter for plain signs, 1.0871 for EF with FP32 auxiliary parameters and one FP32 scale per matrix. Each gradient EF method adds one client estimator and one global server estimator; two-sided EF additionally stores exact X separately from W.

**Current answer:** CNN2 uses 1.087 bits/parameter including auxiliaries; EF adds scales and buffers.

### R1.5 — Reviewer asks which nanoGPT differences are reliable

**Status:** Checked statement; selected.

**Reviewer request:** Separate descriptive gaps from statistical evidence.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** One nanoGPT training run per method gives SignMuon 3.2881 versus SignSGD 3.4049, an observed 0.1168-nat improvement. The 0.0021 gap to EF21-SignMuon does not establish an ordering; there is no compressed-method variance estimate. Upstream Muon variance cannot substitute.

**Current answer:** SignMuon/SignSGD losses are 3.2881/3.4049 (1 run each). A 0.0021 gap to EF21-SignMuon cannot establish a ranking.

### R1.8 — Reviewer requests additional language-model seeds

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Repeat training rather than repeated validation of one run.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No additional independent compressed-method training runs are available. Repeated evaluations and upstream Muon runs cannot substitute for them.

**Current answer:** No additional independent compressed-method training runs are available. Repeated evaluations and upstream Muon runs cannot substitute for them.

### R2.1 — Reviewer asks to identify sign as an LMO earlier

**Status:** Checked statement; selected.

**Reviewer request:** Define the geometry in Section 2.

**Sources:** W1: submitted Section 2 and current signmuon_body.tex Section 4.1.

**Finding, meaning and limits:** We will state in Section 2 that $-\mathrm{sign}(G)$ minimizes $\langle G,D\rangle$ over $\|D\|_\infty\le1$.

**Current answer:** We will introduce sign as the LMO for the entrywise infinity-norm ball.

### R2.4 — Reviewer asks to distinguish raw signs from scaled EF messages

**Status:** Checked statement; selected.

**Reviewer request:** Explain the EF scale when the variants are first introduced.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** Plain signs send bits only; EF sends the residual signs plus its mean absolute value per matrix. FP32 auxiliary parameters are also counted.

**Current answer:** We will distinguish sign bits from EF messages with 1 scale per matrix and count FP32 auxiliaries.

### R2.5 — Reviewer asks to distinguish centralized formulas from distributed algorithms

**Status:** Checked statement; selected.

**Reviewer request:** Explain the N=1 correspondence and nonlinear aggregation.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR-retuned comparison, local updates and partial participation remain outstanding. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** Algorithms A2/A4/A5 define centralized placements. At N=1 the distributed definitions reduce to them; at N>1 local sign/LMO and majority voting define different algorithms. We will label both settings explicitly.

**Current answer:** We will separate single-worker formulas from worker voting: applying signs or orthogonalization before averaging changes the update.

### R2.6 — Reviewer questions the linear objective under lower boundedness

**Status:** Checked statement; selected.

**Reviewer request:** Explain why the linear tests do not satisfy Assumption 1.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The linear examples test descent outside Assumption 1; $f\to-\infty$ is literal there. We also proved lower-bounded replacements: $f(X)=\log(1+e^{\langle G,X\rangle})$ retains the plain placements’ ascent directions for every positive step schedule and EMA/Nesterov momentum.

**Current answer:** The linear examples test descent outside Assumption 1, so uncompressed Muon has $f\to-\infty$ there. Since submission, we verified the original ascent results in Lean and added smooth, lower-bounded examples for all 3 plain sign placements.

### R2.7 — Reviewer suggests consistent names for sign placements

**Status:** Checked statement; selected.

**Reviewer request:** Make the composition order readable without changing identities.

**Sources:** W1: current signmuon_body.tex equations (5)–(6); method table in 02_REBUTTAL_PLAN.md.

**Finding, meaning and limits:** We will retain labels to keep tables/code traceable and add an operation-order legend: SignMuon = sign(polar(M)); MuonUSign = polar(sign(M)); MuonSign = sign(polar(sign(M))).

**Current answer:** We will retain method names for consistency with code and add an operation-order legend.

### R2.2 — Reviewer suggests larger centralized and federated experiments

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Additional models and truly distributed language-model training would strengthen the evidence.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR-retuned comparison, local updates and partial participation remain outstanding. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No new large-model or distributed-language-model run was completed. nanoGPT evaluates the centralized optimizer after full-precision averaging, so it cannot fill this request.

**Current answer:** No new large-model or distributed-language-model run was completed. nanoGPT evaluates the centralized optimizer after full-precision averaging, so it cannot fill this request.

### R2.3 — Reviewer asks for progress on the theory-practice explanation

**Status:** Checked statement; selected.

**Reviewer request:** Give a positive condition without claiming a complete explanation.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** With joint smoothness, unbiased finite-variance noise, exact polar and horizon-tuned steps/momentum, we prove SignMuon convergence if $\langle M,\mathrm{sign}(\mathrm{polar}(M))\rangle\ge\delta\|M\|_*$ uniformly. It suffices that all diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ exceed $\delta>0$ for $M=U\Sigma V^\top$. Verification in our networks remains open.

**Current answer:** In response to your review, we now prove an $O(T^{-1/4})$ bound on SignMuon’s expected gradient norm when signing retains a fixed positive fraction of the orthogonalized update’s alignment with momentum. The theorem allows interacting layers and unbiased bounded-variance noise, with parameters chosen for $T$ iterations. The condition remains unmeasured in networks. Submitted ResNet18/CIFAR-10 accuracies are 94.60% for SignMuon, 94.35% for Muon and 93.37% for SignSGD (3 seeds). Added heterogeneous tests favor gradient-side EF; see the table in our reply to 7S1b, Q2. This explains when the choice of compression placement matters.

### R2.8 — Reviewer wants a realistic explanation of observed success

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Check the proposed sufficient condition on trained networks.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The conditional proof supplies a mathematical intermediate result. It does not establish that network momentum, approximate polar factors or distributed votes satisfy it.

**Current answer:** The conditional proof supplies a mathematical intermediate result. It does not establish that network momentum, approximate polar factors or distributed votes satisfy it.

### R3.2 — Reviewer identifies the reversed MuonSign versus SignSGD ranking

**Status:** Checked statement; selected.

**Reviewer request:** Correct the main table and every affected comparison.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** Table 2 is stale. Five independent submitted runs, each averaged over the final five evaluations at the 2000-round horizon, give MuonSign 82.9424 ± 0.1868 and SignSGD 81.4356 ± 0.1466 sample SD. These agree with A8, reversing the main-table sentence. SignMuon 85.7156 ± .2444 versus Muon 85.9836 ± .2608; no significance claim.

**Current answer:** The 5 submitted runs reproduce A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%. SignMuon reaches 85.72±0.24% versus Muon 85.98±0.26%. We will correct Table 2 and its reversed comparison. The heterogeneous results added after submission favor gradient-side EF; see the table in our reply to 7S1b, Q2.

### R3.1 — Reviewer asks which learning-rate rule proves the squared-gradient rate

**Status:** Checked statement; selected.

**Reviewer request:** Do not transfer the sharp-stepsize result to cosine schedules.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The squared dual-gradient rate O(T^-1/2) has a separate second-moment argument with gradient-scaled radii and horizon-tuned momentum. It does not follow for the cosine settings used in the experiments.

**Current answer:** In response to your review, we extended EF21 convergence to fixed EMA and Nesterov momentum with deterministic learning rates, including constant and cosine schedules. The underlying tracking/convergence proofs are verified in Lean. These bounds account for noise and orthogonalization error but retain separable layers. Our added SignMuon theorem permits interacting layers when signing preserves enough alignment with momentum. The numerical solver needs an error bound; nanoGPT’s changing momentum and auxiliary Adam updates need further analysis.

### R3.3 — Reviewer identifies the separability forced by layer-wise smoothness

**Status:** Checked statement; selected.

**Reviewer request:** Replace the restriction with an explicit whole-model formulation.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies.

**Finding, meaning and limits:** The manuscript’s block-only smoothness forces separability, and the revised EF21 proofs retain it. The author addendum explicitly calls EF21 cross-layer coupling an open target. Only the added conditional plain SignMuon theorem uses joint Frobenius smoothness and permits interacting layers. The separate J1 candidate must not be described as a completed manuscript extension.

**Current answer:** In response to your review, we extended EF21 convergence to fixed EMA and Nesterov momentum with deterministic learning rates, including constant and cosine schedules. The underlying tracking/convergence proofs are verified in Lean. These bounds account for noise and orthogonalization error but retain separable layers. Our added SignMuon theorem permits interacting layers when signing preserves enough alignment with momentum. The numerical solver needs an error bound; nanoGPT’s changing momentum and auxiliary Adam updates need further analysis.

### R3.4 — Reviewer rejects squaring an expected first-moment bound

**Status:** Checked statement; selected.

**Reviewer request:** Remove the unsupported inference and state two distinct results.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies.

**Finding, meaning and limits:** A first-moment gradient bound cannot be squared to obtain a second-moment bound. The squared O(T^-1/2) result has a separate proof and gradient-scaled steps. Under the original separable assumptions, ideal fixed-momentum EF21 has a deterministic-schedule first-moment bound: a transient plus residuals, or O(T^-1/4) with eta_peak proportional to T^-3/4 and 1-mu=T^-1/2.

**Current answer:** We will remove the claim that an expected-norm bound can be squared. The $O(T^{-1/2})$ expected squared-gradient result has a separate proof and gradient-dependent steps. Our post-submission bounds instead control the expected gradient norm for constant or cosine schedules; suitable learning rate and momentum choices for the iteration budget make the exact-update bound vanish.

### R3.5 — Reviewer asks to distinguish ideal EF21 from the implementations

**Status:** Checked statement; selected.

**Reviewer request:** State the remaining oracle and optimizer qualifications.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Lean’s fixed-momentum trajectory theorems allow approximate directions through explicit additive error or positive oracle-gain assumptions. This is not a certificate of a sufficiently small error for the actual finite polynomial implementation. The actual nanoGPT momentum changes and auxiliary Adam updates still need additional analysis. The multiple-block EF21 result retains separability.

**Current answer:** In response to your review, we extended EF21 convergence to fixed EMA and Nesterov momentum with deterministic learning rates, including constant and cosine schedules. The underlying tracking/convergence proofs are verified in Lean. These bounds account for noise and orthogonalization error but retain separable layers. Our added SignMuon theorem permits interacting layers when signing preserves enough alignment with momentum. The numerical solver needs an error bound; nanoGPT’s changing momentum and auxiliary Adam updates need further analysis.

### R4.1 — Reviewer asks whether the experimental schedules have the proved rate

**Status:** Checked statement; selected.

**Reviewer request:** Identify the distinction between squared-gradient steps and deterministic schedules.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** We generalized the EF21 analysis to EMA/Nesterov and deterministic learning rates. Its tracking and convergence arguments are formalized in Lean. Constant/cosine corollaries follow by schedule specialization, whose printed cosine sum is not itself a complete exported Lean theorem. Approximate directions are admitted through explicit error/quality assumptions. The squared-gradient rate still needs gradient-dependent steps and the manuscript retains separability.

**Current answer:** In response to your review, we generalized EF21 convergence to fixed EMA and Nesterov momentum with deterministic learning rates, including constant and cosine schedules, and verified the core proofs in Lean. The bounds retain separable layers; numerical orthogonalization needs an error bound. nanoGPT varies momentum, and the squared-gradient rate needs gradient-dependent steps.

### R4.2 — Reviewer asks whether nanoGPT Nesterov momentum is covered

**Status:** Checked statement; selected.

**Reviewer request:** Separate a fixed Nesterov coefficient from momentum warmup.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The generalized analysis and Lean trajectory proofs cover a fixed momentum coefficient in EMA or Nesterov. nanoGPT changes that coefficient during training, so the implementation is not certified by the fixed-momentum result.

**Current answer:** In response to your review, we generalized EF21 convergence to fixed EMA and Nesterov momentum with deterministic learning rates, including constant and cosine schedules, and verified the core proofs in Lean. The bounds retain separable layers; numerical orthogonalization needs an error bound. nanoGPT varies momentum, and the squared-gradient rate needs gradient-dependent steps.

### R4.3 — Reviewer identifies inconsistent main and supplementary results

**Status:** Checked statement; selected.

**Reviewer request:** Reconstruct the five runs and correct the ranking.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** The original submitted five-seed archive agrees with A8. Table 2 is stale; final-five-evaluation means give 82.9424 ± .1868 vs 81.4356 ± .1466. The main-table comparison must be reversed.

**Current answer:** The submitted 5-seed logs give MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8. We will correct Table 2 and the reversed comparison. SignMuon reaches 85.72% versus Muon 85.98%.

### R4.4 — Reviewer asks for useful sufficient conditions for SignMuon

**Status:** Checked statement; selected.

**Reviewer request:** Supply a structural condition and a conditional result.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The full addendum gives the weaker covariance/spread criterion epsilon*rho<1 and a complete stochastic fixed-momentum first-moment proof. The simpler diagonal lower bound in the export is sufficient and self-contained.

**Current answer:** After submission, we proved SignMuon converges under smoothness and unbiased bounded-variance noise when signing preserves enough alignment with momentum. This permits interacting layers; the condition is unmeasured in networks. We also added heterogeneous tests: gradient-side EF reaches 83.23% at IID-selected rates (table in our reply to 7S1b, Q2). These results explain why the compression placement matters.

### R4.15 — Reviewer asks why the favorable quadratic alignment extends to neural networks

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Distinguish a sufficient theorem from evidence along actual training trajectories.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The sufficient margin has not been checked on these neural-network trajectories, approximate solvers or distributed votes. The favorable quadratic result does not establish it.

**Current answer:** The sufficient margin has not been checked on these neural-network trajectories, approximate solvers or distributed votes. The favorable quadratic result does not establish it.

### R4.6 — Reviewer questions novelty relative to EF21 and concurrent SignMuon

**Status:** Checked statement; selected.

**Reviewer request:** Credit existing frameworks and the latest concurrent theory.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Practical SignMuon and the general EF21 framework are prior/concurrent work. Our supported contribution is the controlled comparison of six placements/EF variants, strong sign-based performance and distinct sign/polar failure mechanisms. Mishra v2 has conditional actual-vote stochastic momentum theory; the current manuscript’s v1-only comparison must be updated. The added conditional SignMuon theorem is new to this submission, not claimed as the first conditional SignMuon result.

**Current answer:** After submission, we proved SignMuon converges under smoothness and unbiased bounded-variance noise when signing preserves enough alignment with momentum. This permits interacting layers; the condition is unmeasured in networks. We also added heterogeneous tests: gradient-side EF reaches 83.23% at IID-selected rates (table in our reply to 7S1b, Q2). These results explain why the compression placement matters.

### R4.5 — Reviewer says nanoGPT does not test the proposed distributed compression

**Status:** Checked statement; selected.

**Reviewer request:** Make the full-precision transport explicit in the main text.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The reviewer is correct. Full-precision gradient averaging precedes the owning-rank optimizer; parameter all-gather follows. Local compress-then-vote semantics and actual one-bit communication gains are not tested by this experiment.

**Current answer:** nanoGPT tests centralized updates after full-precision averaging: SignMuon loss 3.2881 versus SignSGD 3.4049. CNN2 tests worker voting.

### R4.7 — Reviewer asks for definitions before theorem references

**Status:** Checked statement; selected.

**Reviewer request:** Introduce the operations before the introduction cites alignment results.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Introduce the operations before the introduction cites alignment results. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

### R4.8 — Reviewer objects to broad impossibility wording

**Status:** Checked statement; selected.

**Reviewer request:** Replace “unrepairable” with the specified fixed-parameter output-EF statement.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Replace “unrepairable” with the specified fixed-parameter output-EF statement. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

### R4.9 — Reviewer rejects presenting empirical ordering as a reversed theoretical prediction

**Status:** Checked statement; selected.

**Reviewer request:** The existence of failure examples does not predict method ranking.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** The existence of failure examples does not predict method ranking. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

### R4.10 — Reviewer asks for a direct conclusion without repetitive caveats

**Status:** Checked statement; selected.

**Reviewer request:** Remove repeated “document rather than resolve” language.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Remove repeated “document rather than resolve” language. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

### R4.11 — Reviewer asks which of X and W is updated and analyzed

**Status:** Checked statement; selected.

**Reviewer request:** Explain exact server X, compressed broadcast W and their roles.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Explain exact server X, compressed broadcast W and their roles. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

### R4.12 — Reviewer asks to explain target iteration counts before using them

**Status:** Checked statement; selected.

**Reviewer request:** Distinguish finite-target success and stationary error floors from asymptotic rate.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Distinguish finite-target success and stationary error floors from asymptotic rate. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

### R4.13 — Reviewer asks for a clearer statement of the Section 5.2 results

**Status:** Checked statement; selected.

**Reviewer request:** Lead with measured accuracy instead of theorem qualifications.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Lead with measured accuracy instead of theorem qualifications. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

### R4.14 — Reviewer asks to move essential explanations into the main paper

**Status:** Checked statement; selected.

**Reviewer request:** Prepare a coverage table and shorten captions.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Prepare a coverage table and shorten captions. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

### R5.1 — Reviewer asks what is new beyond the EF21-Muon framework

**Status:** Checked statement; selected.

**Reviewer request:** State the new proofs and credit inherited positive rates.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The contribution combines strong measured sign-based optimization with a systematic comparison of three sign placements and their EF variants, and explicit placement-specific ascent/output-EF divergence proofs. In the submitted IID CNN2 study, at the same payload SignMuon gains 4.28 points over momentum SignSGD and is 0.27 points below Muon. This is evidence among tested methods, not a universal ranking or priority claim for the SignMuon idea.

**Current answer:** SignMuon achieves 85.72% at 1 bit per matrix entry, versus Muon 85.98% and SignSGD 81.44% (5 submitted seeds). Our controlled comparison of sign and error-feedback placements explains why the choice matters. In response to the reviews, we added heterogeneous tests: gradient-side EF reaches 83.23%, versus SignMuon 35.92% at IID-selected rates (table in our reply to 7S1b, Q2). Our submitted theorems prove all 3 sign placements can ascend and output-level EF can diverge. After submission, we extended EF21 to fixed EMA/Nesterov and deterministic rates, retaining separable layers, with core proofs verified in Lean.

### R5.3 — Reviewer asks what optimizer-design lesson the counterexamples provide

**Status:** Checked statement; selected.

**Reviewer request:** Explain what is tracked and the consequences of its variation.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The example proves that sharing one residual scale while tracking a rapidly changing polar output can create a persistent wrong-sign component. Gradient-level EF avoids this particular tracking obstacle under smoothness; the guarantee still requires its stated assumptions.

**Current answer:** SignMuon achieves 85.72% at 1 bit per matrix entry, versus Muon 85.98% and SignSGD 81.44% (5 submitted seeds). Our controlled comparison of sign and error-feedback placements explains why the choice matters. In response to the reviews, we added heterogeneous tests: gradient-side EF reaches 83.23%, versus SignMuon 35.92% at IID-selected rates (table in our reply to 7S1b, Q2). Our submitted theorems prove all 3 sign placements can ascend and output-level EF can diverge. After submission, we extended EF21 to fixed EMA/Nesterov and deterministic rates, retaining separable layers, with core proofs verified in Lean.

### R5.2 — Reviewer asks whether Theorem 4 includes decreasing learning rates

**Status:** Checked statement; selected.

**Reviewer request:** Write the quantifiers and identify their limit.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** For each fixed L, eta and mu there is an objective and either momentum variant diverges. The smooth objective may depend on those parameters. The scaled gradient floor is Omega(c_mu L eta), not a horizon-independent positive constant. No decreasing-schedule result is established for output EF.

**Current answer:** For any fixed learning rate and momentum, Theorem 4 constructs a smooth, lower-bounded objective where EF21-SignMuon diverges. The objective can depend on these parameters. This rules out universal constant-step rules based on smoothness and momentum; it does not exclude decreasing steps or parameters chosen for a finite budget.

### R5.4 — Reviewer says parameter-dependent divergence need not preclude finite-time rates

**Status:** Checked statement; selected.

**Reviewer request:** Acknowledge the dependence of the constructed objective and its constants.

**Sources:** N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The new floor calculation makes the limitation explicit. It shrinks with eta and does not defeat horizon-tuned convergence theorems whose objective gap and constants are fixed.

**Current answer:** For any fixed learning rate and momentum, Theorem 4 constructs a smooth, lower-bounded objective where EF21-SignMuon diverges. The objective can depend on these parameters. This rules out universal constant-step rules based on smoothness and momentum; it does not exclude decreasing steps or parameters chosen for a finite budget.

### R5.5 — Reviewer rejects a theoretical prediction of the empirical ranking

**Status:** Checked statement; selected.

**Reviewer request:** Separate worst-case statements from comparative accuracy.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.

**Finding, meaning and limits:** The theory does not predict which optimizer wins on a task. The positive SignMuon theorem is conditional and compatible with the counterexamples. A new fixed softplus objective proves arbitrary-schedule nonstationarity only for each plain placement; this does not strengthen the output-EF quantifiers.

**Current answer:** The theorems do not predict an accuracy ranking. Our new heterogeneous results favor EF21-MuonUSign for uplink-only compression and EF21-MuonSign for both links; IID results favor SignMuon.

### R5.6 — Reviewer objects to presenting sufficient compressor assumptions as necessary

**Status:** Checked statement; selected.

**Reviewer request:** Retract claims that only changing compressors can help.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** Failure of spectral contractivity does not prove nonconvergence under every analysis or parameter choice. c_proj concentration is observational. No initialization/step/compressor ablation proves causation or uniqueness of a remedy.

**Current answer:** We will remove blanket “unrepairable” and “only changing the compressor” claims: sufficient conditions need not be necessary.

### AI.1 — Reviewer asks for the missing stochastic assumptions

**Status:** Checked statement; selected.

**Reviewer request:** State the noise assumptions where the theorem is stated.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Unbiased bounded-variance assumptions are in submitted A.10 and must also appear in the main statement. Revised EF21 bounds retain the original separable smoothness; the added plain SignMuon theorem instead uses joint Frobenius smoothness and conditional centering/finite variance. The two results must be distinguished.

**Current answer:** After submission, we extended EF21 to fixed EMA/Nesterov and deterministic learning rates, with core proofs verified in Lean. The bounds retain separable layers and include unbiased bounded-variance noise and orthogonalization error.

### AI.2 — Reviewer says the layer smoothness forces separability

**Status:** Checked statement; selected.

**Reviewer request:** Use jointly coupled smoothness instead of the original restrictive assumption.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The manuscript does not remove EF21 separability. Its fixed-radius proof explicitly uses block independence, and the author addendum marks a joint-smooth EF21 extension open. The added conditional SignMuon theorem allows interacting layers. J1 is a separate candidate derivation and is excluded from public claims.

**Current answer:** After submission, we extended EF21 to fixed EMA/Nesterov and deterministic learning rates, with core proofs verified in Lean. The bounds retain separable layers and include unbiased bounded-variance noise and orthogonalization error.

### AI.7 — Reviewer asks for an inexact polar-oracle guarantee

**Status:** Checked statement; selected.

**Reviewer request:** Do not identify the finite polynomial solver with the exact polar theorem.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Lean’s schedule/trajectory results include approximation-quality or error terms rather than requiring every direction to equal an exact polar factor. No trajectory-wide small-error certificate is available for the actual five-stage implementation, and no full nanoGPT theorem covers changing momentum plus auxiliary Adam. State the proved generalization and the remaining assumptions separately.

**Current answer:** After submission, we extended EF21 to fixed EMA/Nesterov and deterministic learning rates, with core proofs verified in Lean. The bounds retain separable layers and include unbiased bounded-variance noise and orthogonalization error.

### AI.3 — Reviewer says the guaranteed iterate differs from the queried model

**Status:** Checked statement; selected.

**Reviewer request:** Explain X versus W without transferring the guarantee.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies.

**Finding, meaning and limits:** X takes the exact server update; W tracks it through compressed downlink; workers evaluate gradients at W. The guarantee is stationarity at X, not W.

**Current answer:** EF21-MuonSign queries broadcast W; its theorem concerns server X. Their losses in the same nanoGPT run are 3.3213 and 5.5198.

### AI.4 — Reviewer identifies the large nanoGPT X/W loss gap

**Status:** Checked statement; selected.

**Reviewer request:** Acknowledge that the practically successful model is not the guaranteed one.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The two reported losses are evaluations of the same EF21-MuonSign run, not two independent runs. The gap is 2.1985 nats; c_proj residual concentration is an observation, not a causal initialization/compressor ablation.

**Current answer:** EF21-MuonSign queries broadcast W; its theorem concerns server X. Their losses in the same nanoGPT run are 3.3213 and 5.5198.

### AI.5 — Reviewer asks to frame positive EF21 results as an instantiation

**Status:** Checked statement; selected.

**Reviewer request:** Identify the paper-specific contribution.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The arbitrary-norm EF21 framework and gradient-side EF before a matrix transform are credited to prior work. Our empirical contribution is a controlled six-variant comparison with strong SignMuon performance; the new submitted proofs diagnose sign/polar ascent and output-EF divergence.

**Current answer:** We prove all 3 sign placements can ascend and output-level error feedback can diverge. Heterogeneous tests show why EF placement matters.

### AI.18 — Reviewer cites gradient-side EF before full-matrix preconditioners

**Status:** Checked statement; selected.

**Reviewer request:** Acknowledge Modoranu et al. explicitly.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Modoranu et al. 2024 establishes a precedent for EF compression before a structured preconditioner. It does not supply the sign/polar negative construction; cite both facts accurately.

**Current answer:** We prove all 3 sign placements can ascend and output-level error feedback can diverge. Heterogeneous tests show why EF placement matters.

### AI.6 — Reviewer requests the modern SSVR sign baseline

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Compare with Jiang et al. rather than claiming all gains arise from geometry.

**Sources:** P2: Jiang et al., NeurIPS 2024, Efficient Sign-based Optimization; code/common/optimizers.py SignSGD baseline.

**Finding, meaning and limits:** No SSVR run has been completed. Our SignSGD already uses momentum, but this does not substitute for a variance-reduced baseline; add the citation and remove exclusive causal attribution.

**Current answer:** No SSVR run has been completed. Our SignSGD already uses momentum, but this does not substitute for a variance-reduced baseline; add the citation and remove exclusive causal attribution.

### AI.19 — Reviewer rejects attributing gains exclusively to matrix geometry

**Status:** Checked statement; selected.

**Reviewer request:** Explain the existing momentum baseline and the missing comparison.

**Sources:** P2: code/common/optimizers.py and code/federated/algorithms.py; Jiang et al. 2024.

**Finding, meaning and limits:** The existing SignSGD baseline uses the same EMA momentum coefficient. Thus the comparison is not against memoryless gradient signs, but it does not isolate geometry against all variance-reduction choices.

**Current answer:** SignSGD already uses EMA. SSVR remains untested and needs matched gradient-query budgets; we will qualify the geometry attribution.

### AI.8 — Reviewer asks to limit broad federated claims

**Status:** Checked statement; selected.

**Reviewer request:** State the actual data and update regime.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR-retuned comparison, local updates and partial participation remain outstanding. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** New supplied CNN2 transfer adds five-seed Dirichlet(0.1) label skew using unchanged IID rates, still 11 fully participating clients and no local updates. EF21-MuonSign reaches 83.23±0.36% versus SignMuon 35.92±8.49%; retuning is pending. nanoGPT remains centralized optimization after full-precision averaging, not an LLM one-bit transport test. Revision 4 exports all 10 transfer results in 7S1b Q2, ordered by mean accuracy, with sample SD, 5 seeds, 11 clients, Dirichlet(0.1), 2000 rounds, no local updates and IID-selected rates. Gradient-side EF leads this transfer comparison. Retuned optima and local-update behavior are not established.

**Current answer:** After submission, we added Dirichlet(0.1) tests: EF21-MuonSign reaches 83.23% versus SignMuon 35.92% at IID rates. See the table in our reply to 7S1b, Q2. Clients take no local updates; retuning is pending. nanoGPT averages full-precision gradients.

### AI.9 — Reviewer requests a method matrix

**Status:** Checked statement; selected.

**Reviewer request:** Prepare the matrix of operations, payloads, states and guaranteed iterate.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput. W1: 02_REBUTTAL_PLAN.md method matrix.

**Finding, meaning and limits:** A concrete table is prepared internally with all six methods, EF targets, polar location, payloads, memory and theoretical iterate. It should replace the current scattered explanations.

**Current answer:** We will add the method/coverage table and move implementation to setup.

### AI.11 — Reviewer asks to move Section 4.1 implementation details

**Status:** Checked statement; selected.

**Reviewer request:** Restore a clear sequence of definitions and theoretical results.

**Sources:** W1: 02_REBUTTAL_PLAN.md proposed presentation changes.

**Finding, meaning and limits:** Move solver and layer-scaling details from Section 4.1 to the experimental setup. Keep theory definitions before counterexamples and EF results.

**Current answer:** We will add the method/coverage table and move implementation to setup.

### AI.10 — Reviewer asks for a structural condition explaining positive alignment

**Status:** Checked statement; selected.

**Reviewer request:** Give a sufficient condition with explicit scope.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The diagonal lower bound implies alignment by a singular-value-weighted sum. The full covariance criterion and stochastic first-moment theorem are in the author addendum. Credit OLion v2; it is not an empirical explanation for our networks.

**Current answer:** Our post-submission SignMuon theorem allows interacting layers under a uniform positive alignment with momentum; this is unmeasured.

### AI.20 — Reviewer asks to connect the structural condition to realistic training

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Measure the margin for the actual polar approximation and vote.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The geometric criterion has not been measured on the actual trajectories. Xiao et al. 2025 concerns SignSGD and does not by itself justify the SignMuon condition.

**Current answer:** The geometric criterion has not been measured on the actual trajectories. Xiao et al. 2025 concerns SignSGD and does not by itself justify the SignMuon condition.

### AI.13 — Reviewer asks for the zero-input oracle selection

**Status:** Checked statement; selected.

**Reviewer request:** Define the exact thin-SVD selection and distinguish sign ties.

**Sources:** W1: thin-SVD convention in aaai_article/ef21_signmuon_divergence.tex; code/common/optimizers.py.

**Finding, meaning and limits:** Use polar(0)=0 in the exact theory. For plain strict-binary sign ties at zero, the implementation can randomize; this is distinct from the polar selection.

**Current answer:** We will define polar(0)=0 and count EF scales. nanoGPT has 1 run/method; its 0.0021 gap cannot rank methods. A5’s 16.6 s/epoch is not network timing. MuonSign and SGD reach 80% in 5/5 seeds.

### AI.14 — Reviewer questions the single-seed 0.0021 nanoGPT ordering

**Status:** Checked statement; selected.

**Reviewer request:** Do not infer a reliable ranking from that difference.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** nanoGPT has one training run per compressed method. EF21-SignMuon 3.2860 versus SignMuon 3.2881 is an observed gap of 0.0021, not evidence of a reproducible ranking. SignMuon improves the observed SignSGD loss by 0.1168. No variance estimate is available for the compressed variants.

**Current answer:** We will define polar(0)=0 and count EF scales. nanoGPT has 1 run/method; its 0.0021 gap cannot rank methods. A5’s 16.6 s/epoch is not network timing. MuonSign and SGD reach 80% in 5/5 seeds.

### AI.15 — Reviewer asks for timing evidence behind the wall-clock claim

**Status:** Checked statement; selected.

**Reviewer request:** Use submitted timing data with the correct scope.

**Sources:** E4: submitted supplement Table A5, seconds/epoch (SignMuon16.6, SGD12.2, SignSGD12.5, Adam12.7); centralized export table_cifar.csv.

**Finding, meaning and limits:** Elapsed seconds/epoch are already in A5, so timing data are not entirely absent. The old main-table threshold numbers are stale. Withdraw the generic approximately 2x end-to-end speed claim and report per-epoch timing/threshold separately; compilation/validation and network transport differ across settings.

**Current answer:** We will define polar(0)=0 and count EF scales. nanoGPT has 1 run/method; its 0.0021 gap cannot rank methods. A5’s 16.6 s/epoch is not network timing. MuonSign and SGD reach 80% in 5/5 seeds.

### AI.16 — Reviewer asks to count scales in one-bit claims

**Status:** Checked statement; selected.

**Reviewer request:** Include one scale per matrix and full-precision auxiliary parameters.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** CNN2 has 762560 matrix and 2146 auxiliary parameters and 3 compressed blocks. Bits/channel are 1.0870 plain and 1.0871 EF, not exactly one per total parameter; uplink-only round-trip gain is 1.93x, not 29.4x.

**Current answer:** We will define polar(0)=0 and count EF scales. nanoGPT has 1 run/method; its 0.0021 gap cannot rank methods. A5’s 16.6 s/epoch is not network timing. MuonSign and SGD reach 80% in 5/5 seeds.

### AI.17 — Reviewer flags rounds-to-target conditioned on successful seeds

**Status:** Checked statement; selected.

**Reviewer request:** Correct the stale main table and report successes.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** Corrected submitted archive: MuonSign rounds80 860, SGD1240, both 5/5 successes; SignSGD1140 and SignMuon540, also5/5. Do not use the stale main-table 4/5 and3/5 denominators.

**Current answer:** We will define polar(0)=0 and count EF scales. nanoGPT has 1 run/method; its 0.0021 gap cannot rank methods. A5’s 16.6 s/epoch is not network timing. MuonSign and SGD reach 80% in 5/5 seeds.

### AI.12 — Reviewer asks to scope impossibility to the analyzed EF repair

**Status:** Checked statement; selected.

**Reviewer request:** Avoid ruling out all repairs and decreasing schedules.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.

**Finding, meaning and limits:** Theorem4 is forall fixed L,eta,mu exists parameter-dependent f for the specified scaled-sign output estimator. It does not rule out horizon-tuned finite-time rates, other EF schemes or different analyses.

**Current answer:** Theorem 4 does not exclude decreasing steps.

## Primary external sources checked on 29 October 2026

These links are internal evidence only and are excluded from every exported rebuttal. They were used to check attribution and the most recent concurrent theory, not to invent new empirical support.

- [Gruntkowska et al., Error Feedback for Muon and Friends](https://arxiv.org/html/2510.00643v1): arbitrary-norm distributed EF21 framework and its smooth stochastic theorems.
- [Wang, Shen and Zhang, OLion v2](https://arxiv.org/html/2602.01105v2): conditional diagonal-correlation geometry and spectral-spread interaction. Our own network margin is not measured.
- [Mishra et al., SignMuon v2](https://arxiv.org/html/2605.16311v2): conditional actual-vote theory, stochastic momentum and joint-block extension. The favorable condition is not automatic; our negative examples remain compatible with it.
- [Modoranu et al., ICML2024](https://proceedings.mlr.press/v235/modoranu24a.html): error feedback for compressed structured preconditioners.
- [Jiang et al., NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b7db05c0e383518d789b6e93131f1f0-Abstract-Conference.html): variance-reduced sign methods; no new comparison is available here.
- [Xiao et al., ICML2025](https://proceedings.mlr.press/v267/xiao25c.html): SignSGD risk analysis, not a guarantee for signing a polar factor.
