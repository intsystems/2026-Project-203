# Rebuttal plan and concern-to-evidence ledger — revision 7

## Contribution and response logic

SignMuon retains Muon-level quality at a sign budget and leads the tested sign methods in the submitted IID vision experiments. The completed heterogeneous study shows that gradient-side EF becomes essential for the strongest accuracy under Dirichlet(0.1) label skew: the advantage survives validation-based LR selection. The contribution connects the performance of these sign-based methods to the placement of sign and error feedback and to the conditions in their convergence analysis.

The six self-standing replies keep the scientific explanations, named setups and five conditional human score requests. The full tuned table appears once in **7S1b Q2** and is referenced from the five other replies and AC comment. Public replies contain no external links and fit 1500 UTF-16 characters. The addendum reports transfer and tuned results side by side. Its original author state is preserved before this update.

## Completed heterogeneous tuning

Source: `code/results/rebuttal_results4.zip`, frozen as `evidence/heterogeneous_tuning_results.zip`. SHA-256: `9d7c55c609b9a7fff8761c70b7d596bda13d972803e4dd7f36dfd3956f001bec`. The independent `verify_heterogeneous_tuning.py` audit recomputes all 167 raw histories: 72 validation trials and 95 unique final runs. There are 20 five-seed comparison rows; five Muon final runs are shared because its transfer and tuned rates coincide. All original 50 transfer metric files and statistics are unchanged.

CNN2/CIFAR-10; 11 fully participating Dirichlet(0.1) clients; 2000 rounds; three minibatches of 64 accumulated at the broadcast model; no local parameter steps. Momentum stays 0.9, with cosine scheduling, unit-gain scaling, zero weight decay and auxiliary AdamW LR 0.001. LR alone is selected from last-five validation accuracy, using seed 0 and 5000 held-out images. Finals retrain on all 50000 images with seeds 0–4. Report means and sample SDs of the final-five-evaluation mean for each seed.

The base 1–2–5 search starts with five rates and allows three two-point boundary extensions. MuonSign received two further validation rates, 1e-5 and 2e-5, to resolve its remaining endpoint. Its selected 5e-5 and final test results did not change. All ten selected rates are now interior. This is a completed LR-selected comparison at fixed momentum, rather than a claim of a joint LR/momentum optimum.

| Method | IID-rate transfer (%) | Selected LR | LR-tuned accuracy (%) |
| --- | ---: | ---: | ---: |
| EF21-MuonSign | 83.23±0.36 | 0.02 | 84.03±0.52 |
| EF21-MuonUSign | 82.99±0.46 | 0.02 | 83.84±0.32 |
| SGD | 81.03±0.62 | 0.1 | 79.71±2.03 |
| Adam | 71.17±7.16 | 0.0005 | 75.16±1.07 |
| EF21-SignMuon | 45.46±7.66 | 0.2 | 49.73±6.69 |
| Muon | 46.02±8.55 | 0.1 | 46.02±8.55 |
| MuonUSign | 26.21±2.75 | 0.001 | 44.23±3.28 |
| SignMuon | 35.92±8.49 | 0.005 | 42.96±5.94 |
| MuonSign | 25.63±3.98 | 5e-05 | 40.75±0.98 |
| SignSGD | 26.42±3.23 | 0.0005 | 36.24±4.64 |

Only EF21-MuonSign and EF21-MuonUSign exceed 80% after tuning. This supports using gradient-side EF under this label skew, while SignMuon remains strong on the submitted IID/centralized tasks. Local model updates and partial participation remain untested. One tuning seed is distinct from the five final seeds; repeated checkpoints are not independent runs.

## Evidence register

| Tag | Source | Result and scope |
| --- | --- | --- |
| E1 | Submitted A8, original federated logs, `verified_claims.json` | Five independent IID CNN2 seeds; corrected main table and ranking reproduce A8. All 175 original/current histories match. |
| E2 | Submitted A.11/A.14–15 and federated code | 11 synchronized clients, gradient accumulation at the broadcast model, no local parameter steps. |
| E3 | Submitted Table 3/A.16–17, `code/results/nanogpt/` | One run/method after full-precision gradient averaging; centralized optimizer evidence. |
| E4 | Submitted A5 and original article export | Centralized ResNet18/CIFAR-10, three seeds and per-epoch timing. |
| E5 | `heterogeneous_transfer_results.zip`, `heterogeneous_transfer_verified.json` | Original fifty transferred-IID-rate runs remain unchanged. |
| E6 | `heterogeneous_tuning_results.zip`, `heterogeneous_tuning_verified.json` | All ten validation LR searches and five-seed finals completed; means and sample SDs reproduced from raw histories. |
| C1 | Communication appendix and code | Logical payload includes FP32 auxiliaries and EF scales: about 1.087 bits per total CNN2 parameter on a compressed channel. |
| N1 | Submitted failure results and exact certificates | Three plain sign placements can ascend; specified output EF can diverge on a smooth lower-bounded objective. |
| N2 | Addendum softplus extension | One fixed smooth lower-bounded objective per plain placement defeats every positive step schedule. |
| N4 | Frozen `ef21_vanishing_snapshot/`, `new_quadratic_certificate.json` | Exact quadratic certificate: Frobenius smoothness 1, initial gap 4/7, gradient norm at least 1 under the documented fixed/horizon-tuned and decreasing schedules. |
| G1 | Addendum conditional SignMuon theorem and OLion geometry | Joint smoothness, uniform alignment, unbiased bounded-variance noise and horizon tuning; network margin unmeasured. |
| T1 | Revised Nesterov/fixed-radius/rate TeX | EF21 first- and second-moment guarantees have separate proofs and tuning. Fixed EMA/Nesterov and deterministic schedules are analyzed. |
| F1 | Lean statement audit and paper correspondence | Complete EF21 tracking/convergence proofs and original failure results are formalized; printed constant/cosine substitutions are manual specializations. No Lean claim for the separate plain SignMuon, softplus or N4 additions. |
| J1 | Separate internal candidate | Do not claim removal of EF21 layer separability; not used as a completed manuscript result. |
| P1 | Existing source/bibliography audit | General EF theory, concurrent SignMuon and OLion attribution remain documented internally. |
| P2 | SSVR/SignSGD sources and baseline code | SignSGD uses EMA. A matched-query SSVR implementation and experiment remain unavailable. |
| W1 | Current manuscript and addendum | Main Table 2/ranking corrections are present. This revision updates the addendum's empirical section and rebuilds its PDF. |

## Submitted quantitative comparisons

- Federated CNN2/CIFAR-10, 11 IID clients, five seeds: SignMuon 85.72±0.24%, Muon 85.98±0.26%, SignSGD 81.44±0.15%, MuonSign 82.94±0.19%.
- Centralized ResNet18/CIFAR-10, three seeds: SignMuon 94.60±0.15%, Muon 94.35±0.27%, SignSGD 93.37±0.26%. SignMuon timing is 16.6 s/epoch, not network throughput.
- nanoGPT, one run/method: SignMuon 3.2881, Muon 3.2785, SignSGD 3.4049, EF21-SignMuon 3.2860. The 0.0021 gap does not establish a ranking. EF21-MuonSign queries broadcast W (3.3213) while its theorem controls server X (5.5198), in the same run.

## Theory scope and remaining evidence

EF21 uses the paper's layerwise smoothness, which forces separability. Its fixed EMA/Nesterov tracking and convergence proofs are Lean verified. The first-moment deterministic-schedule result controls weighted expected nuclear gradient norms; the squared-gradient result uses gradient-dependent steps. Approximate directions enter through stated quality/error conditions; the implemented finite solver has no new error certificate from this empirical revision.

The separate conditional plain SignMuon theorem permits interacting layers under joint Frobenius smoothness and a uniform positive alignment margin. It bounds the average expected sum of nuclear block-gradient norms. Softplus preserves the plain ascent witnesses by positive gradient rescaling. N4 has a separate exact algebraic certificate. These analytic additions are distinct from the existing Lean proof audit.

The eight pending concern flags remain: R1.6, R1.7, R1.8, R2.2, R2.8, R4.15, AI.6 and AI.20. For R1.7, heterogeneous LR tuning is completed; its local-update/broader-participation part remains outstanding. The seven intentionally omitted presentation requests retain their IDs, notes and exact source links. No new training or full Lean build was run by this preparation.

## Per-concern ledger
### R1.1 — Reviewer asks for gradient structure that permits SignMuon descent

**Status:** Checked statement; selected.

**Reviewer request:** Explain a sufficient condition beyond the random-quadratic observation.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The diagonal-correlation condition implies alignment by weighting each diagonal entry by its nonnegative singular value. Uniform alignment, joint smoothness, centered finite-variance noise, exact polar and horizon-tuned fixed EMA/Nesterov yield an expected first-moment nuclear-gradient rate O(T^-1/4). It is a conditional result; actual trajectory verification and distributed-vote alignment remain open.

**Current answer:** After your review, we proved SignMuon convergence when signed updates remain uniformly aligned with momentum, under smoothness and unbiased bounded-variance noise. Layers may interact; network alignment is unmeasured.

### R1.6 — Reviewer requests neural-network alignment and spectral measurements

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Measure the proposed condition along actual network trajectories.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No neural-network measurement of the full geometric margin is available. Proposed diagnostics must record momentum/gradient alignment, diagonal correlations and spectral spread for the actual approximate solver and vote.

**Current answer:** No neural-network measurement of the full geometric margin is available. Proposed diagnostics must record momentum/gradient alignment, diagonal correlations and spectral spread for the actual approximate solver and vote.

### R1.2 — Reviewer asks whether federation changes preserve the ordering

**Status:** Checked statement; selected.

**Reviewer request:** State what the 11-client experiment supports.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR tuning is complete; local updates and partial participation remain outstanding. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** Both gradient-side EF methods lead after validation LR selection, not only with transferred IID rates. At momentum 0.9, EF21-MuonSign gives 84.0348 +/- 0.5187 and EF21-MuonUSign 83.8420 +/- 0.3230; SGD 79.7124 +/- 2.0324, Adam 75.1560 +/- 1.0702, Muon 46.0220 +/- 8.5462, SignMuon 42.9648 +/- 5.9431. All ten methods have five full-data final seeds; LR selection uses seed 0 and a fixed 5000-image validation set. The public 7S1b Q2 table reports all ten tuned results, ordered by mean, with the two EF rows bold. Only those two methods exceed 80% in this comparison. The original transfer runs remain unchanged and appear beside tuned results in the addendum. No local parameter steps or partial participation were tested.

**Current answer:** We tuned LRs on federated CNN2/CIFAR-10: 11 Dirichlet(0.1) clients, 2000 rounds, no local updates. Rates use validation seed 0; momentum 0.9; mean±SD over 5 final seeds:

|Method|Accuracy (%)|
|---|---:|
|**EF21-MuonSign**|**84.03±0.52**|
|**EF21-MuonUSign**|**83.84±0.32**|
|SGD|79.71±2.03|
|Adam|75.16±1.07|
|EF21-SignMuon|49.73±6.69|
|Muon|46.02±8.55|
|MuonUSign|44.23±3.28|
|SignMuon|42.96±5.94|
|MuonSign|40.75±0.98|
|SignSGD|36.24±4.64|

**EF21-Muon(U)Sign are the only methods above 80% after tuning.**

### R1.7 — Reviewer requests heterogeneous and local-training experiments

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Broaden the federated evaluation beyond synchronized homogeneous clients.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR tuning is complete; local updates and partial participation remain outstanding. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** The heterogeneous part is completed: transfer and validation-LR-tuned ten-method comparisons, 11 Dirichlet(0.1) clients, 2000 rounds and five final seeds. The pending flag now concerns the untested local parameter updates and broader participation settings. Do not mark the entire combined request resolved or promise an unrun experiment.

**Current answer:** Five-seed non-IID transfer and validation LR tuning are complete. Local parameter updates, partial participation and larger client populations remain untested.

### R1.3 — Reviewer asks which positive results are inherited from EF21

**Status:** Checked statement; selected.

**Reviewer request:** Credit the existing arbitrary-norm framework and isolate extensions.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The base positive EF21 rates instantiate prior work. The revised manuscript adds fixed EMA/Nesterov and deterministic constant/cosine bounds under its original block-separable assumptions. It does not remove EF21 layer separability. Revision 6: Complete tracking and convergence proofs are formalized; remove the misleading core qualifier. This schedule statement does not claim that every solver configuration has a certified approximation error, or that EF21 separability was removed.

**Current answer:** After your review, we proved EF21-Muon(U)Sign convergence for constant/cosine rates and fixed EMA/Nesterov, and verified the proofs in Lean. These bounds control expected gradient norms.

### R1.9 — Reviewer asks what technical contribution is new

**Status:** Checked statement; selected.

**Reviewer request:** Distinguish the negative construction from positive framework instantiation.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses for the plain placements. Output-EF decreasing-step and finite-budget results use the separate N4 construction. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The contribution is the controlled comparison of three sign placements and their EF variants, demonstrating Muon-level quality at a sign budget, plus placement-specific ascent and output-EF divergence proofs. The author addendum adds conditional SignMuon analysis with joint smoothness; the general EF21 framework and practical SignMuon idea are credited to prior/concurrent work. Revision 6: Complete tracking and convergence proofs are formalized; remove the misleading core qualifier. This schedule statement does not claim that every solver configuration has a certified approximation error, or that EF21 separability was removed.

**Current answer:** After your review, we proved EF21-Muon(U)Sign convergence for constant/cosine rates and fixed EMA/Nesterov, and verified the proofs in Lean. These bounds control expected gradient norms.

### R1.4 — Reviewer asks how to choose a communication variant

**Status:** Checked statement; selected.

**Reviewer request:** Recommend choices with explicit evidence limits.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput. E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** Under an uplink constraint EF21-MuonUSign retains an exact/full-precision downlink; under a bidirectional constraint EF21-MuonSign compresses both sides. SignMuon is the empirical candidate if conditional rather than general guarantees are acceptable. No universal ranking is predicted.

**Current answer:** SignMuon is effective on IID CNN2/CIFAR-10; for label skew, use EF21-MuonUSign for uplink-only compression or EF21-MuonSign for both links.

### R1.10 — Reviewer requests scales and memory in a comparison table

**Status:** Checked statement; selected.

**Reviewer request:** Account for payloads, precision, estimators and polar location.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** The internal method table is prepared in 02_REBUTTAL_PLAN.md. A compressed CNN2 channel costs 1.0870 bits/parameter for plain signs, 1.0871 for EF with FP32 auxiliary parameters and one FP32 scale per matrix. Each gradient EF method adds one client estimator and one global server estimator; two-sided EF additionally stores exact X separately from W.

**Current answer:** Messages use 1.087 bits/parameter, including auxiliaries and scales.

### R1.5 — Reviewer asks which nanoGPT differences are reliable

**Status:** Checked statement; selected.

**Reviewer request:** Separate descriptive gaps from statistical evidence.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** One nanoGPT training run per method gives SignMuon 3.2881 versus SignSGD 3.4049, an observed 0.1168-nat improvement. The 0.0021 gap to EF21-SignMuon does not establish an ordering; there is no compressed-method variance estimate. Upstream Muon variance cannot substitute.

**Current answer:** nanoGPT validation losses are 3.2881 for SignMuon and 3.4049 for SignSGD (1 run each). The 0.0021 gap to EF21-SignMuon cannot rank methods.

### R1.8 — Reviewer requests additional language-model seeds

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Repeat training rather than repeated validation of one run.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** No additional independent compressed-method training runs are available. Repeated evaluations and upstream Muon runs cannot substitute for them.

**Current answer:** No additional independent compressed-method training runs are available. Repeated evaluations and upstream Muon runs cannot substitute for them.

### R2.1 — Reviewer asks to identify sign as an LMO earlier

**Status:** Presentation/definition request retained internally; deliberately omitted from export.

**Reviewer request:** Define the geometry in Section 2.

**Sources:** W1: submitted Section 2 and current signmuon_body.tex Section 4.1.

**Finding, meaning and limits:** We will state in Section 2 that $-\mathrm{sign}(G)$ minimizes $\langle G,D\rangle$ over $\|D\|_\infty\le1$. Author requested keeping routine presentation/definition work out of the exported rebuttal. The request and planned edit remain in the fragment notes and source mapping; it is unchecked for export, not marked as missing scientific evidence.

**Current answer:** We will introduce sign as the LMO for the entrywise infinity-norm ball.

### R2.4 — Reviewer asks to distinguish raw signs from scaled EF messages

**Status:** Presentation/definition request retained internally; deliberately omitted from export.

**Reviewer request:** Explain the EF scale when the variants are first introduced.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** Plain signs send bits only; EF sends the residual signs plus its mean absolute value per matrix. FP32 auxiliary parameters are also counted. Author requested keeping routine presentation/definition work out of the exported rebuttal. The request and planned edit remain in the fragment notes and source mapping; it is unchecked for export, not marked as missing scientific evidence.

**Current answer:** We will distinguish sign bits from EF messages with 1 scale per matrix and count FP32 auxiliaries.

### R2.5 — Reviewer asks to distinguish centralized formulas from distributed algorithms

**Status:** Presentation/definition request retained internally; deliberately omitted from export.

**Reviewer request:** Explain the N=1 correspondence and nonlinear aggregation.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR tuning is complete; local updates and partial participation remain outstanding. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** Algorithms A2/A4/A5 define centralized placements. At N=1 the distributed definitions reduce to them; at N>1 local sign/LMO and majority voting define different algorithms. We will label both settings explicitly. Author requested keeping routine presentation/definition work out of the exported rebuttal. The request and planned edit remain in the fragment notes and source mapping; it is unchecked for export, not marked as missing scientific evidence. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** We will separate single-worker formulas from worker voting: applying signs or orthogonalization before averaging changes the update.

### R2.6 — Reviewer questions the linear objective under lower boundedness

**Status:** Checked statement; selected.

**Reviewer request:** Explain why the linear tests do not satisfy Assumption 1.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses for the plain placements. Output-EF decreasing-step and finite-budget results use the separate N4 construction. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Compose each original linear witness <G,X> with softplus log(1+exp(z)). The result is convex, globally smooth with Frobenius L <= ||G||_F^2/4 and bounded below by 0. Its gradient is sigmoid(<G,X>) G, a positive multiple of the original gradient, so each sign/polar map retains its ascent direction. One fixed objective per method defeats all positive step schedules, with either EMA or Nesterov. The original ascent constructions are Lean-verified; this analytic softplus extension is not included in that Lean claim.

**Current answer:** In response to your review, we made the linear counterexamples lower bounded by composing each objective with softplus, $z\mapsto\log(1+e^z)$. Its derivative is positive, so it only rescales the gradients and leaves the sign/polar directions unchanged. All 3 sign placements therefore still move uphill on smooth objectives bounded below by 0, even with decreasing steps. Each method has one fixed objective that defeats every positive step schedule. The original ascent constructions are now also verified in Lean.

### R2.7 — Reviewer suggests consistent names for sign placements

**Status:** Presentation/definition request retained internally; deliberately omitted from export.

**Reviewer request:** Make the composition order readable without changing identities.

**Sources:** W1: current signmuon_body.tex equations (5)–(6); method table in 02_REBUTTAL_PLAN.md.

**Finding, meaning and limits:** We will retain labels to keep tables/code traceable and add an operation-order legend: SignMuon = sign(polar(M)); MuonUSign = polar(sign(M)); MuonSign = sign(polar(sign(M))). Author requested keeping routine presentation/definition work out of the exported rebuttal. The request and planned edit remain in the fragment notes and source mapping; it is unchecked for export, not marked as missing scientific evidence.

**Current answer:** We will retain method names for consistency with code and add an operation-order legend.

### R2.2 — Reviewer suggests larger centralized and federated experiments

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Additional models and truly distributed language-model training would strengthen the evidence.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR tuning is complete; local updates and partial participation remain outstanding. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** No new large-model or distributed-language-model run was completed. nanoGPT evaluates the centralized optimizer after full-precision averaging, so it cannot fill this request. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** No new large-model or distributed-language-model run was completed. nanoGPT evaluates the centralized optimizer after full-precision averaging, so it cannot fill this request.

### R2.3 — Reviewer asks for progress on the theory-practice explanation

**Status:** Checked statement; selected.

**Reviewer request:** Give a positive condition without claiming a complete explanation.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** With joint smoothness, unbiased finite-variance noise, exact polar and horizon-tuned steps/momentum, we prove SignMuon convergence if $\langle M,\mathrm{sign}(\mathrm{polar}(M))\rangle\ge\delta\|M\|_*$ uniformly. It suffices that all diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ exceed $\delta>0$ for $M=U\Sigma V^\top$. Verification in our networks remains open. Revision 6: Name the nuclear block-gradient norm in add:eq:rate; joint Frobenius describes the smoothness assumption. This analytic theorem is separate from the Lean-verified EF21 convergence proofs. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** We now prove an $O(T^{-1/4})$ bound on SignMuon’s average expected nuclear gradient norm over $T$ iterations, when signing retains a fixed positive fraction of the polar update’s alignment with momentum. The theorem permits interacting layers and unbiased bounded-variance noise, with learning rate and momentum chosen for $T$. This gives a sufficient convergence condition; verifying it along network training remains open. On centralized ResNet18/CIFAR-10, SignMuon reaches 94.60% versus Muon 94.35% and SignSGD 93.37% (3 seeds). The added heterogeneous CNN2/CIFAR-10 tests show **only EF21-Muon(U)Sign above 80%** after validation LR tuning (table: 7S1b, Q2). Thus the useful placement depends on the data regime.

### R2.8 — Reviewer wants a realistic explanation of observed success

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Check the proposed sufficient condition on trained networks.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The conditional proof supplies a mathematical intermediate result. It does not establish that network momentum, approximate polar factors or distributed votes satisfy it.

**Current answer:** The conditional proof supplies a mathematical intermediate result. It does not establish that network momentum, approximate polar factors or distributed votes satisfy it.

### R3.2 — Reviewer identifies the reversed MuonSign versus SignSGD ranking

**Status:** Checked statement; selected.

**Reviewer request:** Correct the main table and every affected comparison.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** Table 2 is stale. Five independent submitted runs, each averaged over the final five evaluations at the 2000-round horizon, give MuonSign 82.9424 ± 0.1868 and SignSGD 81.4356 ± 0.1466 sample SD. These agree with A8, reversing the main-table sentence. SignMuon 85.7156 ± .2444 versus Muon 85.9836 ± .2608; no significance claim. Experiment identity is now stated explicitly: the 85.72/85.98/81.44% results are federated CNN2/CIFAR-10 with 11 IID clients and 5 seeds, not centralized ResNet18. The current main table and Section 5.2 prose have already been corrected; the public reply now reports this completed revision. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** On federated CNN2/CIFAR-10 (11 IID clients, 5 seeds), the submitted logs reproduce A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%. SignMuon reaches 85.72±0.24% versus Muon 85.98±0.26%. We corrected Table 2 and its reversed comparison in the revision. On the new label-skew split, **only EF21-Muon(U)Sign exceed 80%** after validation LR tuning (7S1b, Q2).

### R3.1 — Reviewer asks which learning-rate rule proves the squared-gradient rate

**Status:** Checked statement; selected.

**Reviewer request:** Do not transfer the sharp-stepsize result to cosine schedules.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The squared dual-gradient rate O(T^-1/2) has a separate second-moment argument with gradient-scaled radii and horizon-tuned momentum. It does not follow for the cosine settings used in the experiments. Revision 6: Original Ueq5 review raises schedules, separability and rate moments, not nanoGPT momentum warmup or auxiliary Adam. Remove that unsolicited aside. State completed proof extensions and what their terms measure; preserve the EF21 versus plain SignMuon distinction.

**Current answer:** To address the schedule mismatch, we generalized EF21-MuonUSign and EF21-MuonSign convergence to fixed EMA/Nesterov and deterministic learning rates, including constant/cosine schedules. We verified the tracking and convergence proofs in Lean. The bounds quantify initialization, stochastic noise and orthogonalization accuracy. EF21 uses the submitted separable-layer assumption; our separate SignMuon theorem allows interacting layers under joint smoothness and uniform alignment.

### R3.3 — Reviewer identifies the separability forced by layer-wise smoothness

**Status:** Checked statement; selected.

**Reviewer request:** Replace the restriction with an explicit whole-model formulation.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies.

**Finding, meaning and limits:** The manuscript’s block-only smoothness forces separability, and the revised EF21 proofs retain it. The author addendum explicitly calls EF21 cross-layer coupling an open target. Only the added conditional plain SignMuon theorem uses joint Frobenius smoothness and permits interacting layers. The separate J1 candidate must not be described as a completed manuscript extension. Revision 6: Original Ueq5 review raises schedules, separability and rate moments, not nanoGPT momentum warmup or auxiliary Adam. Remove that unsolicited aside. State completed proof extensions and what their terms measure; preserve the EF21 versus plain SignMuon distinction.

**Current answer:** To address the schedule mismatch, we generalized EF21-MuonUSign and EF21-MuonSign convergence to fixed EMA/Nesterov and deterministic learning rates, including constant/cosine schedules. We verified the tracking and convergence proofs in Lean. The bounds quantify initialization, stochastic noise and orthogonalization accuracy. EF21 uses the submitted separable-layer assumption; our separate SignMuon theorem allows interacting layers under joint smoothness and uniform alignment.

### R3.4 — Reviewer rejects squaring an expected first-moment bound

**Status:** Checked statement; selected.

**Reviewer request:** Remove the unsupported inference and state two distinct results.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies.

**Finding, meaning and limits:** A first-moment gradient bound cannot be squared to obtain a second-moment bound. The squared O(T^-1/2) result has a separate proof and gradient-scaled steps. Under the original separable assumptions, ideal fixed-momentum EF21 has a deterministic-schedule first-moment bound: a transient plus residuals, or O(T^-1/4) with eta_peak proportional to T^-3/4 and 1-mu=T^-1/2. Revision 6: Name both algorithms, the nuclear first-moment quantity, horizon tuning and exact polar assumption. The O(T^-1/4) statement follows cor:radius_schedules(iii); the O(T^-1/2) squared-norm result has its own proof. Drop the obvious squaring admonition and unclear claims-separation label.

**Current answer:** For EF21-MuonUSign and EF21-MuonSign with exact polar updates, the new constant/cosine theorem bounds the learning-rate-weighted average expected nuclear gradient norm. Choosing the peak learning rate and fixed momentum for $T$ iterations gives $O(T^{-1/4})$ convergence. A separate proof gives an $O(T^{-1/2})$ average expected squared nuclear gradient norm with gradient-dependent steps.

### R3.5 — Reviewer asks to distinguish ideal EF21 from the implementations

**Status:** Checked statement; selected.

**Reviewer request:** State the remaining oracle and optimizer qualifications.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Lean’s fixed-momentum trajectory theorems allow approximate directions through explicit additive error or positive oracle-gain assumptions. This is not a certificate of a sufficiently small error for the actual finite polynomial implementation. The actual nanoGPT momentum changes and auxiliary Adam updates still need additional analysis. The multiple-block EF21 result retains separability. Revision 6: Original Ueq5 review raises schedules, separability and rate moments, not nanoGPT momentum warmup or auxiliary Adam. Remove that unsolicited aside. State completed proof extensions and what their terms measure; preserve the EF21 versus plain SignMuon distinction.

**Current answer:** To address the schedule mismatch, we generalized EF21-MuonUSign and EF21-MuonSign convergence to fixed EMA/Nesterov and deterministic learning rates, including constant/cosine schedules. We verified the tracking and convergence proofs in Lean. The bounds quantify initialization, stochastic noise and orthogonalization accuracy. EF21 uses the submitted separable-layer assumption; our separate SignMuon theorem allows interacting layers under joint smoothness and uniform alignment.

### R4.1 — Reviewer asks whether the experimental schedules have the proved rate

**Status:** Checked statement; selected.

**Reviewer request:** Identify the distinction between squared-gradient steps and deterministic schedules.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** We generalized the EF21 analysis to EMA/Nesterov and deterministic learning rates. Its tracking and convergence arguments are formalized in Lean. Constant/cosine corollaries follow by schedule specialization, whose printed cosine sum is not itself a complete exported Lean theorem. Approximate directions are admitted through explicit error/quality assumptions. The squared-gradient rate still needs gradient-dependent steps and the manuscript retains separability. Revision 6: Answer pqje Q1 positively with completed schedule/momentum extensions and their measured quantity. Unlike Ueq5, pqje explicitly asks about nanoGPT Nesterov; the direct answer on fixed coefficients versus warmup is in Q4. Formal correspondence still retains the underlying layerwise smoothness and approximation-quality assumptions.

**Current answer:** After your review, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and constant/cosine rates, and verified the proofs in Lean. This gives expected nuclear gradient norm bounds, with explicit noise and polar approximation terms. We also proved the squared-gradient rate for gradient-dependent steps.

### R4.2 — Reviewer asks whether nanoGPT Nesterov momentum is covered

**Status:** Checked statement; selected.

**Reviewer request:** Separate a fixed Nesterov coefficient from momentum warmup.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The generalized analysis and Lean trajectory proofs cover a fixed momentum coefficient in EMA or Nesterov. nanoGPT changes that coefficient during training, so the implementation is not certified by the fixed-momentum result. Revision 6: Answer pqje Q1 positively with completed schedule/momentum extensions and their measured quantity. Unlike Ueq5, pqje explicitly asks about nanoGPT Nesterov; the direct answer on fixed coefficients versus warmup is in Q4. Formal correspondence still retains the underlying layerwise smoothness and approximation-quality assumptions.

**Current answer:** After your review, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and constant/cosine rates, and verified the proofs in Lean. This gives expected nuclear gradient norm bounds, with explicit noise and polar approximation terms. We also proved the squared-gradient rate for gradient-dependent steps.

### R4.3 — Reviewer identifies inconsistent main and supplementary results

**Status:** Checked statement; selected.

**Reviewer request:** Reconstruct the five runs and correct the ranking.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** The original submitted five-seed archive agrees with A8. Table 2 is stale; final-five-evaluation means give 82.9424 ± .1868 vs 81.4356 ± .1466. The main-table comparison must be reversed. Experiment identity is now stated explicitly: the 85.72/85.98/81.44% results are federated CNN2/CIFAR-10 with 11 IID clients and 5 seeds, not centralized ResNet18. The current main table and Section 5.2 prose have already been corrected; the public reply now reports this completed revision.

**Current answer:** The submitted federated CNN2/CIFAR-10 logs (11 IID clients, 5 seeds) give MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8. We corrected Table 2 and the reversed comparison. SignMuon reaches 85.72% versus Muon 85.98%.

### R4.4 — Reviewer asks for useful sufficient conditions for SignMuon

**Status:** Checked statement; selected.

**Reviewer request:** Supply a structural condition and a conditional result.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** The full addendum gives the weaker covariance/spread criterion epsilon*rho<1 and a complete stochastic fixed-momentum first-moment proof. The simpler diagonal lower bound in the export is sufficient and self-contained. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** After submission, we proved SignMuon converges if signed updates retain enough alignment with momentum, under joint smoothness and unbiased bounded-variance noise. This permits interacting layers; network alignment remains unmeasured. On label-skewed CNN2/CIFAR-10, **only EF21-Muon(U)Sign exceed 80%** after validation LR tuning (7S1b, Q2).

### R4.15 — Reviewer asks why the favorable quadratic alignment extends to neural networks

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Distinguish a sufficient theorem from evidence along actual training trajectories.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The sufficient margin has not been checked on these neural-network trajectories, approximate solvers or distributed votes. The favorable quadratic result does not establish it.

**Current answer:** The sufficient margin has not been checked on these neural-network trajectories, approximate solvers or distributed votes. The favorable quadratic result does not establish it.

### R4.6 — Reviewer questions novelty relative to EF21 and concurrent SignMuon

**Status:** Checked statement; selected.

**Reviewer request:** Credit existing frameworks and the latest concurrent theory.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** Practical SignMuon and the general EF21 framework are prior/concurrent work. Our supported contribution is the controlled comparison of six placements/EF variants, strong sign-based performance and distinct sign/polar failure mechanisms. Mishra v2 has conditional actual-vote stochastic momentum theory; the current manuscript’s v1-only comparison must be updated. The added conditional SignMuon theorem is new to this submission, not claimed as the first conditional SignMuon result. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** After submission, we proved SignMuon converges if signed updates retain enough alignment with momentum, under joint smoothness and unbiased bounded-variance noise. This permits interacting layers; network alignment remains unmeasured. On label-skewed CNN2/CIFAR-10, **only EF21-Muon(U)Sign exceed 80%** after validation LR tuning (7S1b, Q2).

### R4.5 — Reviewer says nanoGPT does not test the proposed distributed compression

**Status:** Checked statement; selected.

**Reviewer request:** Make the full-precision transport explicit in the main text.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The reviewer is correct. Full-precision gradient averaging precedes the owning-rank optimizer; parameter all-gather follows. Local compress-then-vote semantics and actual one-bit communication gains are not tested by this experiment. Revision 6: pqje Q1 explicitly raises nanoGPT Nesterov and weakness 2 raises centralized averaging versus worker voting. Answer those issues only; omit auxiliary Adam, which this review does not ask about.

**Current answer:** nanoGPT tests centralized updates after full-precision gradient averaging: validation loss 3.2881 for SignMuon versus 3.4049 for SignSGD. Our extension covers fixed Nesterov momentum; nanoGPT uses momentum warmup. Federated CNN2 tests worker voting.

### R4.7 — Reviewer asks for definitions before theorem references

**Status:** Checked statement; selected.

**Reviewer request:** Introduce the operations before the introduction cites alignment results.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Introduce the operations before the introduction cites alignment results. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

### R4.8 — Reviewer objects to broad impossibility wording

**Status:** Checked statement; selected.

**Reviewer request:** Replace “unrepairable” with the specified fixed-parameter output-EF statement.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Replace “unrepairable” with the specified fixed-parameter output-EF statement. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

### R4.9 — Reviewer rejects presenting empirical ordering as a reversed theoretical prediction

**Status:** Checked statement; selected.

**Reviewer request:** The existence of failure examples does not predict method ranking.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** The existence of failure examples does not predict method ranking. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

### R4.10 — Reviewer asks for a direct conclusion without repetitive caveats

**Status:** Checked statement; selected.

**Reviewer request:** Remove repeated “document rather than resolve” language.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Remove repeated “document rather than resolve” language. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

### R4.11 — Reviewer asks which of X and W is updated and analyzed

**Status:** Checked statement; selected.

**Reviewer request:** Explain exact server X, compressed broadcast W and their roles.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Explain exact server X, compressed broadcast W and their roles. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

### R4.12 — Reviewer asks to explain target iteration counts before using them

**Status:** Checked statement; selected.

**Reviewer request:** Distinguish finite-target success and stationary error floors from asymptotic rate.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Distinguish finite-target success and stationary error floors from asymptotic rate. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

### R4.13 — Reviewer asks for a clearer statement of the Section 5.2 results

**Status:** Checked statement; selected.

**Reviewer request:** Lead with measured accuracy instead of theorem qualifications.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Lead with measured accuracy instead of theorem qualifications. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

### R4.14 — Reviewer asks to move essential explanations into the main paper

**Status:** Checked statement; selected.

**Reviewer request:** Prepare a coverage table and shorten captions.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. W1: 02_REBUTTAL_PLAN.md replacement prose.

**Finding, meaning and limits:** Prepare a coverage table and shorten captions. Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.

**Current answer:** EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

### R5.1 — Reviewer asks what is new beyond the EF21-Muon framework

**Status:** Checked statement; selected.

**Reviewer request:** State the new proofs and credit inherited positive rates.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified. N4: evidence/ef21_vanishing_snapshot/theorem.tex and certificate.py; evidence/new_quadratic_certificate.json and verification.log. Exact algebraic certificate checks 128 strict sign inequalities; 20 independent 2000-step SVD trajectories pass. For any budget-tuned constant learning rate and fixed EMA/Nesterov momentum, an 8x8 convex quadratic has Frobenius L=1, initial gap 4/7 and gradient norm >=1 at every iterate. The instance may depend on the parameters; the class constants are uniform. Arbitrary decreasing steps to zero are covered without momentum; suitable inverse-power decay is covered for each fixed momentum coefficient. Exact gradients/polar; not a Lean claim. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** The contribution combines strong measured sign-based optimization with a systematic comparison of three sign placements and their EF variants, and explicit placement-specific ascent/output-EF divergence proofs. In the submitted IID CNN2 study, at the same payload SignMuon gains 4.28 points over momentum SignSGD and is 0.27 points below Muon. This is evidence among tested methods, not a universal ranking or priority claim for the SignMuon idea. Experiment identity is now stated explicitly: the 85.72/85.98/81.44% results are federated CNN2/CIFAR-10 with 11 IID clients and 5 seeds, not centralized ResNet18. Revision 6: State verification of the convergence proofs without the core qualifier. This review asks about novelty and output-EF quantifiers; the public answer need not repeat unsolicited scope caveats. EF21 separability remains documented internally and in the Ueq5/AI answers. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** On federated CNN2/CIFAR-10 (11 IID clients, 5 seeds), SignMuon reaches 85.72% at 1 bit per matrix entry, versus Muon 85.98% and SignSGD 81.44%. We isolate how sign and error-feedback placement affect performance. After validation LR tuning on the heterogeneous split, **EF21-MuonSign reaches 84.03% and EF21-MuonUSign 83.84%**, the only methods above 80% (7S1b, Q2). After your review, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and constant/cosine rates, with proofs verified in Lean.

### R5.3 — Reviewer asks what optimizer-design lesson the counterexamples provide

**Status:** Checked statement; selected.

**Reviewer request:** Explain what is tracked and the consequences of its variation.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified. N4: evidence/ef21_vanishing_snapshot/theorem.tex and certificate.py; evidence/new_quadratic_certificate.json and verification.log. Exact algebraic certificate checks 128 strict sign inequalities; 20 independent 2000-step SVD trajectories pass. For any budget-tuned constant learning rate and fixed EMA/Nesterov momentum, an 8x8 convex quadratic has Frobenius L=1, initial gap 4/7 and gradient norm >=1 at every iterate. The instance may depend on the parameters; the class constants are uniform. Arbitrary decreasing steps to zero are covered without momentum; suitable inverse-power decay is covered for each fixed momentum coefficient. Exact gradients/polar; not a Lean claim. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** The example proves that sharing one residual scale while tracking a rapidly changing polar output can create a persistent wrong-sign component. Gradient-level EF avoids this particular tracking obstacle under smoothness; the guarantee still requires its stated assumptions. Revision 6: State verification of the convergence proofs without the core qualifier. This review asks about novelty and output-EF quantifiers; the public answer need not repeat unsolicited scope caveats. EF21 separability remains documented internally and in the Ueq5/AI answers. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** On federated CNN2/CIFAR-10 (11 IID clients, 5 seeds), SignMuon reaches 85.72% at 1 bit per matrix entry, versus Muon 85.98% and SignSGD 81.44%. We isolate how sign and error-feedback placement affect performance. After validation LR tuning on the heterogeneous split, **EF21-MuonSign reaches 84.03% and EF21-MuonUSign 83.84%**, the only methods above 80% (7S1b, Q2). After your review, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and constant/cosine rates, with proofs verified in Lean.

### R5.2 — Reviewer asks whether Theorem 4 includes decreasing learning rates

**Status:** Checked statement; selected.

**Reviewer request:** Write the quantifiers and identify their limit.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses for the plain placements. Output-EF decreasing-step and finite-budget results use the separate N4 construction. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified. N4: evidence/ef21_vanishing_snapshot/theorem.tex and certificate.py; evidence/new_quadratic_certificate.json and verification.log. Exact algebraic certificate checks 128 strict sign inequalities; 20 independent 2000-step SVD trajectories pass. For any budget-tuned constant learning rate and fixed EMA/Nesterov momentum, an 8x8 convex quadratic has Frobenius L=1, initial gap 4/7 and gradient norm >=1 at every iterate. The instance may depend on the parameters; the class constants are uniform. Arbitrary decreasing steps to zero are covered without momentum; suitable inverse-power decay is covered for each fixed momentum coefficient. Exact gradients/polar; not a Lean claim.

**Finding, meaning and limits:** Submitted Theorem 4 alone uses fixed parameters and a parameter-dependent objective. The separate added N4 construction now gives a uniform nonstationarity obstruction: Frobenius smoothness 1, initial gap exactly 4/7, dimension 8x8 and gradient norm >=1 at every iterate, for any budget-tuned constant rate and fixed EMA/Nesterov coefficient. The instance may depend on the parameters; constants are uniform across the class. Without momentum it covers any strictly decreasing positive schedule tending to zero. For each fixed momentum it covers suitably shifted power decay. It does not assert arbitrary changing momentum or finite-polar trajectories. The exact certificate is checked; this addition is not in the Lean audit.

**Current answer:** Theorem 4 alone concerns fixed parameters. In response, for any constant learning rate and fixed EMA/Nesterov momentum chosen for the run length, we construct an 8×8 convex quadratic for exact-polar EF21-SignMuon with Frobenius smoothness 1, initial gap 4/7 and gradient norm ≥1 at every iterate. Its gradient cannot approach zero. We also cover strictly decreasing steps to zero without momentum and power decay with fixed momentum.

### R5.4 — Reviewer says parameter-dependent divergence need not preclude finite-time rates

**Status:** Checked statement; selected.

**Reviewer request:** Acknowledge the dependence of the constructed objective and its constants.

**Sources:** N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses for the plain placements. Output-EF decreasing-step and finite-budget results use the separate N4 construction. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified. N4: evidence/ef21_vanishing_snapshot/theorem.tex and certificate.py; evidence/new_quadratic_certificate.json and verification.log. Exact algebraic certificate checks 128 strict sign inequalities; 20 independent 2000-step SVD trajectories pass. For any budget-tuned constant learning rate and fixed EMA/Nesterov momentum, an 8x8 convex quadratic has Frobenius L=1, initial gap 4/7 and gradient norm >=1 at every iterate. The instance may depend on the parameters; the class constants are uniform. Arbitrary decreasing steps to zero are covered without momentum; suitable inverse-power decay is covered for each fixed momentum coefficient. Exact gradients/polar; not a Lean claim.

**Finding, meaning and limits:** Submitted Theorem 4 alone uses fixed parameters and a parameter-dependent objective. The separate added N4 construction now gives a uniform nonstationarity obstruction: Frobenius smoothness 1, initial gap exactly 4/7, dimension 8x8 and gradient norm >=1 at every iterate, for any budget-tuned constant rate and fixed EMA/Nesterov coefficient. The instance may depend on the parameters; constants are uniform across the class. Without momentum it covers any strictly decreasing positive schedule tending to zero. For each fixed momentum it covers suitably shifted power decay. It does not assert arbitrary changing momentum or finite-polar trajectories. The exact certificate is checked; this addition is not in the Lean audit.

**Current answer:** Theorem 4 alone concerns fixed parameters. In response, for any constant learning rate and fixed EMA/Nesterov momentum chosen for the run length, we construct an 8×8 convex quadratic for exact-polar EF21-SignMuon with Frobenius smoothness 1, initial gap 4/7 and gradient norm ≥1 at every iterate. Its gradient cannot approach zero. We also cover strictly decreasing steps to zero without momentum and power decay with fixed momentum.

### R5.5 — Reviewer rejects a theoretical prediction of the empirical ranking

**Status:** Checked statement; selected.

**Reviewer request:** Separate worst-case statements from comparative accuracy.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses for the plain placements. Output-EF decreasing-step and finite-budget results use the separate N4 construction.

**Finding, meaning and limits:** The theory does not predict which optimizer wins on a task. The positive SignMuon theorem is conditional and compatible with the counterexamples. A new fixed softplus objective proves arbitrary-schedule nonstationarity only for each plain placement; this does not strengthen the output-EF quantifiers.

**Current answer:** Use SignMuon on IID data, EF21-MuonUSign for label-skewed uplink compression and EF21-MuonSign for both links. These choices follow experiments; failure examples do not predict a ranking.

### R5.6 — Reviewer objects to presenting sufficient compressor assumptions as necessary

**Status:** Checked statement; selected.

**Reviewer request:** Retract claims that only changing compressors can help.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** Failure of spectral contractivity does not prove nonconvergence under every analysis or parameter choice. c_proj concentration is observational. No initialization/step/compressor ablation proves causation or uniqueness of a remedy.

**Current answer:** A sufficient compressor condition need not be necessary; the impossibility concerns only the specified output-EF scheme and schedules.

### AI.1 — Reviewer asks for the missing stochastic assumptions

**Status:** Checked statement; selected.

**Reviewer request:** State the noise assumptions where the theorem is stated.

**Sources:** T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum for the separate squared-moment result; never infer it by squaring a first moment. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Unbiased bounded-variance assumptions are in submitted A.10 and must also appear in the main statement. Revised EF21 bounds retain the original separable smoothness; the added plain SignMuon theorem instead uses joint Frobenius smoothness and conditional centering/finite variance. The two results must be distinguished. Revision 6: The AI review explicitly raises missing noise, layer coupling and inexact polar assumptions. State the completed general theorem and its explicit approximation-error contribution positively; do not imply a new certificate for the implemented five-stage solver. Complete generic trajectory/tracking proofs are Lean verified; printed cosine substitutions remain algebraic specializations.

**Current answer:** After submission, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and deterministic rates, and verified the proofs in Lean. The bounds use separable-layer smoothness and unbiased bounded-variance noise, with an explicit polar approximation term.

### AI.2 — Reviewer says the layer smoothness forces separability

**Status:** Checked statement; selected.

**Reviewer request:** Use jointly coupled smoothness instead of the original restrictive assumption.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The manuscript does not remove EF21 separability. Its fixed-radius proof explicitly uses block independence, and the author addendum marks a joint-smooth EF21 extension open. The added conditional SignMuon theorem allows interacting layers. J1 is a separate candidate derivation and is excluded from public claims. Revision 6: The AI review explicitly raises missing noise, layer coupling and inexact polar assumptions. State the completed general theorem and its explicit approximation-error contribution positively; do not imply a new certificate for the implemented five-stage solver. Complete generic trajectory/tracking proofs are Lean verified; printed cosine substitutions remain algebraic specializations.

**Current answer:** After submission, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and deterministic rates, and verified the proofs in Lean. The bounds use separable-layer smoothness and unbiased bounded-variance noise, with an explicit polar approximation term.

### AI.7 — Reviewer asks for an inexact polar-oracle guarantee

**Status:** Checked statement; selected.

**Reviewer request:** Do not identify the finite polynomial solver with the exact polar theorem.

**Sources:** T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies. G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Lean’s schedule/trajectory results include approximation-quality or error terms rather than requiring every direction to equal an exact polar factor. No trajectory-wide small-error certificate is available for the actual five-stage implementation, and no full nanoGPT theorem covers changing momentum plus auxiliary Adam. State the proved generalization and the remaining assumptions separately. Revision 6: The AI review explicitly raises missing noise, layer coupling and inexact polar assumptions. State the completed general theorem and its explicit approximation-error contribution positively; do not imply a new certificate for the implemented five-stage solver. Complete generic trajectory/tracking proofs are Lean verified; printed cosine substitutions remain algebraic specializations.

**Current answer:** After submission, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and deterministic rates, and verified the proofs in Lean. The bounds use separable-layer smoothness and unbiased bounded-variance noise, with an explicit polar approximation term.

### AI.3 — Reviewer says the guaranteed iterate differs from the queried model

**Status:** Checked statement; selected.

**Reviewer request:** Explain X versus W without transferring the guarantee.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. T1/G1: aaai_article/ef21_fixed_radius.tex (explicit separability step), ef21_nesterov_convergence.tex, and signmuon_addendum_body.tex (conditional SignMuon theorem and open EF21 cross-layer target). J1: separate candidate derivation in evidence/EF21_JOINT_SMOOTHNESS.md; not a revised-manuscript result and not relied on by the replies.

**Finding, meaning and limits:** X takes the exact server update; W tracks it through compressed downlink; workers evaluate gradients at W. The guarantee is stationarity at X, not W.

**Current answer:** EF21-MuonSign queries broadcast W but guarantees server X. Their nanoGPT losses are 3.3213/5.5198 in the same run, after full-precision gradient averaging.

### AI.4 — Reviewer identifies the large nanoGPT X/W loss gap

**Status:** Checked statement; selected.

**Reviewer request:** Acknowledge that the practically successful model is not the guaranteed one.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The two reported losses are evaluations of the same EF21-MuonSign run, not two independent runs. The gap is 2.1985 nats; c_proj residual concentration is an observation, not a causal initialization/compressor ablation.

**Current answer:** EF21-MuonSign queries broadcast W but guarantees server X. Their nanoGPT losses are 3.3213/5.5198 in the same run, after full-precision gradient averaging.

### AI.5 — Reviewer asks to frame positive EF21 results as an instantiation

**Status:** Checked statement; selected.

**Reviewer request:** Identify the paper-specific contribution.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** The arbitrary-norm EF21 framework and gradient-side EF before a matrix transform are credited to prior work. Our empirical contribution is a controlled six-variant comparison with strong SignMuon performance; the new submitted proofs diagnose sign/polar ascent and output-EF divergence.

**Current answer:** All 3 sign placements can ascend; output-level EF can fail with decreasing steps or budget tuning.

### AI.18 — Reviewer cites gradient-side EF before full-matrix preconditioners

**Status:** Checked statement; selected.

**Reviewer request:** Acknowledge Modoranu et al. explicitly.

**Sources:** P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. The current main article comparison to Mishra v1 is stale and must be replaced. F1: lean_proofs/README.md, PAPER_AUDIT.md, comparator/VERIFICATION.json (45 statements, passed 2026-10-23; reference and dependency hashes rechecked), and research/nesterov/lean-verification.log. Original ascent/output-EF divergence results and the generalized EF21 tracking/convergence arguments have Lean proofs. The printed cosine-sum specialization is a manual composition. The added conditional plain SignMuon theorem and softplus extensions are not claimed Lean-verified.

**Finding, meaning and limits:** Modoranu et al. 2024 establishes a precedent for EF compression before a structured preconditioner. It does not supply the sign/polar negative construction; cite both facts accurately.

**Current answer:** All 3 sign placements can ascend; output-level EF can fail with decreasing steps or budget tuning.

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

**Current answer:** SignSGD already uses EMA. We cannot isolate geometry without a matched-query SSVR comparison, which remains untested.

### AI.8 — Reviewer asks to limit broad federated claims

**Status:** Checked statement; selected.

**Reviewer request:** State the actual data and update regime.

**Sources:** E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. E5: evidence/heterogeneous_transfer_results.zip (supplied during revision 2), verified by verify_heterogeneous_transfer.py against all 50 raw run configs/histories; heterogeneous_transfer_verified.json. Dirichlet(0.1), 11 clients, five paired seeds, IID learning rates held fixed. LR tuning is complete; local updates and partial participation remain outstanding. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer. E6: evidence/heterogeneous_tuning_results.zip (supplied rebuttal_results4.zip), verify_heterogeneous_tuning.py and heterogeneous_tuning_verified.json. All 167 unique 2000-round jobs: 72 validation trials, 95 final runs, 20 five-seed comparison rows with five Muon finals reused. Only LR tuned, at momentum 0.9; validation seed 0 and 5000 held-out images; finals retrain on 50000. All ten selected rates interior after two additional MuonSign validation trials. EF21-MuonSign 84.0348+/-0.5187, EF21-MuonUSign 83.8420+/-0.3230; SGD 79.7124+/-2.0324, Muon 46.0220+/-8.5462, SignMuon 42.9648+/-5.9431. Local parameter updates, partial participation and joint LR/momentum tuning remain untested.

**Finding, meaning and limits:** New supplied CNN2 transfer adds five-seed Dirichlet(0.1) label skew using unchanged IID rates, still 11 fully participating clients and no local updates. EF21-MuonSign reaches 83.23±0.36% versus SignMuon 35.92±8.49%; LR tuning is complete. nanoGPT remains centralized optimization after full-precision averaging, not an LLM one-bit transport test. Revision 4 exports all 10 transfer results in 7S1b Q2, ordered by mean accuracy, with sample SD, 5 seeds, 11 clients, Dirichlet(0.1), 2000 rounds, no local updates and IID-selected rates. Gradient-side EF leads this transfer comparison. The validation LR-selected comparison is complete; local-update behavior is not established. The revision 5 table bolds both EF21-MuonSign and EF21-MuonUSign values and names. Their means 83.23% and 82.99% are the only Muon-variant means above 80% in this test. SGD also exceeds 80%, so the claim is explicitly scoped to Muon variants. The reused IID learning rates stay in the protocol; LR tuning is complete in revision 7. Revision 7: LR tuning is complete for all ten methods. Both gradient-side EF variants exceed 83.8% after validation-only selection; every other tuned method is below 80%. Local parameter steps and partial participation remain untested. The new public shared table reports the tuned comparison, not the earlier transferred settings.

**Current answer:** We completed validation LR tuning on federated CNN2/CIFAR-10: 11 Dirichlet(0.1) clients, 2000 rounds, no local updates, 5 final seeds. **Only EF21-Muon(U)Sign exceed 80%** (table: 7S1b, Q2).

### AI.9 — Reviewer requests a method matrix

**Status:** Presentation/definition request retained internally; deliberately omitted from export.

**Reviewer request:** Prepare the matrix of operations, payloads, states and guaranteed iterate.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput. W1: 02_REBUTTAL_PLAN.md method matrix.

**Finding, meaning and limits:** A concrete table is prepared internally with all six methods, EF targets, polar location, payloads, memory and theoretical iterate. It should replace the current scattered explanations. Author requested keeping routine presentation/definition work out of the exported rebuttal. The request and planned edit remain in the fragment notes and source mapping; it is unchecked for export, not marked as missing scientific evidence.

**Current answer:** We will add the method/coverage table and move implementation to setup.

### AI.11 — Reviewer asks to move Section 4.1 implementation details

**Status:** Presentation/definition request retained internally; deliberately omitted from export.

**Reviewer request:** Restore a clear sequence of definitions and theoretical results.

**Sources:** W1: 02_REBUTTAL_PLAN.md proposed presentation changes.

**Finding, meaning and limits:** Move solver and layer-scaling details from Section 4.1 to the experimental setup. Keep theory definitions before counterexamples and EF results. Author requested keeping routine presentation/definition work out of the exported rebuttal. The request and planned edit remain in the fragment notes and source mapping; it is unchecked for export, not marked as missing scientific evidence.

**Current answer:** We will add the method/coverage table and move implementation to setup.

### AI.10 — Reviewer asks for a structural condition explaining positive alignment

**Status:** Checked statement; selected.

**Reviewer request:** Give a sufficient condition with explicit scope.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.

**Finding, meaning and limits:** The diagonal lower bound implies alignment by a singular-value-weighted sum. The full covariance criterion and stochastic first-moment theorem are in the author addendum. Credit OLion v2; it is not an empirical explanation for our networks. Revision 6: Explain the positive meaning of the conditional plain SignMuon theorem. The unmeasured network-alignment request AI.20 remains excluded and needs_check=true; this conditional statement does not assert that training trajectories satisfy the condition.

**Current answer:** Our separate theorem proves SignMuon convergence for interacting layers when signing preserves a uniform fraction of the polar update’s alignment with momentum.

### AI.20 — Reviewer asks to connect the structural condition to realistic training

**Status:** Additional evidence pending; excluded from export.

**Reviewer request:** Measure the margin for the actual polar approximation and vote.

**Sources:** G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin. E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** The geometric criterion has not been measured on the actual trajectories. Xiao et al. 2025 concerns SignSGD and does not by itself justify the SignMuon condition.

**Current answer:** The geometric criterion has not been measured on the actual trajectories. Xiao et al. 2025 concerns SignSGD and does not by itself justify the SignMuon condition.

### AI.13 — Reviewer asks for the zero-input oracle selection

**Status:** Presentation/definition request retained internally; deliberately omitted from export.

**Reviewer request:** Define the exact thin-SVD selection and distinguish sign ties.

**Sources:** W1: thin-SVD convention in aaai_article/ef21_signmuon_divergence.tex; code/common/optimizers.py.

**Finding, meaning and limits:** Use polar(0)=0 in the exact theory. For plain strict-binary sign ties at zero, the implementation can randomize; this is distinct from the polar selection. Author requested keeping routine presentation/definition work out of the exported rebuttal. The request and planned edit remain in the fragment notes and source mapping; it is unchecked for export, not marked as missing scientific evidence.

**Current answer:** We will define polar(0)=0 and count EF scales. nanoGPT has 1 run/method; its 0.0021 gap cannot rank methods. A5’s 16.6 s/epoch is not network timing. MuonSign and SGD reach 80% in 5/5 seeds.

### AI.14 — Reviewer questions the single-seed 0.0021 nanoGPT ordering

**Status:** Checked statement; selected.

**Reviewer request:** Do not infer a reliable ranking from that difference.

**Sources:** E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.

**Finding, meaning and limits:** nanoGPT has one training run per compressed method. EF21-SignMuon 3.2860 versus SignMuon 3.2881 is an observed gap of 0.0021, not evidence of a reproducible ranking. SignMuon improves the observed SignSGD loss by 0.1168. No variance estimate is available for the compressed variants.

**Current answer:** nanoGPT has 1 run/method; a 0.0021 gap cannot rank methods. CNN2’s 1.087 bits/parameter includes auxiliaries and EF scales. A5 reports SignMuon 16.6 s/epoch on centralized ResNet18/CIFAR-10. MuonSign and SGD reach 80% in all 5 IID seeds.

### AI.15 — Reviewer asks for timing evidence behind the wall-clock claim

**Status:** Checked statement; selected.

**Reviewer request:** Use submitted timing data with the correct scope.

**Sources:** E4: submitted supplement Table A5, seconds/epoch (SignMuon16.6, SGD12.2, SignSGD12.5, Adam12.7); centralized export table_cifar.csv.

**Finding, meaning and limits:** Elapsed seconds/epoch are already in A5, so timing data are not entirely absent. The old main-table threshold numbers are stale. Withdraw the generic approximately 2x end-to-end speed claim and report per-epoch timing/threshold separately; compilation/validation and network transport differ across settings.

**Current answer:** nanoGPT has 1 run/method; a 0.0021 gap cannot rank methods. CNN2’s 1.087 bits/parameter includes auxiliaries and EF scales. A5 reports SignMuon 16.6 s/epoch on centralized ResNet18/CIFAR-10. MuonSign and SGD reach 80% in all 5 IID seeds.

### AI.16 — Reviewer asks to count scales in one-bit claims

**Status:** Checked statement; selected.

**Reviewer request:** Include one scale per matrix and full-precision auxiliary parameters.

**Sources:** C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. Counts are simulated payloads, not measured packed-network throughput.

**Finding, meaning and limits:** CNN2 has 762560 matrix and 2146 auxiliary parameters and 3 compressed blocks. Bits/channel are 1.0870 plain and 1.0871 EF, not exactly one per total parameter; uplink-only round-trip gain is 1.93x, not 29.4x.

**Current answer:** nanoGPT has 1 run/method; a 0.0021 gap cannot rank methods. CNN2’s 1.087 bits/parameter includes auxiliaries and EF scales. A5 reports SignMuon 16.6 s/epoch on centralized ResNet18/CIFAR-10. MuonSign and SGD reach 80% in all 5 IID seeds.

### AI.17 — Reviewer flags rounds-to-target conditioned on successful seeds

**Status:** Checked statement; selected.

**Reviewer request:** Correct the stale main table and report successes.

**Sources:** E1: submitted Table A8 and code/results/federated_export_results.zip: table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.

**Finding, meaning and limits:** Corrected submitted archive: MuonSign rounds80 860, SGD1240, both 5/5 successes; SignSGD1140 and SignMuon540, also5/5. Do not use the stale main-table 4/5 and3/5 denominators.

**Current answer:** nanoGPT has 1 run/method; a 0.0021 gap cannot rank methods. CNN2’s 1.087 bits/parameter includes auxiliaries and EF scales. A5 reports SignMuon 16.6 s/epoch on centralized ResNet18/CIFAR-10. MuonSign and SGD reach 80% in all 5 IID seeds.

### AI.12 — Reviewer asks to scope impossibility to the analyzed EF repair

**Status:** Checked statement; selected.

**Reviewer request:** Avoid ruling out all repairs and decreasing schedules.

**Sources:** N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; evidence/verified_claims.json exact rational ascent/cycle certificates. Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows from Theorem 4 alone; N4 is separate. N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses for the plain placements. Output-EF decreasing-step and finite-budget results use the separate N4 construction. N4: evidence/ef21_vanishing_snapshot/theorem.tex and certificate.py; evidence/new_quadratic_certificate.json and verification.log. Exact algebraic certificate checks 128 strict sign inequalities; 20 independent 2000-step SVD trajectories pass. For any budget-tuned constant learning rate and fixed EMA/Nesterov momentum, an 8x8 convex quadratic has Frobenius L=1, initial gap 4/7 and gradient norm >=1 at every iterate. The instance may depend on the parameters; the class constants are uniform. Arbitrary decreasing steps to zero are covered without momentum; suitable inverse-power decay is covered for each fixed momentum coefficient. Exact gradients/polar; not a Lean claim.

**Finding, meaning and limits:** Submitted Theorem 4 alone uses fixed parameters and a parameter-dependent objective. The separate added N4 construction now gives a uniform nonstationarity obstruction: Frobenius smoothness 1, initial gap exactly 4/7, dimension 8x8 and gradient norm >=1 at every iterate, for any budget-tuned constant rate and fixed EMA/Nesterov coefficient. The instance may depend on the parameters; constants are uniform across the class. Without momentum it covers any strictly decreasing positive schedule tending to zero. For each fixed momentum it covers suitably shifted power decay. It does not assert arbitrary changing momentum or finite-polar trajectories. The exact certificate is checked; this addition is not in the Lean audit.

**Current answer:** The negative result concerns the specified output-EF scheme.
