# Revision 2 scientific and editorial audit

## The separability correction

The previous answer conflated a separate agent-generated candidate with manuscript results. The statement that the revised EF21 analysis “removes layer separability” was not supported by the paper.

1. `aaai_article/signmuon_body.tex`, Assumption 2: block-only Lipschitz control for all tuples forces each gradient block to be independent of the other blocks.
2. `aaai_article/ef21_fixed_radius.tex`: the tracking proof explicitly invokes this independence for simultaneous blocks.
3. `aaai_article/ef21_nesterov_convergence.tex`: the theorem retains Assumptions 1–3.
4. `aaai_article/signmuon_addendum_body.tex`, “EF21 cross-layer coupling: open target”: explicitly says the current block-separable proofs do not establish the joint-smooth extension.
5. The same author addendum’s **conditional plain SignMuon theorem** does assume joint Frobenius smoothness and permits interacting layers. Under a uniform positive alignment margin, conditional unbiasedness/finite variance and exact polar, it gives an average expected nuclear-gradient-norm $O(T^{-1/4})$ bound with horizon-tuned EMA/Nesterov.

Accordingly, the replies and AC comment retain EF21 separability and distinguish the plain SignMuon theorem. The candidate `EF21_JOINT_SMOOTHNESS.md` is preserved for internal review and excluded from public claims. This audit does not assert that no joint-smooth EF21 theorem could be proved.

## Contribution and numbers

The rebuttal centers the supported practical contribution: SignMuon achieves Muon-level quality at a sign budget and leads the tested sign methods in the submitted vision experiments, within a controlled comparison of three sign placements and their EF variants. Prior/concurrent SignMuon and the general EF21 framework are credited. Global superiority over untested sign methods is not claimed.

- Centralized A5/original export: SignMuon 94.60±0.15%, SignSGD 93.37±0.26%, Muon 94.35±0.27%; 3 seeds.
- Federated A8/original five-seed logs: SignMuon 85.72±0.24%, SignSGD 81.44±0.15%, Muon 85.98±0.26%. Exact SignMuon/SignSGD difference is 4.2800 points.
- MuonSign is 82.94±0.19%, above SignSGD by 1.5068 points, reversing the submitted main-table sentence. The submitted supplement/logs already contain the corrected values.
- nanoGPT: SignMuon 3.2881 versus SignSGD 3.4049, observed gain 0.1168 nats; Muon 3.2785. One training run/method. The 0.0021 gap to EF21-SignMuon is not used to rank those methods.

The main PDF, revised body, supplement, original archives and recorded exact certificates were checked. Current article/source files were read and not edited. No training was run. No additional nanoGPT-seed, geometry or SSVR result was invented. New heterogeneous transfer results were supplied during the audit and independently verified; they are distinguished from submitted data and pending LR retuning.

## Response structure and tone

All displayed answer headings are numbered. All five human replies have active, conditional higher-score requests; the AI response does not. Each human reply includes quantitative evidence relevant to its questions. Loaded uncertainty adjectives are replaced by exact study scope; immutable reviewer quotes and historical drafts are preserved. Export counts include source LaTeX and final newlines and must remain within 1500 UTF-16 characters.

Theorem 4 retains its fixed-parameter, parameter-dependent-objective quantifiers. First- and second-moment rates use distinct proofs and tunings. The revised ideal EF21 deterministic-schedule bounds retain exact polar, fixed-within-run momentum and separability; fixed parameters give residuals. Actual nanoGPT uses changing momentum, auxiliary Adam and full-precision collectives.

## Concurrent addendum update and transfer evidence

The author addendum gained heterogeneous transfer results during the audit. Its EF21 coupling extension remains an open target. The supplied 185538-byte archive is preserved as `heterogeneous_transfer_results.zip`; `verify_heterogeneous_transfer.py` checks all 50 complete run configs/histories and reproduces the means/sample SDs. At unchanged IID rates, EF21-MuonSign has 83.23±0.36% versus SignMuon 35.92±8.49%; LR retuning is pending. R1, AI and the optional AC comment now include this evidence. No training was run by this rebuttal preparation.
