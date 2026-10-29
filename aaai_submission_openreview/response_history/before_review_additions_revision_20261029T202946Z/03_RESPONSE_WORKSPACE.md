# Rebuttal workspace — readability revision 3

All six replies have been rewritten in full sentences. They explain what we proved, what it means and how it answers the review. Headings now say “Re our contribution.” Numbers and conditional higher-score requests remain in all five human replies; the AI reply has no score request. The replies stand alone without access to the revised article.

## Current drafts

Open `start_workspace.cmd`, or read `rebuttals/rebuttal_*.md`.

| Response | UTF-16 characters /1500 | Checked concerns selected | Requests still needing evidence |
| --- | ---: | ---: | --- |
| R1 ·7S1b | 1492 | 7/10 | R1.6, R1.7, R1.8 |
| R2 ·D4r2 | 1494 | 6/8 | R2.2, R2.8 |
| R3 ·Ueq5 | 1497 | 5/5 | None |
| R4 ·pqje | 1497 | 14/15 | R4.15 |
| R5 ·Tx34 | 1492 | 6/6 | None |
| AI reviewer | 1492 | 18/20 | AI.6, AI.20 |

## Theory stated clearly

- We prove that every sign placement can produce an ascent direction, and that error feedback after orthogonalization can diverge on a smooth, lower-bounded objective. The original failure results are verified in Lean.
- We generalized the EF21 convergence analysis to EMA and Nesterov momentum with deterministic learning rates. Its underlying tracking/convergence proofs are verified in Lean; constant and cosine schedules are specializations. Printed cosine sums are composed manually, so the replies do not claim every printed corollary is a complete exported Lean declaration.
- The bounds account for noise and numerical-direction error. The actual approximate solver still needs a certified error/quality bound. Fixed momentum and separable layers remain assumptions of these EF21 results.
- The separate SignMuon theorem permits interacting layers if signing retains enough alignment with momentum. It is not claimed Lean-verified; the network alignment measurements remain outstanding.
- The squared-gradient rate uses a separate proof with gradient-dependent steps. A bound on the expected norm cannot simply be squared.

## Evidence and remaining work

The existing submitted logs/A5/A8 support all quoted vision numbers; the table correction remains MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%. The new transfer archive contains 50 checked runs with IID rates held fixed: EF21-MuonSign 83.23% versus SignMuon 35.92% under Dirichlet(0.1). LR retuning remains pending. nanoGPT has 1 run per method and tests centralized updates after full-precision averaging.

The 8 evidence flags remain visible: network alignment (R1.6, R2.8, R4.15, AI.20), broader federation/models and retuning (R1.7, R2.2), additional nanoGPT seeds (R1.8), and matched SSVR experiments (AI.6). No missing experiment is marked complete by the wording revision.

The Lean audit record passed 45 frozen statements on 23 October 2026. Reference and dependency hashes were rechecked for this edit; the full Lean build was not rerun. See `evidence/revision_3_readability.md` for the precise formalization scope.

## Preservation and checks

The current author state, including R5 revision 8, was backed up before editing in `response_history/before_readability_revision_20261029T200001Z/`. Review IDs, source passages, selections and pending flags are preserved. R5 advances to revision 9; each other reply also advances from its current revision.

The independently editable confidential draft has 1377 UTF-16 characters. Its actual field limit/readers remain unknown; check the destination form before using it.

Build, browser inspection and workspace checks passed for this revision. All 7 editor pages have matching exports and counts, working local links, rendered math and no desktop/mobile overflow or JavaScript exceptions. Results are recorded in `evidence/revision_3_validation.json`. After further edits, rebuild with `python -X utf8 aaai_submission_openreview/primer/workspace.py build`, inspect the pages and run the same command with `check`. The evidence ledger is `02_REBUTTAL_PLAN.md`.

Save browser edits before pulling or merging, coordinate work on `response_fragments.json`, and restart the editor after rebuilding. Saves are local. Nothing is submitted to OpenReview automatically; the configured deadline remains 31 October 2026 AoE. Read `README.md` for the collaboration workflow.
