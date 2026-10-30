# Rebuttal workspace — proofreading revision 8

All six public replies and the AC comment have been proofread for direct, self-standing explanations. The public contribution paragraphs now lead with the comparisons of sign/error-feedback placement and their theoretical results. Tx34's reply explains why the quadratic construction prevents convergence. The pqje and AI replies identify Theorem 5 and Corollary 1 from the submitted supplement and define the roles of server X and broadcast W.

## Current drafts

Open `start_workspace.cmd`, or read `rebuttals/rebuttal_*.md`. The full ten-method heterogeneous table remains in **7S1b Q2**, referenced from every other reply and the AC comment. Inline bold claims are removed; EF table rows remain highlighted. Five human replies retain conditional requests to raise the score; the AI reply has no score request.

| Reviewer | UTF-16 characters /1500 | Selected concerns | Missing evidence |
| --- | ---: | ---: | --- |
| 7S1b | 1456 | 7 | R1.6, R1.7, R1.8 |
| D4r2 | 1385 | 2 | R2.2, R2.8 |
| Ueq5 | 1435 | 5 | None |
| pqje | 1495 | 14 | R4.15 |
| Tx34 | 1496 | 5 | None |
| AI_Reviewer | 1480 | 14 | AI.6, AI.20 |

## Evidence and scope

The completed heterogeneous study remains CNN2/CIFAR-10, 11 Dirichlet(0.1) clients, 2000 rounds, no local updates, momentum 0.9. Learning rates use validation seed 0 and 5000 held-out images. Five final seeds retrain on all 50000 images. EF21-MuonSign gives 84.03±0.52% and EF21-MuonUSign 83.84±0.32%, the only tuned means above 80%. The immutable archive and independent 167-run audit are unchanged.

The N4 explanation follows the checked theorem: a small alternating gradient component creates alternating polar factors; the compressed estimator stays orthogonal to a fixed nonzero gradient component. Thus the gradient norm stays at least 1, with fixed smoothness and initial gap across the examples. Constant rates/fixed momentum chosen for the run length and the documented decreasing schedules are covered with exact polar updates.

EF21 retains the submitted separable-layer assumption. The separate conditional SignMuon theorem permits interacting layers. Complete EF21 tracking/convergence proofs and original failure results retain their existing Lean scope. The printed cosine substitutions, conditional SignMuon theorem, softplus extension and N4 construction keep their previously documented verification status. This proofreading does not change theory or experiment sources, or the addendum/PDF.

## Remaining work and preservation

The same eight evidence flags remain: network alignment measurements, broader local-update/participation settings, additional large-model seeds and the matched-budget SSVR comparison. R5.6 and AI.12 are now omitted at the author's request to cut defensive qualifications. There are 47 selected concerns and 9 editorial omissions; all 64 IDs and 79 exact passage mappings remain.

The pre-edit JSON, exports and handoff documents are saved in `response_history/before_proofreading_20261030T081326Z/`. Existing author changes in `primer/` are preserved. The confidential comment has 2762 UTF-16 characters; its field limit/readers remain unknown, so the configured limit stays `null`.

## Checks and collaboration

Rebuild with `python -X utf8 aaai_submission_openreview/primer/workspace.py build`, inspect the editor pages, then run the same command with `check`. Completed source-preservation, count and browser checks are recorded in `evidence/revision_8_validation.json`. No new training, Lean build or addendum rebuild is needed for this prose-only update.

Save browser edits before merging, coordinate changes to `response_fragments.json`, and reload after rebuilding. Saves are local; nothing is submitted or pushed. The configured deadline remains 31 October 2026 AoE.
