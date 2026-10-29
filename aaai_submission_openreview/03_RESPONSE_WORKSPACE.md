# Rebuttal workspace — post-submission additions revision 4

All six replies and the confidential AC comment have been revised. The text marks completed additions as work done after submission or in response to the reviews, states what the extended theorems establish, and explains why the new heterogeneous results change the practical recommendation. Credit-only sentences have been removed from the exports. All five human replies retain conditional higher-score requests; the AI reply has none.

## Current drafts

Open `start_workspace.cmd`, or read `rebuttals/rebuttal_*.md`.

| Response | UTF-16 characters /1500 | Checked concerns selected | Requests still needing evidence |
| --- | ---: | ---: | --- |
| R1 ·7S1b | 1491 | 7/10 | R1.6, R1.7, R1.8 |
| R2 ·D4r2 | 1482 | 6/8 | R2.2, R2.8 |
| R3 ·Ueq5 | 1473 | 5/5 | None |
| R4 ·pqje | 1499 | 14/15 | R4.15 |
| R5 ·Tx34 | 1498 | 6/6 | None |
| AI reviewer | 1493 | 18/20 | AI.6, AI.20 |

## Shared heterogeneous table

**The full table is in the reply to 7S1b, Question 2.** It contains all 10 tested methods, with accuracy mean±sample SD over 5 seeds. All other replies and the AC comment refer to that reviewer/question. There are no external links in the exports.

The added test uses CNN2/CIFAR-10, 11 clients, Dirichlet(0.1), 2000 rounds, no local updates and the IID-selected learning rates. EF21-MuonSign reaches 83.23±0.36%, EF21-MuonUSign 82.99±0.46%, SGD 81.03±0.62%, Muon 46.02±8.55% and SignMuon 35.92±8.49%. Gradient-side EF leads this transfer comparison. Retuning is pending; the results do not rank separately retuned heterogeneous optima.

The table is generated from the previously verified 50-run archive. The archive hash was rechecked; no training or full Lean build was run during this writing revision.

## Theory and timing

- Submitted ascent/divergence results are identified as submitted results. Their Lean verification and the EF21 momentum/schedule extensions are described as additions after submission.
- In response to the reviews, the replies explain that EF21 convergence was extended to fixed EMA/Nesterov and deterministic learning rates, including constant/cosine schedules. Core tracking/convergence proofs are verified in Lean. The printed cosine sums are manual compositions of formalized results.
- EF21 still retains separable layers and a fixed momentum coefficient. Actual numerical orthogonalization needs a quality/error bound. nanoGPT’s changing momentum and auxiliary Adam updates need further analysis.
- The separate post-submission SignMuon theorem permits interacting layers under uniform alignment. It is not claimed Lean-verified; the network condition remains unmeasured.
- The squared-gradient rate has a separate proof with gradient-dependent steps. An expected-norm rate cannot simply be squared.

## Remaining evidence and preservation

The 8 evidence flags remain: network alignment (R1.6, R2.8, R4.15, AI.20), additional federation/models and retuning (R1.7, R2.2), nanoGPT seeds (R1.8), matched SSVR experiments (AI.6). Local updates, partial participation and retuned heterogeneous optima remain untested. No missing experiment is marked complete.

The prior author state is preserved in `response_history/before_review_additions_revision_20261029T202946Z/`. IDs, source passages, selections and pending flags are preserved. R5 advances to revision 10; all other replies also advance by one revision.

The independent confidential draft has 1949 UTF-16 characters. Its actual field limit/readers remain unknown; configuration keeps the limit `null`. Check the destination form before using it.

## Checks and collaboration

Build, inspection of all 7 editor pages and workspace checks passed. The table renders as columns and its cells match the 10 verified results. The 6 portable checks and the relocated browser save checks also passed. Results are recorded in `evidence/revision_4_validation.json`. After further edits, build with `python -X utf8 aaai_submission_openreview/primer/workspace.py build`, inspect the pages and run the same command with `check`. The evidence ledger is `02_REBUTTAL_PLAN.md`.

Save browser edits before pulling or merging, coordinate edits to `response_fragments.json`, and restart the editor after rebuilding. Saves are local; nothing is submitted automatically. The configured deadline remains 31 October 2026 AoE. Read `README.md` for the collaboration workflow.

## Separate completed quadratic extension

The addendum changed while this writing revision was in progress. The new exact-polar EF21-SignMuon construction has been checked and is now stated in Tx34 Q2, the AI scope answer and the AC comment. For every budget-tuned constant learning rate and fixed EMA/Nesterov coefficient, an 8×8 convex quadratic has Frobenius smoothness 1, initial gap 4/7 and gradient norm at least 1 at every iterate. The family has uniform constants; the objective can depend on the parameters. Decreasing schedules are covered without momentum, and suitable power decay with each fixed momentum coefficient. This is a separate result from submitted Theorem 4.

Frozen proof/certificate bytes and the passing log are in `evidence/ef21_vanishing_snapshot/`. The exact certificate checks all 128 strict sign inequalities; 20 numerical optimizer trajectories corroborate the recurrence. This new construction is not part of the Lean audit. The preparation did not edit the live scientific sources.
