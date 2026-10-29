# Optional confidential AC/SPC comment — revision 4

## Purpose and content

The comment asks the AC/SPC to weigh strong sign-based performance, the newly added heterogeneous results and the precise theorem extensions together. It foregrounds the completed Dirichlet(0.1) transfer tests and refers to the full 10-method table in the reply to **7S1b, Q2**. Gradient-side EF reaches about 83% under the IID-selected rates; retuning remains pending.

The comment distinguishes submitted results from post-submission work. It identifies the fixed EMA/Nesterov and deterministic-schedule EF21 extensions, their core Lean proofs, and the separate conditional SignMuon theorem. EF21 separability remains; the conditional theorem is not claimed Lean-verified. The submitted failure theorems and their Lean verification are stated explicitly.

It also resolves the concrete A8/main-table discrepancy raised by Ueq5 and pqje and scopes Theorem 4 to fixed parameters. It does not repeat credit-only sentences, infer optimizer rankings from worst-case counterexamples or claim missing experiments are complete.

## Use and limits

The current comment has 1949 UTF-16 characters including its final newline. Its actual field limit/readers remain unknown, so `workspace.json` keeps its limit `null`. Confirm those destination details before posting. The comment has not been posted.

Local references in `workspace.json` now include 7S1b Q2 as well as the relevant theory/table concerns; these create workspace links without adding URLs to the export. The independent source is `confidential_comment.json`. Public replies each retain their own 1500-character allowance.

The previous comment, strategy and complete author state are backed up in `response_history/before_review_additions_revision_20261029T202946Z/`. Saving is local. The scientific evidence is in `02_REBUTTAL_PLAN.md`, and verification is recorded in `evidence/revision_4_validation.json` after the build/browser checks.

## Separate completed quadratic extension

The addendum changed while this writing revision was in progress. The new exact-polar EF21-SignMuon construction has been checked and is now stated in Tx34 Q2, the AI scope answer and the AC comment. For every budget-tuned constant learning rate and fixed EMA/Nesterov coefficient, an 8×8 convex quadratic has Frobenius smoothness 1, initial gap 4/7 and gradient norm at least 1 at every iterate. The family has uniform constants; the objective can depend on the parameters. Decreasing schedules are covered without momentum, and suitable power decay with each fixed momentum coefficient. This is a separate result from submitted Theorem 4.

Frozen proof/certificate bytes and the passing log are in `evidence/ef21_vanishing_snapshot/`. The exact certificate checks all 128 strict sign inequalities; 20 numerical optimizer trajectories corroborate the recurrence. This new construction is not part of the Lean audit. The preparation did not edit the live scientific sources.
