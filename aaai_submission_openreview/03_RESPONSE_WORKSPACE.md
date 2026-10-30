# Rebuttal workspace — exploration and proof clarity, revision 9

The contribution paragraphs describe the exploration of combining Muon with sign compression and error feedback, and the observed difference between centralized deep learning and heterogeneous federated learning. The extra SignMuon–Muon percentage comparison has been removed from the table-correction replies; the requested MuonSign–SignSGD correction remains.

## Current drafts

Open `start_workspace.cmd`, or read `rebuttals/rebuttal_*.md`. The full ten-method tuned heterogeneous table remains in **7S1b Q2**, with a natural cross-reference from every other reply and the AC comment. Five human replies retain requests to raise the score; the AI reply has no score request.

| Reviewer | UTF-16 characters /1500 | Selected concerns | Missing evidence |
| --- | ---: | ---: | --- |
| 7S1b | 1476 | 7 | R1.6, R1.7, R1.8 |
| D4r2 | 1385 | 2 | R2.2, R2.8 |
| Ueq5 | 1389 | 5 | None |
| pqje | 1460 | 14 | R4.15 |
| Tx34 | 1486 | 5 | None |
| AI_Reviewer | 1482 | 14 | AI.6, AI.20 |

## Counterexample explanation

The Tx34 answer now identifies the gradient projection onto a fixed matrix direction in parameter space. Compressed updates are perpendicular to that direction, leaving this projection unchanged and the full gradient norm at least 1. It then separates two extensions of fixed-parameter Theorem 4: failure with decreasing learning rates and failure of constant learning-rate/momentum choices tuned to a finite budget. The latter keeps smoothness and initial gap fixed across the examples.

Decreasing-step statements retain their actual scope: any strictly decreasing schedule to zero without momentum, or shifted power decay with fixed EMA/Nesterov. Exact polar updates are explicit. These construction proofs have their existing exact-certificate status; this prose revision adds no Lean claim.

## Evidence and preservation

SignMuon leads the tested centralized ResNet18/CIFAR-10 comparison by mean accuracy. In the completed heterogeneous CNN2/CIFAR-10 study (11 Dirichlet(0.1) clients, 2000 rounds, no local updates), only EF21-MuonSign and EF21-MuonUSign exceed 80% after validation LR tuning: 84.03±0.52% and 83.84±0.32% over five final seeds. The immutable archive and 167-run audit are unchanged. Centralized performance is described as an empirical result, without asserting that every network update descends.

EF21 retains the submitted separable-layer assumption; the separate SignMuon theorem permits interacting layers under joint smoothness and uniform alignment. Existing proof verification, source reviews, all 64 IDs and 79 exact passage mappings remain unchanged. So do the 47 selections, 9 editorial omissions and 8 missing-evidence flags. Network alignment, local-update/participation extensions, more large-model seeds and the SSVR experiment remain open.

The pre-edit JSON, exports and handoff documents are backed up in `response_history/before_exploration_revision_20261030T083924Z/`. Existing author changes in `primer/` are preserved. The confidential draft has 2915 UTF-16 characters; its field limit/readers remain unknown and the limit stays `null`.

## Checks and collaboration

Build with `python -X utf8 aaai_submission_openreview/primer/workspace.py build`, inspect the seven editor pages, then run the same command with `check`. Completed checks are recorded in `evidence/revision_9_validation.json`. All six public drafts fit 1500 characters. This prose update requires no training, Lean build or addendum rebuild.

Save browser edits before merging, coordinate changes to `response_fragments.json`, and reload after rebuilding. Saves are local; nothing is submitted or pushed. The configured deadline remains 31 October 2026 AoE.
