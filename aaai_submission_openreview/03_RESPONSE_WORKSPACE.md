# Rebuttal workspace — focused scientific replies, revision 5

All six replies and the AC comment have been revised around the scientific answers. Results name the model, dataset and regime. Routine presentation promises are excluded from exports and retained in notes. The human score requests remain active; the AI reply has none.

## Current drafts

Open `start_workspace.cmd`, or read `rebuttals/rebuttal_*.md`.

| Response | UTF-16 characters /1500 | Selected concerns | Missing evidence | Deliberately omitted presentation requests |
| --- | ---: | ---: | --- | --- |
| R1 ·7S1b | 1496 | 7/10 | R1.6, R1.7, R1.8 | None |
| R2 ·D4r2 | 1462 | 2/8 | R2.2, R2.8 | R2.1, R2.4, R2.5, R2.7 |
| R3 ·Ueq5 | 1490 | 5/5 | None | None |
| R4 ·pqje | 1497 | 14/15 | R4.15 | None |
| R5 ·Tx34 | 1496 | 6/6 | None | None |
| AI reviewer | 1473 | 15/20 | AI.6, AI.20 | AI.9, AI.11, AI.13 |

## What changed

- The 85.72% accuracy is explicitly federated CNN2/CIFAR-10, 11 IID clients, 5 seeds. Centralized ResNet18/CIFAR-10 and nanoGPT validation losses are labeled separately.
- Both EF results are bold in the full table in **7S1b Q2**. The conclusion states that EF21-Muon(U)Sign are the only Muon variants that work well here, with the criterion made concrete as above 80% accuracy. The reused IID rates remain in the caption; no local updates are used. SGD also exceeds 80%, so the scope is deliberate.
- The softplus paragraph explains how the linear witnesses were made lower bounded and why their ascent directions survive. One fixed objective per plain method defeats all positive step schedules.
- Replies report the post-submission EF21 momentum/schedule extensions, conditional SignMuon result and new quadratic output-EF obstruction. Core EF21 and original witness Lean proofs remain distinct from the analytic softplus and N4 additions.
- Presentation-only commitments are kept in notes rather than taking response space. The 7 deliberately unchecked concerns retain their IDs, links and planned edits. The 8 missing-evidence flags remain unchanged.

## Evidence and limits

The shared heterogeneous table is generated from the previously verified 50 runs: CNN2/CIFAR-10, 11 Dirichlet(0.1) clients, 2000 rounds, no local updates, IID-selected rates. Retuned heterogeneous optima, local updates and partial participation remain untested. The network alignment condition is not measured. nanoGPT has one run/method and uses full-precision gradient averaging. SSVR remains an untested matched-query comparison.

The frozen N4 proof/certificate is in `evidence/ef21_vanishing_snapshot/`. It gives Frobenius smoothness 1, initial gap 4/7 and gradient norm at least 1 for budget-tuned constant rates/fixed EMA or Nesterov. Decreasing schedules are covered as stated in its proof. N4 and the softplus extension are not claimed Lean-verified. The existing 45-statement Lean audit covers the documented core results. No training or full Lean build was run for this prose revision.

## Preservation and checks

The source author JSON, evidence records and prior exported replies are backed up in `response_history/before_focused_revision_20261029T212828Z/`. All 64 concern IDs, 79 exact passage mappings, review wording, question references and pending flags are preserved. Selection changes are limited to the 7 presentation/definition omissions above; every response revision advances by one.

The independently editable confidential draft has 2145 UTF-16 characters. Its actual field limit/readers remain unknown; the configured limit stays `null`.

Build with `python -X utf8 aaai_submission_openreview/primer/workspace.py build`, inspect the editor pages and run the same command with `check`. Build and check passed; all 7 editor pages passed export, math, anchor and desktop/mobile checks. Both EF rows and the conclusion render bold. All 6 portable checks and the relocated browser/save checks passed. The record is `evidence/revision_5_validation.json`. The canonical Markdown and character counts retain the bold markup.

Save browser edits before pulling or merging, coordinate changes to `response_fragments.json`, and reload after rebuilding. Saves are local; nothing is submitted automatically. The configured deadline remains 31 October 2026 AoE. Read `README.md` for collaboration instructions.
