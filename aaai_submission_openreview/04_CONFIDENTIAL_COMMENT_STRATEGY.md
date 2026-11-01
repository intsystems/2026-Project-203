# Confidential AC/SPC comment — approved final draft

The author confirmed separate limits of 1,500 UTF-16 characters for the comment
and 250 for its short title. The approved body is 1,492 characters including its
final newline; the title is 63. Blank lines are included in those counts.

The comment connects each concern to the corresponding rebuttal: novelty and
prior-work credit (Tx34 Q1, pqje); the 8×8 finite-budget construction (Tx34 Q2);
the smooth, lower-bounded extensions (D4r2); heterogeneous tests and method choice
(7S1b Q2); conditional SignMuon convergence (7S1b Q1, D4r2); EF21 theorem coverage
(Ueq5, pqje Q1); and table/rate corrections (Ueq5, pqje Q2). It retains the
remaining alignment, separability and exact-polar qualifications. The closing
expresses hope that the novelty is sufficient and the paper timely.

The progress claim is an assessment of these results, not a claim that a global
state-of-the-art benchmark has been established. Worst-case theorems do not
predict empirical rankings. No identifying dates of the authors' own work are
included in the comment.

`confidential_comment.json` stores the body; `workspace.json` stores
`confidential.export_title` and the separate limits. The Markdown export retains
the title as an H1 without including it in the body count. Direct Markdown edits
can be reconciled using `evidence/sync_confidential_draft.py --dry-run`, then
`--apply`, with backups and an increased review revision. Previous drafts and
the pre-import state remain in `response_history/`.

## Archived revision 14 strategy

The AAAI-27 email allows one Confidential Comment to the SPC and AC. The author confirmed that the field is limited to 1500 characters, and `workspace.json` now enforces it. The draft is 1497 UTF-16 characters.

The comment is written before the discussion, so it is a guide for the AC, not a complaint. It borrows the opening and closing of the authors' ICML comment ("Thank you for your time and consideration"; "We fully respect the decision-making process") and tells the contribution story in four paragraphs:

1. Contribution and novelty: Muon with sign compression at one bit per parameter; sign and the polar factor each descend alone, yet all three placements can move uphill (Theorems 1–3); error feedback on the LMO output can diverge because the polar factor can change by Θ(1) per step (Theorem 4), while on the gradient it works. The placements, methods and Theorems 1–4 are new; the EF21 rates follow from EF21-Muon. SignMuon (Mishra et al.) is a May 2026 preprint developed independently of this work, without identifying details of our own work.
2. The shared theory–practice concern (7S1b, D4r2, pqje): the SignMuon convergence condition, and the heterogeneous results in which only EF21-MuonSign (84.03%) and EF21-MuonUSign (83.84%) keep their IID accuracy, while SignMuon falls from 85.72% to 42.96%.
3. Tables (Ueq5, pqje): the submitted logs reproduce A5/A8, and main Tables 1–2 are corrected; the Theorem 4 extension (Tx34).
4. Closing: respect for the decision process.

It avoids score arithmetic, the AI review, our own dates, and any claim that the centralized success of SignMuon is explained in networks. Source: `confidential_comment.json`; export: `rebuttals/confidential_ac_comment.md`. The pre-edit state is in `response_history/before_revision_14_icml_style_20261030T214955Z/`.
