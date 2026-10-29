# Prepared rebuttal workspace — revision 2

Prepared 29 October 2026 for AAAI-27 submission 48060. The prior collaboration workspace is committed as `b60118b`; this revision corrects the separability claim and strengthens the response structure. No material has been submitted to OpenReview or pushed. The replies stand alone without access to the working article.

## Current drafts

Open `start_workspace.cmd`. Copy-ready text is in `rebuttals/rebuttal_*.md`; evidence and concrete article revisions are in `02_REBUTTAL_PLAN.md`.

| Response | UTF-16 characters /1500 | Checked concerns selected | Unfinished requests excluded |
| --- | ---: | ---: | --- |
| R1 ·7S1b | 1477 | 7/10 | R1.6, R1.7, R1.8 |
| R2 ·D4r2 | 1443 | 6/8 | R2.2, R2.8 |
| R3 ·Ueq5 | 1432 | 5/5 | None |
| R4 ·pqje | 1497 | 14/15 | R4.15 |
| R5 ·Tx34 | 1468 | 6/6 | None |
| AI reviewer | 1491 | 18/20 | AI.6, AI.20 |

There are 64 concerns and 79 exact source links. All 5 human replies now have active conditional requests to raise the score; the AI reply has none. All displayed answer headings are numbered. Counts include every heading, newline, space and LaTeX source character. Shared answers cover related concerns without repeated text.

## What changed

- **Corrected the scientific claim:** revised EF21 proofs retain layer separability. Only the added conditional plain SignMuon theorem uses joint smoothness and allows interacting layers. The separate candidate EF21 note is excluded from rebuttal claims.
- **Restored the contribution:** SignMuon’s strong measured quality at a sign budget, compared systematically across 3 sign placements and their EF variants, anchors the replies. The theory explains sufficient alignment and specific failure mechanisms; it does not predict the experimental ranking.
- **Added concrete evidence:** 85.72% versus SignSGD 81.44% in federation; 94.60% versus 93.37% centrally; nanoGPT loss 3.2881 versus 3.4049. Vision values come from already-submitted A5/A8 and logs. Main tables will be synchronized, including MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%.
- **Removed loaded uncertainty wording:** nanoGPT is described precisely as one run per method; no ranking is based on its 0.0021 SignMuon/EF21-SignMuon gap. The supplied transfer runs are separately identified; no extra nanoGPT seeds are claimed.
- **Kept mathematical scope explicit:** separate first-/second-moment proofs, gradient-scaled sharp steps, residuals with fixed practical parameters, exact-polar assumptions and fixed-parameter Theorem 4 quantifiers.

## Evidence still missing

| Request | Visible flags | Completion evidence |
| --- | --- | --- |
| Network alignment/geometry | R1.6, R2.8, R4.15, AI.20 | Measure the full uniform margin on actual momentum/oracle inputs, approximate polar outputs and votes. |
| Broader federation/models | R1.7, R2.2 | LR-retuned label-skew, local-update/partial-participation or broader-model runs. Five-seed non-IID transfer is now available. |
| More compressed nanoGPT seeds | R1.8 | Independent training runs, not repeated evaluation or upstream Muon variance. |
| Variance-reduced sign baseline | AI.6 | Matched SSVR implementation, query budgets and independent runs. |

New transfer evidence supplied during the audit gives EF21-MuonSign 83.23±0.36% versus SignMuon 35.92±8.49% under Dirichlet(0.1) with IID rates unchanged. All 50 runs were independently recomputed; LR retuning remains pending. R1 and AI federation answers, the AC comment and the ledger include this evidence. The original SignMuon performance claim is scoped to the submitted experiments.

The manuscript’s EF21 cross-layer extension, a finite-polar error certificate, and the changing-momentum/auxiliary-optimizer full-run analysis also remain open. Checked replies clarify these scopes without marking missing evidence complete.

## Optional AC/SPC draft

The independent confidential draft has 1374 UTF-16 characters. Its field limit/readers remain unknown (`null` in configuration); check the actual form before using it. It emphasizes measured contribution, submitted-log corrections and precise theory. See `04_CONFIDENTIAL_COMMENT_STRATEGY.md`.

## Verification and preservation

Revision 1 editable files and exports were backed up before replacement in `response_history/before_revision_2_20261029T181259Z/`. Review IDs, exact source passages, original PDF/ZIP/JSON and annotations are unchanged. Article/addendum and training code are left intact by this preparation. The addendum changed concurrently to report transfer results; those supplied results were audited and incorporated. `evidence/revision_2_audit.md` records the proof distinction and sources.

Run from the repository root:

```powershell
python -X utf8 aaai_submission_openreview/evidence/verify_rebuttal_evidence.py
python -X utf8 aaai_submission_openreview/primer/workspace.py build
python -X utf8 aaai_submission_openreview/evidence/inspect_browser.py
python -X utf8 aaai_submission_openreview/primer/workspace.py check
```

Revision 2 passed original input hashes, five-seed statistics, 175 unchanged submitted histories, exact ascent/cycle certificates, all 50 supplied transfer histories, workspace build/check and all 7 browser pages. Exports/counts, rendered math, local anchors and desktop/mobile layout passed, with no JavaScript exceptions or author-data changes. Review files, IDs and annotations/rules are preserved. The concurrently updated addendum was rechecked; EF21 separability remains. Details: `evidence/revision_2_validation.json` and `evidence/browser_inspection.json`. Preparation scripts are historical migrations with overwrite guards; continue editing the JSON/browser rather than rerunning them.

## Editing and sharing

Double-click `start_workspace.cmd` and leave its Python window open. The server saves source JSON and valid Markdown locally, with history and revision-conflict detection. Browser storage uses localStorage. Save before pulling/merging; coordinate edits to the shared `response_fragments.json`, then rebuild/check and restart the editor. Read `README.md` for the portable workflow. A valid export does not imply that every concern is resolved.

The configured author-supplied deadline is 31 October 2026 AoE. Nothing saves to OpenReview automatically.
