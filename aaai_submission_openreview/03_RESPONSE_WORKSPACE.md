# Rebuttal workspace — final confidential comment and synchronized exports

Rules are now confirmed by the AAAI-27 author-response email: one response per review, at most 1500 characters each; one Confidential Comment to the SPC and AC; no external links (papers with links in responses are rejected); no uploads; editable until 31 October 2026 AoE.

## Current drafts

Open `start_workspace.cmd` or read `rebuttals/rebuttal_*.md`. The ten-method IID-versus-heterogeneous table is in **7S1b Q2**; other replies refer to it.

| Reviewer | UTF-16 characters /1500 | Selected concerns | Missing evidence |
| --- | ---: | ---: | --- |
| 7S1b | 1492 | 7/10 | R1.6, R1.7, R1.8 |
| D4r2 | 1494 | 6/8 | R2.2, R2.8 |
| Ueq5 | 1491 | 5/5 | None |
| pqje | 1494 | 14/15 | R4.15 |
| Tx34 | 1496 | 6/6 | None |
| AI_Reviewer | 1497 | 18/20 | AI.6, AI.20 |
| AC/SPC body | 1492 | Separate title: 63/250 | None selected |

## Final confidential comment and repository cleanup

The approved Markdown is preserved byte for byte. It now leads with progress
toward optimization with sign compression and the centralized/federated SignSGD
comparisons. Reviewer references connect novelty, the 8×8 construction, the
smooth lower-bounded extensions, the heterogeneous tests, conditional SignMuon
convergence, theorem coverage, and table/rate corrections. The closing asks the
AC/SPC to consider the novelty and timeliness.

The body is synchronized to `confidential_comment.json` at review revision 16;
the separate title is `confidential.export_title` in `workspace.json`.
Rebuilds and both save modes retain the title without counting it against the
body limit. `evidence/sync_confidential_draft.py` provides a dry-run import with
backups. The eight pre-existing evidence flags above remain open and excluded;
none was relabeled as resolved during cleanup.

Current checks and immutable-input hashes are recorded in
[`evidence/repository_cleanup_validation.json`](evidence/repository_cleanup_validation.json).

The original 668 kissing-number research files were relocated, with every file
verified by SHA-256, to `D:/GitHubD/KissingNumber27/research/kissing43/`. Its
transfer receipt is stored there. Local research jobs may recreate the old path;
it is excluded from SignMuon history. The ICML writing example is in
`reference_materials/`.

## What changed and why

Each reply now answers the reviewer's literal question in its first sentence, concedes valid points plainly, and labels post-submission results.

- **7S1b:** the federated ordering does not persist. The table shows IID next to Dirichlet(0.1). It explains that methods applying sign or polar per client before averaging lose 35–45 points, while the guaranteed EF21 methods, which orthogonalize once on the server, lose nothing. That is consistent with Theorem 5, which has no client-dissimilarity bound.
- **D4r2:** their one required item, the writing list, was previously unanswered. All five points now receive commitments; the renaming is declined politely in favour of a placement table.
- **Ueq5:** the reply does exactly what their justification asks: reconciles the tables (including Table 1 against A5) and separates the two rates. The squaring error is conceded first.
- **pqje:** Q1 is answered table by table (none fully covered as submitted; Tables 1–2 covered by the new bounds for exact polar updates; Newton–Schulz error and Table 3 momentum warmup remain outside the guarantee). SignMuon is identified as concurrent (Mishra et al. is a May 2026 preprint developed in parallel); the replies do not mention our own arXiv version or start date, to keep the review double-blind. All of their writing requests receive concrete edits.
- **Tx34:** the reply separates inherited from new results and states the design lesson (compress the gradient; orthogonalize after aggregation). It says exactly what the submitted Theorem 4 covers before adding the 8×8 construction, and Q3 says we will clarify our statements on the empirical ranking and on sufficient versus necessary conditions.
- **AI:** opens with a technical introduction, without a salutation; concessions on separability, exact polar factors and X versus W; credit to Modoranu et al. and Jiang et al.; the method matrix is committed; minor points are answered with submitted A5/A10 facts.

## Revision 14

The replies keep the revision 13 content in the wording of the authors' ICML response (specific thanks, "We agree", "We respectfully disagree", gentle score wording for pqje and Tx34 only). The confidential comment now tells the contribution story within its 1500-character limit. See the Revision 14 section of `02_REBUTTAL_PLAN.md`.

## Revision 13

The revision 11–12 logic is now phrased in the committed (Codex) wording, reusing its sentences wherever the content is unchanged. See the Revision 13 section of `02_REBUTTAL_PLAN.md`.

## Revision 12

A language pass: complete sentences throughout. Minor content was dropped where space ran out (the table SDs and repeated statements of the SignMuon condition). pqje point 4 now cites the heterogeneous federated CNN2 test of distributed compression. Concurrency is stated as independent of this work. See the Revision 12 section of `02_REBUTTAL_PLAN.md`.

## Scientific notes

See the **Revision 11** section of `02_REBUTTAL_PLAN.md`. In particular, the submitted main Table 1 also disagrees with Table A5, and the replies now say both main tables were corrected. "Guaranteed EF21 methods" always means EF21-MuonUSign and EF21-MuonSign. The heterogeneous mechanism is an observation from the code and the table; the `muonserver` control was not run under heterogeneity.

## Defaults applied for author decisions

D4r2 renaming declined; Tx34 W3–W4 agreement reinstated (R5.6); joint-smoothness reduction not used. Edit these in the browser editor if you prefer otherwise.

## Evidence and handoff

All 64 IDs and 79 source mappings remain; 56 concerns are selected, there are no editorial omissions, and the 8 missing-evidence flags are unchanged. No experiments, proofs, addendum or PDF were changed. The pre-edit state is in `response_history/before_revision_11_rewrite_20261030T122944Z/`; per-review history entries were added as the editor does. Build with `python -X utf8 aaai_submission_openreview/primer/workspace.py build`, then run `check`; results are in `evidence/revision_11_validation.json`. Reload the editor after pulling: any browser copy based on an older revision will be reported as a conflict rather than overwriting the files. Saves are local; nothing was posted to OpenReview.
