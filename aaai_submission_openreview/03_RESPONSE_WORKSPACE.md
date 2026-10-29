# Prepared rebuttal workspace — author handoff

Prepared 29 October 2026. Five human replies, one AI reply, and an optional AC/SPC draft are prepared for AAAI-27 submission 48060. Nothing was submitted, committed or pushed. Existing article/addendum files and live code work were left intact. The replies state their answers directly without relying on reviewers accessing the updated article.

## Prepared drafts

Open `start_workspace.cmd`. Ready-to-copy text is in `rebuttals/rebuttal_*.md`; complete evidence and proposed article wording/tables are in `02_REBUTTAL_PLAN.md`. Version correspondence is in `01_SUBMISSION_MAP.md`. The new coupled-layer EF21 proof is `evidence/EF21_JOINT_SMOOTHNESS.md`.

| Response | UTF-16 characters /1500 | Checked concerns selected | Unfinished requests excluded |
| --- | ---: | ---: | --- |
| R1 ·7S1b |1483 |7/10 |R1.6,R1.7,R1.8 |
| R2 ·D4r2 |1425 |6/8 |R2.2,R2.8 |
| R3 ·Ueq5 |1483 |5/5 |None |
| R4 ·pqje |1490 |14/15 |R4.15 |
| R5 ·Tx34 |1409 |6/6 |None |
| AI reviewer |1499 |18/20 |AI.6,AI.20 |

There are 64 concrete concerns,79 exact passage mappings, 56 selected checked concerns and 8 omitted concerns visibly marked **needs evidence**. Counts include headings, spaces, final newlines and LaTeX source. Shared answers cover related concerns without duplicating text. A successful check/export does not mean all requested experiments are complete.

The optional confidential draft is 1301 characters including its final newline. Its factual content is checked, but the actual field limit/readers remain unknown (`confidential_character_limit: null`). Check the form before using it; the local “ready” message does not certify unknown conference rules. See `04_CONFIDENTIAL_COMMENT_STRATEGY.md`.

## Most useful supported answers

- **The table error is settled by submitted evidence.** Recomputed five-seed logs give MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%; the stale main-table ranking must reverse. SignMuon 85.72±0.24% versus Muon 85.98±0.26% preserves the main practical finding. All175 original/current federated training histories match. No new training was needed. The central main table also needs synchronization with submitted A5/export values.
- **A positive SignMuon condition is available.** A lower bound on polar singular-vector diagonal correlations implies a descent margin. The existing author addendum supplies conditional stochastic first-moment convergence. The replies give a sufficient condition directly; its margin has not been measured in our networks.
- **Ideal EF21 can handle coupled layers.** The new note groups the full tuple in one norm and proves an explicit jointly smooth deterministic-schedule bound for fixed EMA/Nesterov and either downlink. The coupled squared-rate formulation uses a common radius scaled by the whole dual norm, changing the original per-layer sharp-step rule. Fixed peak LR/momentum leave a residual; vanishing first-moment rates require horizon tuning. Finite polar solvers, changing nanoGPT momentum and auxiliary Adam remain outside the full-run guarantee.
- **Theorem 4 is defended with exact quantifiers.** For each fixed learning rate/momentum, a parameter-dependent lower-bounded smooth objective defeats the specified output-EF recursion. Decreasing schedules and horizon-tuned rates are not excluded. New arbitrary-schedule softplus witnesses apply only to the three plain placements.
- **Prior work and empirical scope are stated accurately.** EF21's general framework is credited. Mishra v2 now has conditional actual-vote theory, requiring replacement of the current article's v1 comparison. nanoGPT tests centralized updates after full-precision averaging; single-run rankings remain preliminary and the X/W gap is acknowledged.

## Evidence still missing

| Request | Visible flags | What would complete it |
| --- | --- | --- |
| Neural-network geometry/alignment |R1.6,R2.8,R4.15,AI.20 |Measure the full margin, spectra and actual approximate-oracle/vote alignment along training. |
| Broader federation and models |R1.7,R2.2 |Completed non-IID/local-update/partial-participation/larger-client or distributed-transformer results with independent seeds. A runner is not a result. |
| More compressed nanoGPT seeds |R1.8 |Independent training runs; repeated validation and upstream Muon variance do not substitute. |
| Modern variance-reduced sign baseline |AI.6 |A fair SSVR comparison. Existing SignSGD has momentum but is not SSVR. |

Also unresolved: a trajectory-wide finite-polar error certificate, changing-momentum/auxiliary-optimizer full-run analysis, and a causal explanation of the X/W gap. Selected replies state these limitations. They do not promise unperformed experiments or claim that changing compressors is the only remedy.

## Checks performed

From the repository root:

```powershell
python -X utf8 aaai_submission_openreview/evidence/verify_rebuttal_evidence.py
python -X utf8 aaai_submission_openreview/primer/workspace.py build
python -X utf8 aaai_submission_openreview/evidence/inspect_browser.py
python -X utf8 aaai_submission_openreview/primer/workspace.py check
```

Passed: original hashes, reconstructed five-seed means/sample SDs/thresholds,175 matching histories, rational ascent and output-EF cycle certificates. The analytic proofs establish the theorems; numerical norm samples are only sanity checks.

Edge inspection of all 7 actual editor pages passed: exact browser/export equality and counts, all draft/source math rendered without errors, local anchors resolved, desktop/mobile layout without horizontal overflow, and no JavaScript exceptions. The fresh-profile inspection made no edits/saves and left author JSON hashes unchanged. Details: `evidence/browser_inspection.json`; screenshot: `evidence/browser_R3.png`. No primer software was changed, so save-behavior software tests were unnecessary.

The baseline JSON, summaries and strategy files are preserved in `response_history/preparation_baseline_20261029T134259Z/`. All original IDs and original review blocks are retained; empty imported scaffold headings/praise/reference entries were reassigned to genuine concerns. `evidence/concern_evidence.json` records the migration's concrete result. Preparation scripts guard against replacing later author edits; continue editing through the browser/JSON rather than resetting the preparation.

The configured author-supplied deadline remains 31 October 2026 AoE. Before posting, read each exported response alone and check the actual destination field's count. No local action sends these drafts to OpenReview.

# Editing and saving

Double-click `start_workspace.cmd` and leave the Python window open. The browser navigator lists every configured human and AI review plus the optional confidential comment.

Follow a summary concern to the exact highlighted review passage and back, or open its response field. Separate answers, editable “Re …” headings and the full response stay linked. The full rendered preview supports inline and display LaTeX with bundled KaTeX. Counts use the exact source, including headings and line breaks.

Opening/closing options are editable. Click **Use this opening/closing** to apply one; editing an option alone leaves the active draft as it is. Inspect the count after changing wording.

The local server autosaves `response_fragments.json` or `confidential_comment.json`, and writes valid text to `rebuttals/`. It retains prior versions in `response_history/`, detects revision conflicts, and keeps the previous valid Markdown if current edits fail the configured checks. The page identifies omitted and unchecked concerns. An unfinished imported draft is usable in the editor but has no valid response Markdown yet.

Browser copies use localStorage, not cookies. Clearing site data removes that copy. For a directly opened HTML page in desktop Edge/Chrome, **Save to repo** can write to the selected submission folder. Other browsers can use the local server. JSON downloads and restores provide extra backups. Browser copies are separated by paper identity and browser address.

**Repo means local files.** Git commit, push and OpenReview submission are separate author actions.

To rebuild after changes to summaries, mappings or software, run `python primer/workspace.py build`. Run `python primer/workspace.py check` for final coverage/export checks. Restart the local server after updating the primer. Existing drafts and scientific notes stay outside the primer folder.
