# Agent guide: prepare a paper-specific rebuttal workspace

Start with the parent `AGENT_START.md` and `workspace.json`. The initial imported passages are a scaffold. Their titles explicitly say summaries are needed, and all answers remain empty and unverified.

## 1. Establish the source record

Read the downloaded manuscript, technical supplement, reproducibility checklist, code archive, repository code and original review JSON. Make `01_SUBMISSION_MAP.md` identify their correspondence, version differences, source hashes, relevant tables/sections, and which files are authoritative. Do not assume the repository is identical to the submitted archive. Record the actual reviewer count and IDs from configured note IDs, including AI reviews.

Treat source reviews as immutable. `workspace.json` records the original JSON hash; a change fails checks. If reviews are updated, preserve the previous export, reconcile IDs and passages, document the update, then deliberately record the new hash. Do not silently edit a review to make a quote match.

Read the conference rules supplied by the author. Set deadline/timezone, per-response character limit, external-link permission and the confidential-comment limit separately. Unknown limits stay unknown. A different conference is not evidence of the rules for this one.

## 2. Summarize every concern, with exact links

For each `reviews/<stem>.md`, preserve the headings `## Brief summary`, `## Every concern / request`, and `## Response priorities`. The reader parses that structure. A concern line looks like:

```markdown
1. **R1.1 — Reviewer asks how error feedback affects the convergence claim.** They ask (1) which theorem applies and (2) how the experiment tests it. (Feedback 1.)
```

Write concrete summaries of each distinct question or criticism. Include favorable assessments in the overview. Use selectable headings that identify the issue, such as “Reviewer asks …”; distinguish a reviewer's concern from an author decision. Use (1), (2), (3) inside multi-part sentences. Avoid vague jargon (“control”, “precedents”, “protocol”, “calibration”, “mechanism”, “interface”) in your own summaries. Name the task, algorithm, quantity or comparison.

`annotations.json` maps each concern to one or more exact `[field, quote]` entries. Where the same quote repeats, `[field, quote, occurrence]` selects a zero-based occurrence. The quote must fit within one original paragraph. Split cross-paragraph requests into several links. Repeated mentions of one concern should link back to the same concern. The builder rejects missing, ambiguous or misplaced quotes.

Keep one fragment per concern and map each concern once under `reply_topics`. Follow the order of Specific Points of Feedback for Rebuttal, preserving question numbers; group related weaknesses under the matching question. AI reviews without that section need topics derived from their own text. Short `heading` values such as `1. Re the error-feedback assumption:` are editable and count toward the limit. `question`, `source`, optional `occurrence` and concern links retain the original review mapping.

## 3. Establish evidence before drafting

Use `02_REBUTTAL_PLAN.md` as a concern-to-evidence ledger. For each point, record (1) submitted paper/supplement section or code location, (2) the exact finding and relevant values, (3) what it establishes, (4) any missing evidence or proposed experiment, and (5) the intended concise answer.

Before promising to correct a citation, inspect the bibliography in the actual submitted PDF. Check what venue is already printed there; a preprint-like bibliography key or an old arXiv identifier does not show that the submitted citation omits the journal or conference. Keep the submitted citation and any later publication-status check distinct.

Distinguish independent training/evolution runs from repeated evaluation of the same result. Distinguish model or algorithm effects from optimizer/settings changes. State what computing time includes. Credit human edits and distinguish unedited results. Keep numerical observations separate from exact mathematical certificates. Do not invent experiments, proofs, significance, or support from the reviews.

## 4. Draft a response the reviewer can follow

Use a direct answer, then evidence and its relevance. Define unfamiliar notation locally. Lead with completed actions (“We repeated …”, “We verified …”, “Our method achieved …”) and identify the comparison. Explain proposed revisions as future work on the text; do not claim they have already been made.

Check the logic against the exact question. An answer about an additional baseline must first say whether that baseline can be used, what adaptation it requires, and why the existing comparison was chosen. If introducing a different method or result, explain how it answers that baseline request. A favorable result on another comparison is not sufficient by itself. For a straightforward presentation request, state the concrete change we will make; describing the current abstract or defending its contents does not answer a request to shorten it. Remove unrelated facts even when they are favorable.

Be firm about the scientific contribution and cordial toward reviewers. State what the paper establishes, correct mistaken premises courteously, and explain why the evidence supports the contribution. Give warm, review-specific thanks. Invite score reassessment gently and conditionally, for example: “If these answers address your concerns, would you be open to reassessing your score?” Avoid demands for a higher score, reflexive agreement, elaborate praise, and answering every request with “we will”. Make concrete commitments for justified presentation improvements. Mention existing content where it answers the question. Stronger wording must preserve genuine technical gaps and must not invent proofs, experiments or completed revisions.

Make each answer understandable without surrounding context. Name the methods behind phrases such as “alternative designs” and identify whose advantage is being discussed. Define dimensions, point counts and objective values locally. Explain what was optimized and how before giving a numerical result. Use digits for numerical counts, with readable spaces in lists and formulas. Build a logical sequence from contribution to comparison, experiment and conclusion; remove unrelated facts rather than squeezing them into an opaque paragraph.

Defend the paper's supported contribution. Correct mistaken premises courteously and directly. A request for broader comparisons does not erase the results already established. Answer valid technical questions honestly; keep unfinished answers marked `needs_check: true`. Aim close to the allowed length while preserving clarity. Do not fill the space with opaque abbreviations or unsupported concessions.

Keep internal evidence and qualifications in `notes`, excluded from export. Author-facing omissions remain visible through `include: false`. A valid Markdown file can omit concerns; inspect coverage. Do not claim all questions are resolved when some are excluded or unfinished.

Tailor `frame_variants` to each review: cordial specific thanks, concise thanks, a gentle conditional reassessment request, a grateful conditional request to consider a higher score, and an option to omit the closing. Keep the AI review free of score requests. Applying a variant changes the active text; merely editing the option does not. Final assembled source, headings, line breaks and LaTeX all count.

## 5. Decide whether a confidential comment helps

Read `04_CONFIDENTIAL_COMMENT_STRATEGY.md`. A concrete factual correction or a concise synthesis across reviews can help the AC/SPC. Use reviewer IDs and exact question numbers. Avoid score arithmetic or speculation about motives. Do not use confidential text to claim unfinished issues are resolved.

Add local reference metadata to `workspace.json`:

```json
{"stem": "review_R1_example", "point": "R1.1", "label": "Reviewer example · Feedback 1: error-feedback assumption"}
```

Put these records in `confidential.references`. They create browser links without adding URLs to the exported comment. `confidential_comment.json` stores the independent editable text. Check the actual confidential-field limit and intended readers before posting.

## 6. Verify and hand off

Run `python primer/workspace.py build`, inspect the pages, then `python primer/workspace.py check`. Validate source mappings and links, math, word/character counts, omitted concerns and evidence flags. Check saving on a temporary workspace if you changed software. A final pass must read each exported response without relying on internal notes.

Explain how to launch `start_workspace.cmd`, where local saves go, and what remains unfinished. Browser storage uses localStorage, not cookies. Use JSON/file saves and backups for durable copies. No automatic submission, Git commit or push is implied by these actions.
