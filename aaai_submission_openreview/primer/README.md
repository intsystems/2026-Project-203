# Portable rebuttal primer

Copy this **primer folder** beside a paper's downloaded submission and review JSON. The software works offline and needs Python 3.9 or newer; the browser loads bundled KaTeX locally. No API credentials, npm install, or OpenReview login are needed.

The primer contains reusable software, templates and instructions. All paper-specific work lives **outside** it. Updating the primer therefore preserves the paper's summaries, evidence, edits and history.

## Start on another paper

1. Copy `primer/` into that paper's submission folder. Preserve the `primer` folder name.
2. Run this from the submission folder, replacing the example metadata and source filename:

   ```powershell
   python primer/workspace.py init --source reviews.json --title "Full paper title" --short-title "Short name" --venue "AAAI-27" --submission "12345" --limit 1500 --deadline "2026-10-31 AoE" --port 8766
   ```

3. Read the generated `AGENT_START.md`, or ask an agent to follow it. Initialization imports source passages and creates explicitly unfinished editors. It does not invent scientific summaries or responses.
4. Double-click the generated **start_workspace.cmd**. Leave its Python window open. The browser opens the navigator; edits save to local JSON and valid Markdown. Git commit and push are separate actions.

For multiple papers, use distinct `--port` values. Port `0` selects a free port; the console prints its address. Browser storage is tied to the browser's address, so saving to files is the durable copy when the port changes.

The importer accepts a downloaded OpenReview `notes` list (wrapped in an object or directly) and selects `Official_Review` / `AI_Review` invitations. It supports any number of human and AI reviews and both wrapped and plain content values. Other export formats need an explicit importer; they are not silently guessed.

## What is reusable

- Selectable summaries, exact highlighted review passages and return links.
- Linked concern editors and numbered, editable “Re …” headings.
- Editable full response synchronized with separate answers, plus rendered LaTeX.
- Editable opening/closing alternatives and review-specific score requests.
- Word/character counts; configurable character limits and external-link rules.
- Local-server autosave, browser localStorage, Edge/Chrome folder saving, JSON backups, revision conflicts and last-valid Markdown preservation.
- Separate confidential AC/SPC editor, configurable limit and links to review passages.
- One identity per paper for browser copies; no embedded paper names or reviewer lists.

The same editing safeguards apply on another paper. Scientific evidence, interpretations, figures, prose and score arguments must be prepared for that paper.

## Files and update boundary

```text
submission_folder/
  paper.pdf, supplement.pdf, code.zip, reviews.json   existing inputs
  primer/                                           reusable software
    AGENT_GUIDE.md, templates/, runtime/, tests/
    workspace.py, install.py, primer_manifest.json
  workspace.json                                    paper identity, rules and review list
  inputs_manifest.json                              source inventory and hashes
  annotations.json                                  exact review passage mappings
  reviews/*.md, reviews/*.html                       summaries and browser pages
  response_fragments.json, confidential_comment.json editable author data
  rebuttals/*.md                                    valid text for pasting
  response_history/                                saved prior author drafts
  AGENT_START.md, 01_*.md, 02_*.md, 03_*.md, 04_*.md evidence and instructions
```

`init` refuses to overwrite existing workspace files. `build` regenerates HTML from current source data; normal browser saves need no rebuild. `check` verifies exact mappings, generated pages, draft coverage and exports. Unfinished responses fail final checks while remaining editable.

```powershell
python primer/workspace.py build
python primer/workspace.py check
```

Character limits count UTF-16 source characters, including LaTeX and line breaks, matching the browser counter. Word counts are informational. Venues with word-based limits require a deliberate extension; the current version will not treat a word limit as a character limit.

## Update the primer later

An agent should read [AGENT_GUIDE.md](AGENT_GUIDE.md) and [AGENTS.md](AGENTS.md). Update runtime or templates, run the portable checks, then run `python primer/refresh_manifest.py`.

```powershell
python primer/tests/test_portable.py
python primer/refresh_manifest.py
```

An optional browser check exercises a relocated workspace in Edge, including native folder saving and a non-default response limit:

```powershell
python primer/tests/check_browser.py
```

That development check needs the `websocket-client` Python package and installed Edge; normal workspace use needs only Python's standard library.

To transfer the updated software from its source copy:

```powershell
python primer/install.py --destination "C:/path/to/other/paper/submission_folder/primer" --update
```

The installer verifies the source manifest and archives the existing destination primer under `primer_history/` before replacing its software files. Paper data outside `primer/` is preserved. Template updates do not replace already-created paper notes. If the data format changes, write a separate, explicit migration with backups; do not rerun initialization over an existing paper.

Restart running servers and rebuild each paper after a runtime update. Keep its `workspace_id`, review IDs and concern IDs stable to retain browser and draft mappings.

## Privacy and source material

The primer distribution contains no submission PDFs, code archives, reviews, evidence, response text, personal paths or browser history. It retains the bundled KaTeX license. Saving never posts to OpenReview. Final copy-and-paste remains an author action.
