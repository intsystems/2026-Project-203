# SignMuon rebuttal collaboration

This folder contains the prepared AAAI-27 rebuttal workspace: five human replies,
one AI reply, an optional AC/SPC comment, the original submission/reviews, and the
supporting evidence. The editor works offline with Python 3.9 or newer and a browser.
It needs no API credentials, OpenReview login, or additional Python packages.

## Open and edit

On Windows, double-click **`start_workspace.cmd`** and leave its Python window open.

On macOS/Linux, run this from the repository root:

```sh
python3 -X utf8 aaai_submission_openreview/primer/workspace.py serve --open
```

The server prints the local address. The configured port is 8766; use `--port 0`
to choose another available port. Browser storage is tied to that address, so save
to files before changing ports or clearing browser data.

Read [the author handoff](03_RESPONSE_WORKSPACE.md) for the current draft counts,
remaining evidence requests, and verification results. Read
[the evidence ledger](02_REBUTTAL_PLAN.md) before changing scientific claims.

| File or folder | Purpose |
| --- | --- |
| `response_fragments.json` | Human and AI responses, internal evidence notes, selections and review-specific wording options |
| `confidential_comment.json` | Editable AC/SPC body, limited to 1,500 characters; the separate title and its 250-character limit are configured in `workspace.json` |
| `reviews/*.md`, `annotations.json` | Concern summaries and exact links to the immutable review export |
| `rebuttals/*.md` | Last valid exported text for each response |
| `reviews/*.html` | Generated browser pages; rebuild after pulling changed source files |
| `evidence/` | Proofs, audit scripts, extracted tables and recorded checks |
| `response_history/`, `primer_history/` | Preserved author baseline and earlier editor versions |
| `reference_materials/` | Writing examples from other papers, kept apart from SignMuon evidence |

The local server saves author JSON and valid Markdown automatically. **Save to
repo means local files**; it never submits to OpenReview, commits, or pushes Git.
Unchecked or overlength edits may leave the previous valid Markdown on disk, so
check the editor's readiness status before copying an export.

The confidential Markdown contains an H1 title followed by the body. Copy these
into the separate title and comment fields: the 1,500-character limit applies to
the body, including blank lines, not to the heading. The current approved title
is 63 characters and the body is 1,492 including its final newline. Both editor
save modes and rebuilds preserve that Markdown layout.

To reconcile future direct Markdown edits, first run
`python -X utf8 aaai_submission_openreview/evidence/sync_confidential_draft.py --dry-run`
from the repository root. Review the counts, then use `--apply`; it backs up the
previous JSON/configuration and Markdown and increases the editor revision.

## Share changes

1. Save browser edits before switching branches, pulling, or merging. Close the
   editor/server while Git changes the workspace files.
2. Coordinate who edits each review. All human/AI reviews share
   `response_fragments.json`; inspect overlapping changes instead of replacing the
   entire file with one person's version.
3. After pulling or merging, rebuild and check from the repository root:

   ```sh
   python -X utf8 aaai_submission_openreview/primer/workspace.py build
   python -X utf8 aaai_submission_openreview/primer/workspace.py check
   ```

   Use `python3` where that is your Python command. Commit the edited source JSON,
   any updated notes/summaries, rebuilt pages, and exported Markdown together.
4. Restart the server and reopen the editor after pulling changes. If a saved
   browser copy conflicts with repository data, inspect or back it up before
   choosing which version to restore. Keep review IDs and `workspace_id` stable.

Original PDF/ZIP/JSON inputs and their hashes are immutable. The repository's
`.gitattributes` preserves this folder's exact bytes, including line endings, so
source and editor-manifest verification works across operating systems. Python
caches and temporary logs remain ignored by the repository's existing rules.

The evidence audit scripts are optional development tools. Unlike the editor,
some need NumPy, SymPy, PyMuPDF, Edge, `websocket-client`, or locally restored result
bundles. Their recorded results are included; collaborators can edit and validate
the rebuttals without rerunning those audits. Do not rerun preparation scripts to
reset later author edits.

All six active replies fit the configured 1,500-character limit, with unsupported
experimental requests visibly flagged. No material in this folder has been
submitted to OpenReview by the workspace.
