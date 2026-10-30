# Editing and saving

Double-click `start_workspace.cmd` and leave the Python window open. The browser navigator lists every configured human and AI review plus the optional confidential comment.

Follow a summary concern to the exact highlighted review passage and back, or open its response field. Separate answers, editable “Re …” headings and the full response stay linked. The full rendered preview supports inline and display LaTeX with bundled KaTeX. Counts use the exact source, including headings and line breaks.

Opening/closing options are editable. Click **Use this opening/closing** to apply one; editing an option alone leaves the active draft as it is. Inspect the count after changing wording.

The local server autosaves `response_fragments.json` or `confidential_comment.json`, and writes valid text to `rebuttals/`. It retains prior versions in `response_history/`, detects revision conflicts, and keeps the previous valid Markdown if current edits fail the configured checks. The page identifies omitted and unchecked concerns. An unfinished imported draft is usable in the editor but has no valid response Markdown yet.

Browser copies use localStorage, not cookies. Clearing site data removes that copy. For a directly opened HTML page in desktop Edge/Chrome, **Save to repo** can write to the selected submission folder. Other browsers can use the local server. JSON downloads and restores provide extra backups. Browser copies are separated by paper identity and browser address.

**Repo means local files.** Git commit, push and OpenReview submission are separate author actions.

To rebuild after changes to summaries, mappings or software, run `python primer/workspace.py build`. Run `python primer/workspace.py check` for final coverage/export checks. Restart the local server after updating the primer. Existing drafts and scientific notes stay outside the primer folder.
