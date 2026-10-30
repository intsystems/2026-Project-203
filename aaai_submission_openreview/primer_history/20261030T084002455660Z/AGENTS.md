# Instructions for agents changing the primer

This folder contains reusable software. Paper-specific claims, sources, review text and authored drafts belong in the parent workspace, never in this distribution.

Read `README.md` and `AGENT_GUIDE.md` before changes. Preserve the division between primer code and paper data. Keep paper identity, reviewer lists, filenames, limits, deadline and rule choices in `workspace.json`; do not hard-code them in runtime modules or browser scripts.

Preserve exact source review text, concern IDs, editable heading links, browser drafts, revision checks, backups and last-valid exports. Never perform Git commits, pushes or OpenReview submissions merely because the editor has a save action.

Use standard Python and the bundled browser assets. A schema change requires a documented migration with a dry-run path and backups. Test against multiple paper names, different numbers of reviews, different character limits, duplicate quotes and separate workspace identities. Refresh `primer_manifest.json` only after inspecting the changes and passing the checks.

`install.py --update` changes only the destination primer and first archives its previous software. It must not replace files in the parent paper workspace. Do not delete old author data or imported source material during an update.
