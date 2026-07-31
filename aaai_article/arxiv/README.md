# arXiv edition

`make_arxiv.py` turns the AAAI submission into a single arXiv-ready document.
It reads the AAAI sources and never writes to them, so the submission build in
`aaai_article/` is unaffected by anything here.

```sh
cd aaai_article
python arxiv/make_arxiv.py
```

That writes `arxiv/build/` — the directory you upload — plus
`arxiv/signmuon_arxiv.pdf` (preview, deliberately kept *outside* the build so
arXiv does not read the submission as PDF-only) and
`arxiv/signmuon_arxiv.tar.gz` (the same directory, packed).

## What it changes

| AAAI | arXiv |
| --- | --- |
| main paper + separate supplementary PDF | one document, appendix continuing after the references |
| two reference lists, one per document | one, after the Conclusion, serving body and appendix |
| `xr` cross-document references | ordinary `\ref`, both halves being in one file |
| Anonymous Submission / Anonymous Institution | `arxiv_authors.tex` |
| `[submission]` copyright slug | `[preprint]`: names shown, no slug |
| — | acknowledgments, author contributions, code availability |
| plain text references | clickable refs, citations, URLs, PDF bookmarks |
| `Theorem 1` — only the `1` is a link | the whole `Theorem 1` is the link |

Type, spacing and layout are the AAAI style's, untouched. The generated
`main.tex` runs 41 pages, the same as `v2_SignMuon_AAAI_full.pdf`.

## The two files you edit

Everything else is derived from the AAAI sources automatically — including the
title, so it is never stated twice.

- **`arxiv_authors.tex`** — `\author` and `\affiliations`, the plain-text
  author list for the PDF metadata, and the repository URL. `\equalcontrib`
  (from `aaai2027.sty`) sets the shared "contributed equally" footnote.
- **`arxiv_backmatter.tex`** — acknowledgments, author contributions, code
  availability. Unnumbered sections, spliced in between the Conclusion and the
  references.

Both currently carry drafts. Any `TODO` left in them is listed at the end of
every build.

## Checks

The build fails rather than ship a broken document: it verifies that exactly
one `\bibliography` survives, that no `\input` or `xr` reference is left, and
that no anonymous placeholder remains. After compiling it reports undefined
references, undefined citations, duplicate hyperlink destinations and overfull
boxes; all four should read 0.

## Options

```
--no-compile   assemble the sources without running latexmk
--no-bib       drop references.bib; arXiv uses the shipped main.bbl anyway
--no-tar       skip the tarball
--keep-work    keep .build-work/ (the scratch compile) for debugging
-o DIR         build somewhere other than arxiv/build
```

## Whole-name links

`Theorem~\ref{th:1}` sets the word as ordinary text and links only the number,
which leaves the reader a two-character target. The converter rewrites it to
`\hyperref[th:1]{Theorem~\ref*{th:1}}`, so the whole "Theorem 1" is one link.
243 of the paper's references are rewritten this way; the 28 that are not are
the tails of ranges and lists — the second `\ref` of `Theorems~\ref{a}--\ref{b}`
— which are a bare number and correctly stay one.

Not done with `\cref`, though cleveref is loaded and would also link the whole
phrase: cleveref resolves the name itself, and the paper's 46
`Appendix~\ref{...}` all point at subsections, which cleveref would rename to
"Subsection". The word the author wrote is the word that stays, and the printed
output is character-for-character what the AAAI PDF prints.

`\eqref` is rebuilt around `\ref*` for the same reason — stock `\eqref` prints
"(13)" but links only the 13, leaving the parentheses cold.

The rewrite touches neither comments nor line count, and refuses to run if it
does not conserve the reference count exactly.

## Notes on two things that look odd

`aaai2027.sty` raises `\PackageError` if hyperref is loaded, from a hook that
fires after any `\AtBeginDocument` we could register — so the check cannot be
undone once it runs. The generated preamble instead loads hyperref, loads
cleveref after it (that order is what makes `\cref` link), and then lets
`\ver@hyperref.sty` — the flag the check reads — to `\relax`. Both packages
stay fully functional and the style file is shipped byte-identical to AAAI's.

`algorithmicx` restarts its line counter in every algorithm, so hyperref would
mint one destination name per algorithm *per line number* and keep only the
first. The preamble qualifies `\theHALG@line` with the algorithm number.
Without it the build reports 96 duplicate destinations.
