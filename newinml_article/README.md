# NewInML @ NeurIPS 2026 edition

`make_newinml.py` puts a finished paper on the NeurIPS 2026 workshop template.
It converts two sources today and takes a third as a table entry, not a code
path.

```sh
cd newinml_article
python make_newinml.py                  # the OPT edition of the SignMuon paper
python make_newinml.py --source icomp   # the ICOMP article, from EDMGrokking
```

Each writes `build/<source>/`, a compiled `<source>_newinml.pdf` and a
`<source>_newinml.zip` that compiles on its own in an empty directory.

## The venue

| | |
| --- | --- |
| Call | <https://newinml.github.io/NewInML2026NeurIPS/> |
| Template | NeurIPS 2026 workshop: `\documentclass{article}` + `\usepackage[dblblindworkshop]{neurips_2026}` |
| Length | 2 to 8 pages excluding references |
| Review | double-blind on OpenReview, full anonymization |
| Deadline | 29 August 2026, 11:59pm AoE |
| Archival | non-archival; concurrent submission elsewhere is allowed |

`style/` holds the official `neurips_2026.sty`, taken from
[`Formatting_Instructions_For_NeurIPS_2026.zip`](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
on media.neurips.cc, with the template and checklist it ships beside it. The
whole zip is kept for provenance.

## Why it converts more than one paper

The converter takes **one shape of input: a standalone LaTeX article**, from
`\documentclass` through `\end{document}`. Everything specific to a source
lives in one `SOURCES` entry:

```python
"icomp": {
    "label":    "the ICOMP 2026 article",
    "assemble": None,                       # already standalone
    "main":     ".../icomp_v2/report.tex",
    "assets":   ".../icomp_v2",
    "styles":   ["icomp2026_conference"],   # drop these \usepackage lines
    "provides": [],                         # what that style file loaded for it
    "retire":   [],                         # machinery only its class needed
},
```

The OPT edition is not standalone, so its entry runs
`opt_article/make_opt.py --no-compile` first and converts what that assembles.
That is the whole coupling: **edit the OPT edition, rerun one line here, and
the NewInML version follows.**

To add a third source, add an entry. `styles` is the old venue's style file,
`provides` is what that style file loaded on the document's behalf, and
`retire` names blocks that existed only to survive the old class.

## What the conversion does

| | |
| --- | --- |
| class line | `\documentclass{article}` + `\usepackage[dblblindworkshop]{neurips_2026}` + `\workshoptitle` |
| the old style file | dropped, with `times` and `natbib`, which neurips_2026 loads itself |
| what the old class provided | added: for OPT that is graphicx, amsmath, amssymb, url and hyperref, none of which the document asks for, because jmlr.cls asked for them |
| old-class workarounds | retired: the algorithm2e name release and the jmlrutils theorem-counter release are no-ops once neither package is loaded |
| `\title[short]{full}` | `\title{full}`. article's `\title` has no optional argument and would typeset the short title, leaving the real one loose in the document |
| `\optauthor`, `\Name`, `\Email`, `\addr`, `\acks` | shimmed with `\providecommand` |
| the author block | withheld under `anon`; under `final` taken from `newinml_authors_<source>.tex` |
| `\bibliographystyle` | `plainnat`, the old `.bst` having stayed behind |
| `float`, `subfig` | lifted above `hyperref` (see below) |
| theorem and float destinations | made unique (see below) |

## Two link defects it repairs

Both are invisible in the source venue and appear when the paper moves.

**Duplicate destinations.** hyperref names a destination after the raw counter,
so two objects sharing a counter value share a destination and the second is
dropped: every link to it lands on the first. A theorem restated in an appendix
under the main text's number does it, and so does an appendix that renumbers
floats. The conversion qualifies `\theH<counter>` with the section, which
changes nothing visible.

**`float` after `hyperref`.** Both patch `\@caption`, and in that order every
float's destination is written twice. **The ICOMP article as it stands has 36
of these**, one per table and figure, so 36 of its `\cref`s land on the wrong
float. Loading `float` first fixes all 36. That is worth carrying back into
`report.tex` itself.

## Modes

```sh
python make_newinml.py --source opt                 # anon (default)
python make_newinml.py --source opt --mode final    # camera-ready
```

Under `anon` the style file prints "Anonymous Author(s)", numbers the lines for
reviewers, and hides `\ack`. The footer reads "Submitted to 40th Conference on
Neural Information Processing Systems (NeurIPS 2026). Do not distribute." That
is the style file's own behaviour for a submission: `\workshoptitle` reaches
the footer only under `final`, where it reads "…(NeurIPS 2026). Workshop:
NewInML."

`--workshop "Some Other Workshop"` retargets it, which is the only change any
other double-blind NeurIPS workshop needs.

## The author blocks

`newinml_authors_opt.tex` and `newinml_authors_icomp.tex`, one per source: two
papers do not share an author list. Read only under `--mode final`. NeurIPS
separates authors with `\And` or `\AND` and takes the affiliation as further
lines of the same block, which is not how jmlr or ICOMP spell it, so these are
written here rather than converted. The ICOMP one carries TODOs, and the build
lists them until they are filled in.

## Checks

The build fails rather than ship a broken document: one `\documentclass`, one
balanced `document`, one `\maketitle`, one `\bibliographystyle`, no surviving
`\title[`, `\optauthor`, old style file or `\usepackage{times}`, no `\author`
in an anonymous build, and no label referenced but defined nowhere. After
compiling it reports undefined references, undefined citations, duplicate
destinations, overfull boxes, and the length of the main text against the 2 to
8 pages the call allows.

Where both sources stand today:

```
opt    48 pages,  main text 6,  0 undefined, 0 duplicate, 0 overfull
icomp  35 pages,  main text 9,  0 undefined, 0 duplicate, 0 overfull   <-- 1 over
```

The ICOMP article is a nine-page conference paper, so it needs a page cut
before it is a NewInML submission. The converter reports the number and does
not attempt the cut.
