# OPT 2026 edition

`make_opt.py` builds the OPT 2026 paper. The **appendix** is derived from the
AAAI sources automatically; the **body** is a fork, `opt_body.tex`, because
twelve pages of main text do not become six by rule. Nothing here writes to
`aaai_article/`, so the AAAI submission and the arXiv build are unaffected.

```sh
cd opt_article
python make_opt.py
```

That writes `build/`: main.tex, the three class files, `references.bib`,
`main.bbl`, and the thirteen images the document uses. Beside it go
`signmuon_opt.pdf`, the compiled preview, and `signmuon_opt.zip`.

## The venue

| | |
| --- | --- |
| Style | `opt2026.cls`, from <https://opt-ml.org/opt2026_style.zip>, unpacked into `style/` |
| Class it derives from | `jmlr.cls`: single column, `article`-based |
| Length | 5 pages at submission (soft), 6 for the camera-ready (hard), references and appendix excluded |
| Review | double-blind |
| Deadline | 4 September 2026 (AoE); notification 29 September; camera-ready 27 November |

**The main text is 6 pages**, down from 12. Every build reports it, read out of
`main.aux` at the label the script emits at the end of the Conclusion:

```
  main text      : 6 pages, limit 5   <-- 1 over
```

The limit the build compares against is 5 for `anon` and `named` and 6 for
`final`, which is why the camera-ready build reports no overage.

## What changes, and what does not

| AAAI | OPT |
| --- | --- |
| two columns | one |
| `aaai2027.sty` | `opt2026.cls` |
| author-year citations, `aaai2027.bst` | numeric, `plainnat`: `[7]`, not `(Author 2026)` |
| main paper + separate supplementary PDF | one document, appendix after the references |
| `xr` cross-document references | ordinary `\ref` |
| Anonymous Submission / Anonymous Institution | the class's own "author names withheld" |
| `figure*`, `table*` | `figure`, `table`; 24 of them |
| `\columnwidth` = 3.35in, one column of two | `\aaaicolumnwidth`, the same 3.35in, not the 6.02in single-column measure |
| acknowledgments, contributions, code availability | none, in any mode |
| 12-page body, 3 figures and 3 tables | `opt_body.tex`, 6 pages, 3 figures |

Theorem numbering is unchanged: Theorems 1–5, Lemmas 1–5, Corollaries 1–2 and
the rest carry the same numbers as in the AAAI and arXiv PDFs, so a reference
to "Theorem 4" means the same thing in all three.

## The files you edit

- **`opt_body.tex`**, the forked body. Its header lists what was cut and where
  each piece went, and which labels must survive because the appendix points at
  them.
- **`opt_appendix_extra.tex`**, holding what the cut displaced and the AAAI
  appendix had no home for: the full related work, the Sign A iteration, the federated
  setting with its table, and the CIFAR-10 and nanoGPT tables. Spliced in
  immediately after `\section{Appendix}`, ahead of the AAAI appendix's own
  subsections.
- **`opt_authors.tex`**, holding `\optauthor`, the running head, the PDF author
  string and the repository URL. `\opttitle` and `\optshorttitle` are read in
  every mode; the author block only in `named` and `final`. `\opttitle` carries
  the arXiv edition's title,
  *Sign compression for Muon: SignMuon, MuonSign, and the Limits of Error
  Feedback*, so the two agree word for word; the AAAI one, locked to its
  OpenReview record, is no longer used here.
- **`style/`**, the OPT style file as shipped. Replace wholesale if OPT
  reissues it.

There is deliberately **no back matter**. Acknowledgments, author contributions
and code availability each name a person or a repository, so none of them
belongs in a double-blind submission; none is built in any mode, which is a
stronger guarantee than switching them off for `anon` alone.

## What the cut moved

Two thirds of it was deletion rather than motion: the material already had a
home in the AAAI appendix, so the body lost a block and gained a pointer.

| From the body | To |
| --- | --- |
| Related Work, 3 paragraphs | `opt_appendix_extra.tex` |
| the Sign A family and its display | `opt_appendix_extra.tex` |
| the federated problem (2) | `opt_appendix_extra.tex` |
| §4.1, the centralized setting, and the unit-gain justification | already in `app:lrscale` |
| §4.5, the federated mechanics | already in `app:alg` |
| the EF21 estimator display, memory cost, the √r penalty | already in `app:hyp` |
| the two-model display (11) | `opt_appendix_extra.tex` |
| §5.2, the federated experiments, with its table | `opt_appendix_extra.tex` |
| the CIFAR-10 and nanoGPT tables | `opt_appendix_extra.tex` |

The two experiment figures came the other way, out of the appendix and into
§4.1 and §4.2: at five pages a curve earns its space and the digits behind it
do not. What remained was then cut by about a third again, sentence by
sentence. The body keeps three figures and no tables: the counterexample trajectories,
the CIFAR-10 curves and the nanoGPT curves. The Theory section now opens at
"Three placements of the sign", the generic Sign A iteration having gone to
`app:opt:signa`.

The AAAI appendix is trimmed too, by `APPENDIX_CUTS`: 458 words of the
reproducibility section, which is written to a conference checklist. It keeps what a
workshop reader needs: infrastructure, seeds, schedules, the tuning protocol,
the conventions with numerical consequences. Out go the case for the
benchmarks, the federated selection margins and the argument for reporting a
threshold column.

## The fork, and keeping it honest

Forking the body means the OPT edition no longer follows an AAAI revision by
itself. Three mechanisms make that visible rather than silent.

**Drift.** `opt_body.origin` records the AAAI body's SHA-256 as it stood when
the fork was last reconciled, and `opt_body.origin.tex` keeps the text that
hash belongs to. Every build compares and, when they differ, sizes the change:

```
FORK DRIFT
  signmuon_body.tex has changed since opt_body.tex was last reconciled with it:
  1 hunk(s), +2 -0 lines.
  Read them with   python make_opt.py --drift
```

`python make_opt.py --drift` prints the diff between the AAAI body as it was at
the fork and as it is now, which is the list of things to consider porting into
`opt_body.tex`. Not all of them belong in a six-page paper; that is the point of
the fork. When you have ported what does, `python make_opt.py --reconciled`
records the new state and the warning stops.

The warning also travels: `make_newinml.py` builds the OPT edition on its way to
the workshop version, and repeats anything the OPT build warned about, so
running only the NewInML build cannot hide it.

**Lost labels.** The appendix refers to nineteen labels the body defines,
`tab:exp_3` alone thirteen times. If the cut drops what defines one, the build
stops before compiling and names it, instead of leaving an undefined reference
twenty minutes of reading later.

**Appendix patches and cuts.** One sentence in the AAAI appendix says "Table 2
**in the main text**", which stopped being true when the table moved.
`APPENDIX_PATCHES` rewrites it; `APPENDIX_CUTS` drops the spans of the
reproducibility section the workshop edition does not need. Every anchor must
match exactly once, and a cut that would delete a `\label` is refused outright,
so a reworded appendix stops the build instead of silently losing a paragraph
or leaving a dangling reference. Each build prints what it cut:

```
  appendix cut   : 164 words  [why these three benchmarks: a justification, not a reproduction detail]
```

## Modes

```sh
python make_opt.py                 # anon  (default) -- what you submit
python make_opt.py --mode named    # author names shown
python make_opt.py --mode final    # camera-ready: [final], editor line cleared
```

`anon` writes `signmuon_opt.pdf`; the other two add their name to it
(`signmuon_opt_final.pdf`), so a camera-ready build cannot quietly overwrite the
anonymous one. In `anon` the class prints "author names withheld", swallows
`\optauthor` and `\acks`, and the script writes no `/Author` into the PDF
metadata. Nothing of `opt_authors.tex` reaches an anonymous build but the two
titles.

## The three collisions between the paper and the class

`jmlr.cls` is a busy class: it loads natbib, graphicx, url, amsmath, amssymb,
xcolor, hyperref, nameref and algorithm2e, and `jmlrutils` on top. Three of
those overlap with what the paper declares, and each would break the document
silently rather than loudly.

**algorithm2e owns the pseudocode vocabulary.** `\For`, `\While`, `\If`,
`\Else`, `\Return` and two dozen more are algorithm2e's, and algorithmicx
declines to redefine a command that already exists, so `\For` would open an
algorithm2e block that `\EndFor` cannot close, and all nine algorithms would
fail with *Missing number, treated as zero*. The class's algorithm2e goes
unused here, so the generated preamble releases those names before algorithmicx
loads. `\listofalgorithms` goes with them: both packages define it, and
`algorithm.sty` stops outright on the clash.

**jmlrutils shares one counter across the theorem environments.** Its
`theorem`, `lemma`, `proposition`, `corollary`, `remark` and `definition` all
count together, so the first Lemma after Theorem 1 would print as "Lemma 2" and
the `\setcounter` calls that keep Theorems 1–5 in one sequence across body and
appendix would land on the wrong numbers. Each name the paper declares with
`\newtheorem` is released first. The list is read out of the AAAI preamble, so
adding an environment there needs no change here.

**jmlrutils owns `\subfigure`.** It declares the command and the counters, and
subcaption, finding the counters taken, then declines to define
`\begin{subfigure}`. `nosubfloats` is passed to jmlrutils before the class
loads, which switches that layer off and lets subcaption own it.

## Checks

The build fails rather than ship a broken document: exactly one
`\bibliography`, no surviving `\input`, `xr` reference, `aaai2027`,
`\onecolumn`, `\affiliations`, `\pdfinfo` or starred float, one
`\documentclass` and one `\maketitle`, no anonymous placeholder, and no
referenced-but-undefined label. After compiling it reports undefined
references, undefined citations, duplicate hyperlink destinations, overfull
boxes and the length of the main text. The first three read 0.

A `\usepackage` the script has no rule for is carried into the OPT preamble
*and* listed at the end of the build, so a package added to the AAAI paper
later is something you are told about rather than something that vanishes.

## Whole-name links

Shared with the arXiv build, along with the flattener and the comment-aware
line scanner: `make_opt.py` imports them from `aaai_article/arxiv/make_arxiv.py`
rather than keeping a second copy, so a fix to one edition is a fix to both.

`Theorem~\ref{th:1}` sets the word as ordinary text and links only the number,
which leaves the reader a two-character target. The converter rewrites it to
`\hyperref[th:1]{Theorem~\ref*{th:1}}`, so the whole "Theorem 1" is one link.
267 of the paper's references are rewritten this way; the rest are the tails of
ranges and lists, which are a bare number and correctly stay one. `\eqref` is
rebuilt around `\ref*` for the same reason.

## The narrower measure

The AAAI appendix was set one column wide at 7in; `jmlr.cls` gives it 6.02in,
14% less. That difference arrived as eighteen overfull boxes: theorem
statements whose inline formulas offer no break point, two wide tables, and one
2.4-inch typewriter string. Four of them protruded by more than 30pt, which is
half an inch of text in the margin.

Four changes, none of which touches a word of the paper, bring it to **zero**:

| | |
| --- | --- |
| `\emergencystretch=3.5em`, `\tolerance=1200` | lets TeX stretch a line rather than overrun the margin, in a final pass used only where the alternative is protrusion |
| `\tabcolsep` 6pt → 4pt | 28pt of padding recovered in a seven-column table, without changing what any table contains |
| the alignment table `\small` → `\footnotesize` | the one table that does not come back inside the measure on padding alone |
| `\allowbreak` in `federated.algorithms.communication_bits` | one break point, at the module boundary a reader would break it at anyway |

The trade is eight loose lines where there were sixteen protruding ones. Loose
lines are ordinary in justified text; text in the margin is not.

## Known cosmetics

`\algnarrow` narrows an algorithm float to 0.72 of the text width, and takes
effect only for the algorithms whose `Input:`/`Output:` lines sit inside
`algorithmic`; the others come out full width. This is inherited: the arXiv PDF
prints them the same way.

The class prints an empty copyright line, "©", at the foot of the first page of
an anonymous build, `\jmlryear` and the author list both being blank. OPT's own
`sample.pdf` does the same.
