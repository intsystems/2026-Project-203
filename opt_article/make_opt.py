#!/usr/bin/env python3
r"""Build the OPT 2026 edition of the AAAI/arXiv paper.

OPT 2026 ships a JMLR-derived class, ``opt2026.cls``: single column, numbered
citations, and its own hyperref, natbib, algorithm2e and sub-float machinery.
The AAAI submission is two-column, author-year, and built on ``aaai2027.sty``.
None of the paper's text has to change for the move.  The preamble does, and
this script performs that translation from the AAAI sources without writing to
them, so the same command can be run again after the AAAI paper is revised.

What it does, in order:

  1.  Flattens signmuon_preamble / signmuon_body / signmuon_appendix and every
      file they \input into one main.tex -- the same flattening the arXiv build
      does, and for the same reason: one document, one reference list, no xr.
  2.  Rewrites the preamble.  The AAAI class line becomes
      \documentclass[anon]{opt2026}; the packages opt2026.cls already provides
      (natbib, graphicx, url, amsmath, amssymb, caption) are dropped, as are
      the unused listings block and AAAI's \pdfinfo; everything the *paper*
      declares -- its theorem environments, \crefname list, summarythm,
      \algnarrow, the float fractions -- is carried over verbatim.
  3.  Repairs the two collisions between the paper and the class:
        * jmlr.cls loads algorithm2e, which owns \For, \While, \If, \Else and
          the rest of the vocabulary algorithmicx wants.  algorithmicx will not
          redefine a command that already exists, so the paper's twelve
          algorithms break.  Those names are released before algorithmicx
          loads.  \listofalgorithms goes too: both packages define it, and
          algorithm.sty aborts on the clash.
        * jmlrutils predeclares theorem, lemma, proposition, corollary and
          remark on one shared counter.  The paper numbers each kind
          separately, and the \setcounter calls in the appendix depend on it,
          so the predeclarations are cleared and the paper's own \newtheorem
          lines take effect.
      A third, sub-floats, is settled by passing `nosubfloats` to jmlrutils
      before the class loads, which lets subcaption own \begin{subfigure}.
  4.  Drops the star from figure*/table*/algorithm*.  Those are the two-column
      floats; in a single-column document a starred float may only reach the
      top of a page or a float page, which strands them at the end.
  5.  Substitutes the title -- read from the AAAI preamble unless
      opt_authors.tex overrides it -- and, outside anonymous mode, the author
      block.  There is no back matter in any mode: acknowledgments, author
      contributions and code availability each name a person or a repository.
  6.  Adds what jmlr.cls's own link layer does not: PDF bookmarks, the \eqref
      rebuild, the algorithmicx destination fix, and whole-name reference links
      ("Theorem 1" is the link, not just the "1").
  7.  Copies opt2026.cls, jmlr.cls, jmlrutils.sty, references.bib and exactly
      those images the document includes into the build directory, compiles a
      scratch copy so the build directory stays clean, and reports what the log
      complained about -- including where the main text ends, which is the
      number OPT's page limit is about.

    python make_opt.py                  # anonymous submission, compiled
    python make_opt.py --mode final     # camera-ready: author names shown
    python make_opt.py --no-compile     # assemble the sources only

Nothing here writes to the AAAI sources.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # opt_article
REPO = HERE.parent                              # repository root
SRC = REPO / "aaai_article"                     # read, never written
STYLE = HERE / "style"                          # opt2026.cls as OPT ships it

# Sources of the AAAI build.
PREAMBLE = "signmuon_preamble.tex"
BODY = "signmuon_body.tex"          # forked, see OPT_BODY: read only to detect drift
APPENDIX = "signmuon_appendix.tex"
BIB = "references.bib"

# The body is a fork.  Twelve pages of main text do not become five by rule,
# so opt_body.tex is written and maintained by hand; the appendix is still
# derived from the AAAI source, and opt_appendix_extra.tex holds what the cut
# displaced but the appendix still refers to.  opt_body.origin records the
# AAAI body as it stood when the fork was last reconciled, so that a later
# revision of the AAAI paper is reported rather than silently ignored.
OPT_BODY = HERE / "opt_body.tex"
OPT_APPENDIX_EXTRA = HERE / "opt_appendix_extra.tex"
ORIGIN = HERE / "opt_body.origin"           # the hash
ORIGIN_TEXT = HERE / "opt_body.origin.tex"  # and the text it hashes, to diff

# The class files OPT ships, copied into the build so that it compiles
# anywhere, including on OpenReview's side and on a machine without them.
SUPPORT = ["opt2026.cls", "jmlr.cls", "jmlrutils.sty"]

# The one file holding what the anonymous AAAI sources cannot.  There is no
# back matter: acknowledgments, author contributions and code availability each
# name a person or a repository, so none of them belongs in a double-blind
# submission, and none is built in any mode.
AUTHORS = HERE / "opt_authors.tex"

JOBNAME = "main"
DEFAULT_BUILD = HERE / "build"
WORKDIR = HERE / ".build-work"

# The flattener, the comment-aware line scanner and the whole-name reference
# rewrite are the arXiv build's.  They are shared rather than copied, so that a
# fix to one edition is a fix to both.
ARXIV_TOOLS = SRC / "arxiv" / "make_arxiv.py"

# Where the main text has to stop.  OPT 2026 asks for 5 pages at submission and
# allows 6 for the camera-ready, references and appendix excluded.
PAGE_LIMIT = {"anon": 5, "named": 5, "final": 6}


def _load_arxiv_tools():
    if not ARXIV_TOOLS.exists():
        raise SystemExit(
            f"cannot find {ARXIV_TOOLS}.\n"
            "make_opt.py shares the flattener and the reference rewriter with "
            "the arXiv build; if that script moved, point ARXIV_TOOLS at it."
        )
    spec = importlib.util.spec_from_file_location("make_arxiv", ARXIV_TOOLS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ax = _load_arxiv_tools()
code_of = _ax.code_of                     # the part of a line TeX acts on
flatten = _ax.flatten                     # \input expansion
linkify_named_refs = _ax.linkify_named_refs
match_brace = _ax.match_brace
split_preamble = _ax.split_preamble
title_of = _ax.title_of
report_log = _ax.report_log
_INPUT = _ax._INPUT
_GRAPHICS = _ax._GRAPHICS

_COMMENT = re.compile(r"(?<!\\)%")

# Where the main text ends.  Emitted just before the bibliography; the .aux
# then records the page, which is the one OPT's limit is counted against.
BODY_END_LABEL = "opt:endofmaintext"

# Sentences in the AAAI appendix that stop being true once the cut moves what
# they point at.  Each must match exactly once; the build stops if one does
# not, rather than shipping a document that says the opposite of where a table
# is.  Applied before the reference links are rewritten, so the patterns are
# the source's own spelling.
APPENDIX_PATCHES = [
    (r"Table~\ref{tab:exp_3} in the main text",
     r"Table~\ref{tab:exp_3}",
     "the federated table sits in the appendix now, not the main text"),
    # The alignment table is the one that does not come back inside the measure
    # on \tabcolsep alone: seven columns, one of them headed "% of steps with
    # rho_t < 0".  A size down is the whole fix.
    ("\\small\n\\begin{tabular}{@{}lrrrrrl@{}}",
     "\\footnotesize\n\\begin{tabular}{@{}lrrrrrl@{}}",
     "the seven-column alignment table is 9pt wider than jmlr's measure at "
     "\\small"),
    # A 2.4-inch unbreakable word in a 6-inch measure: the line before it has
    # to stretch to the margin whatever else is on it.  One break point, at the
    # module boundary the reader would break it at anyway, settles the
    # paragraph.
    (r"\texttt{federated.algorithms.communication\_bits}",
     r"\texttt{federated.algorithms.\allowbreak communication\_bits}",
     "the longest typewriter string in the appendix has no break point"),
]

# Spans of the AAAI appendix that the OPT edition drops: from the start of the
# first anchor to the start of the second, which is kept.  Each anchor must
# occur exactly once, so a reworded appendix stops the build instead of losing
# a paragraph nobody meant to lose.  The reproducibility section is written for
# a conference submission's checklist; what a workshop reader needs from it is
# the protocol, not the case for the protocol.
APPENDIX_CUTS = [
    (r"\paragraph{Choice of benchmarks.}",
     r"\paragraph{Computing infrastructure.}",
     "why these three benchmarks: a justification, not a reproduction detail"),
    ("One selection margin requires comment.",
     r"\paragraph{Conventions with numerical consequences.}",
     "federated selection margins, the spread of selected rates, and the case "
     "for reporting a threshold column"),
]


BANNER = r"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GENERATED FILE -- DO NOT EDIT.
%
%  OPT 2026 edition of "{title}" (mode: {mode}).
%  Assembled by opt_article/make_opt.py from the AAAI sources:
%      {sources}
%  plus opt_article/opt_authors.tex.
%
%  Edit those, not this file, and rerun:  python opt_article/make_opt.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

CLASS_HEADER = r"""% jmlrutils declares its own \subfigure and \subtable and the counters that go
% with them, which makes subcaption skip \begin{{subfigure}} altogether -- the
% paper's two sub-figures would come out as an undefined environment.  The
% class's sub-float layer is switched off here so that subcaption can own it.
% Both options have to be set before the class loads.
\PassOptionsToPackage{{nosubfloats}}{{jmlrutils}}
\PassOptionsToPackage{{hyphens}}{{url}}   % jmlr.cls loads url without options
\documentclass[{options}]{{opt2026}}
"""

# --------------------------------------------------------------------------
# Preamble translation.
#
# Every line of signmuon_preamble.tex is either dropped -- the AAAI class line,
# the packages opt2026.cls already loads, template boilerplate -- or carried
# over unchanged.  A \usepackage this script has no rule for is carried over
# *and reported*, so a package added to the AAAI paper later shows up in the
# build log instead of silently vanishing.
# --------------------------------------------------------------------------

# Packages opt2026.cls -> jmlr.cls loads for itself.  Loading them again is at
# best a no-op and at worst an option clash.
CLASS_PROVIDES = ["url", "graphicx", "natbib", "amsmath", "amssymb", "caption"]

# Packages this script knows the paper needs and has checked under opt2026.
KNOWN_KEEP = ["algorithm", "algorithmicx", "algpseudocode", "booktabs",
              "multirow", "subcaption", "cleveref"]

DROP_LINE = [
    (r"^\s*\\documentclass", "the OPT class line replaces it"),
    (r"\\usepackage\s*(\[[^\]]*\])?\s*\{aaai2027\}", "AAAI style file"),
    (r"\\usepackage\s*(\[[^\]]*\])?\s*\{(" + "|".join(CLASS_PROVIDES) + r")\}",
     "loaded by opt2026.cls"),
    (r"\\urlstyle\s*\{", "AAAI's url styling; jmlr.cls sets its own"),
    (r"\\def\s*\\UrlFont", r"sets \rm, which jmlr.cls flags as obsolete"),
    (r"\\usepackage\s*\{(newfloat|listings)\}", "the listing float is unused"),
    (r"\\DeclareCaptionStyle", "belongs to the unused listing float"),
    (r"\\floatstyle\s*\{", "belongs to the unused listing float"),
    (r"\\newfloat\s*\{listing\}", "the listing float is unused"),
    (r"\\floatname\s*\{listing\}", "the listing float is unused"),
]

# Constructs removed whole, brace-matched from the opening pattern.
DROP_GROUP = [
    (re.compile(r"\\pdfinfo\s*\{"), "AAAI template version stamp"),
    (re.compile(r"\\lstset\s*\{"), "belongs to the unused listing float"),
]

# What a removed group leaves behind: a line that reads as code, so the block
# it stood in is still recognized as one that carried a declaration, and is
# then dropped like any other, taking its explanatory comment with it.
GROUP_SENTINEL = "\\@optdroppedgroup"

# Comment lines that are AAAI template furniture and say nothing about this
# paper.  One match condemns the whole run of comment lines it sits in, since
# the template's instructions run over two and three lines and half of one is
# worse than none; a comment block with no such line in it is the authors' own
# and is kept.
DROP_COMMENT = re.compile(
    r"^\s*%.*(DO NOT CHANGE THIS|DISALLOWED|aaai2027|AAAI Press|AAAI Style|"
    r"Written by AAAI|TemplateVersion|acceptable font size|may not be used|"
    r"will not be published|specifically forbidden|No page breaks|"
    r"No negative value|embeds links such as|fonts are loaded automatically)"
)

_IS_COMMENT = re.compile(r"^\s*%")

ALGO_FIX = r"""%%% ---- algorithmicx under jmlr (inserted by make_opt.py) ------------------
% jmlr.cls loads algorithm2e, which owns \For, \While, \If, \Else, \Return and
% the rest of the pseudocode vocabulary.  algorithmicx declines to redefine a
% command that already exists, so \For would open an algorithm2e block that
% algpseudocode's \EndFor cannot close and every algorithm in the paper would
% fail.  The class's algorithm2e goes unused here, so its names are released.
% \listofalgorithms goes with them: both packages define it, and algorithm.sty
% stops with "command already defined" on the clash.
\makeatletter
\@for\@optkw:={For,ForAll,ForEach,While,Repeat,Until,If,ElseIf,ElsIf,Else,%
Loop,Function,Procedure,Return,Comment,Call,State,Statex,Require,Ensure,Do,%
Switch,Case,Other,Begin,Break,Continue,Block,Data,Result,Output,Input,KwTo,%
listofalgorithms}\do{\expandafter\let\csname\@optkw\endcsname\relax}
\makeatother
%%% -------------------------------------------------------------------------
"""

THM_FIX_HEAD = r"""%%% ---- theorem counters under jmlr (inserted by make_opt.py) --------------
% jmlrutils predeclares theorem, lemma, proposition, corollary, remark and
% definition on one shared counter, so its first Lemma after Theorem 1 would
% print as "Lemma 2".  This paper numbers each kind on its own counter, and the
% \setcounter calls in the appendix -- which keep Theorems 1-5 in one sequence
% across the body and the appendix -- depend on that.  Every name the paper
% declares below is released here first; releasing one jmlrutils never took is
% a no-op, so this list follows the \newtheorem lines automatically.
\makeatletter
\newcommand{\optreleasethm}[1]{%
  \expandafter\let\csname #1\endcsname\relax
  \expandafter\let\csname end#1\endcsname\relax
  \expandafter\let\csname the#1\endcsname\relax
  \expandafter\let\csname c@#1\endcsname\relax
  \expandafter\let\csname p@#1\endcsname\relax
  \expandafter\let\csname jmlr@thm@#1@body@font\endcsname\relax}
"""

THM_FIX_TAIL = r"""\makeatother
%%% -------------------------------------------------------------------------
"""


def link_layer(title: str, pdf_author: str | None) -> str:
    r"""What jmlr.cls's hyperref setup does not already do.

    The class loads hyperref and colours the links itself, so only the
    paper-specific repairs are left.  In anonymous mode no /Author is written:
    the PDF metadata is as much a part of a double-blind submission as the
    title page is.
    """
    author = (f"            pdfauthor={{{pdf_author}}},\n" if pdf_author else
              "            % pdfauthor deliberately unset: double-blind.\n")
    return (
        r"""%%% ---- link layer (inserted by make_opt.py) --------------------------------
% jmlr.cls loads hyperref and colours the links already.  This is the rest.
% bookmarks itself is a load-time option, and jmlr.cls has already loaded
% hyperref with it on, so setting it here would only draw a warning.
\hypersetup{breaklinks=true,
            bookmarksnumbered=true,
            pdfstartview=FitH,
"""
        + author
        + "            pdftitle={" + title + "}}\n"
        + r"""\makeatletter
% algorithmicx restarts its line counter in every algorithm, so hyperref would
% mint the destination ALG@line.3 once per float and keep only the first -- a
% \ref to a line of Algorithm 5 would jump to Algorithm 1.  \theH<counter> is
% the name hyperref uses for the destination; qualifying it with the algorithm
% number separates them.  Nothing visible changes.
\providecommand{\theHALG@line}{}
\renewcommand{\theHALG@line}{alg.\thealgorithm.\arabic{ALG@line}}
% \eqref prints "(13)" but hyperref links only the 13, leaving the parentheses
% outside the hot area.  Rebuilt around \ref*, so that the whole "(13)" is one
% target.  \tagform@ is amsmath's parenthesizer, kept so that the form of the
% tag stays whatever amsmath says it is.
\DeclareRobustCommand{\eqref}[1]{\hyperref[#1]{\textup{\tagform@{\ref*{#1}}}}}
% Four appendix subsection titles read "Proof of Theorem~\ref{...}", and titles
% are also PDF bookmarks, where markup is not allowed.  Without this, every one
% of them would warn and drop a token.
\pdfstringdefDisableCommands{\def\hyperref[#1]#2{#2}}
\makeatother
%%% -------------------------------------------------------------------------
"""
    )


def theorem_names(preamble: str) -> list[str]:
    r"""The environments the paper declares with \newtheorem, in order."""
    names: list[str] = []
    for line in preamble.splitlines():
        for hit in re.finditer(r"\\newtheorem\*?\s*\{([^}]*)\}", code_of(line)):
            if hit.group(1) not in names:
                names.append(hit.group(1))
    return names


def find_in_code(text: str, pat: re.Pattern) -> int | None:
    """Offset of the first match outside a comment, or None."""
    offset = 0
    for line in text.splitlines(keepends=True):
        hit = pat.search(code_of(line))
        if hit is not None:
            return offset + hit.start()
        offset += len(line)
    return None


def strip_groups(text: str, dropped: list[str]) -> str:
    r"""Remove the brace groups of DROP_GROUP whole, however many lines they run.

    \pdfinfo{...} and \lstset{...} are the two multi-line declarations to go,
    and neither can be recognized a line at a time.
    """
    for pat, why in DROP_GROUP:
        while (start := find_in_code(text, pat)) is not None:
            end = match_brace(text, text.index("{", start))
            gone = text[start:end].splitlines()
            dropped.append(f"{gone[0].strip()}  [{why}]"
                           + (f" (+{len(gone) - 1} more lines)" if len(gone) > 1 else ""))
            line_start = text.rfind("\n", 0, start) + 1
            before = text[line_start:start] if text[line_start:start].strip() else ""
            # The sentinel keeps the block it stood in marked as one that had
            # code, so that the comment explaining it goes when it does.
            text = text[:line_start] + before + GROUP_SENTINEL + text[end:]
    return text


def split_blocks(text: str) -> list[list[str]]:
    """The preamble as blank-line-separated blocks of lines."""
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.strip() == "":
            blocks.append([])
        else:
            blocks[-1].append(line)
    return [b for b in blocks if b]


def drop_comment_runs(block: list[str]) -> list[str]:
    """Drop each run of consecutive comment lines that carries a template line."""
    out: list[str] = []
    run: list[str] = []
    for line in block + ["\\@endblock"]:
        if _IS_COMMENT.match(line):
            run.append(line)
            continue
        if run and not any(DROP_COMMENT.match(ln) for ln in run):
            out += run
        run = []
        out.append(line)
    return out[:-1]


def transform_preamble(text: str) -> tuple[str, list[str], list[str]]:
    r"""AAAI preamble -> OPT preamble.

    Returns the translated text, the dropped lines each with its reason, and
    the packages carried over that this script has no opinion about -- which is
    what the build log asks you to look at when the AAAI paper gains one.

    Blank-line-separated blocks are the unit of work, because that is how the
    AAAI template is written: a comment explaining a group of \usepackage
    lines, then the lines.  A block whose code is entirely dropped takes its
    explanation with it, rather than leaving "These are recommended to typeset
    listings" standing over nothing.
    """
    names = theorem_names(text)
    thm_fix = (THM_FIX_HEAD
               + "".join(f"\\optreleasethm{{{n}}}\n" for n in names)
               + THM_FIX_TAIL)

    out: list[str] = []
    dropped: list[str] = []
    passed: list[str] = []
    algo_done = thm_done = False

    for block in split_blocks(strip_groups(text, dropped)):
        kept: list[str] = []
        had_code = False
        for line in drop_comment_runs(block):
            code = code_of(line)
            if not _IS_COMMENT.match(line) and code.strip():
                had_code = True
            if line.strip() == GROUP_SENTINEL:   # already recorded as dropped
                continue

            reason = next((why for pat, why in DROP_LINE
                           if re.search(pat, code)), None)
            if reason is not None:
                dropped.append(f"{line.strip()}  [{reason}]")
                continue

            # The two injections go immediately before what they fix.
            if not algo_done and re.search(r"\\usepackage\s*\{algorithm\}", code):
                kept.append(ALGO_FIX)
                algo_done = True
            if not thm_done and re.search(r"\\newtheorem", code):
                kept.append(thm_fix)
                thm_done = True

            pkg = re.search(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", code)
            if pkg:
                passed += [name.strip() for name in pkg.group(1).split(",")
                           if name.strip() not in KNOWN_KEEP]

            # A kept line may still carry an AAAI instruction as a trailing
            # comment.
            kept.append(re.sub(r"\s*%\s*DO NOT CHANGE THIS.*$", "", line))

        survives = any(not _IS_COMMENT.match(ln) and code_of(ln).strip()
                       for ln in kept)
        if kept and (survives or not had_code):
            out.append("\n".join(kept))

    # Whatever comment blocks close the preamble annotate the \title and author
    # block, which this script replaces; the AAAI template's advice on mixed
    # case would be left explaining a line that is no longer there.
    while out and not any(not _IS_COMMENT.match(ln) and code_of(ln).strip()
                          for ln in out[-1].splitlines()):
        out.pop()

    text_out = "\n\n".join(out)

    if not algo_done:
        raise SystemExit(
            r"no \usepackage{algorithm} in the AAAI preamble: the algorithm2e "
            "clash fix has nowhere to go, and the paper's algorithms would "
            "break silently.  Check make_opt.py against the preamble.")
    if not thm_done:
        raise SystemExit(
            r"no \newtheorem in the AAAI preamble: the jmlrutils theorem "
            "counters would stay shared.  Check make_opt.py against the "
            "preamble.")

    return text_out, dropped, passed


# --------------------------------------------------------------------------
# Single-column adjustments.
# --------------------------------------------------------------------------

_STARRED = re.compile(r"\\(begin|end)\{(figure|table|algorithm)\*\}")
_FLOAT_BEGIN = re.compile(r"\\begin\{(?:figure|table|algorithm)(\*?)\}")
_FLOAT_END = re.compile(r"\\end\{(?:figure|table|algorithm)\*?\}")
_COLUMNWIDTH = re.compile(r"\\columnwidth(?![a-zA-Z])")

# What \columnwidth meant in the AAAI sources, as a fraction of OPT's measure.
# aaai2027 sets \textwidth to 7in in two columns, so a column is 3.35in; OPT's
# text block is 433.62pt = 6.02in, and 0.556 of it is 3.35in again.  A figure
# drawn for one AAAI column therefore keeps the physical width it was drawn
# for, instead of being blown up to the full single-column measure -- which for
# the nanoGPT figure meant half a page.
AAAI_COLUMN_FRACTION = 0.556
AAAI_COLUMN = r"\aaaicolumnwidth"

SINGLE_COLUMN_FIX = r"""%%% ---- the AAAI column width (inserted by make_opt.py) --------------------
% \columnwidth in the AAAI sources is one column of two: 3.35in, aaai2027
% setting \textwidth to 7in.  Under opt2026 there is one column and
% \columnwidth is the whole 6.02in measure, so a figure asking for
% \columnwidth would come out at nearly twice the size it was drawn at.  Every
% \columnwidth outside a two-column float is rewritten to this length, which is
% the AAAI column again to within a hundredth of an inch.
\newlength{\aaaicolumnwidth}
\setlength{\aaaicolumnwidth}{""" + f"{AAAI_COLUMN_FRACTION}" + r"""\textwidth}
%
% The same 14% is why the appendix overflows its measure.  It was set one
% column wide at aaai2027's 7in; jmlr gives it 6.02in, and the prose is full of
% inline formulas that offer no break point --- a theorem statement whose only
% comma falls a third of the way in has to set the rest on one line or stretch
% the first past what LaTeX will normally accept.  \emergencystretch licenses
% exactly that stretch, in a final pass used only where the alternative is a
% line running into the margin; nothing else is touched, and a paragraph that
% already fits is set identically.
\emergencystretch=3.5em
\tolerance=1200
% Table columns keep 6pt of padding on each side by default, which is 84pt of
% white space in a seven-column table -- affordable at 7in, not at 6.02in.  The
% appendix's widest tables come back inside the measure at 4pt without any
% change to what they contain.
\setlength{\tabcolsep}{4pt}
%%% -------------------------------------------------------------------------
"""


def single_column_floats(text: str) -> tuple[str, int, int]:
    r"""Adapt the paper's floats to a one-column page.

    Two rewrites, in one pass because the second depends on the first's
    bookkeeping.  ``figure*`` becomes ``figure``, and likewise table and
    algorithm: a starred float is the two-column one, and although LaTeX
    accepts the star in a single-column document it will then place the float
    only at the top of a page or on a float page of its own, which in a paper
    this float-dense defers most of them to the end.

    And ``\columnwidth`` becomes ``\aaaicolumnwidth``, the width an AAAI column
    actually had -- except inside a starred float, which is the one place where
    AAAI's ``\columnwidth`` already meant the full text block and still does.

    Comments are left alone.
    """
    lines, unstarred, rescaled = [], 0, 0
    stack: list[bool] = []                    # one entry per open float, starred?
    for line in text.split("\n"):
        hit = _COMMENT.search(line)
        code, comment = (line, "") if hit is None else (line[:hit.start()], line[hit.start():])

        for begun in _FLOAT_BEGIN.finditer(code):
            stack.append(begun.group(1) == "*")
        if not (stack and stack[-1]):
            code, n = _COLUMNWIDTH.subn(AAAI_COLUMN.replace("\\", "\\\\"), code)
            rescaled += n
        for _ in _FLOAT_END.finditer(code):
            if stack:
                stack.pop()

        code, n = _STARRED.subn(r"\\\1{\2}", code)
        unstarred += n
        lines.append(code + comment)
    return "\n".join(lines), unstarred, rescaled


# --------------------------------------------------------------------------
# Front matter.
# --------------------------------------------------------------------------

def macro_value(text: str, name: str) -> str | None:
    r"""The body of \newcommand{\name}{...}, if the file defines it."""
    for hit in re.finditer(r"\\newcommand\s*\{\s*\\" + name + r"\s*\}", text):
        line_start = text.rfind("\n", 0, hit.start()) + 1
        if _COMMENT.search(text[line_start:hit.start()]):
            continue
        open_brace = text.index("{", hit.end())
        return " ".join(text[open_brace + 1: match_brace(text, open_brace) - 1].split())
    return None


def front_matter(aaai_preamble: str, mode: str) -> tuple[str, str]:
    r"""The \title / \optauthor block, and the title as plain text."""
    authors_src = AUTHORS.read_text(encoding="utf-8")
    _head, title_cmd, _tail = split_preamble(aaai_preamble)
    title = macro_value(authors_src, "opttitle") or title_of(title_cmd)
    short = macro_value(authors_src, "optshorttitle") or title

    block = [
        f"%%% Title.  Read from {PREAMBLE} unless opt_authors.tex defines",
        r"%%% \opttitle.  The bracketed short title is the running head, which",
        "%%% opt2026.cls suppresses; it is set for the PDF outline's sake.",
        f"\\title[{short}]{{{title}}}",
        "",
    ]
    if mode == "anon":
        block += [
            "%%% Authors: withheld.  opt2026.cls's [anon] option prints",
            r'%%% "author names withheld", and defines \optauthor and \acks to',
            "%%% swallow their argument, so nothing of opt_authors.tex reaches",
            "%%% this build but the two titles above.  Build with --mode final",
            "%%% to typeset the names.",
        ]
    else:
        block += [
            "%%% >>> begin opt_article/opt_authors.tex",
            authors_src.rstrip(),
            "%%% <<< end opt_article/opt_authors.tex",
        ]
    return "\n".join(block), title


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def patch_appendix(text: str) -> tuple[str, list[str]]:
    """Apply APPENDIX_PATCHES and APPENDIX_CUTS, each anchor exactly once."""
    for old, new, why in APPENDIX_PATCHES:
        found = text.count(old)
        if found != 1:
            raise SystemExit(
                f"appendix patch matched {found} times, expected 1:\n"
                f"  {old!r}\n  ({why})\n"
                "The AAAI appendix has been reworded. Update APPENDIX_PATCHES.")
        text = text.replace(old, new)

    cut: list[str] = []
    for opening, closing, why in APPENDIX_CUTS:
        for anchor in (opening, closing):
            if text.count(anchor) != 1:
                raise SystemExit(
                    f"appendix cut anchor occurs {text.count(anchor)} times, "
                    f"expected 1:\n  {anchor!r}\n  ({why})\n"
                    "The AAAI appendix has been reworded. Update APPENDIX_CUTS.")
        start, end = text.index(opening), text.index(closing)
        if end <= start:
            raise SystemExit(
                f"appendix cut runs backwards: {closing!r} precedes {opening!r}")
        gone = text[start:end]
        if "\\label{" in gone:
            raise SystemExit(
                "appendix cut would delete a \\label, and something still "
                f"points at it:\n  {gone[:80]!r}...")
        cut.append(f"{len(gone.split())} words  [{why}]")
        text = text[:start] + text[end:]
    return text, cut


def splice_extra(text: str) -> str:
    r"""Insert opt_appendix_extra.tex just after the appendix's \section line.

    It goes first, ahead of the AAAI appendix's own subsections, because what
    it holds is main-text material: a reader looking for the federated results
    the five-page limit displaced should not have to pass the proofs first.
    """
    extra = OPT_APPENDIX_EXTRA.read_text(encoding="utf-8").rstrip()
    for lineno, line in enumerate(text.splitlines()):
        if re.search(r"\\section\s*\{", code_of(line)):
            lines = text.splitlines()
            return "\n".join(
                lines[: lineno + 1]
                + ["", f"%%% >>> begin {OPT_APPENDIX_EXTRA.name}", extra,
                   f"%%% <<< end {OPT_APPENDIX_EXTRA.name}", ""]
                + lines[lineno + 1:])
    raise SystemExit(
        r"no \section in the appendix, so opt_appendix_extra.tex has nowhere "
        "to go; the floats it carries are referenced from the appendix and "
        "would leave undefined references behind them.")


def record_origin() -> None:
    """Record the AAAI body as it stands: its hash, and a copy to diff against."""
    import hashlib
    raw = (SRC / BODY).read_bytes()
    ORIGIN.write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
    with open(ORIGIN_TEXT, "wb") as fh:
        fh.write(raw)


def body_diff() -> list[str]:
    """The AAAI body's changes since the fork was last reconciled."""
    import difflib
    if not ORIGIN_TEXT.exists():
        return []
    then = ORIGIN_TEXT.read_text(encoding="utf-8", errors="replace").splitlines()
    now = (SRC / BODY).read_text(encoding="utf-8", errors="replace").splitlines()
    return list(difflib.unified_diff(then, now, fromfile=f"{BODY} (at fork)",
                                     tofile=f"{BODY} (now)", lineterm="", n=1))


def body_drift() -> str | None:
    """Whether the AAAI body has moved since the fork was last reconciled.

    The appendix is derived from the AAAI source and follows it automatically;
    the body is a fork and does not.  That is the whole cost of forking, so the
    build reports it, and reports it in a form that can be acted on: the number
    of changed hunks now, the diff itself under --drift.
    """
    import hashlib
    current = hashlib.sha256((SRC / BODY).read_bytes()).hexdigest()
    if not ORIGIN.exists() or not ORIGIN_TEXT.exists():
        record_origin()
        return None
    if ORIGIN.read_text(encoding="utf-8").strip() == current:
        return None

    diff = body_diff()
    hunks = sum(1 for line in diff if line.startswith("@@"))
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    return (f"{BODY} has changed since opt_body.tex was last reconciled with it:\n"
            f"  {hunks} hunk(s), +{added} -{removed} lines.\n"
            f"  Read them with   python make_opt.py --drift\n"
            f"  Port what belongs in the OPT edition into {OPT_BODY.name}, then\n"
            f"  rerun with --reconciled to record the new state.")


def build_document(mode: str) -> tuple[str, str, dict]:
    aaai_preamble = (SRC / PREAMBLE).read_text(encoding="utf-8")
    head, _title_cmd, _tail = split_preamble(aaai_preamble)
    preamble, dropped, passed = transform_preamble(head)
    titles, title = front_matter(aaai_preamble, mode)

    pdf_author = None
    if mode != "anon":
        pdf_author = macro_value(AUTHORS.read_text(encoding="utf-8"), "optpdfauthor")

    body, n_body = linkify_named_refs(flatten(OPT_BODY))
    patched, cut = patch_appendix(flatten(SRC / APPENDIX))
    appendix, n_app = linkify_named_refs(splice_extra(patched))
    body, s_body, w_body = single_column_floats(body)
    appendix, s_app, w_app = single_column_floats(appendix)

    options = {"anon": "anon", "named": "", "final": "final"}[mode]
    doc = [
        BANNER.format(title=title, mode=mode,
                      sources=", ".join([PREAMBLE, APPENDIX])),
        CLASS_HEADER.format(options=options),
        preamble,
        SINGLE_COLUMN_FIX,
        link_layer(title, pdf_author),
        titles,
        "",
        r"\begin{document}",
        "",
        f"%%% >>> begin {OPT_BODY.name}   (it opens with \\maketitle and the "
        "abstract)",
        body.lstrip("\ufeff").rstrip(),
        f"%%% <<< end {OPT_BODY.name}",
        "%%% The page the main text ends on, which is the number OPT's limit is",
        "%%% counted against; make_opt.py reads it back out of main.aux.  No",
        "%%% blank line above: a comment does not end a paragraph but a blank",
        "%%% line does, and a \\label in a paragraph of its own is pushed to the",
        "%%% next page whenever the Conclusion ends at the foot of one.",
        f"\\label{{{BODY_END_LABEL}}}",
        "",
        "%%% No back matter.  Acknowledgments, author contributions and code",
        "%%% availability all name people or a repository, and OPT reviews",
        "%%% double-blind; none of the three is carried into any build, so",
        "%%% there is no mode in which one can be left in by accident.",
        "",
        "%%% The single reference list, between the body and the appendix,",
        "%%% where OPT's sample.tex puts it.  jmlr.cls selects plainnat and",
        "%%% passes natbib the numbers option: citations are [7] here, not",
        "%%% (Author 2026) as under AAAI.",
        r"\bibliography{references}",
        "",
        f"%%% >>> begin {APPENDIX}   (it opens with \\appendix)",
        appendix,
        f"%%% <<< end {APPENDIX}",
        "",
        r"\end{document}",
        "",
    ]
    stats = {
        "dropped": dropped,
        "passed": passed,
        "linked": n_body + n_app,
        "unstarred": s_body + s_app,
        "rescaled": w_body + w_app,
        "cut": cut,
        "theorems": theorem_names(head),
    }
    return "\n".join(doc), title, stats


def sanity_check(doc: str, mode: str) -> None:
    """Guard the invariants the conversion is supposed to establish."""
    code = "\n".join(code_of(line) for line in doc.splitlines())
    problems = []
    if code.count(r"\bibliography{") != 1:
        problems.append("expected one \\bibliography, found "
                        f"{code.count(chr(92) + 'bibliography{')}")
    if _INPUT.search(code):
        problems.append("an \\input survived flattening")
    for banned, why in [
        (r"\externaldocument", "xr cross-document references are unnecessary now"),
        (r"\usepackage{xr}", "xr is unnecessary now"),
        ("aaai2027", "the AAAI style file must not be loaded"),
        (r"\onecolumn", "opt2026 is single-column already"),
        (r"\affiliations", "an AAAI author block survived"),
        (r"\pdfinfo", "the AAAI template stamp survived"),
    ]:
        if banned in code:
            problems.append(f"{banned!r} still present: {why}")
    if code.count(r"\documentclass") != 1:
        problems.append("expected exactly one \\documentclass")
    if code.count(r"\begin{document}") != 1 or code.count(r"\end{document}") != 1:
        problems.append("document environment is not balanced")
    if code.count(r"\maketitle") != 1:
        problems.append("expected exactly one \\maketitle (the body opens with it)")
    if re.search(r"\\begin\{(figure|table|algorithm)\*\}", code):
        problems.append("a two-column float survived unstarring")

    # The fork's own failure mode: opt_body.tex drops a float or a display the
    # appendix still points at.  LaTeX would report it as an undefined
    # reference twenty minutes of reading later; this names it now.
    # Only inside the document: the preamble's \pdfstringdefDisableCommands
    # redefines \hyperref, and its #1 is an argument, not a label.
    inside = code.split(r"\begin{document}", 1)[-1]
    defined = set(re.findall(r"\\label\{([^}]*)\}", inside))
    referenced = set(re.findall(r"\\(?:ref|eqref|cref|Cref|autoref)\*?\{([^}]*)\}", inside))
    referenced |= set(re.findall(r"\\hyperref\[([^\]]*)\]", inside))
    for lost in sorted(referenced - defined):
        n = inside.count("{" + lost + "}") + inside.count("[" + lost + "]")
        problems.append(
            f"label {lost!r} is referenced {n} times and defined nowhere -- "
            "the cut dropped what defined it")
    if mode == "anon":
        for leak in ("Anonymous Submission", "Anonymous Institution"):
            if leak in code:
                problems.append(f"{leak!r} survived: opt2026.cls anonymizes by itself")
    if problems:
        raise SystemExit("assembled document failed its checks:\n  - "
                         + "\n  - ".join(problems))


def todos(mode: str) -> list[str]:
    if mode == "anon":
        return []
    return [f"{AUTHORS.name}:{lineno}: {line.strip()}"
            for lineno, line in enumerate(
                AUTHORS.read_text(encoding="utf-8").splitlines(), 1)
            if "TODO" in code_of(line)]


# --------------------------------------------------------------------------
# Assets and compilation.
# --------------------------------------------------------------------------

def copy_assets(doc: str, build: Path, keep_bib: bool) -> list[str]:
    for name in SUPPORT:
        source = STYLE / name
        if not source.exists():
            raise SystemExit(
                f"missing class file: {source}\n"
                "unpack style/opt2026_style.zip, or fetch it again from "
                "https://opt-ml.org/opt2026_style.zip")
        shutil.copy2(source, build / name)
    if keep_bib:
        shutil.copy2(SRC / BIB, build / BIB)

    wanted = sorted({m.group(1).strip()
                     for line in doc.splitlines()
                     for m in _GRAPHICS.finditer(code_of(line))})
    copied, missing = [], []
    for rel in wanted:
        source = SRC / rel
        if not source.exists():
            missing.append(rel)
            continue
        target = build / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel)
    if missing:
        raise SystemExit("images referenced but not found:\n  - " + "\n  - ".join(missing))
    return copied


def compile_pdf(build: Path, keep_work: bool, mode: str) -> Path:
    """Compile a throwaway copy, so `build` holds only the sources."""
    if shutil.which("latexmk") is None:
        raise SystemExit("latexmk not on PATH; rerun with --no-compile")
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    shutil.copytree(build, WORKDIR)

    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
           f"{JOBNAME}.tex"]
    run = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True)
    log = WORKDIR / f"{JOBNAME}.log"
    if run.returncode != 0:
        tail = (log.read_text(encoding="utf-8", errors="replace").splitlines()
                if log.exists() else [])
        errors = [ln for ln in tail if ln.startswith("!")] or tail[-30:]
        raise SystemExit("latexmk failed:\n  " + "\n  ".join(errors[:20])
                         + f"\n\nfull log: {log}")

    bbl = WORKDIR / f"{JOBNAME}.bbl"
    if not bbl.exists():
        raise SystemExit(f"no {JOBNAME}.bbl was produced; the bibliography did not run")
    shutil.copy2(bbl, build / f"{JOBNAME}.bbl")

    pdf = HERE / preview_name(mode, ".pdf")
    shutil.copy2(WORKDIR / f"{JOBNAME}.pdf", pdf)
    report_log(log)
    report_length(WORKDIR / f"{JOBNAME}.aux", mode)
    if not keep_work:
        shutil.rmtree(WORKDIR)
    return pdf


def report_length(aux: Path, mode: str) -> None:
    """How long the main text runs, against what OPT allows.

    References and appendix do not count, so the figure that matters is the
    page the reference list starts on.
    """
    limit = PAGE_LIMIT[mode]
    if not aux.exists():
        return
    text = aux.read_text(encoding="utf-8", errors="replace")
    hit = re.search(r"\\newlabel\{" + re.escape(BODY_END_LABEL)
                    + r"\}\{\{[^{}]*\}\{(\d+)\}", text)
    if hit is None:
        print(f"  main text      : ? pages (OPT allows {limit})")
        return
    pages = int(hit.group(1))
    over = "" if pages <= limit else f"   <-- {pages - limit} over"
    print(f"  main text      : {pages} pages, limit {limit}{over}")


def preview_name(mode: str, suffix: str) -> str:
    """The anonymous build is the submission, so it keeps the plain name."""
    return "signmuon_opt" + ("" if mode == "anon" else f"_{mode}") + suffix


def pack_zip(build: Path, mode: str) -> Path:
    archive = HERE / preview_name(mode, ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(build.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(build)).replace("\\", "/"))
    return archive


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["anon", "named", "final"], default="anon",
                    help="anon: double-blind submission (default).  named: "
                         "author names shown.  final: camera-ready -- names "
                         "shown and the proceedings' editor line cleared.")
    ap.add_argument("-o", "--build", type=Path, default=DEFAULT_BUILD,
                    help=f"output directory (default: {DEFAULT_BUILD})")
    ap.add_argument("--no-compile", action="store_true",
                    help="assemble the sources but do not run latexmk")
    ap.add_argument("--no-bib", action="store_true",
                    help="omit references.bib; the shipped main.bbl is enough")
    ap.add_argument("--no-zip", action="store_true", help="skip the archive")
    ap.add_argument("--drift", action="store_true",
                    help="print what has changed in the AAAI body since the "
                         "fork was last reconciled, then stop")
    ap.add_argument("--reconciled", action="store_true",
                    help="record the AAAI body as reconciled with opt_body.tex; "
                         "pass it once you have ported an AAAI revision into "
                         "the fork, to clear the drift warning")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the scratch compile directory for debugging")
    args = ap.parse_args()

    if args.drift:
        diff = body_diff()
        if diff:
            print("\n".join(diff))
        else:
            print(f"{BODY} is unchanged since opt_body.tex was reconciled "
                  "with it.")
        return 0

    build: Path = args.build.resolve()
    print(f"reading  {SRC}")
    drift = body_drift()
    doc, title, stats = build_document(args.mode)
    sanity_check(doc, args.mode)

    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    with open(build / f"{JOBNAME}.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    images = copy_assets(doc, build, keep_bib=not args.no_bib)

    print(f"writing  {build}")
    print(f"  title          : {title}")
    print(f"  mode           : {args.mode}")
    print(f"  {JOBNAME}.tex       : {len(doc.splitlines())} lines")
    print(f"  images         : {len(images)}")
    print(f"  preamble       : {len(stats['dropped'])} lines dropped, "
          f"{len(stats['theorems'])} theorem environments re-declared")
    print(f"  floats         : {stats['unstarred']} two-column floats unstarred, "
          f"{stats['rescaled']} \\columnwidth held to the AAAI column")
    print(f"  named refs     : {stats['linked']} linked whole (\"Theorem 1\")")
    for item in stats["cut"]:
        print(f"  appendix cut   : {item}")

    if not args.no_compile:
        pdf = compile_pdf(build, args.keep_work, args.mode)
        print(f"  preview        : {pdf}")

    if not args.no_zip:
        archive = pack_zip(build, args.mode)
        print(f"  zip            : {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    unknown = sorted({p for p in stats["passed"]})
    if unknown:
        print("\nPackages carried over from the AAAI preamble that this script "
              "has no rule for.\nThey are in the document; check that they "
              "behave under opt2026.cls:")
        for name in unknown:
            print(f"  {name}")

    left = todos(args.mode)
    if left:
        print("\nTODO left in the front matter -- edit before submitting:")
        for item in left:
            print(f"  {item}")

    if args.reconciled:
        record_origin()
        print(f"\nrecorded {BODY} as reconciled with {OPT_BODY.name}.")
    elif drift is not None:
        print(f"\nFORK DRIFT\n  {drift}")

    print(f"\nSubmit {HERE / preview_name(args.mode, '.pdf')} to OpenReview; "
          f"the sources are in {build}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
