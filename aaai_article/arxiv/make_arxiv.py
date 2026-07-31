#!/usr/bin/env python3
"""Build the arXiv edition of the AAAI paper.

The AAAI submission is three documents sharing four source files: a 7-page
main paper, a separate supplementary PDF, and v2_SignMuon_AAAI_full.tex, which
already rejoins the two.  arXiv wants something the submission cannot be: one
self-contained document, with one reference list, named authors, and links you
can click.  This script produces it without touching a single source file, so
the AAAI build stays exactly as it is.

What it does, in order:

  1.  Flattens signmuon_preamble / signmuon_body / signmuon_appendix and every
      file they \\input into one main.tex.  No \\input survives, and neither
      does xr -- the cross-document reference dance is unnecessary once the
      appendix is in the same document.
  2.  Emits one \\bibliography, placed after the Conclusion as AAAI puts it, so
      that citations from the appendix land in the same list as the body's.
  3.  Replaces the "Anonymous Submission" block with arxiv_authors.tex, and
      switches the aaai2027 option from `submission` to `preprint`, which is
      what shows the author names and drops the "anonymized submission"
      copyright slug.  The title is still read out of the preamble.
  4.  Splices arxiv_backmatter.tex (acknowledgments, author contributions,
      code availability) in before the references.
  5.  Adds the link layer: hyperref, loaded ahead of cleveref and then hidden
      from the \\PackageError that aaai2027.sty raises against it.  This is the
      only change to how the paper looks -- refs, citations and URLs become
      coloured links, and nothing moves on the page.
  6.  Copies aaai2027.sty, aaai2027.bst, references.bib and exactly those
      images the document actually includes into the build directory.
  7.  Compiles in a scratch copy, so the build directory itself stays free of
      .aux clutter, copies the resulting main.bbl back beside main.tex (arXiv
      wants the .bbl, not a bibtex run), and reports what the log complained
      about.

The build directory is then what you upload; a tarball of it is written
alongside for convenience, and the compiled PDF is left outside it so arXiv
does not mistake the submission for a PDF-only one.

    python make_arxiv.py                 # build, compile, check, pack
    python make_arxiv.py --no-compile    # just assemble the sources
    python make_arxiv.py --keep-work     # leave the scratch build for debugging

Nothing here writes to the AAAI sources.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent      # aaai_article/arxiv
SRC = HERE.parent                           # aaai_article

# Sources of the AAAI build, read and never written.
PREAMBLE = "signmuon_preamble.tex"
BODY = "signmuon_body.tex"
APPENDIX = "signmuon_appendix.tex"
BIB = "references.bib"
SUPPORT = ["aaai2027.sty", "aaai2027.bst"]

# The two files that hold what the anonymous sources cannot.
AUTHORS = HERE / "arxiv_authors.tex"
BACKMATTER = HERE / "arxiv_backmatter.tex"

JOBNAME = "main"
DEFAULT_BUILD = HERE / "build"
WORKDIR = HERE / ".build-work"

BANNER = r"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GENERATED FILE -- DO NOT EDIT.
%
%  arXiv edition of "{title}".
%  Assembled by aaai_article/arxiv/make_arxiv.py from the AAAI sources:
%      {sources}
%  plus arxiv/arxiv_authors.tex and arxiv/arxiv_backmatter.tex.
%
%  Edit those, not this file, and rerun:  python arxiv/make_arxiv.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# hyperref is forbidden by aaai2027.sty, which tests \@ifpackageloaded at
# \begin{document} -- after every \AtBeginDocument hook we could register, so
# the check cannot be undone later.  It reads \ver@hyperref.sty, though, and
# that control sequence has no other job once hyperref and cleveref have both
# had their say.  Letting it to \relax leaves both packages fully working and
# the style file byte-identical to the one AAAI ships.
LINK_LAYER = r"""%%% ---- arXiv link layer (inserted by arxiv/make_arxiv.py) -----------------
% The one deviation from the AAAI submission: internal references, citations
% and URLs become clickable.  Type, spacing and layout are untouched.
%
% aaai2027.sty raises \PackageError if hyperref is loaded, and does so from a
% hook that fires after any \AtBeginDocument we could add.  So hyperref is
% loaded here, cleveref is loaded after it (that order is what makes \cref
% produce links), and then \ver@hyperref.sty -- the flag \@ifpackageloaded
% reads, and all that the check reads -- is let to \relax.  Both packages stay
% fully functional; the style file is not modified.
\definecolor{arxivlinkcolor}{rgb}{0.10,0.25,0.60}
\definecolor{arxivcitecolor}{rgb}{0.10,0.40,0.20}
\definecolor{arxivurlcolor}{rgb}{0.55,0.10,0.10}
\usepackage[colorlinks=true,
            linkcolor=arxivlinkcolor,
            citecolor=arxivcitecolor,
            urlcolor=arxivurlcolor,
            breaklinks=true,
            bookmarks=true,
            bookmarksnumbered=true,
            pdfstartview=FitH]{hyperref}
\usepackage{cleveref}  % after hyperref, so that \cref and \Cref link
\makeatletter
% algorithmicx restarts its line counter in every algorithm, so hyperref would
% mint the destination ALG@line.3 once per float and keep only the first --
% a \ref to a line of Algorithm 5 would jump to Algorithm 1.  \theH<counter> is
% the name hyperref uses for the destination; qualifying it with the algorithm
% number separates them.  Nothing visible changes.
\providecommand{\theHALG@line}{}
\renewcommand{\theHALG@line}{alg.\thealgorithm.\arabic{ALG@line}}
% \eqref prints "(13)" but hyperref links only the 13, leaving the parentheses
% outside the hot area. Rebuilt here around \ref*, so that the whole "(13)" is
% one target. \tagform@ is amsmath's parenthesizer, kept so the form of the
% tag is whatever amsmath says it is.
\DeclareRobustCommand{\eqref}[1]{\hyperref[#1]{\textup{\tagform@{\ref*{#1}}}}}
% Four appendix subsection titles read "Proof of Theorem~\ref{...}", and
% titles are also PDF bookmarks, where markup is not allowed. Without this,
% every one of them would warn and drop a token; with it, the bookmark reads
% the reference plainly.
\pdfstringdefDisableCommands{\def\hyperref[#1]#2{#2}}
\expandafter\let\csname ver@hyperref.sty\endcsname\relax
\makeatother
%%% ---- end arXiv link layer -----------------------------------------------
"""

# Names the paper puts in front of a \ref. Taken from what the sources
# actually write, plus the obvious neighbours, so that adding a "Definition~"
# or an "Example~" to the paper later needs no change here. Plurals are listed
# because "Theorems~\ref{a}--\ref{b}" is as common as the singular.
REF_NAMES = [
    "Appendix", "Appendices", "Section", "Sections", "Subsection", "Subsections",
    "Theorem", "Theorems", "Proposition", "Propositions", "Lemma", "Lemmas",
    "Corollary", "Corollaries", "Assumption", "Assumptions", "Remark", "Remarks",
    "Hypothesis", "Hypotheses", "Definition", "Definitions", "Example", "Examples",
    "Figure", "Figures", "Table", "Tables", "Algorithm", "Algorithms",
    "Equation", "Equations", "Part", "Parts", "Step", "Steps", "Line", "Lines",
    "Item", "Items", "Chapter", "Chapters",
]

# "Theorem~\ref{th:1}" -> the word is literal text and only the number is a
# link. The name, whichever way it is capitalized, then the separator the
# source used (a tie, a thin space, or a line break), then the \ref.
# (?<![A-Za-z\\]) keeps "\Table" and the tail of a longer word out.
_NAMED_REF = re.compile(
    r"(?<![A-Za-z\\])("
    + "|".join(f"[{n[0]}{n[0].lower()}]{n[1:]}" for n in REF_NAMES)
    + r")(~|\\,|\\ |[ \t]*\n[ \t]*|[ \t]+)\\ref\{([^}\\]+)\}"
)


# --------------------------------------------------------------------------
# LaTeX source scanning.  Everything below reads TeX with comments stripped,
# so that a commented-out \input or \includegraphics is not acted on.
# --------------------------------------------------------------------------

_COMMENT = re.compile(r"(?<!\\)%")
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
_GRAPHICS = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")


def code_of(line: str) -> str:
    """The part of a line TeX acts on: everything before an unescaped %."""
    hit = _COMMENT.search(line)
    return line if hit is None else line[: hit.start()]


def flatten(path: Path, seen: list[str] | None = None) -> str:
    """Expand \\input and \\include recursively, one file per standalone line.

    Each expansion is fenced by comments naming the file it came from, so the
    generated main.tex can still be read against the sources.
    """
    seen = [] if seen is None else seen
    if path.name in seen:
        raise SystemExit(f"input loop: {' -> '.join(seen + [path.name])}")
    if not path.exists():
        raise SystemExit(f"missing source: {path}")

    out: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        hit = _INPUT.search(code_of(line))
        if hit is None:
            out.append(line)
            continue
        before = code_of(line)[: hit.start()].strip()
        after = code_of(line)[hit.end():].strip()
        if before or after:
            raise SystemExit(
                f"{path.name}:{lineno}: \\input shares a line with other code; "
                "put it on a line of its own so it can be flattened."
            )
        child = hit.group(1).strip()
        child_path = path.parent / (child if child.endswith(".tex") else child + ".tex")
        out.append(f"%%% >>> begin {child_path.name} (was \\input on {path.name}:{lineno})")
        out.append(flatten(child_path, seen + [path.name]))
        out.append(f"%%% <<< end {child_path.name}")
    return "\n".join(out)


def linkify_named_refs(text: str) -> tuple[str, int]:
    """Pull the name of a cross-reference inside its hyperlink.

    "Theorem~\\ref{th:1}" sets the word as ordinary text and links only the
    number, so the reader has a two-character target to hit. Rewritten as
    \\hyperref[th:1]{Theorem~\\ref*{th:1}} the whole "Theorem 1" is the link.

    Printed output is identical, which is the reason for doing it this way
    rather than with \\cref: cleveref would resolve the name itself, and the
    paper's 46 "Appendix~\\ref{...}" point at subsections, which cleveref would
    rename to "Subsection". The word the author wrote is the word that stays.

    Comments are left alone -- a commented-out reference stays commented out --
    and so is the line count, which the assertion below enforces: the
    separator is reproduced verbatim, so a reference split across two lines
    stays split across the same two lines.
    """
    code: list[str] = []
    comments: list[str] = []
    for line in text.split("\n"):
        hit = _COMMENT.search(line)
        if hit is None:
            code.append(line)
            comments.append("")
        else:
            code.append(line[: hit.start()])
            comments.append(line[hit.start():])

    def replace(m: re.Match) -> str:
        name, sep, label = m.group(1), m.group(2), m.group(3)
        return f"\\hyperref[{label}]{{{name}{sep}\\ref*{{{label}}}}}"

    before = "\n".join(code)
    joined, count = _NAMED_REF.subn(replace, before)
    rebuilt = joined.split("\n")
    if len(rebuilt) != len(comments):
        raise SystemExit("linkify changed the line count; refusing to guess where")
    # Every rewrite turns one \ref into one \ref* and adds one \hyperref:
    # no reference may be invented, duplicated or lost.
    if (joined.count(r"\ref{"), joined.count(r"\ref*{"), joined.count(r"\hyperref[")) != (
        before.count(r"\ref{") - count, count, count
    ):
        raise SystemExit("linkify did not conserve the references; not shipping that")
    return "\n".join(c + k for c, k in zip(rebuilt, comments)), count


def match_brace(text: str, start: int) -> int:
    """Index just past the '}' matching the '{' at `start`, comments ignored."""
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "%":
            nl = text.find("\n", i)
            i = len(text) if nl < 0 else nl + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise SystemExit(f"unbalanced braces starting at offset {start}")


def split_preamble(text: str) -> tuple[str, str, str]:
    """Return (head, \\title{...}, tail) around the first \\title.

    The tail is the anonymous author block and the commented-out AAAI examples,
    which arxiv_authors.tex replaces wholesale.
    """
    for hit in re.finditer(r"\\title\s*\{", text):
        line_start = text.rfind("\n", 0, hit.start()) + 1
        if "%" in code_of(text[line_start:hit.start()]):
            continue  # a \title inside a comment
        if _COMMENT.search(text[line_start:hit.start()]):
            continue
        end = match_brace(text, text.index("{", hit.start()))
        return text[:line_start], text[line_start:end], text[end:]
    raise SystemExit(f"no \\title found in {PREAMBLE}")


def title_of(title_cmd: str) -> str:
    """The title as plain text, for the PDF metadata."""
    inner = title_cmd[title_cmd.index("{") + 1: -1]
    return " ".join(inner.split())


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def build_preamble() -> tuple[str, str]:
    """The patched preamble, and the paper title as plain text."""
    text = (SRC / PREAMBLE).read_text(encoding="utf-8")

    # `submission` anonymizes the authors and prints the review-only slug;
    # `preprint` shows the names and prints no slug, which is what arXiv wants.
    patched, n = re.subn(
        r"(\\usepackage\s*\[)([^\]]*)(\]\s*\{aaai2027\})",
        lambda m: m.group(1) + m.group(2).replace("submission", "preprint") + m.group(3),
        text,
    )
    if n != 1 or "preprint" not in patched:
        raise SystemExit(
            "could not switch aaai2027 to [preprint]; check the \\usepackage "
            f"line in {PREAMBLE}"
        )
    text = patched

    # hyperref must precede cleveref, so the link layer takes the latter over.
    if text.count("\\usepackage{cleveref}") != 1:
        raise SystemExit(f"expected exactly one \\usepackage{{cleveref}} in {PREAMBLE}")
    text = text.replace("\\usepackage{cleveref}", LINK_LAYER.rstrip("\n"), 1)

    head, title_cmd, _drop = split_preamble(text)
    title = title_of(title_cmd)

    authors = AUTHORS.read_text(encoding="utf-8").rstrip()
    meta = (
        "\n%%% PDF metadata, from the title above and arxiv_authors.tex.\n"
        "\\hypersetup{%\n"
        f"  pdftitle={{{title}}},\n"
        "  pdfauthor={\\arxivpdfauthor},\n"
        "  pdfcreator={pdfLaTeX}}\n"
    )
    return "".join([head, title_cmd, "\n\n", authors, "\n", meta]), title


def build_document() -> tuple[str, str, int]:
    preamble, title = build_preamble()
    body, n_body = linkify_named_refs(flatten(SRC / BODY))
    appendix, n_app = linkify_named_refs(flatten(SRC / APPENDIX))
    backmatter = BACKMATTER.read_text(encoding="utf-8").rstrip()

    banner = BANNER.format(
        title=title,
        sources=", ".join([PREAMBLE, BODY, APPENDIX]),
    )

    doc = "\n".join([
        banner,
        preamble,
        r"\begin{document}",
        "",
        f"%%% >>> begin {BODY}",
        body,
        f"%%% <<< end {BODY}",
        "",
        "%%% >>> begin arxiv/arxiv_backmatter.tex",
        backmatter,
        "%%% <<< end arxiv/arxiv_backmatter.tex",
        "",
        "%%% The single reference list.  It sits after the Conclusion, where",
        "%%% AAAI puts it, and serves the appendix below as well: with both",
        "%%% halves in one document there is nothing left to duplicate.",
        "%%% aaai2027.sty selects aaai2027.bst on its own.",
        r"\bibliography{references}",
        "",
        r"\onecolumn",
        f"%%% >>> begin {APPENDIX}",
        appendix,
        f"%%% <<< end {APPENDIX}",
        "",
        r"\end{document}",
        "",
    ])
    return doc, title, n_body + n_app


def sanity_check(doc: str) -> None:
    """Guard the invariants the conversion is supposed to establish."""
    code = "\n".join(code_of(line) for line in doc.splitlines())
    problems = []
    if code.count(r"\bibliography{") != 1:
        problems.append(
            f"expected one \\bibliography, found {code.count(chr(92) + 'bibliography{')}"
        )
    if _INPUT.search(code):
        problems.append("an \\input survived flattening")
    for banned, why in [
        (r"\externaldocument", "xr cross-document references are unnecessary now"),
        (r"\usepackage{xr}", "xr is unnecessary now"),
        ("Anonymous Submission", "the anonymous author block survived"),
        ("Anonymous Institution", "the anonymous affiliation survived"),
    ]:
        if banned in code:
            problems.append(f"{banned!r} still present: {why}")
    if code.count(r"\begin{document}") != 1 or code.count(r"\end{document}") != 1:
        problems.append("document environment is not balanced")
    if problems:
        raise SystemExit("assembled document failed its checks:\n  - " + "\n  - ".join(problems))


def todos() -> list[str]:
    found = []
    for path in (AUTHORS, BACKMATTER):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "TODO" in code_of(line):
                found.append(f"{path.name}:{lineno}: {line.strip()}")
    return found


# --------------------------------------------------------------------------
# Assets and compilation.
# --------------------------------------------------------------------------

def copy_assets(doc: str, build: Path, keep_bib: bool) -> list[str]:
    for name in SUPPORT:
        shutil.copy2(SRC / name, build / name)
    if keep_bib:
        shutil.copy2(SRC / BIB, build / BIB)

    wanted = sorted({
        m.group(1).strip()
        for line in doc.splitlines()
        for m in _GRAPHICS.finditer(code_of(line))
    })
    copied = []
    missing = []
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


def compile_pdf(build: Path, keep_work: bool) -> Path:
    """Compile a throwaway copy, so `build` holds only what arXiv receives."""
    if shutil.which("latexmk") is None:
        raise SystemExit("latexmk not on PATH; rerun with --no-compile")
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    shutil.copytree(build, WORKDIR)

    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{JOBNAME}.tex"]
    run = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True)
    log = WORKDIR / f"{JOBNAME}.log"
    if run.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        errors = [ln for ln in tail if ln.startswith("!")] or tail[-30:]
        raise SystemExit(
            "latexmk failed:\n  " + "\n  ".join(errors[:20])
            + f"\n\nfull log: {log}"
        )

    bbl = WORKDIR / f"{JOBNAME}.bbl"
    if not bbl.exists():
        raise SystemExit(f"no {JOBNAME}.bbl was produced; the bibliography did not run")
    shutil.copy2(bbl, build / f"{JOBNAME}.bbl")

    pdf = HERE / "signmuon_arxiv.pdf"
    shutil.copy2(WORKDIR / f"{JOBNAME}.pdf", pdf)
    report_log(log)
    if not keep_work:
        shutil.rmtree(WORKDIR)
    return pdf


def report_log(log: Path) -> None:
    text = log.read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on .*?\((\d+) pages", text)
    print(f"  pages          : {pages.group(1) if pages else '?'}")

    def count(pattern: str) -> int:
        return len(re.findall(pattern, text))

    for label, pattern in [
        ("undefined refs", r"Reference `[^']*' on page \d+ undefined"),
        ("undefined cites", r"Citation `[^']*' on page \d+ undefined"),
        ("duplicate links", r"destination with the same identifier"),
        ("overfull hbox", r"Overfull \\hbox"),
    ]:
        n = count(pattern)
        flag = "" if n == 0 or label == "overfull hbox" else "   <-- fix this"
        print(f"  {label:15}: {n}{flag}")


def pack(build: Path) -> Path:
    tarball = HERE / "signmuon_arxiv.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for path in sorted(build.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(build)).replace("\\", "/"))
    return tarball


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--build", type=Path, default=DEFAULT_BUILD,
                    help=f"output directory (default: {DEFAULT_BUILD})")
    ap.add_argument("--no-compile", action="store_true",
                    help="assemble the sources but do not run latexmk")
    ap.add_argument("--no-bib", action="store_true",
                    help="omit references.bib; the shipped main.bbl is what arXiv uses")
    ap.add_argument("--no-tar", action="store_true", help="skip the tarball")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the scratch compile directory for debugging")
    args = ap.parse_args()

    build: Path = args.build.resolve()
    print(f"reading  {SRC}")
    doc, title, linked = build_document()
    sanity_check(doc)

    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    # Explicit LF: arXiv's TeX runs on Unix, and CRLF in a .tex is a needless
    # way to surprise it.
    with open(build / f"{JOBNAME}.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    images = copy_assets(doc, build, keep_bib=not args.no_bib)

    print(f"writing  {build}")
    print(f"  title          : {title}")
    print(f"  {JOBNAME}.tex       : {len(doc.splitlines())} lines")
    print(f"  images         : {len(images)}")
    # Number-only survivors are the tails of ranges and lists -- the second
    # \ref of "Theorems~\ref{a}--\ref{b}", which is a number and nothing else.
    bare = "\n".join(code_of(line) for line in doc.splitlines()).count(r"\ref{")
    print(f"  named refs     : {linked} linked whole (\"Theorem 1\"), {bare} number-only")

    if not args.no_compile:
        pdf = compile_pdf(build, args.keep_work)
        print(f"  preview        : {pdf}")

    if not args.no_tar:
        tarball = pack(build)
        size = tarball.stat().st_size / 1e6
        print(f"  tarball        : {tarball} ({size:.1f} MB)")

    left = todos()
    if left:
        print("\nTODO left in the front/back matter -- edit before posting:")
        for item in left:
            print(f"  {item}")

    print(f"\nUpload the contents of {build} to arXiv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
