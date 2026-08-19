#!/usr/bin/env python3
r"""Build the NewInML @ NeurIPS 2026 edition of a paper.

NewInML asks for 2--8 pages excluding references, double-blind, on the NeurIPS
2026 workshop template: \documentclass{article} with
\usepackage[dblblindworkshop]{neurips_2026}.  Any paper already written for
another single-column venue is a preamble away from that, so this script does
the preamble and nothing else.

    python make_newinml.py                 # the OPT edition, anonymous
    python make_newinml.py --source icomp  # the ICOMP article instead
    python make_newinml.py --mode final    # camera-ready: names shown

It takes ONE shape of input: a standalone LaTeX article, preamble through
\end{document}.  That is what makes it work for more than one source.  The OPT
edition is not standalone, so the `opt` profile runs opt_article/make_opt.py
first and converts what that assembles; the ICOMP article already is one, and
is read where it lies.  A third source is a SOURCES entry, not a code path.

What the conversion does:

  1.  Replaces the class line with the NeurIPS one, and adds \workshoptitle,
      which is what the style file prints in the first-page footer.
  2.  Drops the packages the old venue's style file provided and the new one
      provides again -- times, natbib -- and the old style file itself.
  3.  Adds what the old style file provided and NeurIPS does not.  jmlr.cls
      carries graphicx, amsmath, amssymb, url and hyperref, so a document
      written against it never loads them and would fail here without them.
  4.  Removes the fixes that existed only to survive the old class: the
      algorithm2e name release and the jmlrutils theorem-counter release are
      no-ops once neither package is loaded.
  5.  Repairs the front matter.  \title[short]{full} is jmlr's; article's
      \title takes no optional argument and would typeset the short title and
      leave the real one loose in the document.  \optauthor, \Name, \Email,
      \addr and \acks are jmlr's too and get shims.
  6.  Normalizes the bibliography to plainnat, since the old .bst is gone.
  7.  Copies the style file, the .bib and exactly those images the document
      includes, compiles a scratch copy, and reports the length against the
      8-page limit.

Nothing here writes to the source article.
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

HERE = Path(__file__).resolve().parent          # newinml_article
REPO = HERE.parent                              # repository root
STYLE = HERE / "style"                          # neurips_2026.sty as NeurIPS ships it

JOBNAME = "main"
DEFAULT_BUILD = HERE / "build"
WORKDIR = HERE / ".build-work"
# One per source: two papers do not share an author list.
def authors_file(source):
    return HERE / ("newinml_authors_" + source + ".tex")

# The venue.  2-8 pages excluding references, double-blind, NeurIPS 2026
# workshop template; https://newinml.github.io/NewInML2026NeurIPS/
WORKSHOP = "NewInML"
PAGE_MIN, PAGE_MAX = 2, 8
TRACK = {"anon": "dblblindworkshop", "final": "dblblindworkshop, final"}

# The flattener and the comment-aware line scanner are the arXiv build's, shared
# by every edition of this paper rather than copied into each.
ARXIV_TOOLS = REPO / "aaai_article" / "arxiv" / "make_arxiv.py"


def _load_arxiv_tools():
    if not ARXIV_TOOLS.exists():
        raise SystemExit(
            f"cannot find {ARXIV_TOOLS}.\n"
            "make_newinml.py shares the flattener with the arXiv build; if that "
            "script moved, point ARXIV_TOOLS at it.")
    spec = importlib.util.spec_from_file_location("make_arxiv", ARXIV_TOOLS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ax = _load_arxiv_tools()
code_of = _ax.code_of
flatten = _ax.flatten
match_brace = _ax.match_brace
report_log = _ax.report_log
_GRAPHICS = _ax._GRAPHICS

_COMMENT = re.compile(r"(?<!\\)%")

# Emitted just before the bibliography; main.aux then records the page the main
# text ends on, which is the number the 8-page limit is counted against.
BODY_END_LABEL = "newinml:endofmaintext"


# --------------------------------------------------------------------------
# Sources.
#
# Each entry says where a standalone article comes from and which packages the
# old venue's style file supplied.  `provides` is what that style file loaded
# on the document's behalf: those must be added here, because the document
# never asks for them itself.  `retire` is the reverse -- machinery that exists
# only to survive the old class, and is dead weight under the new one.
# --------------------------------------------------------------------------

SOURCES = {
    "opt": {
        "label": "the OPT 2026 edition",
        "assemble": ["python", str(REPO / "opt_article" / "make_opt.py"),
                     "--no-compile", "--no-zip", "-o", "{out}"],
        "main": "{out}/main.tex",
        "assets": "{out}",
        "styles": ["opt2026", "jmlr", "jmlrutils"],
        "provides": ["graphicx", "amsmath", "amssymb", "url", "hyperref"],
        "retire": ["jmlrutils", "algorithm2e"],
    },
    "icomp": {
        "label": "the ICOMP 2026 article",
        "assemble": None,
        "main": str(Path.home() / "Documents" / "GitHub" / "EDMGrokking"
                    / "icomp_v2" / "report.tex"),
        "assets": str(Path.home() / "Documents" / "GitHub" / "EDMGrokking"
                      / "icomp_v2"),
        "styles": ["icomp2026_conference"],
        "provides": [],          # report.tex loads its own hyperref, amsmath, ...
        "retire": [],
    },
}

# Packages neurips_2026.sty loads for itself.  Loading them again is at best a
# no-op and at worst an option clash.
NEURIPS_PROVIDES = ["times", "natbib"]

# Packages that have to be loaded before hyperref.  float patches \@caption,
# and if it does so after hyperref has patched it too, every float gets its
# destination written twice and the second is dropped: 36 dead links in the
# ICOMP article as it stands, one per table and figure.  Loading float first
# fixes all of them and changes nothing else.
PRECEDE_HYPERREF = ["float", "subfig", "subfigure"]


BANNER = r"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GENERATED FILE -- DO NOT EDIT.
%
%  {workshop} @ NeurIPS 2026 edition of "{title}" (mode: {mode}).
%  Converted by newinml_article/make_newinml.py from {label}.
%
%  Edit the source, not this file, and rerun:
%      python newinml_article/make_newinml.py --source {source}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

CLASS_HEADER = r"""\documentclass{{article}}

% NewInML reviews double-blind, so the track is dblblindworkshop: the style
% file then withholds the authors and numbers the lines for the reviewers.
% \workshoptitle is what it prints in the first-page footer.
\usepackage[{track}]{{neurips_2026}}
\workshoptitle{{{workshop}}}
"""

LINK_FIXES = r"""%%% ---- unique link destinations (inserted by make_newinml.py) --------------
% hyperref names a destination after the raw counter, so two objects that share
% a counter value share a destination and the second one is dropped: every link
% to it lands on the first.  Two habits of these papers cause it.  A theorem
% restated in the appendix under the main text's number (\setcounter{theorem}{3}
% and a second Theorem 4) collides with the original.  An appendix that restarts
% float numbering as A1, A2 collides with Tables 1, 2 of the body, which is 36
% dropped destinations in the ICOMP article as it stands.
%
% \theH<counter> is the name hyperref uses, and it is the printed number that
% has to go into it.  Qualifying by section separates an appendix copy from a
% body one whatever the numbering scheme.  Nothing visible changes.
\makeatletter
\@for\@nimlc:={theorem,lemma,corollary,proposition,remark,assumption,%
hypothesis,definition,table,figure,algorithm}\do{%
  \expandafter\providecommand\csname theH\@nimlc\endcsname{}%
  \expandafter\edef\csname theH\@nimlc\endcsname{%
    \noexpand\thesection.\expandafter\noexpand\csname the\@nimlc\endcsname}}
\makeatother
%%% -------------------------------------------------------------------------
"""

SHIMS = r"""%%% ---- what the old style file used to define (inserted by make_newinml.py)
% \title[short]{full} is jmlr's; article's \title has no optional argument, so
% the conversion rewrites those.  These four are the rest of jmlr's front
% matter vocabulary, kept as shims so an author block written for it still
% sets, and \acks still produces a section.
\providecommand{\Name}[1]{#1}
\providecommand{\Email}[1]{\texttt{#1}}
\providecommand{\addr}{}
\providecommand{\acks}[1]{\section*{Acknowledgments}#1}
%%% -------------------------------------------------------------------------
"""


# --------------------------------------------------------------------------
# Reading the source.
# --------------------------------------------------------------------------

def obtain_source(source: str, scratch: Path) -> tuple[str, Path]:
    """Return the standalone article's text, and the directory its assets sit in."""
    spec = SOURCES[source]
    if spec["assemble"]:
        out = scratch / "assembled"
        cmd = [a.format(out=str(out)) for a in spec["assemble"]]
        run = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        if run.returncode != 0:
            raise SystemExit(f"assembling {spec['label']} failed:\n"
                             + (run.stdout or "") + (run.stderr or ""))
        main = Path(spec["main"].format(out=str(out)))
        assets = Path(spec["assets"].format(out=str(out)))
    else:
        main, assets = Path(spec["main"]), Path(spec["assets"])
    if not main.exists():
        raise SystemExit(f"no such source document: {main}")
    return flatten(main), assets


def find_in_code(text: str, pat: re.Pattern) -> int | None:
    """Offset of the first match outside a comment, or None."""
    offset = 0
    for line in text.splitlines(keepends=True):
        hit = pat.search(code_of(line))
        if hit is not None:
            return offset + hit.start()
        offset += len(line)
    return None


def split_document(text: str) -> tuple[str, str]:
    r"""Preamble and body, split at \begin{document}."""
    for lineno, line in enumerate(text.splitlines()):
        if r"\begin{document}" in code_of(line):
            lines = text.splitlines()
            return "\n".join(lines[:lineno]), "\n".join(lines[lineno + 1:])
    raise SystemExit(r"no \begin{document} in the source article")


# --------------------------------------------------------------------------
# Preamble surgery.
# --------------------------------------------------------------------------

def strip_block(text: str, opening: re.Pattern, closing: str) -> tuple[str, int]:
    """Drop each region from a line matching `opening` to one containing `closing`.

    Matched against the raw line rather than its code: both blocks this removes
    open with a comment banner, and code_of would see nothing there.
    """
    out, gone, dropping = [], 0, False
    for line in text.splitlines():
        if dropping:
            if closing in line:
                dropping = False
            gone += 1
            continue
        if opening.search(line):
            dropping, gone = True, gone + 1
            continue
        out.append(line)
    return "\n".join(out), gone


def convert_preamble(preamble: str, spec: dict, mode: str,
                     workshop: str) -> tuple[str, dict]:
    drop_pkgs = set(spec["styles"]) | set(NEURIPS_PROVIDES)
    stats = {"dropped": [], "retired": 0, "added": list(spec["provides"])}

    # Machinery that only existed to survive the old class.  Both blocks are
    # marked by make_opt.py with a %%% ---- banner and closed by another.
    for tag in spec["retire"]:
        if tag == "algorithm2e":
            preamble, n = strip_block(
                preamble, re.compile(r"algorithmicx under jmlr"), "%%% ------")
            stats["retired"] += n
        if tag == "jmlrutils":
            preamble, n = strip_block(
                preamble, re.compile(r"theorem counters under jmlr"), "%%% ------")
            stats["retired"] += n

    out = []
    for line in preamble.splitlines():
        code = code_of(line)

        if re.search(r"\\documentclass", code):
            continue                                   # the header replaces it

        # \PassOptionsToPackage aimed at a style file that is no longer loaded.
        opt = re.search(r"\\PassOptionsToPackage\s*\{[^}]*\}\s*\{([^}]*)\}", code)
        if opt and opt.group(1) in drop_pkgs:
            stats["dropped"].append(line.strip())
            continue

        pkg = re.search(r"\\usepackage\s*(\[[^\]]*\])?\s*\{([^}]*)\}", code)
        if pkg:
            names = [n.strip() for n in pkg.group(2).split(",")]
            keep = [n for n in names if n not in drop_pkgs]
            if not keep:
                stats["dropped"].append(line.strip())
                continue
            if keep != names:                          # a mixed list: rewrite it
                stats["dropped"].append(
                    line.strip() + "  [kept " + ", ".join(keep) + "]")
                out.append("\\usepackage" + (pkg.group(1) or "")
                           + "{" + ", ".join(keep) + "}")
                continue

        # jmlr's two-argument \title.
        title = re.search(r"\\title\s*\[[^\]]*\]\s*\{", code)
        if title:
            start = code.index("{", title.start())
            out.append("\\title{" + code[start + 1: match_brace(code, start) - 1] + "}")
            continue

        out.append(line)

    return "\n".join(out), stats


_HYPERREF = re.compile(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{[^}]*\bhyperref\b[^}]*\}", re.S)


def hyperref_line(lines: list[str]) -> int | None:
    r"""Index of the line that opens the \usepackage loading hyperref.

    Found on the joined text, because the call spans two lines whenever the
    option list is long, which is the usual case.
    """
    text = "\n".join(lines)
    hit = _HYPERREF.search(text)
    if hit is None:
        return None
    return text[: hit.start()].count("\n")


def reorder_before_hyperref(preamble: str) -> tuple[str, list[str]]:
    """Lift the packages of PRECEDE_HYPERREF above the hyperref call."""
    lines = preamble.splitlines()
    where = hyperref_line(lines)
    if where is None:
        return preamble, []

    moved, kept = [], []
    for i, line in enumerate(lines):
        pkg = re.search(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", code_of(line))
        names = [n.strip() for n in pkg.group(1).split(",")] if pkg else []
        if i > where and any(n in PRECEDE_HYPERREF for n in names):
            moved.append(line.strip())
            continue
        kept.append(line)
    if not moved:
        return preamble, []

    where = hyperref_line(kept)
    note = ["% Lifted above hyperref by make_newinml.py: loaded after it, these",
            "% patch \\@caption a second time and every float loses its link."]
    kept[where:where] = note + moved
    return "\n".join(kept), moved


def package_block(spec: dict) -> str:
    """What the old style file loaded on the document's behalf."""
    if not spec["provides"]:
        return ""
    lines = [r"%%% ---- packages the old style file loaded (inserted by make_newinml.py)",
             "% The source document never asks for these: its class did it. "
             "neurips_2026",
             "% loads only times and natbib, so without them the conversion "
             "would fail on",
             "% the first \\includegraphics or the first \\text{...}."]
    for name in spec["provides"]:
        if name == "hyperref":
            lines.append(r"\usepackage[colorlinks=true, linkcolor=blue, "
                         r"citecolor=blue, urlcolor=magenta]{hyperref}")
        else:
            lines.append("\\usepackage{" + name + "}")
    lines.append(r"%%% -------------------------------------------------------------------------")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Body surgery.
# --------------------------------------------------------------------------

def convert_authors(preamble: str, mode: str, source: str) -> tuple[str, str]:
    r"""Settle the author block, which both sources keep in the preamble.

    Brace-matched, because ICOMP's runs over four lines and dropping the line
    that opens it would leave the address behind.  Under dblblindworkshop the
    style file withholds the authors, so an anonymous build keeps none of it;
    a final build keeps the source's, or newinml_authors.tex if that exists.
    """
    replacement = None
    authors = authors_file(source)
    if mode == "final" and authors.exists():
        replacement = authors.read_text(encoding="utf-8").rstrip()

    seen = 0
    while True:
        hit = find_in_code(preamble, re.compile(r"\\(?:opt)?author\s*\{"))
        if hit is None:
            break
        open_brace = preamble.index("{", hit)
        end = match_brace(preamble, open_brace)
        seen += 1
        if mode == "anon":
            preamble = preamble[:hit] + preamble[end:]
        elif replacement is not None:
            preamble = preamble[:hit] + (replacement if seen == 1 else "") + preamble[end:]
        else:
            inner = preamble[open_brace + 1: end - 1]
            preamble = preamble[:hit] + "\\author{" + inner + "}" + preamble[end:]
            break

    if mode != "final":
        return preamble, "withheld by the style file"
    if replacement is None:
        return preamble, "the source" if seen else "none in the source"
    if seen == 0:
        # An anonymous source has no author command to replace: the OPT edition
        # omits \optauthor entirely under [anon].  Append instead.
        preamble = preamble.rstrip() + "\n\n" + replacement + "\n"
    return preamble, authors.name


def convert_body(body: str, mode: str) -> tuple[str, dict]:
    """Drop the old venue's bibliography style; plainnat is emitted instead."""
    stats = {"bibstyle": 0}
    out = []
    for line in body.splitlines():
        if re.search(r"\\bibliographystyle\s*\{", code_of(line)):
            stats["bibstyle"] += 1
            continue
        out.append(line)
    return "\n".join(out), stats


def normalize_title(preamble: str) -> tuple[str, bool]:
    r"""Take the manual line breaks out of the title.

    ICOMP's title carries three \\ because that style file sets it in small
    caps at about forty characters to the line and hyphenates it otherwise.
    NeurIPS centres a title of its own width, where the same breaks fall in
    arbitrary places.
    """
    hit = find_in_code(preamble, re.compile(r"\\title\s*\{"))
    if hit is None:
        return preamble, False
    open_brace = preamble.index("{", hit)
    end = match_brace(preamble, open_brace)
    inner = preamble[open_brace + 1: end - 1]
    flat = " ".join(inner.replace(r"\\", " ").split())
    if flat == " ".join(inner.split()):
        return preamble, False
    return preamble[:open_brace + 1] + flat + preamble[end - 1:], True


def title_of(preamble: str) -> str:
    m = re.search(r"\\title\s*\{", preamble)
    if not m:
        return "(untitled)"
    start = preamble.index("{", m.start())
    raw = preamble[start + 1: match_brace(preamble, start) - 1]
    return " ".join(raw.replace(r"\\", " ").split())


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def build_document(source: str, mode: str, workshop: str,
                   scratch: Path) -> tuple[str, str, Path, dict]:
    spec = SOURCES[source]
    text, assets = obtain_source(source, scratch)
    preamble, body = split_document(text)
    preamble, pstats = convert_preamble(preamble, spec, mode, workshop)
    preamble, who = convert_authors(preamble, mode, source)
    body, bstats = convert_body(body, mode)

    # The label goes on the line before \bibliography, with no blank line
    # between: a comment does not end a paragraph but a blank line does, and a
    # \label alone in one is pushed to the next page.
    marker = ("%%% The page the main text ends on: the 8-page limit is counted\n"
              "%%% against it, references excluded.\n"
              f"\\label{{{BODY_END_LABEL}}}\n")
    hit = re.search(r"(?m)^\s*\\bibliography\s*\{", body)
    if hit:
        body = body[:hit.start()] + marker + body[hit.start():]
    else:
        raise SystemExit(r"no \bibliography in the source: nothing to measure against")

    full_preamble = package_block(spec) + preamble.strip("\n")
    full_preamble, moved = reorder_before_hyperref(full_preamble)
    full_preamble, unwrapped = normalize_title(full_preamble)
    title = title_of(full_preamble)

    doc = "\n".join([
        BANNER.format(workshop=workshop, title=title, mode=mode,
                      label=spec["label"], source=source),
        CLASS_HEADER.format(track=TRACK[mode], workshop=workshop),
        full_preamble,
        "",
        LINK_FIXES,
        SHIMS,
        r"\bibliographystyle{plainnat}",
        "",
        r"\begin{document}",
        body.strip("\n"),
        "",
    ])
    stats = {**pstats, **bstats, "author": who, "moved": moved,
             "unwrapped": unwrapped}
    return doc, title, assets, stats


def sanity_check(doc: str, mode: str, source: str) -> None:
    code = "\n".join(code_of(line) for line in doc.splitlines())
    problems = []
    if code.count(r"\documentclass") != 1:
        problems.append("expected exactly one \\documentclass")
    if code.count(r"\begin{document}") != 1 or code.count(r"\end{document}") != 1:
        problems.append("document environment is not balanced")
    if code.count(r"\maketitle") != 1:
        problems.append("expected exactly one \\maketitle")
    if code.count(r"\bibliographystyle") != 1:
        problems.append("expected exactly one \\bibliographystyle")
    for banned, why in [
        (r"\title[", "jmlr's two-argument title survived"),
        (r"\optauthor", "jmlr's author command survived"),
        ("opt2026", "the OPT style file must not be loaded"),
        ("icomp2026_conference", "the ICOMP style file must not be loaded"),
        (r"\usepackage{times}", "neurips_2026 loads times itself"),
    ]:
        if banned in code:
            problems.append(f"{banned!r} still present: {why}")
    if mode == "anon" and re.search(r"(?m)^\s*\\author\s*\{", code):
        problems.append(r"\author survived an anonymous build")

    # \cref takes a comma-separated list, so each group has to be split before
    # its members are looked up.
    inside = code.split(r"\begin{document}", 1)[-1]
    defined = set(re.findall(r"\\label\{([^}]+)\}", inside))
    referenced = set()
    for group in (re.findall(r"\\(?:ref|eqref|cref|Cref|autoref)\*?\{([^}]+)\}", inside)
                  + re.findall(r"\\hyperref\[([^\]]+)\]", inside)):
        referenced |= {name.strip() for name in group.split(",") if name.strip()}
    for lost in sorted(referenced - defined):
        problems.append(f"label {lost!r} is referenced and defined nowhere")

    if problems:
        raise SystemExit("converted document failed its checks:\n  - "
                         + "\n  - ".join(problems))


# --------------------------------------------------------------------------
# Assets and compilation.
# --------------------------------------------------------------------------

def copy_assets(doc: str, assets: Path, build: Path) -> tuple[list[str], list[str]]:
    sty = STYLE / "neurips_2026.sty"
    if not sty.exists():
        raise SystemExit(
            f"missing {sty}\nfetch it from "
            "https://media.neurips.cc/Conferences/NeurIPS2026/"
            "Formatting_Instructions_For_NeurIPS_2026.zip")
    shutil.copy2(sty, build / sty.name)

    bibs = []
    for m in re.finditer(r"\\bibliography\s*\{([^}]*)\}", doc):
        for name in (n.strip() for n in m.group(1).split(",")):
            source = assets / (name + ".bib")
            if not source.exists():
                raise SystemExit(f"bibliography not found: {source}")
            shutil.copy2(source, build / source.name)
            bibs.append(source.name)

    wanted = sorted({m.group(1).strip()
                     for line in doc.splitlines()
                     for m in _GRAPHICS.finditer(code_of(line))})
    copied, missing = [], []
    for rel in wanted:
        found = None
        for candidate in (assets / rel, *(assets.glob(rel + ".*"))):
            if candidate.exists() and candidate.is_file():
                found = candidate
                break
        if found is None:
            missing.append(rel)
            continue
        target = build / rel
        if not target.suffix:
            target = target.with_suffix(found.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, target)
        copied.append(rel)
    if missing:
        raise SystemExit("images referenced but not found under "
                         f"{assets}:\n  - " + "\n  - ".join(missing))
    return copied, bibs


def compile_pdf(build: Path, keep_work: bool, workshop: str, source: str) -> Path:
    if shutil.which("latexmk") is None:
        raise SystemExit("latexmk not on PATH; rerun with --no-compile")
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    shutil.copytree(build, WORKDIR)

    run = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         f"{JOBNAME}.tex"], cwd=WORKDIR, capture_output=True, text=True)
    log = WORKDIR / f"{JOBNAME}.log"
    if run.returncode != 0:
        tail = (log.read_text(encoding="utf-8", errors="replace").splitlines()
                if log.exists() else [])
        errors = [ln for ln in tail if ln.startswith("!")] or tail[-30:]
        raise SystemExit("latexmk failed:\n  " + "\n  ".join(errors[:20])
                         + f"\n\nfull log: {log}")

    bbl = WORKDIR / f"{JOBNAME}.bbl"
    if bbl.exists():
        shutil.copy2(bbl, build / bbl.name)
    pdf = HERE / f"{source}_newinml.pdf"
    shutil.copy2(WORKDIR / f"{JOBNAME}.pdf", pdf)
    report_log(log)
    report_length(WORKDIR / f"{JOBNAME}.aux")
    if not keep_work:
        shutil.rmtree(WORKDIR)
    return pdf


def report_length(aux: Path) -> None:
    if not aux.exists():
        return
    text = aux.read_text(encoding="utf-8", errors="replace")
    hit = re.search(r"\\newlabel\{" + re.escape(BODY_END_LABEL)
                    + r"\}\{\{[^{}]*\}\{(\d+)\}", text)
    if hit is None:
        print(f"  main text      : ? pages (NewInML allows {PAGE_MIN}--{PAGE_MAX})")
        return
    pages = int(hit.group(1))
    if pages > PAGE_MAX:
        note = f"   <-- {pages - PAGE_MAX} over"
    elif pages < PAGE_MIN:
        note = f"   <-- {PAGE_MIN - pages} short"
    else:
        note = ""
    print(f"  main text      : {pages} pages, limit {PAGE_MIN}--{PAGE_MAX}{note}")


def pack_zip(build: Path, source: str) -> Path:
    archive = HERE / f"{source}_newinml.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(build.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(build)).replace("\\", "/"))
    return archive


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=sorted(SOURCES), default="opt",
                    help="which article to convert (default: opt)")
    ap.add_argument("--mode", choices=["anon", "final"], default="anon",
                    help="anon: double-blind submission (default). final: "
                         "camera-ready, author names shown.")
    ap.add_argument("--workshop", default=WORKSHOP,
                    help=f"name for the footer line (default: {WORKSHOP})")
    ap.add_argument("-o", "--build", type=Path, default=None,
                    help="output directory (default: build/<source>)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    build = (args.build or DEFAULT_BUILD / args.source).resolve()
    scratch = HERE / ".source-work"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    try:
        print(f"reading  {SOURCES[args.source]['label']}")
        doc, title, assets, stats = build_document(
            args.source, args.mode, args.workshop, scratch)
        sanity_check(doc, args.mode, args.source)

        if build.exists():
            shutil.rmtree(build)
        build.mkdir(parents=True)
        with open(build / f"{JOBNAME}.tex", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(doc)
        images, bibs = copy_assets(doc, assets, build)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"writing  {build}")
    print(f"  title          : {title}")
    print(f"  workshop       : {args.workshop} @ NeurIPS 2026 ({TRACK[args.mode]})")
    print(f"  {JOBNAME}.tex       : {len(doc.splitlines())} lines")
    print(f"  images         : {len(images)}   bibliography: {', '.join(bibs)}")
    print(f"  preamble       : {len(stats['dropped'])} lines dropped, "
          f"{stats['retired']} retired, {len(stats['added'])} packages added")
    print(f"  authors        : {stats['author']}")
    if stats["moved"]:
        print(f"  reordered      : {len(stats['moved'])} package(s) lifted above hyperref")

    if not args.no_compile:
        pdf = compile_pdf(build, args.keep_work, args.workshop, args.source)
        print(f"  preview        : {pdf}")
    if not args.no_zip:
        archive = pack_zip(build, args.source)
        print(f"  zip            : {archive} "
              f"({archive.stat().st_size / 1e6:.1f} MB)")

    if stats["dropped"]:
        print("\nDropped from the preamble, the old venue's style file having "
              "provided them:")
        for line in stats["dropped"]:
            print(f"  {line}")

    todos = []
    if args.mode == "final":
        authors = authors_file(args.source)
        if authors.exists():
            todos = [f"{authors.name}:{n}: {l.strip()}"
                     for n, l in enumerate(
                         authors.read_text(encoding="utf-8").splitlines(), 1)
                     if "TODO" in code_of(l)]
    if todos:
        print("\nTODO left in the author block -- edit before submitting:")
        for item in todos:
            print(f"  {item}")

    print(f"\nSubmit {HERE / (args.source + '_newinml.pdf')} to OpenReview.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
