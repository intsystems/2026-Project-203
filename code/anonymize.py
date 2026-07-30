"""Check this tree for identifying material, and build the anonymous supplement.

    python3 -m anonymize --check                  # scan; exits 1 on a hit
    python3 -m anonymize --build ../supplement.zip # clean bundle for submission

Run both from ``code/``. ``--check`` is also wired into ``tests/test_code.py``, so a
new author name or absolute path in a docstring fails the test suite rather than
reaching a reviewer.

Why a script rather than a checklist
------------------------------------
Double-blind leaks are almost never in the places a checklist looks. They are in a
notebook's *output* cell holding ``/home/<user>/project``, in a generated MANIFEST
quoting an absolute source path, in a filename, or in a comment crediting a
colleague by first name. All four were present here. A regex sweep finds them, and
running it in CI is what keeps them found.

What is deliberately NOT flagged
--------------------------------
* **Upstream URLs and third-party author names.** ``github.com/KellerJordan/modded-nanogpt``
  is a citation, not a self-identification, and removing it would misattribute the
  code this port is built on. Only *this project's* repository is flagged.
* **Hardware and driver strings** (``NVIDIA H100``, ``Driver Version: ...``) in the
  run logs. AAAI's reproducibility checklist asks for exactly this, and it does not
  identify anyone.
* **Dates.** They constrain the submission window, not the authors.

The bundle
----------
``--build`` writes a zip containing only what a reviewer needs to run the code, with
every generated artifact, dataset, checkpoint and cache excluded (see ``EXCLUDE``),
and with **every notebook output stripped** -- the working copies are untouched. It
also writes a ``MANIFEST.md`` into the zip recording what was excluded, so the
omissions are visible rather than silent.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parent          # code/

__all__ = ["Finding", "scan_tree", "build_bundle", "PATTERNS", "EXCLUDE"]


# --------------------------------------------------------------------------
# What counts as identifying
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str
    why: str
    flags: int = re.IGNORECASE


#: Names, handles and institutions belonging to *this* project. Add to one of these
#: two lists rather than to the generic rules below -- a specific string is cheap to
#: check and never fires on an upstream citation.
#:
#: Distinctive enough to match ANYWHERE, with no boundary conditions at all.
#: None of these is a substring of an ordinary English word, so gluing cannot
#: create a false positive -- but it can very easily hide a real leak
#: (``AlexKravatsky``, ``kravatsky2026``, ``by_legeartis``), which is why these
#: are matched bare.
PROJECT_IDENTIFIERS: Sequence[str] = (
    "smirnova", "kravatsk", "legeartis", "stsix",
    "miriai", "intsystems", "2026-Project-203",
)

#: Short or ambiguous enough that they need a non-letter boundary, or they would
#: fire on ordinary words (``alex`` inside ``Alexandria``, ``lesha`` inside a
#: transliteration). Bare given names live here.
WORD_IDENTIFIERS: Sequence[str] = (
    "alex", "alexey", "alesha", "lesha",
)

PATTERNS: Sequence[Rule] = (
    Rule("project-identifier",
         "(" + "|".join(PROJECT_IDENTIFIERS) + ")",
         "an author name, handle, institution or repository belonging to this project"),
    # NOT \b...\b. `\b` counts `_` as a word character, so a name inside a
    # *filename* (`<name>_nanogpt.py`) never matches -- which is exactly the shape a
    # leaked name takes. The lookarounds exclude letters only, so underscores,
    # digits, dots and hyphens cannot shield a name either (`<name>1`, `<name>.tex`).
    Rule("project-identifier",
         r"(?<![A-Za-z])(" + "|".join(WORD_IDENTIFIERS) + r")(?![A-Za-z])",
         "an author name belonging to this project"),
    Rule("email",
         r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
         "an email address"),
    Rule("home-path",
         r"(/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9._-]+)",
         "an absolute path containing a username"),
    # Any owner, not just the lab account: a personal fork
    # (``github.com/<handle>/SignMuon``) de-anonymizes just as effectively.
    Rule("project-repo",
         r"github\.com/(intsystems[A-Za-z0-9._/-]*"
         r"|[A-Za-z0-9._-]+/[A-Za-z0-9._-]*(signmuon|2026-project-203))",
         "a link to this project's own repository"),
    Rule("orcid",
         r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b",
         "an ORCID identifier"),
)

#: Substrings that make a hit a false positive. Kept narrow and explicit so that
#: silencing something is a visible decision.
ALLOW = (
    "KellerJordan",           # upstream modded-nanogpt
    "NoahAmsel",              # upstream PolarExpress
    "varunneal", "ragulpr",   # upstream contributors credited in the port
    "@torch", "@dataclass", "@property", "@staticmethod", "@contextmanager",
    "@classmethod", "@abstractmethod",
    # Invented placeholders in `common/paths.py` and the test that pins it. That
    # code exists to strip real home paths out of the export bundles, so it has to
    # quote the shape of one; these three are not anybody's username.
    "/home/someone/", "/home/u/", "/home/user/",
)

#: Files skipped by the scan itself. ``anonymize.py`` has to spell out the names it
#: searches for, so scanning it would always fail; it is also excluded from the
#: bundle, since the identifier list is the one place the names must not ship.
SKIP_SCAN_PATHS = ("anonymize.py",)

#: Files never scanned: binary, generated, or third-party verbatim.
SKIP_SCAN = (
    "*.png", "*.pdf", "*.jpg", "*.jpeg", "*.gif", "*.pt", "*.pth", "*.bin",
    "*.npy", "*.npz", "*.zip", "*.tar", "*.tar.gz", "*.pyc", "*.so", "*.ttf",
)

#: Paths excluded from BOTH the scan and the bundle: caches, datasets, run output.
EXCLUDE = (
    "**/__pycache__/**", "**/.pytest_cache/**", "**/.ipynb_checkpoints/**",
    "**/.git/**", "**/.venv/**", "**/venv/**",
    "results/**", "results_old/**", "saves/**", "saves_*/**", "output_grid/**",
    # Result bundles unpacked *outside* `results/`. The trailing `*` is
    # load-bearing: a bundle parked as `article_export.stale_2026-07-29` is the same
    # run output as `article_export`, and it carries the absolute paths of the box
    # that wrote it.
    "**/article_export*/**", "**/article_export*.tar.gz",
    "**/federated_export*/**", "**/federated_export*.zip",
    "**/synthetic_results*.zip",
    "data/**", "data_federated/**", "**/fineweb10B/**",
    "*.pt", "*.pth", "*.bin", "*.npz",
)

#: Re-included after ``EXCLUDE``, and checked before it.
#:
#: ``results/**`` above drops run output because every bit of it is regenerated by
#: a command in ``REPRODUCE.md`` -- except this one. The eight nanoGPT logs cost an
#: 8xH100 node and cannot be re-created from this bundle, and they *are* the
#: evidence behind the language-modelling table: ``parse_logs.py`` and
#: ``make_tables.py`` turn them into it. They embed the training script and the
#: optimizer definitions verbatim, and the scan covers them (they are not excluded
#: from it either), so they are as anonymous as the source they quote.
INCLUDE_ANYWAY = (
    "results/nanogpt/**",
)

#: Excluded from the bundle only -- present and useful locally, but not something a
#: reviewer needs.
EXCLUDE_FROM_BUNDLE = (
    "*.log",
)


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    why: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.text.strip()[:110]}"


def _matches(posix: str, pat: str) -> bool:
    if fnmatch(posix, pat) or fnmatch(posix, pat.lstrip("*/")):
        return True
    # "results/**" should also match the directory itself and one-level files
    head = pat.split("/**")[0]
    return pat.endswith("/**") and (posix == head or posix.startswith(head + "/"))


def _excluded(rel: str, extra: Sequence[str] = ()) -> bool:
    posix = rel.replace("\\", "/")
    # Checked FIRST: a re-inclusion has to beat the broad `results/**` rule it is
    # carved out of, and `extra` (the bundle-only rules) still applies afterwards
    # only through its own loop below -- so `*.log` inside a re-included tree would
    # still ship, which is intended: there are none.
    if any(_matches(posix, pat) for pat in INCLUDE_ANYWAY):
        return False
    for pat in tuple(EXCLUDE) + tuple(extra):
        if fnmatch(posix, pat) or fnmatch(posix, pat.lstrip("*/")):
            return True
        # "results/**" should also match the directory itself and one-level files
        head = pat.split("/**")[0]
        if pat.endswith("/**") and (posix == head or posix.startswith(head + "/")):
            return True
    return False


def _scannable(path: Path) -> bool:
    return not any(fnmatch(path.name, pat) for pat in SKIP_SCAN)


def _iter_files(root: Path, extra_exclude: Sequence[str] = ()) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if _excluded(rel, extra_exclude):
            continue
        yield p


#: Inline opt-out. A line carrying this marker is skipped, which is how the
#: scanner's own test fixtures live in the tree without failing it. Deliberately a
#: visible per-line annotation rather than a file-level exclusion: exempting a line
#: should be a decision a reader can see, and the rest of the file stays scanned.
PRAGMA = "anonymize: allow"


def _allowed_spans(line: str) -> List[tuple]:
    """Character ranges of the ``ALLOW`` substrings occurring in ``line``."""
    spans = []
    for a in ALLOW:
        start = 0
        while True:
            k = line.find(a, start)
            if k < 0:
                break
            spans.append((k, k + len(a)))
            start = k + 1
    return spans


def scan_text(rel: str, text: str) -> List[Finding]:
    """Findings in ``text``, one per (line, rule).

    ``ALLOW`` is applied **per match**, not per line: a hit is silenced only when
    it falls *inside* an allowed substring. Silencing the whole line would exempt
    everything else on it -- and "ported from KellerJordan's repo by <author>
    <email>" is exactly the shape a real leak takes here, so a line-level allow
    hides precisely the case the scan exists to catch.
    """
    out: List[Finding] = []
    for i, line in enumerate(text.splitlines(), 1):
        if PRAGMA in line:
            continue
        allowed = _allowed_spans(line)
        for rule in PATTERNS:
            for m in re.finditer(rule.pattern, line, rule.flags):
                if any(lo <= m.start() and m.end() <= hi for lo, hi in allowed):
                    continue
                out.append(Finding(rel, i, rule.name, rule.why, line))
                break
    return out


def scan_tree(root: Path = ROOT, extra_exclude: Sequence[str] = ()
              ) -> tuple[List[Finding], List[str]]:
    """``(findings, notebooks_with_outputs)`` for the tree under ``root``.

    Notebooks are scanned **as the bundle will ship them**, i.e. with outputs
    stripped. That is the honest object to check: an output cell holding
    ``/home/<user>/project`` is invisible to a reader skimming the source and is
    shipped verbatim inside a raw ``.ipynb``, but ``--build`` removes it. Scanning
    the raw file instead would make the check fail forever on a leak that cannot
    reach a reviewer -- and a check that always fails is a check nobody runs.

    The notebooks that still carry outputs are returned separately so the caller
    can say so out loud rather than relying on the stripping being remembered.
    """
    findings: List[Finding] = []
    with_outputs: List[str] = []
    for path in _iter_files(root, extra_exclude):
        rel = str(path.relative_to(root))
        if not _scannable(path) or rel.replace("\\", "/") in SKIP_SCAN_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix == ".ipynb":
            cleaned = strip_notebook(text)
            if cleaned != text:
                with_outputs.append(rel.replace("\\", "/"))
            text = cleaned
        findings.extend(scan_text(rel, text))
    return findings, with_outputs


# --------------------------------------------------------------------------
# Bundling
# --------------------------------------------------------------------------


def strip_notebook(text: str) -> str:
    """Clear every output and execution count, leaving the source intact.

    Notebook outputs are the single most common double-blind leak in supplementary
    material: they carry absolute paths, usernames, hostnames and wall-clock
    timestamps that no one reads before zipping.
    """
    try:
        nb = json.loads(text)
    except json.JSONDecodeError:
        return text
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def build_bundle(out_path: Path, root: Path = ROOT, top: str = "code") -> dict:
    """Write the anonymous supplement zip; return a summary dict."""
    included: List[str] = []
    stripped: List[str] = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in _iter_files(root, EXCLUDE_FROM_BUNDLE):
            rel = path.relative_to(root)
            if rel.as_posix() in SKIP_SCAN_PATHS:
                continue          # the identifier list must not ship
            arc = f"{top}/{rel.as_posix()}"
            if path.suffix == ".ipynb":
                text = path.read_text(encoding="utf-8", errors="replace")
                cleaned = strip_notebook(text)
                if cleaned != text:
                    stripped.append(rel.as_posix())
                z.writestr(arc, cleaned)
            else:
                z.write(path, arc)
            included.append(rel.as_posix())

        z.writestr(f"{top}/MANIFEST.md", _manifest(included, stripped))

    return {"zip": str(out_path), "files": len(included),
            "notebooks_stripped": stripped,
            "bytes": out_path.stat().st_size}


def _manifest(included: Sequence[str], stripped: Sequence[str]) -> str:
    lines = [
        "# Supplementary material — manifest",
        "",
        "Built by `python3 -m anonymize --build`. Everything below is stated so that",
        "the omissions are visible rather than silent.",
        "",
        f"* **{len(included)} files** included.",
        "* **Excluded**: caches (`__pycache__`, `.pytest_cache`), datasets",
        "  (`data/`, `data_federated/`, FineWeb shards), run output (`results/`,",
        "  `results_old/`, `article_export/`), model checkpoints (`*.pt`) and stray",
        "  `*.log` files.",
        "  None of these is needed to run anything; every one is regenerated by the",
        "  commands in `REPRODUCE.md`. `results/nanogpt/` IS included -- those eight",
        "  logs are the evidence behind the language-modelling table and cost an",
        "  8xH100 node, so they are the one artefact this bundle cannot regenerate.",
        f"* **Notebook outputs stripped**: {len(stripped)} notebook(s)"
        + (" — " + ", ".join(f"`{s}`" for s in stripped) if stripped else "") + ".",
        "  Source cells are untouched. Outputs are removed because they carry",
        "  absolute paths and hostnames from the machine that ran them.",
        "",
        "## Where to start",
        "",
        "`README.md` is the map; `REPRODUCE.md` has the exact command for every table",
        "and figure in the paper. `python3 -m tests.test_code` runs the CPU test suite",
        "in about a minute and needs no GPU and no downloads.",
        "",
        "## Anonymity",
        "",
        "`python3 -m anonymize --check` re-runs the scan that produced this bundle:",
        "author names, emails, absolute paths containing usernames, the project's own",
        "repository URL, and ORCIDs. Upstream citations (e.g. the modded-nanogpt",
        "repository this port builds on) are deliberately kept — removing them would",
        "misattribute the work.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def scrub_results(root: Path, write: bool = False) -> List[tuple]:
    """Rewrite machine-specific paths out of an existing ``results/`` tree.

    The writers scrub as they go, but a tree predating that carries the paths of
    the box that made it -- and `results/` now ships with the code, so those files
    are read rather than merely stored. Deleting them is the other option and a
    worse one: they are results.

    Text files only, in place, using the same `common.paths.scrub_text` the writers
    use, so this cannot disagree with them. Returns ``(relative path, hits)`` per
    file changed; ``write=False`` reports without touching anything.
    """
    from common.paths import scrub_text

    results = root / "results"
    changed = []
    for path in sorted(results.rglob("*")) if results.is_dir() else []:
        if not path.is_file() or not _scannable(path):
            continue
        try:
            before = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        after = scrub_text(before)
        if after == before:
            continue
        rel = path.relative_to(root).as_posix()
        changed.append((rel, sum(1 for a, b in zip(before.splitlines(),
                                                   after.splitlines()) if a != b)))
        if write:
            path.write_text(after, encoding="utf-8")
    return changed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="Scan for identifying material; exit 1 if anything is found")
    p.add_argument("--scrub-results", action="store_true",
                   help="Rewrite machine-specific paths out of results/ in place. "
                        "Reports what it would change; add --write to apply")
    p.add_argument("--write", action="store_true",
                   help="With --scrub-results, actually rewrite the files")
    p.add_argument("--build", type=str, metavar="ZIP", default=None,
                   help="Write the anonymous supplement to this path")
    p.add_argument("--root", type=str, default=str(ROOT))
    p.add_argument("--list-rules", action="store_true")
    args = p.parse_args()

    if args.list_rules:
        for rule in PATTERNS:
            print(f"{rule.name:<20} {rule.why}\n{'':<20} /{rule.pattern}/")
        return 0

    if not args.check and not args.build and not args.scrub_results:
        p.error("nothing to do: pass --check, --scrub-results and/or --build ZIP")

    root = Path(args.root).resolve()
    status = 0

    if args.scrub_results:
        changed = scrub_results(root, write=args.write)
        if not changed:
            print(f"results/ under {root} already carries no machine-specific path.")
        else:
            verb = "Rewrote" if args.write else "Would rewrite"
            print(f"{verb} {len(changed)} file(s):")
            for rel, hits in changed:
                print(f"    {rel}  ({hits} line(s))")
            if not args.write:
                print("\nNothing was written. Re-run with --write to apply.")

    if args.check:
        findings, with_outputs = scan_tree(root)
        if with_outputs:
            print(f"note: {len(with_outputs)} notebook(s) still carry outputs and were "
                  f"scanned as --build will ship them (outputs stripped):")
            for nb in with_outputs:
                print(f"    {nb}")
            print()
        if findings:
            print(f"FAIL: {len(findings)} identifying item(s) found in {root}\n")
            by_rule: dict = {}
            for f in findings:
                by_rule.setdefault(f.rule, []).append(f)
            for rule, items in by_rule.items():
                print(f"  [{rule}] {items[0].why}")
                for f in items:
                    print(f"    {f}")
                print()
            print("Fix these before building the supplement. If a hit is an upstream\n"
                  "citation rather than a self-identification, add it to ALLOW in\n"
                  "anonymize.py with a one-line reason.")
            status = 1
        else:
            print(f"OK: no identifying material found in {root}")

    if args.build:
        if status:
            print("\nRefusing to build a bundle while --check fails.")
            return status
        info = build_bundle(Path(args.build), root)
        print(f"\nWrote {info['zip']} "
              f"({info['files']} files, {info['bytes'] / 1e6:.1f} MB)")
        if info["notebooks_stripped"]:
            print(f"Stripped outputs from {len(info['notebooks_stripped'])} notebook(s): "
                  + ", ".join(info["notebooks_stripped"]))
        print("Re-check the zip itself before submitting:\n"
              "  python3 -c \"import zipfile,sys; "
              "print(len(zipfile.ZipFile(sys.argv[1]).namelist()))\" "
              + str(args.build))

    return status


if __name__ == "__main__":
    sys.exit(main())
