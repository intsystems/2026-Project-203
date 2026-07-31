"""Check this tree for identifying material, and build the anonymous supplement.

    python3 -m anonymize --check                  # scan; exits 1 on a hit
    python3 -m anonymize --build ../supplement.zip # clean bundle for submission
    python3 -m anonymize --build-dir ../anonymous_code  # the same bundle, unpacked

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

__all__ = ["Finding", "scan_tree", "build_bundle", "build_dir", "PATTERNS", "EXCLUDE"]


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

#: The anonymity tooling: withheld from the bundle as one unit, because each part
#: gives away what the others hide.
#:
#: * ``anonymize.py`` is the list of names itself.
#: * ``tests/test_anonymize.py`` quotes its rules and would import it.
#: * ``ANONYMIZATION.md`` is the write-up, and a write-up of what leaked leaks:
#:   it records that a docstring credited a colleague by first name, that a
#:   filename carried one, and that Russian-language comments were translated --
#:   the last of which narrows the author pool exactly as it warns against.
#:
#: A reviewer needs none of it. What they do need -- that the code was scanned,
#: for what, and that upstream citations were kept deliberately -- is in the
#: bundle's own ``MANIFEST.md``, written by `_manifest` below.
ANONYMITY_TOOLING = (
    "anonymize.py",
    "tests/test_anonymize.py",
    "ANONYMIZATION.md",
)

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
#: ``results/**`` above drops run output, which is right for the *raw* trees: they
#: are hundreds of megabytes, they accumulate runs from superseded sessions, and
#: `REPRODUCE.md` has a command that regenerates each. What ships instead is the
#: curated end of each arm.
#:
#: * **The three export archives.** `REPRODUCE.md` §§3-5 all have the same shape --
#:   compute on the GPU box, bring back one archive, plot anywhere -- and these are
#:   the archives. Each is written by the exporter that also decides what belongs in
#:   it (`federated.export_article` drops runs made under a superseded sign
#:   convention, and records every exclusion with its reason), so an archive is the
#:   curated run set, not the tree it came from. Without them a reviewer cannot
#:   redraw a single CIFAR, federated or synthetic figure without a GPU-night.
#: * **The nanoGPT logs.** They cost an 8xH100 node and cannot be re-created from
#:   this bundle at all. `parse_logs.py` and `make_tables.py` turn them into the
#:   language-modelling table, and they embed the training script and the optimizer
#:   definitions verbatim.
#:
#: All of it is scanned: `scan_tree` reads the text members of an included archive,
#: so shipping a zip does not smuggle past the gate what a loose file could not.
INCLUDE_ANYWAY = (
    "results/nanogpt/**",
    "results/synthetic_results.zip",
    "results/article_export.tar.gz",
    "results/federated_export_results.zip",
)

#: Excluded from the bundle only -- present and useful locally, but not something a
#: reviewer needs. ``*.stackdump`` is a Cygwin/MSYS crash dump: `.gitignore` already
#: drops them, so one sits in the tree unnoticed and would otherwise ship.
EXCLUDE_FROM_BUNDLE = (
    "*.log", "*.stackdump",
) + ANONYMITY_TOOLING


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


#: Archives whose *members* are scanned when the archive itself is shipped. An
#: export bundle is a zip of JSON, CSV and Markdown written on the machine that ran
#: the study, so it carries exactly the material this tool exists to catch --
#: invisibly, since `_scannable` (rightly) refuses to read a zip as text. Before
#: these shipped, excluding them was defence enough; now the gate has to open them.
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")


def _is_archive(rel: str) -> bool:
    return rel.replace("\\", "/").lower().endswith(ARCHIVE_SUFFIXES)


def _archive_members(path: Path) -> Iterable[tuple]:
    """``(member name, text)`` for every text member, or nothing if unreadable.

    Members that are binary, or too large to be prose, are skipped by the same
    ``SKIP_SCAN`` suffixes used for loose files. A corrupt or unsupported archive
    yields nothing rather than raising: the file still ships, and a scan that dies
    on one archive would stop covering the rest.
    """
    import tarfile

    try:
        if path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if info.is_dir() or not _scannable(Path(info.filename)):
                        continue
                    yield info.filename, z.read(info).decode("utf-8", "replace")
        else:
            with tarfile.open(path) as t:
                for member in t.getmembers():
                    if not member.isfile() or not _scannable(Path(member.name)):
                        continue
                    handle = t.extractfile(member)
                    if handle is not None:
                        yield member.name, handle.read().decode("utf-8", "replace")
    except Exception:                                    # noqa: BLE001
        return


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
        if rel.replace("\\", "/") in SKIP_SCAN_PATHS:
            continue
        if _is_archive(rel):
            # Reached only for an archive that survived EXCLUDE, i.e. one that ships.
            for member, text in _archive_members(path):
                findings.extend(scan_text(f"{rel}:{member}", text))
            continue
        if not _scannable(path):
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


#: What replaces a git revision in the bundle. Not an empty string: a reader
#: should see that the field was removed, not that it was never recorded.
REDACTED_REVISION = "<redacted for review>"

#: The exporters stamp the commit they ran at into every `metrics.json`,
#: `configs.json`, `environment.json` and `MANIFEST.json` -- provenance, and the
#: right thing for a run tree to record. In an anonymous bundle it is the one
#: piece of identifying material that reproducibility does not pay for: a reviewer
#: has the code, not the repository, so a revision resolves to nothing for them,
#: while a full SHA pasted into a commit search resolves to the repository, its
#: name and its authors. So it goes, and `MANIFEST.md` says that it went.
_REVISION_JSON = re.compile(
    r'("(?:git_)?(?:commit|sha|revision)"\s*:\s*")([0-9a-f]{7,40})(")', re.IGNORECASE)
_REVISION_PROSE = re.compile(
    r"(?i)\b(commit\s+`)([0-9a-f]{7,40})(`)")


def redact_revisions(text: str) -> str:
    """Replace recorded git revisions, leaving everything else byte-identical.

    Keyed on the field name rather than on "looks like hex", so a run identifier
    (``EF21-MuonSign_lr0.06_e6770317.txt``) or a driver hash is untouched -- those
    name a run, not a repository.
    """
    text = _REVISION_JSON.sub(rf"\g<1>{REDACTED_REVISION}\g<3>", text)
    return _REVISION_PROSE.sub(rf"\g<1>{REDACTED_REVISION}\g<3>", text)


def _redacted_bytes(data: bytes, name: str) -> bytes:
    """``data`` with revisions redacted, or ``data`` itself.

    Returns the original object unless a redaction actually fired, so a file the
    rule does not touch ships byte-for-byte -- no decode/encode round trip to
    normalize a line ending or mangle a byte that is not valid UTF-8.
    """
    if not _scannable(Path(name)):
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    redacted = redact_revisions(text)
    return data if redacted == text else redacted.encode("utf-8")


def _redacted_archive(path: Path) -> bytes:
    """The archive rebuilt with `redact_revisions` applied to its text members.

    Rebuilt rather than patched in place: a zip entry's length changes, and the
    three archives are small enough that reading and rewriting them costs less
    than the machinery for editing one. Binary members are copied verbatim.
    """
    import tarfile

    raw = path.read_bytes()
    out = io.BytesIO()
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as src, \
                zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                dst.writestr(info, _redacted_bytes(src.read(info), info.filename))
    else:
        with tarfile.open(fileobj=io.BytesIO(raw)) as src, \
                tarfile.open(fileobj=out, mode="w:gz") as dst:
            for member in src.getmembers():
                if not member.isfile():
                    dst.addfile(member)
                    continue
                handle = src.extractfile(member)
                data = _redacted_bytes(b"" if handle is None else handle.read(),
                                       member.name)
                member.size = len(data)
                dst.addfile(member, io.BytesIO(data))
    return out.getvalue()


def _bundle_entries(root: Path) -> Iterable[tuple]:
    """``(relative path, bytes to write, whether a notebook was stripped)``.

    The one definition of *what the bundle contains*, so that the zip and the
    directory below cannot come to disagree about a file, an exclusion or a
    stripped output.
    """
    for path in _iter_files(root, EXCLUDE_FROM_BUNDLE):
        rel = path.relative_to(root)
        if path.suffix == ".ipynb":
            text = path.read_text(encoding="utf-8", errors="replace")
            cleaned = strip_notebook(text)
            yield rel, cleaned.encode("utf-8"), cleaned != text
        elif _is_archive(rel.as_posix()):
            yield rel, _redacted_archive(path), False
        else:
            yield rel, _redacted_bytes(path.read_bytes(), rel.as_posix()), False


def build_bundle(out_path: Path, root: Path = ROOT, top: str = "code") -> dict:
    """Write the anonymous supplement zip; return a summary dict."""
    included: List[str] = []
    stripped: List[str] = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, payload, was_stripped in _bundle_entries(root):
            z.writestr(f"{top}/{rel.as_posix()}", payload)
            included.append(rel.as_posix())
            if was_stripped:
                stripped.append(rel.as_posix())

        z.writestr(f"{top}/MANIFEST.md", _manifest(included, stripped))

    return {"zip": str(out_path), "files": len(included),
            "notebooks_stripped": stripped,
            "bytes": out_path.stat().st_size}


def build_dir(out_dir: Path, root: Path = ROOT, top: str = "code") -> dict:
    """Write the same bundle as `build_bundle`, unpacked, into ``out_dir``.

    Same file set, same notebook stripping, same ``MANIFEST.md`` -- the zip and the
    directory are two spellings of one bundle, so a reviewer's copy cannot depend on
    which one was built. Useful when the archive is made later (OpenReview wants a
    zip, but one that has been looked at first), or when the contents want checking
    with ordinary tools rather than through `zipfile`.

    Refuses a destination that already exists and is not empty. Emptying it here
    would mean a script deleting a directory tree named on the command line, and a
    mistyped path is not worth the convenience; remove it yourself and re-run.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} exists and is not empty; remove it and re-run so that "
            "nothing from an earlier build survives into this one")

    included: List[str] = []
    stripped: List[str] = []
    base = out_dir / top

    for rel, payload, was_stripped in _bundle_entries(root):
        dest = base / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        included.append(rel.as_posix())
        if was_stripped:
            stripped.append(rel.as_posix())

    (base / "MANIFEST.md").write_text(_manifest(included, stripped), encoding="utf-8")

    return {"dir": str(out_dir), "files": len(included),
            "notebooks_stripped": stripped,
            "bytes": sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())}


def _manifest(included: Sequence[str], stripped: Sequence[str]) -> str:
    lines = [
        "# Supplementary material — manifest",
        "",
        "Built by `python3 -m anonymize --build` (or `--build-dir`, which writes the",
        "same file set unpacked). Everything below is stated so that the omissions are",
        "visible rather than silent.",
        "",
        f"* **{len(included)} files** included.",
        "* **Excluded**: caches (`__pycache__`, `.pytest_cache`), datasets",
        "  (`data/`, `data_federated/`, FineWeb shards), the raw per-run trees",
        "  (`results/federated/`, `results/synthetic/`, `results_old/`) and the",
        "  unpacked export bundles, model checkpoints (`*.pt`), and stray `*.log`",
        "  and `*.stackdump` files.",
        "* **Included from `results/`**: the three export archives",
        "  (`synthetic_results.zip`, `article_export.tar.gz`,",
        "  `federated_export_results.zip`) and `nanogpt/`. The archives are what",
        "  `REPRODUCE.md` §§3-5 plot from, so every CIFAR, federated and synthetic",
        "  table and figure redraws from this bundle without a GPU. Each is the run",
        "  set its exporter chose: the federated one excludes runs made under a",
        "  superseded sign convention and lists each exclusion in its",
        "  `MANIFEST.json`. The nanoGPT logs are included because they cost an",
        "  8xH100 node and are the one artefact nothing here can regenerate.",
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
        "This tree was scanned before it was packaged, for author names, emails,",
        "absolute paths containing usernames, the project's own repository URL, and",
        "ORCIDs — inside the export archives as well as in the source. Upstream",
        "citations (e.g. the modded-nanogpt repository this port builds on) are",
        "deliberately kept: removing them would misattribute the work.",
        "",
        "The scanner, its tests and its write-up are **not** in this bundle. Between",
        "them they hold the list of names to look for and a record of what once",
        "leaked, so shipping them would undo the scan. Nothing else references them,",
        "and the suite above has no import that fails to resolve as a result.",
        "",
        "**Git revisions are redacted.** The exporters stamp the commit each run was",
        "made at into `metrics.json`, `configs.json`, `environment.json` and each",
        "`MANIFEST.json`; in the archives above every such field now reads",
        f"`{REDACTED_REVISION}`. A revision resolves to nothing without the",
        "repository, which is not part of this bundle, so it bought reproducibility",
        "nothing here and identified the authors to anyone who searched for it.",
        "Everything else in the archives is exactly what the exporter wrote.",
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
    p.add_argument("--build-dir", type=str, metavar="DIR", default=None,
                   help="Write the same bundle unpacked into this directory, to be "
                        "inspected and archived afterwards. Refuses a non-empty DIR")
    p.add_argument("--root", type=str, default=str(ROOT))
    p.add_argument("--list-rules", action="store_true")
    args = p.parse_args()

    if args.list_rules:
        for rule in PATTERNS:
            print(f"{rule.name:<20} {rule.why}\n{'':<20} /{rule.pattern}/")
        return 0

    if not (args.check or args.build or args.build_dir or args.scrub_results):
        p.error("nothing to do: pass --check, --scrub-results, --build ZIP "
                "and/or --build-dir DIR")

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

    for flag, target, build in (("--build", args.build, build_bundle),
                                ("--build-dir", args.build_dir, build_dir)):
        if not target:
            continue
        if status:
            print(f"\nRefusing to build ({flag}) while --check fails.")
            return status
        info = build(Path(target), root)
        print(f"\nWrote {info.get('zip') or info['dir']} "
              f"({info['files']} files, {info['bytes'] / 1e6:.1f} MB)")
        if info["notebooks_stripped"]:
            print(f"Stripped outputs from {len(info['notebooks_stripped'])} notebook(s): "
                  + ", ".join(info["notebooks_stripped"]))

    if args.build:
        print("Re-check the zip itself before submitting:\n"
              "  python3 -c \"import zipfile,sys; "
              "print(len(zipfile.ZipFile(sys.argv[1]).namelist()))\" "
              + str(args.build))
    if args.build_dir:
        print("Re-check the directory itself before archiving it: grep it for "
              "usernames and\nhome paths with your own tools, not with the one "
              "that wrote it.")

    return status


if __name__ == "__main__":
    sys.exit(main())
