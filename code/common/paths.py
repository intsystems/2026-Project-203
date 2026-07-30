"""Make the absolute paths inside an export bundle repo-relative.

A run tree records where it ran: the driver's `state.json` stores a `log` and a
`metrics` path per job, the exporters stamp their source tree into a manifest,
and `configs.json` carries whatever the CLI was given. On a Linux box those are
all rooted at a home directory, so they carry a username -- and export bundles
travel with anonymous submissions.

`repo_relative` rewrites any absolute path that reaches a ``results/`` or
``results_old/`` segment down to that segment, which is where the repository
would put it anyway::

    /home/someone/SignMuon/code/results/centralized  ->  results/centralized
    D:\\work\\SignMuon\\code\\results\\run\\m.json    ->  results/run/m.json

Paths that never reach such a segment are left alone: guessing at an anchor
would be worse than leaving a path a reader can see is absolute.

This is defence in depth, not the primary control. `anonymize.py` excludes
`results/` from the anonymous bundle outright; this keeps a leak out of the
files themselves, so that a bundle copied by hand is safe too.

No torch import, deliberately: both exporters run on machines that have none.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

__all__ = ["repo_relative", "results_root", "scrub"]


def results_root() -> Path:
    """``code/results``, or wherever ``SIGNMUON_RESULTS`` points.

    The one definition. `common.utils.results_root` delegates here, and so do the
    exporters and plotters, which cannot import `common.utils` because it pulls in
    torch and they are meant to run on a laptop that has none. Duplicating the
    lookup instead is how the override came to be honoured by the driver but not
    by the export it launches, which produced an empty bundle at the end of a
    redirected run.
    """
    override = os.environ.get("SIGNMUON_RESULTS")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / "results"

#: Everything from a drive letter or root up to (not including) a ``results``
#: segment. Three details earn their keep:
#:
#: * ``pre`` requires the path to start at a delimiter, and is put back. Without
#:   it a LaTeX control sequence reads as a root -- ``\path{/a/results/x}`` has
#:   the regex swallow ``\path{`` as two directories.
#: * braces are excluded from directory names, for the same reason.
#: * the inner group is lazy, so the *first* ``results`` segment wins, and the
#:   class excludes quotes and whitespace so a match cannot run off the end of a
#:   path embedded in JSON.
_LEADING = re.compile(
    r"""(?P<pre>^|[\s"'`=:,\[\](){}<>])  # start of line, or a delimiter we keep
        (?:[A-Za-z]:)?[\\/]              # drive-and-root, or bare root
        (?:[^\s"'\\/{}]+[\\/])*?         # intermediate directories, lazily
        (?=results(?:_old)?[\\/])        # stop before results/ or results_old/
    """,
    re.VERBOSE | re.MULTILINE,
)


#: The remainder of a path once `_LEADING` has taken the prefix off, so that the
#: separators inside *it* can be normalized without touching the rest of the
#: text. Scoping this matters: a blanket ``replace("\\", "/")`` over a whole file
#: silently destroys every LaTeX command in a `hardware.tex` and every escape in
#: a JSON string.
_TAIL = re.compile(r"results(?:_old)?[\\/][^\s\"']*")


def repo_relative(text: str) -> str:
    """Strip the machine-specific prefix from every path in ``text``.

    Takes a string rather than a path so it can be run over a whole file: a
    `state.json` holds hundreds of these, and rewriting them one key at a time
    would mean knowing the schema. Only the matched paths are rewritten -- every
    other character, backslashes included, is returned untouched.
    """
    return _TAIL.sub(lambda m: m.group(0).replace("\\", "/"),
                     _LEADING.sub(r"\g<pre>", text))


def scrub(value: Any) -> Any:
    """`repo_relative` over the strings of a JSON-shaped value, recursively.

    For data already parsed into Python. Dictionary *keys* are rewritten too:
    the drivers key jobs by run name, but a caller may hand us a mapping keyed
    by path.
    """
    if isinstance(value, str):
        return repo_relative(value)
    if isinstance(value, dict):
        return {scrub(k): scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value
