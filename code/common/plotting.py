"""Shared figure style: one palette, one method-to-colour map, one save helper.

Every plotting script in this repository imports from here, so a method keeps the
same colour whether it appears in a synthetic-benchmark figure, a federated
accuracy curve or the centralized learning-rate sweep. That consistency is the
only reason this module exists -- a reader who has learned that orange is
EF21-MuonUSign in one figure should not have to relearn it in the next.

Nothing here writes into ``aaai_article/``. The paper's figures are copied over
deliberately; a plotting run must never silently replace one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "SURFACE", "INK", "INK_2", "MUTED", "GRID", "AXIS", "REFERENCE", "SERIES",
    "METHOD_LABEL", "METHOD_COLOR", "METHOD_ORDER",
    "label_of", "color_of", "order_methods", "style_axes", "save_figure",
]

# -- palette ---------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCE = "#898781"
INK, INK_2, MUTED = "#2b2a27", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

#: Display names, spelled as the paper spells them. ``SignMuon`` signs AFTER the
#: LMO, ``MuonUSign`` signs BEFORE it, ``MuonSign`` signs on both sides -- the
#: names are not interchangeable and a figure legend is exactly where a reader
#: would be misled by the old convention.
METHOD_LABEL: Dict[str, str] = {
    "signmuon": "SignMuon",
    "ef21signmuon": "EF21-SignMuon",
    "muonusign": "MuonUSign",
    "muonsign": "MuonSign",
    "ef21muonusign": "EF21-MuonUSign",
    "ef21muonsign": "EF21-MuonSign",
    "muon": "Muon",
    "muonserver": "Muon (server LMO)",
    "signsgd": "SignSGD",
    "sgd": "SGD",
    "adam": "Adam",
}

#: Hue encodes the family: the sign-around-the-LMO methods are cool, the
#: error-feedback methods are warm, the uncompressed references are green/grey.
METHOD_COLOR: Dict[str, str] = {
    "signmuon": "#2a78d6",
    "muonusign": "#6f4ecf",
    "muonsign": "#0f9bd0",
    "ef21signmuon": "#d4342b",
    "ef21muonusign": "#eb6834",
    "ef21muonsign": "#c9761a",
    "muon": "#1baf7a",
    "muonserver": "#0f7a58",
    "signsgd": "#8a6d3b",
    "sgd": "#898781",
    "adam": "#52514e",
}

#: The order the paper lists them in: six proposed methods, then the references.
METHOD_ORDER: List[str] = list(METHOD_LABEL)

#: Pre-refactor spellings that still appear in older ``metrics.json`` files and
#: in the nanoGPT logs. Resolved so an old result plots under its current name.
_ALIASES = {
    "signmuon_cl": "signmuon",
    "signmuon_ef_21": "ef21signmuon",
    "signmuon_ef_ud": "ef21muonsign",
    "ef_usignmuon": "ef21muonusign",
    "ef_udsignmuon": "ef21muonsign",
    "muon_server": "muonserver",
    "muonlmoserver": "muonserver",
}


def _canonical(method: str) -> str:
    key = str(method).strip().lower().replace("-", "").replace(" ", "")
    return _ALIASES.get(str(method).strip().lower(), key)


def label_of(method: str) -> str:
    """Display name, falling back to the raw key for anything unregistered."""
    return METHOD_LABEL.get(_canonical(method), str(method))


def color_of(method: str, fallback: str = REFERENCE) -> str:
    return METHOD_COLOR.get(_canonical(method), fallback)


def order_methods(methods: Iterable[str]) -> List[str]:
    """Sort into the paper's order; unknown names keep their relative order last."""
    known = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(methods,
                  key=lambda m: (known.get(_canonical(m), len(known)), str(m)))


# -- drawing ---------------------------------------------------------------


def style_axes(ax, logx: bool = False, logy: bool = False) -> None:
    """The house axis style: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor("none")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def save_figure(fig, out_dir: Path, stem: str,
                formats: Sequence[str] = ("pdf", "png"), dpi: int = 200) -> List[Path]:
    """Write ``stem`` into ``out_dir`` in each format, and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        written.append(path)
    return written


def legend(ax, *, loc: str = "best", ncol: int = 1, title: Optional[str] = None,
           outside: bool = False):
    """A legend with the same muted styling as the axes.

    ``outside=True`` parks it to the right of the axes. With ten methods on one
    plot there is no interior corner that does not cover data, and a legend that
    hides the curves it labels is worse than no legend.
    """
    if outside:
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=ncol,
                        title=title, frameon=False, fontsize=8,
                        borderaxespad=0.0)
    else:
        leg = ax.legend(loc=loc, ncol=ncol, title=title, frameon=False, fontsize=8)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(INK_2)
        if leg.get_title() is not None:
            leg.get_title().set_color(MUTED)
            leg.get_title().set_fontsize(8)
    return leg
