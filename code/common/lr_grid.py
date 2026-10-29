"""The shared 1-2-5 learning-rate lattice; no training-library imports.

Extracted unchanged from centralized.tune so lightweight experiment planners
use exactly the same grids and boundary extensions as the homogeneous tuner.
"""

import math
from typing import List, Sequence

ROUND_MANTISSAS = (1.0, 2.0, 5.0)


def lattice_value(k: int) -> float:
    """The k-th 1-2-5 point (k=0 is 1); negative k is allowed."""
    return float(f"{ROUND_MANTISSAS[k % 3] * 10.0 ** (k // 3):.6g}")


def lattice_index(x: float) -> int:
    """Index of the lattice point closest to x in log space."""
    if x <= 0:
        raise ValueError(f"learning rate must be positive; got {x}")
    approx = 3 * math.log10(x)
    target = math.log10(x)
    candidates = range(math.floor(approx) - 2, math.ceil(approx) + 3)
    return min(candidates, key=lambda k: abs(math.log10(lattice_value(k)) - target))


def round_grid(anchor: float, points: int) -> List[float]:
    """Consecutive lattice values centered on the point nearest anchor."""
    if points < 1:
        return [float(f"{anchor:.6g}")]
    k0 = lattice_index(anchor)
    half = (points - 1) // 2
    return [lattice_value(k0 - half + i) for i in range(points)]


def extend_grid(grid: Sequence[float], *, low: bool, points: int = 2) -> List[float]:
    """Continue grid by points lattice steps beyond one endpoint."""
    g = sorted(grid)
    if not g:
        return []
    if low:
        k = lattice_index(g[0])
        return sorted([lattice_value(k - i - 1) for i in range(points)] + g)
    k = lattice_index(g[-1])
    return sorted(g + [lattice_value(k + i + 1) for i in range(points)])
