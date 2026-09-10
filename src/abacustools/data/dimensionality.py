"""Vacuum analysis and dimensionality classification of a structure.

The largest empty span along every lattice direction says how many directions
are still periodic: a slab keeps a vacuum along one of them, a wire along two
and a molecule along all three.  The work-function workflow uses the same
analysis to find the direction that carries the electrostatic vacuum.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from abacustools.io.stru import AbacusSTRU


DIRECTIONS = ("a", "b", "c")
DIMENSIONALITY_LABELS = {
    "bulk": "3D bulk",
    "slab": "2D slab",
    "wire": "1D wire",
    "molecule": "0D molecule or cluster",
}
_DIMENSIONALITY_NAMES = ("molecule", "wire", "slab", "bulk")


def vacuum_gaps(structure: AbacusSTRU) -> list[dict[str, Any]]:
    """Return the largest empty span of every lattice direction.

    Args:
        structure: Structure to inspect.

    Returns:
        list: One entry per lattice direction with the ``direction`` name, the
        gap ``thickness`` and its ``top`` and ``bottom`` boundaries, all in
        Angstrom.  An empty list is returned for a structure without atoms.
    """
    direct = np.asarray(structure.coords_direct, dtype=float)
    cell = np.asarray(structure.cell, dtype=float)
    if direct.ndim != 2 or direct.shape[1] != 3:
        raise ValueError("structure must contain atoms with three-dimensional coordinates")
    if len(direct) == 0:
        return []
    gaps = []
    for axis, direction in enumerate(DIRECTIONS):
        values = np.sort(np.mod(direct[:, axis], 1.0))
        differences = np.append(np.diff(values), values[0] + 1.0 - values[-1])
        index = int(np.argmax(differences))
        if index == len(values) - 1:
            top, bottom = values[0], values[-1]
        else:
            top, bottom = values[index + 1], values[index]
        length = float(np.linalg.norm(cell[axis]))
        gaps.append(
            {
                "direction": direction,
                "thickness": float(differences[index]) * length,
                "top": float(top) * length,
                "bottom": float(bottom) * length,
            }
        )
    return gaps


def largest_vacuum(structure: AbacusSTRU) -> dict[str, Any]:
    """Return the lattice direction with the widest empty span.

    Args:
        structure: Structure to inspect.

    Returns:
        dict: The entry of :func:`vacuum_gaps` with the largest thickness.
    """
    gaps = vacuum_gaps(structure)
    if not gaps:
        raise ValueError("structure contains no atoms")
    return max(gaps, key=lambda gap: gap["thickness"])


def classify_dimensionality(
    structure: AbacusSTRU,
    *,
    min_vacuum: float = 5.0,
) -> dict[str, Any]:
    """Classify a structure as a bulk, slab, wire or molecule.

    A lattice direction is non-periodic when its largest empty span reaches
    ``min_vacuum``.  The number of such directions gives the dimensionality,
    which is three minus the number of vacuum directions.

    Args:
        structure: Structure to inspect.
        min_vacuum: Empty span in Angstrom that counts as vacuum.

    Returns:
        dict: The ``dimensionality`` (``bulk``, ``slab``, ``wire`` or
        ``molecule``), its ``label``, the ``vacuum_directions`` and
        ``periodic_directions``, the largest ``vacuum_thickness`` and the
        per-direction ``gaps``.
    """
    if not np.isfinite(min_vacuum) or min_vacuum <= 0:
        raise ValueError("min_vacuum must be a positive finite number")
    cell = np.asarray(structure.cell, dtype=float)
    dimension = abs(float(np.linalg.det(cell))) if cell.shape == (3, 3) else 0.0
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or dimension <= 1e-12:
        return {
            "available": True,
            "dimensionality": "molecule",
            "label": DIMENSIONALITY_LABELS["molecule"],
            "periodic": False,
            "periodic_dimensions": 0,
            "periodic_directions": [],
            "vacuum_directions": [],
            "vacuum_thickness": None,
            "gaps": [],
            "min_vacuum": float(min_vacuum),
        }
    gaps = vacuum_gaps(structure)
    vacuum_directions = [
        gap["direction"] for gap in gaps if gap["thickness"] >= min_vacuum
    ]
    periodic_directions = [
        direction for direction in DIRECTIONS if direction not in vacuum_directions
    ]
    name = _DIMENSIONALITY_NAMES[len(periodic_directions)]
    vacuum_thickness = max((gap["thickness"] for gap in gaps), default=None)
    return {
        "available": True,
        "dimensionality": name,
        "label": DIMENSIONALITY_LABELS[name],
        "periodic": True,
        "periodic_dimensions": len(periodic_directions),
        "periodic_directions": periodic_directions,
        "vacuum_directions": vacuum_directions,
        "vacuum_thickness": None if vacuum_thickness is None else float(vacuum_thickness),
        "gaps": gaps,
        "min_vacuum": float(min_vacuum),
    }


__all__ = [
    "DIRECTIONS",
    "classify_dimensionality",
    "largest_vacuum",
    "vacuum_gaps",
]
