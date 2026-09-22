"""The main fields of a structure, for listing many of them at once.

``abacustools file info`` reports one structure in full. Listing a batch of
structures is a different job: it only needs the fields that make structures
comparable, so this module reads the cell, the elements and the space group and
leaves the Wyckoff positions, the coordination, the dimensionality and the
magnetic analysis to the full report.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from abacustools.data.symmetry import space_group_summary
from abacustools.io.stru import AbacusSTRU


def _cell_information(structure: AbacusSTRU) -> dict[str, Any]:
    """Return the cell parameters of a structure.

    Args:
        structure: Structure to measure.

    Returns:
        dict: Lengths in Angstrom, angles in degree, the volume in Angstrom**3
        and whether the cell is periodic. Everything is ``None``, and
        ``periodic`` is ``False``, for a structure without a usable cell.
    """
    cell = np.asarray(structure.cell, dtype=float)
    volume = None
    if cell.shape == (3, 3) and np.all(np.isfinite(cell)):
        determinant = abs(float(np.linalg.det(cell)))
        volume = None if determinant <= 1e-12 else determinant
    parameters = None if volume is None else structure.get_cell_param()
    return {
        "periodic": volume is not None,
        "lengths_angstrom": (
            None if parameters is None else [float(value) for value in parameters[:3]]
        ),
        "angles_degree": (
            None if parameters is None else [float(value) for value in parameters[3:]]
        ),
        "volume_angstrom3": volume,
    }


def structure_summary(
    filename: str | Path,
    *,
    input_format: Optional[str] = None,
    cell: Optional[Sequence[float]] = None,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
) -> dict[str, Any]:
    """Read a structure and return its main fields.

    Args:
        filename: Structure file to read.
        input_format: Explicit input format, inferred from the name when
            omitted.
        cell: Nine cell-vector components for formats without a cell, such as
            XYZ.
        symprec: Symmetry distance tolerance in Angstrom.
        angle_tolerance: Symmetry angle tolerance in degrees.

    Returns:
        dict: The file, number of atoms, formula and element counts, the space
        group with its number and crystal system, and the cell parameters.
        A field the structure cannot provide is ``None`` instead of a guess.

    Raises:
        RuntimeError: When the file cannot be read as a structure.
    """
    structure = AbacusSTRU.read(
        filename,
        fmt=input_format,
        cell=None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3),
    )
    if structure is None:
        raise RuntimeError(f"failed to read structure: {filename}")
    elements = [atom.element or atom.label for atom in structure.atoms]
    symmetry = space_group_summary(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    return {
        "file": str(Path(filename).absolute()),
        "natoms": structure.natoms,
        "formula": " ".join(
            f"{element}{count if count != 1 else ''}"
            for element, count in Counter(elements).items()
        ),
        "element_counts": dict(Counter(elements)),
        "space_group": symmetry["space_group_symbol"],
        "space_group_number": symmetry["space_group_number"],
        "crystal_system": symmetry["crystal_system"],
        "cell": _cell_information(structure),
    }
