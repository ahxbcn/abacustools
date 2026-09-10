"""Composition, mass and density of a structure."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from abacustools.core.constant import AMU_TO_GRAM
from abacustools.io.stru import AbacusSTRU


def _cell_volume(structure: AbacusSTRU) -> float:
    """Return the cell volume in Angstrom^3, or zero without a periodic cell."""
    cell = np.asarray(structure.cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        return 0.0
    volume = abs(float(np.linalg.det(cell)))
    return 0.0 if volume <= 1e-12 else volume


def _cell_mass(structure: AbacusSTRU, composition: Any) -> float:
    """Return the mass of the cell in amu, preferring the masses of the file."""
    masses = [atom.mass for atom in structure.atoms]
    if masses and all(mass is not None for mass in masses):
        return float(sum(float(mass) for mass in masses))
    return float(composition.weight)


def composition_summary(structure: AbacusSTRU) -> dict[str, Any]:
    """Summarise the composition of a structure.

    The formula unit is the reduced composition of the cell, so a primitive
    Fe6O8 cell reports Fe3O4 twice.  The prototype is the composition with its
    elements replaced by A, B, C, ... in the order of electronegativity, as
    prototype formulas are usually written.

    Args:
        structure: Structure to summarise.

    Returns:
        dict: Formula unit, number of formula units per cell, prototype, cell
        mass and density, or ``available=False`` when a label is not a known
        element.
    """
    from pymatgen.core import Composition

    counts = Counter(str(atom.element or atom.label) for atom in structure.atoms)
    result: dict[str, Any] = {
        "available": True,
        "error": None,
        "formula_unit": None,
        "formula_units_per_cell": None,
        "prototype": None,
        "mass_amu": None,
        "density_g_cm3": None,
    }
    try:
        composition = Composition({element: count for element, count in counts.items()})
    except Exception as error:
        result["available"] = False
        result["error"] = f"the structure has an unknown element: {error}"
        return result
    formula_unit, factor = composition.get_reduced_composition_and_factor()
    mass = _cell_mass(structure, composition)
    volume = _cell_volume(structure)
    result["formula_unit"] = formula_unit.reduced_formula
    result["formula_units_per_cell"] = int(round(float(factor)))
    result["prototype"] = composition.anonymized_formula
    result["mass_amu"] = mass
    if volume > 0:
        result["density_g_cm3"] = mass * AMU_TO_GRAM / (volume * 1.0e-24)
    return result


__all__ = ["composition_summary"]
