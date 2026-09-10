"""Vacancy formation energy helpers."""

from __future__ import annotations

import copy
from typing import Optional, cast

from abacustools.io.stru import AbacusATOM, AbacusSTRU


ELEMENT_CRYSTAL_STRUCTURES = {
    "Li": {"crystal": "bcc", "a": 3.51},
    "Be": {"crystal": "hcp", "a": 2.27, "c": 3.59},
    "Na": {"crystal": "bcc", "a": 4.23},
    "Mg": {"crystal": "hcp", "a": 3.21, "c": 5.21},
    "Al": {"crystal": "fcc", "a": 4.05},
    "Si": {"crystal": "diamond", "a": 5.43},
    "K": {"crystal": "bcc", "a": 5.23},
    "Ca": {"crystal": "fcc", "a": 5.58},
    "Sc": {"crystal": "hcp", "a": 3.31, "c": 5.27},
    "Ti": {"crystal": "hcp", "a": 2.95, "c": 4.68},
    "V": {"crystal": "bcc", "a": 3.03},
    "Cr": {"crystal": "bcc", "a": 2.88},
    "Mn": {"crystal": "bcc", "a": 2.91},
    "Fe": {"crystal": "bcc", "a": 2.87},
    "Co": {"crystal": "hcp", "a": 2.51, "c": 4.07},
    "Ni": {"crystal": "fcc", "a": 3.52},
    "Cu": {"crystal": "fcc", "a": 3.61},
    "Zn": {"crystal": "hcp", "a": 2.66, "c": 4.95},
    "Ga": {"crystal": "fcc", "a": 4.50},
    "Ge": {"crystal": "diamond", "a": 5.69},
    "Rb": {"crystal": "bcc", "a": 5.58},
    "Sr": {"crystal": "fcc", "a": 6.08},
    "Y": {"crystal": "hcp", "a": 3.65, "c": 5.73},
    "Zr": {"crystal": "hcp", "a": 3.23, "c": 5.15},
    "Nb": {"crystal": "bcc", "a": 3.30},
    "Mo": {"crystal": "bcc", "a": 3.15},
    "Tc": {"crystal": "hcp", "a": 2.74, "c": 4.44},
    "Ru": {"crystal": "hcp", "a": 2.71, "c": 4.28},
    "Rh": {"crystal": "fcc", "a": 3.80},
    "Pd": {"crystal": "fcc", "a": 3.89},
    "Ag": {"crystal": "fcc", "a": 4.09},
    "Cd": {"crystal": "hcp", "a": 2.98, "c": 5.62},
    "In": {"crystal": "bcc", "a": 3.25},
    "Sn": {"crystal": "diamond", "a": 6.49},
    "Cs": {"crystal": "bcc", "a": 6.05},
    "Ba": {"crystal": "bcc", "a": 5.02},
    "La": {"crystal": "hcp", "a": 3.75, "c": 5.75},
    "Ce": {"crystal": "fcc", "a": 5.16},
    "Pr": {"crystal": "hcp", "a": 3.65, "c": 5.75},
    "Nd": {"crystal": "hcp", "a": 3.65, "c": 5.73},
    "Pm": {"crystal": "hcp", "a": 3.65, "c": 5.73},
    "Sm": {"crystal": "hcp", "a": 3.65, "c": 5.73},
    "Eu": {"crystal": "bcc", "a": 4.58},
    "Gd": {"crystal": "hcp", "a": 3.63, "c": 5.78},
    "Tb": {"crystal": "hcp", "a": 3.60, "c": 5.70},
    "Dy": {"crystal": "hcp", "a": 3.59, "c": 5.65},
    "Ho": {"crystal": "hcp", "a": 3.58, "c": 5.62},
    "Er": {"crystal": "hcp", "a": 3.56, "c": 5.59},
    "Tm": {"crystal": "hcp", "a": 3.54, "c": 5.56},
    "Yb": {"crystal": "fcc", "a": 5.48},
    "Lu": {"crystal": "hcp", "a": 3.50, "c": 5.55},
    "Hf": {"crystal": "hcp", "a": 3.19, "c": 5.05},
    "Ta": {"crystal": "bcc", "a": 3.30},
    "W": {"crystal": "bcc", "a": 3.16},
    "Re": {"crystal": "hcp", "a": 2.76, "c": 4.46},
    "Os": {"crystal": "hcp", "a": 2.74, "c": 4.32},
    "Ir": {"crystal": "fcc", "a": 3.84},
    "Pt": {"crystal": "fcc", "a": 3.92},
    "Au": {"crystal": "fcc", "a": 4.08},
    "Hg": {"crystal": "hcp", "a": 2.99, "c": 5.01},
    "Tl": {"crystal": "hcp", "a": 3.46, "c": 5.52},
    "Pb": {"crystal": "fcc", "a": 4.95},
}


def build_elemental_crystal(
    element: str,
    pp: Optional[str] = None,
    orb: Optional[str] = None,
) -> AbacusSTRU:
    """Build the most stable elemental crystal structure for ``element``.

    Uses the ASE ``bulk`` builder with the reference lattice parameters from
    :data:`ELEMENT_CRYSTAL_STRUCTURES` (the same table used by abacus-test).
    """
    from ase.build import bulk

    if element not in ELEMENT_CRYSTAL_STRUCTURES:
        raise ValueError(f"element {element!r} is not supported for reference crystals")
    info = ELEMENT_CRYSTAL_STRUCTURES[element]
    crystal = info["crystal"]
    if crystal == "bcc":
        atoms = bulk(element, "bcc", a=info["a"])
    elif crystal == "fcc":
        atoms = bulk(element, "fcc", a=info["a"])
    elif crystal == "hcp":
        atoms = bulk(element, "hcp", a=info["a"], c=info["c"])
    elif crystal == "diamond":
        atoms = bulk(element, "diamond", a=info["a"])
    else:
        raise ValueError(f"crystal structure {crystal!r} is not supported")

    if pp is not None:
        atoms.info["pp"] = {element: pp}
    if orb is not None:
        atoms.info["orb"] = {element: orb}
    return AbacusSTRU.from_ase(atoms)


def set_atom_empty(structure: AbacusSTRU, index: int) -> AbacusSTRU:
    """Return a copy with the atom at 0-based ``index`` turned into an empty atom.

    The atom keeps its position but is renamed to ``<element>_empty`` and moved
    to the end of the structure. ABACUS treats any label containing ``empty`` as
    an empty element, which is the same convention abacus-test uses for
    vacancies.
    """
    if index < 0 or index >= structure.natoms:
        raise IndexError(f"atom index {index} is out of range for {structure.natoms} atoms")

    new = copy.deepcopy(structure)
    atom = cast(AbacusATOM, new[index])
    element = atom.element or atom.label
    vacancy = atom.model_copy(deep=True)
    vacancy.label = f"{element}_empty"
    vacancy.element = element
    del new[index]
    new.append(vacancy)
    return new


def vacancy_formation_energy(
    defect_energy: float,
    original_energy: float,
    supercell_factor: int,
    reference_atom_energy: float,
) -> float:
    """Vacancy formation energy in eV.

    ``E_f = (E_defect + mu_removed) - E_original * supercell_factor`` where
    ``mu_removed`` is the reference energy per atom of the removed element and
    ``supercell_factor`` is the number of primitive cells in the supercell.
    """
    return (defect_energy + reference_atom_energy) - original_energy * supercell_factor
