"""Hirshfeld and CM5 atomic charges from an ABACUS charge density.

Hirshfeld (stockholder) charges partition the electron density with the weights

``w_A(r) = rho_A^pro(r) / sum_B rho_B^pro(r)``,

where the promolecule density ``rho_A^pro`` is the spherically averaged
free-atom density of atom A.  ABACUS stores that density in the pseudopotential
(``PP_RHOATOM``), so the proatoms here come from the same pseudopotentials the
calculation used; for a periodic cell they are summed over lattice images so
the promolecule is periodic too.

CM5 (Marenich, Jerome, Cramer and Truhlar, *J. Chem. Theory Comput.* **2012**,
8, 527) maps the Hirshfeld charges onto class IV charges with

``q_k^CM5 = q_k^Hirshfeld + sum_{k' != k} T_{k k'} B_{k k'}``,

where ``B_{k k'} = exp[-alpha (r_{k k'} - R_{Zk} - R_{Zk'})]`` is Pauling's bond
order, built from the covalent radii ``R_Z`` and the interatomic distance, and
``T_{k k'} = D_{Zk Zk'}`` for the H/C/N/O pairs (antisymmetric, zero for equal
atoms) or ``T_{k k'} = D_{Zk} - D_{Zk'}`` otherwise.  The parameters of Table 1
of the paper (``D_Z``, ``D_{Z Z'}``, ``alpha = 2.474 1/Angstrom``) are bundled
here; the covalent radii are the Cordero single-bond values used by the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.charge import read_job_density, valence_electrons
from abacustools.io.abacus import ReadInput
from abacustools.io.pseudo import UPF
from abacustools.io.stru import AbacusSTRU


@dataclass(frozen=True)
class HirshfeldResult:
    """Hirshfeld and CM5 charges of one job.

    Attributes:
        job: Job directory.
        elements: Element symbol of every atom.
        charges: Hirshfeld charge of every atom.
        volumes: Hirshfeld volume of every atom, in Angstrom^3.
        cm5: CM5 charges, when a parameter table was supplied.
        valence: Valence electron count of every atom.
        grid: Shape of the density grid.
    """

    job: str
    elements: tuple[str, ...]
    charges: np.ndarray
    volumes: np.ndarray
    cm5: Optional[np.ndarray]
    valence: np.ndarray
    grid: tuple[int, int, int]


def _scalar(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _proatom(job: Path, inputs: Mapping, structure: AbacusSTRU, element: str):
    """Return the spherical proatom density of one element.

    The pseudopotential stores ``4 pi r^2 rho(r)``; this returns ``rho(r)`` in
    ``e/Angstrom^3`` on a radial grid in Angstrom.
    """
    for atom in structure.atoms:
        if str(atom.element) != element:
            continue
        pseudo_dir = inputs.get("pseudo_dir")
        base = Path(str(pseudo_dir)) if pseudo_dir else job
        if not base.is_absolute():
            base = job / base
        upf = UPF.read_from_file(base / atom.pp)
        r_bohr = np.asarray(upf.r, dtype=float)
        rhoatom = np.asarray(upf.rhoatom, dtype=float)
        if r_bohr.size < 2:
            raise ValueError(f"{atom.pp} has no radial grid for the proatom density")
        # rho = rhoatom / (4 pi r^2); the r = 0 value is the r -> 0 limit.
        rho = np.zeros_like(r_bohr)
        rho[1:] = rhoatom[1:] / (4.0 * np.pi * r_bohr[1:] ** 2)
        rho[0] = rho[1]
        r_ang = r_bohr * BOHR_TO_ANG
        rho_ang = rho / BOHR_TO_ANG ** 3
        return r_ang, rho_ang
    raise ValueError(f"no atom of element {element} found for the proatom density")


#: Table 1 of the CM5 paper: atom-wise parameters D_Z (dimensionless).
CM5_ATOMIC_PARAMETERS: dict[str, float] = {
    "H": 0.0056, "He": -0.1543, "Li": 0.0, "Be": 0.0333, "B": -0.1030,
    "C": -0.0446, "N": -0.1072, "O": -0.0802, "F": -0.0629, "Ne": -0.1088,
    "Na": 0.0184, "Mg": 0.0, "Al": -0.0726, "Si": -0.0790, "P": -0.0756,
    "S": -0.0565, "Cl": -0.0444, "Ar": -0.0767, "K": 0.0130, "Ca": 0.0,
    "Zn": 0.0, "Ge": -0.0557, "As": -0.0533, "Se": -0.0399, "Br": -0.0313,
    "I": -0.0220,
}

#: Table 1 of the CM5 paper: pairwise parameters D_{Z Z'} for the H/C/N/O pairs.
#: The order is the one the table lists; D_{Z Z'} = -D_{Z' Z}.
CM5_PAIR_PARAMETERS: dict[tuple[str, str], float] = {
    ("H", "C"): 0.0502, ("H", "N"): 0.1747, ("H", "O"): 0.1671,
    ("C", "N"): 0.0556, ("C", "O"): 0.0234, ("N", "O"): -0.0346,
}

#: Exponent of Pauling's bond order, in 1/Angstrom.
CM5_ALPHA = 2.474

#: Elements whose pairs use the tabulated D_{Z Z'} instead of D_Z - D_Z'.
CM5_PAIR_ELEMENTS = frozenset({"H", "C", "N", "O"})

#: Single-bond covalent radii in Angstrom (Cordero et al., the values behind
#: the CRC table the CM5 paper cites for R_Z).
CM5_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Zn": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20,
    "Br": 1.20, "I": 1.39,
}


def _cm5_pair_parameter(first: str, second: str) -> float:
    """Return ``D_{first second}`` with the antisymmetry of the table."""
    if first == second:
        return 0.0
    if (first, second) in CM5_PAIR_PARAMETERS:
        return CM5_PAIR_PARAMETERS[(first, second)]
    if (second, first) in CM5_PAIR_PARAMETERS:
        return -CM5_PAIR_PARAMETERS[(second, first)]
    raise KeyError(f"no CM5 pair parameter for {first}-{second}")


def _cm5_charges(
    elements: Sequence[str],
    positions: np.ndarray,
    cell: np.ndarray,
    charges: np.ndarray,
) -> np.ndarray:
    """Apply the CM5 pairwise correction to Hirshfeld charges."""
    corrected = np.array(charges, dtype=float)
    natom = len(elements)
    for k in range(natom):
        for other in range(natom):
            if k == other:
                continue
            first, second = elements[k], elements[other]
            delta = positions[other] - positions[k]
            # Minimum-image distance for the periodic cell.
            fractional = np.linalg.solve(cell.T, delta)
            fractional -= np.round(fractional)
            distance = float(np.linalg.norm(fractional @ cell))
            radius = CM5_COVALENT_RADII[first] + CM5_COVALENT_RADII[second]
            bond_order = float(np.exp(-CM5_ALPHA * (distance - radius)))
            if first in CM5_PAIR_ELEMENTS and second in CM5_PAIR_ELEMENTS:
                coefficient = _cm5_pair_parameter(first, second)
            else:
                coefficient = CM5_ATOMIC_PARAMETERS[first] - CM5_ATOMIC_PARAMETERS[second]
            corrected[k] += coefficient * bond_order
    return corrected


def hirshfeld_charges(
    job: str | Path,
    *,
    images: int = 1,
    grid_shape: Optional[Sequence[int]] = None,
    lat0: Optional[float] = None,
    cm5: bool = True,
) -> HirshfeldResult:
    """Compute Hirshfeld charges of an ABACUS job.

    Args:
        job: ABACUS job directory.
        images: Lattice images of the proatoms to sum over, in each direction.
        grid_shape: FFT grid, when the density has to be rebuilt from a restart.
        lat0: Lattice constant, when the density has to be rebuilt from a restart.
        cm5: Also compute the CM5 charges (default ``True``).

    Returns:
        The Hirshfeld charges, volumes and, when possible, the CM5 charges.
    """
    job_path = Path(job).resolve()
    inputs = ReadInput(str(job_path / "INPUT"))
    structure = AbacusSTRU.read(str(job_path / str(_scalar(inputs.get("stru_file", "STRU")))))
    if structure is None:
        raise ValueError("cannot read the structure")

    density = read_job_density(
        job_path,
        grid_shape=tuple(grid_shape) if grid_shape is not None else None,
        lat0=lat0,
    )
    total = density.total()
    rho = np.asarray(total.data, dtype=float)          # e/Angstrom^3
    cell = np.asarray(total.cell, dtype=float)         # Angstrom
    positions = np.asarray(total.atom_positions, dtype=float)
    # The cube reader does not carry the valence charges, so read them from
    # the pseudopotentials (this is what the restart path uses too).
    valence = np.asarray(
        valence_electrons(structure, inputs.get("pseudo_dir"), job_path), dtype=float
    )
    shape = rho.shape
    natom = positions.shape[0]

    # Grid points in Cartesian Angstrom (the cube grid is regular in fractional
    # coordinates).
    i = np.arange(shape[0]) / shape[0]
    j = np.arange(shape[1]) / shape[1]
    k = np.arange(shape[2]) / shape[2]
    frac = np.stack(np.meshgrid(i, j, k, indexing="ij"), axis=-1)
    points = frac.reshape(-1, 3) @ cell

    elements = tuple(str(element) for element in structure.elements)
    unique = sorted(set(elements))
    proatoms = {element: _proatom(job_path, inputs, structure, element) for element in unique}

    shifts = [
        np.asarray((a, b, c), dtype=float) @ cell
        for a in range(-images, images + 1)
        for b in range(-images, images + 1)
        for c in range(-images, images + 1)
    ]

    promolecule = np.zeros((natom, points.shape[0]))
    for atom in range(natom):
        r_grid, rho_grid = proatoms[elements[atom]]
        for shift in shifts:
            distance = np.linalg.norm(points - (positions[atom] + shift), axis=1)
            promolecule[atom] += np.interp(distance, r_grid, rho_grid, right=0.0)
    denominator = promolecule.sum(axis=0)
    denominator = np.where(denominator <= 0.0, 1.0, denominator)

    volume_element = abs(float(np.linalg.det(cell))) / (shape[0] * shape[1] * shape[2])
    flat = rho.reshape(-1)
    populations = (promolecule / denominator) @ flat * volume_element
    volumes = (promolecule / denominator).sum(axis=1) * volume_element
    charges = valence - populations

    cm5 = _cm5_charges(elements, positions, cell, charges) if cm5 else None

    return HirshfeldResult(
        job=str(job_path),
        elements=elements,
        charges=charges,
        volumes=volumes,
        cm5=cm5,
        valence=valence,
        grid=tuple(int(n) for n in shape),
    )


__all__ = ["HirshfeldResult", "hirshfeld_charges"]
