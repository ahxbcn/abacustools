"""Hirshfeld and CM5 atomic charges from an ABACUS charge density.

Hirshfeld (stockholder) charges partition the electron density with the weights

``w_A(r) = rho_A^pro(r) / sum_B rho_B^pro(r)``,

where the promolecule density ``rho_A^pro`` is the spherically averaged
free-atom density of atom A.  ABACUS stores that density in the pseudopotential
(``PP_RHOATOM``), so the proatoms here come from the same pseudopotentials the
calculation used; for a periodic cell they are summed over lattice images so
the promolecule is periodic too.

CM5 (Marenich, Jerome, Cramer and Truhlar, *J. Chem. Theory Comput.* **2012**,
8, 527) is the Hirshfeld charge plus a pairwise correction,

``q_k^CM5 = q_k^Hirshfeld + sum_{k' != k} T_{k k'}``.

The correction parameters are element-pair specific and must be supplied as a
JSON table (see :func:`read_cm5_parameters`); the values published in the CM5
paper are not bundled here.
"""

from __future__ import annotations

import json
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


def read_cm5_parameters(path: str | Path) -> dict[tuple[str, str], float]:
    """Read a CM5 parameter table.

    The file is JSON mapping element pairs to the correction coefficient, for
    example ``{"C-H": 0.1234, "H-C": 0.1234}``.  Both orders of a pair are
    looked up, so listing one is enough.

    Args:
        path: JSON file with the CM5 coefficients.

    Returns:
        The parameter table, keyed by ``(element_a, element_b)``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[tuple[str, str], float] = {}
    for key, value in raw.items():
        if isinstance(key, (list, tuple)) and len(key) == 2:
            first, second = str(key[0]), str(key[1])
        else:
            first, second = (part.strip() for part in str(key).replace("_", "-").split("-", 1))
        table[(first, second)] = float(value)
        table.setdefault((second, first), float(value))
    return table


def hirshfeld_charges(
    job: str | Path,
    *,
    images: int = 1,
    grid_shape: Optional[Sequence[int]] = None,
    lat0: Optional[float] = None,
    cm5_parameters: Optional[Mapping[tuple[str, str], float]] = None,
) -> HirshfeldResult:
    """Compute Hirshfeld charges of an ABACUS job.

    Args:
        job: ABACUS job directory.
        images: Lattice images of the proatoms to sum over, in each direction.
        grid_shape: FFT grid, when the density has to be rebuilt from a restart.
        lat0: Lattice constant, when the density has to be rebuilt from a restart.
        cm5_parameters: Optional CM5 correction table; when given, CM5 charges
            are computed as well.

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

    cm5 = None
    if cm5_parameters is not None:
        cm5 = np.array(charges, dtype=float)
        for atom in range(natom):
            for other in range(natom):
                if atom == other:
                    continue
                key = (elements[atom], elements[other])
                coefficient = cm5_parameters.get(key)
                if coefficient is None:
                    raise KeyError(f"no CM5 parameter for the pair {key}")
                cm5[atom] += coefficient * (charges[atom] - charges[other])

    return HirshfeldResult(
        job=str(job_path),
        elements=elements,
        charges=charges,
        volumes=volumes,
        cm5=cm5,
        valence=valence,
        grid=tuple(int(n) for n in shape),
    )


__all__ = ["HirshfeldResult", "hirshfeld_charges", "read_cm5_parameters"]
