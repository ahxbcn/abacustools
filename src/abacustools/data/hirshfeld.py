"""Hirshfeld, Hirshfeld-I and CM5 atomic charges from an ABACUS charge density.

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

Hirshfeld-I (Bultinck, Van Alsenoy, Ayers and Carbo-Dorca, *J. Chem. Phys.*
**2007**, 126, 144111) removes the arbitrariness of the neutral promolecule by
letting the molecular density determine its own reference.  Each iteration
rebuilds the atomic reference density at the population the previous iteration
assigned to the atom,

``rho_A^N(r) = rho_A^floor(N)(r) * (ceil(N) - N) + rho_A^ceil(N)(r) * (N - floor(N))``,

which needs the spherical atomic densities of the element at the integer
valence populations around ``N``.  For a norm-conserving pseudopotential those
come from the pseudo-atomic wavefunctions (``PP_PSWFC``), filled by Aufbau
around the neutral configuration; the radial shapes are frozen, which is the
usual approximation when no separate atomic calculations are run.
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


@dataclass(frozen=True)
class HirshfeldIResult:
    """Hirshfeld-I charges of one job.

    Attributes:
        job: Job directory.
        elements: Element symbol of every atom.
        charges: Hirshfeld-I charge of every atom.
        populations: AIM electron population of every atom.
        valence: Valence electron count of every atom.
        iterations: Number of iterations that were run.
        converged: Whether the populations met the convergence threshold.
        grid: Shape of the density grid.
        reference_source: Where the integer-population reference densities came
            from, for reporting.
    """

    job: str
    elements: tuple[str, ...]
    charges: np.ndarray
    populations: np.ndarray
    valence: np.ndarray
    iterations: int
    converged: bool
    grid: tuple[int, int, int]
    reference_source: str


@dataclass(frozen=True)
class AtomicReference:
    """Spherical reference densities of one element at integer populations.

    Attributes:
        element: Chemical symbol.
        r: Radial grid in Angstrom.
        densities: Maps an integer valence population to ``rho(r)`` in
            ``e/Angstrom^3`` on ``r``.
    """

    element: str
    r: np.ndarray
    densities: Mapping[int, np.ndarray]

    @property
    def populations(self) -> tuple[int, ...]:
        """Return the integer populations that have a reference density."""
        return tuple(sorted(self.densities))

    def at(self, population: float) -> np.ndarray:
        """Return the reference density linearly interpolated at ``population``.

        The two integer populations that bracket ``population`` are blended as
        in Bultinck's Eq. (19).  A population outside the tabulated range is an
        error, so the caller decides whether to clamp or to fail.
        """
        low = int(np.floor(population))
        high = int(np.ceil(population))
        available = self.densities
        if low == high:
            if low not in available:
                raise KeyError(
                    f"{self.element} has no reference density at population {low}"
                )
            return np.asarray(available[low], dtype=float)
        missing = [n for n in (low, high) if n not in available]
        if missing:
            raise KeyError(
                f"{self.element} has no reference density at population "
                f"{missing[0]}; available populations are {self.populations}"
            )
        fraction = population - low
        return (1.0 - fraction) * np.asarray(available[low], dtype=float) + fraction * np.asarray(
            available[high], dtype=float
        )


def _scalar(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _upf_for_element(
    job: Path, inputs: Mapping, structure: AbacusSTRU, element: str
) -> UPF:
    """Return the pseudopotential that the first atom of ``element`` uses."""
    for atom in structure.atoms:
        if str(atom.element) != element:
            continue
        pseudo_dir = inputs.get("pseudo_dir")
        base = Path(str(pseudo_dir)) if pseudo_dir else job
        if not base.is_absolute():
            base = job / base
        return UPF.read_from_file(base / atom.pp)
    raise ValueError(f"no atom of element {element} found in the structure")


def _radial_from_rhoatom(upf: UPF) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r, rho)`` of ``PP_RHOATOM``, in Angstrom and e/Angstrom^3.

    The pseudopotential stores ``4 pi r^2 rho(r)``; this returns ``rho(r)`` on a
    radial grid in Angstrom.
    """
    r_bohr = np.asarray(upf.r, dtype=float)
    rhoatom = np.asarray(upf.rhoatom, dtype=float)
    if r_bohr.size < 2:
        raise ValueError(f"{upf.element} has no radial grid for the proatom density")
    # rho = rhoatom / (4 pi r^2); the r = 0 value is the r -> 0 limit.
    rho = np.zeros_like(r_bohr)
    rho[1:] = rhoatom[1:] / (4.0 * np.pi * r_bohr[1:] ** 2)
    rho[0] = rho[1]
    return r_bohr * BOHR_TO_ANG, rho / BOHR_TO_ANG ** 3


def _proatom(job: Path, inputs: Mapping, structure: AbacusSTRU, element: str):
    """Return the spherical proatom density of one element.

    The pseudopotential stores ``4 pi r^2 rho(r)``; this returns ``rho(r)`` in
    ``e/Angstrom^3`` on a radial grid in Angstrom.
    """
    upf = _upf_for_element(job, inputs, structure, element)
    return _radial_from_rhoatom(upf)


def _reference_occupations(
    waves: Sequence[Mapping], neutral: Sequence[float], target: float
) -> list[float]:
    """Fill the pseudo-atomic orbitals of an element to ``target`` electrons.

    The neutral occupations of ``PP_PSWFC`` are the starting point; electrons
    are then added to the lowest-lying orbitals with spare capacity, or removed
    from the highest-lying occupied ones, ordered by the pseudo-atomic energies
    of the pseudopotential.
    """
    occupations = [float(value) for value in neutral]
    capacities = [2.0 * (2.0 * float(wave["l"]) + 1.0) for wave in waves]

    def energy(index: int) -> float:
        value = waves[index].get("pseudo_energy")
        return float(value) if value is not None else float("inf")

    order = sorted(
        range(len(waves)),
        key=lambda index: (energy(index), float(waves[index]["l"]), index),
    )
    remaining = float(target) - sum(occupations)
    if remaining > 0.0:
        for index in order:
            room = capacities[index] - occupations[index]
            take = min(room, remaining)
            occupations[index] += take
            remaining -= take
            if remaining <= 1e-10:
                break
    elif remaining < 0.0:
        for index in reversed(order):
            take = min(occupations[index], -remaining)
            occupations[index] -= take
            remaining += take
            if remaining >= -1e-10:
                break
    if abs(remaining) > 1e-9:
        raise ValueError(
            f"cannot fill the pseudo-atomic orbitals to {target} electrons; "
            "the pseudopotential does not have enough angular channels"
        )
    return occupations


def pseudo_atomic_references(
    upf: UPF, *, charges: Sequence[int] = (-2, -1, 0, 1, 2)
) -> AtomicReference:
    """Build the integer-population reference densities of one element.

    The neutral population is ``PP_RHOATOM``; the other populations come from
    the ``PP_PSWFC`` pseudo-atomic wavefunctions, whose occupations are refilled
    by Aufbau while their radial shapes stay frozen.  This is the approximation
    used when the reference densities are not computed with separate atomic
    calculations.

    Args:
        upf: Pseudopotential of the element.
        charges: Valence charges ``q = z_valence - N`` to tabulate.  Populations
            below zero are dropped.

    Returns:
        The reference densities on the radial grid of the pseudopotential.

    Raises:
        ValueError: If the pseudopotential has no ``PP_PSWFC`` wavefunctions and
            a non-neutral population is requested.
    """
    r_ang, rho_neutral = _radial_from_rhoatom(upf)
    valence = float(upf.valence)
    waves = list(upf.pseudo_wavefunctions)
    rab = np.asarray(upf.rab, dtype=float)

    targets: dict[int, float] = {}
    for charge in charges:
        population = valence - float(charge)
        if population < -1e-9:
            continue
        targets[int(round(population))] = population

    densities: dict[int, np.ndarray] = {}
    orbital_shapes: Optional[list[np.ndarray]] = None
    for population in sorted(targets):
        if abs(population - valence) < 1e-9:
            densities[int(round(population))] = rho_neutral
            continue
        if not waves:
            raise ValueError(
                f"{upf.element} has no PP_PSWFC wavefunctions, so only the "
                "neutral reference density is available; pass explicit "
                "reference densities for the charged populations"
            )
        if orbital_shapes is None:
            r_bohr = np.asarray(upf.r, dtype=float)
            orbital_shapes = []
            for wave in waves:
                data = np.asarray(wave["data"], dtype=float)
                if rab.size == r_bohr.size:
                    norm = float(np.sum(data ** 2 * rab))
                else:
                    norm = float(np.trapezoid(data ** 2, r_bohr))
                orbital_shapes.append(data ** 2 / norm)
        try:
            occupations = _reference_occupations(
                waves, [float(wave["occupation"]) for wave in waves], population
            )
        except ValueError:
            # The pseudopotential has no angular channel for this population
            # (for example H beyond 1s^2); leave it out of the table.
            continue
        rhoatom_bohr = np.zeros_like(r_ang)
        for occupation, shape in zip(occupations, orbital_shapes):
            rhoatom_bohr += occupation * shape
        rho = np.zeros_like(rhoatom_bohr)
        r_bohr = np.asarray(upf.r, dtype=float)
        rho[1:] = rhoatom_bohr[1:] / (4.0 * np.pi * r_bohr[1:] ** 2)
        rho[0] = rho[1]
        densities[int(round(population))] = rho / BOHR_TO_ANG ** 3

    return AtomicReference(element=upf.element, r=r_ang, densities=densities)


def read_reference_densities(directory: str | Path) -> dict[str, AtomicReference]:
    """Read integer-population reference densities from a directory.

    Every ``*.dat`` file is named ``<element>_<population>.dat`` and holds two
    whitespace-separated columns, ``r`` in Angstrom and ``rho(r)`` in
    ``e/Angstrom^3``.  Files of the same element must share one radial grid.

    Args:
        directory: Directory that holds the ``<element>_<population>.dat`` files.

    Returns:
        A mapping of element symbol to its :class:`AtomicReference`.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"reference density directory not found: {directory}")
    collected: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for path in sorted(directory.glob("*.dat")):
        element, _, count = path.stem.rpartition("_")
        if not element:
            continue
        try:
            population = int(count)
        except ValueError:
            continue
        data = np.loadtxt(path, ndmin=2)
        if data.shape[1] < 2:
            raise ValueError(f"{path} must have two columns, r and rho(r)")
        collected.setdefault(element, {})[population] = (data[:, 0], data[:, 1])

    references: dict[str, AtomicReference] = {}
    for element, table in collected.items():
        grid = next(iter(table.values()))[0]
        for r, _ in table.values():
            if r.shape != grid.shape or not np.allclose(r, grid):
                raise ValueError(
                    f"reference densities of {element} do not share one radial grid"
                )
        references[element] = AtomicReference(
            element=element,
            r=grid,
            densities={population: rho for population, (_, rho) in table.items()},
        )
    return references


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


@dataclass
class _DensityContext:
    """The pieces of an ABACUS job that both charge schemes read."""

    job_path: Path
    inputs: Mapping
    structure: AbacusSTRU
    rho: np.ndarray
    cell: np.ndarray
    positions: np.ndarray
    valence: np.ndarray
    points: np.ndarray
    elements: tuple[str, ...]

    @property
    def grid(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.rho.shape)


def _load_context(
    job: str | Path,
    grid_shape: Optional[Sequence[int]],
    lat0: Optional[float],
) -> _DensityContext:
    """Read the density, structure and valence counts of a job."""
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
    elements = tuple(str(element) for element in structure.elements)

    # Grid points in Cartesian Angstrom (the cube grid is regular in fractional
    # coordinates).
    i = np.arange(rho.shape[0]) / rho.shape[0]
    j = np.arange(rho.shape[1]) / rho.shape[1]
    k = np.arange(rho.shape[2]) / rho.shape[2]
    frac = np.stack(np.meshgrid(i, j, k, indexing="ij"), axis=-1)
    points = frac.reshape(-1, 3) @ cell

    return _DensityContext(
        job_path=job_path,
        inputs=inputs,
        structure=structure,
        rho=rho,
        cell=cell,
        positions=positions,
        valence=valence,
        points=points,
        elements=elements,
    )


def _lattice_shifts(cell: np.ndarray, images: int) -> list[np.ndarray]:
    """Return the lattice-image translations to sum the proatoms over."""
    return [
        np.asarray((a, b, c), dtype=float) @ cell
        for a in range(-images, images + 1)
        for b in range(-images, images + 1)
        for c in range(-images, images + 1)
    ]


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
    context = _load_context(job, grid_shape, lat0)
    rho = context.rho
    cell = context.cell
    positions = context.positions
    valence = context.valence
    shape = rho.shape
    natom = positions.shape[0]

    elements = context.elements
    unique = sorted(set(elements))
    proatoms = {
        element: _proatom(context.job_path, context.inputs, context.structure, element)
        for element in unique
    }

    shifts = _lattice_shifts(cell, images)

    promolecule = np.zeros((natom, context.points.shape[0]))
    for atom in range(natom):
        r_grid, rho_grid = proatoms[elements[atom]]
        for shift in shifts:
            distance = np.linalg.norm(context.points - (positions[atom] + shift), axis=1)
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
        job=str(context.job_path),
        elements=elements,
        charges=charges,
        volumes=volumes,
        cm5=cm5,
        valence=valence,
        grid=context.grid,
    )


def hirshfeld_i_charges(
    job: str | Path,
    *,
    references: Optional[Mapping[str, AtomicReference]] = None,
    charges: Sequence[int] = (-2, -1, 0, 1, 2),
    images: int = 1,
    grid_shape: Optional[Sequence[int]] = None,
    lat0: Optional[float] = None,
    max_iter: int = 200,
    tol: float = 5e-4,
    mixing: float = 1.0,
) -> HirshfeldIResult:
    """Compute Hirshfeld-I charges of an ABACUS job.

    The atomic reference densities are refined against the molecular density
    until the population of every atom stops changing, following Bultinck et al.
    The reference densities at integer populations come from the
    ``PP_PSWFC`` pseudo-atomic wavefunctions of the pseudopotentials by default;
    pass ``references`` to use densities from separate atomic calculations
    instead (see :func:`read_reference_densities`).

    Args:
        job: ABACUS job directory.
        references: Per-element integer-population reference densities.  Built
            from the pseudopotentials when omitted.
        charges: Valence charges ``q = z_valence - N`` to tabulate per element.
        images: Lattice images of the reference atoms to sum over, in each direction.
        grid_shape: FFT grid, when the density has to be rebuilt from a restart.
        lat0: Lattice constant, when the density has to be rebuilt from a restart.
        max_iter: Maximum number of self-consistent iterations.
        tol: Convergence threshold on the largest population change, in electrons.
        mixing: Linear mixing of the population update, in ``(0, 1]``.  The
            default ``1.0`` is Bultinck's undamped iteration; lower it when a
            system oscillates.

    Returns:
        The Hirshfeld-I charges, populations and iteration count.
    """
    if not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must lie in (0, 1]")
    context = _load_context(job, grid_shape, lat0)
    elements = context.elements
    positions = context.positions
    cell = context.cell
    valence = np.array(context.valence, dtype=float)
    natom = positions.shape[0]

    if references is None:
        table = {
            element: pseudo_atomic_references(
                _upf_for_element(context.job_path, context.inputs, context.structure, element),
                charges=charges,
            )
            for element in sorted(set(elements))
        }
        reference_source = "PP_PSWFC pseudo-atomic orbitals"
    else:
        table = dict(references)
        reference_source = "explicit reference densities"
    for element in sorted(set(elements)):
        if element not in table:
            raise ValueError(f"no reference densities for element {element}")

    shifts = _lattice_shifts(cell, images)
    # Distances to every lattice image are precomputed once and reused by the
    # iterations.  Images farther from the cell than the reference radial grid
    # are dropped, so a large cell keeps only the one or few images that reach
    # it and the memory stays bounded.
    atom_images: list[list[np.ndarray]] = []
    for atom in range(natom):
        reference = table[elements[atom]]
        cutoff = float(reference.r[-1])
        per_image: list[np.ndarray] = []
        for shift in shifts:
            origin = positions[atom] + shift
            fractional = np.linalg.solve(cell.T, origin)
            nearest = np.clip(fractional, 0.0, 1.0) @ cell
            if np.linalg.norm(origin - nearest) > cutoff:
                continue
            per_image.append(
                np.linalg.norm(context.points - origin, axis=1).astype(np.float32)
            )
        atom_images.append(per_image)

    flat = context.rho.reshape(-1)
    volume_element = abs(float(np.linalg.det(cell))) / (
        context.rho.shape[0] * context.rho.shape[1] * context.rho.shape[2]
    )

    populations = valence.copy()
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        reference_on_grid = np.empty((natom, context.points.shape[0]))
        for atom in range(natom):
            reference = table[elements[atom]]
            available = reference.populations
            population = float(np.clip(populations[atom], available[0], available[-1]))
            radial = reference.at(population)
            accumulated = np.zeros(context.points.shape[0])
            for distance in atom_images[atom]:
                accumulated += np.interp(distance, reference.r, radial, right=0.0)
            reference_on_grid[atom] = accumulated
        denominator = reference_on_grid.sum(axis=0)
        denominator = np.where(denominator <= 0.0, 1.0, denominator)
        weights = reference_on_grid / denominator
        new_populations = weights @ flat * volume_element
        delta = float(np.max(np.abs(new_populations - populations)))
        populations = populations + mixing * (new_populations - populations)
        iterations = iteration
        if delta < tol:
            converged = True
            break

    return HirshfeldIResult(
        job=str(context.job_path),
        elements=elements,
        charges=valence - populations,
        populations=populations,
        valence=valence,
        iterations=iterations,
        converged=converged,
        grid=context.grid,
        reference_source=reference_source,
    )


__all__ = [
    "AtomicReference",
    "HirshfeldIResult",
    "HirshfeldResult",
    "hirshfeld_charges",
    "hirshfeld_i_charges",
    "pseudo_atomic_references",
    "read_reference_densities",
]
