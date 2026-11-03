"""Periodic Natural Bond Orbital analysis from ABACUS LCAO output.

The implementation follows the periodic NBO construction of Dunnington and
Schmidt. ABACUS already supplies atom-centred LCAO matrices, so the
plane-wave-to-AO projection step is not needed. The analysis builds an
orthogonal Natural Atomic Orbital basis from the periodic density, searches
one- and two-centre NBOs in real space, and optionally evaluates periodic
donor-acceptor interactions from the LCAO Fock matrix.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, is_dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from abacustools.core.constant import RY_TO_EV
from abacustools.data.charge import valence_electrons
from abacustools.data.mayer import (
    NAOData,
    atom_ranges,
    read_nao_file,
    read_overlap_matrix,
    read_overlap_matrix_develop,
)
from abacustools.data.population import (
    BondOrder,
    KPointDensity,
    load_kpoint_densities_from_output,
)
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


KCAL_PER_MOL_TO_EV = 0.0433641039005932


class PeriodicNBOError(RuntimeError):
    """Raised when a Periodic NBO analysis cannot be completed."""


@dataclass(frozen=True)
class PeriodicNBOOptions:
    """Options controlling Periodic NBO construction."""

    cutoff: float = 3.0
    r_shells: int = 1
    one_center_threshold: float = 1.8
    two_center_threshold: float = 1.8
    rydberg_threshold: float = 1.0e-3
    maximum_rydberg_per_atom: int = 4
    donor_acceptor: bool = True
    e2_threshold_ev: float = 0.05
    maximum_donor_acceptor: int = 100


@dataclass(frozen=True)
class AtomPopulation:
    """Atomic population and charge from the periodic population analysis."""

    index: int
    element: str
    population: float
    charge: float
    valence_electrons: float


@dataclass(frozen=True)
class NBOShell:
    """Basis functions belonging to one atom, l and radial shell."""

    atom: int
    element: str
    l: int
    shell: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class NBOComponent:
    """One local component of a periodic NBO."""

    atom: int
    image: tuple[int, int, int]
    coefficients: tuple[complex, ...]


@dataclass(frozen=True)
class NBOWavefunction:
    """One localised NBO, antibond or Rydberg orbital."""

    index: int
    kind: str
    occupancy: float
    centers: tuple[tuple[int, tuple[int, int, int]], ...]
    components: tuple[NBOComponent, ...]
    distance: Optional[float] = None
    hybridizations: tuple[str, ...] = ()
    paired_index: Optional[int] = None
    spin: Optional[int] = None


@dataclass(frozen=True)
class DonorAcceptorInteraction:
    """One periodic second-order donor-acceptor interaction."""

    donor: int
    acceptor: int
    donor_label: str
    acceptor_label: str
    donor_atom: int
    acceptor_atom: int
    acceptor_image: tuple[int, int, int]
    coupling_ev: float
    donor_energy_ev: float
    acceptor_energy_ev: float
    stabilization_ev: float
    stabilization_kcal_mol: float


@dataclass(frozen=True)
class NaturalOrbital:
    """One basis function of the final orthogonal NAO basis."""

    index: int
    atom: int
    element: str
    occupancy: float
    direction: str


@dataclass(frozen=True)
class PeriodicNBOAnalysis:
    """Complete Periodic NBO report for one ABACUS LCAO job."""

    job: str
    nspin: int
    gamma_only: bool
    kpoint_count: int
    basis_functions: int
    total_electrons: float
    occupancy_convention: str
    atoms: tuple[AtomPopulation, ...]
    natural_orbitals: tuple[NaturalOrbital, ...]
    wavefunctions: tuple[NBOWavefunction, ...]
    donor_acceptor: tuple[DonorAcceptorInteraction, ...]
    data_files: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    population_method: str = "NPA"
    occupancy_method: str = "periodic pre-NAO + one-/two-centre NHO projection + Rydberg completion"
    unit_energy: str = "eV"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report."""

        def clean(value: Any) -> Any:
            if is_dataclass(value):
                return clean(asdict(value))
            if isinstance(value, np.ndarray):
                return clean(value.tolist())
            if isinstance(value, np.generic):
                return clean(value.item())
            if isinstance(value, complex):
                if abs(value.imag) < 1.0e-10:
                    return float(value.real)
                return [float(value.real), float(value.imag)]
            if isinstance(value, tuple):
                return [clean(item) for item in value]
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {str(key): clean(item) for key, item in value.items()}
            return value

        return {
            "format": 1,
            "kind": "periodic_nbo",
            "job": self.job,
            "nspin": self.nspin,
            "gamma_only": self.gamma_only,
            "kpoint_count": self.kpoint_count,
            "basis_functions": self.basis_functions,
            "total_electrons": self.total_electrons,
            "occupancy_convention": self.occupancy_convention,
            "population_method": self.population_method,
            "occupancy_method": self.occupancy_method,
            "unit_energy": self.unit_energy,
            "atoms": [clean(item.__dict__) for item in self.atoms],
            "natural_orbitals": [clean(item.__dict__) for item in self.natural_orbitals],
            "wavefunctions": [clean(item.__dict__) for item in self.wavefunctions],
            "donor_acceptor": [clean(item.__dict__) for item in self.donor_acceptor],
            "data_files": list(self.data_files),
            "warnings": list(self.warnings),
        }


@dataclass
class _NaturalBasis:
    transform: np.ndarray
    occupancies: np.ndarray
    atoms: np.ndarray
    directions: tuple[str, ...]
    nbas: int
    pre_transform: np.ndarray | None = None
    pre_occupancies: np.ndarray | None = None


@dataclass
class _SearchResult:
    wavefunctions: list[NBOWavefunction]
    density_blocks: dict[tuple[int, int, int], np.ndarray]
    used: dict[tuple[int, tuple[int, int, int]], list[np.ndarray]]
    donors: list[NBOWavefunction]
    acceptors: list[NBOWavefunction]


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _output_directory(job: Path, inputs: dict[str, Any]) -> Path:
    suffix = str(_scalar(inputs.get("suffix", "ABACUS")))
    return job / f"OUT.{suffix}"


def _resolve_orbital(job: Path, inputs: dict[str, Any], filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    orbital_dir = inputs.get("orbital_dir")
    base = Path(str(orbital_dir)) if orbital_dir else job
    if not base.is_absolute():
        base = job / base
    return base / path


def _read_structure_and_orbitals(
    job: Path,
    inputs: dict[str, Any],
) -> tuple[AbacusSTRU, dict[str, NAOData], list[tuple[int, int]], int]:
    stru_name = str(_scalar(inputs.get("stru_file", "STRU")))
    stru_path = job / stru_name
    if not stru_path.is_file():
        raise PeriodicNBOError(f"structure file does not exist: {stru_path}")
    structure = AbacusSTRU.read(str(stru_path))
    if structure is None:
        raise PeriodicNBOError(f"cannot read structure: {stru_path}")
    nao_by_file: dict[str, NAOData] = {}
    for index, atom in enumerate(structure.atoms, 1):
        if not atom.orb:
            raise PeriodicNBOError(f"atom {index} has no numerical orbital file")
        if atom.orb not in nao_by_file:
            path = _resolve_orbital(job, inputs, atom.orb)
            if not path.is_file():
                raise PeriodicNBOError(f"numerical orbital file does not exist: {path}")
            nao_by_file[atom.orb] = read_nao_file(path)
    ranges, nbas = atom_ranges(structure, nao_by_file)
    return structure, nao_by_file, ranges, nbas


def _shell_layout(
    structure: AbacusSTRU,
    nao_by_file: dict[str, NAOData],
    ranges: Sequence[tuple[int, int]],
) -> tuple[NBOShell, ...]:
    shells: list[NBOShell] = []
    for atom_index, (atom, atom_range) in enumerate(zip(structure.atoms, ranges)):
        if not atom.orb:
            raise PeriodicNBOError(f"atom {atom_index + 1} has no orbital file")
        nao = nao_by_file[atom.orb]
        offset = atom_range[0]
        for l, count in enumerate(nao.orbitals_per_l):
            width = 2 * l + 1
            for radial in range(count):
                indices = tuple(range(offset, offset + width))
                if indices[-1] >= atom_range[1]:
                    raise PeriodicNBOError(
                        f"atom {atom_index + 1}: orbital layout exceeds the NAO range"
                    )
                shells.append(
                    NBOShell(
                        atom=atom_index,
                        element=str(atom.element or atom.label),
                        l=l,
                        shell=radial,
                        indices=indices,
                    )
                )
                offset += width
        if offset != atom_range[1]:
            raise PeriodicNBOError(
                f"atom {atom_index + 1}: orbital layout ends at {offset}, "
                f"but the atom range ends at {atom_range[1]}"
            )
    return tuple(shells)


def _expected_electrons(
    inputs: dict[str, Any],
    structure: AbacusSTRU,
    job: Path,
) -> Optional[float]:
    nelec = float(_scalar(inputs.get("nelec", 0.0)) or 0.0)
    if nelec > 0.0:
        return nelec + float(_scalar(inputs.get("nelec_delta", 0.0)) or 0.0)
    try:
        return float(sum(valence_electrons(structure, inputs.get("pseudo_dir"), job)))
    except (OSError, ValueError, KeyError):
        return None


def _density_normalisation(
    kpoints: Sequence[KPointDensity],
    expected: Optional[float],
) -> tuple[float, bool, str, float]:
    """Determine the occupation scaling and k-weight convention of D(k)."""

    candidates: list[tuple[float, bool, float, float]] = []
    for apply_weight in (False, True):
        for scale in (0.5, 1.0, 2.0):
            total = 0.0
            for kpoint in kpoints:
                factor = float(kpoint.weight) if apply_weight else 1.0
                trace = sum(
                    float(np.real(np.trace(density @ kpoint.overlap)))
                    for density in kpoint.densities.values()
                )
                total += factor * scale * trace
            error = float("inf") if expected is None else abs(total - expected)
            candidates.append((error, apply_weight, scale, total))
    if expected is None:
        apply_weight, scale = False, 1.0
        total = sum(
            sum(
                float(np.real(np.trace(density @ kpoint.overlap)))
                for density in kpoint.densities.values()
            )
            for kpoint in kpoints
        )
    else:
        _error, apply_weight, scale, total = min(candidates)
        tolerance = 1.0e-4 * max(1.0, abs(expected)) + 1.0e-6
        if abs(total - expected) > tolerance:
            raise PeriodicNBOError(
                "cannot match the declared electron count with the LCAO "
                "density matrices; check nelec, nelec_delta and occupations"
            )
    convention = (
        "k-weighted D(k)" if apply_weight else "weighted WFC/D(k)"
    ) + f", spin/occupation scale {scale:g}"
    return scale, apply_weight, convention, float(total)


def kpoint_matrices_to_real_space(
    kpoints: Sequence[tuple[tuple[float, float, float], float, np.ndarray]],
    r_blocks: Iterable[tuple[int, int, int]],
    *,
    apply_weight: bool = True,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Fourier-transform k-resolved matrices to real-space R blocks.

    ABACUS wavefunction-derived D(k) already contains the Brillouin-zone
    weight, whereas overlap and Hamiltonian matrices do not.
    """

    if not kpoints:
        raise PeriodicNBOError("at least one k-point is required")
    weights = [float(weight) for _direct, weight, _matrix in kpoints]
    if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
        raise PeriodicNBOError("k-point weights must be finite and non-negative")
    if abs(sum(weights) - 1.0) > 1.0e-7:
        raise PeriodicNBOError(f"k-point weights sum to {sum(weights):.12g}, not one")
    shape = np.asarray(kpoints[0][2]).shape
    for _direct, _weight, matrix in kpoints:
        if np.asarray(matrix).shape != shape:
            raise PeriodicNBOError("all k-point matrices must have the same shape")
    result: dict[tuple[int, int, int], np.ndarray] = {}
    for block in r_blocks:
        key = tuple(int(value) for value in block)
        total = np.zeros(shape, dtype=np.complex128)
        for direct, weight, matrix in kpoints:
            phase = np.exp(2j * np.pi * np.dot(direct, key))
            factor = float(weight) if apply_weight else 1.0
            total += factor * phase * np.asarray(matrix, dtype=np.complex128)
        result[key] = total
    return result


def _hermitian_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    scale = max(1.0, float(np.max(np.abs(values))))
    if np.min(values) <= 1.0e-9 * scale:
        raise PeriodicNBOError("natural-orbital metric is singular; check the LCAO overlap matrix")
    inverse = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
    return 0.5 * (inverse + inverse.conj().T)


def _weighted_orthogonalizer(
    overlap: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    subset = 0.5 * (overlap + overlap.conj().T)
    safe_weights = np.clip(np.asarray(weights, dtype=float), 1.0e-4, None)
    diagonal = np.diag(safe_weights)
    weighted = diagonal @ subset @ diagonal
    return diagonal @ _hermitian_inverse_sqrt(weighted)


def _orthonormalise_weighted(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if coefficients.shape[1] == 0:
        return coefficients.copy()
    subset_overlap = coefficients.conj().T @ overlap @ coefficients
    transform = _weighted_orthogonalizer(subset_overlap, weights)
    return coefficients @ transform


def _natural_minimum_basis_count(atomic_number: int, valence: int) -> int:
    if atomic_number >= 87:
        raise PeriodicNBOError("Periodic NBO supports atomic numbers below 87")
    if atomic_number >= 55:
        count = 43
    elif atomic_number >= 37:
        count = 27
    elif atomic_number >= 19:
        count = 18
    elif atomic_number >= 11:
        count = 9
    elif atomic_number >= 3:
        count = 5
    else:
        count = 1
    core_electrons = max(0, atomic_number - int(round(valence)))
    count -= core_electrons // 2
    return max(1, count)


def _natural_basis(
    structure: AbacusSTRU,
    shells: Sequence[NBOShell],
    ranges: Sequence[tuple[int, int]],
    density: np.ndarray,
    overlap: np.ndarray,
    atomic_numbers: Sequence[int],
    valences: Sequence[float],
) -> _NaturalBasis:
    nbas = density.shape[0]
    coefficients = np.zeros((nbas, nbas), dtype=np.complex128)
    column_atoms = np.zeros(nbas, dtype=int)
    directions: list[str] = [""] * nbas

    by_atom_l: dict[tuple[int, int], list[NBOShell]] = {}
    for shell in shells:
        by_atom_l.setdefault((shell.atom, shell.l), []).append(shell)

    for (atom, l), atom_shells in by_atom_l.items():
        atom_shells = sorted(atom_shells, key=lambda item: item.shell)
        nshell = len(atom_shells)
        rho_at = np.zeros((nshell, nshell), dtype=float)
        s_at = np.zeros((nshell, nshell), dtype=float)
        width = 2 * l + 1
        for first in range(nshell):
            for second in range(nshell):
                rho_value = 0.0
                s_value = 0.0
                for m in range(width):
                    i = atom_shells[first].indices[m]
                    j = atom_shells[second].indices[m]
                    rho_value += float(np.real(density[i, j])) / width
                    s_value += float(np.real(overlap[i, j])) / width
                rho_at[first, second] = rho_value
                s_at[first, second] = s_value
        s_eval, s_vec = np.linalg.eigh(0.5 * (s_at + s_at.T))
        if np.min(s_eval) <= 1.0e-10:
            raise PeriodicNBOError(
                f"atom {atom + 1}, l={l}: overlapping natural-orbital block is singular"
            )
        s_half_inv = (s_vec * (1.0 / np.sqrt(s_eval))) @ s_vec.T
        whitened = s_half_inv @ rho_at @ s_half_inv
        occupancies, vectors = np.linalg.eigh(0.5 * (whitened + whitened.T))
        order = np.argsort(occupancies)[::-1]
        vectors = vectors[:, order]
        transform = s_half_inv @ vectors
        for source_shell in range(nshell):
            for natural_shell in range(nshell):
                for m in range(width):
                    row = atom_shells[source_shell].indices[m]
                    column = atom_shells[natural_shell].indices[m]
                    column_atoms[column] = atom
                    directions[column] = f"l={l} zeta={natural_shell + 1}"
                    coefficients[row, column] = transform[source_shell, natural_shell]

    raw_occupancies = np.real(np.diag(coefficients.conj().T @ density @ coefficients))
    per_atom_columns: dict[int, list[int]] = {}
    for column in range(nbas):
        per_atom_columns.setdefault(int(column_atoms[column]), []).append(column)
    ordered: list[int] = []
    nmb_columns: list[int] = []
    nrb_columns: list[int] = []
    for atom in range(structure.natoms):
        columns = per_atom_columns.get(atom, [])
        columns.sort(key=lambda column: raw_occupancies[column], reverse=True)
        ordered.extend(columns)
        nmb_count = _natural_minimum_basis_count(
            int(atomic_numbers[atom]), int(round(valences[atom]))
        )
        nmb_count = min(nmb_count, len(columns))
        nmb_columns.extend(columns[:nmb_count])
        nrb_columns.extend(columns[nmb_count:])

    ordered_array = np.asarray(ordered, dtype=int)
    reordered = coefficients[:, ordered_array]
    n_nmb = len(nmb_columns)
    nmb = reordered[:, :n_nmb]
    nrb = reordered[:, n_nmb:]
    nmb_occ = raw_occupancies[np.asarray(nmb_columns, dtype=int)]
    nrb_occ = raw_occupancies[np.asarray(nrb_columns, dtype=int)]
    nmb = _orthonormalise_weighted(nmb, overlap, nmb_occ)
    if nrb.shape[1]:
        nrb = nrb - nmb @ (nmb.conj().T @ overlap @ nrb)
        nrb = _orthonormalise_weighted(
            nrb,
            overlap,
            np.where(nrb_occ > 1.0e-4, nrb_occ, 1.0),
        )
    final = np.hstack((nmb, nrb))
    final_directions = tuple(directions[column] for column in nmb_columns) + tuple(
        directions[column] for column in nrb_columns
    )
    atom_major: list[int] = []
    for atom in range(structure.natoms):
        atom_major.extend(
            index for index, column in enumerate(nmb_columns) if int(column_atoms[column]) == atom
        )
        atom_major.extend(
            n_nmb + index
            for index, column in enumerate(nrb_columns)
            if int(column_atoms[column]) == atom
        )
    grid = np.asarray(atom_major, dtype=int)
    final = final[:, grid]
    final_directions = tuple(final_directions[index] for index in grid)
    final_atoms = np.concatenate(
        [
            np.full(
                len([column for column in ordered if int(column_atoms[column]) == atom]),
                atom,
                dtype=int,
            )
            for atom in range(structure.natoms)
        ]
    )
    inverse = np.linalg.inv(final)
    final_occ = np.real(np.diag(inverse @ density @ inverse.conj().T))
    return _NaturalBasis(
        transform=final,
        occupancies=final_occ,
        atoms=final_atoms,
        directions=final_directions,
        nbas=nbas,
        pre_transform=coefficients[:, ordered_array],
        pre_occupancies=raw_occupancies[ordered_array],
    )


def _hybridization(
    coefficients: np.ndarray,
    atom: int,
    shells: Sequence[NBOShell],
    start: int,
) -> str:
    totals: dict[int, float] = {}
    for shell in shells:
        if shell.atom != atom:
            continue
        local = np.asarray(shell.indices, dtype=int) - start
        value = float(np.sum(np.abs(coefficients[local]) ** 2))
        totals[shell.l] = totals.get(shell.l, 0.0) + value
    norm = sum(totals.values())
    if norm <= 1.0e-14:
        return "unknown"
    labels = "spdfghijklmn"
    parts = [
        f"{labels[l] if l < len(labels) else f'l{l}'} {100.0 * value / norm:.1f}%"
        for l, value in sorted(totals.items())
        if value > 1.0e-8
    ]
    return ", ".join(parts)


def _orbital_label(orbital: NBOWavefunction) -> str:
    if orbital.kind in {"bond", "antibond"}:
        first, second = orbital.centers
        return (
            f"{orbital.kind} {first[0] + 1}-{second[0] + 1}"
            f"[{second[1][0]},{second[1][1]},{second[1][2]}]"
        )
    atom = orbital.centers[0][0]
    return f"{orbital.kind} {atom + 1}"


def _best_image(
    structure: AbacusSTRU,
    first: int,
    second: int,
    r_shells: int,
) -> tuple[tuple[int, int, int], float]:
    lattice = np.asarray(structure.cell, dtype=float)
    coords = np.asarray(structure.coords_direct, dtype=float)
    best: tuple[tuple[int, int, int], float] | None = None
    for image in product(range(-r_shells, r_shells + 1), repeat=3):
        delta = coords[second] + np.asarray(image) - coords[first]
        distance = float(np.linalg.norm(delta @ lattice))
        if best is None or distance < best[1]:
            best = (image, distance)
    if best is None:
        raise PeriodicNBOError("could not determine a periodic atom-pair image")
    return best


def _periodic_images_within_cutoff(
    structure: AbacusSTRU,
    first: int,
    second: int,
    cutoff: float,
    r_shells: int,
) -> list[tuple[tuple[int, int, int], float]]:
    """Return every periodic image of an atom pair inside the cutoff.

    A Wiberg pair value from the population analysis is integrated over the
    periodic images of the same atom pair. A high-symmetry crystal such as MgO
    therefore has several symmetry-equivalent nearest neighbours represented by
    one pair entry. The NBO search must expand that entry before assigning one
    bonding pair per neighbour.
    """

    lattice = np.asarray(structure.cell, dtype=float)
    coords = np.asarray(structure.coords_direct, dtype=float)
    tolerance = max(1.0e-8, 1.0e-8 * float(cutoff))
    images: list[tuple[tuple[int, int, int], float]] = []
    for image in product(range(-r_shells, r_shells + 1), repeat=3):
        if first == second and image == (0, 0, 0):
            continue
        delta = coords[second] + np.asarray(image) - coords[first]
        distance = float(np.linalg.norm(delta @ lattice))
        if distance <= cutoff + tolerance:
            images.append((image, distance))
    images.sort(key=lambda item: item[1])
    if not images:
        image, distance = _best_image(structure, first, second, r_shells)
        images.append((image, distance))
    return images


def _normalise_atomic_component(
    vector: np.ndarray,
    atom: int,
    ranges: Sequence[tuple[int, int]],
    overlap: np.ndarray,
) -> np.ndarray:
    start, stop = ranges[atom]
    full = np.asarray(vector, dtype=np.complex128).copy()
    local = full[start:stop]
    metric = overlap[start:stop, start:stop]
    norm = float(np.sqrt(max(0.0, float(np.real(local.conj() @ metric @ local)))))
    if norm <= 1.0e-12:
        return full * 0.0
    return full / norm


def _generalized_pair_occupancies(
    density: np.ndarray,
    overlap: np.ndarray,
) -> tuple[float, float]:
    """Project a two-orbital density block and return its occupations.

    The density and overlap blocks are expressed in a pair of atom-centred
    natural hybrids. Diagonalising the generalized eigenproblem gives the
    occupations of the in-phase and out-of-phase combinations directly from
    the density matrix, without using a Wiberg bond order as an occupation.

    Args:
        density: Two-orbital density block.
        overlap: Two-orbital overlap block.

    Returns:
        Bonding and antibonding occupations ordered from high to low.
    """

    density = np.asarray(density, dtype=np.complex128)
    overlap = np.asarray(overlap, dtype=np.complex128)
    if density.shape != (2, 2) or overlap.shape != (2, 2):
        raise PeriodicNBOError("pair density and overlap blocks must be 2x2")
    hermitian_overlap = 0.5 * (overlap + overlap.conj().T)
    values, vectors = np.linalg.eigh(hermitian_overlap)
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) <= 1.0e-9 * scale:
        raise PeriodicNBOError("periodic pair hybrid metric is singular")
    inverse = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
    standard = inverse.conj().T @ density @ inverse
    occupations = np.linalg.eigvalsh(0.5 * (standard + standard.conj().T))
    order = np.argsort(occupations)[::-1]
    high = float(np.clip(occupations[order[0]], 0.0, 2.0))
    low = float(np.clip(occupations[order[1]], 0.0, 2.0))
    return high, low


def _pair_budget_occupancies(bond_order: float, high: float, low: float) -> tuple[float, float]:
    """Scale a projected pair split to its periodic two-electron budget."""

    budget = min(2.0, 2.0 * max(0.0, float(bond_order)))
    total = max(1.0e-12, high + low)
    return (
        float(np.clip(budget * high / total, 0.0, 2.0)),
        float(np.clip(budget * low / total, 0.0, 2.0)),
    )


def _remove_hybrid_direction(
    subspace: np.ndarray,
    hybrid: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    """Return the metric-orthonormal remainder of a hybrid subspace."""

    if subspace.shape[1] <= 1:
        return np.zeros((subspace.shape[0], 0), dtype=np.complex128)
    norm = float(np.sqrt(max(0.0, float(np.real(hybrid.conj() @ overlap @ hybrid)))))
    if norm <= 1.0e-12:
        return subspace.copy()
    unit = hybrid / norm
    remainder = subspace - unit[:, None] * (unit.conj() @ overlap @ subspace)[None, :]
    metric = remainder.conj().T @ overlap @ remainder
    values, vectors = np.linalg.eigh(0.5 * (metric + metric.conj().T))
    keep = values > 1.0e-9
    if not np.any(keep):
        return np.zeros((subspace.shape[0], 0), dtype=np.complex128)
    return remainder @ (vectors[:, keep] * (1.0 / np.sqrt(values[keep])))


def _valence_lone_pair_count(valence: float, incident_order: float) -> int:
    """Estimate lone pairs from valence electrons not used in sigma bonds."""

    nonbonding = max(0.0, float(valence) - max(0.0, float(incident_order)))
    return max(0, int(round(nonbonding / 2.0)))


def _search_nbos_from_population(
    structure: AbacusSTRU,
    ranges: Sequence[tuple[int, int]],
    shells: Sequence[NBOShell],
    natural: _NaturalBasis,
    density_blocks: dict[tuple[int, int, int], np.ndarray],
    overlap_blocks: dict[tuple[int, int, int], np.ndarray],
    bond_orders: Sequence[BondOrder],
    atom_populations: Sequence[Any],
    atomic_numbers: Sequence[int],
    options: PeriodicNBOOptions,
) -> _SearchResult:
    """Build NBOs by projecting the periodic density into NHO subspaces.

    Each atom is first represented by an orthonormal natural-hybrid subspace
    built from its pre-NAO minimal basis. The density block of every periodic
    atom pair is then projected into the corresponding NHO subspaces and
    diagonalised as a generalized two-orbital problem. Its eigenvalues give
    the bond/antibond occupations directly. One-centre natural hybrids above
    the valence lone-pair count provide the lone pairs. The remaining density
    is assigned to Rydberg candidates so that the complete NBO set reproduces
    the total electron count.
    """

    nbas = natural.nbas
    blocks = {key: np.asarray(value, dtype=np.complex128) for key, value in density_blocks.items()}
    overlaps = {
        key: np.asarray(value, dtype=np.complex128) for key, value in overlap_blocks.items()
    }
    zero_density = blocks[(0, 0, 0)]
    zero_overlap = np.asarray(overlaps[(0, 0, 0)].real, dtype=float)

    incident_orders: dict[int, float] = {atom: 0.0 for atom in range(structure.natoms)}
    pair_data: list[tuple[int, int, tuple[int, int, int], float, float]] = []
    for bond in bond_orders:
        first = bond.atom1 - 1
        second = bond.atom2 - 1
        images = _periodic_images_within_cutoff(
            structure,
            first,
            second,
            options.cutoff,
            options.r_shells,
        )
        order_per_image = float(bond.value) / len(images)
        for image, distance in images:
            incident_orders[first] += order_per_image
            if second != first:
                incident_orders[second] += order_per_image
            pair_data.append((first, second, image, distance, order_per_image))

    lone_pair_counts = {
        atom: min(
            _valence_lone_pair_count(
                float(atom_populations[atom].valence_electrons),
                incident_orders[atom],
            ),
            ranges[atom][1] - ranges[atom][0],
        )
        for atom in range(structure.natoms)
    }

    hybrid_coefficients: dict[int, np.ndarray] = {}
    hybrid_occupancies: dict[int, np.ndarray] = {}
    remaining_hybrids: dict[int, np.ndarray] = {}
    for atom, (start, stop) in enumerate(ranges):
        count = _natural_minimum_basis_count(
            int(atomic_numbers[atom]),
            int(round(float(atom_populations[atom].valence_electrons))),
        )
        count = min(count, stop - start)
        raw = natural.pre_transform[start:stop, start : start + count]
        metric = raw.conj().T @ zero_overlap[start:stop, start:stop] @ raw
        values, vectors = np.linalg.eigh(0.5 * (metric + metric.conj().T))
        if float(np.min(values)) <= 1.0e-10:
            raise PeriodicNBOError(f"atom {atom + 1}: natural-hybrid overlap block is singular")
        orthonormal = raw @ ((vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T)
        local_density = orthonormal.conj().T @ zero_density[start:stop, start:stop] @ orthonormal
        occupations, hybrid_vectors = np.linalg.eigh(0.5 * (local_density + local_density.conj().T))
        order = np.argsort(occupations)[::-1]
        coefficients = np.zeros((nbas, count), dtype=np.complex128)
        coefficients[start:stop] = orthonormal @ hybrid_vectors[:, order]
        hybrid_coefficients[atom] = coefficients
        hybrid_occupancies[atom] = np.clip(occupations[order], 0.0, 2.0)

    wavefunctions: list[NBOWavefunction] = []
    donors: list[NBOWavefunction] = []
    acceptors: list[NBOWavefunction] = []
    used: dict[tuple[int, tuple[int, int, int]], list[np.ndarray]] = {}
    assigned_electrons = 0.0

    for atom, (start, stop) in enumerate(ranges):
        coefficient_block = hybrid_coefficients[atom]
        for column in range(min(lone_pair_counts[atom], coefficient_block.shape[1])):
            vector = coefficient_block[:, column]
            occupancy = float(hybrid_occupancies[atom][column])
            orbital = NBOWavefunction(
                index=len(wavefunctions) + 1,
                kind="lone_pair",
                occupancy=occupancy,
                centers=((atom, (0, 0, 0)),),
                components=(
                    NBOComponent(
                        atom,
                        (0, 0, 0),
                        tuple(complex(x) for x in vector[start:stop]),
                    ),
                ),
                hybridizations=(_hybridization(vector[start:stop], atom, shells, start),),
            )
            wavefunctions.append(orbital)
            donors.append(orbital)
            assigned_electrons += occupancy
        remaining_hybrids[atom] = coefficient_block[:, lone_pair_counts[atom] :]

    for first, second, image, distance, bond_order in sorted(
        pair_data,
        key=lambda item: (item[3], item[0], item[1], item[2]),
    ):
        first_hybrids = remaining_hybrids[first]
        second_hybrids = remaining_hybrids[second]
        if first_hybrids.shape[1] == 0:
            first_hybrids = hybrid_coefficients[first]
        if second_hybrids.shape[1] == 0:
            second_hybrids = hybrid_coefficients[second]
        negative_image = tuple(-value for value in image)
        coupling = first_hybrids.conj().T @ blocks[negative_image] @ second_hybrids
        left_vectors, _singular_values, right_vectors = np.linalg.svd(coupling, full_matrices=False)
        if left_vectors.shape[1] == 0 or right_vectors.shape[0] == 0:
            raise PeriodicNBOError(
                f"atoms {first + 1}-{second + 1}: no natural hybrids are available"
            )
        first_hybrid = first_hybrids @ left_vectors[:, 0]
        second_hybrid = second_hybrids @ right_vectors.conj().T[:, 0]
        pair_density = np.asarray(
            [
                [
                    first_hybrid.conj() @ zero_density @ first_hybrid,
                    first_hybrid.conj() @ blocks[negative_image] @ second_hybrid,
                ],
                [
                    second_hybrid.conj() @ blocks[image] @ first_hybrid,
                    second_hybrid.conj() @ zero_density @ second_hybrid,
                ],
            ],
            dtype=np.complex128,
        )
        pair_overlap = np.asarray(
            [
                [
                    first_hybrid.conj() @ zero_overlap @ first_hybrid,
                    first_hybrid.conj() @ overlaps[negative_image] @ second_hybrid,
                ],
                [
                    second_hybrid.conj() @ overlaps[image] @ first_hybrid,
                    second_hybrid.conj() @ zero_overlap @ second_hybrid,
                ],
            ],
            dtype=np.complex128,
        )
        high, low = _generalized_pair_occupancies(pair_density, pair_overlap)
        bond_occupancy, anti_occupancy = _pair_budget_occupancies(bond_order, high, low)

        hermitian_pair_overlap = 0.5 * (pair_overlap + pair_overlap.conj().T)
        values, vectors = np.linalg.eigh(hermitian_pair_overlap)
        inverse = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
        standard = inverse.conj().T @ pair_density @ inverse
        pair_values, pair_vectors = np.linalg.eigh(0.5 * (standard + standard.conj().T))
        order = np.argsort(pair_values)[::-1]
        pair_components: list[tuple[np.ndarray, np.ndarray]] = []
        for column in order[:2]:
            reduced = inverse @ pair_vectors[:, column]
            combined = first_hybrid * reduced[0] + second_hybrid * reduced[1]
            first_component = combined[ranges[first][0] : ranges[first][1]]
            second_component = combined[ranges[second][0] : ranges[second][1]]
            pair_components.append((first_component, second_component))

        bond_first, bond_second = pair_components[0]
        anti_first, anti_second = pair_components[1]
        bond = NBOWavefunction(
            index=len(wavefunctions) + 1,
            kind="bond",
            occupancy=bond_occupancy,
            centers=((first, (0, 0, 0)), (second, image)),
            components=(
                NBOComponent(
                    first,
                    (0, 0, 0),
                    tuple(complex(x) for x in bond_first),
                ),
                NBOComponent(
                    second,
                    image,
                    tuple(complex(x) for x in bond_second),
                ),
            ),
            distance=distance,
            hybridizations=(
                _hybridization(bond_first, first, shells, ranges[first][0]),
                _hybridization(bond_second, second, shells, ranges[second][0]),
            ),
        )
        wavefunctions.append(bond)
        donors.append(bond)
        anti = NBOWavefunction(
            index=len(wavefunctions) + 1,
            kind="antibond",
            occupancy=anti_occupancy,
            centers=((first, (0, 0, 0)), (second, image)),
            components=(
                NBOComponent(
                    first,
                    (0, 0, 0),
                    tuple(complex(x) for x in anti_first),
                ),
                NBOComponent(
                    second,
                    image,
                    tuple(complex(x) for x in anti_second),
                ),
            ),
            distance=distance,
            hybridizations=(
                _hybridization(anti_first, first, shells, ranges[first][0]),
                _hybridization(anti_second, second, shells, ranges[second][0]),
            ),
            paired_index=bond.index,
        )
        wavefunctions.append(anti)
        acceptors.append(anti)
        assigned_electrons += bond_occupancy + anti_occupancy
        remaining_hybrids[first] = _remove_hybrid_direction(
            first_hybrids, first_hybrid, zero_overlap
        )
        remaining_hybrids[second] = _remove_hybrid_direction(
            second_hybrids, second_hybrid, zero_overlap
        )
        used.setdefault((first, (0, 0, 0)), []).append(first_hybrid)
        used.setdefault((second, image), []).append(second_hybrid)

    rydberg_candidates: list[tuple[int, int, np.ndarray, float]] = []
    for atom, (start, stop) in enumerate(ranges):
        count = min(
            _natural_minimum_basis_count(
                int(atomic_numbers[atom]),
                int(round(float(atom_populations[atom].valence_electrons))),
            ),
            stop - start,
        )
        offsets = list(range(count, stop - start))
        offsets.sort(
            key=lambda offset: float(natural.pre_occupancies[start + offset]), reverse=True
        )
        for offset in offsets[: options.maximum_rydberg_per_atom]:
            weight = float(natural.pre_occupancies[start + offset])
            if weight < options.rydberg_threshold:
                continue
            vector = np.asarray(
                natural.pre_transform[start:stop, start + offset],
                dtype=np.complex128,
            )
            vector = _normalise_atomic_component(
                np.pad(vector, (start, nbas - stop)),
                atom,
                ranges,
                zero_overlap,
            )
            rydberg_candidates.append((atom, offset, vector, weight))

    total_electrons = sum(float(item.population) for item in atom_populations)
    residual = max(0.0, total_electrons - assigned_electrons)
    rydberg_occupancies = np.zeros(len(rydberg_candidates), dtype=float)
    if rydberg_candidates and residual > 1.0e-10:
        weights = np.asarray([item[3] for item in rydberg_candidates], dtype=float)
        weights = np.clip(weights, 1.0e-8, None)
        remaining = residual
        active = np.ones(len(weights), dtype=bool)
        for _iteration in range(len(weights) + 1):
            if remaining <= 1.0e-10 or not np.any(active):
                break
            share = remaining * weights[active] / float(np.sum(weights[active]))
            occupied = rydberg_occupancies[active]
            room = np.clip(2.0 - occupied, 0.0, None)
            addition = np.minimum(share, room)
            rydberg_occupancies[active] += addition
            remaining -= float(np.sum(addition))
            active = rydberg_occupancies < 2.0 - 1.0e-10
        residual = remaining

    for (atom, _offset, vector, _weight), occupancy in zip(rydberg_candidates, rydberg_occupancies):
        start, stop = ranges[atom]
        orbital = NBOWavefunction(
            index=len(wavefunctions) + 1,
            kind="rydberg",
            occupancy=float(occupancy),
            centers=((atom, (0, 0, 0)),),
            components=(
                NBOComponent(
                    atom,
                    (0, 0, 0),
                    tuple(complex(x) for x in vector[start:stop]),
                ),
            ),
            hybridizations=(_hybridization(vector[start:stop], atom, shells, start),),
        )
        wavefunctions.append(orbital)
        acceptors.append(orbital)

    occupancy_by_index = {item.index: item.occupancy for item in wavefunctions}
    antibond_index = {
        int(item.paired_index): item.index
        for item in wavefunctions
        if item.kind == "antibond" and item.paired_index is not None
    }
    bond_groups: dict[tuple[tuple[str, str], float], list[NBOWavefunction]] = {}
    for orbital in wavefunctions:
        if orbital.kind != "bond" or orbital.distance is None:
            continue
        first = orbital.centers[0][0]
        second = orbital.centers[1][0]
        pair = tuple(
            sorted(
                (
                    str(structure.atoms[first].element or structure.atoms[first].label),
                    str(structure.atoms[second].element or structure.atoms[second].label),
                )
            )
        )
        key = (pair, round(float(orbital.distance), 4))
        bond_groups.setdefault(key, []).append(orbital)
    for group in bond_groups.values():
        if len(group) < 2:
            continue
        average_bond = float(np.mean([item.occupancy for item in group]))
        average_anti = float(
            np.mean(
                [
                    occupancy_by_index[antibond_index[item.index]]
                    for item in group
                    if item.index in antibond_index
                ]
            )
        )
        for item in group:
            occupancy_by_index[item.index] = average_bond
            if item.index in antibond_index:
                occupancy_by_index[antibond_index[item.index]] = average_anti
    wavefunctions = [
        replace(item, occupancy=float(occupancy_by_index[item.index])) for item in wavefunctions
    ]
    donors = [item for item in wavefunctions if item.kind in {"bond", "lone_pair"}]
    acceptors = [item for item in wavefunctions if item.kind in {"antibond", "rydberg"}]

    return _SearchResult(
        wavefunctions=wavefunctions,
        density_blocks=density_blocks,
        used=used,
        donors=donors,
        acceptors=acceptors,
    )


def _fock_element(
    first: NBOWavefunction,
    second: NBOWavefunction,
    fock_blocks: dict[tuple[int, int, int], np.ndarray],
    ranges: Sequence[tuple[int, int]],
) -> float:
    total = 0.0
    for left in first.components:
        l0, l1 = ranges[left.atom]
        left_coefficients = np.asarray(left.coefficients, dtype=np.complex128)
        for right in second.components:
            r0, r1 = ranges[right.atom]
            relative = tuple(right.image[index] - left.image[index] for index in range(3))
            block = fock_blocks.get(tuple(-value for value in relative))
            if block is None:
                continue
            right_coefficients = np.asarray(right.coefficients, dtype=np.complex128)
            total += float(
                np.real(
                    left_coefficients.conj() @ np.asarray(block[l0:l1, r0:r1]) @ right_coefficients
                )
            )
    return total


def _donor_acceptor_interactions(
    donors: Sequence[NBOWavefunction],
    acceptors: Sequence[NBOWavefunction],
    fock_blocks: dict[tuple[int, int, int], np.ndarray],
    ranges: Sequence[tuple[int, int]],
    options: PeriodicNBOOptions,
) -> list[DonorAcceptorInteraction]:
    result: list[DonorAcceptorInteraction] = []
    for donor in donors:
        donor_energy = _fock_element(donor, donor, fock_blocks, ranges)
        for acceptor in acceptors:
            if acceptor.index == donor.index:
                continue
            coupling = _fock_element(donor, acceptor, fock_blocks, ranges)
            acceptor_energy = _fock_element(acceptor, acceptor, fock_blocks, ranges)
            gap = acceptor_energy - donor_energy
            if gap <= 1.0e-4 or abs(coupling) <= 1.0e-8:
                continue
            stabilization = -float(donor.occupancy) * coupling**2 / gap
            if stabilization < options.e2_threshold_ev:
                continue
            acceptor_atom, acceptor_image = acceptor.centers[-1]
            result.append(
                DonorAcceptorInteraction(
                    donor=donor.index,
                    acceptor=acceptor.index,
                    donor_label=_orbital_label(donor),
                    acceptor_label=_orbital_label(acceptor),
                    donor_atom=donor.centers[0][0],
                    acceptor_atom=acceptor_atom,
                    acceptor_image=acceptor_image,
                    coupling_ev=coupling,
                    donor_energy_ev=donor_energy,
                    acceptor_energy_ev=acceptor_energy,
                    stabilization_ev=stabilization,
                    stabilization_kcal_mol=stabilization / KCAL_PER_MOL_TO_EV,
                )
            )
    result.sort(key=lambda item: item.stabilization_ev, reverse=True)
    return result[: options.maximum_donor_acceptor]


def _hamiltonian_files(
    output: Path,
    kpoints: Sequence[KPointDensity],
    nspin: int,
) -> dict[tuple[int, int], Path]:
    files: dict[tuple[int, int], Path] = {}
    for kpoint in kpoints:
        for spin in (1,) if nspin == 1 else (1, 2):
            candidates = [
                output / f"data-{kpoint.index - 1}-H",
                output / f"data-{kpoint.index}-H",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    files[(kpoint.index, spin)] = candidate
                    break

    entries = sorted(output.glob("hk*_nao.txt"), key=lambda path: path.name)
    if entries:
        by_index: dict[int, dict[int, Path]] = {}
        for path in entries:
            match = re.match(
                r"^hk(?P<ik>\d+)?(?:s(?P<is>\d+))?(?:g\d+)?_nao\.txt$",
                path.name,
                re.IGNORECASE,
            )
            if match is None:
                continue
            ik = int(match.group("ik") or 1)
            spin = int(match.group("is") or 1)
            by_index.setdefault(ik, {})[spin] = path
        for kpoint in kpoints:
            entry = by_index.get(kpoint.index)
            if entry is None:
                continue
            for spin in (1,) if nspin == 1 else (1, 2):
                if spin in entry:
                    files[(kpoint.index, spin)] = entry[spin]
    return files


def _load_raw_fock_blocks(
    output: Path,
    kpoints: Sequence[KPointDensity],
    nspin: int,
    basis_size: int,
    r_blocks: Sequence[tuple[int, int, int]],
) -> dict[tuple[int, int, int], np.ndarray] | None:
    """Load the AO-basis Fock/Hamiltonian matrix in real space and eV."""

    paths = _hamiltonian_files(output, kpoints, nspin)
    matrices: list[tuple[tuple[float, float, float], float, np.ndarray]] = []
    for kpoint in kpoints:
        path = paths.get((kpoint.index, 1))
        if path is None:
            return None
        try:
            if path.name.startswith("data-"):
                hamiltonian = read_overlap_matrix(path)
            else:
                hamiltonian = read_overlap_matrix_develop(path)
        except (OSError, ValueError):
            return None
        if hamiltonian.shape != (basis_size, basis_size):
            return None
        matrix = hamiltonian * RY_TO_EV
        matrix = 0.5 * (matrix + matrix.conj().T)
        matrices.append((kpoint.direct, kpoint.weight, matrix))
    blocks = kpoint_matrices_to_real_space(matrices, r_blocks)
    result: dict[tuple[int, int, int], np.ndarray] = {}
    for key, value in blocks.items():
        if np.max(np.abs(value.imag)) > 1.0e-6:
            return None
        matrix = np.asarray(value.real, dtype=float)
        result[key] = 0.5 * (matrix + matrix.T)
    return result


def _load_kpoint_fock_matrices(
    output: Path,
    kpoints: Sequence[KPointDensity],
    nspin: int,
    basis_size: int,
) -> list[np.ndarray] | None:
    """Load AO-basis Fock/Hamiltonian matrices in k space and eV."""

    if nspin != 1:
        return None
    paths = _hamiltonian_files(output, kpoints, nspin)
    matrices: list[np.ndarray] = []
    for kpoint in kpoints:
        path = paths.get((kpoint.index, 1))
        if path is None:
            return None
        try:
            if path.name.startswith("data-"):
                matrix = read_overlap_matrix(path)
            else:
                matrix = read_overlap_matrix_develop(path)
        except (OSError, ValueError):
            return None
        if matrix.shape != (basis_size, basis_size):
            return None
        matrix = matrix * RY_TO_EV
        matrices.append(0.5 * (matrix + matrix.conj().T))
    return matrices


def analyze_periodic_nbo(
    job: str | Path,
    *,
    options: Optional[PeriodicNBOOptions] = None,
) -> PeriodicNBOAnalysis:
    """Run the original periodic NAO/NBO/NPA algorithm on an ABACUS LCAO job.

    The pre-NAO construction follows ``pre_nao.f90`` and the NBO search follows
    ``nbo.f90`` from Dunnington and Schmidt (2012).  ABACUS substitutes its own
    k-resolved LCAO density and overlap matrices for the plane-wave projection
    stage of the reference workflow.

    Args:
        job: ABACUS job directory.
        options: Search thresholds and periodic-neighbour cutoff.

    Returns:
        NPA populations, final NAOs, one- and two-centre NBOs, antis bonds,
        Rydberg orbitals, and optional Fock-based donor-acceptor interactions.

    Raises:
        PeriodicNBOError: If the job is not LCAO or required matrices are
            missing or inconsistent.
    """

    options = options or PeriodicNBOOptions()
    if options.cutoff <= 0.0:
        raise PeriodicNBOError("cutoff must be positive")
    if options.r_shells < 0:
        raise PeriodicNBOError("r_shells must be non-negative")
    job_path = Path(job).resolve()
    if not job_path.is_dir():
        raise PeriodicNBOError(f"job directory does not exist: {job_path}")
    input_path = job_path / "INPUT"
    if not input_path.is_file():
        raise PeriodicNBOError(f"INPUT file does not exist: {input_path}")
    inputs = ReadInput(str(input_path))
    if not inputs:
        raise PeriodicNBOError(f"cannot read INPUT: {input_path}")
    if str(_scalar(inputs.get("basis_type", "pw"))).lower() != "lcao":
        raise PeriodicNBOError("Periodic NBO requires basis_type=lcao")
    nspin = int(_scalar(inputs.get("nspin", 1)) or 1)
    if nspin not in (1, 2):
        raise PeriodicNBOError(f"Periodic NBO supports nspin=1 or 2, got {nspin}")
    gamma_only = bool(int(_scalar(inputs.get("gamma_only", 0)) or 0))

    structure, nao_by_file, ranges, nbas = _read_structure_and_orbitals(job_path, inputs)
    shells = _shell_layout(structure, nao_by_file, ranges)
    output = _output_directory(job_path, inputs)
    if not output.is_dir():
        raise PeriodicNBOError(f"ABACUS output directory does not exist: {output}")

    kpoints = load_kpoint_densities_from_output(output, nspin, gamma_only)
    expected = _expected_electrons(inputs, structure, job_path)
    valences = valence_electrons(structure, inputs.get("pseudo_dir"), job_path)

    from abacustools.data.nbo_search import (
        PeriodicNBOSearchOptions,
        search_periodic_nbos,
    )
    from abacustools.data.periodic_nao import build_periodic_nao

    fock_matrices = _load_kpoint_fock_matrices(output, kpoints, nspin, nbas)
    try:
        nao = build_periodic_nao(
            structure,
            ranges,
            shells,
            kpoints,
            r_shells=options.r_shells,
            fock_matrices=fock_matrices,
            valences=valences,
        )
    except ValueError as error:
        raise PeriodicNBOError(str(error)) from error

    total_electrons = float(np.sum(nao.populations))
    if expected is not None and abs(total_electrons - expected) > 1.0e-4 * max(1.0, abs(expected)):
        raise PeriodicNBOError(
            "the NPA electron count does not match the INPUT/pseudopotential count "
            f"({total_electrons:.8f} vs {expected:.8f})"
        )

    search_options = PeriodicNBOSearchOptions(
        one_center_threshold=options.one_center_threshold,
        two_center_threshold=options.two_center_threshold,
        cutoff=options.cutoff,
    )
    wavefunctions: list[NBOWavefunction] = []
    assigned_electrons = 0.0
    try:
        if nspin == 1:
            search = search_periodic_nbos(
                structure,
                ranges,
                shells,
                nao,
                nspin=1,
                options=search_options,
            )
            wavefunctions.extend(search.wavefunctions)
            assigned_electrons = search.assigned_electrons
        else:
            if nao.spin_density_blocks is None:
                raise PeriodicNBOError(
                    "spin-resolved NBO search requires both spin density channels"
                )
            for spin, spin_blocks in nao.spin_density_blocks.items():
                spin_nao = replace(nao, density_blocks=spin_blocks)
                spin_search = search_periodic_nbos(
                    structure,
                    ranges,
                    shells,
                    spin_nao,
                    nspin=2,
                    options=search_options,
                )
                wavefunctions.extend(
                    replace(orbital, spin=spin) for orbital in spin_search.wavefunctions
                )
                assigned_electrons += spin_search.assigned_electrons
    except ValueError as error:
        raise PeriodicNBOError(str(error)) from error

    atom_entries = tuple(
        AtomPopulation(
            index=index + 1,
            element=str(atom.element or atom.label),
            population=float(nao.populations[index]),
            charge=float(nao.charges[index]),
            valence_electrons=float(valences[index]),
        )
        for index, atom in enumerate(structure.atoms)
    )

    basis_atom = np.zeros(nbas, dtype=int)
    basis_shell = np.zeros(nbas, dtype=int)
    for atom, (start, stop) in enumerate(ranges):
        basis_atom[start:stop] = atom
        basis_shell[start:stop] = np.arange(stop - start)
    natural_orbitals = tuple(
        NaturalOrbital(
            index=index + 1,
            atom=int(basis_atom[index]) + 1,
            element=str(
                structure.atoms[int(basis_atom[index])].element
                or structure.atoms[int(basis_atom[index])].label
            ),
            occupancy=float(nao.weights[index]),
            direction=(
                f"l={int(nao.basis_l[index])} "
                f"basis={int(nao.basis_shell[index])} "
                f"shell={int(basis_shell[index]) + 1}"
            ),
        )
        for index in range(nbas)
    )

    donors = [orbital for orbital in wavefunctions if orbital.kind in {"bond", "lone_pair", "core"}]
    acceptors = [orbital for orbital in wavefunctions if orbital.kind in {"antibond", "rydberg"}]
    donor_acceptor: list[DonorAcceptorInteraction] = []
    warnings: list[str] = []
    if options.donor_acceptor and nspin == 1 and nao.fock_blocks is not None:
        donor_acceptor = _donor_acceptor_interactions(
            donors,
            acceptors,
            nao.fock_blocks,
            ranges,
            options,
        )
    elif options.donor_acceptor:
        warnings.append(
            "Fock/Hamiltonian matrices were unavailable; donor-acceptor interactions were skipped"
        )

    if abs(assigned_electrons - total_electrons) > 1.0e-4 * max(1.0, total_electrons):
        warnings.append(
            "the localized NBO occupancies do not reproduce the NPA electron count "
            f"({assigned_electrons:.8f} vs {total_electrons:.8f})"
        )
    if any(item.occupancy < -1.0e-7 for item in wavefunctions):
        warnings.append(
            "one or more localized orbitals have a negative occupancy from numerical noise"
        )

    data_files = tuple(
        dict.fromkeys(
            (
                *(file for kpoint in kpoints for file in kpoint.data_files),
                *((str(path) for path in _hamiltonian_files(output, kpoints, nspin).values())),
            )
        )
    )
    return PeriodicNBOAnalysis(
        job=str(job_path),
        nspin=nspin,
        gamma_only=gamma_only,
        kpoint_count=len(kpoints),
        basis_functions=nbas,
        total_electrons=total_electrons,
        occupancy_convention=(
            "ABACUS k-weighted LCAO density; periodic Fourier blocks use the "
            "reference exp(-2*pi*i k.R) convention"
        ),
        atoms=atom_entries,
        natural_orbitals=natural_orbitals,
        wavefunctions=tuple(wavefunctions),
        donor_acceptor=tuple(donor_acceptor),
        data_files=data_files,
        warnings=tuple(warnings),
        population_method="NPA",
        occupancy_method=(
            "periodic pre-NAO + one-/two-centre NHO projection + Rydberg completion"
            + ("; spin-resolved NBO search" if nspin == 2 else "")
        ),
    )


__all__ = [
    "AtomPopulation",
    "DonorAcceptorInteraction",
    "NBOComponent",
    "NBOShell",
    "NBOWavefunction",
    "NaturalOrbital",
    "PeriodicNBOAnalysis",
    "PeriodicNBOError",
    "PeriodicNBOOptions",
    "analyze_periodic_nbo",
    "kpoint_matrices_to_real_space",
]
