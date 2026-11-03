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
from dataclasses import asdict, dataclass, is_dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from abacustools.core.constant import RY_TO_EV
from abacustools.data.charge import element_numbers, valence_electrons
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
    analyze_population,
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
    population_method: str = "mulliken"
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


def _bond_occupancies(bond_order: float) -> tuple[float, float]:
    """Return the electron occupations of one bonding/antibonding pair.

    One Wiberg bond represents one bonding pair with a total of two electrons.
    A bond order above one therefore saturates the bonding NBO and leaves the
    antibonding NBO empty. A multiple bond needs more than one bonding pair;
    the electron-count check at the analysis level reports when this simple
    topology cannot represent it.

    Args:
        bond_order: Periodic Wiberg bond order.

    Returns:
        Occupations of the bonding and antibonding orbitals.
    """

    order = max(0.0, float(bond_order))
    bond = min(2.0, 2.0 * min(1.0, order))
    return bond, max(0.0, 2.0 - bond)


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
    """Build NBOs from periodic Wiberg bond orders and pre-NAO hybrids.

    The pre-NAO natural orbitals supply the atomic hybrid space. Each periodic
    Wiberg bond determines a two-centre reduced density block whose leading
    natural orbital is used as the bond shape. Wiberg order controls the
    bonding/antibonding topology, while occupations use an explicit
    two-electron budget per bond and a valence-based lone-pair count. This is
    an approximate ABACUS-native construction, not a standard NBO deletion
    analysis.
    """

    nbas = natural.nbas
    blocks = {key: np.asarray(value, dtype=np.complex128) for key, value in density_blocks.items()}
    overlaps = {
        key: np.asarray(value, dtype=np.complex128) for key, value in overlap_blocks.items()
    }
    wavefunctions: list[NBOWavefunction] = []
    donors: list[NBOWavefunction] = []
    acceptors: list[NBOWavefunction] = []
    used: dict[tuple[int, tuple[int, int, int]], list[np.ndarray]] = {}

    minimal: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for atom, (start, stop) in enumerate(ranges):
        count = _natural_minimum_basis_count(
            int(atomic_numbers[atom]),
            int(round(float(atom_populations[atom].valence_electrons))),
        )
        count = min(count, stop - start)
        order = np.argsort(natural.pre_occupancies[start:stop])[::-1][:count]
        coefficients = natural.pre_transform[start:stop, start:stop][:, order]
        weights = natural.pre_occupancies[start:stop][order]
        minimal[atom] = (coefficients, weights)

    incident_orders: dict[int, float] = {atom: 0.0 for atom in range(structure.natoms)}
    pair_data: list[tuple[int, int, tuple[int, int, int], float, float]] = []
    for bond in bond_orders:
        first = bond.atom1 - 1
        second = bond.atom2 - 1
        image, distance = _best_image(structure, first, second, options.r_shells)
        incident_orders[first] += float(bond.value)
        incident_orders[second] += float(bond.value)
        pair_data.append((first, second, image, distance, float(bond.value)))

    coordination_labels: dict[int, str | None] = {}
    for atom in range(structure.natoms):
        nonbonding = max(
            0.0,
            float(atom_populations[atom].valence_electrons) - incident_orders[atom],
        )
        coordination = int(round(incident_orders[atom]))
        labels = {1: "s", 2: "sp", 3: "sp2", 4: "sp3"}
        coordination_labels[atom] = (
            labels.get(coordination) if nonbonding <= 0.2 and coordination in labels else None
        )

    for first, second, image, distance, bond_order in pair_data:
        first_basis, first_weights = minimal[first]
        second_basis, second_weights = minimal[second]
        first_count = first_basis.shape[1]
        second_count = second_basis.shape[1]
        first_range = ranges[first]
        second_range = ranges[second]
        zero_overlap = overlaps[(0, 0, 0)]
        pair_overlap = np.block(
            [
                [
                    zero_overlap[
                        first_range[0] : first_range[1],
                        first_range[0] : first_range[1],
                    ],
                    overlaps[image][
                        first_range[0] : first_range[1],
                        second_range[0] : second_range[1],
                    ],
                ],
                [
                    overlaps[tuple(-value for value in image)][
                        second_range[0] : second_range[1],
                        first_range[0] : first_range[1],
                    ],
                    zero_overlap[
                        second_range[0] : second_range[1],
                        second_range[0] : second_range[1],
                    ],
                ],
            ]
        )
        zero_density = blocks[(0, 0, 0)]
        pair_density = np.block(
            [
                [
                    zero_density[
                        first_range[0] : first_range[1],
                        first_range[0] : first_range[1],
                    ],
                    blocks[image][
                        first_range[0] : first_range[1],
                        second_range[0] : second_range[1],
                    ],
                ],
                [
                    blocks[tuple(-value for value in image)][
                        second_range[0] : second_range[1],
                        first_range[0] : first_range[1],
                    ],
                    zero_density[
                        second_range[0] : second_range[1],
                        second_range[0] : second_range[1],
                    ],
                ],
            ]
        )
        transform = np.zeros(
            (
                first_range[1] - first_range[0] + second_range[1] - second_range[0],
                first_count + second_count,
            ),
            dtype=np.complex128,
        )
        transform[: first_range[1] - first_range[0], :first_count] = first_basis
        transform[first_range[1] - first_range[0] :, first_count:] = second_basis
        reduced_overlap = transform.conj().T @ pair_overlap @ transform
        reduced_density = transform.conj().T @ pair_density @ transform
        values, vectors = np.linalg.eigh(0.5 * (reduced_overlap + reduced_overlap.conj().T))
        inverse = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
        standard = inverse.conj().T @ reduced_density @ inverse
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (standard + standard.conj().T))
        order = np.argsort(eigenvalues)[::-1]
        leading_reduced = inverse @ eigenvectors[:, order[0]]
        trailing_reduced = inverse @ eigenvectors[:, order[-1]]
        leading_full = np.real_if_close(transform @ leading_reduced, tol=1000)
        trailing_full = np.real_if_close(transform @ trailing_reduced, tol=1000)
        first_size = first_range[1] - first_range[0]
        first_vector = leading_full[:first_size]
        second_vector = leading_full[first_size:]
        first_vector = _normalise_atomic_component(
            np.pad(first_vector, (first_range[0], nbas - first_range[1])),
            first,
            ranges,
            zero_overlap,
        )
        second_vector = _normalise_atomic_component(
            np.pad(second_vector, (second_range[0], nbas - second_range[1])),
            second,
            ranges,
            zero_overlap,
        )
        occupancy, anti_occupancy = _bond_occupancies(bond_order)
        anti_values = trailing_full
        anti_first = anti_values[:first_size]
        anti_second = anti_values[first_size:]
        anti_first = _normalise_atomic_component(
            np.pad(anti_first, (first_range[0], nbas - first_range[1])),
            first,
            ranges,
            zero_overlap,
        )
        anti_second = _normalise_atomic_component(
            np.pad(anti_second, (second_range[0], nbas - second_range[1])),
            second,
            ranges,
            zero_overlap,
        )
        bond = NBOWavefunction(
            index=len(wavefunctions) + 1,
            kind="bond",
            occupancy=occupancy,
            centers=((first, (0, 0, 0)), (second, image)),
            components=(
                NBOComponent(
                    first,
                    (0, 0, 0),
                    tuple(complex(x) for x in first_vector[first_range[0] : first_range[1]]),
                ),
                NBOComponent(
                    second,
                    image,
                    tuple(complex(x) for x in second_vector[second_range[0] : second_range[1]]),
                ),
            ),
            distance=distance,
            hybridizations=(
                coordination_labels.get(first)
                or _hybridization(
                    first_vector[first_range[0] : first_range[1]],
                    first,
                    shells,
                    first_range[0],
                ),
                coordination_labels.get(second)
                or _hybridization(
                    second_vector[second_range[0] : second_range[1]],
                    second,
                    shells,
                    second_range[0],
                ),
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
                    tuple(complex(x) for x in anti_first[first_range[0] : first_range[1]]),
                ),
                NBOComponent(
                    second,
                    image,
                    tuple(complex(x) for x in anti_second[second_range[0] : second_range[1]]),
                ),
            ),
            distance=distance,
            hybridizations=(
                coordination_labels.get(first)
                or _hybridization(
                    anti_first[first_range[0] : first_range[1]],
                    first,
                    shells,
                    first_range[0],
                ),
                coordination_labels.get(second)
                or _hybridization(
                    anti_second[second_range[0] : second_range[1]],
                    second,
                    shells,
                    second_range[0],
                ),
            ),
            paired_index=bond.index,
        )
        wavefunctions.append(anti)
        acceptors.append(anti)

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
    total_electrons = sum(float(item.population) for item in atom_populations)
    bond_electrons = sum(sum(_bond_occupancies(bond_order)) for *_rest, bond_order in pair_data)
    remaining_electrons = max(0.0, total_electrons - bond_electrons)
    lone_pair_slots = sum(lone_pair_counts.values())
    lone_pair_occupancy = min(
        2.0,
        remaining_electrons / max(1, lone_pair_slots),
    )

    for atom, (start, stop) in enumerate(ranges):
        lone_pair_count = lone_pair_counts[atom]
        pre_order = np.argsort(natural.pre_occupancies[start:stop])[::-1]
        for column in pre_order[:lone_pair_count]:
            vector = np.asarray(
                natural.pre_transform[start:stop, start + column],
                dtype=np.complex128,
            )
            vector = _normalise_atomic_component(
                np.pad(vector, (start, nbas - stop)),
                atom,
                ranges,
                overlap_blocks[(0, 0, 0)].real,
            )
            component = NBOComponent(
                atom,
                (0, 0, 0),
                tuple(complex(x) for x in vector[start:stop]),
            )
            orbital = NBOWavefunction(
                index=len(wavefunctions) + 1,
                kind="lone_pair",
                occupancy=float(lone_pair_occupancy),
                centers=((atom, (0, 0, 0)),),
                components=(component,),
                hybridizations=(_hybridization(vector[start:stop], atom, shells, start),),
            )
            wavefunctions.append(orbital)
            donors.append(orbital)

    for atom, (start, stop) in enumerate(ranges):
        added = 0
        order = np.argsort(natural.pre_occupancies[start:stop])[::-1]
        for offset in order:
            occupancy = float(natural.pre_occupancies[start + offset])
            if occupancy < options.rydberg_threshold:
                continue
            if occupancy >= options.one_center_threshold:
                continue
            vector = np.asarray(
                natural.pre_transform[start:stop, start + offset],
                dtype=np.complex128,
            )
            vector = _normalise_atomic_component(
                np.pad(vector, (start, nbas - stop)),
                atom,
                ranges,
                overlap_blocks[(0, 0, 0)].real,
            )
            component = NBOComponent(
                atom,
                (0, 0, 0),
                tuple(complex(x) for x in vector[start:stop]),
            )
            orbital = NBOWavefunction(
                index=len(wavefunctions) + 1,
                kind="rydberg",
                occupancy=0.0,
                centers=((atom, (0, 0, 0)),),
                components=(component,),
                hybridizations=(_hybridization(vector[start:stop], atom, shells, start),),
            )
            wavefunctions.append(orbital)
            acceptors.append(orbital)
            added += 1
            if added >= options.maximum_rydberg_per_atom:
                break

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
            block = fock_blocks.get(relative)
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


def analyze_periodic_nbo(
    job: str | Path,
    *,
    options: Optional[PeriodicNBOOptions] = None,
) -> PeriodicNBOAnalysis:
    """Run a periodic NBO/NPA analysis from an ABACUS LCAO job.

    Args:
        job: ABACUS job directory.
        options: Thresholds and neighbour-shell settings.

    Returns:
        Populations, NAOs, NBOs, antibonds, Rydberg orbitals and optional
        periodic donor-acceptor interactions.

    Raises:
        PeriodicNBOError: If the job is not LCAO, required matrices are
            missing, or the natural-orbital basis is singular.
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
    density_scale, density_weighted, convention, total_electrons = _density_normalisation(
        kpoints, expected
    )
    for kpoint in kpoints:
        if kpoint.overlap.shape != (nbas, nbas):
            raise PeriodicNBOError(
                f"k-point {kpoint.index}: overlap dimension "
                f"{kpoint.overlap.shape} does not match basis size {nbas}"
            )

    r_blocks = list(product(range(-options.r_shells, options.r_shells + 1), repeat=3))
    total_kpoints = [
        (
            kpoint.direct,
            kpoint.weight,
            density_scale * sum(kpoint.densities.values()),
        )
        for kpoint in kpoints
    ]
    density_blocks = kpoint_matrices_to_real_space(
        total_kpoints,
        r_blocks,
        apply_weight=density_weighted,
    )
    overlap_kpoints = [(kpoint.direct, kpoint.weight, kpoint.overlap) for kpoint in kpoints]
    overlap_blocks = kpoint_matrices_to_real_space(overlap_kpoints, r_blocks)
    d0 = np.asarray(density_blocks[(0, 0, 0)].real, dtype=float)
    s0 = np.asarray(overlap_blocks[(0, 0, 0)].real, dtype=float)
    d0 = 0.5 * (d0 + d0.T)
    s0 = 0.5 * (s0 + s0.T)

    valence = valence_electrons(structure, inputs.get("pseudo_dir"), job_path)
    atomic_numbers = element_numbers([atom.element or atom.label for atom in structure.atoms])
    natural = _natural_basis(
        structure,
        shells,
        ranges,
        d0,
        s0,
        atomic_numbers,
        valence,
    )
    population = analyze_population(
        job_path,
        method="wiberg",
        cutoff=options.cutoff,
        spin="total",
    )
    atomic_population = analyze_population(
        job_path,
        method="mulliken",
        cutoff=options.cutoff,
        spin="total",
    )
    atom_entries = [
        AtomPopulation(
            index=atom.index,
            element=atom.element,
            population=atom.population,
            charge=float(atom.charge or 0.0),
            valence_electrons=float(valence[atom.index - 1]),
        )
        for atom in atomic_population.atoms
    ]

    search = _search_nbos_from_population(
        structure,
        ranges,
        shells,
        natural,
        density_blocks,
        overlap_blocks,
        population.bonds,
        atom_entries,
        atomic_numbers,
        options,
    )

    donor_acceptor: list[DonorAcceptorInteraction] = []
    warnings: list[str] = []
    assigned_electrons = sum(item.occupancy for item in search.wavefunctions)
    if abs(assigned_electrons - total_electrons) > 1.0e-4 * max(1.0, total_electrons):
        warnings.append(
            "the occupancy assignment does not reproduce the total electron count "
            f"({assigned_electrons:.8f} vs {total_electrons:.8f}); the Wiberg "
            "topology probably contains a multiple bond not represented by a "
            "single bond/antibond pair"
        )
    if any(
        item.occupancy < -1.0e-8 or item.occupancy > 2.0 + 1.0e-8 for item in search.wavefunctions
    ):
        warnings.append(
            "one or more localized orbitals have an unphysical occupancy outside [0, 2]"
        )
    if options.donor_acceptor and nspin == 1:
        fock_blocks = _load_raw_fock_blocks(output, kpoints, nspin, nbas, r_blocks)
        if fock_blocks is None:
            warnings.append(
                "Fock/Hamiltonian matrices were not available; "
                "donor-acceptor interactions were skipped"
            )
        else:
            donor_acceptor = _donor_acceptor_interactions(
                search.donors,
                search.acceptors,
                fock_blocks,
                ranges,
                options,
            )
    elif options.donor_acceptor:
        warnings.append(
            "spin-resolved donor-acceptor analysis is not implemented; "
            "interactions were skipped for nspin=2"
        )

    natural_orbitals = tuple(
        NaturalOrbital(
            index=index + 1,
            atom=int(natural.atoms[index]) + 1,
            element=str(
                structure.atoms[int(natural.atoms[index])].element
                or structure.atoms[int(natural.atoms[index])].label
            ),
            occupancy=float(natural.pre_occupancies[index]),
            direction=natural.directions[index],
        )
        for index in range(nbas)
    )
    data_files = tuple(dict.fromkeys((*atomic_population.data_files, *population.data_files)))
    return PeriodicNBOAnalysis(
        job=str(job_path),
        nspin=nspin,
        gamma_only=gamma_only,
        kpoint_count=len(kpoints),
        basis_functions=nbas,
        total_electrons=total_electrons,
        occupancy_convention=convention,
        atoms=tuple(atom_entries),
        natural_orbitals=natural_orbitals,
        wavefunctions=tuple(search.wavefunctions),
        donor_acceptor=tuple(donor_acceptor),
        data_files=data_files,
        warnings=tuple(warnings),
        population_method="mulliken",
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
