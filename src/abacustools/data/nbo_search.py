"""Reference periodic NBO/NHO search in an orthogonal NAO basis.

The search follows ``nbo.f90`` from the periodic NBO reference implementation:
one-centre strongly occupied hybrids are located and removed, two-centre NHOs
are searched over the central cell and its nearest periodic images, and the
remaining one-centre density supplies Rydberg orbitals.  The routine assumes
the orthogonal natural-atomic-orbital basis produced by
:mod:`abacustools.data.periodic_nao`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from abacustools.data.charge import element_numbers
from abacustools.data.nbo import NBOShell, NBOWavefunction, NBOComponent
from abacustools.data.periodic_nao import PeriodicNAOResult
from abacustools.io.stru import AbacusSTRU


CORE_THRESHOLD = 0.995
POLARIZATION_THRESHOLD = 0.99
PROJECTION_THRESHOLD = 0.20


@dataclass(frozen=True)
class PeriodicNBOSearchOptions:
    """Thresholds of the original periodic NBO search.

    The values are closed-shell occupancies.  For a spin-polarized
    calculation the reference implementation applies half of each threshold
    to every spin channel.
    """

    one_center_threshold: float = 1.80
    two_center_threshold: float = 1.80
    cutoff: float = 3.0


@dataclass(frozen=True)
class _Hybrid:
    atom: int
    coefficients: np.ndarray
    occupancy: float


@dataclass(frozen=True)
class _BondCandidate:
    first: int
    second: int
    image: tuple[int, int, int]
    first_hybrid: int
    second_hybrid: int
    occupancy: float


@dataclass(frozen=True)
class _OrbitalRecord:
    kind: str
    occupancy: float
    centers: tuple[tuple[int, tuple[int, int, int]], ...]
    components: tuple[NBOComponent, ...]
    distance: float | None = None
    paired_index: int | None = None


@dataclass(frozen=True)
class PeriodicNBOSearchResult:
    """Complete original-algorithm NBO search result."""

    wavefunctions: tuple[NBOWavefunction, ...]
    density_blocks: dict[tuple[int, int, int], np.ndarray]
    lone_pairs: int
    bonds: int
    antibonds: int
    rydberg: int
    assigned_electrons: float


def _hermitian_real(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    hermitian = 0.5 * (array + array.conj().T)
    if float(np.max(np.abs(hermitian.imag))) > 1.0e-7:
        raise ValueError("periodic NBO search requires real real-space density blocks")
    return np.asarray(hermitian.real, dtype=float)


def _diagonalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(_hermitian_real(matrix))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def _projection_expectation(
    previous: Sequence[np.ndarray],
    proposed: np.ndarray,
) -> float:
    projector = np.eye(proposed.size)
    for hybrid in previous:
        projector = (np.eye(proposed.size) - np.outer(hybrid, hybrid)) @ projector
    return float(np.real(proposed.conj() @ projector @ proposed))


def _local_component(
    atom: int,
    image: tuple[int, int, int],
    coefficients: np.ndarray,
) -> NBOComponent:
    return NBOComponent(
        atom=atom,
        image=image,
        coefficients=tuple(complex(value) for value in coefficients),
    )


def _hybridization(
    coefficients: np.ndarray,
    atom: int,
    shells: Sequence[NBOShell],
    ranges: Sequence[tuple[int, int]],
) -> str:
    start, _stop = ranges[atom]
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
    return ", ".join(
        f"{labels[l] if l < len(labels) else f'l{l}'} {100.0 * value / norm:.1f}%"
        for l, value in sorted(totals.items())
        if value > 1.0e-8
    )


def _image_distance(
    structure: AbacusSTRU,
    first: int,
    second: int,
    image: tuple[int, int, int],
) -> float:
    lattice = np.asarray(structure.cell, dtype=float)
    coords = np.asarray(structure.coords_direct, dtype=float)
    delta = coords[second] + np.asarray(image, dtype=float) - coords[first]
    return float(np.linalg.norm(delta @ lattice))


def _pair_block(
    density: dict[tuple[int, int, int], np.ndarray],
    first: int,
    second: int,
    image: tuple[int, int, int],
    ranges: Sequence[tuple[int, int]],
) -> np.ndarray:
    i0, i1 = ranges[first]
    j0, j1 = ranges[second]
    central = density[(0, 0, 0)]
    if image == (0, 0, 0) and first == second:
        return central[i0:i1, i0:i1].copy()
    block = np.zeros((i1 - i0 + j1 - j0, i1 - i0 + j1 - j0), dtype=float)
    ioff = i1 - i0
    block[:ioff, :ioff] = central[i0:i1, i0:i1]
    block[ioff:, ioff:] = central[j0:j1, j0:j1]
    coupling = density[image][i0:i1, j0:j1]
    block[:ioff, ioff:] = coupling
    block[ioff:, :ioff] = coupling.T
    return 0.5 * (block + block.T)


def _write_pair_block(
    density: dict[tuple[int, int, int], np.ndarray],
    first: int,
    second: int,
    image: tuple[int, int, int],
    ranges: Sequence[tuple[int, int]],
    block: np.ndarray,
) -> None:
    i0, i1 = ranges[first]
    j0, j1 = ranges[second]
    ioff = i1 - i0
    central = density[(0, 0, 0)]
    central[i0:i1, i0:i1] = block[:ioff, :ioff]
    central[j0:j1, j0:j1] = block[ioff:, ioff:]
    density[image][i0:i1, j0:j1] = block[:ioff, ioff:]
    negative = tuple(-value for value in image)
    density[negative][j0:j1, i0:i1] = block[ioff:, :ioff]


def _search_one_center(
    density: dict[tuple[int, int, int], np.ndarray],
    ranges: Sequence[tuple[int, int]],
    cutoff: float,
    core_threshold: float,
    core_counts: Sequence[int],
) -> tuple[list[list[_Hybrid]], list[_OrbitalRecord]]:
    hybrids: list[list[_Hybrid]] = [[] for _ in ranges]
    records: list[_OrbitalRecord] = []
    central = density[(0, 0, 0)]
    projected_directions: list[tuple[int, int]] = []
    for atom, (start, stop) in enumerate(ranges):
        values, vectors = _diagonalize(central[start:stop, start:stop])
        found_core = 0
        for column, occupancy in enumerate(values):
            if occupancy <= cutoff:
                continue
            coefficients = vectors[:, column]
            hybrids[atom].append(_Hybrid(atom, coefficients, float(occupancy)))
            is_core = occupancy > core_threshold and found_core < core_counts[atom]
            if is_core:
                found_core += 1
            kind = "core" if is_core else "lone_pair"
            records.append(
                _OrbitalRecord(
                    kind=kind,
                    occupancy=float(occupancy),
                    centers=((atom, (0, 0, 0)),),
                    components=(_local_component(atom, (0, 0, 0), coefficients),),
                )
            )
            projected_directions.append((atom, len(hybrids[atom]) - 1))

    projector = np.eye(central.shape[0])
    for atom, hybrid_index in projected_directions:
        start, stop = ranges[atom]
        vector = np.zeros(central.shape[0])
        vector[start:stop] = hybrids[atom][hybrid_index].coefficients
        projector = (np.eye(central.shape[0]) - np.outer(vector, vector)) @ projector
    central[:] = projector @ central @ projector.T
    return hybrids, records


def _search_two_center_candidates(
    density: dict[tuple[int, int, int], np.ndarray],
    ranges: Sequence[tuple[int, int]],
    hybrids: list[list[_Hybrid]],
    cutoff: float,
    *,
    structure: AbacusSTRU,
    distance_cutoff: float,
) -> list[_BondCandidate]:
    candidates: list[_BondCandidate] = []
    for first, (i0, i1) in enumerate(ranges):
        isize = i1 - i0
        for image in product(range(-1, 2), repeat=3):
            for second, (j0, j1) in enumerate(ranges):
                if image == (0, 0, 0) and first >= second:
                    continue
                if _image_distance(structure, first, second, image) > distance_cutoff:
                    continue
                jsize = j1 - j0
                block = _pair_block(density, first, second, image, ranges)
                values, vectors = _diagonalize(block)
                for column, occupancy in enumerate(values):
                    if occupancy <= cutoff:
                        continue
                    polarization = float(np.sum(vectors[:isize, column] ** 2))
                    if max(polarization, 1.0 - polarization) > POLARIZATION_THRESHOLD:
                        continue
                    first_hybrid = vectors[:isize, column] / np.sqrt(polarization)
                    second_hybrid = vectors[isize:, column] / np.sqrt(1.0 - polarization)
                    if (
                        _projection_expectation(
                            [hybrid.coefficients for hybrid in hybrids[first]],
                            first_hybrid,
                        )
                        <= PROJECTION_THRESHOLD
                    ):
                        continue
                    if (
                        _projection_expectation(
                            [hybrid.coefficients for hybrid in hybrids[second]],
                            second_hybrid,
                        )
                        <= PROJECTION_THRESHOLD
                    ):
                        continue
                    if len(hybrids[first]) >= isize or len(hybrids[second]) >= jsize:
                        continue
                    hybrids[first].append(_Hybrid(first, first_hybrid, float(occupancy)))
                    hybrids[second].append(_Hybrid(second, second_hybrid, float(occupancy)))
                    candidates.append(
                        _BondCandidate(
                            first=first,
                            second=second,
                            image=image,
                            first_hybrid=len(hybrids[first]) - 1,
                            second_hybrid=len(hybrids[second]) - 1,
                            occupancy=float(occupancy),
                        )
                    )
    return candidates


def _symmetrically_orthogonalize_atomic_hybrids(
    hybrids: list[list[_Hybrid]],
) -> None:
    """Symmetric orthogonalization of every atom's complete NHO set."""

    for atom, atom_hybrids in enumerate(hybrids):
        if not atom_hybrids:
            continue
        vectors = np.column_stack([hybrid.coefficients for hybrid in atom_hybrids])
        overlap = vectors.T @ vectors
        values, eigenvectors = np.linalg.eigh(0.5 * (overlap + overlap.T))
        if np.min(values) <= 1.0e-12:
            raise ValueError(f"atom {atom + 1}: NHO overlap is singular")
        inverse = (eigenvectors * (1.0 / np.sqrt(values))) @ eigenvectors.T
        vectors = vectors @ inverse
        hybrids[atom] = [
            _Hybrid(atom, vectors[:, index], hybrid.occupancy)
            for index, hybrid in enumerate(atom_hybrids)
        ]


def _search_bonds(
    density: dict[tuple[int, int, int], np.ndarray],
    ranges: Sequence[tuple[int, int]],
    hybrids: Sequence[Sequence[_Hybrid]],
    candidates: Sequence[_BondCandidate],
    cutoff: float,
    *,
    structure: AbacusSTRU,
) -> list[_OrbitalRecord]:
    records: list[_OrbitalRecord] = []
    for candidate in candidates:
        first = candidate.first
        second = candidate.second
        image = candidate.image
        i0, i1 = ranges[first]
        j0, j1 = ranges[second]
        isize = i1 - i0
        first_hybrid = hybrids[first][candidate.first_hybrid].coefficients
        second_hybrid = hybrids[second][candidate.second_hybrid].coefficients
        transform = np.zeros((isize + j1 - j0, 2))
        transform[:isize, 0] = first_hybrid
        transform[isize:, 1] = second_hybrid
        block = _pair_block(density, first, second, image, ranges)
        pair_density = transform.T @ block @ transform
        pair_density = 0.5 * (pair_density + pair_density.T)
        values, vectors = _diagonalize(pair_density)
        if float(np.max(values)) <= cutoff:
            continue
        for column, occupancy in enumerate(values):
            if -1.0e-3 < occupancy < 0.0:
                occupancy = 0.0
            coefficients = transform @ vectors[:, column]
            first_component = coefficients[:isize]
            second_component = coefficients[isize:]
            polarization = float(np.sum(first_component**2))
            if max(polarization, 1.0 - polarization) > POLARIZATION_THRESHOLD:
                continue
            kind = "bond" if occupancy > cutoff else "antibond"
            records.append(
                _OrbitalRecord(
                    kind=kind,
                    occupancy=float(occupancy),
                    centers=((first, (0, 0, 0)), (second, image)),
                    components=(
                        _local_component(first, (0, 0, 0), first_component),
                        _local_component(second, image, second_component),
                    ),
                    distance=_image_distance(structure, first, second, image),
                )
            )
            local_projector = np.eye(coefficients.size) - np.outer(coefficients, coefficients)
            block = local_projector @ block @ local_projector.T
            _write_pair_block(density, first, second, image, ranges, block)
    return records


def _search_rydberg(
    density: dict[tuple[int, int, int], np.ndarray],
    ranges: Sequence[tuple[int, int]],
    hybrids: list[list[_Hybrid]],
) -> list[_OrbitalRecord]:
    records: list[_OrbitalRecord] = []
    central = density[(0, 0, 0)]
    for atom, (start, stop) in enumerate(ranges):
        values, vectors = _diagonalize(central[start:stop, start:stop])
        projected = len(hybrids[atom])
        redundant = np.argsort(np.abs(values))[:projected]
        keep = np.ones(values.size, dtype=bool)
        keep[redundant] = False
        for column in np.flatnonzero(keep):
            coefficients = vectors[:, column]
            hybrids[atom].append(_Hybrid(atom, coefficients, float(values[column])))
            records.append(
                _OrbitalRecord(
                    kind="rydberg",
                    occupancy=float(values[column]),
                    centers=((atom, (0, 0, 0)),),
                    components=(_local_component(atom, (0, 0, 0), coefficients),),
                )
            )
    return records


def _orthogonalize_atomic_hybrids(
    hybrids: list[list[_Hybrid]],
    initial_counts: Sequence[int],
) -> None:
    for atom, atom_hybrids in enumerate(hybrids):
        if not atom_hybrids:
            continue
        trial = np.column_stack([hybrid.coefficients for hybrid in atom_hybrids])
        overlap = trial.T @ trial
        occupied = int(initial_counts[atom])
        if occupied >= overlap.shape[0]:
            continue
        leading = overlap[:occupied, :occupied]
        projection = np.linalg.solve(leading, overlap[:occupied, occupied:])
        coefficients = np.eye(overlap.shape[0])
        coefficients[:occupied, occupied:] = -projection
        trial = trial @ coefficients
        low_overlap = trial[:, occupied:].T @ trial[:, occupied:]
        low_values, low_vectors = np.linalg.eigh(0.5 * (low_overlap + low_overlap.T))
        if np.min(low_values) <= 1.0e-12:
            continue
        inverse = (low_vectors * (1.0 / np.sqrt(low_values))) @ low_vectors.T
        trial[:, occupied:] = trial[:, occupied:] @ inverse
        hybrids[atom] = [
            _Hybrid(atom, trial[:, index], hybrid.occupancy)
            for index, hybrid in enumerate(atom_hybrids)
        ]


def search_periodic_nbos(
    structure: AbacusSTRU,
    ranges: Sequence[tuple[int, int]],
    shells: Sequence[NBOShell],
    nao: PeriodicNAOResult,
    *,
    nspin: int = 1,
    options: PeriodicNBOSearchOptions | None = None,
) -> PeriodicNBOSearchResult:
    """Run the original one-centre, two-centre and Rydberg searches."""

    if nspin not in (1, 2):
        raise ValueError("periodic NBO search supports nspin=1 or 2")
    options = options or PeriodicNBOSearchOptions()
    spin_factor = 1.0 if nspin == 1 else 0.5
    one_center_cutoff = options.one_center_threshold * spin_factor
    two_center_cutoff = options.two_center_threshold * spin_factor
    core_threshold = CORE_THRESHOLD + float(2 - nspin)

    density = {
        key: np.asarray(value.real, dtype=float).copy() for key, value in nao.density_blocks.items()
    }
    atomic_numbers = element_numbers([atom.element or atom.label for atom in structure.atoms])
    valences = np.asarray(nao.populations) + np.asarray(nao.charges)
    core_counts = [
        max(0, int(round((atomic_numbers[index] - valences[index]) / 2.0)))
        for index in range(len(ranges))
    ]
    hybrids, one_center_records = _search_one_center(
        density,
        ranges,
        one_center_cutoff,
        core_threshold,
        core_counts,
    )
    candidates = _search_two_center_candidates(
        density,
        ranges,
        hybrids,
        two_center_cutoff,
        structure=structure,
        distance_cutoff=options.cutoff,
    )
    _symmetrically_orthogonalize_atomic_hybrids(hybrids)
    bond_records = _search_bonds(
        density,
        ranges,
        hybrids,
        candidates,
        two_center_cutoff,
        structure=structure,
    )
    counts_before_rydberg = [len(atom_hybrids) for atom_hybrids in hybrids]
    rydberg_records = _search_rydberg(density, ranges, hybrids)
    _orthogonalize_atomic_hybrids(hybrids, counts_before_rydberg)

    records: list[_OrbitalRecord] = []
    records.extend(one_center_records)
    records.extend(bond_records)
    records.extend(rydberg_records)

    wavefunctions: list[NBOWavefunction] = []
    for record in records:
        hybridizations: list[str] = []
        for component in record.components:
            hybridizations.append(
                _hybridization(
                    np.asarray(component.coefficients),
                    component.atom,
                    shells,
                    ranges,
                )
            )
        wavefunctions.append(
            NBOWavefunction(
                index=len(wavefunctions) + 1,
                kind=record.kind,
                occupancy=record.occupancy,
                centers=record.centers,
                components=record.components,
                distance=record.distance,
                hybridizations=tuple(hybridizations),
                paired_index=record.paired_index,
            )
        )

    lone_pairs = sum(record.kind == "lone_pair" for record in records)
    bonds = sum(record.kind == "bond" for record in records)
    antibonds = sum(record.kind == "antibond" for record in records)
    rydberg = sum(record.kind == "rydberg" for record in records)
    return PeriodicNBOSearchResult(
        wavefunctions=tuple(wavefunctions),
        density_blocks=density,
        lone_pairs=int(lone_pairs) + sum(record.kind == "core" for record in records),
        bonds=int(bonds),
        antibonds=int(antibonds),
        rydberg=int(rydberg),
        assigned_electrons=float(sum(record.occupancy for record in records)),
    )


__all__ = [
    "PeriodicNBOSearchOptions",
    "PeriodicNBOSearchResult",
    "search_periodic_nbos",
]
