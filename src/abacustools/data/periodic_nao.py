"""Periodic natural atomic orbitals following the original NBO algorithm.

This module implements the five stages of ``pre_nao.f90`` from the original
periodic NBO implementation.  The input is a set of ABACUS k-resolved LCAO
density and overlap matrices.  The output contains the k-dependent AO-to-NAO
transformations, NPA populations and real-space density blocks in the final
orthonormal NAO basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from abacustools.data.charge import element_numbers
from abacustools.data.nbo import NBOShell
from abacustools.data.population import KPointDensity
from abacustools.io.stru import AbacusSTRU


W_THRESHOLD = 1.0e-2


@dataclass(frozen=True)
class PeriodicNAOResult:
    """Result of the original periodic pre-NAO construction."""

    transformations: tuple[np.ndarray, ...]
    kpoint_overlaps: tuple[np.ndarray, ...]
    occupancies: np.ndarray
    populations: np.ndarray
    charges: np.ndarray
    density_blocks: dict[tuple[int, int, int], np.ndarray]
    fock_blocks: dict[tuple[int, int, int], np.ndarray] | None
    overlap_at_origin: np.ndarray
    basis_l: np.ndarray
    basis_shell: np.ndarray
    nmb_mask: np.ndarray
    cusp: str = "periodic-pre-nao"


def _inverse_sqrt_hermitian(matrix: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) <= 1.0e-10 * scale:
        raise ValueError("overlap matrix is singular in pre-NAO construction")
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T


def _generalized_eigh(
    density: np.ndarray,
    overlap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    density = 0.5 * (density + density.conj().T)
    overlap = 0.5 * (overlap + overlap.conj().T)
    inverse = _inverse_sqrt_hermitian(overlap)
    standard = inverse.conj().T @ density @ inverse
    values, vectors = np.linalg.eigh(0.5 * (standard + standard.conj().T))
    order = np.argsort(values)[::-1]
    return values[order], inverse @ vectors[:, order]


def _schmidt_orthog_matrix(overlap: np.ndarray, occupied: int) -> np.ndarray:
    size = overlap.shape[0]
    if occupied <= 0 or occupied >= size:
        return np.eye(size, dtype=overlap.dtype)
    leading = overlap[:occupied, :occupied]
    projection = np.linalg.solve(leading, overlap[:occupied, occupied:])
    raw = np.eye(size, dtype=overlap.dtype)
    raw[:occupied, occupied:] = -projection
    return raw.conj().T


def _symmetric_orthog_matrix(overlap: np.ndarray, weights: np.ndarray) -> np.ndarray:
    diagonal = np.diag(np.asarray(weights, dtype=float))
    weighted = diagonal @ overlap @ diagonal
    return diagonal @ _inverse_sqrt_hermitian(weighted)


def _partition_basis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_indices = np.flatnonzero(mask)
    false_indices = np.flatnonzero(~mask)
    mapping = np.concatenate((true_indices, false_indices))
    inverse = np.empty_like(mapping)
    inverse[mapping] = np.arange(mapping.size)
    return mapping, inverse


def _remap_matrix(matrix: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return matrix[np.ix_(mapping, mapping)]


def _remap_matrix_back(matrix: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return matrix[np.ix_(mapping, mapping)]


def _build_basis_metadata(
    ranges: Sequence[tuple[int, int]],
    shells: Sequence[NBOShell],
) -> tuple[np.ndarray, np.ndarray]:
    basis_l = np.zeros(ranges[-1][1], dtype=int)
    basis_shell = np.zeros(ranges[-1][1], dtype=int)
    shell_counter = 0
    for shell in shells:
        shell_counter += 1
        for index in shell.indices:
            basis_l[index] = shell.l
            basis_shell[index] = shell_counter
    return basis_l, basis_shell


def _nmb_count(atomic_number: int, valence: float) -> int:
    if atomic_number >= 87:
        raise ValueError("periodic pre-NAO supports atomic numbers below 87")
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
    count -= int(round((atomic_number - valence) / 2.0))
    return max(1, count)


def _load_average_block(
    rho0: np.ndarray,
    s0: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    basis_l: np.ndarray,
    basis_shell: np.ndarray,
    atom: int,
    l: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    start, stop = ranges[atom]
    shell_starts: list[int] = []
    seen_shells: set[int] = set()
    for index in range(start, stop):
        if int(basis_l[index]) != l:
            continue
        shell = int(basis_shell[index])
        if shell not in seen_shells:
            seen_shells.add(shell)
            shell_starts.append(index)
    nshell = len(shell_starts)
    rho_block = np.zeros((nshell, nshell), dtype=float)
    s_block = np.zeros((nshell, nshell), dtype=float)
    width = 2 * l + 1
    for first, first_index in enumerate(shell_starts):
        for second, second_index in enumerate(shell_starts):
            for m in range(width):
                rho_block[first, second] += float(np.real(rho0[first_index + m, second_index + m]))
                s_block[first, second] += float(np.real(s0[first_index + m, second_index + m]))
    rho_block /= width
    s_block /= width
    return rho_block, s_block, shell_starts


def _pre_nao_step2(
    rho0: np.ndarray,
    s0: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    basis_l: np.ndarray,
    basis_shell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nbas = rho0.shape[0]
    transform = np.zeros((nbas, nbas), dtype=float)
    weights = np.zeros(nbas, dtype=float)
    for atom, (start, stop) in enumerate(ranges):
        for l in range(int(np.max(basis_l[start:stop])) + 1):
            rho_block, s_block, shell_starts = _load_average_block(
                rho0, s0, ranges, basis_l, basis_shell, atom, l
            )
            if not shell_starts:
                continue
            values, vectors = _generalized_eigh(rho_block, s_block)
            width = 2 * l + 1
            for source, source_start in enumerate(shell_starts):
                for target, target_start in enumerate(shell_starts):
                    for m in range(width):
                        weights[source_start + m] = values[source]
                        transform[source_start + m, target_start + m] = vectors[source, target]
    return transform, weights


def _create_nmb_mask(
    weights: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    atomic_numbers: Sequence[int],
    valences: Sequence[float],
) -> np.ndarray:
    mask = np.zeros(weights.size, dtype=bool)
    for atom, (start, stop) in enumerate(ranges):
        count = min(_nmb_count(int(atomic_numbers[atom]), float(valences[atom])), stop - start)
        order = np.argsort(weights[start:stop])[::-1]
        mask[start + order[:count]] = True
    return mask


def _apply_to_kpoints(
    matrices: list[np.ndarray],
    transform: np.ndarray,
) -> None:
    for index, matrix in enumerate(matrices):
        matrices[index] = transform @ matrix @ transform.conj().T


def _transform_density(matrix: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return transform @ matrix @ transform.conj().T


def _real_space_block(
    kpoints: Sequence[KPointDensity],
    matrices: Sequence[np.ndarray],
    block: tuple[int, int, int],
) -> np.ndarray:
    total = np.zeros_like(matrices[0], dtype=np.complex128)
    for kpoint, matrix in zip(kpoints, matrices):
        phase = np.exp(
            2j
            * np.pi
            * (
                kpoint.direct[0] * block[0]
                + kpoint.direct[1] * block[1]
                + kpoint.direct[2] * block[2]
            )
        )
        total += kpoint.weight * phase * matrix
    return total


def build_periodic_nao(
    structure: AbacusSTRU,
    ranges: Sequence[tuple[int, int]],
    shells: Sequence[NBOShell],
    kpoints: Sequence[KPointDensity],
    *,
    r_shells: int = 1,
    fock_matrices: Sequence[np.ndarray] | None = None,
    valences: Sequence[float] | None = None,
) -> PeriodicNAOResult:
    """Construct periodic NAOs and NPA populations using the original steps."""

    if not kpoints:
        raise ValueError("periodic pre-NAO requires at least one k-point")
    nbas = kpoints[0].overlap.shape[0]
    atomic_numbers = element_numbers([atom.element or atom.label for atom in structure.atoms])
    if valences is None:
        raise ValueError("valences are required for periodic pre-NAO construction")
    atom_valences = np.asarray(valences, dtype=float)
    basis_l, basis_shell = _build_basis_metadata(ranges, shells)

    density_k: list[np.ndarray] = []
    overlap_k: list[np.ndarray] = []
    for kpoint in kpoints:
        total_density = sum(kpoint.densities.values())
        overlap = np.asarray(kpoint.overlap, dtype=np.complex128)
        density_k.append(overlap @ total_density @ overlap)
        overlap_k.append(overlap)
    transformations = [np.eye(nbas, dtype=np.complex128) for _ in kpoints]

    rho_origin = _real_space_block(kpoints, density_k, (0, 0, 0)).real
    s_origin = _real_space_block(kpoints, overlap_k, (0, 0, 0)).real
    rho_origin = 0.5 * (rho_origin + rho_origin.T)
    s_origin = 0.5 * (s_origin + s_origin.T)

    transform, weights = _pre_nao_step2(rho_origin, s_origin, ranges, basis_l, basis_shell)
    for index, matrix in enumerate(density_k):
        density_k[index] = _transform_density(matrix, transform.astype(np.complex128))
    _apply_to_kpoints(overlap_k, transform.astype(np.complex128))
    for index, matrix in enumerate(transformations):
        transformations[index] = matrix @ transform

    nmb_mask = _create_nmb_mask(weights, ranges, atomic_numbers, atom_valences)
    ordering, inverse = _partition_basis(nmb_mask)
    weights = weights[ordering]
    nvalence = int(np.count_nonzero(nmb_mask))
    for kpoint_index, (density, overlap) in enumerate(zip(density_k, overlap_k)):
        density_r = _remap_matrix(density, ordering)
        overlap_r = _remap_matrix(overlap, ordering)
        os_matrix = _schmidt_orthog_matrix(overlap_r, nvalence)
        density_r = _transform_density(density_r, os_matrix)
        overlap_r = os_matrix @ overlap_r @ os_matrix.T
        global_transform = np.eye(nbas, dtype=np.complex128)
        global_transform[np.ix_(ordering, ordering)] = os_matrix
        transformations[kpoint_index] = transformations[kpoint_index] @ global_transform
        density_k[kpoint_index] = _remap_matrix_back(density_r, inverse)
        overlap_k[kpoint_index] = _remap_matrix_back(overlap_r, inverse)
    weights = weights[inverse]

    nmb_mask = _create_nmb_mask(weights, ranges, atomic_numbers, atom_valences)
    ordering, inverse = _partition_basis(nmb_mask)
    weights = weights[ordering]
    nvalence = int(np.count_nonzero(nmb_mask))
    rydberg_mask = weights > W_THRESHOLD
    rydberg_mask[:nvalence] = True
    rydberg_ordering, rydberg_inverse = _partition_basis(rydberg_mask)
    weights = weights[rydberg_ordering]
    nhiryd = int(np.count_nonzero(rydberg_mask))
    weights[nhiryd:] = 1.0
    for kpoint_index, (density, overlap) in enumerate(zip(density_k, overlap_k)):
        density_r = _remap_matrix(density, ordering)
        overlap_r = _remap_matrix(overlap, ordering)
        density_r = _remap_matrix(density_r, rydberg_ordering)
        overlap_r = _remap_matrix(overlap_r, rydberg_ordering)
        transform_r = np.eye(nbas, dtype=np.complex128)

        ow = _symmetric_orthog_matrix(overlap_r[:nvalence, :nvalence], weights[:nvalence])
        block = np.eye(nbas, dtype=np.complex128)
        block[:nvalence, :nvalence] = ow
        transform_r = transform_r @ block
        density_r = _transform_density(density_r, block)
        overlap_r = block.T @ overlap_r @ block

        ow_high = _symmetric_orthog_matrix(
            overlap_r[nvalence:nhiryd, nvalence:nhiryd],
            weights[nvalence:nhiryd],
        )
        block = np.eye(nbas, dtype=np.complex128)
        block[nvalence:nhiryd, nvalence:nhiryd] = ow_high
        transform_r = transform_r @ block
        density_r = _transform_density(density_r, block)
        overlap_r = block.T @ overlap_r @ block

        os_low = _schmidt_orthog_matrix(overlap_r, nhiryd)
        transform_r = transform_r @ os_low
        density_r = _transform_density(density_r, os_low)
        overlap_r = os_low.T @ overlap_r @ os_low

        ow_low = _symmetric_orthog_matrix(overlap_r[nhiryd:, nhiryd:], weights[nhiryd:])
        block = np.eye(nbas, dtype=np.complex128)
        block[nhiryd:, nhiryd:] = ow_low
        transform_r = transform_r @ block
        density_r = _transform_density(density_r, block)
        overlap_r = block.T @ overlap_r @ block

        density_r = _remap_matrix_back(density_r, rydberg_inverse)
        overlap_r = _remap_matrix_back(overlap_r, rydberg_inverse)
        density_k[kpoint_index] = _remap_matrix_back(density_r, inverse)
        overlap_k[kpoint_index] = _remap_matrix_back(overlap_r, inverse)
        transform_global = np.eye(nbas, dtype=np.complex128)
        transform_global[np.ix_(ordering, ordering)] = transform_r
        transformations[kpoint_index] = transformations[kpoint_index] @ transform_global
    weights = weights[rydberg_inverse]
    weights = weights[inverse]

    for index in range(len(overlap_k)):
        overlap_k[index] = np.eye(nbas, dtype=np.complex128)
    rho_origin = _real_space_block(kpoints, density_k, (0, 0, 0)).real
    rho_origin = 0.5 * (rho_origin + rho_origin.T)
    transform2, weights2 = _pre_nao_step2(
        rho_origin,
        np.eye(nbas),
        ranges,
        basis_l,
        basis_shell,
    )
    for index, matrix in enumerate(density_k):
        density_k[index] = _transform_density(matrix, transform2.astype(np.complex128))
    for index, matrix in enumerate(transformations):
        transformations[index] = matrix @ transform2

    blocks = list(product(range(-r_shells, r_shells + 1), repeat=3))
    density_blocks = {block: _real_space_block(kpoints, density_k, block) for block in blocks}
    fock_blocks = None
    if fock_matrices is not None:
        fock_k = [matrix.astype(np.complex128) for matrix in fock_matrices]
        for kpoint_index, matrix in enumerate(fock_k):
            transform_k = transformations[kpoint_index]
            fock_k[kpoint_index] = transform_k.T @ matrix @ transform_k
        fock_blocks = {block: _real_space_block(kpoints, fock_k, block) for block in blocks}

    rho_origin = _real_space_block(kpoints, density_k, (0, 0, 0)).real
    rho_origin = 0.5 * (rho_origin + rho_origin.T)
    occupancies = np.diag(rho_origin).real
    populations = np.asarray(
        [float(np.trace(rho_origin[start:stop, start:stop])) for start, stop in ranges]
    )
    charges = atom_valences - populations
    return PeriodicNAOResult(
        transformations=tuple(transformations),
        kpoint_overlaps=tuple(overlap_k),
        occupancies=occupancies,
        populations=populations,
        charges=charges,
        density_blocks=density_blocks,
        fock_blocks=fock_blocks,
        overlap_at_origin=np.eye(nbas),
        basis_l=basis_l,
        basis_shell=basis_shell,
        nmb_mask=_create_nmb_mask(weights2, ranges, atomic_numbers, atom_valences),
    )


__all__ = ["PeriodicNAOResult", "build_periodic_nao"]
