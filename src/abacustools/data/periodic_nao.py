"""Periodic natural atomic orbitals following the original NBO algorithm.

This module implements the pre-NAO construction of Dunnington and Schmidt,
J. Chem. Theory Comput. 8, 1902 (2012), directly from ABACUS LCAO matrices.
No plane-wave projection is required because ABACUS already supplies the
k-resolved density and overlap matrices in the numerical-orbital basis.

All matrix transformations use one convention. If ``coefficients`` maps the
input AO basis to a new basis,

    S' = coefficients^H S coefficients,
    rho' = coefficients^H rho coefficients,
    F' = coefficients^H F coefficients.

The cumulative ``transformations`` returned by :func:`build_periodic_nao`
are these coefficient matrices, so that the final overlap is the identity.
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
    spin_density_blocks: dict[int, dict[tuple[int, int, int], np.ndarray]] | None
    overlap_at_origin: np.ndarray
    basis_l: np.ndarray
    basis_shell: np.ndarray
    nmb_mask: np.ndarray
    weights: np.ndarray
    cusp: str = "periodic-pre-nao"


def _congruence(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Transform a covariant matrix with coefficient matrix ``coefficients``."""

    return coefficients.conj().T @ matrix @ coefficients


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
    """Solve ``density c = overlap c w`` with overlap-normalized vectors."""

    density = 0.5 * (density + density.conj().T)
    overlap = 0.5 * (overlap + overlap.conj().T)
    inverse = _inverse_sqrt_hermitian(overlap)
    standard = inverse.conj().T @ density @ inverse
    values, vectors = np.linalg.eigh(0.5 * (standard + standard.conj().T))
    order = np.argsort(values)[::-1]
    return values[order], inverse @ vectors[:, order]


def _schmidt_coefficient_matrix(overlap: np.ndarray, occupied: int) -> np.ndarray:
    """Return the coefficient transform for partial Gram-Schmidt."""

    size = overlap.shape[0]
    if occupied <= 0 or occupied >= size:
        return np.eye(size, dtype=overlap.dtype)
    leading = overlap[:occupied, :occupied]
    projection = np.linalg.solve(leading, overlap[:occupied, occupied:])
    coefficients = np.eye(size, dtype=overlap.dtype)
    coefficients[:occupied, occupied:] = -projection
    return coefficients


def _symmetric_orthog_coefficients(
    overlap: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the occupancy-weighted symmetric orthogonalizer.

    For ``M = W S W`` this is ``M^(-1/2) W``.  It is the transpose of the
    row transform ``W M^(-1/2)`` used in the reference Fortran code.
    """

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


def _permutation_matrix(mapping: np.ndarray) -> np.ndarray:
    matrix = np.zeros((mapping.size, mapping.size), dtype=float)
    matrix[np.arange(mapping.size), mapping] = 1.0
    return matrix


def _global_coefficient(
    transform: np.ndarray,
    outer_mapping: np.ndarray,
    inner_mapping: np.ndarray | None = None,
) -> np.ndarray:
    """Embed a subspace coefficient transform in the original basis."""

    outer = _permutation_matrix(outer_mapping)
    if inner_mapping is None:
        return outer.T @ transform @ outer
    inner = _permutation_matrix(inner_mapping)
    return outer.T @ inner.T @ transform @ inner @ outer


def _build_basis_metadata(
    ranges: Sequence[tuple[int, int]],
    shells: Sequence[NBOShell],
) -> tuple[np.ndarray, np.ndarray]:
    nbas = ranges[-1][1]
    basis_l = np.zeros(nbas, dtype=int)
    basis_shell = np.zeros(nbas, dtype=int)
    for shell_index, shell in enumerate(shells, start=1):
        for index in shell.indices:
            basis_l[index] = shell.l
            basis_shell[index] = shell_index
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
    """Construct the block-diagonal pre-NAO transform from atom/l blocks."""

    nbas = rho0.shape[0]
    coefficients = np.zeros((nbas, nbas), dtype=float)
    weights = np.zeros(nbas, dtype=float)
    for atom, (start, stop) in enumerate(ranges):
        maximum_l = int(np.max(basis_l[start:stop]))
        for l in range(maximum_l + 1):
            rho_block, s_block, shell_starts = _load_average_block(
                rho0,
                s0,
                ranges,
                basis_l,
                basis_shell,
                atom,
                l,
            )
            if not shell_starts:
                continue
            values, vectors = _generalized_eigh(rho_block, s_block)
            width = 2 * l + 1
            for source, source_start in enumerate(shell_starts):
                for target, target_start in enumerate(shell_starts):
                    for m in range(width):
                        weights[source_start + m] = values[source]
                        coefficients[source_start + m, target_start + m] = vectors[source, target]
    return coefficients, weights


def _create_nmb_mask(
    weights: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    atomic_numbers: Sequence[int],
    valences: Sequence[float],
) -> np.ndarray:
    """Select the natural minimal basis using the reference threshold rule."""

    mask = np.zeros(weights.size, dtype=bool)
    for atom, (start, stop) in enumerate(ranges):
        count = min(
            _nmb_count(int(atomic_numbers[atom]), float(valences[atom])),
            stop - start,
        )
        ordered = np.sort(weights[start:stop])[::-1]
        threshold = ordered[count - 1]
        mask[start:stop] = weights[start:stop] >= threshold
    return mask


def _real_space_block(
    kpoints: Sequence[KPointDensity],
    matrices: Sequence[np.ndarray],
    block: tuple[int, int, int],
    *,
    apply_weight: bool = True,
) -> np.ndarray:
    """Fourier-transform k-resolved matrices to one periodic G block."""

    total = np.zeros_like(matrices[0], dtype=np.complex128)
    for kpoint, matrix in zip(kpoints, matrices):
        phase = np.exp(
            -2j
            * np.pi
            * (
                kpoint.direct[0] * block[0]
                + kpoint.direct[1] * block[1]
                + kpoint.direct[2] * block[2]
            )
        )
        factor = kpoint.weight if apply_weight else 1.0
        total += factor * phase * matrix
    return total


def _normalise_density(
    kpoints: Sequence[KPointDensity],
    expected: float,
) -> tuple[float, bool]:
    """Find the ABACUS density weighting and spin/occupation scale."""

    candidates: list[tuple[float, float, bool]] = []
    for apply_weight in (False, True):
        for scale in (0.5, 1.0, 2.0):
            total = 0.0
            for kpoint in kpoints:
                factor = kpoint.weight if apply_weight else 1.0
                trace = sum(
                    float(np.real(np.trace(density @ kpoint.overlap)))
                    for density in kpoint.densities.values()
                )
                total += factor * scale * trace
            candidates.append((abs(total - expected), scale, apply_weight))
    error, scale, apply_weight = min(candidates)
    if error > 1.0e-4 * max(1.0, abs(expected)) + 1.0e-6:
        raise ValueError("cannot match the declared electron count with the LCAO density matrices")
    return scale, apply_weight


def _renormalize_origin(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


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
    """Construct periodic NAOs and NPA populations from ABACUS LCAO data."""

    if not kpoints:
        raise ValueError("periodic pre-NAO requires at least one k-point")
    if valences is None:
        raise ValueError("valences are required for periodic pre-NAO construction")
    if r_shells < 0:
        raise ValueError("r_shells must be non-negative")
    if fock_matrices is not None and len(fock_matrices) != len(kpoints):
        raise ValueError("fock_matrices and kpoints must have the same length")

    nbas = kpoints[0].overlap.shape[0]
    atomic_numbers = element_numbers([atom.element or atom.label for atom in structure.atoms])
    atom_valences = np.asarray(valences, dtype=float)
    density_scale, density_weighted = _normalise_density(kpoints, float(np.sum(atom_valences)))
    basis_l, basis_shell = _build_basis_metadata(ranges, shells)

    density_k: list[np.ndarray] = []
    overlap_k: list[np.ndarray] = []
    for kpoint in kpoints:
        overlap = np.asarray(kpoint.overlap, dtype=np.complex128)
        total_density = sum(
            density_scale * np.asarray(matrix, dtype=np.complex128)
            for matrix in kpoint.densities.values()
        )
        if overlap.shape != (nbas, nbas) or total_density.shape != (nbas, nbas):
            raise ValueError("all k-point matrices must have the same square shape")
        density_k.append(overlap @ total_density @ overlap)
        overlap_k.append(overlap)
    transformations = [np.eye(nbas, dtype=np.complex128) for _ in kpoints]

    rho_origin = _renormalize_origin(
        _real_space_block(kpoints, density_k, (0, 0, 0), apply_weight=density_weighted).real
    )
    s_origin = _renormalize_origin(_real_space_block(kpoints, overlap_k, (0, 0, 0)).real)
    pre_coefficients, weights = _pre_nao_step2(
        rho_origin,
        s_origin,
        ranges,
        basis_l,
        basis_shell,
    )
    pre_coefficients_c = pre_coefficients.astype(np.complex128)
    for index in range(len(kpoints)):
        density_k[index] = _congruence(density_k[index], pre_coefficients_c)
        overlap_k[index] = _congruence(overlap_k[index], pre_coefficients_c)
        transformations[index] = transformations[index] @ pre_coefficients_c

    nmb_mask = _create_nmb_mask(weights, ranges, atomic_numbers, atom_valences)
    ordering, inverse = _partition_basis(nmb_mask)
    nvalence = int(np.count_nonzero(nmb_mask))
    ordered_weights = weights[ordering]
    for index in range(len(kpoints)):
        density_ordered = density_k[index][np.ix_(ordering, ordering)]
        overlap_ordered = overlap_k[index][np.ix_(ordering, ordering)]
        schmidt = _schmidt_coefficient_matrix(overlap_ordered, nvalence)
        global_transform = _global_coefficient(schmidt, ordering)
        density_k[index] = _congruence(density_k[index], global_transform)
        overlap_k[index] = _congruence(overlap_k[index], global_transform)
        transformations[index] = transformations[index] @ global_transform
    weights = ordered_weights[inverse]

    nmb_mask = _create_nmb_mask(weights, ranges, atomic_numbers, atom_valences)
    ordering, inverse = _partition_basis(nmb_mask)
    nvalence = int(np.count_nonzero(nmb_mask))
    ordered_weights = weights[ordering]
    rydberg_mask = ordered_weights > W_THRESHOLD
    rydberg_mask[:nvalence] = True
    rydberg_ordering, rydberg_inverse = _partition_basis(rydberg_mask)
    occupied_weights = ordered_weights[rydberg_ordering].copy()
    nhiryd = int(np.count_nonzero(rydberg_mask))
    occupied_weights[nhiryd:] = 1.0

    for index in range(len(kpoints)):
        density_ordered = density_k[index][np.ix_(ordering, ordering)]
        overlap_ordered = overlap_k[index][np.ix_(ordering, ordering)]
        density_ryd = density_ordered[np.ix_(rydberg_ordering, rydberg_ordering)]
        overlap_ryd = overlap_ordered[np.ix_(rydberg_ordering, rydberg_ordering)]

        coefficient_steps: list[np.ndarray] = []
        nmb_transform = np.eye(nbas, dtype=np.complex128)
        nmb_transform[:nvalence, :nvalence] = _symmetric_orthog_coefficients(
            overlap_ryd[:nvalence, :nvalence],
            occupied_weights[:nvalence],
        )
        coefficient_steps.append(nmb_transform)
        density_ryd = _congruence(density_ryd, nmb_transform)
        overlap_ryd = _congruence(overlap_ryd, nmb_transform)

        high_transform = np.eye(nbas, dtype=np.complex128)
        if nhiryd > nvalence:
            high_transform[nvalence:nhiryd, nvalence:nhiryd] = _symmetric_orthog_coefficients(
                overlap_ryd[nvalence:nhiryd, nvalence:nhiryd],
                occupied_weights[nvalence:nhiryd],
            )
        coefficient_steps.append(high_transform)
        density_ryd = _congruence(density_ryd, high_transform)
        overlap_ryd = _congruence(overlap_ryd, high_transform)

        schmidt_low = _schmidt_coefficient_matrix(overlap_ryd, nhiryd)
        coefficient_steps.append(schmidt_low)
        density_ryd = _congruence(density_ryd, schmidt_low)
        overlap_ryd = _congruence(overlap_ryd, schmidt_low)

        low_transform = np.eye(nbas, dtype=np.complex128)
        if nhiryd < nbas:
            low_transform[nhiryd:, nhiryd:] = _symmetric_orthog_coefficients(
                overlap_ryd[nhiryd:, nhiryd:],
                occupied_weights[nhiryd:],
            )
        coefficient_steps.append(low_transform)
        density_ryd = _congruence(density_ryd, low_transform)
        overlap_ryd = _congruence(overlap_ryd, low_transform)

        ordered_transform = np.eye(nbas, dtype=np.complex128)
        for step in coefficient_steps:
            ordered_transform = ordered_transform @ step
        global_transform = _global_coefficient(
            ordered_transform,
            ordering,
            rydberg_ordering,
        )
        density_k[index] = _congruence(density_k[index], global_transform)
        overlap_k[index] = _congruence(overlap_k[index], global_transform)
        transformations[index] = transformations[index] @ global_transform
    weights = ordered_weights[rydberg_inverse]
    weights = weights[inverse]

    rho_origin = _renormalize_origin(
        _real_space_block(kpoints, density_k, (0, 0, 0), apply_weight=density_weighted).real
    )
    final_rotation, final_weights = _pre_nao_step2(
        rho_origin,
        np.eye(nbas),
        ranges,
        basis_l,
        basis_shell,
    )
    final_rotation_c = final_rotation.astype(np.complex128)
    for index in range(len(kpoints)):
        density_k[index] = _congruence(density_k[index], final_rotation_c)
        overlap_k[index] = _congruence(overlap_k[index], final_rotation_c)
        transformations[index] = transformations[index] @ final_rotation_c

    blocks = list(product(range(-r_shells, r_shells + 1), repeat=3))
    density_blocks = {
        block: _real_space_block(kpoints, density_k, block, apply_weight=density_weighted)
        for block in blocks
    }

    spin_density_blocks: dict[int, dict[tuple[int, int, int], np.ndarray]] | None = None
    spin_values = sorted(kpoints[0].densities)
    if len(spin_values) > 1:
        spin_density_blocks = {}
        for spin in spin_values:
            transformed_spin: list[np.ndarray] = []
            for kpoint, transform in zip(kpoints, transformations):
                overlap = np.asarray(kpoint.overlap, dtype=np.complex128)
                density = np.asarray(kpoint.densities[spin], dtype=np.complex128)
                transformed_spin.append(
                    _congruence(
                        overlap @ (density_scale * density) @ overlap,
                        transform,
                    )
                )
            spin_density_blocks[spin] = {
                block: _real_space_block(
                    kpoints, transformed_spin, block, apply_weight=density_weighted
                )
                for block in blocks
            }

    fock_blocks = None
    if fock_matrices is not None:
        transformed_fock = [
            _congruence(
                np.asarray(matrix, dtype=np.complex128),
                transformations[index],
            )
            for index, matrix in enumerate(fock_matrices)
        ]
        fock_blocks = {
            block: _real_space_block(kpoints, transformed_fock, block) for block in blocks
        }

    rho_origin = _renormalize_origin(
        _real_space_block(kpoints, density_k, (0, 0, 0), apply_weight=density_weighted).real
    )
    occupancies = np.real(np.diag(rho_origin))
    populations = np.asarray(
        [float(np.real(np.trace(rho_origin[start:stop, start:stop]))) for start, stop in ranges]
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
        spin_density_blocks=spin_density_blocks,
        overlap_at_origin=np.eye(nbas),
        basis_l=basis_l,
        basis_shell=basis_shell,
        nmb_mask=_create_nmb_mask(
            final_weights,
            ranges,
            atomic_numbers,
            atom_valences,
        ),
        weights=final_weights,
    )


__all__ = ["PeriodicNAOResult", "build_periodic_nao"]
