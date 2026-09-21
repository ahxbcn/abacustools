"""Symmetry handling of the piezoelectric tensor.

The piezoelectric tensor ``d_ijk = dP_i / d eps_jk`` is a polar third-rank
tensor, symmetric in its last two indices, so Voigt notation leaves 18
components in a 3x6 matrix.  The point group of the crystal reduces them to a
few independent components and forbids the rest - one component for ``-43m``,
three for ``6mm``, four for ``3m`` and so on - so a fit of all 18 components
returns numbers that the symmetry requires to vanish, alongside violations of
the relations between the components that do not vanish.  This module puts the
symmetry back, exactly as :mod:`abacustools.data.elastic` does for the
stiffness tensor.

The transformation is carried out on the full ``d_ijk`` tensor, where the
rotation of a third-rank tensor is unambiguous and no Voigt factor can be
lost, and the components that survive are named the usual way (``d14``,
``d15``, ``d31``, ``d33``, ...).

References:
    Nye, *Physical Properties of Crystals* (1985), and the tables of the
    International Tables for Crystallography, volume D.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from abacustools.data.elastic import (
    DEFAULT_TOLERANCE,
    RANK_TOLERANCE,
    VOIGT_PAIRS,
    numerical_rank,
)


def tensor_from_voigt(matrix: Any) -> np.ndarray:
    """Expand a 3x6 piezoelectric matrix into its third-rank tensor.

    Args:
        matrix: Three-by-six matrix, in C/m^2.

    Returns:
        The ``d_ijk`` tensor, symmetric in its last two indices.
    """
    values = np.asarray(matrix, dtype=float)
    if values.shape != (3, 6):
        raise ValueError("a piezoelectric tensor must be a 3x6 matrix")
    tensor = np.zeros((3, 3, 3), dtype=float)
    for index, (first, second) in enumerate(VOIGT_PAIRS):
        tensor[:, first, second] = values[:, index]
        tensor[:, second, first] = values[:, index]
    return tensor


def voigt_from_tensor(tensor: np.ndarray) -> np.ndarray:
    """Collapse a third-rank piezoelectric tensor to its 3x6 matrix."""
    values = np.asarray(tensor, dtype=float)
    if values.shape != (3, 3, 3):
        raise ValueError("a third-rank tensor must be 3x3x3")
    matrix = np.zeros((3, 6), dtype=float)
    for index, (first, second) in enumerate(VOIGT_PAIRS):
        matrix[:, index] = values[:, first, second]
    return matrix


def project_tensor(matrix: Any, rotations: np.ndarray) -> np.ndarray:
    """Project a piezoelectric matrix onto the invariant subspace of a group.

    Args:
        matrix: Three-by-six piezoelectric matrix, in C/m^2.
        rotations: Point group rotations of the reference structure.

    Returns:
        The symmetrised 3x6 matrix, in C/m^2.
    """
    tensor = tensor_from_voigt(matrix)
    group = np.asarray(rotations, dtype=float)
    if group.ndim != 3 or group.shape[1:] != (3, 3):
        raise ValueError("rotations must be an array of 3x3 matrices")
    total = np.zeros_like(tensor)
    for rotation in group:
        total += np.einsum("ia,jb,kc,abc->ijk", rotation, rotation, rotation, tensor)
    projected = total / len(group)
    # A polar third-rank tensor is symmetric in its last two indices, which the
    # average only preserves up to the numerical noise of the group.
    projected = 0.5 * (projected + projected.transpose(0, 2, 1))
    return voigt_from_tensor(projected)


def symmetrize_piezoelectric_tensor(matrix: Any, rotations: np.ndarray) -> np.ndarray:
    """Symmetrise a fitted piezoelectric matrix with the crystal symmetry.

    Args:
        matrix: Three-by-six piezoelectric matrix, in C/m^2.
        rotations: Point group rotations of the reference structure.

    Returns:
        The symmetrised 3x6 matrix, in C/m^2.
    """
    return project_tensor(matrix, rotations)


def symmetrization_residual(raw: Any, symmetrized: Any) -> float:
    """Return the largest component changed by the symmetrisation, in C/m^2."""
    difference = np.asarray(raw, dtype=float) - np.asarray(symmetrized, dtype=float)
    return float(np.max(np.abs(difference)))


def piezoelectric_basis(
    rotations: np.ndarray,
    *,
    tolerance: float = RANK_TOLERANCE,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Return a basis of the piezoelectric matrices a point group allows.

    The basis is built by walking the components in the usual order and
    keeping those that add a new direction to the projected subspace, so its
    length is the number of independent components of the point group and the
    entries are named ``d14``, ``d15``, ``d31``, ``d33`` and so on.

    Args:
        rotations: Point group rotations of the reference structure.
        tolerance: Relative tolerance of the rank test.

    Returns:
        The matrices of the basis and the ``(row, column)`` index of the
        component each one belongs to.
    """
    basis: List[np.ndarray] = []
    flat: List[np.ndarray] = []
    pairs: List[Tuple[int, int]] = []
    rank = 0
    for row in range(3):
        for column in range(6):
            element = np.zeros((3, 6), dtype=float)
            element[row, column] = 1.0
            projected = project_tensor(element, rotations)
            new_rank = numerical_rank(
                np.asarray(flat + [projected.ravel()], dtype=float), tolerance
            )
            if new_rank <= rank:
                continue
            basis.append(projected)
            flat.append(projected.ravel())
            pairs.append((row, column))
            rank = new_rank
    return basis, pairs


def independent_component_count(rotations: np.ndarray) -> int:
    """Return how many piezoelectric components a point group leaves free."""
    basis, _ = piezoelectric_basis(rotations)
    return len(basis)


def component_name(row: int, column: int) -> str:
    """Return the usual name of a piezoelectric component, e.g. ``d31``."""
    return f"d{row + 1}{column + 1}"


def independent_components(
    matrix: Any,
    rotations: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, float]:
    """Return the independent components of a symmetrised matrix.

    Args:
        matrix: Three-by-six piezoelectric matrix, in C/m^2.
        rotations: Point group rotations of the reference structure.
        tolerance: Components below this magnitude are left out, in C/m^2.

    Returns:
        Mapping of names such as ``d33`` to their C/m^2 values.
    """
    values = np.asarray(matrix, dtype=float)
    if values.shape != (3, 6):
        raise ValueError("a piezoelectric tensor must be a 3x6 matrix")
    _, pairs = piezoelectric_basis(rotations)
    components: Dict[str, float] = {}
    for row, column in pairs:
        value = float(values[row, column])
        if abs(value) > tolerance:
            components[component_name(row, column)] = value
    return components


def independent_strain_modes(
    rotations: np.ndarray,
    *,
    tolerance: float = RANK_TOLERANCE,
) -> List[int]:
    """Return the strain modes needed for the independent components.

    Staining along the Voigt mode ``J`` measures the column ``d[:, J]``, so a
    mode contributes the information ``M[J][i, k] = B_k[i, J]`` built from the
    basis matrices.  The modes are taken in the usual order and kept while
    they raise the rank of the stacked information, which stops as soon as the
    independent components are determined.

    Args:
        rotations: Point group rotations of the reference structure.
        tolerance: Relative tolerance of the rank test.

    Returns:
        The Voigt indices of the strain modes, in increasing order.
    """
    basis, _ = piezoelectric_basis(rotations, tolerance=tolerance)
    stacked: List[np.ndarray] = []
    modes: List[int] = []
    rank = 0
    for column in range(6):
        block = np.asarray([matrix[:, column] for matrix in basis], dtype=float).T
        new_rank = numerical_rank(np.vstack(stacked + [block]), tolerance)
        if new_rank <= rank:
            continue
        stacked.append(block)
        modes.append(column)
        rank = new_rank
        if rank == len(basis):
            break
    if rank < len(basis):
        raise ValueError(
            "the strain modes do not determine the independent piezoelectric "
            f"components (rank {rank} of {len(basis)})"
        )
    return modes


def fit_independent_piezoelectric(
    matrix: Any,
    modes: Sequence[int],
    rotations: np.ndarray,
    *,
    tolerance: float = RANK_TOLERANCE,
) -> np.ndarray:
    """Fit only the independent components of a piezoelectric matrix.

    The matrix is written as a combination of the symmetry allowed basis
    matrices, ``d = sum_k a_k B_k``, and the coefficients are fitted to the
    columns that were measured in one least squares.

    Args:
        matrix: Three-by-six matrix holding the measured columns, in C/m^2.
        modes: Voigt indices of the columns that were measured.
        rotations: Point group rotations of the reference structure.
        tolerance: Relative tolerance of the rank test that builds the basis.

    Returns:
        The 3x6 matrix reconstructed from the independent components.

    Raises:
        ValueError: When the measured columns do not determine every
            independent component.
    """
    values = np.asarray(matrix, dtype=float)
    if values.shape != (3, 6):
        raise ValueError("a piezoelectric tensor must be a 3x6 matrix")
    measured = [int(mode) for mode in modes]
    if not measured:
        raise ValueError("at least one strain mode is needed")
    basis, pairs = piezoelectric_basis(rotations, tolerance=tolerance)
    design = np.zeros((3 * len(measured), len(basis)), dtype=float)
    target = np.zeros(3 * len(measured), dtype=float)
    for slot, mode in enumerate(measured):
        for column, block in enumerate(basis):
            design[slot * 3 : slot * 3 + 3, column] = block[:, mode]
        target[slot * 3 : slot * 3 + 3] = values[:, mode]
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < len(basis):
        labels = ", ".join(component_name(row, column) for row, column in pairs)
        raise ValueError(
            "the measured strain modes do not determine the independent "
            f"components {labels} (rank {rank} of {len(basis)})"
        )
    result = np.zeros((3, 6), dtype=float)
    for coefficient, block in zip(solution, basis):
        result = result + coefficient * block
    return result
