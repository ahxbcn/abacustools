"""Symmetry handling of the elastic stiffness tensor.

The stress-strain workflow fits the 6x6 stiffness matrix from a set of
strained calculations.  That fit is unconstrained: it returns 36 numbers, so
the numerical noise of the stresses shows up both as components the crystal
symmetry forbids and as pairs that violate the major symmetry
``C_ijkl = C_klij``.  This module puts the crystal symmetry back.

The symmetrisation is the Reynolds projection of the fitted tensor over the
point group operations of the reference cell,

    C_sym = (1 / |G|) * sum_g R_g C R_g^T,

carried out on the full fourth-rank tensor ``C_ijkl``, where the
transformation is unambiguous and no Voigt factor can be lost: Voigt notation
couples the engineering shear strains to the shear stresses, which makes the
6x6 transformation matrices easy to get wrong.  The projection is a
projector, so applying it twice changes nothing, and it preserves the major
symmetry of its input.

The number of independent constants that survives the projection is a
property of the Laue class: 21 for triclinic, 13 for monoclinic, 9 for
orthorhombic, 7 (4/m) or 6 (4/mmm) for tetragonal, 7 (-3) or 6 (-3m) for
trigonal, 5 for hexagonal and 3 for cubic crystals.  See Nye, *Physical
Properties of Crystals* (1985), and Mouhat & Coudert, Phys. Rev. B **90**,
224104 (2014).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from abacustools.io.stru import AbacusSTRU


#: Voigt index -> the pair of Cartesian directions it abbreviates.
VOIGT_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)

#: Tolerance below which a component counts as zero.
DEFAULT_TOLERANCE = 1.0e-8

#: Relative tolerance of the rank test that picks the independent components.
#: The point group operations of a relaxed cell are never exact - the cell is
#: only symmetric to the tolerance of the relaxation that produced it - so the
#: projection leaks a little of every forbidden basis tensor.  Those leaks are
#: orders of magnitude smaller than a genuinely independent direction, and the
#: rank test has to sit between the two.
RANK_TOLERANCE = 1.0e-3

#: Absolute floor of the rank test.  The basis tensors are built from unit
#: components, so a direction that survives a projection has a norm of order
#: one, while a direction the symmetry forbids projects onto numerical zero.
#: Without the floor such a zero would still count as a new direction, because
#: the tolerance of the rank test is relative to the largest singular value of
#: the matrix it is given.
RANK_FLOOR = 1.0e-9


def numerical_rank(
    matrix: np.ndarray,
    relative_tolerance: float = RANK_TOLERANCE,
    *,
    floor: float = RANK_FLOOR,
) -> int:
    """Return the numerical rank with a tolerance scaled to the matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return 0
    singular = np.linalg.svd(values, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0
    threshold = max(relative_tolerance * singular[0], floor)
    return int(np.count_nonzero(singular > threshold))


def voigt_index(first: int, second: int) -> int:
    """Return the Voigt index of a pair of Cartesian directions.

    Args:
        first: First Cartesian direction, 0 to 2.
        second: Second Cartesian direction, 0 to 2.

    Returns:
        Index into a six-component Voigt vector.
    """
    for index, (left, right) in enumerate(VOIGT_PAIRS):
        if {first, second} == {left, right}:
            return index
    raise ValueError(f"invalid Cartesian directions: {first}, {second}")


def component_label(first: int, second: int) -> str:
    """Return the usual name of an elastic component, e.g. ``C11``."""
    left = voigt_index(first, second) + 1
    return f"C{left}"


def elastic_component_names() -> List[str]:
    """Return the names of the 21 independent components, ``C11`` to ``C66``."""
    return [
        f"C{first + 1}{second + 1}"
        for first in range(6)
        for second in range(first, 6)
    ]


def stress_voigt(stress: np.ndarray) -> List[float]:
    """Convert a 3x3 stress matrix to its six Voigt components."""
    values = np.asarray(stress, dtype=float)
    if values.shape != (3, 3):
        raise ValueError("a stress tensor must be a 3x3 matrix")
    return [
        float(values[0, 0]),
        float(values[1, 1]),
        float(values[2, 2]),
        float(values[1, 2]),
        float(values[0, 2]),
        float(values[0, 1]),
    ]


def matrix_from_voigt(values: Any) -> np.ndarray:
    """Build a symmetric 6x6 stiffness matrix from Voigt components.

    Args:
        values: Either 36 row-major components, or 21 components in the order
            given by :func:`elastic_component_names`.

    Returns:
        The 6x6 stiffness matrix in GPa.
    """
    data = np.asarray(values, dtype=float).ravel()
    matrix = np.zeros((6, 6), dtype=float)
    if data.size == 36:
        return data.reshape(6, 6)
    if data.size == 21:
        for value, (first, second) in zip(data, _upper_triangle_pairs()):
            matrix[first, second] = value
            matrix[second, first] = value
        return matrix
    raise ValueError("an elastic tensor needs 21 or 36 components")


def _upper_triangle_pairs() -> List[Tuple[int, int]]:
    """Return the pairs of the upper triangle of a 6x6 matrix."""
    return [(first, second) for first in range(6) for second in range(first, 6)]


def tensor4_from_voigt(voigt: np.ndarray) -> np.ndarray:
    """Expand a 6x6 stiffness matrix into its fourth-rank tensor.

    Args:
        voigt: Six-by-six stiffness matrix, in GPa.

    Returns:
        The ``C_ijkl`` tensor, with both minor symmetries filled in.
    """
    matrix = np.asarray(voigt, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError("an elastic tensor must be a 6x6 matrix")
    tensor = np.zeros((3, 3, 3, 3), dtype=float)
    for first, (i, j) in enumerate(VOIGT_PAIRS):
        for second, (k, l) in enumerate(VOIGT_PAIRS):
            value = matrix[first, second]
            tensor[i, j, k, l] = value
            tensor[j, i, k, l] = value
            tensor[i, j, l, k] = value
            tensor[j, i, l, k] = value
    return tensor


def voigt_from_tensor4(tensor: np.ndarray) -> np.ndarray:
    """Collapse a fourth-rank stiffness tensor back to Voigt notation."""
    values = np.asarray(tensor, dtype=float)
    if values.shape != (3, 3, 3, 3):
        raise ValueError("a fourth-rank elastic tensor must be 3x3x3x3")
    matrix = np.zeros((6, 6), dtype=float)
    for first, (i, j) in enumerate(VOIGT_PAIRS):
        for second, (k, l) in enumerate(VOIGT_PAIRS):
            matrix[first, second] = values[i, j, k, l]
    return matrix


def point_group_operations(
    structure: AbacusSTRU,
    *,
    symprec: float = 1.0e-5,
    angle_tolerance: float = 5.0,
) -> np.ndarray:
    """Return the point group operations of a structure, in Cartesian axes.

    Args:
        structure: Structure to analyse.
        symprec: Symmetry distance tolerance in Angstrom.
        angle_tolerance: Symmetry angle tolerance in degrees.

    Returns:
        Array of shape ``(n, 3, 3)`` holding the rotation matrices in the
        Cartesian frame of the cell, which is the frame the fitted elastic
        tensor lives in.

    Raises:
        RuntimeError: When the symmetry analysis is not available.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(
        structure.to("pymatgen"),
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    operations = analyzer.get_point_group_operations(cartesian=True)
    rotations = np.asarray(
        [operation.rotation_matrix for operation in operations], dtype=float
    )
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise RuntimeError("the point group operations could not be determined")
    return rotations


def project_tensor(voigt: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Project a stiffness matrix onto the invariant subspace of a point group.

    Args:
        voigt: Six-by-six stiffness matrix, in GPa.
        rotations: Point group rotations, as returned by
            :func:`point_group_operations`.

    Returns:
        The symmetrised 6x6 stiffness matrix, in GPa.
    """
    tensor = tensor4_from_voigt(voigt)
    group = np.asarray(rotations, dtype=float)
    if group.ndim != 3 or group.shape[1:] != (3, 3):
        raise ValueError("rotations must be an array of 3x3 matrices")
    total = np.zeros_like(tensor)
    for rotation in group:
        total += np.einsum(
            "ai,bj,ck,dl,ijkl->abcd", rotation, rotation, rotation, rotation, tensor
        )
    projected = total / len(group)
    # The average of a polar tensor is symmetric in (ij) and (kl); the tiny
    # numerical asymmetry that is left is removed so the result is exactly a
    # stiffness matrix.
    return voigt_from_tensor4(0.5 * (projected + projected.transpose(1, 0, 3, 2)))


def symmetrize_elastic_tensor(
    voigt: np.ndarray,
    rotations: np.ndarray,
    *,
    enforce_major_symmetry: bool = True,
) -> np.ndarray:
    """Symmetrise a fitted stiffness matrix with the crystal symmetry.

    Args:
        voigt: Six-by-six stiffness matrix fitted from the stresses, in GPa.
        rotations: Point group rotations of the reference structure.
        enforce_major_symmetry: Average the tensor with its transpose first.
            The elastic tensor of an unstrained equilibrium crystal satisfies
            ``C_ij = C_ji``, so averaging removes half of the noise before the
            projection.

    Returns:
        The symmetrised 6x6 stiffness matrix, in GPa.
    """
    matrix = np.asarray(voigt, dtype=float)
    if enforce_major_symmetry:
        matrix = 0.5 * (matrix + matrix.T)
    return project_tensor(matrix, rotations)


def symmetrization_residual(raw: np.ndarray, symmetrized: np.ndarray) -> float:
    """Return the largest component changed by the symmetrisation, in GPa."""
    difference = np.asarray(raw, dtype=float) - np.asarray(symmetrized, dtype=float)
    return float(np.max(np.abs(difference)))


def independent_components(
    voigt: np.ndarray,
    rotations: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, float]:
    """Return the components a symmetry class leaves independent.

    A component is independent when its basis tensor adds a new direction to
    the invariant subspace spanned by the components that come before it in
    the usual order (``C11``, ``C12``, ... ``C66``).  This reproduces the
    textbook lists: three components for cubic, five for hexagonal with ``C66``
    left out because it follows from ``C11`` and ``C12``, six for the ``-3m``
    class with ``C14`` kept and ``C15`` dropped, and so on.

    Args:
        voigt: Six-by-six stiffness matrix, in GPa.
        rotations: Point group rotations of the reference structure.
        tolerance: Components below this magnitude are left out, in GPa.

    Returns:
        Mapping of component names such as ``C11`` to their GPa values.
    """
    matrix = np.asarray(voigt, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError("an elastic tensor must be a 6x6 matrix")
    _, pairs = independent_basis(rotations)
    components: Dict[str, float] = {}
    for first, second in pairs:
        value = 0.5 * (matrix[first, second] + matrix[second, first])
        if abs(value) > tolerance:
            components[f"C{first + 1}{second + 1}"] = float(value)
    return components


def independent_component_count(rotations: np.ndarray) -> int:
    """Return how many elastic constants a point group leaves independent.

    The count is the rank of the projection, which is obtained by projecting
    the 21 basis tensors of the symmetric 6x6 space.

    Args:
        rotations: Point group rotations of the reference structure.

    Returns:
        The number of independent elastic constants.
    """
    basis, _ = independent_basis(rotations)
    return len(basis)


def independent_basis(
    rotations: np.ndarray,
    *,
    tolerance: float = RANK_TOLERANCE,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Return a basis of the elastic tensors a point group allows.

    The basis is built by walking the usual component order and keeping the
    basis tensors that add a new direction to the projected subspace, so the
    basis tensors come in the order ``C11``, ``C12``, ... and their number is
    the number of independent elastic constants of the Laue class.

    Args:
        rotations: Point group rotations of the reference structure.
        tolerance: Relative tolerance of the rank test.

    Returns:
        The matrices of the basis, and the pair of Voigt indices of the
        component each one belongs to.
    """
    basis: List[np.ndarray] = []
    pairs: List[Tuple[int, int]] = []
    flat: List[np.ndarray] = []
    rank = 0
    for first, second in _upper_triangle_pairs():
        element = np.zeros((6, 6), dtype=float)
        element[first, second] = 1.0
        element[second, first] = 1.0
        projected = project_tensor(element, rotations)
        new_rank = numerical_rank(
            np.asarray(flat + [projected.ravel()], dtype=float), tolerance
        )
        if new_rank <= rank:
            continue
        basis.append(projected)
        flat.append(projected.ravel())
        pairs.append((first, second))
        rank = new_rank
    return basis, pairs


def strain_tensor(index: int) -> np.ndarray:
    """Return the 3x3 strain tensor whose Voigt form is the unit vector ``e``.

    The Voigt shear components carry the engineering factor of two, so the
    tensor of ``e4`` is ``(1/2)(e_y e_z + e_z e_y)`` and its Voigt form is
    ``(0, 0, 0, 1, 0, 0)``.

    Args:
        index: Voigt index between 0 and 5.

    Returns:
        The symmetric strain tensor.
    """
    if not 0 <= index < 6:
        raise ValueError("a Voigt index runs from 0 to 5")
    tensor = np.zeros((3, 3), dtype=float)
    first, second = VOIGT_PAIRS[index]
    if first == second:
        tensor[first, second] = 1.0
    else:
        tensor[first, second] = 0.5
        tensor[second, first] = 0.5
    return tensor


def strain_voigt(tensor: np.ndarray) -> np.ndarray:
    """Return the six engineering Voigt components of a strain tensor."""
    values = np.asarray(tensor, dtype=float)
    if values.shape != (3, 3):
        raise ValueError("a strain tensor must be a 3x3 matrix")
    symmetric = 0.5 * (values + values.T)
    return np.array(
        [
            symmetric[0, 0],
            symmetric[1, 1],
            symmetric[2, 2],
            2.0 * symmetric[1, 2],
            2.0 * symmetric[0, 2],
            2.0 * symmetric[0, 1],
        ]
    )


def _unit_strain(index: int) -> np.ndarray:
    """Return the unit Voigt vector of one strain direction."""
    direction = strain_voigt(strain_tensor(index))
    return direction / np.linalg.norm(direction)


def _rotated_strain(index: int, rotation: np.ndarray) -> np.ndarray:
    """Return the unit Voigt vector of a strain direction after a rotation."""
    tensor = rotation @ strain_tensor(index) @ rotation.T
    direction = strain_voigt(tensor)
    return direction / np.linalg.norm(direction)


def independent_strain_modes(
    rotations: np.ndarray,
    *,
    tolerance: float = RANK_TOLERANCE,
) -> List[int]:
    """Return the strain directions needed for the independent constants.

    A strain direction ``eps`` contributes the equations
    ``sigma_i = sum_k a_k (B_k eps)_i``, so its information is the matrix
    ``M[eps][i, k] = (B_k eps)_i`` built from the symmetry allowed basis
    tensors.  The directions are taken in the usual Voigt order and kept when
    they raise the rank of the stacked information, which stops as soon as the
    independent constants are determined: two directions for a cubic crystal
    (``xx`` and ``yz``), three for a hexagonal one, four for a tetragonal one,
    and all six when no symmetry relates any of them.

    Args:
        rotations: Point group rotations of the reference structure.
        tolerance: Relative tolerance of the rank test.

    Returns:
        The Voigt indices of the strain directions, in increasing order.

    Raises:
        ValueError: When the six single-component strains do not determine
            every independent constant, which cannot happen for a point group
            of a three-dimensional crystal.
    """
    basis, _ = independent_basis(rotations, tolerance=tolerance)
    stacked: List[np.ndarray] = []
    rank = 0
    modes: List[int] = []
    for index in range(6):
        direction = strain_voigt(strain_tensor(index))
        block = np.asarray([tensor @ direction for tensor in basis], dtype=float).T
        new_rank = numerical_rank(np.vstack(stacked + [block]), tolerance)
        if new_rank <= rank:
            continue
        stacked.append(block)
        modes.append(index)
        rank = new_rank
        if rank == len(basis):
            break
    if rank < len(basis):
        raise ValueError(
            "the single-component strains do not determine the independent "
            f"constants (rank {rank} of {len(basis)})"
        )
    return modes


def fit_independent_stress_strain(
    strain_values: np.ndarray,
    stress_values: np.ndarray,
    rotations: np.ndarray,
    *,
    equilibrium_stress: Optional[np.ndarray] = None,
    tolerance: float = RANK_TOLERANCE,
) -> np.ndarray:
    """Fit only the independent constants of a stiffness matrix.

    The stiffness matrix is written as a combination of the symmetry allowed
    basis tensors, ``C = sum_k a_k B_k``, and the coefficients are fitted to
    the measured stresses in one least squares.  A strain set that only
    contains representatives of the strain orbits still determines every
    coefficient, which is what makes the reduced strain sets of
    :func:`independent_strain_modes` usable.

    Args:
        strain_values: Array of shape ``(n, 6)`` with the applied strains.
        stress_values: Array of shape ``(n, 6)`` with the stresses, in GPa.
        rotations: Point group rotations of the reference structure.
        equilibrium_stress: Stress of the unstrained cell, in Voigt form and
            in GPa, subtracted from every stress before fitting.
        tolerance: Relative tolerance of the rank test that builds the basis.

    Returns:
        The 6x6 stiffness matrix, in GPa, following ``sigma_i = C_ij eps_j``.

    Raises:
        ValueError: When the strain and stress arrays do not match, or when
            the data does not determine every independent constant.
    """
    strains = np.asarray(strain_values, dtype=float)
    stresses = np.asarray(stress_values, dtype=float)
    if strains.ndim != 2 or strains.shape[1] != 6:
        raise ValueError("strains must have shape (n, 6)")
    if stresses.shape != strains.shape:
        raise ValueError("strains and stresses must have the same shape")
    if equilibrium_stress is not None:
        reference = np.asarray(equilibrium_stress, dtype=float).ravel()
        if reference.size != 6:
            raise ValueError("the equilibrium stress needs six components")
        stresses = stresses - reference

    basis, pairs = independent_basis(rotations, tolerance=tolerance)
    design = np.zeros((strains.shape[0] * 6, len(basis)), dtype=float)
    target = np.zeros(strains.shape[0] * 6, dtype=float)
    for state, (strain, stress) in enumerate(zip(strains, stresses)):
        for column, tensor in enumerate(basis):
            design[state * 6 : state * 6 + 6, column] = tensor @ strain
        target[state * 6 : state * 6 + 6] = stress
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < len(basis):
        labels = ", ".join(f"C{first + 1}{second + 1}" for first, second in pairs)
        raise ValueError(
            "the strain states do not determine the independent constants "
            f"{labels} (rank {rank} of {len(basis)})"
        )
    tensor = np.zeros((6, 6), dtype=float)
    for coefficient, block in zip(solution, basis):
        tensor = tensor + coefficient * block
    return tensor


def strain_matrix(voigt: Any) -> np.ndarray:
    """Return the 3x3 strain tensor of a six-component engineering strain."""
    values = np.asarray(voigt, dtype=float).ravel()
    if values.size != 6:
        raise ValueError("a Voigt strain needs six components")
    tensor = np.zeros((3, 3), dtype=float)
    for index, (first, second) in enumerate(VOIGT_PAIRS):
        if first == second:
            tensor[first, second] = values[index]
        else:
            tensor[first, second] = 0.5 * values[index]
            tensor[second, first] = 0.5 * values[index]
    return tensor


def deformation_gradient(voigt: Any) -> np.ndarray:
    """Return the deformation gradient of a Lagrangian strain.

    The gradient is ``F = (I + 2 eps)^(1/2)``, computed by diagonalising the
    symmetric matrix ``I + 2 eps``, so that ``F^T F - I = 2 eps`` holds for the
    strain that is applied to the cell.

    Args:
        voigt: Six-component engineering strain.

    Returns:
        The 3x3 deformation gradient.
    """
    matrix = np.eye(3) + 2.0 * strain_matrix(voigt)
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0.0):
        raise ValueError("the strain is too large for a deformation gradient")
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def energy_strain_patterns(
    rotations: np.ndarray,
    *,
    tolerance: float = RANK_TOLERANCE,
) -> List[np.ndarray]:
    """Return the strain patterns whose energy curvature fixes the constants.

    A strain pattern ``u`` strained by an amplitude ``e`` has the energy
    ``E = E_0 + V_0 (u . sigma_0) e + (V_0 / 2) (u^T C u) e^2``, so its
    curvature measures the quadratic form ``u^T C u = sum_k a_k u^T B_k u``.
    Walking the single-component strains and then the pairwise combinations,
    this keeps the patterns that raise the rank of that information until every
    independent constant is covered: two patterns for a cubic crystal, three
    for a hexagonal one, four for a tetragonal one, and more for lower
    symmetry.

    Args:
        rotations: Point group rotations of the reference structure.
        tolerance: Relative tolerance of the rank test.

    Returns:
        Unit patterns, in the order they were selected.

    Raises:
        ValueError: When the pattern pool does not determine every constant.
    """
    basis, _ = independent_basis(rotations, tolerance=tolerance)
    candidates: List[np.ndarray] = []
    for index in range(6):
        pattern = np.zeros(6, dtype=float)
        pattern[index] = 1.0
        candidates.append(pattern)
    for first in range(6):
        for second in range(first + 1, 6):
            pattern = np.zeros(6, dtype=float)
            pattern[first] = 1.0
            pattern[second] = 1.0
            candidates.append(pattern)

    stacked: List[np.ndarray] = []
    patterns: List[np.ndarray] = []
    rank = 0
    for candidate in candidates:
        direction = candidate / np.linalg.norm(candidate)
        row = np.asarray(
            [direction @ tensor @ direction for tensor in basis], dtype=float
        )
        new_rank = numerical_rank(np.asarray(stacked + [row], dtype=float), tolerance)
        if new_rank <= rank:
            continue
        stacked.append(row)
        patterns.append(direction)
        rank = new_rank
        if rank == len(basis):
            break
    if rank < len(basis):
        raise ValueError(
            "the strain patterns do not determine the independent constants "
            f"(rank {rank} of {len(basis)})"
        )
    return patterns


def fit_energy_strain(
    strains: np.ndarray,
    energies: np.ndarray,
    volume: float,
    rotations: np.ndarray,
    *,
    tolerance: float = RANK_TOLERANCE,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit the stiffness matrix from the curvature of the total energy.

    The energies of the strained cells are fitted with

        E(e) = E0 + (V0 / 2) sum_k a_k (e^T B_k e),

    which is linear in the coefficients of the symmetry allowed basis tensors.
    A stress of the reference cell adds a term that is odd in the amplitude,
    and every pattern is strained with amplitudes that are symmetric about
    zero, so that term is orthogonal to the constant and quadratic columns of
    the fit and cannot leak into the curvature.  The reference stress is
    instead reported from the odd part of each pattern, where it is
    ``V0 (u . sigma0)``.

    Args:
        strains: Array of shape ``(n, 6)`` with the applied strains.
        energies: Total energies of the strained cells, in eV.
        volume: Volume of the reference cell, in Angstrom^3.
        rotations: Point group rotations of the reference structure.
        tolerance: Relative tolerance of the rank test that builds the basis.

    Returns:
        ``(tensor, pattern_stress, residual)`` with the stiffness matrix in
        GPa, the stress of the reference cell projected on every strain
        pattern (in GPa, in the order the patterns appear in the strains), and
        the root mean square energy residual in eV.
    """
    strain_values = np.asarray(strains, dtype=float)
    energy_values = np.asarray(energies, dtype=float).ravel()
    if strain_values.ndim != 2 or strain_values.shape[1] != 6:
        raise ValueError("strains must have shape (n, 6)")
    if energy_values.size != strain_values.shape[0]:
        raise ValueError("every strain needs one energy")
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("the reference volume must be positive")
    patterns, groups = _strain_pattern_groups(strain_values, tolerance=tolerance)
    basis, pairs = independent_basis(rotations, tolerance=tolerance)
    from abacustools.data.eos import EV_PER_ANGSTROM3_TO_GPA

    # Split the energy of every pattern into its odd and even parts.  The odd
    # part is the stress of the reference cell acting along the pattern; the
    # even part is what the elastic constants are fitted from.
    even_energies = energy_values.copy()
    stresses = []
    for pattern, group in zip(patterns, groups):
        amplitudes = np.asarray(
            [strain_values[state] @ pattern for state in group], dtype=float
        )
        denominator = float(np.sum(amplitudes**2))
        linear = (
            float(np.sum(amplitudes * energy_values[group]) / denominator)
            if denominator > 0.0
            else 0.0
        )
        even_energies[group] = even_energies[group] - linear * amplitudes
        stresses.append(linear / volume * EV_PER_ANGSTROM3_TO_GPA)

    columns = 1 + len(basis)
    design = np.zeros((strain_values.shape[0], columns), dtype=float)
    for state, strain in enumerate(strain_values):
        design[state, 0] = 1.0
        for column, tensor in enumerate(basis):
            design[state, 1 + column] = 0.5 * volume * (strain @ tensor @ strain)
    solution, _, rank, _ = np.linalg.lstsq(design, even_energies, rcond=None)
    if rank < columns:
        labels = ", ".join(f"C{first + 1}{second + 1}" for first, second in pairs)
        raise ValueError(
            "the strain states do not determine the independent constants "
            f"{labels} (rank {rank} of {columns}); use more strain patterns"
        )

    tensor = np.zeros((6, 6), dtype=float)
    for coefficient, block in zip(solution[1:], basis):
        tensor = tensor + coefficient * block
    residual = float(np.sqrt(np.mean((even_energies - design @ solution) ** 2)))
    return (
        tensor * EV_PER_ANGSTROM3_TO_GPA,
        np.asarray(stresses, dtype=float),
        residual,
    )


def _strain_pattern_groups(
    strains: np.ndarray, *, tolerance: float
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Group a strain set by the pattern each state belongs to.

    Args:
        strains: Array of shape ``(n, 6)`` with the applied strains.
        tolerance: Relative tolerance used to recognise a repeated pattern.

    Returns:
        The unit strain patterns and, for every pattern, the indices of its
        states.
    """
    patterns: List[np.ndarray] = []
    groups: List[List[int]] = []
    for state, strain in enumerate(strains):
        norm = float(np.linalg.norm(strain))
        if norm == 0.0:
            continue
        direction = strain / norm
        for index, pattern in enumerate(patterns):
            if abs(abs(float(np.dot(direction, pattern))) - 1.0) < tolerance:
                groups[index].append(state)
                break
        else:
            if direction[np.flatnonzero(direction != 0.0)[0]] < 0.0:
                # A strain direction is only defined up to its sign.
                direction = -direction
            patterns.append(direction)
            groups.append([state])
    return patterns, groups


def elastic_moduli(tensor: np.ndarray) -> Dict[str, float]:
    """Return the Voigt bulk, shear, Young's and Poisson moduli, in GPa.

    Args:
        tensor: Six-by-six stiffness matrix, in GPa.

    Returns:
        Mapping with ``bulk_modulus``, ``shear_modulus``, ``young_modulus``
        and ``poisson_ratio``.
    """
    matrix = np.asarray(tensor, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError("an elastic tensor must be a 6x6 matrix")
    diagonal = np.trace(matrix[:3, :3])
    off_diagonal = matrix[0, 1] + matrix[0, 2] + matrix[1, 2]
    shear = matrix[3, 3] + matrix[4, 4] + matrix[5, 5]
    bulk_modulus = (diagonal + 2.0 * off_diagonal) / 9.0
    shear_modulus = (diagonal - off_diagonal + 3.0 * shear) / 15.0
    denominator = 3.0 * bulk_modulus + shear_modulus
    if denominator == 0:
        raise ValueError("cannot calculate Young's modulus from the fitted tensor")
    young_modulus = 9.0 * bulk_modulus * shear_modulus / denominator
    poisson_ratio = (3.0 * bulk_modulus - 2.0 * shear_modulus) / (
        2.0 * denominator
    )
    return {
        "bulk_modulus": float(bulk_modulus),
        "shear_modulus": float(shear_modulus),
        "young_modulus": float(young_modulus),
        "poisson_ratio": float(poisson_ratio),
    }


def fit_stress_strain(
    strain_values: np.ndarray,
    stress_values: np.ndarray,
    *,
    equilibrium_stress: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit the stiffness matrix from strain and stress data.

    The strain states of the workflow carry one Voigt component each, so the
    column ``j`` of the stiffness matrix is the slope of every stress
    component against the strain component ``j``.  The fit is unconstrained
    and covers all 36 components; :func:`symmetrize_elastic_tensor` removes
    the part the crystal symmetry forbids.

    Args:
        strain_values: Array of shape ``(n, 6)`` with the applied strains.
        stress_values: Array of shape ``(n, 6)`` with the stresses, in GPa.
        equilibrium_stress: Stress of the unstrained cell, in Voigt form and
            in GPa, subtracted from every stress before fitting.

    Returns:
        The 6x6 stiffness matrix, in GPa, following ``sigma_i = C_ij eps_j``.
    """
    strains = np.asarray(strain_values, dtype=float)
    stresses = np.asarray(stress_values, dtype=float)
    if strains.ndim != 2 or strains.shape[1] != 6:
        raise ValueError("strains must have shape (n, 6)")
    if stresses.shape != strains.shape:
        raise ValueError("strains and stresses must have the same shape")
    if equilibrium_stress is not None:
        reference = np.asarray(equilibrium_stress, dtype=float).ravel()
        if reference.size != 6:
            raise ValueError("the equilibrium stress needs six components")
        stresses = stresses - reference

    tensor = np.zeros((6, 6), dtype=float)
    for component in range(6):
        mask = np.abs(strains[:, component]) > 0.0
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"insufficient strain data for component {component}")
        columns = [strains[mask, component]]
        # The strain states of a single Voigt component are uncoupled, but the
        # fit keeps the other components as regressors so that a strain set
        # that mixes them still gives the full stiffness matrix.
        for other in range(6):
            if other == component:
                continue
            if np.any(np.abs(strains[mask, other]) > 0.0):
                columns.append(strains[mask, other])
        design = np.column_stack(columns + [np.ones(np.count_nonzero(mask))])
        for stress_component in range(6):
            solution = np.linalg.lstsq(
                design,
                stresses[mask, stress_component],
                rcond=None,
            )[0]
            tensor[stress_component, component] = solution[0]
    return tensor
