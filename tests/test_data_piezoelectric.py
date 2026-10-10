"""Tests for the symmetry-aware piezoelectric tensor helpers."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from abacustools.data.elastic import point_group_operations
from abacustools.data.piezoelectric import (
    fit_independent_piezoelectric,
    independent_component_count,
    independent_components,
    independent_strain_modes,
    project_tensor,
    symmetrize_piezoelectric_tensor,
    symmetrization_residual,
    tensor_from_voigt,
    voigt_from_tensor,
)
from abacustools.io.stru import AbacusSTRU


WURTZITE = Structure.from_spacegroup(
    186,
    Lattice.hexagonal(3.82, 6.26),
    ["Zn", "S"],
    [[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, 0.375]],
)
ZINCBLENDE = Structure.from_spacegroup(
    216, Lattice.cubic(5.41), ["Zn", "S"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
)
ROCK_SALT = Structure.from_spacegroup(
    225, Lattice.cubic(5.64), ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)
TETRAGONAL = Structure.from_spacegroup(
    99,
    Lattice.tetragonal(3.99, 4.03),
    ["Ba", "Ti", "O"],
    [[0.5, 0.5, 0.5], [0.5, 0.5, 0.52], [0.5, 0.5, 0.02]],
)
TRIGONAL = Structure.from_spacegroup(
    161,
    Lattice.hexagonal(5.15, 13.86),
    ["Li", "Nb", "O"],
    [[0.0, 0.0, 0.2829], [0.0, 0.0, 0.0], [0.0492, 0.3417, 0.0635]],
)
TRICLINIC = Structure.from_spacegroup(
    1,
    Lattice.from_parameters(5.0, 6.0, 7.0, 80.0, 95.0, 110.0),
    ["Si", "O"],
    [[0.1, 0.2, 0.3], [0.4, 0.15, 0.6]],
)


def _rotations(structure: Structure) -> np.ndarray:
    """Return the point group rotations of a pymatgen structure."""
    return point_group_operations(AbacusSTRU.from_pymatgen(structure))


def test_tensor_round_trip_keeps_every_component() -> None:
    matrix = np.arange(1.0, 19.0).reshape(3, 6)

    tensor = tensor_from_voigt(matrix)

    assert tensor.shape == (3, 3, 3)
    np.testing.assert_allclose(tensor[:, 1, 2], tensor[:, 2, 1])
    np.testing.assert_allclose(voigt_from_tensor(tensor), matrix)


def test_tensor_from_voigt_rejects_a_wrong_shape() -> None:
    with pytest.raises(ValueError):
        tensor_from_voigt(np.zeros((6, 3)))


@pytest.mark.parametrize(
    ("structure", "expected"),
    [
        (WURTZITE, 3),
        (ZINCBLENDE, 1),
        (TETRAGONAL, 3),
        (TRIGONAL, 4),
        (TRICLINIC, 18),
        (ROCK_SALT, 0),
    ],
)
def test_number_of_independent_components_follows_the_point_group(
    structure: Structure, expected: int
) -> None:
    assert independent_component_count(_rotations(structure)) == expected


def test_centrosymmetric_crystals_have_no_piezoelectricity() -> None:
    rotations = _rotations(ROCK_SALT)
    rng = np.random.default_rng(5)
    noisy = rng.normal(scale=0.05, size=(3, 6))

    projected = project_tensor(noisy, rotations)

    assert independent_strain_modes(rotations) == []
    np.testing.assert_allclose(projected, np.zeros((3, 6)), atol=1e-12)


def test_strain_modes_of_the_piezoelectric_point_groups() -> None:
    assert independent_strain_modes(_rotations(WURTZITE)) == [0, 2, 3]
    assert independent_strain_modes(_rotations(ZINCBLENDE)) == [3]
    assert independent_strain_modes(_rotations(TETRAGONAL)) == [0, 2, 3]
    assert independent_strain_modes(_rotations(TRIGONAL)) == [0, 2, 3]


def _wurtzite_tensor() -> np.ndarray:
    """Return the 6mm form with the Materials Project values of ZnS."""
    matrix = np.zeros((3, 6), dtype=float)
    matrix[0, 4] = -0.05587
    matrix[1, 3] = -0.05588
    matrix[2, 0] = matrix[2, 1] = -0.06096
    matrix[2, 2] = 0.08304
    return matrix


def test_projection_enforces_the_relations_of_the_point_group() -> None:
    rotations = _rotations(WURTZITE)
    matrix = _wurtzite_tensor()

    projected = symmetrize_piezoelectric_tensor(matrix, rotations)

    assert projected[0, 4] == pytest.approx(projected[1, 3], abs=1e-6)
    assert projected[2, 1] == pytest.approx(projected[2, 0], abs=1e-6)
    for row, column in ((0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (2, 3), (2, 5)):
        assert projected[row, column] == pytest.approx(0.0, abs=1e-8)
    components = independent_components(projected, rotations)
    assert set(components) == {"d15", "d31", "d33"}
    # The two measurements of d15 differ by 1e-5 in the reference data.
    assert symmetrization_residual(matrix, projected) < 1e-5


def test_noisy_centrosymmetric_tensor_is_projected_to_zero() -> None:
    rotations = _rotations(ROCK_SALT)
    rng = np.random.default_rng(11)
    noisy = rng.normal(scale=0.01, size=(3, 6))

    projected = symmetrize_piezoelectric_tensor(noisy, rotations)

    assert symmetrization_residual(noisy, projected) > 0.0
    np.testing.assert_allclose(projected, np.zeros((3, 6)), atol=1e-12)
    assert independent_components(projected, rotations) == {}


def test_independent_fit_recovers_a_wurtzite_tensor_from_three_modes() -> None:
    rotations = _rotations(WURTZITE)
    matrix = _wurtzite_tensor()
    modes = independent_strain_modes(rotations)
    measured = np.zeros((3, 6), dtype=float)
    for mode in modes:
        measured[:, mode] = matrix[:, mode]

    fitted = fit_independent_piezoelectric(measured, modes, rotations)

    components = independent_components(fitted, rotations)
    assert set(components) == {"d15", "d31", "d33"}
    assert components["d33"] == pytest.approx(0.08304, abs=1e-6)
    assert components["d31"] == pytest.approx(-0.06096, abs=1e-6)
    # The reference d15 and d24 differ by 1e-5, which the fit resolves by
    # taking the measured column.
    assert components["d15"] == pytest.approx(-0.05588, abs=1e-6)
    # The duplicate column the fit never measured follows from the relation.
    assert fitted[1, 3] == pytest.approx(fitted[0, 4], abs=1e-9)
    assert fitted[2, 1] == pytest.approx(fitted[2, 0], abs=1e-9)


def test_independent_fit_rejects_too_few_measured_modes() -> None:
    rotations = _rotations(WURTZITE)
    measured = np.zeros((3, 6), dtype=float)
    measured[:, 0] = _wurtzite_tensor()[:, 0]

    with pytest.raises(ValueError, match="do not determine"):
        fit_independent_piezoelectric(measured, [0], rotations)
