"""Tests for the symmetry-aware elastic tensor helpers."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from abacustools.data.elastic import (
    deformation_gradient,
    directional_moduli_summary,
    elastic_component_names,
    elastic_moduli,
    energy_strain_patterns,
    fit_energy_strain,
    fit_independent_stress_strain,
    fit_stress_strain,
    independent_component_count,
    independent_components,
    independent_strain_modes,
    in_plane_voigt_indices,
    matrix_from_voigt,
    point_group_operations,
    project_tensor,
    stress_voigt,
    strain_matrix,
    symmetrize_elastic_tensor,
    symmetrization_residual,
    tensor4_from_voigt,
    two_dimensional_moduli,
    two_dimensional_tensor,
    voigt_from_tensor4,
)
from abacustools.io.stru import AbacusSTRU


CUBIC = Structure.from_spacegroup(227, Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])
HEXAGONAL = Structure.from_spacegroup(
    186,
    Lattice.hexagonal(3.82, 6.26),
    ["Zn", "S"],
    [[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, 0.375]],
)
TETRAGONAL = Structure.from_spacegroup(
    136,
    Lattice.tetragonal(4.59, 2.96),
    ["Ti", "O"],
    [[0.0, 0.0, 0.0], [0.305, 0.305, 0.0]],
)
TETRAGONAL_LOW = Structure.from_spacegroup(
    75, Lattice.tetragonal(4.59, 2.96), ["Si"], [[0.1, 0.2, 0.3]]
)
TRIGONAL = Structure.from_spacegroup(
    154,
    Lattice.hexagonal(4.9134, 5.4052),
    ["Si", "O"],
    [[0.4697, 0.0, 0.0], [0.4135, 0.2669, 0.1191]],
)
TRIGONAL_LOW = Structure.from_spacegroup(
    147,
    Lattice.hexagonal(4.9, 5.4),
    ["Si", "O"],
    [[0.1, 0.2, 0.3], [0.4, 0.15, 0.6]],
)
ORTHORHOMBIC = Structure.from_spacegroup(
    62, Lattice.orthorhombic(5.0, 6.0, 7.0), ["Si"], [[0.1, 0.2, 0.3]]
)
MONOCLINIC = Structure.from_spacegroup(
    14, Lattice.monoclinic(5.0, 6.0, 7.0, 100.0), ["Si"], [[0.1, 0.2, 0.3]]
)
TRICLINIC = Structure.from_spacegroup(
    1,
    Lattice.from_parameters(5.0, 6.0, 7.0, 80.0, 95.0, 110.0),
    ["Si"],
    [[0.1, 0.2, 0.3]],
)


def _rotations(structure: Structure) -> np.ndarray:
    """Return the point group rotations of a pymatgen structure."""
    rotations = point_group_operations(AbacusSTRU.from_pymatgen(structure))
    return rotations


def _cubic_tensor(c11: float = 161.0, c12: float = 64.0, c44: float = 76.0) -> np.ndarray:
    """Return the elastic tensor of a cubic crystal, in GPa."""
    tensor = np.zeros((6, 6), dtype=float)
    tensor[:3, :3] = c12
    np.fill_diagonal(tensor[:3, :3], c11)
    tensor[3, 3] = tensor[4, 4] = tensor[5, 5] = c44
    return tensor


def _relaxed_like(structure: Structure, strain: float = 2.0e-4) -> Structure:
    """Return a structure whose cell is only symmetric to a few 1e-4."""
    cell = np.array(structure.lattice.matrix, dtype=float)
    cell[0] = cell[0] * (1.0 + strain)
    return Structure(Lattice(cell), structure.species, structure.frac_coords)


def test_elastic_component_names_lists_the_upper_triangle() -> None:
    names = elastic_component_names()

    assert len(names) == 21
    assert names[0] == "C11"
    assert names[5] == "C16"
    assert names[-1] == "C66"
    assert names.count("C44") == 1


def test_voigt_round_trip_keeps_every_component() -> None:
    matrix = np.arange(1.0, 37.0).reshape(6, 6)

    assert np.allclose(matrix_from_voigt(matrix), matrix)

    tensor = tensor4_from_voigt(matrix)
    assert tensor.shape == (3, 3, 3, 3)
    # C_44 abbreviates pairs of yz and zx directions, with both minor
    # symmetries filled in.
    assert tensor[1, 2, 1, 2] == pytest.approx(matrix[3, 3])
    assert tensor[2, 1, 1, 2] == pytest.approx(matrix[3, 3])
    assert tensor[1, 2, 2, 1] == pytest.approx(matrix[3, 3])
    assert tensor[0, 1, 0, 2] == pytest.approx(matrix[5, 4])
    assert np.allclose(voigt_from_tensor4(tensor), matrix)


def test_matrix_from_voigt_accepts_the_independent_components() -> None:
    values = [float(index) for index in range(21)]

    matrix = matrix_from_voigt(values)

    assert matrix.shape == (6, 6)
    assert np.allclose(matrix, matrix.T)
    assert matrix[0, 0] == 0.0
    assert matrix[0, 1] == 1.0
    assert matrix[0, 5] == 5.0
    assert matrix[5, 5] == 20.0


def test_matrix_from_voigt_rejects_a_wrong_length() -> None:
    with pytest.raises(ValueError):
        matrix_from_voigt([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("structure", "expected"),
    [
        (CUBIC, 3),
        (HEXAGONAL, 5),
        (TETRAGONAL, 6),
        (TETRAGONAL_LOW, 7),
        (TRIGONAL, 6),
        (TRIGONAL_LOW, 7),
        (ORTHORHOMBIC, 9),
        (MONOCLINIC, 13),
        (TRICLINIC, 21),
    ],
)
def test_number_of_independent_constants_follows_the_laue_class(
    structure: Structure, expected: int
) -> None:
    assert independent_component_count(_rotations(structure)) == expected


def test_projection_is_idempotent() -> None:
    rotations = _rotations(HEXAGONAL)
    rng = np.random.default_rng(2024)
    noisy = rng.normal(scale=10.0, size=(6, 6))
    noisy = 0.5 * (noisy + noisy.T)

    once = project_tensor(noisy, rotations)
    twice = project_tensor(once, rotations)

    assert np.allclose(once, twice, atol=1e-10)
    assert np.allclose(once, once.T, atol=1e-10)


def test_symmetrization_removes_forbidden_cubic_components() -> None:
    rotations = _rotations(CUBIC)
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=0.05, size=(6, 6))
    noise = 0.5 * (noise + noise.T)
    raw = _cubic_tensor() + noise

    symmetrized = symmetrize_elastic_tensor(raw, rotations)

    assert np.allclose(symmetrized, _cubic_tensor(), atol=0.05)
    # Every component the cubic symmetry forbids is gone.
    assert symmetrized[0, 3] == pytest.approx(0.0, abs=1e-8)
    assert symmetrized[0, 5] == pytest.approx(0.0, abs=1e-8)
    assert symmetrized[3, 4] == pytest.approx(0.0, abs=1e-8)
    assert symmetrized[3, 3] == pytest.approx(symmetrized[4, 4])
    assert symmetrized[3, 3] == pytest.approx(symmetrized[5, 5])
    assert symmetrization_residual(raw, symmetrized) > 0.0
    assert set(independent_components(symmetrized, rotations)) == {"C11", "C12", "C44"}


def test_symmetrization_keeps_the_trigonal_c14_of_quartz() -> None:
    """The -3m class keeps C14, which must not be projected away."""
    rotations = _rotations(TRIGONAL)
    c11, c12, c13, c14, c33, c44 = 75.0, 1.0, 6.0, 19.0, 88.0, 52.0
    c66 = 0.5 * (c11 - c12)
    tensor = np.zeros((6, 6), dtype=float)
    tensor[:3, :3] = np.array([[c11, c12, c13], [c12, c11, c13], [c13, c13, c33]])
    tensor[0, 3] = tensor[3, 0] = c14
    tensor[1, 3] = tensor[3, 1] = -c14
    tensor[3, 3] = tensor[4, 4] = c44
    tensor[4, 5] = tensor[5, 4] = c14
    tensor[5, 5] = c66

    symmetrized = symmetrize_elastic_tensor(tensor, rotations)

    assert np.allclose(symmetrized, tensor, atol=1e-8)
    components = independent_components(symmetrized, rotations)
    assert components["C14"] == pytest.approx(c14, abs=1e-6)
    assert "C15" not in components


def test_relaxed_cell_still_gives_the_textbook_component_set() -> None:
    """A cell relaxed to a few 1e-4 must not lose C44 to a leaked C15."""
    relaxed = _relaxed_like(TRIGONAL)
    rotations = point_group_operations(
        AbacusSTRU.from_pymatgen(relaxed), symprec=1e-2
    )
    c11, c12, c13, c14, c33, c44 = 75.0, 1.0, 6.0, 19.0, 88.0, 52.0
    tensor = np.zeros((6, 6), dtype=float)
    tensor[:3, :3] = np.array([[c11, c12, c13], [c12, c11, c13], [c13, c13, c33]])
    tensor[0, 3] = tensor[3, 0] = c14
    tensor[1, 3] = tensor[3, 1] = -c14
    tensor[3, 3] = tensor[4, 4] = c44
    tensor[4, 5] = tensor[5, 4] = c14
    tensor[5, 5] = 0.5 * (c11 - c12)

    symmetrized = symmetrize_elastic_tensor(tensor, rotations)
    components = independent_components(symmetrized, rotations)

    assert independent_component_count(rotations) == 6
    assert set(components) == {"C11", "C12", "C13", "C14", "C33", "C44"}
    assert components["C44"] == pytest.approx(c44, rel=0.05)


@pytest.mark.parametrize(
    ("structure", "expected"),
    [
        (CUBIC, [0, 3]),
        (HEXAGONAL, [0, 2, 3]),
        (TETRAGONAL, [0, 2, 3, 5]),
        (TETRAGONAL_LOW, [0, 2, 3, 5]),
        (TRIGONAL, [0, 2, 3]),
        (ORTHORHOMBIC, [0, 1, 2, 3, 4, 5]),
        (MONOCLINIC, [0, 1, 2, 3, 4, 5]),
        (TRICLINIC, [0, 1, 2, 3, 4, 5]),
    ],
)
def test_strain_modes_cover_the_independent_constants(
    structure: Structure, expected: list[int]
) -> None:
    rotations = _rotations(structure)

    modes = independent_strain_modes(rotations)

    assert modes == expected
    assert len(modes) <= 6
    assert len(modes) >= 2


def _reduced_strain_stress(tensor: np.ndarray, modes, amplitude: float = 0.01):
    """Return strains and stresses for one direction per symmetry orbit."""
    strains, stresses = [], []
    for mode in modes:
        for scale in (-amplitude, -0.5 * amplitude, 0.5 * amplitude, amplitude):
            strain = np.zeros(6, dtype=float)
            strain[mode] = scale
            strains.append(strain)
            stresses.append(tensor @ strain)
    return np.asarray(strains), np.asarray(stresses)


def test_independent_fit_recovers_a_cubic_tensor_from_two_modes() -> None:
    rotations = _rotations(CUBIC)
    tensor = _cubic_tensor()
    modes = independent_strain_modes(rotations)
    strains, stresses = _reduced_strain_stress(tensor, modes)

    fitted = fit_independent_stress_strain(strains, stresses, rotations)

    assert len(modes) == 2
    assert len(strains) == 8
    np.testing.assert_allclose(fitted, tensor, atol=1e-10)


def test_independent_fit_rebuilds_the_derived_hexagonal_c66() -> None:
    """Three modes fix C11, C12, C13, C33, C44 and hence C66."""
    rotations = _rotations(HEXAGONAL)
    c11, c12, c13, c33, c44 = 124.0, 55.6, 41.0, 147.6, 28.7
    tensor = np.zeros((6, 6), dtype=float)
    tensor[:3, :3] = np.array([[c11, c12, c13], [c12, c11, c13], [c13, c13, c33]])
    tensor[3, 3] = tensor[4, 4] = c44
    tensor[5, 5] = 0.5 * (c11 - c12)
    modes = independent_strain_modes(rotations)
    strains, stresses = _reduced_strain_stress(tensor, modes)

    fitted = fit_independent_stress_strain(strains, stresses, rotations)

    assert modes == [0, 2, 3]
    np.testing.assert_allclose(fitted, tensor, atol=1e-8)
    assert fitted[5, 5] == pytest.approx(0.5 * (fitted[0, 0] - fitted[0, 1]))


def test_independent_fit_rejects_an_incomplete_strain_set() -> None:
    rotations = _rotations(CUBIC)
    tensor = _cubic_tensor()
    strains, stresses = _reduced_strain_stress(tensor, [0])

    with pytest.raises(ValueError, match="do not determine"):
        fit_independent_stress_strain(strains, stresses, rotations)


def test_deformation_gradient_reproduces_the_lagrangian_strain() -> None:
    for strain in (
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
        [0.01, 0.01, 0.0, 0.0, 0.0, 0.0],
    ):
        gradient = deformation_gradient(strain)
        recovered = 0.5 * (gradient.T @ gradient - np.eye(3))

        np.testing.assert_allclose(recovered, strain_matrix(strain), atol=1e-12)


@pytest.mark.parametrize(
    ("structure", "count"),
    [
        (CUBIC, 3),
        (HEXAGONAL, 5),
        (TETRAGONAL, 6),
        (TRIGONAL, 6),
        (ORTHORHOMBIC, 9),
    ],
)
def test_energy_patterns_cover_every_independent_constant(
    structure: Structure, count: int
) -> None:
    rotations = _rotations(structure)

    patterns = energy_strain_patterns(rotations)

    assert len(patterns) == count
    for pattern in patterns:
        assert np.linalg.norm(pattern) == pytest.approx(1.0)


def test_energy_strain_recovers_a_cubic_tensor_and_the_reference_stress() -> None:
    from abacustools.data.eos import EV_PER_ANGSTROM3_TO_GPA

    rotations = _rotations(CUBIC)
    volume = 5.43**3
    tensor = _cubic_tensor()
    reference_stress = np.array([0.3, -0.2, 0.1, 0.05, 0.0, 0.0])
    patterns = energy_strain_patterns(rotations)
    strains, energies = [], []
    for pattern in patterns:
        for amplitude in (-0.01, -0.005, 0.005, 0.01):
            strain = pattern * amplitude
            strains.append(strain)
            energies.append(
                volume * reference_stress @ strain / EV_PER_ANGSTROM3_TO_GPA
                + 0.5
                * volume
                * (strain @ tensor @ strain)
                / EV_PER_ANGSTROM3_TO_GPA
            )

    fitted, stresses, residual = fit_energy_strain(
        np.asarray(strains), np.asarray(energies), volume, rotations
    )

    np.testing.assert_allclose(fitted, tensor, atol=1e-8)
    np.testing.assert_allclose(
        stresses, [pattern @ reference_stress for pattern in patterns], atol=1e-10
    )
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_energy_strain_rejects_a_strain_set_that_is_too_small() -> None:
    rotations = _rotations(CUBIC)
    volume = 5.43**3
    tensor = _cubic_tensor()
    strains, energies = [], []
    for amplitude in (-0.01, -0.005, 0.005, 0.01):
        strain = np.zeros(6, dtype=float)
        strain[0] = amplitude
        strains.append(strain)
        energies.append(0.5 * volume * (strain @ tensor @ strain) / 160.21766208)

    with pytest.raises(ValueError):
        fit_energy_strain(
            np.asarray(strains), np.asarray(energies), volume, rotations
        )


@pytest.mark.parametrize(
    ("axis", "expected"),
    [(0, [1, 2, 3]), (1, [0, 2, 4]), (2, [0, 1, 5])],
)
def test_in_plane_modes_of_a_slab(axis: int, expected: list[int]) -> None:
    assert in_plane_voigt_indices(axis) == expected


def test_two_dimensional_block_scales_to_newton_per_metre() -> None:
    tensor = np.zeros((6, 6), dtype=float)
    tensor[0, 0] = tensor[1, 1] = 100.0
    tensor[0, 1] = tensor[1, 0] = 20.0
    tensor[5, 5] = 40.0
    tensor[2, 2] = 3.0  # vacuum direction, must not reach the block
    tensor[3, 3] = tensor[4, 4] = 0.5

    block = two_dimensional_tensor(tensor, 2, height=20.0)

    np.testing.assert_allclose(
        block, [[200.0, 40.0, 0.0], [40.0, 200.0, 0.0], [0.0, 0.0, 80.0]]
    )
    assert two_dimensional_tensor(tensor, 2).shape == (3, 3)
    assert two_dimensional_tensor(tensor, 0).shape == (3, 3)


def test_hexagonal_layer_has_an_isotropic_in_plane_response() -> None:
    c11, c12 = 350.0, 60.0
    block = np.array(
        [
            [c11, c12, 0.0],
            [c12, c11, 0.0],
            [0.0, 0.0, 0.5 * (c11 - c12)],
        ]
    )

    summary = directional_moduli_summary(two_dimensional_moduli(block))

    assert summary["anisotropy"] == pytest.approx(1.0, abs=1e-9)
    assert summary["young_modulus_max"] == pytest.approx(
        (c11**2 - c12**2) / c11, rel=1e-9
    )
    assert summary["poisson_ratio_max"] == pytest.approx(c12 / c11, rel=1e-9)
    assert summary["poisson_ratio_min"] == pytest.approx(c12 / c11, rel=1e-9)


def test_oblique_layer_is_anisotropic() -> None:
    block = np.array(
        [
            [120.0, 30.0, 12.0],
            [30.0, 140.0, -8.0],
            [12.0, -8.0, 45.0],
        ]
    )

    summary = directional_moduli_summary(two_dimensional_moduli(block))

    assert summary["anisotropy"] > 1.2
    assert summary["young_modulus_max_angle"] != summary["young_modulus_min_angle"]


def test_fit_can_restrict_the_components_to_the_in_plane_set() -> None:
    tensor = np.zeros((6, 6), dtype=float)
    tensor[0, 0] = tensor[1, 1] = 120.0
    tensor[0, 1] = tensor[1, 0] = 30.0
    tensor[5, 5] = 45.0
    tensor[2, 2] = 7.0
    tensor[3, 3] = tensor[4, 4] = 2.0
    strains, stresses = [], []
    for component in (0, 1, 5):
        for amplitude in (-0.01, -0.005, 0.005, 0.01):
            strain = np.zeros(6, dtype=float)
            strain[component] = amplitude
            strains.append(strain)
            stresses.append(tensor @ strain)

    fitted = fit_stress_strain(
        np.asarray(strains), np.asarray(stresses), indices=[0, 1, 5]
    )

    np.testing.assert_allclose(fitted[:2, :2], tensor[:2, :2], atol=1e-10)
    assert fitted[5, 5] == pytest.approx(45.0)
    assert fitted[2, 2] == 0.0
    assert fitted[3, 3] == 0.0


def test_independent_modes_can_be_restricted_to_the_plane() -> None:
    rotations = _rotations(HEXAGONAL)

    modes = independent_strain_modes(rotations, allowed=[0, 1, 5])

    # In the plane the hexagonal cell has C66 = (C11 - C12) / 2, so a single
    # normal strain determines both independent constants.
    assert modes == [0]
    assert independent_component_count(rotations, components=[0, 1, 5]) == 2
    assert independent_component_count(rotations) == 5  # the three dimensional count


def test_hexagonal_projection_imposes_the_c66_relation() -> None:
    rotations = _rotations(HEXAGONAL)
    c11, c12, c13, c33, c44 = 130.0, 41.0, 45.0, 145.0, 40.0
    tensor = np.zeros((6, 6), dtype=float)
    tensor[:3, :3] = np.array([[c11, c12, c13], [c12, c11, c13], [c13, c13, c33]])
    tensor[3, 3] = tensor[4, 4] = c44
    tensor[5, 5] = 999.0  # C66 is not independent for a hexagonal crystal

    symmetrized = symmetrize_elastic_tensor(tensor, rotations)

    # The projection moves the tensor along the invariant directions, so C11
    # and C12 change too; the hexagonal relations are what it has to satisfy.
    assert symmetrized[0, 0] == pytest.approx(symmetrized[1, 1])
    assert symmetrized[0, 2] == pytest.approx(symmetrized[1, 2])
    assert symmetrized[3, 3] == pytest.approx(symmetrized[4, 4])
    assert symmetrized[5, 5] == pytest.approx(
        0.5 * (symmetrized[0, 0] - symmetrized[0, 1])
    )
    assert symmetrized[0, 3] == pytest.approx(0.0, abs=1e-8)
    assert symmetrized[3, 4] == pytest.approx(0.0, abs=1e-8)
    components = independent_components(symmetrized, rotations)
    # C66 follows from C11 and C12 for every hexagonal class, so it is not
    # listed among the independent constants.
    assert set(components) == {"C11", "C12", "C13", "C33", "C44"}
    assert components["C11"] == pytest.approx(symmetrized[0, 0])
    assert components["C12"] == pytest.approx(symmetrized[0, 1])


def test_fit_follows_the_stress_strain_convention() -> None:
    expected = np.array(
        [
            [100.0, 20.0, 30.0, 4.0, 5.0, 6.0],
            [21.0, 110.0, 25.0, 7.0, 8.0, 9.0],
            [31.0, 26.0, 120.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 40.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 45.0, 19.0],
            [22.0, 23.0, 24.0, 27.0, 28.0, 50.0],
        ]
    )
    strains = np.zeros((12, 6), dtype=float)
    for component in range(6):
        strains[2 * component, component] = 0.01
        strains[2 * component + 1, component] = -0.005
    equilibrium = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    stresses = strains @ expected.T + equilibrium

    fitted = fit_stress_strain(strains, stresses, equilibrium_stress=equilibrium)

    assert np.allclose(fitted, expected, atol=1e-10)
    assert not np.allclose(fitted, fitted.T)


def test_fit_reports_the_equilibrium_stress_without_shifting_the_slopes() -> None:
    strains = np.zeros((12, 6), dtype=float)
    for component in range(6):
        strains[2 * component, component] = 0.01
        strains[2 * component + 1, component] = -0.01
    tensor = _cubic_tensor()
    stresses = strains @ tensor.T + np.array([5.0, -3.0, 2.0, 1.0, -1.0, 0.5])

    fitted = fit_stress_strain(
        strains, stresses, equilibrium_stress=np.array([5.0, -3.0, 2.0, 1.0, -1.0, 0.5])
    )

    assert np.allclose(fitted, tensor, atol=1e-10)


def test_fit_needs_two_strain_states_per_component() -> None:
    strains = np.zeros((6, 6), dtype=float)
    np.fill_diagonal(strains, 0.01)

    with pytest.raises(ValueError, match="insufficient strain data"):
        fit_stress_strain(strains, np.zeros((6, 6)))


def test_noisy_fit_is_repaired_by_the_symmetrisation() -> None:
    rng = np.random.default_rng(11)
    tensor = _cubic_tensor()
    strains = np.zeros((24, 6), dtype=float)
    index = 0
    for component in range(6):
        for strain in (-0.01, -0.005, 0.005, 0.01):
            strains[index, component] = strain
            index += 1
    stresses = strains @ tensor.T + rng.normal(scale=0.02, size=(24, 6))

    raw = fit_stress_strain(strains, stresses)
    rotations = _rotations(CUBIC)
    symmetrized = symmetrize_elastic_tensor(raw, rotations)

    assert symmetrization_residual(raw, symmetrized) > 0.0
    np.testing.assert_allclose(symmetrized, tensor, atol=1.0)
    assert abs(raw[0, 3]) > 0.0
    assert symmetrized[0, 3] == pytest.approx(0.0, abs=1e-8)


def test_stress_voigt_follows_abacus_ordering() -> None:
    stress = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])

    assert stress_voigt(stress) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_elastic_moduli_use_the_voigt_average() -> None:
    tensor = np.array(
        [
            [100.0, 20.0, 30.0, 0.0, 0.0, 0.0],
            [20.0, 110.0, 25.0, 0.0, 0.0, 0.0],
            [30.0, 25.0, 120.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 40.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 45.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
        ]
    )

    result = elastic_moduli(tensor)

    assert result["bulk_modulus"] == pytest.approx(53.333333333333336)
    assert result["shear_modulus"] == pytest.approx(44.0)
    assert result["young_modulus"] == pytest.approx(103.52941176470588)
    assert result["poisson_ratio"] == pytest.approx(0.17647058823529413)
