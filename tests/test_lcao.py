from __future__ import annotations

import numpy as np
import pytest

from abacustools.data.lcao import (
    LCAODataError,
    embed_fragment_coefficients,
    embed_fragment_matrix,
    kpoint_density_to_real_space,
    load_lcao_kpoint_densities,
    real_space_density_to_kpoints,
    write_csr_matrix_blocks,
)
from abacustools.data.mayer import read_csr_matrix_blocks
from abacustools.io.stru import AbacusATOM, AbacusSTRU


def _structure() -> AbacusSTRU:
    return AbacusSTRU(
        cell=np.diag([4.0, 4.0, 4.0]),
        atoms=[
            AbacusATOM(label="H", coord=(0.0, 0.0, 0.0)),
            AbacusATOM(label="H", coord=(1.0, 0.0, 0.0)),
        ],
        metadata={"lattice_constant": 1.889726, "lattice_name": "test"},
    )


def test_write_csr_matrix_blocks_round_trip(tmp_path):
    path = tmp_path / "dmrs1_nao.csr"
    blocks = {
        (0, 0, 0): np.array([[1.0, 0.5], [0.5, 2.0]]),
        (1, 0, 0): np.array([[0.0, 0.25], [0.25, 0.0]]),
    }

    write_csr_matrix_blocks(path, blocks, structure=_structure(), nspin=1, spin=1)
    actual = read_csr_matrix_blocks(path)

    assert set(actual) == set(blocks)
    for key, expected in blocks.items():
        np.testing.assert_allclose(actual[key], expected, atol=1e-12)


def test_write_csr_matrix_blocks_preserves_complex_values(tmp_path):
    path = tmp_path / "dmrs1_nao.csr"
    expected = np.array([[1.0 + 0.0j, 0.5 + 0.25j], [0.5 - 0.25j, 2.0 + 0.0j]])

    write_csr_matrix_blocks(path, {(0, 0, 0): expected}, structure=_structure())
    actual = read_csr_matrix_blocks(path)[(0, 0, 0)]

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_write_csr_matrix_blocks_writes_empty_block(tmp_path):
    path = tmp_path / "dmrs1_nao.csr"
    write_csr_matrix_blocks(
        path,
        {(0, 0, 0): np.zeros((2, 2), dtype=float)},
        structure=_structure(),
    )

    text = path.read_text()
    assert " 0 0 0 0" in text
    assert "# CSR row pointers" in text
    actual = read_csr_matrix_blocks(path)[(0, 0, 0)]
    np.testing.assert_allclose(actual, 0.0)


def test_write_csr_matrix_blocks_rejects_non_square_matrix(tmp_path):
    with pytest.raises(ValueError, match="square"):
        write_csr_matrix_blocks(
            tmp_path / "dmrs1_nao.csr",
            {(0, 0, 0): np.zeros((2, 3))},
            structure=_structure(),
        )


def test_write_csr_matrix_blocks_rejects_invalid_spin(tmp_path):
    with pytest.raises(ValueError, match="spin"):
        write_csr_matrix_blocks(
            tmp_path / "dmrs1_nao.csr",
            {(0, 0, 0): np.eye(2)},
            structure=_structure(),
            nspin=1,
            spin=2,
        )


def test_kpoint_density_fourier_transform_round_trip_1d():
    d0 = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=complex)
    d1 = np.array([[0.5, -0.25j], [0.25j, 1.5]], dtype=complex)
    kpoints = [
        ((0.0, 0.0, 0.0), 0.5, d0),
        ((0.5, 0.0, 0.0), 0.5, d1),
    ]

    blocks = kpoint_density_to_real_space(kpoints, [(0, 0, 0), (1, 0, 0)])
    restored = real_space_density_to_kpoints(
        blocks,
        [((0.0, 0.0, 0.0), 0.5), ((0.5, 0.0, 0.0), 0.5)],
    )

    np.testing.assert_allclose(restored[0][2], d0, atol=1e-12)
    np.testing.assert_allclose(restored[1][2], d1, atol=1e-12)


def test_kpoint_density_fourier_transform_round_trip_3d():
    rng = np.random.default_rng(7)
    kpoints = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                direct = (0.5 * ix, 0.5 * iy, 0.5 * iz)
                matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                kpoints.append((direct, 0.125, matrix))
    r_blocks = [(rx, ry, rz) for rx in (0, 1) for ry in (0, 1) for rz in (0, 1)]

    blocks = kpoint_density_to_real_space(kpoints, r_blocks)
    restored = real_space_density_to_kpoints(
        blocks,
        [(direct, weight) for direct, weight, _matrix in kpoints],
    )

    for (_direct, _weight, original), (_restored_direct, _restored_weight, recovered) in zip(
        kpoints, restored
    ):
        np.testing.assert_allclose(recovered, original, atol=1e-12)


def test_kpoint_density_fourier_transform_gamma_is_identity():
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
    blocks = kpoint_density_to_real_space(
        [((0.0, 0.0, 0.0), 1.0, matrix)],
        [(0, 0, 0)],
    )
    restored = real_space_density_to_kpoints(
        blocks,
        [((0.0, 0.0, 0.0), 1.0)],
    )

    np.testing.assert_allclose(restored[0][2], matrix, atol=1e-12)


def test_kpoint_density_fourier_transform_rejects_bad_weights():
    with pytest.raises(ValueError, match="sum to one"):
        kpoint_density_to_real_space(
            [((0.0, 0.0, 0.0), 0.4, np.eye(2))],
            [(0, 0, 0)],
        )


def test_kpoint_density_fourier_transform_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        kpoint_density_to_real_space(
            [
                ((0.0, 0.0, 0.0), 0.5, np.eye(2)),
                ((0.5, 0.0, 0.0), 0.5, np.eye(3)),
            ],
            [(0, 0, 0)],
        )


def test_real_space_density_to_kpoints_uses_abacus_minus_phase():
    zero = np.array([[1.0, 0.0], [0.0, 1.0]])
    one = np.array([[0.75, 0.0], [0.0, 0.25]])
    result = real_space_density_to_kpoints(
        {(0, 0, 0): zero, (1, 0, 0): one},
        [((0.5, 0.0, 0.0), 1.0)],
    )

    np.testing.assert_allclose(result[0][2], zero - one, atol=1e-12)


def test_load_lcao_kpoint_densities_rejects_missing_job(tmp_path):
    with pytest.raises(LCAODataError, match="job directory"):
        load_lcao_kpoint_densities(tmp_path / "missing")


def test_load_lcao_kpoint_densities_rejects_missing_input(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    with pytest.raises(LCAODataError, match="INPUT"):
        load_lcao_kpoint_densities(job)


def test_embed_fragment_matrix_maps_selected_atom_blocks():
    full_ranges = [(0, 2), (2, 5), (5, 6)]
    fragment = np.arange(9, dtype=float).reshape(3, 3)
    embedded = embed_fragment_matrix(fragment, [0, 2], full_ranges)

    assert embedded.shape == (6, 6)
    np.testing.assert_allclose(embedded[0:2, 0:2], fragment[0:2, 0:2])
    np.testing.assert_allclose(embedded[0:2, 5:6], fragment[0:2, 2:3])
    np.testing.assert_allclose(embedded[5:6, 0:2], fragment[2:3, 0:2])
    np.testing.assert_allclose(embedded[5:6, 5:6], fragment[2:3, 2:3])
    np.testing.assert_allclose(embedded[2:5, :], 0.0)
    np.testing.assert_allclose(embedded[:, 2:5], 0.0)


def test_embed_fragment_coefficients_maps_rows_only():
    full_ranges = [(0, 2), (2, 5), (5, 6)]
    coefficients = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    embedded = embed_fragment_coefficients(coefficients, [1], full_ranges)

    assert embedded.shape == (6, 2)
    np.testing.assert_allclose(embedded[2:5, :], coefficients)
    np.testing.assert_allclose(embedded[0:2, :], 0.0)
    np.testing.assert_allclose(embedded[5:6, :], 0.0)


def test_embed_fragment_matrix_preserves_complex_dtype():
    fragment = np.array([[1.0 + 0.5j, 0.0], [0.0, 2.0 - 0.25j]])
    embedded = embed_fragment_matrix(fragment, [0], [(0, 2), (2, 3)])

    assert np.iscomplexobj(embedded)
    np.testing.assert_allclose(embedded[0:2, 0:2], fragment)


def test_embed_fragment_matrix_rejects_duplicate_atoms():
    with pytest.raises(ValueError, match="unique"):
        embed_fragment_matrix(np.eye(2), [0, 0], [(0, 2)])


def test_embed_fragment_matrix_rejects_out_of_range_atom():
    with pytest.raises(ValueError, match="outside"):
        embed_fragment_matrix(np.eye(2), [2], [(0, 2)])


def test_embed_fragment_matrix_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension"):
        embed_fragment_matrix(np.eye(3), [0], [(0, 2)])
