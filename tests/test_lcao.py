from __future__ import annotations

import numpy as np
import pytest

from abacustools.data.lcao import write_csr_matrix_blocks
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
