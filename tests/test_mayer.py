"""Tests for robust Mayer postprocessing readers."""

from __future__ import annotations

import numpy as np

from abacustools.data.mayer import (
    calculate_density_matrix_k,
    get_nao_basis_num,
    read_density_matrix,
    read_kpoint_weights,
    read_nao_file,
    read_overlap_matrix,
    read_wfc_nao_k,
)


def test_read_nao_file_uses_summary_and_mesh(tmp_path):
    orbital = tmp_path / "H.orb"
    orbital.write_text(
        """Element H
Energy Cutoff(Ry) 100
Radius Cutoff(a.u.) 6
Lmax 0
Number of Sorbital--> 1
SUMMARY  END

Mesh 3
dr 0.01
Type L N
0 0 0
1.0D+00 2.0 3.0
""",
        encoding="utf-8",
    )
    data = read_nao_file(orbital)
    assert data.element == "H"
    assert data.orbitals_per_l == (1,)
    assert get_nao_basis_num(data) == 1
    np.testing.assert_allclose(data.orbitals[0].values, [1, 2, 3])


def test_read_overlap_matrix_parses_real_and_complex_triangles(tmp_path):
    real = tmp_path / "real-S"
    real.write_text("3 1 0.2 0.3\n1.1 0\n1\n", encoding="utf-8")
    np.testing.assert_allclose(read_overlap_matrix(real), [[1, 0.2, 0.3], [0.2, 1.1, 0], [0.3, 0, 1]])

    complex_matrix = tmp_path / "complex-S"
    complex_matrix.write_text("2 (1,0) (0.1,0.2)\n(1,0)\n", encoding="utf-8")
    result = read_overlap_matrix(complex_matrix)
    assert result[0, 1] == 0.1 + 0.2j
    assert result[1, 0] == 0.1 - 0.2j


def test_read_density_matrix_ignores_abacus_header(tmp_path):
    density = tmp_path / "SPIN1_DM"
    density.write_text("none\n1\n10 0 0\n\n1\n-0.2 (fermi energy)\n\n2 2\n1 2\n3 4\n", encoding="utf-8")
    np.testing.assert_allclose(read_density_matrix(density), [[1, 2], [3, 4]])


def test_read_wfc_and_kpoint_weights(tmp_path):
    wfc = tmp_path / "WFC_NAO_K1.txt"
    wfc.write_text(
        """1 (index of k points)
1 (number of bands)
2 (number of orbitals)
1 (band)
0 (Ry)
0.5 (Occupations)
1 0 0.5 -0.5
""",
        encoding="utf-8",
    )
    coefficients, occupations = read_wfc_nao_k(wfc)
    np.testing.assert_allclose(coefficients[:, 0], [1, 0.5 - 0.5j])
    np.testing.assert_allclose(calculate_density_matrix_k(coefficients, occupations), [[0.5, 0.25 + 0.25j], [0.25 - 0.25j, 0.25]])

    kpoints = tmp_path / "kpoints"
    kpoints.write_text("nkstot now = 2\nKPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT\n1 0 0 0 0.5\n2 0.5 0 0 0.5\n", encoding="utf-8")
    np.testing.assert_allclose(read_kpoint_weights(kpoints), [0.5, 0.5])
