"""Tests for robust Mayer postprocessing readers."""

from __future__ import annotations

import numpy as np
import pytest

from abacustools.data.mayer import (
    _develop_density_files,
    _develop_overlap_file,
    analyze_mayer_bond_order,
    calculate_density_matrix_k,
    detect_matrix_format,
    get_nao_basis_num,
    read_density_matrix,
    read_density_matrix_develop,
    read_kpoint_weights,
    read_nao_file,
    read_overlap_matrix,
    read_overlap_matrix_develop,
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


def test_read_overlap_matrix_develop_upper_triangle(tmp_path):
    """The develop ``sk`` file wraps each triangular row over several lines."""

    matrix = tmp_path / "sk1_nao.txt"
    matrix.write_text(
        """#------------------------------------------------------------------------
# ionic step 1
# filename OUT.ABACUS/sk1_nao.txt
# gamma only 0
# rows 3
# columns 3
#------------------------------------------------------------------------
Row 1
 (1.0,0.0) (0.2,0.3) (0.4,0.0)
Row 2
 (1.1,0.0) (0.5,0.0)
Row 3
 (1.2,0.0)
""",
        encoding="utf-8",
    )
    result = read_overlap_matrix_develop(matrix)
    np.testing.assert_allclose(np.diag(result).real, [1.0, 1.1, 1.2])
    assert result[0, 1] == 0.2 + 0.3j
    assert result[1, 0] == 0.2 - 0.3j
    np.testing.assert_allclose(result[0, 2], 0.4)
    np.testing.assert_allclose(result[:, :], result.conj().T)


def test_read_overlap_matrix_develop_accepts_plain_reals(tmp_path):
    """Gamma-only runs write real numbers without the ``(real,imag)`` wrapper."""

    matrix = tmp_path / "sk_nao.txt"
    matrix.write_text(
        """# rows 2
# columns 2
Row 1
 1.0 0.25
Row 2
 1.5
""",
        encoding="utf-8",
    )
    np.testing.assert_allclose(read_overlap_matrix_develop(matrix), [[1.0, 0.25], [0.25, 1.5]])


def test_read_density_matrix_develop_skips_structure_block(tmp_path):
    """The develop density matrix follows a structure block, eight values per line."""

    density = tmp_path / "dmk1g1_nao.txt"
    density.write_text(
        """ --- Ionic Step 1 ---
 1 # number of spin directions
 1 # spin index
 1 # total k points
 1 # total k points after symmetrized (if open)
 1 # k-point index
 0 0 0 # k point coordinate (Cartesian)
 0 0 0 # k point coordinate (direct)
 2 # weight of this k point
 -0.2 # Fermi energy in Ry
 4 # number of localized basis
 2 2 # size of this matrix

 user_defined_lattice
 1
 10 0 0
 0 10 0
 0 0 10
 H
 2
 Direct
 0.0 0.0 0.0
 0.0 0.0 0.7

 (1.0,0.0) (0.0,0.0)
 (0.0,0.0) (2.0,0.0)
""",
        encoding="utf-8",
    )
    np.testing.assert_allclose(read_density_matrix_develop(density), [[1, 0], [0, 2]])


def test_develop_density_files_read_k_and_spin_indices(tmp_path):
    """The k- and spin-index come from the file name, which varies by run type."""

    for name in [
        "dmg1_nao.txt",
        "dms1g1_nao.txt",
        "dms2g1_nao.txt",
        "dmk1g1_nao.txt",
        "dmk2s1g1_nao.txt",
        "dmk2s2g1_nao.txt",
        "sk1_nao.txt",
    ]:
        (tmp_path / name).write_text("", encoding="utf-8")

    assert _develop_density_files(tmp_path) == [
        (1, 1, tmp_path / "dmg1_nao.txt"),
        (1, 1, tmp_path / "dmk1g1_nao.txt"),
        (1, 1, tmp_path / "dms1g1_nao.txt"),
        (1, 2, tmp_path / "dms2g1_nao.txt"),
        (2, 1, tmp_path / "dmk2s1g1_nao.txt"),
        (2, 2, tmp_path / "dmk2s2g1_nao.txt"),
    ]


def test_develop_overlap_file_prefers_gamma_for_single_k(tmp_path):
    (tmp_path / "sk_nao.txt").write_text("", encoding="utf-8")
    assert _develop_overlap_file(tmp_path, 1) == tmp_path / "sk_nao.txt"


def test_develop_overlap_file_uses_k_index_for_multi_k(tmp_path):
    (tmp_path / "sk3_nao.txt").write_text("", encoding="utf-8")
    assert _develop_overlap_file(tmp_path, 3) == tmp_path / "sk3_nao.txt"


def test_detect_matrix_format_prefers_requested_layout(tmp_path):
    (tmp_path / "dmk1g1_nao.txt").write_text("", encoding="utf-8")
    (tmp_path / "data-0-S").write_text("", encoding="utf-8")

    assert detect_matrix_format(tmp_path, out_dmk=1) == "develop"
    assert detect_matrix_format(tmp_path, out_dmk=0) == "lts"


def test_detect_matrix_format_falls_back_to_available_data(tmp_path):
    (tmp_path / "dmk1g1_nao.txt").write_text("", encoding="utf-8")
    assert detect_matrix_format(tmp_path, out_dmk=0) == "develop"

    lts_only = tmp_path / "lts"
    lts_only.mkdir()
    (lts_only / "data-0-S").write_text("", encoding="utf-8")
    assert detect_matrix_format(lts_only, out_dmk=1) == "lts"


def _write_develop_h2_job(root):
    """Write a two-orbital synthetic job in the develop output layout."""

    (root / "H.orb").write_text(
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
1.0 2.0 3.0
""",
        encoding="utf-8",
    )
    (root / "H.upf").write_text("", encoding="utf-8")
    (root / "STRU").write_text(
        """ATOMIC_SPECIES
H 1.008 H.upf

NUMERICAL_ORBITAL
H.orb

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
10 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Cartesian

H
0.0
2
0.0 0.0 0.0 1 1 1
0.0 0.0 0.7 1 1 1
""",
        encoding="utf-8",
    )
    (root / "INPUT").write_text(
        """INPUT_PARAMETERS
calculation scf
basis_type lcao
nspin 1
gamma_only 1
out_mat_hs 1
out_dmk 1
""",
        encoding="utf-8",
    )

    output = root / "OUT.ABACUS"
    output.mkdir()
    # S = [[1, 0.5], [0.5, 1]] in the develop upper-triangular layout.
    (output / "sk_nao.txt").write_text(
        """#------------------------------------------------------------------------
# ionic step 1
# filename OUT.ABACUS/sk_nao.txt
# gamma only 1
# rows 2
# columns 2
#------------------------------------------------------------------------
Row 1
 1.0 0.5
Row 2
 1.0
""",
        encoding="utf-8",
    )
    # D = [[1, 0.25], [0.25, 1]] behind a structure block.
    (output / "dmg1_nao.txt").write_text(
        """ --- Ionic Step 1 ---
 1 # number of spin directions
 1 # spin index
 1 # total k points
 1 # total k points after symmetrized (if open)
 1 # k-point index
 0 0 0 # k point coordinate (Cartesian)
 0 0 0 # k point coordinate (direct)
 2 # weight of this k point
 -0.2 # Fermi energy in Ry
 2 # number of localized basis
 2 2 # size of this matrix

 user_defined_lattice
 1
 10 0 0
 0 10 0
 0 0 10
 H
 2
 Direct
 0.0 0.0 0.0
 0.0 0.0 0.07

 1.0 0.25
 0.25 1.0
""",
        encoding="utf-8",
    )


def test_analyze_mayer_bond_order_reads_develop_job(tmp_path):
    """A develop-layout job yields the analytic Mayer order of the matrices."""

    _write_develop_h2_job(tmp_path)
    analysis = analyze_mayer_bond_order(tmp_path)

    assert analysis.gamma_only is True
    assert analysis.nspin == 1
    assert analysis.basis_functions == 2
    assert len(analysis.pairs) == 1
    pair = analysis.pairs[0]
    # (D S)[0, 1] = (D S)[1, 0] = 0.75 for the synthetic matrices above.
    np.testing.assert_allclose(pair.bond_order, 0.5625)
    np.testing.assert_allclose(pair.distance, 0.7)


def _stru(cell, fractional):
    """Build a structure whose atoms sit at the given fractional coordinates."""
    from abacustools.io.stru import AbacusATOM, AbacusSTRU

    cell = np.asarray(cell, dtype=float)
    return AbacusSTRU(
        cell=cell.tolist(),
        atoms=[
            AbacusATOM(
                label="H",
                element="H",
                coord=tuple((np.asarray(coords, dtype=float) @ cell).tolist()),
            )
            for coords in fractional
        ],
    )


def test_periodic_distance_is_correct_for_a_strongly_skewed_cell():
    """A skewed cell needs the reduced basis that pymatgen searches in.

    Searching the 27 images of this raw cell gives 4.5574, because the nearest
    image sits two cells away along the second lattice vector.
    """
    from abacustools.data.mayer import _minimum_distance
    from abacustools.io.stru import periodic_lattice

    structure = _stru(
        [[8.384, 0.0, 0.0], [1.742, 3.008, 0.0], [2.194, 4.553, 6.765]],
        [[0.9103, 0.4235, 0.9889], [0.4074, 0.9272, 0.6554]],
    )

    distance = _minimum_distance(
        periodic_lattice(structure), *structure.coords_direct
    )

    np.testing.assert_allclose(distance, 4.4205, atol=1e-4)


def test_periodic_distance_crosses_the_cell_boundary():
    from abacustools.data.mayer import _minimum_distance
    from abacustools.io.stru import periodic_lattice

    structure = _stru(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        [[0.95, 0.0, 0.0], [0.05, 0.0, 0.0]],
    )

    distance = _minimum_distance(
        periodic_lattice(structure), *structure.coords_direct
    )

    np.testing.assert_allclose(distance, 1.0)


def test_select_atom_pairs_uses_the_periodic_distance():
    from abacustools.data.mayer import select_atom_pairs

    structure = _stru(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        [[0.95, 0.0, 0.0], [0.05, 0.0, 0.0]],
    )

    assert select_atom_pairs(structure, cutoff=2.0) == [(0, 1)]
    assert select_atom_pairs(structure, cutoff=0.5) == []


def test_periodic_lattice_rejects_a_degenerate_cell():
    from abacustools.io.stru import periodic_lattice

    structure = _stru(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(ValueError, match="non-singular"):
        periodic_lattice(structure)
