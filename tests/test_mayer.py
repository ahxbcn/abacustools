"""Tests for robust Mayer postprocessing readers."""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace
from pathlib import Path

from abacustools.core.constant import RY_TO_EV
from abacustools.data.mayer import (
    _develop_density_files,
    _develop_overlap_file,
    analyze_mayer_bond_order,
    calculate_density_matrix_from_matrices,
    calculate_density_matrix_k,
    detect_matrix_format,
    exact_kpoint_weights,
    expanded_orders,
    get_nao_basis_num,
    read_csr_matrix,
    read_csr_matrix_blocks,
    read_density_matrix,
    read_density_matrix_develop,
    read_eig_occ,
    read_kpoint_table,
    read_kpoint_weights,
    read_nao_file,
    read_overlap_matrix,
    read_overlap_matrix_develop,
    read_wfc_nao_k,
)
from abacustools.data.symmetry import atom_permutation, space_group_operations
from abacustools.io.stru import AbacusSTRU


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



def test_read_csr_matrix_parses_abacus_text_format(tmp_path):
    matrix = np.asarray(
        [
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 4.0],
            [0.0, 5.0, 0.0],
        ]
    )
    path = tmp_path / "sr_nao.csr"
    _write_csr_matrix(path, matrix)

    np.testing.assert_allclose(read_csr_matrix(path), matrix)



def test_read_csr_matrix_blocks_reads_every_R_block(tmp_path):
    path = tmp_path / "dmrs1_nao.csr"
    _write_csr_matrix_blocks(
        path,
        {
            (0, 0, 0): np.eye(2),
            (0, 0, 1): np.asarray([[0.0, 0.5], [0.5, 0.0]]),
        },
    )

    blocks = read_csr_matrix_blocks(path)
    assert sorted(blocks) == [(0, 0, 0), (0, 0, 1)]
    np.testing.assert_allclose(blocks[(0, 0, 0)], np.eye(2))
    np.testing.assert_allclose(blocks[(0, 0, 1)], [[0.0, 0.5], [0.5, 0.0]])


def test_read_eig_occ_reads_spin_blocks(tmp_path):
    path = tmp_path / "eig_occ.txt"
    path.write_text(
        """1     # ionic step
 Electronic state energy (eV) and occupations
 Spin number 2
 spin=1 k-point=1/1 Cartesian=0 0 0 (1 plane wave)
 1 -1.0 1.0
 2 1.0 0.0
 spin=2 k-point=1/1 Cartesian=0 0 0 (1 plane wave)
 1 -0.5 0.5
 2 0.5 0.5
""",
        encoding="utf-8",
    )

    eigenstates = read_eig_occ(path)
    np.testing.assert_allclose(eigenstates[(1, 1)][0], [-1.0, 1.0])
    np.testing.assert_allclose(eigenstates[(2, 1)][1], [0.5, 0.5])


def test_calculate_density_matrix_from_matrices():
    hamiltonian = np.asarray([[0.0, -0.5], [-0.5, 0.0]])
    overlap = np.eye(2)
    eigenvalues, density = calculate_density_matrix_from_matrices(
        hamiltonian, overlap, np.asarray([2.0, 0.0])
    )

    np.testing.assert_allclose(eigenvalues, [-0.5, 0.5])
    np.testing.assert_allclose(density, [[1.0, 1.0], [1.0, 1.0]])


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
    assert detect_matrix_format(tmp_path, out_dmk=0) == "develop"


def test_detect_matrix_format_falls_back_to_available_data(tmp_path):
    (tmp_path / "dmk1g1_nao.txt").write_text("", encoding="utf-8")
    assert detect_matrix_format(tmp_path, out_dmk=0) == "develop"

    lts_only = tmp_path / "lts"
    lts_only.mkdir()
    (lts_only / "data-0-S").write_text("", encoding="utf-8")
    assert detect_matrix_format(lts_only, out_dmk=1) == "lts"

    dmr_only = tmp_path / "dmr"
    dmr_only.mkdir()
    (dmr_only / "dmrs1_nao.csr").write_text("", encoding="utf-8")
    assert detect_matrix_format(dmr_only, out_dmk=0, out_dmr=1) == "dmr"
    assert detect_matrix_format(dmr_only, out_dmk=0) == "dmr"

    (dmr_only / "sr_nao.csr").write_text("", encoding="utf-8")
    (dmr_only / "hrs1_nao.csr").write_text("", encoding="utf-8")
    assert detect_matrix_format(
        dmr_only, out_dmk=0, out_dmr=1, out_mat_hs2=1
    ) == "dmr"

    csr_only = tmp_path / "csr"
    csr_only.mkdir()
    (csr_only / "srg1_nao.csr").write_text("", encoding="utf-8")
    (csr_only / "hrs1g1_nao.csr").write_text("", encoding="utf-8")
    assert detect_matrix_format(csr_only, out_dmk=0) == "csr"



def _write_develop_density(
    path: Path,
    matrix: np.ndarray,
    *,
    spin: int,
    nspin: int = 2,
    weight: float = 1.0,
) -> None:
    """Write a minimal develop-layout ``dm*_nao.txt`` file."""

    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("test density matrix must be square")
    lines = [
        " --- Ionic Step 1 ---",
        f" {nspin} # number of spin directions",
        f" {spin} # spin index",
        " 1 # total k points",
        " 1 # total k points after symmetrized (if open)",
        " 1 # k-point index",
        " 0 0 0 # k point coordinate (Cartesian)",
        " 0 0 0 # k point coordinate (direct)",
        f" {weight:.16e} # weight of this k point",
        " -0.2 # Fermi energy in Ry",
        f" {matrix.shape[0]} # number of localized basis",
        f" {matrix.shape[0]} {matrix.shape[1]} # size of this matrix",
        "",
    ]
    lines.extend(
        " ".join(f"{value.real:.16e}" for value in row)
        for row in matrix
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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




def test_develop_dmk_reads_csr_overlap_when_text_overlap_is_absent(tmp_path):
    """A DM(k) job can pair with `sr_nao.csr` from out_mat_hs2/out_hsr."""

    _write_develop_h2_job(tmp_path)
    output = tmp_path / "OUT.ABACUS"
    (output / "sk_nao.txt").unlink()
    _write_csr_matrix(output / "sr_nao.csr", np.eye(2))

    analysis = analyze_mayer_bond_order(tmp_path, pairs="1-2")

    assert analysis.pairs[0].bond_order == pytest.approx(0.0625)



def test_develop_dmk_nspin2_sums_both_spin_channels(tmp_path):
    """Two DM(k) spin channels are added, not doubled or averaged."""

    _write_develop_h2_job(tmp_path)
    output = tmp_path / "OUT.ABACUS"
    (tmp_path / "INPUT").write_text(
        """INPUT_PARAMETERS
calculation scf
basis_type lcao
nspin 2
out_dmk 1
""",
        encoding="utf-8",
    )
    (output / "dmg1_nao.txt").unlink()
    density = np.asarray([[0.5, 0.125], [0.125, 0.5]])
    _write_develop_density(output / "dms1g1_nao.txt", density, spin=1)
    _write_develop_density(output / "dms2g1_nao.txt", density, spin=2)

    analysis = analyze_mayer_bond_order(tmp_path, pairs="1-2")

    overlap = np.asarray([[1.0, 0.5], [0.5, 1.0]])
    population = density @ overlap
    spin_order = population[0, 1] * population[1, 0]
    assert analysis.pairs[0].bond_order == pytest.approx(4.0 * spin_order)


def _write_csr_h2_job(root):
    """Write a two-orbital CSR job with the ABACUS v3.11 output names."""

    _write_develop_h2_job(root)
    (root / "INPUT").write_text(
        """INPUT_PARAMETERS
calculation scf
basis_type lcao
nspin 2
gamma_only 1
out_mat_hs2 1
""",
        encoding="utf-8",
    )
    output = root / "OUT.ABACUS"
    hamiltonian = np.asarray([[0.0, -0.5], [-0.5, 0.0]])
    _write_csr_matrix(output / "sr_nao.csr", np.eye(2))
    _write_csr_matrix(output / "hrs1_nao.csr", hamiltonian)
    _write_csr_matrix(output / "hrs2_nao.csr", hamiltonian)
    energies = np.asarray([-0.5, 0.5]) * RY_TO_EV
    lines = [
        "1     # ionic step",
        " Electronic state energy (eV) and occupations",
        " Spin number 2",
    ]
    for spin in (1, 2):
        lines.append(
            f" spin={spin} k-point=1/1 Cartesian=0 0 0 (1 plane wave)"
        )
        for band, (energy, occupation) in enumerate(zip(energies, (1.0, 0.0)), 1):
            lines.append(f" {band} {energy:.16e} {occupation:.16e}")
    (output / "eig_occ.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analyze_mayer_bond_order_reads_csr_job(tmp_path):
    """The ABACUS v3.11 CSR layout is reconstructed without data-0-S."""

    _write_csr_h2_job(tmp_path)
    analysis = analyze_mayer_bond_order(tmp_path, pairs="1-2")

    assert analysis.gamma_only is True
    assert analysis.nspin == 2
    assert analysis.basis_functions == 2
    assert analysis.pairs[0].bond_order == pytest.approx(1.0)
    assert analysis.pairs[0].distance == pytest.approx(0.7)



def _write_dmr_h2_job(root):
    """Write a gamma-only job whose density matrices come from out_dmr."""

    _write_develop_h2_job(root)
    (root / "INPUT").write_text(
        """INPUT_PARAMETERS
calculation scf
basis_type lcao
nspin 2
gamma_only 1
out_dmr 1
""",
        encoding="utf-8",
    )
    output = root / "OUT.ABACUS"
    (output / "dmg1_nao.txt").unlink()
    _write_csr_matrix(output / "sr_nao.csr", np.eye(2))
    density = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    _write_csr_matrix(output / "dmrs1_nao.csr", density)
    _write_csr_matrix(output / "dmrs2_nao.csr", density)


def test_analyze_mayer_bond_order_uses_dmr_by_default(tmp_path):
    """DM(R) is selected without H/S diagonalization when out_dmr is present."""

    _write_dmr_h2_job(tmp_path)
    analysis = analyze_mayer_bond_order(tmp_path, pairs="1-2")

    assert analysis.gamma_only is True
    assert analysis.nspin == 2
    assert analysis.pairs[0].bond_order == pytest.approx(1.0)
    assert all(path.endswith(".csr") for path in analysis.data_files)


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

# ---------------------------------------------------------- k-point symmetry

#: A cubic cell of two hydrogen atoms, which the fractional translation
#: (1/2, 1/2, 1/2) maps onto itself by swapping the two atoms.
SYMMETRIC_STRU = """ATOMIC_SPECIES
H 1.008 H.upf

NUMERICAL_ORBITAL
H.orb

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0

ATOMIC_POSITIONS
Direct

H
0.0
2
0.0 0.0 0.0
0.5 0.5 0.5
"""

SYMMETRIC_ORB = """Element H
Energy Cutoff(Ry) 100
Radius Cutoff(a.u.) 6
Lmax 0
Number of Sorbital--> 1
SUMMARY  END

Mesh 1
dr 0.01
Type L N
0 0 0
1.0
"""

#: Star, seed overlap (off-diagonal) and seed phase of every k-point of a
#: 2x2x2 mesh.  The two coefficients of a seed have the same modulus, so the
#: density matrix is invariant under the atom swap the translations induce.
SYMMETRIC_SEEDS = {
    (0.0, 0.0, 0.0): (0.30, (0.7071067811865476, 0.7071067811865476)),
    (0.5, 0.0, 0.0): (0.10, (0.7071067811865476, -0.7071067811865476)),
    (0.5, 0.5, 0.0): (0.20, (0.7071067811865476, 0.7071067811865476j)),
    (0.5, 0.5, 0.5): (0.40, (0.7071067811865476, -0.7071067811865476j)),
}

#: The 2x2x2 mesh with the star of every point, as ABACUS reduces it.
SYMMETRIC_MESH = (
    ((0.0, 0.0, 0.0), 1),
    ((0.5, 0.0, 0.0), 2),
    ((0.0, 0.5, 0.0), 2),
    ((0.0, 0.0, 0.5), 2),
    ((0.5, 0.5, 0.0), 3),
    ((0.5, 0.0, 0.5), 3),
    ((0.0, 0.5, 0.5), 3),
    ((0.5, 0.5, 0.5), 4),
)


def _kpoint_table_text(ibz, mesh=()):
    """Build the text of an ABACUS ``kpoints`` file."""
    lines = ["                               nkstot now = %d" % len(ibz), "K-POINTS DIRECT COORDINATES",
             " KPOINTS    DIRECT_X    DIRECT_Y    DIRECT_Z  WEIGHT"]
    for index, (point, weight) in enumerate(ibz, start=1):
        lines.append(f"  {index:6d}  {point[0]:.8f}  {point[1]:.8f}  {point[2]:.8f}  {weight:.4f}")
    if mesh:
        lines += ["", "                                   nkstot = %d" % len(mesh),
                  "K-POINTS REDUCTION ACCORDING TO SYMMETRY",
                  "     KPT    DIRECT_X    DIRECT_Y    DIRECT_Z     IBZ    DIRECT_X    DIRECT_Y    DIRECT_Z"]
        for index, (point, star) in enumerate(mesh, start=1):
            reference = ibz[star - 1][0]
            lines.append(
                f"  {index:6d}  {point[0]:.8f}  {point[1]:.8f}  {point[2]:.8f}"
                f"  {star:6d}  {reference[0]:.8f}  {reference[1]:.8f}  {reference[2]:.8f}"
            )
    return "\n".join(lines) + "\n"


def _write_csr_matrix_blocks(
    path: Path,
    blocks: dict[tuple[int, int, int], np.ndarray],
) -> None:
    """Write a minimal ABACUS multi-R CSR matrix file."""

    if not blocks:
        raise ValueError("test CSR file needs at least one block")
    dimension = None
    lines = [
        " --- Ionic Step 1 ---",
        " # print matrix in real space M(R)",
        " 1 # number of spin directions",
        " 1 # spin index",
    ]
    for matrix in blocks.values():
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("test CSR matrix must be square")
        if dimension is None:
            dimension = matrix.shape[0]
        elif matrix.shape[0] != dimension:
            raise ValueError("test CSR blocks must share one dimension")
    assert dimension is not None
    lines.extend(
        [
            f" {dimension} # number of localized basis",
            f" {len(blocks)} # number of Bravais lattice vector R",
            "",
        ]
    )
    for (rx, ry, rz), matrix in blocks.items():
        matrix = np.asarray(matrix)
        rows, columns = np.nonzero(matrix)
        values = matrix[rows, columns]
        row_pointers = np.zeros(matrix.shape[0] + 1, dtype=int)
        for row in rows:
            row_pointers[row + 1] += 1
        row_pointers = np.cumsum(row_pointers)
        lines.extend(
            [
                f" {rx} {ry} {rz} {len(values)}",
                " # CSR values",
                " " + " ".join(f"{value.real:.16e}" for value in values),
                " # CSR column indices",
                " " + " ".join(str(int(column)) for column in columns),
                " # CSR row pointers",
                " " + " ".join(str(int(pointer)) for pointer in row_pointers),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csr_matrix(path: Path, matrix: np.ndarray) -> None:
    """Write a minimal ABACUS gamma-only CSR matrix file."""

    _write_csr_matrix_blocks(path, {(0, 0, 0): matrix})


def _write_overlap(path: Path, off_diagonal: float) -> None:
    path.write_text(f"2 1.0 {off_diagonal}\n1.0\n", encoding="utf-8")


def _write_wfc(path: Path, index: int, coefficients, occupation: float) -> None:
    values = " ".join(f"{value.real:.10f} {value.imag:.10f}" for value in coefficients)
    path.write_text(
        f"{index} (index of k points)\n1 (number of bands)\n2 (number of orbitals)\n"
        f"1 (band)\n0.0 (Ry)\n{occupation:.10f} (Occupations)\n{values}\n",
        encoding="utf-8",
    )


def _write_synthetic_job(root: Path, *, reduced: bool) -> None:
    """Write an ABACUS LCAO job of the two-atom cubic cell."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nnspin 1\nbasis_type lcao\n"
        "out_mat_hs 1\nsymmetry 1\n",
        encoding="utf-8",
    )
    (root / "STRU").write_text(SYMMETRIC_STRU, encoding="utf-8")
    (root / "H.orb").write_text(SYMMETRIC_ORB, encoding="utf-8")
    output = root / "OUT.ABACUS"
    output.mkdir()

    representatives = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.5, 0.5)]
    if reduced:
        counts = [sum(1 for _point, star in SYMMETRIC_MESH if star == index) for index in range(1, 5)]
        nodes = [(point, count / len(SYMMETRIC_MESH)) for point, count in zip(representatives, counts)]
    else:
        # Every mesh point is computed, but the symmetry-equivalent points
        # share the electronic structure of their star representative.
        nodes = [
            (representatives[star - 1], 1.0 / len(SYMMETRIC_MESH))
            for _point, star in SYMMETRIC_MESH
        ]
    mesh = SYMMETRIC_MESH if reduced else ()
    (output / "kpoints").write_text(_kpoint_table_text(nodes, mesh), encoding="utf-8")

    for index, (point, weight) in enumerate(nodes, start=1):
        seed = SYMMETRIC_SEEDS[point]
        _write_overlap(output / f"data-{index - 1}-S", seed[0])
        # ABACUS writes the occupations as twice the k-point weight.
        _write_wfc(output / f"WFC_NAO_K{index}.txt", index, seed[1], 2.0 * weight)


def test_read_kpoint_table_reads_the_symmetry_reduction(tmp_path):
    path = tmp_path / "kpoints"
    nodes = [
        ((0.0, 0.0, 0.0), 0.125),
        ((0.5, 0.0, 0.0), 0.375),
        ((0.5, 0.5, 0.0), 0.375),
        ((0.5, 0.5, 0.5), 0.125),
    ]
    path.write_text(_kpoint_table_text(nodes, SYMMETRIC_MESH), encoding="utf-8")
    table = read_kpoint_table(path)

    assert [kpoint.direct for kpoint in table.ibz] == [node[0] for node in nodes]
    assert [kpoint.weight for kpoint in table.ibz] == [node[1] for node in nodes]
    assert len(table.mesh) == 8
    assert {index for _point, index in table.mesh} == {1, 2, 3, 4}


def test_exact_kpoint_weights_follow_the_symmetry(tmp_path):
    """The printed weights are rounded, the exact ones follow from the stars."""
    weight = 1.0 / 3.0
    path = tmp_path / "reduced"
    path.write_text(
        _kpoint_table_text([((0.0, 0.0, 0.0), 0.3333), ((0.5, 0.0, 0.0), 0.6667)],
                           [((0.0, 0.0, 0.0), 1), ((0.5, 0.0, 0.0), 2), ((0.0, 0.5, 0.0), 2)]),
        encoding="utf-8",
    )
    np.testing.assert_allclose(exact_kpoint_weights(read_kpoint_table(path)), [weight, 2 * weight])

    path = tmp_path / "time-reversal"
    path.write_text(
        _kpoint_table_text(
            [
                ((0.0, 0.0, 0.0), 0.2500),
                ((0.5, 0.0, 0.0), 0.2500),
                ((0.25, 0.0, 0.0), 0.5000),
            ]
        ),
        encoding="utf-8",
    )
    np.testing.assert_allclose(exact_kpoint_weights(read_kpoint_table(path)), [0.25, 0.25, 0.5])

    path = tmp_path / "full-mesh"
    path.write_text(
        _kpoint_table_text([((0.0, 0.0, 0.0), 0.5000), ((0.25, 0.0, 0.0), 0.5000)]),
        encoding="utf-8",
    )
    np.testing.assert_allclose(exact_kpoint_weights(read_kpoint_table(path)), [0.5, 0.5])

    path = tmp_path / "not-reduced"
    path.write_text(
        _kpoint_table_text([((0.0, 0.0, 0.0), 0.5), ((0.25, 0.0, 0.0), 0.5)]),
        encoding="utf-8",
    )
    table = replace(read_kpoint_table(path), kpoint_count=8)
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        exact_kpoint_weights(table)


def test_atom_permutation_follows_the_operations(tmp_path):
    (tmp_path / "STRU").write_text(SYMMETRIC_STRU, encoding="utf-8")
    structure = AbacusSTRU.read(str(tmp_path / "STRU"))

    assert atom_permutation(structure, np.eye(3), [0.0, 0.0, 0.0]) == (0, 1)
    assert atom_permutation(structure, np.eye(3), [0.5, 0.5, 0.5]) == (1, 0)
    assert atom_permutation(structure, np.eye(3), [0.25, 0.0, 0.0]) is None

    permutations = {operation.permutation for operation in space_group_operations(structure)}
    assert (0, 1) in permutations and (1, 0) in permutations


def test_expanded_orders_average_the_star():
    """Every star member contributes through the pair its operation maps onto."""
    values = {0: {(0, 1): 1.0, (1, 2): 4.0, (0, 2): 7.0}}

    def values_of(index, pairs):
        return {pair: values[index][pair] for pair in pairs}

    stars = {1: [(0, 1, 2), (1, 2, 0)]}
    orders = expanded_orders(stars, (0.5,), values_of, [(0, 1)])

    # The identity reaches the pair (0, 1), the three-cycle the pair (1, 2),
    # and the star is averaged and weighted like one computed k-point.
    assert orders[(0, 1)] == pytest.approx((1.0 + 4.0) / 2 / 0.5)


def test_symmetry_reduced_kpoints_reproduce_the_full_mesh(tmp_path):
    """A symmetry=1 run gives the bond order of the full k-mesh."""
    reduced = tmp_path / "reduced"
    full = tmp_path / "full"
    _write_synthetic_job(reduced, reduced=True)
    _write_synthetic_job(full, reduced=False)

    with_reduction = analyze_mayer_bond_order(reduced, pairs="1-2")
    with_full_mesh = analyze_mayer_bond_order(full, pairs="1-2")

    assert len(read_kpoint_table(reduced / "OUT.ABACUS" / "kpoints").ibz) == 4
    assert len(read_kpoint_table(full / "OUT.ABACUS" / "kpoints").ibz) == 8
    assert with_reduction.pairs[0].bond_order == pytest.approx(
        with_full_mesh.pairs[0].bond_order, rel=1e-9
    )

