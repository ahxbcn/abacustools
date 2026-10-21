"""Tests for Mulliken, Loewdin and Wiberg population analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacustools.data.lcao import load_lcao_kpoint_densities
from abacustools.data.population import (
    PopulationAnalysisError,
    _expand_atom_values,
    _expand_pair_values,
    _load_density_data,
    _sqrtm_hermitian,
    analyze_population,
)


H2_STRUCTURE = """ATOMIC_SPECIES
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
0.0 0.0 0.0
0.0 0.0 0.7
"""

H_ORBITAL = """Element H
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

OVERLAP = np.asarray([[1.0, 0.5], [0.5, 1.0]])
DENSITY = np.asarray([[0.875, 0.25], [0.25, 0.875]])


def _write_job(
    root: Path,
    *,
    nspin: int = 1,
    gamma_only: bool = True,
    nelec: float = 2.0,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "STRU").write_text(H2_STRUCTURE, encoding="utf-8")
    (root / "H.orb").write_text(H_ORBITAL, encoding="utf-8")
    (root / "H.upf").write_text("", encoding="utf-8")
    (root / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        f"nspin {nspin}\n"
        f"gamma_only {1 if gamma_only else 0}\n"
        "out_dmk 1\n"
        f"nelec {nelec}\n",
        encoding="utf-8",
    )
    (root / "OUT.ABACUS").mkdir()


def _write_overlap(path: Path, matrix: np.ndarray) -> None:
    rows = []
    for index in range(matrix.shape[0]):
        values = " ".join(
            f"({matrix[index, column].real:.12f},{matrix[index, column].imag:.12f})"
            for column in range(index, matrix.shape[1])
        )
        rows.append(f"Row {index + 1}\n{values}\n")
    path.write_text(
        f"# rows {matrix.shape[0]}\n# columns {matrix.shape[1]}\n" + "".join(rows),
        encoding="utf-8",
    )


def _write_lts_overlap(path: Path, matrix: np.ndarray) -> None:
    lines = []
    for index in range(matrix.shape[0]):
        values = " ".join(
            f"({matrix[index, column].real:.12f},{matrix[index, column].imag:.12f})"
            for column in range(index, matrix.shape[1])
        )
        if index == 0:
            values = f"{matrix.shape[0]} {values}"
        lines.append(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_density(
    path: Path,
    matrix: np.ndarray,
    *,
    spin: int,
    nspin: int,
    kpoint: int,
    total_kpoints: int,
    direct: tuple[float, float, float],
    weight: float,
) -> None:
    lines = [
        " --- Ionic Step 1 ---",
        f" {nspin} # number of spin directions",
        f" {spin} # spin index",
        f" {total_kpoints} # total k points",
        f" {total_kpoints} # total k points after symmetrized (if open)",
        f" {kpoint} # k-point index",
        " 0 0 0 # k point coordinate (Cartesian)",
        f" {direct[0]} {direct[1]} {direct[2]} # k point coordinate (direct)",
        f" {weight:.16e} # weight of this k point",
        " -0.2 # Fermi energy in Ry",
        f" {matrix.shape[0]} # number of localized basis",
        f" {matrix.shape[0]} {matrix.shape[1]} # size of this matrix",
        "",
        " user_defined_lattice",
        " 1",
        " 10 0 0",
        " 0 10 0",
        " 0 0 10",
        " H",
        " 2",
        " Direct",
        " 0.0 0.0 0.0",
        " 0.0 0.0 0.07",
        "",
    ]
    lines.extend(
        " ".join(f"({value.real:.16e},{value.imag:.16e})" for value in row) for row in matrix
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_kpoints(path: Path, weights: list[float]) -> None:
    lines = [
        f" nkstot now = {len(weights)}",
        "K-POINTS DIRECT COORDINATES",
        " KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT",
    ]
    for index, weight in enumerate(weights, start=1):
        direct = 0.0 if index == 1 else 0.5
        lines.append(f" {index} {direct} 0 0 {weight:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_kpoints_with_mesh(
    path: Path,
    nodes: list[tuple[tuple[float, float, float], float]],
    mesh: list[tuple[tuple[float, float, float], int]],
) -> None:
    lines = [
        f" nkstot now = {len(nodes)}",
        "K-POINTS DIRECT COORDINATES",
        " KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT",
    ]
    for index, (point, weight) in enumerate(nodes, start=1):
        lines.append(f" {index} {point[0]} {point[1]} {point[2]} {weight:.4f}")
    lines += [
        "",
        f" nkstot = {len(mesh)}",
        "K-POINTS REDUCTION ACCORDING TO SYMMETRY",
        " KPT DIRECT_X DIRECT_Y DIRECT_Z IBZ DIRECT_X DIRECT_Y DIRECT_Z",
    ]
    for index, (point, star) in enumerate(mesh, start=1):
        reference = nodes[star - 1][0]
        lines.append(
            f" {index} {point[0]} {point[1]} {point[2]} {star} "
            f"{reference[0]} {reference[1]} {reference[2]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gamma_job(root: Path) -> Path:
    _write_job(root)
    output = root / "OUT.ABACUS"
    _write_overlap(output / "sk_nao.txt", OVERLAP)
    _write_density(
        output / "dmg1_nao.txt",
        DENSITY,
        spin=1,
        nspin=1,
        kpoint=1,
        total_kpoints=1,
        direct=(0.0, 0.0, 0.0),
        weight=2.0,
    )
    return root


def _multik_job(root: Path, *, weighted: bool = True) -> Path:
    _write_job(root, gamma_only=False)
    output = root / "OUT.ABACUS"
    _write_kpoints(output / "kpoints", [1.0, 1.0])
    for index, direct in ((1, (0.0, 0.0, 0.0)), (2, (0.5, 0.0, 0.0))):
        _write_overlap(output / f"sk{index}_nao.txt", OVERLAP)
        matrix = 0.5 * DENSITY if weighted else DENSITY
        _write_density(
            output / f"dmk{index}g1_nao.txt",
            matrix,
            spin=1,
            nspin=1,
            kpoint=index,
            total_kpoints=2,
            direct=direct,
            weight=1.0,
        )
    return root


def _nspin2_gamma_job(root: Path) -> Path:
    _write_job(root, nspin=2)
    output = root / "OUT.ABACUS"
    _write_overlap(output / "sk_nao.txt", OVERLAP)
    up = np.asarray([[0.5, 0.125], [0.125, 0.5]])
    down = np.asarray([[0.375, 0.125], [0.125, 0.375]])
    for spin, matrix in ((1, up), (2, down)):
        _write_density(
            output / f"dms{spin}g1_nao.txt",
            matrix,
            spin=spin,
            nspin=2,
            kpoint=1,
            total_kpoints=1,
            direct=(0.0, 0.0, 0.0),
            weight=1.0,
        )
    return root


def _write_wfc(path: Path, coefficients: np.ndarray, occupations: list[float]) -> None:
    lines = [
        "1 (index of k points)",
        f"{len(occupations)} (number of bands)",
        f"{coefficients.shape[0]} (number of orbitals)",
    ]
    for band, occupation in enumerate(occupations, start=1):
        lines.extend(
            [
                f"{band} (band)",
                "0.0 (Ry)",
                f"{occupation:.12f} (Occupations)",
                " ".join(
                    f"{coefficients[row, band - 1].real:.12f} "
                    f"{coefficients[row, band - 1].imag:.12f}"
                    for row in range(coefficients.shape[0])
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lts_multik_job(root: Path) -> Path:
    _write_job(root, gamma_only=False)
    output = root / "OUT.ABACUS"
    _write_kpoints(output / "kpoints", [0.5, 0.5])
    coefficients = np.eye(2, dtype=complex)
    for index in (1, 2):
        _write_lts_overlap(output / f"data-{index - 1}-S", OVERLAP)
        _write_wfc(
            output / f"WFC_NAO_K{index}.txt",
            coefficients,
            [1.0, 1.0],
        )
    return root


def _symmetry_weight_job(root: Path) -> Path:
    _write_job(root, gamma_only=False)
    output = root / "OUT.ABACUS"
    nodes = [
        ((0.0, 0.0, 0.0), 0.25),
        ((0.25, 0.0, 0.0), 0.5),
        ((0.5, 0.0, 0.0), 0.25),
    ]
    mesh = [
        ((0.0, 0.0, 0.0), 1),
        ((0.25, 0.0, 0.0), 2),
        ((0.5, 0.0, 0.0), 3),
        ((-0.25, 0.0, 0.0), 2),
    ]
    _write_kpoints_with_mesh(output / "kpoints", nodes, mesh)
    for index, (point, weight) in enumerate(nodes, start=1):
        _write_overlap(output / f"sk{index}_nao.txt", OVERLAP)
        _write_density(
            output / f"dmk{index}g1_nao.txt",
            weight * DENSITY,
            spin=1,
            nspin=1,
            kpoint=index,
            total_kpoints=4,
            direct=point,
            weight=2.0 * weight,
        )
    return root


def _lts_nspin2_job(root: Path) -> Path:
    _write_job(root, nspin=2, gamma_only=False)
    output = root / "OUT.ABACUS"
    _write_kpoints(output / "kpoints", [0.5, 0.5])
    coefficients = np.eye(2, dtype=complex)
    for index in (1, 2):
        _write_lts_overlap(output / f"data-{index - 1}-S", OVERLAP)
    for index in (1, 2, 3, 4):
        _write_wfc(
            output / f"WFC_NAO_K{index}.txt",
            coefficients,
            [0.5, 0.5],
        )
    return root


def test_mulliken_gamma_population_and_overlap(tmp_path):
    analysis = analyze_population(_gamma_job(tmp_path), method="mulliken-overlap")

    assert analysis.kpoint_count == 1
    assert analysis.expected_electrons == pytest.approx(2.0)
    assert analysis.total_electrons == pytest.approx(2.0)
    np.testing.assert_allclose([atom.population for atom in analysis.atoms], [1.0, 1.0])
    assert analysis.bonds[0].value == pytest.approx(2.0 * 0.25 * 0.5)


def test_mulliken_gamma_bond_order(tmp_path):
    analysis = analyze_population(_gamma_job(tmp_path), method="mulliken")

    product = DENSITY @ OVERLAP
    assert analysis.bonds[0].value == pytest.approx(product[0, 1] * product[1, 0])


def test_lowdin_gamma_wiberg_matches_direct_matrix_product(tmp_path):
    analysis = analyze_population(_gamma_job(tmp_path), method="lowdin")

    root = _sqrtm_hermitian(OVERLAP)
    product = root @ DENSITY @ root
    np.testing.assert_allclose(
        [atom.population for atom in analysis.atoms],
        np.real(np.diag(product)),
    )
    assert analysis.bonds[0].value == pytest.approx(abs(product[0, 1]) ** 2)


@pytest.mark.parametrize("weighted", [False, True])
def test_multik_weight_convention_is_detected(tmp_path, weighted):
    analysis = analyze_population(
        _multik_job(tmp_path, weighted=weighted), method="mulliken-overlap"
    )

    assert analysis.kpoint_count == 2
    assert analysis.total_electrons == pytest.approx(2.0)
    np.testing.assert_allclose([atom.population for atom in analysis.atoms], [1.0, 1.0])
    assert analysis.bonds[0].value == pytest.approx(0.25)


def test_nspin2_total_and_difference(tmp_path):
    total = analyze_population(_nspin2_gamma_job(tmp_path), method="mulliken", spin="total")
    difference = analyze_population(tmp_path, method="mulliken", spin="difference")
    legacy_scale = analyze_population(tmp_path, method="mulliken", spin="total", spin_factor=2.0)

    np.testing.assert_allclose([atom.population for atom in total.atoms], [1.0, 1.0])
    assert total.total_magnetization == pytest.approx(0.25)
    np.testing.assert_allclose([atom.population for atom in difference.atoms], [0.125, 0.125])
    assert difference.total_magnetization == pytest.approx(0.25)
    assert legacy_scale.bonds[0].value == pytest.approx(2.0 * total.bonds[0].value)
    assert legacy_scale.spin_factor == 2.0


def test_lts_wfc_multik_population(tmp_path):
    analysis = analyze_population(_lts_multik_job(tmp_path), method="mulliken-overlap")

    assert analysis.kpoint_count == 2
    assert analysis.total_electrons == pytest.approx(2.0)
    np.testing.assert_allclose([atom.population for atom in analysis.atoms], [1.0, 1.0])
    # The synthetic D = I has no inter-atomic coherence, so its Mulliken
    # overlap population vanishes even though S has an off-diagonal element.
    assert analysis.bonds[0].value == pytest.approx(0.0)


def test_develop_dm_uses_the_recorded_symmetry_weights(tmp_path):
    root = _symmetry_weight_job(tmp_path)
    kpoints = _load_density_data(root / "OUT.ABACUS", 1, False)

    np.testing.assert_allclose([kpoint.weight for kpoint in kpoints], [0.25, 0.5, 0.25])


def test_electron_count_falls_back_to_the_scf_log(tmp_path):
    root = _gamma_job(tmp_path)
    (root / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\nnspin 1\ngamma_only 1\nout_dmk 1\n",
        encoding="utf-8",
    )
    (root / "OUT.ABACUS" / "running_scf.log").write_text(
        "AUTOSET number of electrons: = 2\n",
        encoding="utf-8",
    )

    analysis = analyze_population(root, method="mulliken")

    assert analysis.expected_electrons == pytest.approx(2.0)
    assert analysis.total_electrons == pytest.approx(2.0)


def test_lts_wfc_nspin2_total_and_difference(tmp_path):
    total = analyze_population(_lts_nspin2_job(tmp_path), method="mulliken", spin="total")
    difference = analyze_population(tmp_path, method="mulliken", spin="difference")

    assert total.kpoint_count == 2
    assert total.total_electrons == pytest.approx(2.0)
    np.testing.assert_allclose([atom.population for atom in total.atoms], [1.0, 1.0])
    assert total.total_magnetization == pytest.approx(0.0)
    np.testing.assert_allclose([atom.population for atom in difference.atoms], [0.0, 0.0])


def test_spin_selection_requires_nspin2(tmp_path):
    with pytest.raises(PopulationAnalysisError, match="requires nspin=2"):
        analyze_population(_gamma_job(tmp_path), spin="difference")


def test_star_expansion_maps_atoms_and_pairs():
    kpoints = [
        type(
            "K",
            (),
            {"index": 1, "weight": 0.5},
        )()
    ]
    # The star swaps the two atoms.  The representative values are 1 and 3
    # for the atoms and 2 for the pair; the star average is (1 + 3) / 2 and
    # the pair is mapped back onto itself.
    atoms = _expand_atom_values(
        kpoints,
        {1: [(0, 1), (1, 0)]},
        {1: np.asarray([1.0, 3.0])},
        2,
    )
    pairs = _expand_pair_values(
        kpoints,
        {1: [(0, 1), (1, 0)]},
        {1: {(0, 1): 2.0}},
        [(0, 1)],
        2,
    )
    np.testing.assert_allclose(atoms, [1.0, 1.0])
    assert pairs[(0, 1)] == pytest.approx(1.0)


def _assert_same_kpoint_densities(left, right):
    assert len(left) == len(right)
    for first, second in zip(left, right):
        assert first.index == second.index
        np.testing.assert_allclose(first.direct, second.direct)
        assert first.weight == pytest.approx(second.weight)
        np.testing.assert_allclose(first.overlap, second.overlap)
        assert set(first.densities) == set(second.densities)
        for spin in first.densities:
            np.testing.assert_allclose(first.densities[spin], second.densities[spin])


def test_public_lcao_loader_matches_gamma_population_loader(tmp_path):
    root = _gamma_job(tmp_path)
    public = load_lcao_kpoint_densities(root)
    private = _load_density_data(root / "OUT.ABACUS", 1, True)

    _assert_same_kpoint_densities(public, private)


def test_public_lcao_loader_matches_multik_population_loader(tmp_path):
    root = _multik_job(tmp_path, weighted=False)
    public = load_lcao_kpoint_densities(root)
    private = _load_density_data(root / "OUT.ABACUS", 1, False)

    _assert_same_kpoint_densities(public, private)
