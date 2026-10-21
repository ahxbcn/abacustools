"""Tests for ABACUS LCAO COHP/COOP postprocessing."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.postprocess.cohp import run
from abacustools.data.cohp import analyze_cohp, analyze_cohp_pairs, read_efermi_from_log
from abacustools.core.constant import RY_TO_EV
from abacustools.data.mayer import read_nao_basis_layout


H_ORB = """Element H
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
"""

COHP_STRU = """ATOMIC_SPECIES
H 1.008 H.upf

NUMERICAL_ORBITAL
H.orb

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0

ATOMIC_POSITIONS
Cartesian
H
0.0
2
0.0 0.0 0.0 1 1 1
0.0 0.0 0.74 1 1 1
"""


def _write_basis(job: Path) -> None:
    """Write the structure and orbital file that define the two-atom basis."""
    (job / "H.orb").write_text(H_ORB, encoding="utf-8")
    (job / "STRU").write_text(COHP_STRU, encoding="utf-8")


def _write_job(job: Path, *, nspin: int = 1) -> None:
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    _write_basis(job)
    (job / "INPUT").write_text(
        f"INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\nsuffix ABACUS\nnspin {nspin}\n",
        encoding="utf-8",
    )
    (output / "kpoints").write_text(
        "KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT\n1 0 0 0 0.5\n2 0.5 0 0 0.5\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(" E_Fermi 0.0 2.0\n", encoding="utf-8")
    for index in range(2):
        (output / f"data-{index}-H").write_text("2 0 0.5\n0\n", encoding="utf-8")
        (output / f"data-{index}-S").write_text("2 0 0.25\n0\n", encoding="utf-8")
        (output / f"WFC_NAO_K{index + 1}.txt").write_text(
            "1 (index of k points)\n"
            "2 (number of bands)\n"
            "2 (number of orbitals)\n"
            "1 (band)\n"
            "1 (Ry)\n"
            "1.0 (Occupations)\n"
            "0.7071067811865476 0.7071067811865476\n"
            "2 (band)\n"
            "2 (Ry)\n"
            "0.0 (Occupations)\n"
            "0.7071067811865476 -0.7071067811865476\n",
            encoding="utf-8",
        )


def test_cohp_and_coop_use_different_operators(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    cohp = analyze_cohp(job, [0], [1], method="COHP", smooth=False, de=1.0)
    coop = analyze_cohp(job, [0], [1], method="COOP", smooth=False, de=1.0)

    np.testing.assert_allclose(cohp.raw_energy, [13.605693, 13.605693, 27.211386, 27.211386], rtol=1e-6)
    np.testing.assert_allclose(cohp.raw_values, [0.25 * RY_TO_EV, 0.25 * RY_TO_EV, 0.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(coop.raw_values, [0.125, 0.125, 0.0, 0.0], rtol=1e-12)
    assert cohp.efermi == 2.0
    assert cohp.ico_value == 0.0


def test_cohp_command_writes_plot_and_data(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    assert run(
        Namespace(
            job=job,
            cutoff=None,
            pairs=None,
            pairs_file=None,
            atom_i_orbs=[0],
            atom_j_orbs=[1],
            method="COHP",
            spin="sum",
            de=1.0,
            no_smooth=True,
            smooth_nstddev=3.0,
            emin=-5.0,
            emax=5.0,
            width=None,
            invert=True,
            output="plots/cohp.png",
            data_output="processed/cohp.dat",
            efermi=2.0,
        )
    ) == 0
    assert (job / "plots/cohp.png").stat().st_size > 0
    assert (job / "processed/cohp.dat").stat().st_size > 0
    assert np.loadtxt(job / "processed/cohp.dat").shape[1] == 2


def test_read_efermi_returns_last_value(tmp_path: Path) -> None:
    log = tmp_path / "running.log"
    log.write_text("E_Fermi 0.0 1.0\nE_Fermi = 2.5 eV\n", encoding="utf-8")
    assert read_efermi_from_log(log) == 2.5


def _write_develop_job(job: Path) -> None:
    """Write the same two-orbital job in the develop output layout."""

    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\nsuffix ABACUS\nnspin 1\nout_dmk 1\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(" E_Fermi 0.0 2.0\n", encoding="utf-8")
    for index in range(2):
        for label, value in (("H", 0.5), ("S", 0.25)):
            prefix = "hk" if label == "H" else "sk"
            (output / f"{prefix}{index + 1}_nao.txt").write_text(
                "# rows 2\n# columns 2\n"
                f"Row 1\n 0.0 {value}\n"
                "Row 2\n 0.0\n",
                encoding="utf-8",
            )
        (output / f"wfk{index + 1}_nao.txt").write_text(
            "1 (index of k points)\n"
            "2 (number of bands)\n"
            "2 (number of orbitals)\n"
            "1 (band)\n"
            "1 (Ry)\n"
            "1.0 (Occupations)\n"
            "0.7071067811865476 0.7071067811865476\n"
            "2 (band)\n"
            "2 (Ry)\n"
            "0.0 (Occupations)\n"
            "0.7071067811865476 -0.7071067811865476\n",
            encoding="utf-8",
        )
        (output / f"dmk{index + 1}g1_nao.txt").write_text(
            "1 # total k points\n"
            "0.5 # weight of this k point\n"
            "2 # number of localized basis\n"
            "2 2 # size of this matrix\n",
            encoding="utf-8",
        )


def test_cohp_reads_develop_layout(tmp_path: Path) -> None:
    """The develop ``hk``/``sk``/``wfk`` files give the same curve as the LTS ones."""

    lts_job = tmp_path / "lts"
    lts_job.mkdir()
    _write_job(lts_job)
    develop_job = tmp_path / "develop"
    develop_job.mkdir()
    _write_develop_job(develop_job)

    lts = analyze_cohp(lts_job, [0], [1], method="COHP", smooth=False, de=1.0)
    develop = analyze_cohp(develop_job, [0], [1], method="COHP", smooth=False, de=1.0)

    np.testing.assert_allclose(develop.raw_energy, lts.raw_energy, rtol=1e-12)
    np.testing.assert_allclose(develop.raw_values, lts.raw_values, rtol=1e-12)
    np.testing.assert_allclose(develop.raw_values, [0.25 * RY_TO_EV, 0.25 * RY_TO_EV, 0.0, 0.0], rtol=1e-12)


def test_cohp_atom_pair_maps_to_all_atom_orbitals(tmp_path: Path) -> None:
    """An atom pair expands to the full NAO ranges of both atoms."""
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    selected, results = analyze_cohp_pairs(
        job, pairs="1-2", method="COHP", smooth=False, de=1.0
    )
    reference = analyze_cohp(job, [0], [1], method="COHP", smooth=False, de=1.0)

    assert selected == ((0, 1),)
    assert results[0].atom_pair == (1, 2)
    assert results[0].orbital_i == (0,)
    assert results[0].orbital_j == (1,)
    np.testing.assert_allclose(results[0].raw_energy, reference.raw_energy)
    np.testing.assert_allclose(results[0].raw_values, reference.raw_values)


def test_cohp_command_accepts_atom_pairs(tmp_path: Path) -> None:
    """The CLI derives NAO groups from a one-based atom pair."""
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    assert run(
        Namespace(
            job=job,
            cutoff=None,
            pairs="1-2",
            pairs_file=None,
            atom_i_orbs=None,
            atom_j_orbs=None,
            method="COHP",
            spin="sum",
            de=1.0,
            no_smooth=True,
            smooth_nstddev=3.0,
            emin=-5.0,
            emax=5.0,
            width=None,
            invert=False,
            output=None,
            data_output=None,
            efermi=2.0,
        )
    ) == 0
    assert (job / "cohp.png").is_file()
    assert (job / "cohp.dat").is_file()


def test_cohp_pairs_file_and_cutoff_select_the_same_pair(tmp_path: Path) -> None:
    """Pair files and periodic cutoffs use Mayer's atom-pair selector."""
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)
    pair_file = tmp_path / "pairs.txt"
    pair_file.write_text("1 2\n", encoding="utf-8")

    from_file = analyze_cohp_pairs(
        job, pairs_file=pair_file, method="COOP", smooth=False, de=1.0
    )
    from_cutoff = analyze_cohp_pairs(
        job, cutoff=1.0, method="COOP", smooth=False, de=1.0
    )

    assert from_file[0] == from_cutoff[0] == ((0, 1),)
    np.testing.assert_allclose(
        from_file[1][0].raw_values, from_cutoff[1][0].raw_values
    )


def test_nao_layout_maps_s_and_sp_atoms_to_disjoint_ranges(tmp_path: Path) -> None:
    """The automatic mapping includes every angular component of every atom."""
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\n",
        encoding="utf-8",
    )
    (job / "H.orb").write_text(H_ORB, encoding="utf-8")
    (job / "He.orb").write_text(
        """Element He
Energy Cutoff(Ry) 100
Radius Cutoff(a.u.) 6
Lmax 1
Number of Sorbital--> 1
Number of Porbital--> 1
SUMMARY  END

Mesh 3
dr 0.01
Type L N
0 0 0
1.0 2.0 3.0
Type L N
1 1 0
1.0 2.0 3.0
""",
        encoding="utf-8",
    )
    (job / "STRU").write_text(
        """ATOMIC_SPECIES
H 1.008 H.upf
He 4.0026 He.upf

NUMERICAL_ORBITAL
H.orb
He.orb

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0

ATOMIC_POSITIONS
Cartesian
H
0.0
1
0.0 0.0 0.0 1 1 1
He
0.0
1
0.0 0.0 0.74 1 1 1
""",
        encoding="utf-8",
    )

    _, ranges, basis_functions = read_nao_basis_layout(job)

    assert ranges == [(0, 1), (1, 5)]
    assert basis_functions == 5


def test_multiple_atom_pairs_write_one_curve_per_pair(tmp_path: Path) -> None:
    """Multiple selected pairs do not overwrite each other's output files."""
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    _write_basis(job)
    (job / "STRU").write_text(
        COHP_STRU.replace("0.0 0.0 0.74 1 1 1", "0.0 0.0 0.74 1 1 1\n0.0 0.0 1.48 1 1 1").replace(
            "0.0\n2\n", "0.0\n3\n"
        ),
        encoding="utf-8",
    )
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\nsuffix ABACUS\nnspin 1\n",
        encoding="utf-8",
    )
    (output / "kpoints").write_text(
        "KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT\n1 0 0 0 1.0\n",
        encoding="utf-8",
    )
    (output / "data-0-H").write_text(
        "3 0 0.5 0.25\n0 0.3\n0\n", encoding="utf-8"
    )
    (output / "data-0-S").write_text(
        "3 1 0.25 0.125\n1 0.2\n1\n", encoding="utf-8"
    )
    (output / "WFC_NAO_K1.txt").write_text(
        "1 (index of k points)\n"
        "3 (number of bands)\n"
        "3 (number of orbitals)\n"
        "1 (band)\n1 (Ry)\n2.0 (Occupations)\n1 1 1\n"
        "2 (band)\n2 (Ry)\n2.0 (Occupations)\n1 -1 1\n"
        "3 (band)\n3 (Ry)\n2.0 (Occupations)\n1 1 -1\n",
        encoding="utf-8",
    )

    assert run(
        Namespace(
            job=job,
            cutoff=None,
            pairs="1-2,1-3",
            pairs_file=None,
            atom_i_orbs=None,
            atom_j_orbs=None,
            method="COHP",
            spin="sum",
            de=1.0,
            no_smooth=True,
            smooth_nstddev=3.0,
            emin=-5.0,
            emax=5.0,
            width=None,
            invert=False,
            output=None,
            data_output=None,
            efermi=2.0,
        )
    ) == 0

    assert (job / "cohp_1-2.png").is_file()
    assert (job / "cohp_1-3.png").is_file()
    assert (job / "cohp_1-2.dat").is_file()
    assert (job / "cohp_1-3.dat").is_file()


def test_icohp_is_occupation_weighted_and_grid_independent(tmp_path: Path) -> None:
    """ICOHP sums occupied band terms and does not scale with the plot bin."""
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    coarse = analyze_cohp(
        job, [0], [1], method="COHP", smooth=False, de=1.0, efermi=20.0
    )
    fine = analyze_cohp(
        job, [0], [1], method="COHP", smooth=False, de=0.1, efermi=20.0
    )

    expected = 0.5 * RY_TO_EV
    assert coarse.ico_value == pytest.approx(expected)
    assert fine.ico_value == pytest.approx(expected)
    for result in (coarse, fine):
        below = result.energy < 20.0
        integrated = np.trapezoid(result.values[below], result.energy[below])
        assert integrated == pytest.approx(expected, rel=0.02)
