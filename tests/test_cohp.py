"""Tests for ABACUS LCAO COHP/COOP postprocessing."""

from argparse import Namespace
from pathlib import Path

import numpy as np

from abacustools.commands.postprocess.cohp import run
from abacustools.data.cohp import analyze_cohp, read_efermi_from_log


def _write_job(job: Path, *, nspin: int = 1) -> None:
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
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
            "0 (Occupations)\n"
            "0.7071067811865476 0.7071067811865476\n"
            "2 (band)\n"
            "2 (Ry)\n"
            "0 (Occupations)\n"
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
    np.testing.assert_allclose(cohp.raw_values, [0.125, 0.125, -0.125, -0.125], rtol=1e-12)
    np.testing.assert_allclose(coop.raw_values, [0.0625, 0.0625, -0.0625, -0.0625], rtol=1e-12)
    assert cohp.efermi == 2.0
    assert cohp.ico_value == 0.0


def test_cohp_command_writes_plot_and_data(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    assert run(
        Namespace(
            job=job,
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
            "0 (Occupations)\n"
            "0.7071067811865476 0.7071067811865476\n"
            "2 (band)\n"
            "2 (Ry)\n"
            "0 (Occupations)\n"
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
    np.testing.assert_allclose(develop.raw_values, [0.125, 0.125, -0.125, -0.125], rtol=1e-12)
