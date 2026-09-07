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
