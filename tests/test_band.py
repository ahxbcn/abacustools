"""Tests for ABACUS NSCF band processing and plotting."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.postprocess.band import run
from abacustools.data.band import BandData


STRU = """ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
4 0 0
0 4 0
0 0 4

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0
"""


def _write_job(job: Path, nspin: int = 1, calculation: str = "nscf") -> None:
    (job / "OUT.ABACUS").mkdir()
    (job / "INPUT").write_text(
        f"INPUT_PARAMETERS\ncalculation {calculation}\nsuffix ABACUS\n"
        f"nspin {nspin}\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(STRU, encoding="utf-8")
    (job / "KPT").write_text(
        "K_POINTS\n3\nLine\n"
        "0 0 0 2 # G\n"
        "0.5 0 0 2 # X\n"
        "0.5 0.5 0 1 # M\n",
        encoding="utf-8",
    )
    rows = np.array(
        [
            [1, 0.0, -1.0, 1.0],
            [2, 0.2, -0.8, 1.2],
            [3, 0.4, -0.5, 1.5],
            [4, 0.6, -0.8, 1.2],
            [5, 0.8, -1.0, 1.0],
        ]
    )
    np.savetxt(job / "OUT.ABACUS" / "BANDS_1.dat", rows)
    if nspin == 2:
        np.savetxt(job / "OUT.ABACUS" / "BANDS_2.dat", rows + [0, 0, 0.1, -0.1])


def test_read_nscf_band_handles_spin_channels(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job, nspin=2)

    band = BandData.ReadFromAbacusJob(job, efermi=0.25)

    assert band.band_data.shape == (2, 5, 2)
    np.testing.assert_allclose(band.band_data[0, 0], [-1.25, 0.75])
    assert band.kpaths == [
        {"start": "G", "end": "X", "start_nkpt": 0, "end_nkpt": 1},
        {"start": "X", "end": "M", "start_nkpt": 2, "end_nkpt": 4},
    ]


def test_read_nscf_band_reads_e_fermi_log_format(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)
    (job / "OUT.ABACUS" / "running_nscf.log").write_text(
        " E_Fermi        0.25         4.75\n",
        encoding="utf-8",
    )

    band = BandData.ReadFromAbacusJob(job)

    assert band.efermi == 4.75


def test_band_command_writes_plot_and_processed_data(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    assert run(
        Namespace(
            job=job,
            output="plots/band.png",
            data_output="processed/band.dat",
            kpath_output="processed/KPATH.txt",
            emin=-2.0,
            emax=2.0,
            efermi=0.0,
            gap=False,
            spin_resolved=False,
            effective_mass=None,
            direction=None,
            fit_points=5,
            fat_band=None,
            atom_index=None,
            json=False,
        )
    ) == 0

    assert (job / "plots/band.png").stat().st_size > 0
    assert (job / "processed/band.dat").stat().st_size > 0
    assert (job / "processed/KPATH.txt").stat().st_size > 0


def _analysis_args(job: Path, **overrides) -> Namespace:
    args = dict(
        job=job,
        output=None,
        data_output="band.dat",
        kpath_output="KPATH.txt",
        emin=-2.0,
        emax=2.0,
        efermi=0.0,
        gap=False,
        spin_resolved=False,
        effective_mass=None,
        direction=None,
        fit_points=5,
        fat_band=None,
        atom_index=None,
        json=False,
    )
    args.update(overrides)
    return Namespace(**args)


def test_band_command_reports_band_gap(tmp_path: Path, capsys) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    assert run(_analysis_args(job, gap=True, json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    gap = report["band_gap"]
    assert gap["band_gap"] == pytest.approx(1.5)
    assert gap["direct"] is False
    assert gap["is_metal"] is False
    assert gap["vbm"]["energy"] == pytest.approx(-0.5)
    assert gap["cbm"]["energy"] == pytest.approx(1.0)


def test_band_reader_warns_for_non_nscf_job(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job, calculation="scf")

    with pytest.warns(UserWarning, match="calculation=nscf"):
        band = BandData.ReadFromAbacusJob(job, efermi=0.0)

    assert band.band_data.shape == (1, 5, 2)
