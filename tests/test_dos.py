"""Tests for DOS and PDOS postprocessing."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from abacustools.commands.postprocess.dos import run
from abacustools.data.dos import DOSData, PDOSData


def _args(job: Path, **overrides) -> Namespace:
    args = dict(
        job=job,
        output=None,
        data_output=None,
        emin=-3.0,
        emax=2.0,
        efermi=0.5,
        pdos=None,
        atom_index=None,
        combined=False,
        list_metadata=False,
        json=False,
    )
    args.update(overrides)
    return Namespace(**args)


def _write_job(job: Path, with_pdos: bool = False) -> None:
    output = job / "OUT.ABACUS"
    output.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation nscf\nsuffix ABACUS\nnspin 2\n",
        encoding="utf-8",
    )
    energy = np.array([-2.0, -1.0, 0.0, 1.0])
    up = np.array([0.1, 0.2, 0.3, 0.2])
    down = np.array([0.2, 0.1, 0.2, 0.4])
    np.savetxt(output / "DOS1_smearing.dat", np.column_stack([energy, up, np.cumsum(up)]))
    np.savetxt(output / "DOS2_smearing.dat", np.column_stack([energy, down, np.cumsum(down)]))
    if with_pdos:
        (output / "PDOS").write_text(
            """<pdos>
<energy_values>-2.0 -1.0 0.0 1.0</energy_values>
<orbital index="1" atom_index="1" species="H" l="0" m="0" z="1">
<data>
0.1 0.2
0.2 0.1
0.3 0.2
0.2 0.4
</data>
</orbital>
</pdos>
""",
            encoding="utf-8",
        )


def test_dos_data_reads_spin_channels(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    dos = DOSData.ReadFromAbacusJob(job, efermi=0.5)

    assert dos.dosdata.shape == (4, 2)
    np.testing.assert_allclose(dos.energy, [-2.5, -1.5, -0.5, 0.5])
    np.testing.assert_allclose(dos.dosdata[:, 1], [0.2, 0.1, 0.2, 0.4])


def test_dos_command_writes_plot_and_data(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    assert run(
        Namespace(
            job=job,
            output="plots/DOS.png",
            data_output="processed/DOS.dat",
            emin=-3.0,
            emax=2.0,
            efermi=0.5,
            pdos=None,
            atom_index=None,
            combined=False,
            list_metadata=False,
            json=False,
        )
    ) == 0

    assert (job / "plots/DOS.png").stat().st_size > 0
    assert (job / "processed/DOS.dat").stat().st_size > 0
    assert "DOS_up" in (job / "processed/DOS.dat").read_text()


def test_pdos_command_uses_existing_species_helpers(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job, with_pdos=True)

    assert run(
        Namespace(
            job=job,
            output=None,
            data_output=None,
            emin=-3.0,
            emax=2.0,
            efermi=0.5,
            pdos="species",
            atom_index=None,
            combined=False,
            list_metadata=False,
            json=False,
        )
    ) == 0

    assert (job / "PDOS.png").stat().st_size > 0
    assert (job / "PDOS.dat").stat().st_size > 0
    assert "H_up" in (job / "PDOS.dat").read_text()
    assert isinstance(PDOSData.ReadFromAbacusJob(job, efermi=0.5), PDOSData)


def test_dos_command_lists_metadata(tmp_path: Path, capsys) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job, with_pdos=True)

    assert run(_args(job, list_metadata=True, json=True)) == 0

    metadata = json.loads(capsys.readouterr().out)
    assert metadata["species"] == ["H"]
    assert metadata["shells"]["H"] == [0]
    assert metadata["atoms"] == [1]


def test_dos_command_writes_combined_plot(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job, with_pdos=True)

    assert run(_args(job, combined=True)) == 0

    assert (job / "DOS_PDOS.png").stat().st_size > 0
    assert (job / "DOS_PDOS.dat").stat().st_size > 0


def test_dos_command_atom_shell_mode(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job, with_pdos=True)

    assert run(_args(job, pdos="atom-shell", atom_index=[1])) == 0

    assert (job / "PDOS.png").stat().st_size > 0
    assert (job / "PDOS.dat").stat().st_size > 0
