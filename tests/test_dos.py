"""Tests for DOS and PDOS postprocessing."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

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


def _dos_nao_text(energy: np.ndarray, dos: np.ndarray, istep: int = 1) -> str:
    """Build one develop-branch total-DOS file body."""
    header = (
        f"{istep}    # ionic step\n"
        f"{len(energy)} # number of points\n"
        "#  energy(eV)         dos     dos_int  dos_smear(1/eV)  dos_smear_int\n"
    )
    rows = "\n".join(
        f"{value:>12.6f} {dos:>12.6f} {dos:>12.6f} {dos:>12.6f} {dos:>12.6f}"
        for value, dos in zip(energy, dos)
    )
    return header + rows + "\n"


def _pdos_nao_text(energy: np.ndarray, atoms: List[Tuple[int, str, np.ndarray]]) -> str:
    """Build one develop-branch projected-DOS file body.

    Args:
        energy: Shared energy grid.
        atoms: One ``(atom_index, species, columns)`` per atom, the columns
            holding one PDOS value per energy point and ``(l, m)`` channel.
    """
    header = (
        "# istep: 1\n"
        f"# npoints: {len(energy)}\n"
        "# energy(eV)  atom  species  pdos(1/eV), columns: s(m=0) p(m=0,+1,-1)"
        " d(m=0,+1,-1,+2,-2) f(m=0,+1,-1,+2,-2,+3,-3)\n"
    )
    rows = []
    for atom_index, species, columns in atoms:
        for point in range(len(energy)):
            fields = [f"{energy[point]:12.6f}", f"{atom_index:3d}", f"{species:>4}"]
            fields += [f"{value:10.6f}" for value in columns[point]]
            rows.append("  ".join(fields))
    return header + "\n".join(rows) + "\n"


def _write_develop_job(job: Path, with_pdos: bool = False) -> None:
    """Write an ABACUS develop job with the text DOS and PDOS outputs."""
    output = job / "OUT.ABACUS"
    output.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nsuffix ABACUS\nnspin 2\n",
        encoding="utf-8",
    )
    energy = np.array([-2.0, -1.0, 0.0, 1.0])
    up = np.array([0.1, 0.2, 0.3, 0.2])
    down = np.array([0.2, 0.1, 0.2, 0.4])
    for spin, values in ((1, up), (2, down)):
        (output / f"doss{spin}g1_nao.txt").write_text(
            _dos_nao_text(energy, values), encoding="utf-8"
        )
    if with_pdos:
        # Atom 1 (H) carries one s channel, atom 2 (O) one s and three p.
        # The spin-down columns differ from the up ones so the pairing is
        # checkable channel by channel.
        up_atoms = [
            (1, "H", up.reshape(-1, 1)),
            (2, "O", np.column_stack([up * 2.0, up * 3.0, up * 4.0, up * 5.0])),
        ]
        down_atoms = [
            (1, "H", down.reshape(-1, 1)),
            (2, "O", np.column_stack([down * 2.0, down * 3.0, down * 4.0, down * 5.0])),
        ]
        (output / "pdoss1g1_nao.txt").write_text(
            _pdos_nao_text(energy, up_atoms), encoding="utf-8"
        )
        (output / "pdoss2g1_nao.txt").write_text(
            _pdos_nao_text(energy, down_atoms), encoding="utf-8"
        )


def test_dos_data_reads_spin_channels(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    dos = DOSData.ReadFromAbacusJob(job, efermi=0.5)

    assert dos.dosdata.shape == (4, 2)
    np.testing.assert_allclose(dos.energy, [-2.5, -1.5, -0.5, 0.5])
    np.testing.assert_allclose(dos.dosdata[:, 1], [0.2, 0.1, 0.2, 0.4])


def test_dos_data_reads_develop_nao_files(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_develop_job(job)

    dos = DOSData.ReadFromAbacusJob(job, efermi=0.5)

    assert dos.dosdata.shape == (4, 2)
    np.testing.assert_allclose(dos.dosdata[:, 0], [0.1, 0.2, 0.3, 0.2])
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


def test_pdos_data_reads_develop_nao_files(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_develop_job(job, with_pdos=True)

    pdos = PDOSData.ReadFromAbacusJob(job, efermi=0.5)

    assert pdos.get_species() == ["H", "O"]
    assert pdos.get_species_shell("H") == [0]
    assert pdos.get_species_shell("O") == [0, 1]
    assert pdos.get_species_shell_orbital("O", 1) == [0, 1, 2]
    np.testing.assert_allclose(pdos.energy, [-2.5, -1.5, -0.5, 0.5])
    # the s channel of atom 1 pairs the up and down columns of both files
    orbital = pdos.get_pdos_by_orbital("H", 1, 0)
    np.testing.assert_allclose(
        orbital["data"], np.column_stack([[0.1, 0.2, 0.3, 0.2], [0.2, 0.1, 0.2, 0.4]])
    )
    # the fourth column of atom 2 is the p(m=2) channel
    orbital = pdos.get_pdos_by_orbital("O", 2, 1, m=2)
    np.testing.assert_allclose(
        orbital["data"], np.column_stack([[0.5, 1.0, 1.5, 1.0], [1.0, 0.5, 1.0, 2.0]])
    )
    # the summed species PDOS keeps the spin-channel layout
    np.testing.assert_allclose(
        pdos.get_pdos_by_species("O"),
        np.column_stack([[1.4, 2.8, 4.2, 2.8], [2.8, 1.4, 2.8, 5.6]]),
    )


def test_dos_command_writes_plot_and_data_develop(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    _write_develop_job(job)

    assert run(
        Namespace(
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
    ) == 0

    assert (job / "DOS.png").stat().st_size > 0
    assert (job / "DOS.dat").stat().st_size > 0
    assert "DOS_up" in (job / "DOS.dat").read_text()


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
