"""Tests for data readers that do not depend on abacustest."""

from __future__ import annotations

import numpy as np
import pytest

from abacustools.data.abacus_result import (
    get_result_from_job,
    read_dos_from_job,
    read_orbital_xml,
    read_pdos_from_job,
)
from abacustools.data.band import BandData
from abacustools.data.grid import Charge, Grid
from abacustools.main import main


def test_band_data_module_imports_without_abacustest():
    assert BandData.__name__ == "BandData"


def _job_with_output(tmp_path):
    output = tmp_path / "OUT.ABACUS"
    output.mkdir()
    (tmp_path / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\ngamma_only 1\n",
        encoding="utf-8",
    )
    return output


def test_get_result_returns_none_when_requested_data_is_unavailable(tmp_path):
    output = _job_with_output(tmp_path)
    (output / "running_scf.log").write_text(
        "calculation did not reach an electronic iteration\n",
        encoding="utf-8",
    )

    result = get_result_from_job(
        tmp_path,
        [
            "energy",
            "drho",
            "denergy",
            "scf_steps",
            "converged",
            "normal_end",
            "efermi",
        ],
        version="",
    )

    assert result == {
        "energy": None,
        "drho": None,
        "denergy": None,
        "scf_steps": 0,
        "converged": False,
        "normal_end": False,
        "efermi": None,
    }


def test_get_result_does_not_fail_for_incomplete_job(tmp_path):
    (tmp_path / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nsuffix ABACUS\n",
        encoding="utf-8",
    )

    result = get_result_from_job(tmp_path, None, version="")

    assert result["energy"] is None
    assert result["converged"] is None
    assert result["normal_end"] is None


def test_get_result_reports_normal_end_from_log_footer(tmp_path):
    output = _job_with_output(tmp_path)
    (output / "running_scf.log").write_text(
        "Final Etot = -2.5 eV\n"
        "\n"
        "Start  Time  : 2026-09-07 10:00:00\n"
        "Finish Time  : 2026-09-07 10:01:00\n"
        "Total  Time  : 60\n\n",
        encoding="utf-8",
    )

    result = get_result_from_job(tmp_path, ["normal_end"], version="")

    assert result == {"normal_end": True}


def test_get_result_reports_incomplete_log(tmp_path):
    output = _job_with_output(tmp_path)
    (output / "running_scf.log").write_text(
        "Final Etot = -2.5 eV\n",
        encoding="utf-8",
    )

    result = get_result_from_job(tmp_path, ["normal_end"], version="")

    assert result == {"normal_end": False}


def test_result_command_continues_after_incomplete_job(tmp_path, capsys):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\n",
        encoding="utf-8",
    )

    complete = tmp_path / "complete"
    complete.mkdir()
    output = _job_with_output(complete)
    (output / "running_scf.log").write_text(
        "Final Etot = -2.5 eV\n",
        encoding="utf-8",
    )

    assert main(
        [
            "postprocess",
            "result",
            "-j",
            str(incomplete),
            str(complete),
            "-p",
            "energy",
        ]
    ) == 0

    output_text = capsys.readouterr().out
    assert "incomplete" in output_text
    assert "complete" in output_text
    assert "-2.50000000" in output_text


def test_read_dos_from_job(tmp_path):
    output = _job_with_output(tmp_path)
    np.savetxt(
        output / "DOS1_smearing.dat",
        [[-1.0, 0.1], [0.0, 0.2], [1.0, 0.3]],
    )
    np.savetxt(
        output / "DOS2_smearing.dat",
        [[-1.0, 0.4], [0.0, 0.5], [1.0, 0.6]],
    )

    result = read_dos_from_job(tmp_path)

    np.testing.assert_allclose(result["energy"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        result["data"], [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
    )


def _develop_dos_text(dos_values, istep=1):
    rows = "\n".join(
        f"{-1.0 + 1.0 * point:.6f} {value:.6f} {value:.6f} {value:.6f} {value:.6f}"
        for point, value in enumerate(dos_values)
    )
    return (
        f"{istep}    # ionic step\n3 # number of points\n"
        "#  energy(eV)         dos     dos_int  dos_smear(1/eV)  dos_smear_int\n"
        f"{rows}\n"
    )


def test_read_dos_from_job_develop_nao_files(tmp_path):
    output = _job_with_output(tmp_path)
    (output / "doss1g1_nao.txt").write_text(_develop_dos_text([0.1, 0.2, 0.3]))
    (output / "doss2g1_nao.txt").write_text(_develop_dos_text([0.4, 0.5, 0.6]))

    result = read_dos_from_job(tmp_path)

    np.testing.assert_allclose(result["energy"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        result["data"], [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
    )


def test_read_dos_from_job_develop_pw_single_spin(tmp_path):
    # PW with nspin 1 writes one spinless dos.txt on this branch
    output = _job_with_output(tmp_path)
    (output / "dos.txt").write_text(_develop_dos_text([0.1, 0.2, 0.3]))

    result = read_dos_from_job(tmp_path)

    np.testing.assert_allclose(result["data"], [[0.1], [0.2], [0.3]])


def test_read_dos_from_job_develop_last_ionic_step(tmp_path):
    output = _job_with_output(tmp_path)
    # a relaxation run names one file per ionic step
    (output / "doss1g1_nao.txt").write_text(_develop_dos_text([0.1, 0.2, 0.3], istep=1))
    (output / "doss1g2_nao.txt").write_text(_develop_dos_text([0.2, 0.3, 0.4], istep=2))
    # with out_app_flag 1 the spin-2 channel appends one block per step
    (output / "doss2_nao.txt").write_text(
        _develop_dos_text([0.7, 0.8, 0.9], istep=1)
        + _develop_dos_text([0.9, 1.0, 1.1], istep=2)
    )

    result = read_dos_from_job(tmp_path)

    np.testing.assert_allclose(result["data"][:, 0], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(result["data"][:, 1], [0.9, 1.0, 1.1])


def _develop_pdos_text(atoms, spin_values):
    """Build one develop-branch PDOS file; atoms are (atom_index, species,
    channels-per-point multipliers) and spin_values holds the base
    up-or-down values."""
    rows = []
    for atom_index, species, multipliers in atoms:
        for point, value in enumerate(spin_values):
            fields = [f"{-1.0 + 1.0 * point:.6f}", f"{atom_index}", species]
            fields += [f"{value * factor:.6f}" for factor in multipliers]
            rows.append("  ".join(fields))
    return (
        "# istep: 1\n# npoints: 3\n"
        "# energy(eV)  atom  species  pdos(1/eV), columns: s(m=0) p(m=0,+1,-1)\n"
        + "\n".join(rows)
        + "\n"
    )


def test_read_pdos_from_job_develop_nao_files(tmp_path):
    output = _job_with_output(tmp_path)
    up_atoms = [(1, "H", [1.0]), (2, "O", [1.0, 2.0, 3.0, 4.0])]
    down_atoms = [(1, "H", [10.0]), (2, "O", [10.0, 20.0, 30.0, 40.0])]
    (output / "pdoss1g1_nao.txt").write_text(
        _develop_pdos_text(up_atoms, [0.1, 0.2, 0.3])
    )
    (output / "pdoss2g1_nao.txt").write_text(
        _develop_pdos_text(down_atoms, [0.4, 0.5, 0.6])
    )

    result = read_pdos_from_job(tmp_path)

    np.testing.assert_allclose(result["energy"], [-1.0, 0.0, 1.0])
    metadata = [
        (orbital["atom_index"], orbital["species"], orbital["l"], orbital["m"])
        for orbital in result["orbitals"]
    ]
    assert metadata == [
        (1, "H", 0, 0),
        (2, "O", 0, 0),
        (2, "O", 1, 0),
        (2, "O", 1, 1),
        (2, "O", 1, 2),
    ]
    # each orbital pairs the up and down columns of the two spin files
    np.testing.assert_allclose(result["orbitals"][0]["data"], [[0.1, 4.0], [0.2, 5.0], [0.3, 6.0]])
    np.testing.assert_allclose(result["orbitals"][4]["data"], [[0.4, 16.0], [0.8, 20.0], [1.2, 24.0]])


def test_read_pdos_from_job(tmp_path):
    output = _job_with_output(tmp_path)
    (output / "PDOS").write_text(
        """<pdos>
<energy_values>-1.0 0.0 1.0</energy_values>
<orbital index="1" atom_index="1" species="H" l="0" m="0" z="1">
<data>
0.1
0.2
0.3
</data>
</orbital>
</pdos>
""",
        encoding="utf-8",
    )

    result = read_pdos_from_job(tmp_path)

    np.testing.assert_allclose(result["energy"], [-1.0, 0.0, 1.0])
    assert result["orbitals"][0]["species"] == "H"
    np.testing.assert_allclose(result["orbitals"][0]["data"], [[0.1], [0.2], [0.3]])


def test_read_orbital_xml_without_energy_grid(tmp_path):
    xml_file = tmp_path / "PBANDS_1"
    xml_file.write_text(
        """<projected_bands>
<orbital index="2" atom_index="3" species="O" l="1" m="2" z="1">
<data>
0.1 0.2
0.3 0.4
</data>
</orbital>
</projected_bands>
""",
        encoding="utf-8",
    )

    result = read_orbital_xml(xml_file)

    assert result["energy"] is None
    assert result["orbitals"][0]["atom_index"] == 3
    np.testing.assert_allclose(result["orbitals"][0]["data"], [[0.1, 0.2], [0.3, 0.4]])


def test_grid_from_stru_and_charge_to_potential(tmp_path):
    stru_file = tmp_path / "STRU"
    stru_file.write_text(
        """ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
5 0 0
0 5 0
0 0 5

ATOMIC_POSITIONS
Cartesian

H
0.0
1
1 2 3
""",
        encoding="utf-8",
    )

    grid = Grid.from_stru(stru_file, (2, 2, 2))
    potential = Charge(np.zeros((2, 2, 2)), grid.cell).to_pot()

    assert grid.atom_types.tolist() == [1]
    np.testing.assert_allclose(grid.atom_positions, [[1.0, 2.0, 3.0]], atol=1e-4)
    np.testing.assert_allclose(potential.data, 0.0)


def test_get_result_reads_ewald_energy(tmp_path):
    output = _job_with_output(tmp_path)
    (output / "running_scf.log").write_text(
        "E_Ewald 1.000000 13.605693\n",
        encoding="utf-8",
    )

    result = get_result_from_job(tmp_path, ["ewald_energy"], version="")

    assert result["ewald_energy"] == pytest.approx(13.605693)
