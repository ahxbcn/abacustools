"""Tests for data readers that do not depend on abacustest."""

from __future__ import annotations

import numpy as np

from abacustools.data.abacus_result import read_dos_from_job, read_pdos_from_job
from abacustools.data.band import BandData
from abacustools.data.grid import Charge, Grid


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
