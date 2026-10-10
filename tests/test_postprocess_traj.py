"""Tests for the relaxation-trajectory postprocessing command."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.postprocess.traj import run as postprocess_traj
from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.md import read_relax_trajectory


LATTICE_CONSTANT = 1.8897261255
CELL_ANGSTROM = 5.0 * LATTICE_CONSTANT * BOHR_TO_ANG


DEVELOP_STRU = """# ABACUS version: 3.11.0
# Written at 2026-01-01 00:00:00
# RELAX STEP {step}, Energy: {energy:.8f} eV
# NOTE: stress and forces computed for this geometry
ATOMIC_SPECIES
Si 28.0855 Si.upf upf

LATTICE_CONSTANT
1.8897261255 # in Bohr (= 1 Angstrom); lattice vectors below are in Angstrom

LATTICE_VECTORS # in Angstrom
{a:.9f} 0.000000000 0.000000000
0.000000000 {a:.9f} 0.000000000
0.000000000 0.000000000 {a:.9f}

ATOMIC_POSITIONS
Cartesian_angstrom # positions in Angstrom, forces in eV/Angstrom

Si #label
0.0000 #magnetism (default, overridden by per-atom mag below)
2 #number of atoms
{x:.10f} 0.0000000000 0.0000000000 m 1 1 1 f 0.100000 0.200000 0.300000
0.5000000000 0.5000000000 0.5000000000 m 1 1 1 f -0.100000 -0.200000 -0.300000
"""

LTS_STRU = """ATOMIC_SPECIES
Si 28.0855 Si.upf Si

LATTICE_CONSTANT
1.8897261255

LATTICE_VECTORS
{a:.9f} 0.0 0.0
0.0 {a:.9f} 0.0
0.0 0.0 {a:.9f}

ATOMIC_POSITIONS
Direct

Si #label
0.0 #magnetism
2 #number of atoms
0.0000000000 0.0000000000 0.0000000000 m 1 1 1
0.5000000000 0.5000000000 0.5000000000 m 1 1 1
"""

RELAX_LOG = """
 STEP OF RELAXATION : 1
 final etot is -10.000000 eV
 Relaxation is not converged yet!

 STEP OF RELAXATION : 2
 final etot is -10.250000 eV
 Relaxation is converged!
 Total  Time  : 0 h 0 mins 1 secs
"""


def _job(tmp_path: Path, *, calculation: str = "relax", develop: bool = True) -> Path:
    """Create a relaxation job with per-step structures and a running log."""
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        f"suffix ABACUS\ncalculation {calculation}\nnspin 1\n",
        encoding="utf-8",
    )
    (output / f"running_{calculation}.log").write_text(RELAX_LOG, encoding="utf-8")
    if develop:
        (output / "STRU1").write_text(
            DEVELOP_STRU.format(step=1, energy=-10.0, a=5.0, x=0.0), encoding="utf-8"
        )
        (output / "STRU2").write_text(
            DEVELOP_STRU.format(step=2, energy=-10.25, a=5.5, x=0.2),
            encoding="utf-8",
        )
    else:
        (output / "STRU_ION1_D").write_text(
            LTS_STRU.format(a=5.0), encoding="utf-8"
        )
        (output / "STRU_ION2_D").write_text(
            LTS_STRU.format(a=5.5), encoding="utf-8"
        )
    return job


def test_read_develop_relaxation_trajectory(tmp_path: Path) -> None:
    frames = read_relax_trajectory(_job(tmp_path), version="develop")

    assert [frame.step for frame in frames] == [1, 2]
    assert [frame.energy for frame in frames] == pytest.approx([-10.0, -10.25])
    np.testing.assert_allclose(frames[0].positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(frames[1].positions[0], [0.2, 0.0, 0.0])
    np.testing.assert_allclose(
        frames[0].forces, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
    )
    np.testing.assert_allclose(np.diag(frames[1].cell), [5.5, 5.5, 5.5], atol=1e-5)
    assert frames[0].velocities is None


def test_read_lts_relaxation_trajectory_has_no_forces(tmp_path: Path) -> None:
    frames = read_relax_trajectory(
        _job(tmp_path, develop=False), version="LTS3.10.1"
    )

    assert [frame.step for frame in frames] == [1, 2]
    np.testing.assert_allclose(
        frames[0].positions[1], [CELL_ANGSTROM / 2.0] * 3, atol=1e-6
    )
    assert frames[0].forces is None


def test_cell_relax_uses_the_cell_relax_log(tmp_path: Path) -> None:
    frames = read_relax_trajectory(
        _job(tmp_path, calculation="cell-relax"), version="develop"
    )

    assert [frame.step for frame in frames] == [1, 2]
    assert frames[0].energy == pytest.approx(-10.0)
    assert frames[1].cell[2, 2] > frames[0].cell[2, 2]


def test_no_energy_skips_the_running_log(tmp_path: Path) -> None:
    frames = read_relax_trajectory(_job(tmp_path), version="develop", with_log=False)

    assert all(frame.energy is None for frame in frames)


def test_read_relaxation_trajectory_requires_step_structures(tmp_path: Path) -> None:
    job = _job(tmp_path)
    for name in ("STRU1", "STRU2"):
        (job / "OUT.ABACUS" / name).unlink()

    with pytest.raises(FileNotFoundError, match="out_stru"):
        read_relax_trajectory(job)


def test_read_relaxation_trajectory_rejects_md_jobs(tmp_path: Path) -> None:
    job = _job(tmp_path)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\ncalculation md\nnspin 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relax or cell-relax"):
        read_relax_trajectory(job)


def test_postprocess_traj_writes_a_trajectory(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from ase.io import read

    job = _job(tmp_path)

    assert postprocess_traj(Namespace(
        job=job, output="relax_trajectory.extxyz", format=None, first=None,
        last=None, stride=1, no_energy=False, version="develop", json=True,
    )) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["frames"] == 2
    assert report["atoms"] == 2
    assert report["steps"] == [1, 2]
    assert report["has_forces"] and report["has_energy"]
    assert not report["has_velocities"]
    path = job / "relax_trajectory.extxyz"
    assert path.is_file()
    structures = read(str(path), index=":")
    assert len(structures) == 2
    np.testing.assert_allclose(structures[1].positions[0], [0.2, 0.0, 0.0], atol=1e-6)
    assert structures[0].get_potential_energy() == pytest.approx(-10.0)


def test_postprocess_traj_reports_a_missing_trajectory(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job = _job(tmp_path, calculation="md")

    assert postprocess_traj(Namespace(
        job=job, output="relax_trajectory.extxyz", format=None, first=None,
        last=None, stride=1, no_energy=False, version="develop", json=False,
    )) == 1

    assert "relax or cell-relax" in capsys.readouterr().out
