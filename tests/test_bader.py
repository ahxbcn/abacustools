"""Tests for Bader charge analysis."""

from __future__ import annotations

import stat
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.postprocess.bader import run
from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.data.bader import (
    BaderError,
    analyze_bader,
    fft_grid_from_log,
    find_bader_executable,
    read_acf,
)
from abacustools.data.grid import Charge, RestartCharge


FAKE_BADER = '''#!/usr/bin/env python3
import os
import sys

charge = sys.argv[-1]
with open(charge) as handle:
    lines = handle.readlines()
natom = int(lines[2].split()[0])
atoms = [line.split() for line in lines[6:6 + natom]]
is_spin = "spin" in os.path.basename(charge)
rows = []
for i, atom in enumerate(atoms, start=1):
    z_valence = float(atom[1])
    value = 0.5 if is_spin else z_valence - 0.1
    rows.append((i, float(atom[2]), float(atom[3]), float(atom[4]), value))
out = [
    "    #         X           Y           Z       CHARGE      MIN DIST   ATOMIC VOL",
    " " + "-" * 80,
]
for index, x, y, z, value in rows:
    out.append(f"{index:5d} {x:11.6f} {y:11.6f} {z:11.6f} {value:11.6f} {1.0:11.6f} {2.0:11.6f}")
out.append(" " + "-" * 80)
out.append("    VACUUM CHARGE:               0.0000")
out.append("    VACUUM VOLUME:               0.0000")
out.append(f"    NUMBER OF ELECTRONS:      {sum(row[4] for row in rows):.4f}")
with open("ACF.dat", "w") as handle:
    handle.write("\\n".join(out) + "\\n")
for name in ("BCF.dat", "AVF.dat"):
    open(name, "w").close()
'''


def _fake_bader(tmp_path: Path) -> Path:
    path = tmp_path / "bader"
    path.write_text(FAKE_BADER, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _cube(path: Path, data: np.ndarray, valence: float = 4.0, numbers=(14, 14)) -> None:
    cell = np.diag([4.0, 4.0, 4.0])
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    charges = [valence] * len(numbers)
    Charge(data, cell, positions, list(numbers), charges).save_cube(str(path), format="abacus")


def _namespace(job: Path, **overrides) -> Namespace:
    values = dict(
        job=str(job),
        output=None,
        bader_exe=None,
        cube=None,
        reference=None,
        grid=None,
        lat0=ANG_TO_BOHR,
        vacuum=None,
        workdir=None,
        keep_cubes=False,
        json=True,
    )
    values.update(overrides)
    return Namespace(**values)


def test_read_acf_parses_columns_and_footer(tmp_path: Path) -> None:
    acf = tmp_path / "ACF.dat"
    acf.write_text(
        "    #         X           Y           Z       CHARGE      MIN DIST   ATOMIC VOL\n"
        " " + "-" * 80 + "\n"
        "    1    0.000000    0.000000    0.000000    4.000028     1.803127   132.651936\n"
        "    2   -7.650000    7.650000    7.650000    3.900000     1.803158   132.651936\n"
        " " + "-" * 80 + "\n"
        "    VACUUM CHARGE:               0.0000\n"
        "    VACUUM VOLUME:               0.0000\n"
        "    NUMBER OF ELECTRONS:         7.9000\n",
        encoding="utf-8",
    )
    records, vacuum_charge, vacuum_volume, electrons = read_acf(acf)
    assert len(records) == 2
    assert records[0]["index"] == 1
    assert records[0]["charge"] == pytest.approx(4.000028)
    assert records[1]["position"] == pytest.approx((-7.65, 7.65, 7.65))
    assert vacuum_charge == pytest.approx(0.0)
    assert vacuum_volume == pytest.approx(0.0)
    assert electrons == pytest.approx(7.9)


def test_find_bader_executable_missing_raises() -> None:
    with pytest.raises(BaderError):
        find_bader_executable("definitely-not-a-bader-executable")


def test_fft_grid_from_log(tmp_path: Path) -> None:
    log = tmp_path / "running_scf.log"
    log.write_text(
        "some header\n            fft grid for charge/potential = [ 96, 96, 192 ]\n",
        encoding="utf-8",
    )
    assert fft_grid_from_log(log) == (96, 96, 192)
    assert fft_grid_from_log(tmp_path / "missing.log") is None


def test_fft_grid_from_log_prefers_the_finest_grid(tmp_path: Path) -> None:
    """A restart file holds rho(G) on the finest grid the log reports."""
    log = tmp_path / "running_scf.log"
    log.write_text(
        " | dimensions of FFT grid. The number of FFT grid on each processor |\n"
        "            fft grid for charge/potential = [ 36, 36, 36 ]\n"
        "                        fft grid division = [ 1, 1, 1 ]\n"
        "        big fft grid for charge/potential = [ 54, 54, 54 ]\n"
        "      fft grid for dense charge/potential = [ 54, 54, 54 ]\n",
        encoding="utf-8",
    )
    assert fft_grid_from_log(log) == (54, 54, 54)

    # An old log can report the plain grid as the larger one, such as the
    # 864 x 32 x 54 grid of a slab with a 216 x 8 x 27 "big" grid.
    log.write_text(
        "            fft grid for charge/potential = [ 864, 32, 54 ]\n"
        "        big fft grid for charge/potential = [ 216, 8, 27 ]\n",
        encoding="utf-8",
    )
    assert fft_grid_from_log(log) == (864, 32, 54)


def test_fft_grid_from_develop_log(tmp_path: Path) -> None:
    log = tmp_path / "running_scf.log"
    log.write_text(
        "                              ABACUS v3.11.0-beta9\n"
        "            FFT grid for charge/potential = [ 24, 24, 24 ]\n",
        encoding="utf-8",
    )
    assert fft_grid_from_log(log) == (24, 24, 24)


def test_bader_command_cube_nspin1(tmp_path: Path) -> None:
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 1\n", encoding="utf-8"
    )
    _cube(output / "SPIN1_CHG.cube", np.ones((4, 4, 4)))

    exe = _fake_bader(tmp_path)
    assert run(_namespace(job, bader_exe=str(exe))) == 0

    analysis = analyze_bader(job, exe=str(exe))
    assert analysis.nspin == 1
    assert [atom.element for atom in analysis.atoms] == ["Si", "Si"]
    assert analysis.atoms[0].bader_charge == pytest.approx(3.9)
    assert analysis.atoms[0].net_charge == pytest.approx(0.1)
    assert analysis.atoms[0].spin_moment is None
    assert analysis.number_of_electrons == pytest.approx(7.8)


def test_bader_command_cube_nspin2_reports_spin_moments(tmp_path: Path) -> None:
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 2\n", encoding="utf-8"
    )
    _cube(output / "SPIN1_CHG.cube", np.full((4, 4, 4), 0.6))
    _cube(output / "SPIN2_CHG.cube", np.full((4, 4, 4), 0.4))

    exe = _fake_bader(tmp_path)
    analysis = analyze_bader(job, exe=str(exe))
    assert analysis.nspin == 2
    assert all(atom.spin_moment == pytest.approx(0.5) for atom in analysis.atoms)
    assert analysis.total_net_charge == pytest.approx(0.2)


def test_bader_command_restart_input(tmp_path: Path, monkeypatch) -> None:
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 1\npseudo_dir ./pp\n", encoding="utf-8"
    )
    (job / "STRU").write_text(
        "ATOMIC_SPECIES\n"
        "Si 28.0855 Si.upf\n\n"
        "LATTICE_CONSTANT\n"
        "1.889726\n\n"
        "LATTICE_VECTORS\n"
        "4.0 0.0 0.0\n0.0 4.0 0.0\n0.0 0.0 4.0\n\n"
        "ATOMIC_POSITIONS\nCartesian\nSi\n0.0\n2\n"
        "0.0 0.0 0.0 1 1 1\n2.0 2.0 2.0 1 1 1\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(
        "fft grid for charge/potential = [ 4, 4, 4 ]\n", encoding="utf-8"
    )

    shape = (4, 4, 4)
    fractions = [np.fft.fftfreq(n) * n for n in shape]
    mesh = np.meshgrid(*fractions, indexing="ij")
    miller = np.stack([m.ravel() for m in mesh], axis=1).astype(np.int64)
    reciprocal = np.linalg.inv(np.diag([4.0, 4.0, 4.0]))
    rng = np.random.default_rng(0)
    rhog = rng.standard_normal((1, miller.shape[0])) + 1j * rng.standard_normal((1, miller.shape[0]))
    RestartCharge(rhog, miller, reciprocal).write(str(output / "ABACUS-CHARGE-DENSITY.restart"))

    monkeypatch.setattr(
        "abacustools.data.bader._valence_electrons", lambda *args, **kwargs: [4.0, 4.0]
    )
    exe = _fake_bader(tmp_path)
    analysis = analyze_bader(job, exe=str(exe))
    assert analysis.charge_source.startswith("restart")
    assert [atom.element for atom in analysis.atoms] == ["Si", "Si"]
    assert analysis.atoms[0].net_charge == pytest.approx(0.1)



def test_analyze_bader_restart_uses_lattice_constant_from_stru(
    tmp_path: Path, monkeypatch
) -> None:
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 1\npseudo_dir ./pp\n", encoding="utf-8"
    )
    # LATTICE_CONSTANT 3.779452 Bohr with 4.0 lattice vectors gives an 8 A cell;
    # the old default of 1.889726 Bohr would have produced a 4 A cell.
    (job / "STRU").write_text(
        "ATOMIC_SPECIES\n"
        "Si 28.0855 Si.upf\n\n"
        "LATTICE_CONSTANT\n"
        "3.779452\n\n"
        "LATTICE_VECTORS\n"
        "4.0 0.0 0.0\n0.0 4.0 0.0\n0.0 0.0 4.0\n\n"
        "ATOMIC_POSITIONS\nCartesian\nSi\n0.0\n2\n"
        "0.0 0.0 0.0 1 1 1\n2.0 2.0 2.0 1 1 1\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(
        "fft grid for charge/potential = [ 4, 4, 4 ]\n", encoding="utf-8"
    )
    shape = (4, 4, 4)
    fractions = [np.fft.fftfreq(n) * n for n in shape]
    mesh = np.meshgrid(*fractions, indexing="ij")
    miller = np.stack([m.ravel() for m in mesh], axis=1).astype(np.int64)
    reciprocal = np.linalg.inv(np.diag([4.0, 4.0, 4.0]))
    rng = np.random.default_rng(0)
    rhog = rng.standard_normal((1, miller.shape[0])) + 1j * rng.standard_normal((1, miller.shape[0]))
    RestartCharge(rhog, miller, reciprocal).write(str(output / "ABACUS-CHARGE-DENSITY.restart"))

    monkeypatch.setattr(
        "abacustools.data.bader._valence_electrons", lambda *args, **kwargs: [4.0, 4.0]
    )
    work = tmp_path / "work"
    analyze_bader(job, exe=str(_fake_bader(tmp_path)), workdir=work)
    written = Charge.from_cube(str(work / "charge_total.cube"), format="abacus")
    assert written.cell[0][0] == pytest.approx(8.0)


def test_analyze_bader_rejects_nspin4(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "OUT.ABACUS").mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 4\n", encoding="utf-8"
    )
    with pytest.raises(BaderError):
        analyze_bader(job, exe=str(_fake_bader(tmp_path)))


def test_analyze_bader_requires_charge_density(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "OUT.ABACUS").mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 1\n", encoding="utf-8"
    )
    with pytest.raises(BaderError):
        analyze_bader(job, exe=str(_fake_bader(tmp_path)))


def test_bader_atoms_are_reported_in_angstrom(tmp_path: Path) -> None:
    """ACF.dat holds Bohr and Bohr**3; the analysis reports Angstrom."""
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 1\n", encoding="utf-8"
    )
    _cube(output / "SPIN1_CHG.cube", np.ones((4, 4, 4)))

    analysis = analyze_bader(job, exe=str(_fake_bader(tmp_path)))

    # the cube puts the atoms at (0, 0, 0) and (2, 2, 2) Angstrom, and the fake
    # program writes the 1.0 Bohr distance and 2.0 Bohr**3 volume of its rows
    assert analysis.atoms[0].position == pytest.approx((0.0, 0.0, 0.0))
    assert analysis.atoms[1].position == pytest.approx((2.0, 2.0, 2.0))
    assert analysis.atoms[0].min_distance == pytest.approx(BOHR_TO_ANG)
    assert analysis.atoms[0].atomic_volume == pytest.approx(2.0 * BOHR_TO_ANG**3)
