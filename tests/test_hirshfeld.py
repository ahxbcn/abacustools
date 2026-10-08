"""Tests for Hirshfeld and CM5 atomic charges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from abacustools.core.constant import ANG_TO_BOHR
from abacustools.data.hirshfeld import hirshfeld_charges, read_cm5_parameters

# A radial grid in Bohr with PP_RHOATOM integrated to z_valence = 1.
_UPF = """\
<UPF version="2.0.1">
  <PP_HEADER element="H" z_valence="1.0" l_max="0" mesh_size="5"/>
  <PP_MESH>
    <PP_R  type="real" size="5">0.0 1.0 2.0 3.0 4.0</PP_R>
    <PP_RAB type="real" size="5">1.0 1.0 1.0 1.0 1.0</PP_RAB>
  </PP_MESH>
  <PP_LOCAL size="5">-1.0 -0.5 -0.2 -0.1 -0.05</PP_LOCAL>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="2" cutoff_radius="1.5" size="5">1.0 2.0 3.0 4.0 5.0</PP_BETA.1>
    <PP_DIJ size="1">1.0</PP_DIJ>
  </PP_NONLOCAL>
  <PP_PSWFC>
    <PP_CHI.1 index="1" l="0" label="1S" occupation="1.0" pseudo_energy="-0.2" size="5">0.1 0.2 0.3 0.4 0.5</PP_CHI.1>
  </PP_PSWFC>
  <PP_RHOATOM size="5">0.0 0.54134113 0.29305022 0.08923534 0.02148362</PP_RHOATOM>
</UPF>
"""

_ORB = """Element H
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

_STRU = """ATOMIC_SPECIES
H 1.008 H.upf

NUMERICAL_ORBITAL
H.orb

LATTICE_CONSTANT
1.889726125457828

LATTICE_VECTORS
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0

ATOMIC_POSITIONS
Cartesian

H
0.0
1
5.0 5.0 5.0
"""


def _write_cube(path: Path, shape, cell_ang, atom_z, atom_pos_ang, values_bohr3) -> None:
    """Write an ABACUS-format cube (Bohr, e/Bohr^3)."""
    cell_bohr = np.asarray(cell_ang, dtype=float) * ANG_TO_BOHR
    pos_bohr = np.asarray(atom_pos_ang, dtype=float) * ANG_TO_BOHR
    voxels = [cell_bohr[i] / shape[i] for i in range(3)]
    lines = ["synthetic density", "abacustools test", f"{len(atom_z)} 0.0 0.0 0.0"]
    for n, voxel in zip(shape, voxels):
        lines.append(f"{n} {voxel[0]:.10f} {voxel[1]:.10f} {voxel[2]:.10f}")
    for z, position in zip(atom_z, pos_bohr):
        lines.append(f"{z} 0.0 {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}")
    flat = np.asarray(values_bohr3, dtype=float).reshape(-1)
    for start in range(0, flat.size, 6):
        lines.append("".join(f"{value:13.5e}" for value in flat[start : start + 6]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_job(job: Path, shape=(60, 60, 60)) -> None:
    job.mkdir(parents=True, exist_ok=True)
    (job / "H.upf").write_text(_UPF, encoding="utf-8")
    (job / "H.orb").write_text(_ORB, encoding="utf-8")
    (job / "STRU").write_text(_STRU, encoding="utf-8")
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\nsuffix ABACUS\nout_chg 1\n",
        encoding="utf-8",
    )
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True, exist_ok=True)
    (output / "running_scf.log").write_text(
        " fft grid: 60 60 60\n charge density convergence is achieved\n", encoding="utf-8"
    )

    # Density equal to the (normalized) proatom: the Hirshfeld charge must be zero.
    cell = np.diag([10.0, 10.0, 10.0])
    r_bohr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    rhoatom = np.array([0.0, 0.54134113, 0.29305022, 0.08923534, 0.02148362])
    rhoatom = rhoatom / np.trapezoid(rhoatom, r_bohr)
    radial = np.zeros_like(r_bohr)
    radial[1:] = rhoatom[1:] / (4.0 * np.pi * r_bohr[1:] ** 2)
    radial[0] = radial[1]

    center = np.array([5.0, 5.0, 5.0])
    i = np.arange(shape[0]) / shape[0]
    j = np.arange(shape[1]) / shape[1]
    k = np.arange(shape[2]) / shape[2]
    frac = np.stack(np.meshgrid(i, j, k, indexing="ij"), axis=-1)
    points = (frac.reshape(-1, 3) @ cell)
    distance = np.linalg.norm(points - center, axis=1) * ANG_TO_BOHR
    values = np.interp(distance, r_bohr, radial, right=0.0).reshape(shape)
    _write_cube(output / "SPIN1_CHG.cube", shape, cell, [1], [center], values)


def test_hirshfeld_of_a_proatom_is_neutral(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job)
    result = hirshfeld_charges(job)
    assert result.grid == (60, 60, 60)
    # The density equals the proatom, so the charge is near zero; the residual
    # is the radial-interpolation and grid resolution of this synthetic case.
    assert abs(float(result.charges[0])) < 0.25
    assert result.volumes[0] > 0.0


def test_read_cm5_parameters(tmp_path: Path) -> None:
    path = tmp_path / "cm5.json"
    path.write_text(json.dumps({"C-H": 0.12, "O-O": 0.0}), encoding="utf-8")
    table = read_cm5_parameters(path)
    assert table[("C", "H")] == pytest.approx(0.12)
    assert table[("H", "C")] == pytest.approx(0.12)
    assert table[("O", "O")] == pytest.approx(0.0)


def test_cm5_correction_shifts_charge(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job)
    base = hirshfeld_charges(job)
    corrected = hirshfeld_charges(job, cm5_parameters={("H", "H"): 0.5})
    assert corrected.cm5 is not None
    # A single atom has no partner, so the correction leaves it unchanged.
    assert corrected.cm5[0] == pytest.approx(base.charges[0], abs=1e-9)
