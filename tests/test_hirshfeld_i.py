"""Tests for the Hirshfeld-I atomic charges."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.data.grid import Charge
from abacustools.data.hirshfeld import (
    AtomicReference,
    _reference_occupations,
    hirshfeld_charges,
    hirshfeld_i_charges,
    hirshfeld_i_weights,
    pseudo_atomic_references,
    read_reference_densities,
)
from abacustools.io.pseudo import UPF
from abacustools.io.stru import AbacusSTRU

# A two-orbital norm-conserving pseudopotential whose PP_RHOATOM is exactly the
# occupation-weighted sum of the PP_PSWFC radial densities.
_CHI_S = [0.00, 0.30, 0.50, 0.35, 0.15]
_CHI_P = [0.00, 0.40, 0.55, 0.30, 0.10]
_R = [0.0, 1.0, 2.0, 3.0, 4.0]
_OCC_S, _OCC_P = 2.0, 2.0
_NORM_S = sum(value * value for value in _CHI_S)
_NORM_P = sum(value * value for value in _CHI_P)
_RHOATOM = [_OCC_S * s * s / _NORM_S + _OCC_P * p * p / _NORM_P for s, p in zip(_CHI_S, _CHI_P)]

_UPF = """\
<UPF version="2.0.1">
  <PP_HEADER element="C" z_valence="4.0" l_max="1" mesh_size="5"/>
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
    <PP_CHI.1 index="1" l="0" label="2S" occupation="2.0" pseudo_energy="-1.0" size="5">{s}</PP_CHI.1>
    <PP_CHI.2 index="2" l="1" label="2P" occupation="2.0" pseudo_energy="-0.4" size="5">{p}</PP_CHI.2>
  </PP_PSWFC>
  <PP_RHOATOM size="5">{rho}</PP_RHOATOM>
</UPF>
"""

_UPF_TEXT = _UPF.format(
    s=" ".join(f"{value:.10f}" for value in _CHI_S),
    p=" ".join(f"{value:.10f}" for value in _CHI_P),
    rho=" ".join(f"{value:.10f}" for value in _RHOATOM),
)


def _write_upf(tmp_path: Path) -> Path:
    path = tmp_path / "C.upf"
    path.write_text(_UPF_TEXT, encoding="utf-8")
    return path


def test_pseudo_atomic_references_reproduce_rhoatom(tmp_path: Path) -> None:
    upf = UPF.read_from_file(_write_upf(tmp_path))
    reference = pseudo_atomic_references(upf)
    assert reference.element == "C"
    assert reference.populations == (2, 3, 4, 5, 6)

    r_bohr = np.asarray(upf.r, dtype=float)
    rab = np.asarray(upf.rab, dtype=float)
    for population in reference.populations:
        rho = reference.densities[population]
        # Back to 4 pi r^2 rho(r) in Bohr and integrate on the radial mesh.
        rhoatom = rho * BOHR_TO_ANG**3 * 4.0 * np.pi * r_bohr**2
        assert float(np.sum(rhoatom * rab)) == pytest.approx(population, abs=1e-9)
    # The neutral population is the pseudopotential's own PP_RHOATOM.
    neutral = reference.densities[4] * BOHR_TO_ANG**3 * 4.0 * np.pi * r_bohr**2
    assert np.allclose(neutral, np.asarray(upf.rhoatom), atol=1e-12)


def test_reference_density_interpolates_linearly() -> None:
    r = np.linspace(0.0, 4.0, 5)
    low = np.full_like(r, 1.0)
    high = np.full_like(r, 3.0)
    reference = AtomicReference("C", r, {3: low, 4: high})
    assert np.allclose(reference.at(3.0), low)
    assert np.allclose(reference.at(4.0), high)
    assert np.allclose(reference.at(3.25), 0.75 * low + 0.25 * high)
    with pytest.raises(KeyError):
        reference.at(5.0)


def test_reference_occupations_fill_by_aufbau() -> None:
    waves = [
        {"l": 0, "occupation": 2.0, "pseudo_energy": -1.0},
        {"l": 1, "occupation": 2.0, "pseudo_energy": -0.4},
    ]
    neutral = [2.0, 2.0]
    # Removing one electron comes from the higher 2p orbital.
    assert _reference_occupations(waves, neutral, 3.0) == [2.0, 1.0]
    # Adding one electron goes to the 2p orbital as well.
    assert _reference_occupations(waves, neutral, 5.0) == [2.0, 3.0]
    # Two electrons emptied from the 2p shell.
    assert _reference_occupations(waves, neutral, 2.0) == [2.0, 0.0]


def test_read_reference_densities_round_trip(tmp_path: Path) -> None:
    r = np.linspace(0.0, 3.0, 4)
    for population, scale in ((3, 1.0), (4, 2.0)):
        rows = "\n".join(f"{x:.6f} {scale * (1.0 - x / 4.0):.6f}" for x in r)
        (tmp_path / f"C_{population}.dat").write_text(rows + "\n", encoding="utf-8")
    references = read_reference_densities(tmp_path)
    assert set(references) == {"C"}
    assert references["C"].populations == (3, 4)
    assert np.allclose(
        references["C"].at(3.5),
        0.5 * references["C"].densities[3] + 0.5 * references["C"].densities[4],
    )


# --- A single-atom job whose density is the pseudopotential proatom. ---

_H_UPF = """\
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

_STRU = """ATOMIC_SPECIES
H 1.008 H.upf

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
{count}
{positions}
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
        lines.append("".join(f"{value:20.12e}" for value in flat[start : start + 6]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _proatom_radial(upf_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r, rho)`` of a UPF's PP_RHOATOM, in Angstrom and e/Angstrom^3."""
    lines = upf_text.split("<PP_RHOATOM", 1)[1]
    rhoatom = np.array([float(v) for v in lines.split(">", 1)[1].split("<")[0].split()])
    r_bohr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    rho = np.zeros_like(r_bohr)
    rho[1:] = rhoatom[1:] / (4.0 * np.pi * r_bohr[1:] ** 2)
    rho[0] = rho[1]
    return r_bohr * BOHR_TO_ANG, rho / BOHR_TO_ANG**3


def _write_job(job: Path, positions, weights, shape=(48, 48, 48)) -> float:
    """Write a job whose density is ``sum weights[i] * proatom(r - R_i)``.

    Returns the discrete electron count of that density.
    """
    job.mkdir(parents=True, exist_ok=True)
    (job / "H.upf").write_text(_H_UPF, encoding="utf-8")
    position_lines = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in positions)
    (job / "STRU").write_text(
        _STRU.format(count=len(positions), positions=position_lines), encoding="utf-8"
    )
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\nsuffix ABACUS\nout_chg 1\n",
        encoding="utf-8",
    )
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True, exist_ok=True)
    (output / "running_scf.log").write_text(
        " fft grid: 48 48 48\n charge density convergence is achieved\n", encoding="utf-8"
    )

    cell = np.diag([10.0, 10.0, 10.0])
    r_ang, rho_ang = _proatom_radial(_H_UPF)
    i = np.arange(shape[0]) / shape[0]
    j = np.arange(shape[1]) / shape[1]
    k = np.arange(shape[2]) / shape[2]
    frac = np.stack(np.meshgrid(i, j, k, indexing="ij"), axis=-1)
    points = frac.reshape(-1, 3) @ cell
    values = np.zeros(points.shape[0])
    for position, weight in zip(positions, weights):
        distance = np.linalg.norm(points - np.asarray(position, dtype=float), axis=1)
        values += weight * np.interp(distance, r_ang, rho_ang, right=0.0)
    _write_cube(
        output / "SPIN1_CHG.cube",
        shape,
        cell,
        [1] * len(positions),
        positions,
        (values * BOHR_TO_ANG**3).reshape(shape),
    )
    volume_element = abs(float(np.linalg.det(cell))) / (shape[0] * shape[1] * shape[2])
    return float(values.sum() * volume_element)


def test_hirshfeld_i_of_a_proatom_is_neutral(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job, [(5.0, 5.0, 5.0)], [1.0])
    result = hirshfeld_i_charges(job)
    assert result.converged
    assert result.grid == (48, 48, 48)
    assert abs(float(result.charges[0])) < 0.25
    assert result.populations[0] == pytest.approx(result.valence[0] - result.charges[0])


def test_hirshfeld_i_weights_reproduce_the_populations(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job, [(5.0, 5.0, 5.0), (5.0, 5.0, 7.0)], [1.0, 1.6])
    result = hirshfeld_i_charges(job)
    density = Charge.from_cube(str(job / "OUT.ABACUS" / "SPIN1_CHG.cube"), format="abacus")
    structure = AbacusSTRU.read(str(job / "STRU"))

    partition = hirshfeld_i_weights(density, structure, job=job)

    assert partition.converged
    assert partition.weights.shape == (2, 48**3)
    np.testing.assert_allclose(partition.populations, result.populations)
    occupied = density.data.reshape(-1) > 0.0
    assert np.all(partition.weights[:, occupied].sum(axis=0) <= 1.0 + 1e-12)


def test_hirshfeld_i_conserves_electrons_and_polarises(tmp_path: Path) -> None:
    job = tmp_path / "job"
    electrons = _write_job(job, [(5.0, 5.0, 5.0), (5.0, 5.0, 7.0)], [1.0, 1.6])
    result = hirshfeld_i_charges(job)
    plain = hirshfeld_charges(job)
    assert result.converged
    # The second atom carries the excess density, so it is the negative one.
    assert result.charges[1] < result.charges[0]
    # The sum of the populations equals the electrons the density carries.
    assert float(result.populations.sum()) == pytest.approx(electrons, abs=1e-6)
    # The iterative scheme is not the plain Hirshfeld one.
    assert not np.allclose(result.charges, plain.charges, atol=1e-3)


def test_hirshfeld_i_accepts_explicit_references(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job, [(5.0, 5.0, 5.0)], [1.0])
    r_ang, rho_ang = _proatom_radial(_H_UPF)
    reference = AtomicReference("H", r_ang, {0: rho_ang * 0.0, 1: rho_ang, 2: rho_ang * 2.0})
    result = hirshfeld_i_charges(job, references={"H": reference})
    assert result.reference_source == "explicit reference densities"
    assert result.converged


def test_hirshfeld_i_rejects_unknown_reference(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job, [(5.0, 5.0, 5.0)], [1.0])
    with pytest.raises(ValueError):
        hirshfeld_i_charges(job, references={})
