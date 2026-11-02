"""Tests for the approximate QTAIM bond-critical-point search."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from abacustools.commands.postprocess.qtaim import run
from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.grid import Charge
from abacustools.data.qtaim import (
    CoreRadius,
    analyze_qtaim_bcp,
    resolve_core_radii,
    search_bcp,
)
from abacustools.io.stru import AbacusSTRU


H_UPF = """\
<UPF version="2.0.1">
  <PP_HEADER element="H" z_valence="1.0" l_max="0" mesh_size="3"/>
  <PP_MESH>
    <PP_R type="real" size="3">0.0 1.0 2.0</PP_R>
    <PP_RAB type="real" size="3">1.0 1.0 1.0</PP_RAB>
  </PP_MESH>
  <PP_LOCAL size="3">-1.0 -0.5 -0.2</PP_LOCAL>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="2" cutoff_radius="2.0" size="3">1.0 2.0 3.0</PP_BETA.1>
    <PP_DIJ size="1">1.0</PP_DIJ>
  </PP_NONLOCAL>
  <PP_RHOATOM size="3">0.1 0.2 0.3</PP_RHOATOM>
</UPF>
"""


def _synthetic_job(root: Path, *, separation: float = 1.2) -> Path:
    cell = np.diag([8.0, 6.0, 6.0])
    shape = (80, 32, 32)
    first = np.asarray([4.0 - separation / 2.0, 3.0, 3.0])
    second = np.asarray([4.0 + separation / 2.0, 3.0, 3.0])
    indices = np.stack(np.meshgrid(*(np.arange(value) for value in shape), indexing="ij"), axis=-1)
    cartesian = indices * (cell.diagonal() / np.asarray(shape))
    sigma = 0.32
    density = sum(
        np.exp(-np.sum((cartesian - center) ** 2, axis=-1) / (2.0 * sigma**2))
        for center in (first, second)
    )
    charge = Charge(
        density,
        cell,
        np.asarray([first, second]),
        np.asarray([1, 1]),
        np.asarray([1.0, 1.0]),
    )
    root.mkdir(parents=True)
    (root / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type pw\nnspin 1\n",
        encoding="utf-8",
    )
    (root / "H.upf").write_text(H_UPF, encoding="utf-8")
    (root / "STRU").write_text(
        "ATOMIC_SPECIES\n"
        "H 1.008 H.upf\n\n"
        "LATTICE_CONSTANT\n"
        "1.889726\n\n"
        "LATTICE_VECTORS\n"
        "8 0 0\n"
        "0 6 0\n"
        "0 0 6\n\n"
        "ATOMIC_POSITIONS\n"
        "Cartesian\n"
        "H\n"
        "0.0\n"
        "2\n"
        f"{first[0]} {first[1]} {first[2]} 1 1 1\n"
        f"{second[0]} {second[1]} {second[2]} 1 1 1\n",
        encoding="utf-8",
    )
    charge.save_cube(str(root / "density.cube"), format="abacus")
    return root


def test_bcp_search_finds_the_saddle_between_two_gaussian_atoms(tmp_path: Path) -> None:
    job = _synthetic_job(tmp_path / "job")
    structure = AbacusSTRU.read(str(job / "STRU"))
    density = Charge.from_cube(str(job / "density.cube"), format="abacus")
    assert structure is not None

    result = search_bcp(
        density,
        structure,
        core_radii=(
            CoreRadius(1, "H", 0.2),
            CoreRadius(2, "H", 0.2),
        ),
        pair_cutoff=2.0,
        gradient_tolerance=1.0e-4,
    )

    assert len(result.points) == 1
    point = result.points[0]
    assert (point.atom1, point.atom2) == (1, 2)
    np.testing.assert_allclose(point.position, [4.0, 3.0, 3.0], atol=0.03)
    assert point.rho > 0.0
    assert point.laplacian > 0.0
    assert point.eigenvalues[0] < 0.0
    assert point.eigenvalues[1] < 0.0
    assert point.eigenvalues[2] > 0.0
    assert point.core_clearance > 0.0


def test_core_exclusion_can_leave_no_allowed_search_segment(tmp_path: Path) -> None:
    job = _synthetic_job(tmp_path / "job")
    structure = AbacusSTRU.read(str(job / "STRU"))
    density = Charge.from_cube(str(job / "density.cube"), format="abacus")
    assert structure is not None

    result = search_bcp(
        density,
        structure,
        core_radii=(
            CoreRadius(1, "H", 0.7),
            CoreRadius(2, "H", 0.7),
        ),
        pair_cutoff=2.0,
    )

    assert result.candidate_pairs == 1
    assert result.seeds == 0
    assert result.points == ()


def test_qtaim_command_writes_json_with_an_external_density(tmp_path: Path, capsys) -> None:
    job = _synthetic_job(tmp_path / "job")
    args = Namespace(
        job=job,
        density="density.cube",
        core_radius="0.2",
        cutoff=2.0,
        line_samples=31,
        seeds_per_pair=3,
        gradient_tol=1.0e-4,
        max_iterations=60,
        step_limit=0.25,
        merge_distance=0.08,
        grid=None,
        lat0=None,
        output="qtaim.json",
        json=True,
    )

    assert run(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["method"] == "qtaim-bcp-search"
    assert report["density_source"] == str((job / "density.cube").resolve())
    assert len(report["critical_points"]) == 1
    assert (job / "qtaim.json").is_file()


def test_analyze_qtaim_bcp_accepts_a_scalar_core_radius(tmp_path: Path) -> None:
    job = _synthetic_job(tmp_path / "job")

    result = analyze_qtaim_bcp(
        job,
        density_cube="density.cube",
        core_radius=0.2,
        pair_cutoff=2.0,
        gradient_tolerance=1.0e-4,
    )

    assert len(result.points) == 1
    assert [entry.radius for entry in result.core_radii] == [0.2, 0.2]


def test_auto_core_radius_uses_the_upf_projector_cutoff(tmp_path: Path) -> None:
    job = _synthetic_job(tmp_path / "job")

    radii, source = resolve_core_radii(job, setting="auto")

    assert len(radii) == 2
    np.testing.assert_allclose([entry.radius for entry in radii], [2.0 * BOHR_TO_ANG] * 2)
    assert "UPF non-local projector cutoff" in source
