"""Tests for the lattice thermal conductivity workflow."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.workflow.thermal_conductivity import (
    _conductivity_components,
    _displacement_tasks,
    _kappa_array,
    _mesh_setting,
    _scale_kpoints,
    _temperatures,
    postprocess,
    prepare,
)


def _has_phono3py() -> bool:
    try:
        import phono3py  # noqa: F401
    except ImportError:
        return False
    return True


requires_phono3py = pytest.mark.skipif(
    not _has_phono3py(), reason="phono3py is not installed"
)


def _write_job(job: Path) -> None:
    """Write a minimal one-atom ABACUS input directory."""
    job.mkdir(parents=True, exist_ok=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nscf_thr 1e-6\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(
        """ATOMIC_SPECIES
H 1.0

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
3 0 0
0 3 0
0 0 3

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0
""",
        encoding="utf-8",
    )


def _prepare_namespace(job: Path, **overrides) -> Namespace:
    values = dict(
        job=job,
        supercell_fc3=[2, 2, 2],
        supercell_fc2=None,
        displacement_stepsize_fc3=0.03,
        displacement_stepsize_fc2=None,
        min_supercell_length=10.0,
        override=False,
        generate_scripts=None,
        submission_type=None,
        abacus_command=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _postprocess_namespace(job: Path, **overrides) -> Namespace:
    values = dict(
        job=job,
        version=None,
        mesh=[3, 3, 3],
        tmin=300.0,
        tmax=300.0,
        tstep=100.0,
        lbte=False,
        output="thermal_conductivity_results.json",
        plot="thermal_conductivity.png",
    )
    values.update(overrides)
    return Namespace(**values)


def _write_force_log(task: Path, natoms: int, *, converged: bool = True) -> None:
    """Write one converged ABACUS log holding zero forces."""
    lines = ["E_KohnSham = -1.000000 eV\n", "density error = 1e-9\n"]
    if converged:
        lines.append("charge density convergence is achieved\n")
    lines += [
        "#TOTAL-FORCE (eV/Angstrom)\n",
        "-" * 30 + "\n",
        "  Atoms  Force_x  Force_y  Force_z\n",
        "-" * 30 + "\n",
    ]
    lines += [f"  H{index + 1}  0.0000000  0.0000000  0.0000000\n" for index in range(natoms)]
    lines += ["-" * 30 + "\n", "Total  Time  : 0 h 0 mins 1 secs\n"]
    output = task / "OUT.ABACUS"
    output.mkdir(parents=True, exist_ok=True)
    (output / "running_scf.log").write_text("".join(lines), encoding="utf-8")


def test_temperature_grid() -> None:
    assert _temperatures(200.0, 400.0, 100.0) == [200.0, 300.0, 400.0]
    assert _temperatures(300.0, 300.0, 100.0) == [300.0]
    assert _temperatures(100.0, 350.0, 100.0) == [100.0, 200.0, 300.0]
    with pytest.raises(ValueError):
        _temperatures(-1.0, 300.0, 100.0)
    with pytest.raises(ValueError):
        _temperatures(400.0, 300.0, 100.0)
    with pytest.raises(ValueError):
        _temperatures(300.0, 300.0, 0.0)


def test_scale_kpoints() -> None:
    assert _scale_kpoints([9, 9, 9, 0, 0, 0], "gamma", [3, 3, 3]) == [3, 3, 3, 0, 0, 0]
    assert _scale_kpoints([9, 9, 9, 0, 0, 0], "MP", [1, 1, 1]) == [9, 9, 9, 0, 0, 0]
    assert _scale_kpoints([2, 2, 2, 0, 0, 0], "gamma", [4, 4, 4])[:3] == [1, 1, 1]
    line = [[0.0, 0.0, 0.0, 10], [0.5, 0.0, 0.0, 1]]
    assert _scale_kpoints(line, "line", [2, 2, 2]) == line


def test_mesh_setting() -> None:
    assert _mesh_setting([4]) == [4]
    assert _mesh_setting([4, 5, 6]) == [4, 5, 6]
    with pytest.raises(ValueError):
        _mesh_setting([0])
    with pytest.raises(ValueError):
        _mesh_setting([2, 2])


def test_displacement_tasks_keep_dataset_indices() -> None:
    tasks = _displacement_tasks([object(), None, object()], "fc3-")
    assert tasks == [
        {"task": "fc3-0000", "index": 0},
        {"task": "fc3-0002", "index": 2},
    ]


def test_kappa_array_and_components() -> None:
    kappa = np.arange(12, dtype=float).reshape(1, 2, 6)
    values = _kappa_array(kappa)
    assert values is not None
    assert values.shape == (2, 6)
    components = _conductivity_components(values)
    assert list(components) == ["xx", "yy", "zz", "yz", "xz", "xy"]
    assert components["xx"] == [0.0, 6.0]
    assert _kappa_array(np.zeros((2, 5))) is None


@requires_phono3py
def test_prepare_writes_manifest_and_jobs(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job)

    assert prepare(_prepare_namespace(job)) == 0

    manifest = json.loads((job / "workflow_thermal_conductivity.json").read_text())
    assert manifest["workflow"] == "thermal_conductivity"
    assert manifest["fc3_supercell"] == [2, 2, 2]
    assert manifest["fc2_supercell"] is None
    assert manifest["displacement_stepsize_fc3"] == pytest.approx(0.03)
    assert manifest["fc2_displacements"] == []
    assert manifest["fc3_supercell_count"] == 13
    assert len(manifest["tasks"]) == 13
    assert (job / "phono3py_disp.yaml").is_file()
    assert (job / "fc3-0000" / "STRU").is_file()
    assert "cal_force" in (job / "fc3-0000" / "INPUT").read_text()


@requires_phono3py
def test_prepare_with_independent_fc2_supercell(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job)

    assert prepare(_prepare_namespace(job, supercell_fc2=[1, 1, 1])) == 0

    manifest = json.loads((job / "workflow_thermal_conductivity.json").read_text())
    assert manifest["fc2_supercell"] == [1, 1, 1]
    assert manifest["fc2_supercell_count"] == 1
    assert len(manifest["fc2_displacements"]) == 1
    assert manifest["fc2_displacements"][0]["task"] == "fc2-0000"
    assert (job / "fc2-0000" / "STRU").is_file()
    assert len(manifest["tasks"]) == len(manifest["fc3_displacements"]) + 1


@requires_phono3py
def test_prepare_requires_override_and_consistent_stepsize(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job)
    prepare(_prepare_namespace(job))

    with pytest.raises(RuntimeError, match="use --override"):
        prepare(_prepare_namespace(job))
    assert prepare(_prepare_namespace(job, override=True)) == 0

    with pytest.raises(ValueError, match="supercell-fc2"):
        prepare(
            _prepare_namespace(
                job, override=True, displacement_stepsize_fc2=0.01
            )
        )


@requires_phono3py
def test_postprocess_solves_transport_equation(tmp_path: Path) -> None:
    from abacustools.io.stru import AbacusSTRU

    job = tmp_path / "job"
    _write_job(job)
    prepare(_prepare_namespace(job))
    manifest = json.loads((job / "workflow_thermal_conductivity.json").read_text())
    for name in manifest["tasks"]:
        structure = AbacusSTRU.read(str(job / name / "STRU"))
        _write_force_log(job / name, structure.natoms)

    assert postprocess(_postprocess_namespace(job)) == 0

    result = json.loads((job / "thermal_conductivity_results.json").read_text())
    assert result["method"] == "RTA"
    assert result["mesh"] == [3, 3, 3]
    assert result["temperatures"] == [300.0]
    assert result["fc3_calculations"] == len(manifest["tasks"])
    assert result["fc2_calculations"] == 0
    assert set(result["kappa_w_m_k"]) == {"xx", "yy", "zz", "yz", "xz", "xy"}
    assert len(result["kappa_w_m_k"]["xx"]) == 1
    assert len(result["kappa_components_w_m_k"]) == 1
    assert result["rta_kappa_w_m_k"] is None
    assert (job / "kappa-m333.hdf5").is_file()
    assert (job / "thermal_conductivity.png").is_file()
    assert result["kappa_hdf5"] == str(job / "kappa-m333.hdf5")


@requires_phono3py
def test_postprocess_rejects_unconverged_and_mismatched_jobs(tmp_path: Path) -> None:
    from abacustools.io.stru import AbacusSTRU

    job = tmp_path / "job"
    _write_job(job)
    prepare(_prepare_namespace(job))
    manifest = json.loads((job / "workflow_thermal_conductivity.json").read_text())
    for name in manifest["tasks"]:
        structure = AbacusSTRU.read(str(job / name / "STRU"))
        _write_force_log(job / name, structure.natoms, converged=False)

    with pytest.raises(RuntimeError, match="did not converge"):
        postprocess(_postprocess_namespace(job))

    for name in manifest["tasks"]:
        structure = AbacusSTRU.read(str(job / name / "STRU"))
        _write_force_log(job / name, structure.natoms)
    manifest["fc3_supercell"] = [3, 3, 3]
    (job / "workflow_thermal_conductivity.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="fc3 supercell"):
        postprocess(_postprocess_namespace(job))


@requires_phono3py
def test_postprocess_requires_prepared_dataset(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job)
    manifest = {
        "format": 1,
        "workflow": "thermal_conductivity",
        "tasks": ["fc3-0000"],
        "fc3_supercell": [1, 1, 1],
        "fc3_displacements": [{"task": "fc3-0000", "index": 0}],
    }
    (job / "workflow_thermal_conductivity.json").write_text(json.dumps(manifest))
    (job / "fc3-0000").mkdir()

    with pytest.raises(RuntimeError, match="run the prepare stage first"):
        postprocess(_postprocess_namespace(job))
