"""Tests for the cutoff-energy convergence workflow."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from abacustools.commands.workflow.ecutwfc import postprocess, prepare
from abacustools.io.abacus import ReadInput


STRU = """\
ATOMIC_SPECIES
H 1.0 H.upf

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
"""


def _base_job(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "suffix ABACUS\n"
        "ecutwfc 40\n"
        "kpoint_file KPT\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(STRU, encoding="utf-8")
    (job / "H.upf").write_text("pseudo", encoding="utf-8")
    (job / "KPT").write_text(
        "K_POINTS\n0\nGamma\n2 2 2 0 0 0\n", encoding="utf-8"
    )
    return job


def _write_scf_result(job: Path, energy: float) -> None:
    output = job / "OUT.ABACUS"
    output.mkdir()
    (output / "running_scf.log").write_text(
        f"E_KohnSham = {energy + 0.1:.8f} eV\n"
        "density error = 1.0e-5\n"
        f"E_KohnSham = {energy:.8f} eV\n"
        "density error = 1.0e-8\n"
        "charge density convergence is achieved\n"
        "Total  Time  : 1.0 s\n",
        encoding="utf-8",
    )


def _args(job: Path, values: list[float]) -> Namespace:
    return Namespace(job=job, values=values, override=False)


def test_prepare_writes_independent_scf_jobs(tmp_path: Path) -> None:
    job = _base_job(tmp_path)

    assert prepare(_args(job, [30.0, 50.0])) == 0

    first = ReadInput(job / "ecutwfc_01" / "INPUT")
    second = ReadInput(job / "ecutwfc_02" / "INPUT")
    assert first["calculation"] == "scf"
    assert first["ecutwfc"] == 30
    assert second["ecutwfc"] == 50
    assert (job / "ecutwfc_01" / "KPT").is_symlink()
    manifest = json.loads((job / "workflow_ecutwfc.json").read_text())
    assert manifest["tasks"] == ["ecutwfc_01", "ecutwfc_02"]


def test_postprocess_selects_first_converged_value(tmp_path: Path) -> None:
    job = _base_job(tmp_path)
    prepare(_args(job, [30.0, 40.0, 50.0]))
    _write_scf_result(job / "ecutwfc_01", -10.0)
    _write_scf_result(job / "ecutwfc_02", -10.1)
    _write_scf_result(job / "ecutwfc_03", -10.10005)

    result = postprocess(
        Namespace(
            job=job,
            version="",
            energy_tol=1e-4,
            output="ecut.json",
            plot="ecut.png",
        )
    )

    assert result == 0
    report = json.loads((job / "ecut.json").read_text())
    assert report["selected"]["parameter"] == 40.0
    assert report["all_tasks_complete"] is True
    assert (job / "ecut.png").stat().st_size > 0


def test_postprocess_does_not_abort_when_all_tasks_are_missing(tmp_path: Path) -> None:
    job = _base_job(tmp_path)
    prepare(_args(job, [30.0, 40.0]))

    assert postprocess(
        Namespace(
            job=job,
            version="",
            energy_tol=1e-4,
            output="empty.json",
            plot="empty.png",
        )
    ) == 0
    report = json.loads((job / "empty.json").read_text())
    assert report["selected"] is None
    assert not (job / "empty.png").exists()
