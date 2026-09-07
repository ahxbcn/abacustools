"""Tests for input-only ABACUS job checking."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from abacustools.commands.job.checkinput import run
from abacustools.core.job import check_input


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


def _job(tmp_path: Path, *, input_text: str) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(input_text, encoding="utf-8")
    (job / "STRU").write_text(STRU, encoding="utf-8")
    (job / "H.upf").write_text("pseudo", encoding="utf-8")
    return job


def test_check_input_reports_summary_without_reading_output(tmp_path: Path) -> None:
    job = _job(
        tmp_path,
        input_text=(
            "INPUT_PARAMETERS\n"
            "calculation scf\n"
            "basis_type pw\n"
            "nspin 1\n"
            "ecutwfc 60\n"
            "scf_thr 1e-7\n"
            "gamma_only 1\n"
        ),
    )
    output = job / "OUT.ABACUS"
    output.mkdir()
    (output / "running_scf.log").write_text("not an output", encoding="utf-8")

    report = check_input(job)

    assert report.valid
    assert report.summary["calculation"] == "scf"
    assert report.summary["structure"]["natoms"] == 1
    assert report.summary["structure"]["species"] == {"H": 1}
    assert report.summary["kpoints"] == {"mode": "gamma_only"}


def test_check_input_validates_kpt_and_input_values(tmp_path: Path) -> None:
    job = _job(
        tmp_path,
        input_text=(
            "INPUT_PARAMETERS\n"
            "calculation nonsense\n"
            "nspin 3\n"
            "ecutwfc 0\n"
            "scf_thr -1\n"
            "gamma_only 0\n"
            ""
        ),
    )
    (job / "KPT").write_text("K_POINTS\n0\nGamma\n0 2 2 0 0 0\n", encoding="utf-8")

    report = check_input(job)

    assert not report.valid
    assert {issue.code for issue in report.issues} >= {
        "invalid-calculation",
        "invalid-nspin",
        "invalid-ecutwfc",
        "invalid-scf_thr",
        "invalid-kpt",
    }


def test_check_input_accepts_three_component_kspacing_without_kpt(tmp_path: Path) -> None:
    job = _job(
        tmp_path,
        input_text=(
            "INPUT_PARAMETERS\n"
            "calculation scf\n"
            "kspacing 0.2 0.2 0.2\n"
        ),
    )

    report = check_input(job)

    assert report.valid
    assert report.summary["kpoints"] == {
        "mode": "kspacing",
        "value": [0.2, 0.2, 0.2],
    }


def test_checkinput_json_contains_summary(tmp_path: Path, capsys) -> None:
    job = _job(
        tmp_path,
        input_text="INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
    )

    assert run(Namespace(job=job, strict=False, json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["summary"]["calculation"] == "scf"


def test_checkinput_text_prints_task_information(tmp_path: Path, capsys) -> None:
    job = _job(
        tmp_path,
        input_text="INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
    )

    assert run(Namespace(job=job, strict=False, json=False)) == 0

    output = capsys.readouterr().out
    assert "calculation: scf" in output
    assert "structure: natoms=1, species=H:1" in output
    assert "kpoints: mode=gamma_only" in output
    assert "valid: yes" in output
