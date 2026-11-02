"""Tests for the machine-readable develop artefacts (abacus.json, INPUT.info)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abacustools.data import versions
from abacustools.data.abacus_json import latest_ionic_output, read_abacus_json
from abacustools.data.abacus_result import (
    _calculation,
    get_result_from_job,
)


_SUMMARY = {
    "comment": "Unless otherwise specified, the unit of energy is eV and the "
    "unit of length is Angstrom",
    "init": {"natom": 42, "nband": 247},
    "output": [
        {"e_fermi": 0.10, "energy": -1.0, "force": [[0.0, 0.0, 0.0]]},
        {"e_fermi": 0.75, "energy": -1.5, "force": [[0.0, 0.0, 0.1]]},
    ],
    "general_info": {"version": "v3.12.0"},
}


def test_read_abacus_json_of_absent_file_is_none(tmp_path: Path) -> None:
    assert read_abacus_json(tmp_path) is None


def test_latest_ionic_output_forgives_partial_summaries(tmp_path: Path) -> None:
    assert latest_ionic_output(None) is None
    assert latest_ionic_output({"output": []}) is None
    assert latest_ionic_output({"output": {"step": 1}}) is None
    (tmp_path / "abacus.json").write_text(
        json.dumps(_SUMMARY, indent=4), encoding="utf-8"
    )
    summary = read_abacus_json(tmp_path)
    assert latest_ionic_output(summary)["e_fermi"] == 0.75


def test_read_abacus_json_rejects_broken_files(tmp_path: Path) -> None:
    (tmp_path / "abacus.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="abacus.json"):
        read_abacus_json(tmp_path)

    (tmp_path / "abacus.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_abacus_json(tmp_path)


def _write_json_job(job: Path, summary: dict = _SUMMARY) -> Path:
    job.mkdir(exist_ok=True)
    (job / "abacus.json").write_text(
        json.dumps(summary, indent=4), encoding="utf-8"
    )
    return job


def test_detect_version_prefers_abacus_json(tmp_path: Path) -> None:
    job = _write_json_job(tmp_path / "job")
    out = job / "OUT.ABACUS"
    out.mkdir()
    (out / "running_scf.log").write_text(
        "                              ABACUS v3.10.2\n", encoding="utf-8"
    )

    profile = versions.detect_version(job)
    assert profile is versions.resolve_version("develop")

    # without the summary the same job falls back to its log banner
    (job / "abacus.json").unlink()
    assert versions.detect_version(job) is versions.resolve_version("3.10")


def test_detect_version_skips_json_without_version(tmp_path: Path) -> None:
    _write_json_job(tmp_path / "job", {"general_info": {}})
    assert versions.detect_version(tmp_path / "job") is None


def _scf_job(job: Path, log: str) -> Path:
    job.mkdir(exist_ok=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\n", encoding="utf-8"
    )
    out = job / "OUT.ABACUS"
    out.mkdir(exist_ok=True)
    (out / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\n", encoding="utf-8"
    )
    (out / "running_scf.log").write_text(log, encoding="utf-8")
    return job


_LOG_WITH_EFERMI = """\
                              ABACUS v3.11.0-beta9
  E_Fermi        0.7913484839         10.7668484843
"""


def test_abacus_json_efermi_wins_over_log(tmp_path: Path) -> None:
    job = _scf_job(_write_json_job(tmp_path / "job"), _LOG_WITH_EFERMI)
    assert get_result_from_job(job, ["efermi"])["efermi"] == 0.75


def test_empty_summary_falls_back_to_log_efermi(tmp_path: Path) -> None:
    job = _scf_job(
        _write_json_job(tmp_path / "job", {"output": []}), _LOG_WITH_EFERMI
    )
    assert get_result_from_job(job, ["efermi"])["efermi"] == pytest.approx(
        10.7668484843
    )


def test_calculation_reads_develop_effective_input(tmp_path: Path) -> None:
    out = tmp_path / "OUT.ABACUS"
    out.mkdir()
    (out / "INPUT.info").write_text(
        "INPUT_PARAMETERS\ncalculation   nscf #electronic only\n",
        encoding="utf-8",
    )
    assert _calculation({"calculation": "relax"}, out) == "nscf"


def test_calculation_keeps_lts_effective_input(tmp_path: Path) -> None:
    out = tmp_path / "OUT.ABACUS"
    out.mkdir()
    (out / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation   cell-relax\n", encoding="utf-8"
    )
    assert _calculation({"calculation": "scf"}, out) == "cell-relax"
