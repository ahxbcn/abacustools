"""Tests for ABACUS job preparation and diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from abacustools.core.config import CONFIG
from abacustools.core.input_prep import InputPreparationError, InputPreparer
from abacustools.core.job import status_job, validate_job
from abacustools.io.abacus import ReadInput
from abacustools.main import main


STRU = """\
ATOMIC_SPECIES
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
"""


def _source_and_library(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "water.stru"
    source.write_text(STRU, encoding="utf-8")
    library = tmp_path / "library"
    library.mkdir()
    (library / "H.upf").write_text("pseudo", encoding="utf-8")
    (library / "H.orb").write_text("orbital", encoding="utf-8")
    return source, library


def test_prepare_writes_complete_lcao_job(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    jobs = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=library,
        orb_path=library,
        kpt=[2, 2, 2],
        copy_resources=True,
        folder_syntax="{x[:-5]}",
    ).run()

    job = jobs[0].path
    assert job.name == "water"
    assert (job / "INPUT").is_file()
    assert (job / "STRU").is_file()
    assert (job / "KPT").is_file()
    assert (job / "H.upf").read_text(encoding="utf-8") == "pseudo"
    assert (job / "H.orb").read_text(encoding="utf-8") == "orbital"
    assert ReadInput(job / "INPUT")["basis_type"] == "lcao"
    assert ReadInput(job / "INPUT")["ks_solver"] == "genelpa"
    assert validate_job(job).valid
    assert status_job(job).state == "ready"


def test_prepare_preserves_template_basis_parameters(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    template = tmp_path / "INPUT.template"
    template.write_text(
        "INPUT_PARAMETERS\nbasis_type lcao\nks_solver custom_solver\n",
        encoding="utf-8",
    )
    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        pp_path=library,
        orb_path=library,
        input_template=template,
        kpt=[1, 1, 1],
        copy_resources=True,
    ).run()[0].path

    assert ReadInput(job / "INPUT")["ks_solver"] == "custom_solver"


def test_validate_reports_unknown_keyword_and_missing_resource(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nnot_a_keyword 1\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(STRU, encoding="utf-8")

    report = validate_job(job)
    assert not report.valid
    assert {issue.code for issue in report.issues} == {
        "unknown-input",
        "missing-pseudopotential",
    }
    assert validate_job(job, strict=True).valid is False


def test_status_reads_scf_progress_and_convergence(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        kpt=[1, 1, 1],
        copy_resources=True,
    ).run()[0].path
    output = job / "OUT.ABACUS"
    output.mkdir()
    (output / "running_scf.log").write_text(
        "E_KohnSham = -1.000000\n"
        "density error = 1.0E-4\n"
        "E_KohnSham = -1.100000\n"
        "density error = 1.0E-8\n"
        "charge density convergence is achieved\n",
        encoding="utf-8",
    )

    status = status_job(job)
    assert status.state == "converged"
    assert status.progress["scf_steps"] == 2
    assert status.progress["denergy"] == pytest.approx(-0.1)


def test_prepare_uses_configured_default_library(tmp_path: Path, monkeypatch) -> None:
    source, library = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": library, "orb": library}}},
    )

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        kpt=[1, 1, 1],
        copy_resources=True,
    ).run()[0].path

    assert (job / "H.upf").is_file()
    assert (job / "H.orb").is_file()


def test_prepare_selects_requested_configured_library(tmp_path: Path, monkeypatch) -> None:
    source, library = _source_and_library(tmp_path)
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "H.upf").write_text("selected pseudo", encoding="utf-8")
    (selected / "H.orb").write_text("selected orbital", encoding="utf-8")
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {
            "default": "test",
            "libraries": {
                "test": {"pp": library, "orb": library},
                "selected": {"pp": selected, "orb": selected},
            },
        },
    )

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        library="selected",
        kpt=[1, 1, 1],
        copy_resources=True,
    ).run()[0].path

    assert (job / "H.upf").read_text(encoding="utf-8") == "selected pseudo"
    assert (job / "H.orb").read_text(encoding="utf-8") == "selected orbital"


def test_prepare_rejects_unknown_library(tmp_path: Path, monkeypatch) -> None:
    source, _ = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": None, "orb": None}}},
    )

    with pytest.raises(ValueError, match="unsupported resource library"):
        InputPreparer(source, library="missing")


def test_prepare_reports_unconfigured_library_path(tmp_path: Path, monkeypatch) -> None:
    source, _ = _source_and_library(tmp_path)
    monkeypatch.delenv("ABACUS_PP_PATH", raising=False)
    monkeypatch.delenv("ABACUS_ORB_PATH", raising=False)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": None, "orb": None}}},
    )

    with pytest.raises(InputPreparationError, match=r"resources\.libraries\.test\.pp"):
        InputPreparer(source, filetype="stru", basis="pw", kpt=[1, 1, 1]).run()


def test_explicit_library_does_not_use_legacy_environment_paths(
    tmp_path: Path, monkeypatch
) -> None:
    source, library = _source_and_library(tmp_path)
    monkeypatch.setenv("ABACUS_PP_PATH", str(library))
    monkeypatch.setenv("ABACUS_ORB_PATH", str(library))
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": None, "orb": None}}},
    )

    with pytest.raises(InputPreparationError, match=r"resources\.libraries\.test\.pp"):
        InputPreparer(
            source,
            filetype="stru",
            basis="pw",
            library="test",
            kpt=[1, 1, 1],
        ).run()


def test_job_prepare_help_does_not_expose_resource_paths(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["job", "prepare", "--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--library" in output
    assert "--pp" not in output
    assert "--orb" not in output
