"""Tests for ABACUS job preparation and diagnostics."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from abacustools.core.config import CONFIG
from abacustools.core.input_prep import InputPreparationError, InputPreparer
from abacustools.core.job import status_job, validate_job
from abacustools.io.abacus import IsEnabled, ReadInput
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


STRU_WITH_ORBITAL = """\
ATOMIC_SPECIES
H 1.0 H.upf

NUMERICAL_ORBITAL
H.orb

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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("T.", False),
        ("yes", True),
        ("no", False),
        ("0", False),
        ("2", True),
        ("", False),
        (None, False),
        ([0.0, 0.0, 0.1], True),
        ([0, 0], False),
    ],
)
def test_is_enabled(value, expected: bool) -> None:
    assert IsEnabled(value) is expected


def test_prepare_pw_job_drops_orbitals(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    source.write_text(STRU_WITH_ORBITAL, encoding="utf-8")

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        orb_path=library,
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H.upf").is_file()
    assert not (job / "H.orb").exists()
    assert "NUMERICAL_ORBITAL" not in (job / "STRU").read_text(encoding="utf-8")


def test_prepare_lcao_job_keeps_orbitals(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    source.write_text(STRU_WITH_ORBITAL, encoding="utf-8")

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=library,
        orb_path=library,
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H.orb").is_file()
    assert "NUMERICAL_ORBITAL" in (job / "STRU").read_text(encoding="utf-8")


def test_prepare_warns_about_the_default_kpt_mesh(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    with pytest.warns(UserWarning, match="no KPT file found"):
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
        ).run()[0].path

    assert (job / "KPT").read_text(encoding="utf-8") == "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"


def test_prepare_uses_kspacing_instead_of_a_kpt_file(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
            set_params={"kspacing": 0.1},
        ).run()[0].path

    assert not (job / "KPT").exists()


def test_prepare_writes_the_cell_relax_keyword(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        job_type="cell-relax",
        kpt=[1, 1, 1],
    ).run()[0].path

    assert ReadInput(job / "INPUT")["calculation"] == "cell-relax"


def test_prepare_rejects_a_broken_element_index(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    (library / "H.upf").unlink()
    (library / "element.json").write_text('{"H": "H.upf"}', encoding="utf-8")

    with pytest.raises(InputPreparationError, match="element.json"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
            kpt=[1, 1, 1],
        ).run()

    # The output directory itself is created first, but no job is written.
    assert not (tmp_path / "jobs" / "000000").exists()


def test_prepare_rejects_conflicting_basis_options(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    with pytest.raises(ValueError, match="conflicts with --set basis_type"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=library,
            orb_path=library,
            set_params={"basis_type": "pw"},
        )


def test_prepare_rejects_unknown_basis_type(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    with pytest.raises(ValueError, match="unsupported basis_type: bogus"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            pp_path=library,
            set_params={"basis_type": "bogus"},
        )


def test_prepare_basis_type_choice_uses_matching_solver(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        pp_path=library,
        kpt=[1, 1, 1],
        set_params={"basis_type": "pw"},
    ).run()[0].path

    inputs = ReadInput(job / "INPUT")
    assert inputs["basis_type"] == "pw"
    assert inputs["ks_solver"] == "dav_subspace"
    assert not (job / "H.orb").exists()


def test_prepare_rejects_a_template_with_another_basis(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    template = tmp_path / "INPUT.template"
    template.write_text("INPUT_PARAMETERS\nbasis_type pw\n", encoding="utf-8")

    with pytest.raises(InputPreparationError, match="use a single basis"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=library,
            orb_path=library,
            input_template=template,
        ).run()


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
