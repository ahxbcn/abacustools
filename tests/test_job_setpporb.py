"""Tests for switching a prepared job's pseudopotential/orbital library."""

from __future__ import annotations

from pathlib import Path

import pytest

from abacustools.core.config import CONFIG
from abacustools.core.input_prep import InputPreparationError
from abacustools.core.resource_switch import read_job_basis, switch_job_library
from abacustools.io.stru import AbacusSTRU
from abacustools.main import main


STRU = """\
ATOMIC_SPECIES
Si 28.0855 Si.upf

NUMERICAL_ORBITAL
Si.orb

LATTICE_CONSTANT
1.8897259886

LATTICE_VECTORS
  0.5000000000   0.5000000000   0.0000000000
  0.0000000000   0.5000000000   0.5000000000
  0.5000000000   0.0000000000   0.5000000000

ATOMIC_POSITIONS
Direct

Si
0.0
2
  0.0000000000   0.0000000000   0.0000000000 m 1 1 1
  0.2500000000   0.2500000000   0.2500000000 m 1 1 1
"""

INPUT_LCAO = "INPUT_PARAMETERS\nbasis_type lcao\necutwfc 100\n"
INPUT_PW = "INPUT_PARAMETERS\nbasis_type pw\necutwfc 60\n"


def _library(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    return root


def _configure(monkeypatch, libraries: dict[str, Path], default: str = "a") -> None:
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {
            "default": default,
            "libraries": {
                name: {"pp": str(path), "orb": str(path)} for name, path in libraries.items()
            },
        },
    )


def _job(tmp_path: Path, *, input_text: str = INPUT_LCAO, stru: str = STRU) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    (job / "STRU").write_text(stru, encoding="utf-8")
    if input_text is not None:
        (job / "INPUT").write_text(input_text, encoding="utf-8")
    (job / "Si.upf").write_text("old-pseudo", encoding="utf-8")
    (job / "Si.orb").write_text("old-orbital", encoding="utf-8")
    return job


def test_switch_replaces_stru_and_files(tmp_path: Path, monkeypatch) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    lib_b = _library(
        tmp_path / "b",
        {"Si_ONCV.upf": "b-pseudo", "Si_gga_7au_100Ry_2s2p1d.orb": "b-orbital"},
    )
    _configure(monkeypatch, {"a": lib_a, "b": lib_b})
    job = _job(tmp_path)

    result = switch_job_library(job, library="b", copy_resources=True)

    text = (job / "STRU").read_text(encoding="utf-8")
    assert "Si_ONCV.upf" in text
    assert "Si_gga_7au_100Ry_2s2p1d.orb" in text
    assert (job / "Si_ONCV.upf").read_text(encoding="utf-8") == "b-pseudo"
    assert (job / "Si_gga_7au_100Ry_2s2p1d.orb").read_text(encoding="utf-8") == "b-orbital"
    assert not (job / "Si.upf").exists()
    assert not (job / "Si.orb").exists()
    assert result.pseudopotentials == {"Si": "Si_ONCV.upf"}
    assert result.orbitals == {"Si": "Si_gga_7au_100Ry_2s2p1d.orb"}
    assert result.removed == ["Si.orb", "Si.upf"]
    assert result.basis == "lcao"


def test_files_are_copied_when_the_job_used_copies(tmp_path: Path, monkeypatch) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    lib_b = _library(
        tmp_path / "b",
        {"Si_ONCV.upf": "b-pseudo", "Si_gga_7au_100Ry_2s2p1d.orb": "b-orbital"},
    )
    _configure(monkeypatch, {"a": lib_a, "b": lib_b})
    job = _job(tmp_path)

    switch_job_library(job, library="b")

    assert not (job / "Si_ONCV.upf").is_symlink()
    assert (job / "Si_ONCV.upf").read_text(encoding="utf-8") == "b-pseudo"


def test_files_are_symlinked_when_the_job_used_symlinks(tmp_path: Path, monkeypatch) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    lib_b = _library(
        tmp_path / "b",
        {"Si_ONCV.upf": "b-pseudo", "Si_gga_7au_100Ry_2s2p1d.orb": "b-orbital"},
    )
    _configure(monkeypatch, {"a": lib_a, "b": lib_b})
    job = _job(tmp_path)
    for name in ("Si.upf", "Si.orb"):
        (job / name).unlink()
        (job / name).symlink_to(lib_a / name)

    switch_job_library(job, library="b")

    assert (job / "Si_ONCV.upf").is_symlink()
    assert (job / "Si_ONCV.upf").resolve() == (lib_b / "Si_ONCV.upf").resolve()


def test_variant_selects_the_requested_orbital_set(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "sg15"
    (library / "Si_SZ").mkdir(parents=True)
    (library / "Si_DZP").mkdir(parents=True)
    (library / "Si_SZ" / "Si_gga_7au_100Ry_1s1p.orb").write_text("sz", encoding="utf-8")
    (library / "Si_DZP" / "Si_gga_7au_100Ry_2s2p1d.orb").write_text("dzp", encoding="utf-8")
    (library / "Si.upf").write_text("pseudo", encoding="utf-8")
    _configure(monkeypatch, {"sg15": library, "a": library}, default="sg15")
    job = _job(tmp_path)

    result = switch_job_library(job, library="sg15", variant="SZ", copy_resources=True)

    assert result.orbitals == {"Si": "Si_gga_7au_100Ry_1s1p.orb"}
    assert "Si_gga_7au_100Ry_1s1p.orb" in (job / "STRU").read_text(encoding="utf-8")


def test_plane_wave_job_switches_only_pseudopotentials(tmp_path: Path, monkeypatch) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    lib_b = _library(tmp_path / "b", {"Si_ONCV.upf": "b-pseudo"})
    _configure(monkeypatch, {"a": lib_a, "b": lib_b})
    job = _job(tmp_path, input_text=INPUT_PW)

    result = switch_job_library(job, library="b", copy_resources=True)

    text = (job / "STRU").read_text(encoding="utf-8")
    assert result.basis == "pw"
    assert result.orbitals == {}
    assert "Si_ONCV.upf" in text
    # The leftover orbital reference is not this command's concern.
    assert "Si.orb" in text
    assert (job / "Si.orb").read_text(encoding="utf-8") == "old-orbital"


def test_dry_run_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    lib_b = _library(
        tmp_path / "b",
        {"Si_ONCV.upf": "b-pseudo", "Si_gga_7au_100Ry_2s2p1d.orb": "b-orbital"},
    )
    _configure(monkeypatch, {"a": lib_a, "b": lib_b})
    job = _job(tmp_path)

    result = switch_job_library(job, library="b", dry_run=True)

    assert result.dry_run
    assert result.pseudopotentials == {"Si": "Si_ONCV.upf"}
    assert (job / "STRU").read_text(encoding="utf-8") == STRU
    assert (job / "Si.upf").read_text(encoding="utf-8") == "old-pseudo"


def test_missing_element_leaves_the_job_untouched(tmp_path: Path, monkeypatch) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    empty = tmp_path / "empty"
    empty.mkdir()
    _configure(monkeypatch, {"a": lib_a, "empty": empty})
    job = _job(tmp_path)

    with pytest.raises(InputPreparationError, match="no pp for Si"):
        switch_job_library(job, library="empty", copy_resources=True)

    assert (job / "STRU").read_text(encoding="utf-8") == STRU
    assert (job / "Si.upf").read_text(encoding="utf-8") == "old-pseudo"


def test_warns_when_ecutwfc_is_below_the_new_orbital_cutoff(tmp_path: Path, monkeypatch) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    lib_b = _library(
        tmp_path / "b",
        {"Si_ONCV.upf": "b-pseudo", "Si_gga_7au_300Ry_4s2p2d1f.orb": "b-orbital"},
    )
    _configure(monkeypatch, {"a": lib_a, "b": lib_b})
    job = _job(tmp_path)

    with pytest.warns(UserWarning, match="below the 300 Ry cutoff"):
        switch_job_library(job, library="b", copy_resources=True)


def test_read_job_basis_prefers_input_over_leftover_orbitals(tmp_path: Path) -> None:
    job = _job(tmp_path, input_text=INPUT_PW, stru=STRU)
    structure = AbacusSTRU.read(str(job / "STRU"), fmt="stru")

    assert read_job_basis(job, structure) == "pw"


def test_cli_switches_a_job(tmp_path: Path, monkeypatch, capsys) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    lib_b = _library(
        tmp_path / "b",
        {"Si_ONCV.upf": "b-pseudo", "Si_gga_7au_100Ry_2s2p1d.orb": "b-orbital"},
    )
    _configure(monkeypatch, {"a": lib_a, "b": lib_b})
    job = _job(tmp_path)

    status = main(["job", "setpporb", str(job), "--library", "b", "--copy-resources"])

    capsys.readouterr()
    assert status == 0
    assert "Si_ONCV.upf" in (job / "STRU").read_text(encoding="utf-8")


def test_cli_reports_a_job_without_stru(tmp_path: Path, monkeypatch, capsys) -> None:
    lib = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    _configure(monkeypatch, {"a": lib})
    empty = tmp_path / "empty"
    empty.mkdir()

    status = main(["job", "setpporb", str(empty)])

    assert status == 1
    assert "no STRU file" in capsys.readouterr().out


def test_honours_stru_file_named_by_input(tmp_path: Path, monkeypatch) -> None:
    lib_a = _library(tmp_path / "a", {"Si.upf": "a-pseudo", "Si.orb": "a-orbital"})
    lib_b = _library(
        tmp_path / "b",
        {"Si_ONCV.upf": "b-pseudo", "Si_gga_7au_100Ry_2s2p1d.orb": "b-orbital"},
    )
    _configure(monkeypatch, {"a": lib_a, "b": lib_b})
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nbasis_type lcao\nstru_file STRU_relaxed\necutwfc 100\n",
        encoding="utf-8",
    )
    (job / "STRU_relaxed").write_text(STRU, encoding="utf-8")
    (job / "Si.upf").write_text("old-pseudo", encoding="utf-8")
    (job / "Si.orb").write_text("old-orbital", encoding="utf-8")

    switch_job_library(job, library="b", copy_resources=True)

    assert "Si_ONCV.upf" in (job / "STRU_relaxed").read_text(encoding="utf-8")
    assert not (job / "STRU").exists()
