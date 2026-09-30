"""Tests for the ``file kpt`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abacustools.data.kpt import mesh_from_spacing, read_kpt
from abacustools.io.abacus import WriteKpt
from abacustools.io.stru import AbacusSTRU
from abacustools.main import main


STRU = """\
ATOMIC_SPECIES
Si 28.0855 Si.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
3 0 0
0 3 0
0 0 3

ATOMIC_POSITIONS
Cartesian

Si
0.0
1
0 0 0
"""


def _structure_file(tmp_path: Path) -> Path:
    path = tmp_path / "STRU"
    path.write_text(STRU, encoding="utf-8")
    return path


def _json_output(output: str) -> dict:
    """Return the JSON report a command printed after its banner."""
    return json.loads(output[output.index("{"):])


def test_inspect_reports_a_mesh_and_its_spacing(tmp_path: Path, capsys) -> None:
    structure_file = _structure_file(tmp_path)
    kpt = tmp_path / "KPT"
    WriteKpt([12, 12, 12, 0, 0, 0], str(kpt), "gamma")

    assert main(["file", "kpt", str(kpt), "--structure", str(structure_file), "--json"]) == 0
    report = _json_output(capsys.readouterr().out)

    assert report["model"] == "gamma"
    assert report["mesh"] == [12, 12, 12]
    assert report["shifts"] == [0.0, 0.0, 0.0]
    assert report["mesh_points"] == 1728
    assert report["valid"] is True

    structure = AbacusSTRU.read(str(structure_file))
    assert mesh_from_spacing(structure, report["kspacing_angstrom"]) == [12, 12, 12]


def test_inspect_reports_a_line_path(tmp_path: Path, capsys) -> None:
    kpt = tmp_path / "KPT"
    kpt.write_text(
        "K_POINTS\n2\nLine\n0 0 0 4 # G\n0.5 0 0 1 # X\n", encoding="utf-8"
    )

    assert main(["file", "kpt", str(kpt), "--json"]) == 0
    report = _json_output(capsys.readouterr().out)

    assert report["model"] == "line"
    assert report["nodes"] == 2
    assert report["point_counts"] == [4, 1]
    assert report["labels"] == ["G", "X"]


def test_inspect_flags_an_invalid_mesh(tmp_path: Path, capsys) -> None:
    kpt = tmp_path / "KPT"
    kpt.write_text("K_POINTS\n0\nGamma\n0 0 0 0 0 0\n", encoding="utf-8")

    assert main(["file", "kpt", str(kpt), "--json"]) == 0
    report = _json_output(capsys.readouterr().out)
    assert report["valid"] is False
    assert "positive integers" in report["error"]

    capsys.readouterr()
    assert main(["file", "kpt", str(kpt)]) == 0
    assert "warning:" in capsys.readouterr().out


def test_write_a_mesh_from_a_spacing(tmp_path: Path) -> None:
    structure_file = _structure_file(tmp_path)
    output = tmp_path / "KPT"

    assert main([
        "file", "kpt", "--structure", str(structure_file),
        "--spacing", "0.2", "-o", str(output),
    ]) == 0

    values, model = read_kpt(output)
    assert model == "gamma"
    structure = AbacusSTRU.read(str(structure_file))
    assert [int(value) for value in values[:3]] == mesh_from_spacing(structure, 0.2)


def test_write_a_band_path(tmp_path: Path, capsys) -> None:
    structure_file = _structure_file(tmp_path)
    output = tmp_path / "KPT"

    assert main([
        "file", "kpt", "--structure", str(structure_file),
        "--path", "-o", str(output), "--npoints", "4", "--json",
    ]) == 0
    report = _json_output(capsys.readouterr().out)

    assert report["model"] == "line"
    assert report["npoints"] == 4
    assert report["labels"][0] == "G"

    values, model = read_kpt(output)
    assert model == "line"
    assert len(values) == report["nodes"]
    assert [int(node[3]) for node in values[:-1]] == [4] * (len(values) - 1)
    assert int(values[-1][3]) == 1


def test_generation_refuses_to_overwrite(tmp_path: Path) -> None:
    structure_file = _structure_file(tmp_path)
    output = tmp_path / "KPT"
    output.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists"):
        main([
            "file", "kpt", "--structure", str(structure_file),
            "--mesh", "4", "4", "4", "-o", str(output),
        ])
    assert output.read_text(encoding="utf-8") == "keep me"

    assert main([
        "file", "kpt", "--structure", str(structure_file),
        "--mesh", "4", "4", "4", "-o", str(output), "--override",
    ]) == 0
    values, _ = read_kpt(output)
    assert [int(value) for value in values[:3]] == [4, 4, 4]
