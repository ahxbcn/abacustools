"""Tests for reading and writing ABACUS KPT files."""

from __future__ import annotations

from pathlib import Path

import pytest

from abacustools.core.input_prep import InputPreparer
from abacustools.io.abacus import FormatKpt, ReadKpt, WriteKpt


STRU = """\
ATOMIC_SPECIES
Si 28.0855

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


def _source_and_library(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "silicon.stru"
    source.write_text(STRU, encoding="utf-8")
    library = tmp_path / "library"
    library.mkdir()
    (library / "Si.upf").write_text("pseudo", encoding="utf-8")
    (library / "Si.orb").write_text("orbital", encoding="utf-8")
    return source, library


def test_gamma_and_mp_meshes() -> None:
    assert FormatKpt([1, 1, 1], "gamma") == "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"
    assert (
        FormatKpt([4, 4, 4, 0.5, 0.5, 0.5], "Gamma")
        == "K_POINTS\n0\nGamma\n4 4 4 0.5 0.5 0.5\n"
    )
    assert FormatKpt([2, 2, 2, 1, 1, 1], "mp") == "K_POINTS\n0\nMP\n2 2 2 1 1 1\n"


def test_explicit_kpoints_and_cartesian_alias() -> None:
    direct = FormatKpt([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], "direct")
    assert direct.startswith("K_POINTS\n2\nDirect\n")
    assert "0.50000000000\n" in direct

    # The writer only understood the misspelling "cartessian" before.
    cartesian = FormatKpt([[0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 3.0]], "cartesian")
    assert cartesian.startswith("K_POINTS\n2\nCartesian\n")
    assert "0.25000000000\n" in cartesian
    assert FormatKpt([[0.0, 0.0, 0.0, 1.0]], "cartessian").startswith(
        "K_POINTS\n1\nCartesian\n"
    )


def test_line_kpoints_and_comments() -> None:
    line = FormatKpt([[0.0, 0.0, 0.0, 10, "G"], [0.5, 0.5, 0.0, 1, "//X"]], "line")
    assert line.startswith("K_POINTS\n2\nLine\n")
    assert "10 #G\n" in line
    assert "1 //X\n" in line
    assert FormatKpt([[0.0, 0.0, 0.0, 5], [0.5, 0.0, 0.0, 1]], "line_cartesian").startswith(
        "K_POINTS\n2\nLine_Cartesian\n"
    )


@pytest.mark.parametrize(
    ("kpt", "model", "message"),
    [
        ([1, 1], "gamma", "three or six values"),
        ([[1, 1, 1], [2, 2, 2]], "mp", "single mesh group"),
        ([[0.0, 0.0, 0.0]], "line", "at least two"),
        ([[0.0, 0.0, 0.0, 1.0, 2.0]], "direct", "optional weight"),
        ([[0.0, 0.0, 0.0, 0], [0.5, 0.0, 0.0, 1]], "line", "positive integers"),
        ([1, 1, 1], "bogus", "unsupported KPT model"),
    ],
)
def test_invalid_kpt_input_raises(kpt, model, message) -> None:
    with pytest.raises(ValueError, match=message):
        FormatKpt(kpt, model)


def test_write_kpt_leaves_the_caller_list_alone(tmp_path: Path) -> None:
    kpt = [2, 2, 2]
    WriteKpt(kpt, tmp_path / "KPT", "gamma")
    assert kpt == [2, 2, 2]


@pytest.mark.parametrize(
    ("kpt", "model"),
    [
        ([2, 2, 2, 0, 0, 0], "gamma"),
        ([[0.0, 0.0, 0.0, 0.5], [0.5, 0.0, 0.0, 0.5]], "direct"),
        ([[0.0, 0.0, 0.0, 0.25], [0.5, 0.0, 0.0, 0.75]], "cartesian"),
        ([[0.0, 0.0, 0.0, 10, "G"], [0.5, 0.0, 0.0, 1]], "line"),
        ([[0.0, 0.0, 0.0, 5], [0.5, 0.0, 0.0, 1]], "line_cartesian"),
    ],
)
def test_kpt_round_trip(tmp_path: Path, kpt, model) -> None:
    path = tmp_path / "KPT"
    WriteKpt(kpt, path, model)

    read_back, read_model = ReadKpt(str(path))
    rewritten = tmp_path / "KPT.again"
    WriteKpt(read_back, rewritten, read_model)

    assert read_model == model
    assert rewritten.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_prepare_writes_explicit_and_line_kpt(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    jobs = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        kpt=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        kpt_model="cartesian",
    ).run()

    content = (jobs[0].path / "KPT").read_text(encoding="utf-8")
    assert content.startswith("K_POINTS\n2\nCartesian\n")
    assert "0.50000000000\n" in content

    line_jobs = InputPreparer(
        source,
        output_dir=tmp_path / "line-jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        kpt=[[0.0, 0.0, 0.0, 10], [0.5, 0.5, 0.0, 1]],
        kpt_model="line",
    ).run()
    assert (line_jobs[0].path / "KPT").read_text(encoding="utf-8").startswith(
        "K_POINTS\n2\nLine\n"
    )


def test_prepare_rejects_invalid_kpt_before_writing(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    output = tmp_path / "jobs"

    with pytest.raises(ValueError, match="three or six values"):
        InputPreparer(
            source,
            output_dir=output,
            filetype="stru",
            basis="pw",
            pp_path=library,
            kpt=[1, 1],
        )

    assert not output.exists()
