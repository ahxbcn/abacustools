"""Tests for the ``workflow band`` workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from abacustools.data.kpt import read_kpt
from abacustools.io.abacus import ReadInput, WriteKpt
from abacustools.main import main


INPUT = """\
INPUT_PARAMETERS
calculation scf
suffix ABACUS
basis_type pw
ecutwfc 50
scf_thr 1e-7
gamma_only 0
kpoint_file KPT
"""


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


SLAB_STRU = """\
ATOMIC_SPECIES
Si 28.0855 Si.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
6 0 0
-3 5.196152423 0
0 0 30

ATOMIC_POSITIONS
Cartesian

Si
0.0
1
0 0 0
"""


def _reference_job(tmp_path: Path, stru: str = STRU) -> Path:
    """Return a reference SCF directory the workflow can prepare from."""
    job = tmp_path / "reference"
    job.mkdir()
    (job / "INPUT").write_text(INPUT, encoding="utf-8")
    (job / "STRU").write_text(stru, encoding="utf-8")
    (job / "Si.upf").write_text("pseudo", encoding="utf-8")
    WriteKpt([4, 4, 4, 0, 0, 0], str(job / "KPT"), "gamma")
    return job


def _json_output(output: str) -> dict:
    """Return the JSON report a command printed after its banner."""
    return json.loads(output[output.index("{"):])


def test_prepare_writes_the_scf_and_nscf_jobs(tmp_path: Path) -> None:
    job = _reference_job(tmp_path)

    assert main(["workflow", "band", "prepare", "-j", str(job), "--npoints", "5"]) == 0

    scf = job / "band_scf"
    nscf = job / "band_nscf"
    manifest = json.loads((job / "workflow_band.json").read_text(encoding="utf-8"))

    assert manifest["tasks"] == ["band_scf", "band_nscf"]
    assert manifest["directory"] == "band_nscf"
    assert manifest["scf_directory"] == "band_scf"
    assert manifest["npoints"] == 5
    assert manifest["scf_mesh"] == [4, 4, 4]
    assert manifest["dimensionality"] == "bulk"
    assert manifest["dimensionality_label"] == "3D bulk"
    assert manifest["path_method"] == "seekpath"

    scf_inputs = ReadInput(scf / "INPUT")
    assert scf_inputs["calculation"] == "scf"
    assert scf_inputs["suffix"] == "scf"
    assert str(scf_inputs["out_chg"]) == "1"
    assert str(scf_inputs["gamma_only"]) == "0"
    values, model = read_kpt(scf / "KPT")
    assert model == "gamma"
    assert [int(value) for value in values[:3]] == [4, 4, 4]

    nscf_inputs = ReadInput(nscf / "INPUT")
    assert nscf_inputs["calculation"] == "nscf"
    assert nscf_inputs["suffix"] == "nscf"
    assert nscf_inputs["init_chg"] == "file"
    assert nscf_inputs["read_file_dir"] == "../band_scf/OUT.scf"
    assert str(nscf_inputs["out_band"]) == "1"
    assert "out_chg" not in nscf_inputs
    nodes, model = read_kpt(nscf / "KPT")
    assert model == "line"
    assert len(nodes) == len(manifest["labels"])
    assert [int(node[3]) for node in nodes[:-1]] == [5] * (len(nodes) - 1)
    assert int(nodes[-1][3]) == 1

    assert (scf / "Si.upf").is_symlink()
    assert (nscf / "Si.upf").is_symlink()
    assert (scf / "STRU").is_file()
    assert (nscf / "STRU").is_file()


def test_prepare_uses_a_two_dimensional_path(tmp_path: Path, capsys) -> None:
    job = _reference_job(tmp_path, stru=SLAB_STRU)

    assert main(["workflow", "band", "prepare", "-j", str(job), "--npoints", "3"]) == 0
    stdout = capsys.readouterr().out
    assert "2D slab" in stdout

    manifest = json.loads((job / "workflow_band.json").read_text(encoding="utf-8"))
    assert manifest["dimensionality"] == "slab"
    assert manifest["dimensionality_label"] == "2D slab"
    assert manifest["path_method"] == "2D hexagonal"
    assert manifest["labels"] == ["G", "M", "K", "G"]
    assert manifest["periodic_directions"] == ["a", "b"]

    nodes, model = read_kpt(job / "band_nscf" / "KPT")
    assert model == "line"
    assert [int(node[3]) for node in nodes[:-1]] == [3, 3, 3]
    # The vacuum runs along c, so the path stays in the a-b plane.
    assert all(abs(float(node[2])) < 1e-12 for node in nodes)


def test_prepare_sets_nbands_and_refuses_to_overwrite(tmp_path: Path) -> None:
    job = _reference_job(tmp_path)
    assert main([
        "workflow", "band", "prepare", "-j", str(job), "--bands", "12",
    ]) == 0
    assert str(ReadInput(job / "band_nscf" / "INPUT")["nbands"]) == "12"

    with pytest.raises(RuntimeError, match="already exist"):
        main(["workflow", "band", "prepare", "-j", str(job)])

    assert main(["workflow", "band", "prepare", "-j", str(job), "--override"]) == 0


def _write_band_output(job: Path) -> int:
    """Fill the prepared NSCF directory with a synthetic band result."""
    nscf = job / "band_nscf"
    nodes, _ = read_kpt(nscf / "KPT")
    total = sum(int(node[3]) for node in nodes)
    output = nscf / "OUT.nscf"
    output.mkdir()
    rows = []
    for index in range(total):
        fraction = index / (total - 1)
        # A valence band peaking in the middle and a conduction band with its
        # minimum at the ends, as in the postprocess band tests.
        shape = 1.0 - abs(2.0 * fraction - 1.0)
        rows.append([index + 1, fraction, -1.0 + 0.5 * shape, 1.0 + 0.5 * shape])
    np.savetxt(output / "BANDS_1.dat", np.asarray(rows, dtype=float))
    return total


def test_postprocess_reports_the_band_gap(tmp_path: Path, capsys) -> None:
    job = _reference_job(tmp_path)
    assert main(["workflow", "band", "prepare", "-j", str(job), "--npoints", "2"]) == 0
    total = _write_band_output(job)
    capsys.readouterr()

    assert main([
        "workflow", "band", "postprocess", "-j", str(job), "--efermi", "0.0", "--json",
    ]) == 0
    report = _json_output(capsys.readouterr().out)

    assert report["workflow"] == "band"
    assert report["nkpts"] == total
    assert report["nbands"] == 2
    assert report["energy_unit"] == "eV"
    assert report["npoints"] == 2
    assert report["dimensionality"] == "bulk"
    assert report["path_method"] == "seekpath"
    gap = report["band_gap"]
    assert gap["is_metal"] is False
    assert gap["band_gap"] == pytest.approx(1.5)
    assert gap["vbm"]["energy"] == pytest.approx(-0.5)
    assert gap["cbm"]["energy"] == pytest.approx(1.0)

    results = job / "band_results.json"
    assert results.is_file()
    stored = json.loads(results.read_text(encoding="utf-8"))
    assert stored["band_gap"]["band_gap"] == pytest.approx(1.5)
    assert stored["results"] == str(results)
    assert (job / "band.png").is_file()


def test_postprocess_needs_a_finished_run(tmp_path: Path) -> None:
    job = _reference_job(tmp_path)
    assert main(["workflow", "band", "prepare", "-j", str(job)]) == 0

    with pytest.raises(RuntimeError, match="no ABACUS output"):
        main(["workflow", "band", "postprocess", "-j", str(job)])
