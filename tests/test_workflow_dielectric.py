"""Tests for the electronic dielectric tensor workflow.

The pyatb call itself is replaced by a stub, because it needs an MPI runtime
and a set of ABACUS matrices; everything around it, which is the part this
package owns, runs for real.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.workflow import dielectric as dielectric_command
from abacustools.commands.workflow.dielectric import postprocess, prepare
from abacustools.data.dielectric import dielectric_summary
from abacustools.io.abacus import ReadInput


_STRU = """ATOMIC_SPECIES
Na 22.9898
Cl 35.45

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
0.0 2.8514618194 2.8514618194
2.8514618194 0.0 2.8514618194
2.8514618194 2.8514618194 0.0

ATOMIC_POSITIONS
Direct

Na
0.0
1
0.0 0.0 0.0

Cl
0.0
1
0.5 0.5 0.5
"""

_MATRICES = (
    "data-HR-sparse_SPIN0.csr",
    "data-SR-sparse_SPIN0.csr",
    "data-rR-sparse.csr",
)


def _source_job(tmp_path: Path) -> Path:
    """Write the converged SCF a dielectric calculation is prepared from."""
    job = tmp_path / "job"
    (job / "OUT.ABACUS").mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        "gamma_only 0\n"
        "kspacing 0.14\n"
        "nspin 1\n"
        "suffix ABACUS\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(_STRU, encoding="utf-8")
    return job


def _finish_task(task: Path) -> Path:
    """Add the matrices and the running log a finished task presents."""
    output = task / "OUT.ABACUS"
    output.mkdir(parents=True, exist_ok=True)
    for name in _MATRICES:
        (output / name).write_text("Matrix Dimension of H(R): 1\n", encoding="utf-8")
    (output / "running_scf.log").write_text(
        "AUTOSET number of electrons:  = 16\n"
        "        occupied bands = 8\n"
        " EFERMI = 2.8588620907 eV\n",
        encoding="utf-8",
    )
    return task


def _prepare_args(job: Path, **overrides) -> Namespace:
    """Return the preparation arguments of the dielectric workflow."""
    values = dict(job=job, name="dielectric", override=False)
    values.update(overrides)
    return Namespace(**values)


def _postprocess_args(job: Path, **overrides) -> Namespace:
    """Return the postprocessing arguments of the dielectric workflow."""
    values = dict(
        job=job,
        task=None,
        workdir="pyatb",
        grid=[50, 50, 50],
        omega=[0.0, 80.0],
        domega=0.5,
        eta=0.2,
        max_kpoint_num=8000,
        output="dielectric_results.json",
        pyatb_command=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _stub_pyatb(monkeypatch, tensor: np.ndarray) -> dict:
    """Replace the pyatb call and record the arguments it was given."""
    calls: dict = {}

    def fake(workdir, structure, **kwargs):
        calls["workdir"] = Path(workdir)
        calls["structure"] = structure
        calls.update(kwargs)
        summary = dielectric_summary(tensor)
        summary.update({"grid": list(kwargs["grid"]), "pyatb_version": "1.1.2"})
        return summary

    monkeypatch.setattr(dielectric_command, "dielectric_tensor", fake)
    return calls


def test_prepare_switches_the_matrix_output_on(tmp_path: Path) -> None:
    job = _source_job(tmp_path)

    assert prepare(_prepare_args(job)) == 0

    written = ReadInput(str(job / "dielectric" / "INPUT"))
    assert int(written["out_mat_hs2"]) == 1
    assert int(written["out_mat_r"]) == 1
    assert int(written["symmetry"]) == 0
    # The k mesh of the source job is kept, because the Kubo-Greenwood sum
    # runs on its own dense grid and only the density has to be converged.
    assert float(written["kspacing"]) == pytest.approx(0.14)
    assert (job / "dielectric" / "STRU").is_file()


def test_prepare_records_the_task_in_the_manifest(tmp_path: Path) -> None:
    job = _source_job(tmp_path)

    prepare(_prepare_args(job))

    manifest = json.loads((job / "workflow_dielectric.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "dielectric"
    assert manifest["tasks"] == ["dielectric"]
    assert manifest["matrix_keywords"] == ["out_mat_hs2", "out_mat_r"]
    assert manifest["basis_type"] == "lcao"


def test_prepare_rejects_a_plane_wave_basis(tmp_path: Path) -> None:
    job = _source_job(tmp_path)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nbasis_type pw\ngamma_only 0\nkspacing 0.14\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="basis_type"):
        prepare(_prepare_args(job))


def test_prepare_protects_an_existing_calculation(tmp_path: Path) -> None:
    job = _source_job(tmp_path)
    prepare(_prepare_args(job))

    with pytest.raises(RuntimeError, match="--override"):
        prepare(_prepare_args(job))
    assert prepare(_prepare_args(job, override=True)) == 0


def test_postprocess_writes_the_dielectric_results(tmp_path: Path, monkeypatch) -> None:
    job = _source_job(tmp_path)
    prepare(_prepare_args(job))
    _finish_task(job / "dielectric")
    _stub_pyatb(monkeypatch, np.eye(3) * 2.5325)

    assert postprocess(_postprocess_args(job)) == 0

    result = json.loads((job / "dielectric_results.json").read_text(encoding="utf-8"))
    assert result["workflow"] == "dielectric"
    assert result["task"] == "dielectric"
    assert result["occ_band"] == 8
    assert result["nspin"] == 1
    assert result["fermi_energy_ev"] == pytest.approx(2.8588620907)
    assert result["diagonal_mean"] == pytest.approx(2.5325)
    assert result["isotropic"] is True
    assert result["grid"] == [50, 50, 50]
    assert result["omega_ev"] == [0.0, 80.0]
    assert "dimensionless" in result["units"]["tensor"]
    assert result["source_structure"]["natoms"] == 2
    assert result["source_structure"]["volume_angstrom3"] > 0.0
    np.testing.assert_allclose(result["tensor"], np.eye(3) * 2.5325)


def test_postprocess_copies_the_matrices_into_the_work_directory(
    tmp_path: Path, monkeypatch
) -> None:
    job = _source_job(tmp_path)
    prepare(_prepare_args(job))
    _finish_task(job / "dielectric")
    calls = _stub_pyatb(monkeypatch, np.eye(3) * 2.5325)

    postprocess(_postprocess_args(job))

    workdir = job / "dielectric" / "pyatb"
    assert calls["workdir"] == workdir
    for name in _MATRICES:
        assert (workdir / name).is_file()
    assert calls["occ_band"] == 8
    assert calls["fermi_energy"] == pytest.approx(2.8588620907)
    assert calls["nspin"] == 1
    assert calls["grid"] == [50, 50, 50]
    assert calls["omega"] == (0.0, 80.0)


def test_postprocess_reads_a_job_without_a_manifest(tmp_path: Path, monkeypatch) -> None:
    """A job that already dumped its matrices can be read in place."""
    job = _finish_task(_source_job(tmp_path))
    _stub_pyatb(monkeypatch, np.eye(3) * 2.34)

    assert postprocess(_postprocess_args(job)) == 0

    result = json.loads((job / "dielectric_results.json").read_text(encoding="utf-8"))
    assert result["task"] == "job"
    assert result["diagonal_mean"] == pytest.approx(2.34)


def test_postprocess_reports_a_missing_matrix(tmp_path: Path, monkeypatch) -> None:
    job = _source_job(tmp_path)
    prepare(_prepare_args(job))
    _finish_task(job / "dielectric")
    (job / "dielectric" / "OUT.ABACUS" / "data-rR-sparse.csr").unlink()
    _stub_pyatb(monkeypatch, np.eye(3) * 2.34)

    with pytest.raises(FileNotFoundError, match="rR matrix"):
        postprocess(_postprocess_args(job))


def test_postprocess_warns_about_a_short_photon_energy_window(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    job = _finish_task(_source_job(tmp_path))
    _stub_pyatb(monkeypatch, np.eye(3) * 2.34)

    assert postprocess(_postprocess_args(job, omega=[0.0, 20.0])) == 0

    assert "below the 40 eV" in capsys.readouterr().out


def test_postprocess_reports_the_tensor_with_units(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    job = _finish_task(_source_job(tmp_path))
    _stub_pyatb(monkeypatch, np.eye(3) * 2.5325)

    postprocess(_postprocess_args(job))

    printed = capsys.readouterr().out
    assert "relative permittivity (dimensionless)" in printed
    assert "2.53250000" in printed
    assert "photon energy window: 0 - 80 eV" in printed


def test_postprocess_reports_an_anisotropic_tensor(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    job = _finish_task(_source_job(tmp_path))
    _stub_pyatb(monkeypatch, np.diag([2.0, 2.5, 3.0]))

    postprocess(_postprocess_args(job))

    printed = capsys.readouterr().out
    assert "isotropic average: 2.50000000" in printed
    assert "anisotropy" in printed
