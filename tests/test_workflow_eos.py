"""Tests for the equation-of-state workflow."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from abacustools.commands.workflow.eos import postprocess, prepare
from abacustools.data.eos import EV_PER_ANGSTROM3_TO_GPA, fit_birch_murnaghan
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


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


def _bm3_energy(volume: float, v0: float, e0: float, b0: float, bp: float) -> float:
    """Evaluate the third-order Birch-Murnaghan energy (b0 in eV/Ang^3)."""
    x = (v0 / volume) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((x - 1.0) ** 3 * bp + (x - 1.0) ** 2 * (6.0 - 4.0 * x))


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
    (job / "KPT").write_text("K_POINTS\n0\nGamma\n2 2 2 0 0 0\n", encoding="utf-8")
    return job


def _write_scf_result(job: Path, energy: float) -> None:
    output = job / "OUT.ABACUS"
    output.mkdir()
    (output / "running_scf.log").write_text(
        f"E_KohnSham = {energy + 0.1:.8f} eV\n"
        "density error = 1.0e-5\n"
        f"E_KohnSham = {energy:.8f} eV\n"
        "density error = 1.0e-8\n"
        "charge density convergence is achieved\n",
        encoding="utf-8",
    )


def _prepare_args(job: Path) -> Namespace:
    return Namespace(job=job, start=0.9, end=1.1, step=0.05, relax=False, override=False)


def test_fit_recovers_birch_murnaghan_parameters() -> None:
    v0, e0, b0, bp = 100.0, -10.0, 0.5, 4.0
    volumes = [v0 * scale for scale in (0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1)]
    energies = [_bm3_energy(volume, v0, e0, b0, bp) for volume in volumes]

    fit = fit_birch_murnaghan(volumes, energies)

    assert abs(fit.volume - v0) < 1e-6
    assert abs(fit.energy - e0) < 1e-6
    assert abs(fit.bulk_modulus - b0 * EV_PER_ANGSTROM3_TO_GPA) < 1e-3
    assert abs(fit.bulk_modulus_derivative - bp) < 1e-6
    assert fit.residual < 1e-9


def test_prepare_writes_volume_scaled_jobs(tmp_path: Path) -> None:
    job = _base_job(tmp_path)

    assert prepare(_prepare_args(job)) == 0

    manifest = json.loads((job / "workflow_eos.json").read_text())
    assert manifest["tasks"] == ["eos_0.900", "eos_0.950", "eos_1.000", "eos_1.050", "eos_1.100"]
    assert len(manifest["points"]) == 5
    assert ReadInput(job / "eos_0.900" / "INPUT")["calculation"] == "scf"
    assert (job / "eos_0.900" / "KPT").is_symlink()

    scaled = AbacusSTRU.read(job / "eos_0.900" / "STRU")
    volume = abs(float(np.linalg.det(np.asarray(scaled.cell, dtype=float))))
    assert abs(volume - manifest["equilibrium_volume"] * 0.9) < 1e-6


def test_prepare_relax_sets_calculation(tmp_path: Path) -> None:
    job = _base_job(tmp_path)

    prepare(Namespace(job=job, start=0.9, end=1.1, step=0.05, relax=True, override=False))

    assert ReadInput(job / "eos_1.000" / "INPUT")["calculation"] == "relax"


def test_postprocess_fits_and_writes_report(tmp_path: Path) -> None:
    job = _base_job(tmp_path)
    prepare(_prepare_args(job))
    manifest = json.loads((job / "workflow_eos.json").read_text())
    v0 = manifest["equilibrium_volume"]
    for point in manifest["points"]:
        energy = _bm3_energy(point["volume"], v0, -10.0, 0.5, 4.0)
        _write_scf_result(job / point["name"], energy)

    assert postprocess(Namespace(job=job, version="", output="eos.json", plot="eos.png")) == 0

    report = json.loads((job / "eos.json").read_text())
    assert abs(report["equilibrium_volume"] - v0) < 1e-3
    assert abs(report["bulk_modulus"] - 0.5 * EV_PER_ANGSTROM3_TO_GPA) < 0.5
    assert abs(report["equilibrium_energy"] - (-10.0)) < 1e-3
    assert (job / "eos.png").is_file()
