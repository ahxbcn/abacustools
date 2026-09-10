"""Tests for the vacancy formation energy workflow."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from abacustools.commands.workflow.vacancy import postprocess, prepare
from abacustools.data.vacancy import (
    build_elemental_crystal,
    set_atom_empty,
    vacancy_formation_energy,
)
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


STRU = """\
ATOMIC_SPECIES
Si 28.0855 Si.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
5.43 0 0
0 5.43 0
0 0 5.43

ATOMIC_POSITIONS
Direct

Si
0.0
2
0.0 0.0 0.0
0.25 0.25 0.25
"""


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
    (job / "Si.upf").write_text("pseudo", encoding="utf-8")
    (job / "KPT").write_text("K_POINTS\n0\nGamma\n2 2 2 0 0 0\n", encoding="utf-8")
    return job


def _write_log(job: Path, calculation: str, energy: float) -> None:
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True, exist_ok=True)
    log = "running_scf.log" if calculation == "scf" else f"running_{calculation}.log"
    (output / log).write_text(
        f"E_KohnSham = {energy:.8f} eV\n"
        "charge density convergence is achieved\n"
        f"final etot is {energy:.8f} eV\n",
        encoding="utf-8",
    )


def _prepare_args(job: Path) -> Namespace:
    return Namespace(
        job=job,
        supercell=[1, 1, 1],
        index=[1],
        index_file=None,
        cal_reference=True,
        ref_dir="ref_element",
        max_step=100,
        force_thr_ev=0.01,
        stress_thr_kbar=0.5,
        relax_kspacing=None,
        scf_kspacing=None,
        override=False,
    )


def test_set_atom_empty_renames_and_reorders(tmp_path: Path) -> None:
    path = tmp_path / "STRU"
    path.write_text(STRU, encoding="utf-8")
    structure = AbacusSTRU.read(path)

    defect = set_atom_empty(structure, 0)

    assert defect.natoms == structure.natoms
    assert defect.labels[-1] == "Si_empty"
    assert defect.elements[-1] == "Si"


def test_build_elemental_crystal_uses_reference_table() -> None:
    crystal = build_elemental_crystal("Si", "Si.upf", None)

    assert crystal.natoms == 2
    assert set(crystal.elements) == {"Si"}
    assert crystal.pps[0] == "Si.upf"


def test_vacancy_formation_energy_formula() -> None:
    energy = vacancy_formation_energy(
        defect_energy=-19.0,
        original_energy=-20.0,
        supercell_factor=1,
        reference_atom_energy=-5.0,
    )

    assert energy == -4.0


def test_prepare_creates_original_defect_and_reference(tmp_path: Path) -> None:
    job = _base_job(tmp_path)

    assert prepare(_prepare_args(job)) == 0

    original = job / "vacancy_original_stru"
    assert ReadInput(original / "INPUT")["calculation"] == "cell-relax"
    assert (original / "STRU").is_file()
    assert (original / "final_scf" / "INPUT").is_file()
    assert not (original / "final_scf" / "STRU").exists()

    defect = job / "vacancy_defect_1_Si_1_1_1"
    defect_stru = AbacusSTRU.read(defect / "STRU")
    assert "Si_empty" in defect_stru.labels

    assert (job / "ref_element" / "Si" / "STRU").is_file()

    manifest = json.loads((job / "workflow_vacancy.json").read_text())
    assert manifest["tasks"][0] == "vacancy_original_stru"
    assert "vacancy_defect_1_Si_1_1_1" in manifest["tasks"]
    assert manifest["supercell"] == [1, 1, 1]


def test_postprocess_computes_formation_energy(tmp_path: Path) -> None:
    job = _base_job(tmp_path)
    prepare(_prepare_args(job))

    _write_log(job / "vacancy_original_stru" / "final_scf", "scf", -20.0)
    _write_log(job / "vacancy_defect_1_Si_1_1_1" / "final_scf", "scf", -19.0)
    _write_log(job / "ref_element" / "Si", "cell-relax", -10.0)

    assert postprocess(Namespace(job=job, version="", ref_dir=None, ref_file="ref_energy.txt", output="vac.json")) == 0

    report = json.loads((job / "vac.json").read_text())
    result = report["results"]["vacancy_defect_1_Si_1_1_1"]
    assert result["vacancy_formation_energy"] == -4.0
    assert report["reference_atom_energies"]["Si"] == -5.0
