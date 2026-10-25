"""Tests for the pEDA/NOCV workflow."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from abacustools.commands.workflow.eda import density, postprocess, prepare
from abacustools.data.eda import (
    ElectrostaticCrossTerms,
    LTSStateDensityResult,
    PEDAJobEnergy,
    PEDANOCVResult,
    PEDAEnergyTerms,
)
from abacustools.data.nocv import NOCVEnergyDecomposition


def _make_jobs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    job = tmp_path / "job"
    fragment_a = job / "frag_a"
    fragment_b = job / "frag_b"
    reference_a = job / "ref_a"
    reference_b = job / "ref_b"
    for path in (job, fragment_a, fragment_b, reference_a, reference_b):
        path.mkdir(parents=True, exist_ok=True)
    return job, fragment_a, fragment_b, reference_a, reference_b


def _prepare_args(
    job: Path,
    fragment_a: Path,
    fragment_b: Path,
    reference_a: Path,
    reference_b: Path,
) -> Namespace:
    return Namespace(
        job=job,
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        reference_a=reference_a,
        reference_b=reference_b,
        atoms_a=[1, 2],
        atoms_b=[3, 4],
        override=False,
        version="auto",
    )


def test_registered_command_exposes_both_stages() -> None:
    from abacustools.main import _create_parser

    parser = _create_parser("abacustools")
    prepare_namespace = parser.parse_args(
        [
            "workflow",
            "peda",
            "prepare",
            "-j",
            ".",
            "--fragment-a",
            "frag_a",
            "--fragment-b",
            "frag_b",
            "--atoms-a",
            "1",
            "2",
            "--atoms-b",
            "3",
            "4",
        ]
    )
    assert prepare_namespace.workflow_command == "peda"
    assert prepare_namespace.peda_command == "prepare"
    assert prepare_namespace.atoms_a == [1, 2]
    assert prepare_namespace.version

    density_namespace = parser.parse_args(["workflow", "peda", "density", "-j", "."])
    assert density_namespace.peda_command == "density"
    assert density_namespace.output == "peda_density.json"

    postprocess_namespace = parser.parse_args(["workflow", "peda", "postprocess", "-j", "."])
    assert postprocess_namespace.peda_command == "postprocess"
    assert postprocess_namespace.output == "peda_results.json"


def test_prepare_writes_state_jobs_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, fragment_a, fragment_b, reference_a, reference_b = _make_jobs(tmp_path)
    args = _prepare_args(job, fragment_a, fragment_b, reference_a, reference_b)
    state = SimpleNamespace(direct=(0.0, 0.0, 0.0), channels={1: object()})

    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.read_job_structure",
        lambda path: (
            {"basis_type": "lcao", "gamma_only": 1, "nspin": 1},
            "STRU",
            SimpleNamespace(natoms=4),
        ),
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.build_peda_kpoint_states",
        lambda *args, **kwargs: [state],
    )

    def fake_prepare_state(job_path, states, destination, **kwargs):
        Path(destination).mkdir(parents=True)
        return Path(destination)

    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.prepare_peda_state_job",
        fake_prepare_state,
    )

    assert prepare(args) == 0
    manifest = json.loads((job / "workflow_peda.json").read_text())
    assert manifest["tasks"] == ["peda/states/p0", "peda/states/p_ts"]
    assert manifest["fragment_a_atoms"] == [1, 2]
    assert manifest["fragment_b_atoms"] == [3, 4]
    assert manifest["fragment_a_atom_map"] == [1, 2]
    assert manifest["fragment_b_atom_map"] == [3, 4]
    assert "validated_settings" in manifest
    assert manifest["nocv"] is True
    assert (job / "peda/states/p0").is_dir()
    assert (job / "peda/states/p_ts").is_dir()


def test_prepare_multik_skips_pts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, fragment_a, fragment_b, reference_a, reference_b = _make_jobs(tmp_path)
    args = _prepare_args(job, fragment_a, fragment_b, reference_a, reference_b)
    state = SimpleNamespace(
        direct=(0.5, 0.0, 0.0),
        channels={1: object(), 2: object()},
    )

    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.read_job_structure",
        lambda path: (
            {"basis_type": "lcao", "gamma_only": 0, "nspin": 2},
            "STRU",
            SimpleNamespace(natoms=4),
        ),
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.build_peda_kpoint_states",
        lambda *args, **kwargs: [state],
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.prepare_peda_state_job",
        lambda job_path, states, destination, **kwargs: (
            Path(destination).mkdir(parents=True) or Path(destination)
        ),
    )

    assert prepare(args) == 0
    manifest = json.loads((job / "workflow_peda.json").read_text())
    assert manifest["tasks"] == ["peda/states/p0"]
    assert manifest["nocv"] is False
    assert manifest["nspin"] == 2
    assert not (job / "peda/states/p_ts").exists()


def test_prepare_lts_adds_density_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, fragment_a, fragment_b, reference_a, reference_b = _make_jobs(tmp_path)
    args = _prepare_args(job, fragment_a, fragment_b, reference_a, reference_b)
    args.version = "LTS3.10.1"
    state = SimpleNamespace(direct=(0.0, 0.0, 0.0), channels={1: object()})

    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.read_job_structure",
        lambda path: (
            {"basis_type": "lcao", "gamma_only": 1, "nspin": 1},
            "STRU",
            SimpleNamespace(natoms=4),
        ),
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.build_peda_kpoint_states",
        lambda *args, **kwargs: [state],
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.prepare_lts_p0_density_job",
        lambda job_path, states, destination, **kwargs: (
            Path(destination).mkdir(parents=True) or Path(destination)
        ),
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.prepare_peda_state_job",
        lambda job_path, states, destination, **kwargs: (
            Path(destination).mkdir(parents=True) or Path(destination)
        ),
    )

    assert prepare(args) == 0
    manifest = json.loads((job / "workflow_peda.json").read_text())
    assert manifest["tasks"][0] == "peda/density/p0"
    assert manifest["density_mode"] == "file"
    assert manifest["abacus_version"] == "3.10.1LTS"
    assert (job / "peda/density/p0").is_dir()


def test_density_stage_writes_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, fragment_a, fragment_b, _reference_a, _reference_b = _make_jobs(tmp_path)
    (job / "peda/density/p0").mkdir(parents=True)
    (job / "peda/states/p0").mkdir(parents=True)
    (job / "peda/states/p_ts").mkdir(parents=True)
    (job / "workflow_peda.json").write_text(
        json.dumps(
            {
                "format": 1,
                "workflow": "peda",
                "tasks": ["peda/density/p0", "peda/states/p0", "peda/states/p_ts"],
                "fragment_a_job": "frag_a",
                "fragment_b_job": "frag_b",
                "fragment_a_atoms": [1, 2],
                "fragment_b_atoms": [3, 4],
                "density_mode": "file",
                "density_task": "peda/density/p0",
                "nspin": 1,
                "gamma_only": True,
            }
        )
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.build_peda_kpoint_states",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.assemble_lts_peda_state_densities",
        lambda *args, **kwargs: LTSStateDensityResult(
            density_job=str(job / "peda/density/p0"),
            p0_files={1: "p0.cube"},
            p_ts_files={1: "pts.cube"},
            electron_numbers={1: 2.0},
            band_files={1: ("band.cube",)},
        ),
    )

    assert density(Namespace(job=job, version="LTS3.10.1", output="peda_density.json")) == 0
    result = json.loads((job / "peda_density.json").read_text())
    assert result["electron_numbers"] == {"1": 2.0}


def _fake_result() -> PEDANOCVResult:
    terms = PEDAEnergyTerms(
        preparation=0.1,
        electrostatic=-1.0,
        pauli=2.0,
        orbital=-3.0,
        dispersion=0.0,
        interaction_electronic=-2.0,
        interaction=-2.0,
        bond=-1.9,
        electrostatic_fraction=0.25,
        orbital_fraction=0.75,
        dispersion_fraction=0.0,
    )
    electrostatic = ElectrostaticCrossTerms(
        electron_electron=4.0,
        electron_nucleus_ab=-3.0,
        electron_nucleus_ba=-3.0,
        ion_ion=1.0,
        total=-1.0,
    )
    nocv = {
        1: NOCVEnergyDecomposition(
            channels=(),
            paired_total_ev=-3.0,
            unpaired_total_ev=0.0,
            total_ev=-3.0,
        )
    }
    return PEDANOCVResult(
        terms=terms,
        electrostatic=electrostatic,
        nocv_channels=nocv,
        nocv_total_ev=-3.0,
        orbital_error_ev=0.0,
        energy_ledger={
            "full": PEDAJobEnergy(-10.0, -9.5, -0.5, "d3_bj"),
            "p0": PEDAJobEnergy(-7.0, -7.0, 0.0, "none"),
        },
        dmr_validation={
            "p0": {
                "field": "p0",
                "max_block_error": 1.0e-10,
                "electron_numbers": [4.0],
            }
        },
    )


def test_postprocess_writes_gamma_nocv_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, fragment_a, fragment_b, reference_a, reference_b = _make_jobs(tmp_path)
    for name in ("p0", "p_ts"):
        (job / "peda/states" / name).mkdir(parents=True)
    (job / "workflow_peda.json").write_text(
        json.dumps(
            {
                "format": 1,
                "workflow": "peda",
                "tasks": ["peda/states/p0", "peda/states/p_ts"],
                "fragment_a_job": "frag_a",
                "fragment_b_job": "frag_b",
                "reference_a_job": "ref_a",
                "reference_b_job": "ref_b",
                "fragment_a_atoms": [1, 2],
                "fragment_b_atoms": [3, 4],
                "nspin": 1,
                "gamma_only": True,
                "nocv": True,
                "kpoint_count": 1,
            }
        )
    )

    energies = {
        job: -10.0,
        fragment_a: -4.0,
        fragment_b: -4.0,
        reference_a: -5.0,
        reference_b: -5.0,
    }
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.get_result_from_job",
        lambda path, *args, **kwargs: {
            "energy": energies[Path(path)],
            "converged": True,
        },
    )
    monkeypatch.setattr(
        "abacustools.commands.workflow.eda.analyze_gamma_peda_nocv",
        lambda *args, **kwargs: _fake_result(),
    )

    assert postprocess(Namespace(job=job, version="test", output="peda_results.json")) == 0
    report = json.loads((job / "peda_results.json").read_text())
    assert report["analysis"] == "gamma_peda_nocv"
    assert report["kpoint_count"] == 1
    assert report["e0_ev"] == pytest.approx(-6.5)
    assert report["terms_ev"]["bond"] == pytest.approx(-1.9)
    assert report["nocv_total_ev"] == pytest.approx(-3.0)
    assert report["nocv_channels"]["1"]["total_ev"] == pytest.approx(-3.0)
    assert report["energy_ledger_ev"]["full"]["electronic_ev"] == pytest.approx(-9.5)
    assert report["energy_ledger_ev"]["full"]["dispersion_ev"] == pytest.approx(-0.5)
    assert report["energy_ledger_ev"]["full"]["vdw_method"] == "d3_bj"
    assert report["dmr_validation"]["p0"]["max_block_error"] == pytest.approx(1.0e-10)
