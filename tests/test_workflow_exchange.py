"""Tests for the magnetic exchange coupling workflow."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from abacustools.commands.workflow.exchange import postprocess, prepare
from abacustools.data.exchange import ExchangeError, angle_between
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


STRU = """\
ATOMIC_SPECIES
Fe 55.845 Fe.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
5 0 0
0 5 0
0 0 5

ATOMIC_POSITIONS
Cartesian

Fe
0.0
2
0 0 0 mag 0 0 2.5
2 2 2 mag 2.5 0 0
"""

MOMENT1 = [0.0, 0.0, 2.5]
MOMENT2 = [2.5, 0.0, 0.0]


def _write_job(job: Path, *, nspin: int = 4, stru: str = STRU) -> Path:
    job.mkdir(parents=True, exist_ok=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        f"nspin {nspin}\n"
        "scf_thr 1e-5\n"
        "gamma_only 1\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(stru, encoding="utf-8")
    (job / "Fe.upf").write_text("pseudo", encoding="utf-8")
    return job


def _prepare_args(job: Path, **overrides) -> Namespace:
    arguments = {
        "job": job,
        "pairs": [["Fe", "1", "Fe", "2"]],
        "info_file": "magj.txt",
        "step": 10.0,
        "number": 3,
        "override": False,
        "generate_scripts": False,
        "submission_type": None,
        "abacus_command": None,
    }
    arguments.update(overrides)
    return Namespace(**arguments)


def _postprocess_args(job: Path, **overrides) -> Namespace:
    arguments = {
        "job": job,
        "version": "LTS3.10.1",
        "output": "exchange_results.json",
        "plot": "exchange_fit.png",
        "allow_unconverged": False,
    }
    arguments.update(overrides)
    return Namespace(**arguments)


def _read_moments(path: Path) -> tuple[list[float], list[float]]:
    structure = AbacusSTRU.read(str(path / "STRU"))
    return (
        [float(value) for value in structure.atoms[0].atommag],
        [float(value) for value in structure.atoms[1].atommag],
    )


def test_prepare_writes_the_reference_and_four_state_jobs(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job")
    prepare(_prepare_args(job))

    manifest = json.loads((job / "workflow_magj.json").read_text())
    pair = manifest["pairs"][0]
    assert pair["name"] == "Fe1_Fe2"
    assert (pair["atom1"], pair["atom2"]) == (1, 2)
    assert pair["reference_angle"] == pytest.approx(90.0)
    assert len(manifest["cases"]) == 3 * 3
    assert len(manifest["tasks"]) == 1 + 3 * 3

    original = _read_moments(job / "magj/original")
    assert original[0] == pytest.approx(MOMENT1)
    assert original[1] == pytest.approx(MOMENT2)

    for case in manifest["cases"]:
        moments = _read_moments(job / case["task"])
        # The STRU file stores eight decimals per component.
        assert moments[0] == pytest.approx(case["moment1"], abs=1e-7)
        assert moments[1] == pytest.approx(case["moment2"], abs=1e-7)
        assert angle_between(*moments) == pytest.approx(90.0 + case["tilt"], abs=1e-6)
        inputs = ReadInput(job / case["task"] / "INPUT")
        assert inputs["calculation"] == "scf"
        assert float(inputs["scf_thr"]) <= 1e-7


def test_prepare_reads_the_reference_pair_file(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job")
    (job / "magj.txt").write_text("Fe 2 Fe 1\n", encoding="utf-8")
    prepare(_prepare_args(job, pairs=None, step=20.0, number=2))

    manifest = json.loads((job / "workflow_magj.json").read_text())
    pair = manifest["pairs"][0]
    # The pair is canonicalised to the atom order of the structure.
    assert (pair["atom1"], pair["atom2"]) == (1, 2)
    assert pair["name"] == "Fe1_Fe2"
    assert [case["tilt"] for case in manifest["cases"]] == [20.0] * 3 + [40.0] * 3
    assert (job / "magj/Fe1_Fe2_angle20_both/STRU").is_file()


def test_prepare_drops_duplicate_pairs(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job")
    prepare(
        _prepare_args(
            job,
            pairs=[["Fe", "1", "Fe", "2"], ["Fe", "2", "Fe", "1"]],
            step=10.0,
            number=2,
        )
    )
    manifest = json.loads((job / "workflow_magj.json").read_text())
    assert len(manifest["pairs"]) == 1
    assert len(manifest["cases"]) == 3 * 2


def test_prepare_requires_noncollinear_moments(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job", nspin=2)
    with pytest.raises(RuntimeError, match="nspin 4"):
        prepare(_prepare_args(job))


def test_prepare_accepts_noncolin_and_lspinorb_flags(tmp_path: Path) -> None:
    # ABACUS raises nspin to 4 itself for these flags.
    for flag in ("noncolin 1", "lspinorb 1"):
        job = _write_job(tmp_path / flag.split()[0])
        (job / "INPUT").write_text(
            "INPUT_PARAMETERS\n"
            "calculation scf\n"
            f"{flag}\n"
            "scf_thr 1e-7\n"
            "gamma_only 1\n",
            encoding="utf-8",
        )
        prepare(_prepare_args(job))
        manifest = json.loads((job / "workflow_magj.json").read_text())
        assert len(manifest["pairs"]) == 1


def test_prepare_rejects_a_pair_without_moments(tmp_path: Path) -> None:
    without_moment = STRU.replace("2 2 2 mag 2.5 0 0", "2 2 2 mag 0 0 0")
    job = _write_job(tmp_path / "job", stru=without_moment)
    with pytest.raises(ExchangeError, match="zero magnetic moment"):
        prepare(_prepare_args(job))


def test_prepare_accepts_collinear_style_scalar_moments(tmp_path: Path) -> None:
    # A scalar moment is a moment along z, so both atoms start parallel.
    scalar = STRU.replace("mag 0 0 2.5", "mag 2.5").replace("mag 2.5 0 0", "mag 2.5")
    job = _write_job(tmp_path / "job", stru=scalar)
    prepare(_prepare_args(job))

    manifest = json.loads((job / "workflow_magj.json").read_text())
    assert manifest["pairs"][0]["reference_angle"] == pytest.approx(0.0)
    assert manifest["cases"][0]["pair_angle"] == pytest.approx(10.0)


def test_prepare_reports_a_missing_pair_file(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job")
    with pytest.raises(FileNotFoundError, match="magnetic pair file"):
        prepare(_prepare_args(job, pairs=None))


def test_postprocess_fits_a_known_coupling(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job")
    prepare(_prepare_args(job))
    manifest = json.loads((job / "workflow_magj.json").read_text())

    # Energies of a pure bilinear model E = c * (m1 . m2), for which the
    # four-state fit must return exactly c.
    coefficient = 0.012
    energies = {
        manifest["original_task"]: coefficient * float(np.dot(MOMENT1, MOMENT2))
    }
    for case in manifest["cases"]:
        energies[case["task"]] = coefficient * float(
            np.dot(case["moment1"], case["moment2"])
        )

    def fake_state(path: Path, version: str) -> dict:
        return {"energy": energies[path.relative_to(job).as_posix()], "converged": True}

    with patch("abacustools.commands.workflow.exchange._read_state", side_effect=fake_state):
        postprocess(_postprocess_args(job))

    report = json.loads((job / "exchange_results.json").read_text())
    pair = report["pairs"][0]
    assert pair["name"] == "Fe1_Fe2"
    assert pair["reference_angle"] == pytest.approx(90.0)
    # The fit is defined on the unit directions, so the moments of 2.5 Bohr
    # magneton scale the fitted coefficient by 2.5 * 2.5.
    assert pair["coupling"]["j_mev"] == pytest.approx(
        coefficient * 2.5 * 2.5 * 1000.0, abs=1e-8
    )
    assert pair["coupling"]["r_squared"] == pytest.approx(1.0, abs=1e-10)
    assert pair["coupling"]["points"] == 3
    assert len(pair["points"]) == 3
    assert pair["points"][0]["pair_angle"] == pytest.approx(100.0)
    assert pair["points"][0]["delta_energy_ev"] == pytest.approx(
        pair["coupling"]["slope_ev"] * pair["points"][0]["x"]
        + pair["coupling"]["intercept_ev"],
        abs=1e-12,
    )
    assert (job / "exchange_fit.png").is_file()


def test_postprocess_stops_on_a_missing_calculation(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job")
    prepare(_prepare_args(job))
    with patch(
        "abacustools.commands.workflow.exchange._read_state",
        side_effect=RuntimeError("energy was not found in the output"),
    ):
        with pytest.raises(RuntimeError, match="energy was not found"):
            postprocess(_postprocess_args(job))


def test_postprocess_reports_unconverged_states(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job")
    prepare(_prepare_args(job))
    manifest = json.loads((job / "workflow_magj.json").read_text())
    stalled = manifest["cases"][0]["task"]

    def fake_state(path: Path, version: str) -> dict:
        task = path.relative_to(job).as_posix()
        return {"energy": -10.0, "converged": task != stalled}

    with patch("abacustools.commands.workflow.exchange._read_state", side_effect=fake_state):
        with pytest.raises(RuntimeError, match="did not converge"):
            postprocess(_postprocess_args(job))
        postprocess(_postprocess_args(job, allow_unconverged=True))

    report = json.loads((job / "exchange_results.json").read_text())
    assert report["unconverged_tasks"] == [stalled]
    assert report["pairs"][0]["points"][0]["converged"] is False


def test_registered_command_exposes_both_stages() -> None:
    from abacustools.main import _create_parser

    parser = _create_parser("abacustools")
    namespace = parser.parse_args(
        ["workflow", "exchange", "prepare", "-j", ".", "--pair", "Fe", "1", "Fe", "2"]
    )
    assert namespace.workflow_command == "exchange"
    assert namespace.pairs == [["Fe", "1", "Fe", "2"]]

    alias = parser.parse_args(
        ["workflow", "magj", "postprocess", "-j", "."]
    )
    assert alias.workflow_command == "magj"
    assert alias.output == "exchange_results.json"
