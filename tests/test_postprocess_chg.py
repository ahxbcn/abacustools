"""Tests for the ``postprocess chg`` command."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.postprocess.chg import _AUTO_PLOT, run
from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.grid import Charge, Grid


def _job(tmp_path: Path, *, nspin: int = 1) -> tuple[Path, Path]:
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        f"INPUT_PARAMETERS\nsuffix ABACUS\nnspin {nspin}\n", encoding="utf-8"
    )
    return job, output


def _cube(path: Path, data) -> None:
    Charge(
        np.asarray(data, dtype=float),
        np.diag([4.0, 4.0, 4.0]),
        np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        [14, 14],
        [4.0, 4.0],
    ).save_cube(str(path))


def _args(job: Path, **overrides) -> Namespace:
    arguments = {
        "job": job,
        "cube": None,
        "profile": None,
        "profile_kind": "average",
        "data_output": None,
        "plot": None,
        "json": False,
    }
    arguments.update(overrides)
    return Namespace(**arguments)


def test_summary_reports_the_integrated_electrons(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.full((4, 4, 4), 0.25))

    assert run(_args(job)) == 0

    printed = capsys.readouterr().out
    assert "density source: cube" in printed
    assert "electrons: 16.000000 e" in printed
    assert "valence electrons 8.000000 e" in printed
    assert "deviation +8.000000 e" in printed


def test_summary_as_json_reports_the_source_and_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.full((4, 4, 4), 0.25))

    assert run(_args(job, json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["job"] == str(job)
    assert report["source"] == "cube"
    assert report["cube_files"] == [str(output / "SPIN1_CHG.cube")]
    assert report["nspin"] == 1
    assert report["grid"] == [4, 4, 4]
    assert report["volume_angstrom3"] == pytest.approx(64.0)
    assert report["electrons"] == pytest.approx(16.0)
    assert report["deviation"] == pytest.approx(8.0)


def test_two_spin_channels_are_summed(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path, nspin=2)
    _cube(output / "SPIN1_CHG.cube", np.full((2, 2, 2), 0.75))
    _cube(output / "SPIN2_CHG.cube", np.full((2, 2, 2), 0.25))

    assert run(_args(job, json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["nspin"] == 2
    assert report["electrons"] == pytest.approx(1.0 * 64.0)


def test_cube_export_writes_the_total_density(tmp_path: Path) -> None:
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.full((4, 4, 4), 0.25))

    assert run(_args(job, cube="export.cube")) == 0

    written = job / "export.cube"
    assert written.is_file()
    np.testing.assert_allclose(
        Charge.from_cube(str(written), format="abacus").data, 0.25, rtol=1e-9
    )
    np.testing.assert_allclose(
        Grid.from_cube(str(written)).data, 0.25 * BOHR_TO_ANG**3, rtol=1e-9
    )


def test_profile_writes_the_data_and_an_explicit_plot(tmp_path: Path) -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.broadcast_to(values[None, None, :], (2, 2, 4)))

    assert run(
        _args(job, profile="c", plot="profile.png", data_output="profile.dat")
    ) == 0

    rows = [
        line.split()
        for line in (job / "profile.dat").read_text().splitlines()
        if not line.startswith("#")
    ]
    assert len(rows) == 4
    table = np.array([[float(value) for value in row] for row in rows])
    np.testing.assert_allclose(table[:, 0], np.linspace(0.0, 4.0, 4))
    np.testing.assert_allclose(table[:, 1], values)
    assert (job / "profile.png").is_file()


def test_profile_default_names_and_integral_kind(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.full((2, 2, 2), 0.5))

    assert run(
        _args(job, profile="c", profile_kind="integral", plot=_AUTO_PLOT, json=True)
    ) == 0

    report = json.loads(capsys.readouterr().out)
    profile = report["profile"]
    assert profile["kind"] == "integral"
    assert profile["unit"] == "e"
    assert Path(profile["data_output"]).name == "chg_profile_c_integral.dat"
    assert Path(profile["plot"]).name == "chg_profile_c_integral.png"
    # Every plane slab sums to the integrated number of electrons.
    assert profile["sum"] == pytest.approx(report["electrons"])
    assert (job / "chg_profile_c_integral.png").is_file()


def test_missing_density_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, _ = _job(tmp_path)

    assert run(_args(job)) == 1

    assert "Charge-density analysis failed" in capsys.readouterr().out
