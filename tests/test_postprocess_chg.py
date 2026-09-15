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


def _stru(job: Path) -> None:
    """Write a two-atom Si cell whose second atom sits in the c = 0.5 plane."""
    (job / "STRU").write_text(
        "ATOMIC_SPECIES\nSi 28.0855 Si.upf\n\nLATTICE_CONSTANT\n1.889726\n\n"
        "LATTICE_VECTORS\n4 0 0\n0 4 0\n0 0 4\n\nATOMIC_POSITIONS\n"
        "Cartesian\n\nSi\n0.0\n2\n0.0 0.0 0.0 1 1 1\n2.0 2.0 2.0 1 1 1\n",
        encoding="utf-8",
    )


def _args(job: Path, **overrides) -> Namespace:
    arguments = {
        "job": job,
        "spin": "total",
        "quantity": "density",
        "nci_plot": None,
        "nci_rho_max": 0.05,
        "difference": None,
        "cube": None,
        "profile": None,
        "profile_kind": "average",
        "data_output": None,
        "plot": None,
        "slice": None,
        "slice_index": 0.5,
        "slice_output": None,
        "slice_plot": None,
        "vmin": None,
        "vmax": None,
        "no_atoms": False,
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


def test_spin_choices_select_the_channels(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path, nspin=2)
    _cube(output / "SPIN1_CHG.cube", np.full((2, 2, 2), 0.75))
    _cube(output / "SPIN2_CHG.cube", np.full((2, 2, 2), 0.25))

    assert run(_args(job, spin="up", json=True)) == 0
    assert json.loads(capsys.readouterr().out)["electrons"] == pytest.approx(0.75 * 64.0)

    assert run(_args(job, spin="down", json=True)) == 0
    assert json.loads(capsys.readouterr().out)["electrons"] == pytest.approx(0.25 * 64.0)

    assert run(_args(job, spin="difference", json=True)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["spin"] == "difference"
    assert report["electrons"] == pytest.approx(0.5 * 64.0)
    assert report["magnetization"] == pytest.approx(0.5 * 64.0)
    # The valence comparison only means something for the total density.
    assert report["valence_electrons"] is None
    assert report["deviation"] is None


def test_spin_choice_needs_two_channels(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.ones((2, 2, 2)))

    assert run(_args(job, spin="difference")) == 1

    assert "needs an nspin 2 calculation" in capsys.readouterr().out


def test_difference_of_two_jobs(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.full((2, 2, 2), 0.5))
    other, other_output = _job(tmp_path / "other")
    _cube(other_output / "SPIN1_CHG.cube", np.full((2, 2, 2), 0.125))

    assert run(_args(job, difference=str(other), cube="difference.cube", json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["difference_of"] == str(other)
    assert report["electrons"] == pytest.approx((0.5 - 0.125) * 64.0)
    assert report["valence_electrons"] is None
    np.testing.assert_allclose(
        Charge.from_cube(str(job / "difference.cube"), format="abacus").data,
        0.375,
        rtol=1e-9,
    )


def test_difference_requires_the_same_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.ones((2, 2, 2)))
    other, other_output = _job(tmp_path / "other")
    Charge(
        np.ones((3, 3, 3)),
        np.diag([4.0, 4.0, 4.0]),
        np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        [14, 14],
        [4.0, 4.0],
    ).save_cube(str(other_output / "SPIN1_CHG.cube"))

    assert run(_args(job, difference=str(other))) == 1

    assert "incompatible grid shape for the two jobs" in capsys.readouterr().out


def test_slice_writes_the_data_plot_and_atoms(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.broadcast_to(values[None, None, :], (4, 4, 4)))
    _stru(job)

    assert run(_args(job, slice="c", slice_plot=_AUTO_PLOT, json=True)) == 0

    slc = json.loads(capsys.readouterr().out)["slice"]
    assert slc["axis"] == "c"
    assert slc["index"] == 2
    assert slc["position"] == pytest.approx(0.5)
    assert slc["distance"] == pytest.approx(2.0)
    assert slc["unit"] == "e/Angstrom^3"
    assert slc["points"] == 16
    assert [atom["label"] for atom in slc["atoms"]] == ["Si"]
    assert Path(slc["data_output"]).name == "chg_slice_c_0.5000.dat"
    assert Path(slc["plot"]).name == "chg_slice_c_0.5000.png"
    assert (job / "chg_slice_c_0.5000.png").is_file()

    rows = [
        line.split()
        for line in Path(slc["data_output"]).read_text().splitlines()
        if not line.startswith("#")
    ]
    assert len(rows) == 16
    assert all(len(row) == 3 for row in rows)
    # The density varies along c only, so every value of the slice is the same.
    np.testing.assert_allclose([float(row[2]) for row in rows], 3.0)


def test_slice_can_skip_the_atoms(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    job, output = _job(tmp_path)
    _cube(output / "SPIN1_CHG.cube", np.ones((2, 2, 2)))
    _stru(job)

    assert run(_args(job, slice="c", no_atoms=True, json=True)) == 0

    assert json.loads(capsys.readouterr().out)["slice"]["atoms"] == []


def test_develop_named_job_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path, nspin=2)
    _cube(output / "chgs1.cube", np.full((2, 2, 2), 0.75))
    _cube(output / "chgs2.cube", np.full((2, 2, 2), 0.25))

    assert run(_args(job, json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["source"] == "cube (develop: chgs1.cube, chgs2.cube)"
    assert report["cube_files"] == [str(output / "chgs1.cube"), str(output / "chgs2.cube")]
    assert report["electrons"] == pytest.approx(64.0)


def _varied_cube(path: Path) -> None:
    """Write a density that varies along c, so the NCI fields are not zero."""
    z = np.linspace(0.0, 4.0, 4, endpoint=False)
    values = (0.05 + 0.01 * np.cos(2 * np.pi * z / 4.0))[None, None, :]
    _cube(path, np.broadcast_to(values, (4, 4, 4)).copy())


def test_quantity_rdg_reports_field_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path, nspin=1)
    _varied_cube(output / "SPIN1_CHG.cube")

    assert run(_args(job, quantity="rdg", json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["quantity"] == "rdg"
    assert "electrons" not in report
    assert report["field"]["minimum"] >= 0.0
    assert report["grid"] == [4, 4, 4]


def test_quantity_dori_exports_a_bounded_cube(tmp_path: Path) -> None:
    job, output = _job(tmp_path, nspin=1)
    _varied_cube(output / "SPIN1_CHG.cube")

    assert run(_args(job, quantity="dori", cube="dori.cube")) == 0

    values = Charge.from_cube(str(job / "dori.cube"), format="abacus").data
    assert np.all(values >= 0.0)
    assert np.all(values < 1.0)


def test_nci_plot_is_written_and_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path, nspin=1)
    _varied_cube(output / "SPIN1_CHG.cube")

    assert run(_args(job, nci_plot="auto", json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["nci"]["points"] > 0
    assert report["nci"]["plot"] == str(job / "nci.png")
    assert (job / "nci.png").is_file()


def test_quantity_is_used_by_the_slice_and_its_unit(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path, nspin=1)
    _varied_cube(output / "SPIN1_CHG.cube")

    assert run(_args(job, quantity="sl2rho", slice="c", json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["slice"]["unit"] == "e/Angstrom^3"
    header = Path(report["slice"]["data_output"]).read_text().splitlines()[1]
    assert "e/Angstrom^3" in header


def test_profile_of_a_dimensionless_field_names_its_unit(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job, output = _job(tmp_path, nspin=1)
    _varied_cube(output / "SPIN1_CHG.cube")

    assert run(_args(job, quantity="rdg", profile="c", json=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["profile"]["unit"] == "dimensionless"
