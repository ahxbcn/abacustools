"""Tests for the molecular-dynamics trajectory support."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.file.traj import run as convert_trajectory
from abacustools.commands.postprocess.md import run as postprocess_md
from abacustools.data.md import (
    read_md_dump,
    read_trajectory,
    select_frames,
    write_trajectory,
)


LATTICE_CONSTANT = 1.889726


def _block(step: int, shift: float, *, forces: bool = True, velocities: bool = True) -> str:
    """Return one ``MD_dump`` block with two atoms."""
    header = "INDEX    LABEL    POSITION (Angstrom)"
    if forces:
        header += "    FORCE (eV/Angstrom)"
    if velocities:
        header += "    VELOCITY (Angstrom/fs)"
    rows = []
    for index, (label, position, force, velocity) in enumerate(
        (
            ("Si", (0.0, 0.0, 0.0), (0.1, 0.2, 0.3), (0.01, 0.02, 0.03)),
            ("O", (2.5, 2.5, 2.5), (-0.1, -0.2, -0.3), (-0.01, -0.02, -0.03)),
        )
    ):
        row = [f"{index}", label]
        row.extend(f"{value + shift:.12f}" for value in position)
        if forces:
            row.extend(f"{value:.12f}" for value in force)
        if velocities:
            row.extend(f"{value:.12f}" for value in velocity)
        rows.append("  " + "  ".join(row))
    return "\n".join(
        [
            f"MDSTEP:  {step}",
            f"LATTICE_CONSTANT: {LATTICE_CONSTANT:.12f} Angstrom",
            "LATTICE_VECTORS",
            "  5.000000000000  0.000000000000  0.000000000000",
            "  0.000000000000  5.000000000000  0.000000000000",
            "  0.000000000000  0.000000000000  5.000000000000",
            "VIRIAL (kbar)",
            "  1.000000000000  0.000000000000  0.000000000000",
            "  0.000000000000  1.000000000000  0.000000000000",
            "  0.000000000000  0.000000000000  1.000000000000",
            header,
            *rows,
        ]
    )


def _job(tmp_path: Path, *, dump: bool = True, steps: int = 3, **flags) -> Path:
    """Create an MD job with an ``MD_dump`` or with ``STRU_MD_*`` files."""
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\ncalculation md\nnspin 1\n", encoding="utf-8"
    )
    if dump:
        blocks = [
            _block(step, 0.1 * step, **flags) for step in range(steps)
        ]
        (output / "MD_dump").write_text("\n".join(blocks) + "\n", encoding="utf-8")
    else:
        for step in range(steps):
            (job / f"STRU_MD_{step}").write_text(
                "ATOMIC_SPECIES\nSi 28.0855 Si.upf\n\n"
                "LATTICE_CONSTANT\n"
                f"{LATTICE_CONSTANT}\n\n"
                "LATTICE_VECTORS\n5 0 0\n0 5 0\n0 0 5\n\n"
                "ATOMIC_POSITIONS\nCartesian\n\n"
                "Si\n0.0\n1\n"
                f"{0.1 * step} 0.0 0.0 1 1 1 v 0.01 0.0 0.0\n",
                encoding="utf-8",
            )
    return job


def test_read_md_dump_parses_every_dumped_field(tmp_path: Path) -> None:
    path = _job(tmp_path) / "OUT.ABACUS" / "MD_dump"

    frames = read_md_dump(path)

    assert [frame.step for frame in frames] == [0, 1, 2]
    first = frames[0]
    assert first.symbols == ["Si", "O"]
    np.testing.assert_allclose(first.cell, np.diag([5.0, 5.0, 5.0]) * LATTICE_CONSTANT)
    np.testing.assert_allclose(first.positions[1], [2.5, 2.5, 2.5])
    np.testing.assert_allclose(first.forces[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(first.velocities[1], [-0.01, -0.02, -0.03])
    np.testing.assert_allclose(frames[2].positions[0], [0.2, 0.2, 0.2])
    assert first.virial is not None and first.virial.shape == (3, 3)


def test_read_md_dump_handles_a_position_only_dump(tmp_path: Path) -> None:
    job = _job(tmp_path, forces=False, velocities=False)

    frames = read_md_dump(job / "OUT.ABACUS" / "MD_dump")

    assert len(frames) == 3
    assert frames[0].forces is None
    assert frames[0].velocities is None


def test_trajectory_from_stru_md_files(tmp_path: Path) -> None:
    job = _job(tmp_path, dump=False)

    frames = read_trajectory(job)

    assert [frame.step for frame in frames] == [0, 1, 2]
    assert frames[0].symbols == ["Si"]
    np.testing.assert_allclose(frames[2].positions[0], [0.2, 0.0, 0.0])
    np.testing.assert_allclose(frames[0].velocities[0], [0.01, 0.0, 0.0])
    assert frames[0].forces is None


def test_trajectory_attaches_the_log_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    (job / "OUT.ABACUS" / "running_md.log").write_text("log", encoding="utf-8")
    monkeypatch.setattr(
        "abacustools.data.md.read_md_history",
        lambda *args, **kwargs: [
            {"step": 0, "energy": -10.0, "temperature": 300.0, "pressure": 1.0},
            {"step": 2, "energy": -9.5, "temperature": 320.0, "pressure": 2.0},
        ],
    )

    frames = read_trajectory(job)

    assert frames[0].energy == pytest.approx(-10.0)
    assert frames[0].temperature == pytest.approx(300.0)
    assert frames[1].energy is None
    assert frames[2].pressure == pytest.approx(2.0)


def test_select_frames_by_step_and_stride(tmp_path: Path) -> None:
    frames = read_md_dump(_job(tmp_path) / "OUT.ABACUS" / "MD_dump")

    assert [frame.step for frame in select_frames(frames, first=1)] == [1, 2]
    assert [frame.step for frame in select_frames(frames, last=1)] == [0, 1]
    assert [frame.step for frame in select_frames(frames, stride=2)] == [0, 2]
    with pytest.raises(ValueError, match="positive integer"):
        select_frames(frames, stride=0)


def test_write_trajectory_round_trips_through_ase(tmp_path: Path) -> None:
    from ase.io import read

    frames = read_md_dump(_job(tmp_path) / "OUT.ABACUS" / "MD_dump")
    path = write_trajectory(tmp_path / "trajectory.extxyz", frames)

    structures = read(str(path), index=":")
    assert len(structures) == len(frames)
    np.testing.assert_allclose(structures[1].positions, frames[1].positions)
    np.testing.assert_allclose(structures[1].cell, frames[1].cell)
    np.testing.assert_allclose(
        structures[1].get_forces(), frames[1].forces, atol=1e-6
    )
    np.testing.assert_allclose(
        structures[1].get_velocities(), frames[1].velocities, atol=1e-6
    )


def test_postprocess_md_writes_a_trajectory(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    job = _job(tmp_path)

    assert postprocess_md(Namespace(
        job=job, output="traj.extxyz", format=None, first=None, last=None,
        stride=2, version="LTS3.10.1", json=True,
    )) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["frames"] == 2
    assert report["atoms"] == 2
    assert report["steps"] == [0, 2]
    assert report["has_forces"] and report["has_velocities"] and report["has_virial"]
    assert (job / "traj.extxyz").is_file()


def test_file_traj_converts_between_formats(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from ase.io import read

    frames = read_md_dump(_job(tmp_path) / "OUT.ABACUS" / "MD_dump")
    source = write_trajectory(tmp_path / "trajectory.extxyz", frames)

    assert convert_trajectory(Namespace(
        input=source, output=tmp_path / "trajectory.xyz", input_format=None,
        output_format=None, stride=2, json=True,
    )) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["frames"] == 2
    assert report["input_format"] == "extxyz"
    assert report["output_format"] == "xyz"
    assert len(read(str(tmp_path / "trajectory.xyz"), index=":")) == 2
