"""Tests for the Berry phase polarization data layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacustools.data.polarization import (
    BERRY_LOGS,
    kpoint_mesh,
    polarization_cartesian,
    polarization_delta,
    read_berry_polarization,
    read_task_polarization,
    task_metrics,
)
from abacustools.io.stru import AbacusSTRU


def _structure(cell: float = 4.0) -> AbacusSTRU:
    from abacustools.io.stru import AbacusATOM

    return AbacusSTRU(
        cell=[
            [cell, 0.0, 0.0],
            [0.0, cell, 0.0],
            [0.0, 0.0, cell],
        ],
        atoms=[AbacusATOM(label="H", element="H", coord=(0.0, 0.0, 0.0))],
        metadata={"atom_type": "cartesian"},
    )


def test_polarization_cartesian_maps_lattice_components() -> None:
    # An orthorhombic cell: each lattice component maps onto its own axis.
    cell = [[2.0, 0, 0], [0, 4.0, 0], [0, 0, 8.0]]

    assert polarization_cartesian([1.0, 1.0, 1.0], cell) == [1.0, 1.0, 1.0]


def test_polarization_cartesian_follows_lattice_directions() -> None:
    # A sheared cell tilts the second lattice vector away from y.
    cell = [[2.0, 0, 0], [2.0, 2.0, 0], [0, 0, 2.0]]
    result = polarization_cartesian([0.0, 1.0, 0.0], cell)

    assert result[0] == pytest.approx(1.0 / np.sqrt(2.0))
    assert result[1] == pytest.approx(1.0 / np.sqrt(2.0))
    assert result[2] == pytest.approx(0.0)


def test_polarization_cartesian_rejects_a_bad_cell() -> None:
    with pytest.raises(ValueError, match="three valid vectors"):
        polarization_cartesian([1.0, 1.0, 1.0], [[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="three valid vectors"):
        polarization_cartesian([1.0, 1.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_polarization_delta_wraps_a_crossing() -> None:
    delta = polarization_delta([0.9, 0.0, 0.0], [1.1, 0.0, 0.0], [4.0, 4.0, 4.0])

    assert delta[0] == pytest.approx(0.2)
    assert delta[1:] == [0.0, 0.0]


def test_polarization_delta_wraps_the_far_side() -> None:
    delta = polarization_delta([3.9, 0.0, 0.0], [0.1, 0.0, 0.0], [4.0, 4.0, 4.0])

    assert delta[0] == pytest.approx(0.2)


def test_polarization_delta_leaves_a_zero_quantum_alone() -> None:
    # A zero quantum marks a direction that must not be wrapped.
    delta = polarization_delta([0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 4.0, 4.0])

    assert delta == [5.0, 0.0, 0.0]


def test_polarization_delta_rejects_a_bad_shape() -> None:
    with pytest.raises(ValueError, match="three values"):
        polarization_delta([0.0, 0.0], [0.0, 0.0], [1.0, 1.0, 1.0])


def test_read_berry_polarization_parses_a_log(tmp_path: Path) -> None:
    log = tmp_path / "running_nscf1.log"
    log.write_text(
        "Volume (A^3) = 64.0\n"
        "The calculated polarization direction is in R1 direction\n"
        "P = 0.01 (mod 0.02) (0.0 0.0 0.0) (e/Omega).bohr\n"
        "P = 0.05 (mod 1.0) (0.0 0.0 0.0) C/m^2\n",
        encoding="utf-8",
    )

    values = read_berry_polarization(log)

    assert values["direction"] == 1
    assert values["p_vec"] == pytest.approx(0.01 * 0.529177249)
    assert values["mod"] == pytest.approx(0.02 * 0.529177249)
    assert values["polarization_cm2"] == pytest.approx(0.05)
    assert values["volume"] == pytest.approx(64.0)


def test_read_berry_polarization_requires_a_polarization(tmp_path: Path) -> None:
    log = tmp_path / "running_nscf1.log"
    log.write_text("Volume (A^3) = 64.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        read_berry_polarization(log)


def test_read_task_polarization_reads_all_three_directions(tmp_path: Path) -> None:
    for index, name in enumerate(BERRY_LOGS, start=1):
        output = tmp_path / "OUT.ABACUS"
        output.mkdir(exist_ok=True)
        (output / name).write_text(
            "Volume (A^3) = 64.0\n"
            f"The calculated polarization direction is in R{index} direction\n"
            "P = 0.01 (mod 0.02) (0.0 0.0 0.0) (e/Omega).bohr\n"
            "P = 0.05 (mod 1.0) (0.0 0.0 0.0) C/m^2\n",
            encoding="utf-8",
        )

    values = read_task_polarization(tmp_path, "ABACUS")

    assert values is not None
    assert values["directions"] == [1, 2, 3]
    assert len(values["p_vec"]) == 3
    assert values["volume"] == pytest.approx(64.0)


def test_read_task_polarization_skips_an_incomplete_task(tmp_path: Path, capsys) -> None:
    # Only two of the three directions exist.
    output = tmp_path / "OUT.ABACUS"
    output.mkdir()
    (output / BERRY_LOGS[0]).write_text(
        "Volume (A^3) = 64.0\n"
        "The calculated polarization direction is in R1 direction\n"
        "P = 0.01 (mod 0.02) (0.0 0.0 0.0) (e/Omega).bohr\n",
        encoding="utf-8",
    )

    assert read_task_polarization(tmp_path, "ABACUS") is None
    assert "skipping incomplete BEC task" in capsys.readouterr().out


def test_kpoint_mesh_prefers_gamma_only(tmp_path: Path) -> None:
    values, model = kpoint_mesh(Path("."), {"gamma_only": 1}, _structure())

    assert values == [1, 1, 1, 0.0, 0.0, 0.0]
    assert model == "gamma"


def test_kpoint_mesh_expands_a_kspacing() -> None:
    values, model = kpoint_mesh(Path("."), {"kspacing": 0.2}, _structure(8.0))

    assert model == "gamma"
    assert all(value >= 1 for value in values[:3])


def test_kpoint_mesh_reads_an_explicit_kpt_file(tmp_path: Path) -> None:
    (tmp_path / "KPT").write_text(
        "K_POINTS\n0\nGamma\n4 4 4 0 0 0\n", encoding="utf-8"
    )

    values, model = kpoint_mesh(tmp_path, {}, _structure())

    assert values == [4.0, 4.0, 4.0, 0.0, 0.0, 0.0]
    assert model == "gamma"


def test_task_metrics_reports_a_missing_job_instead_of_raising(tmp_path: Path) -> None:
    # An unreadable job yields empty metrics; the caller decides what to do.
    metrics = task_metrics(tmp_path / "absent", "")

    assert metrics["energy"] is None
    assert metrics["converged"] is None
