"""Tests for the shared periodic-phonon data layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacustools.data.phonon import (
    automatic_supercell,
    collect_forces,
    displacement_task,
    displacement_tasks,
    jsonable,
    moved_mode_indices,
    read_forces,
    validate_displacement_entries,
    validate_mesh,
    validate_positive_float,
    validate_supercell,
)
from abacustools.io.stru import AbacusATOM, AbacusSTRU


def _write_job(job: Path, natoms: int, *, converged: bool = True) -> Path:
    """Write a minimal ABACUS job holding a force block.

    Args:
        job: Directory to create.
        natoms: Number of atoms of the force block.
        converged: Whether the log reports an achieved convergence.

    Returns:
        The job directory.
    """
    job.mkdir(parents=True, exist_ok=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nscf_thr 1e-6\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(
        "ATOMIC_SPECIES\nH 1.0\n\n"
        "LATTICE_CONSTANT\n1.0\n\n"
        "LATTICE_VECTORS\n3 0 0\n0 3 0\n0 0 3\n\n"
        "ATOMIC_POSITIONS\nCartesian\n\nH\n0.0\n1\n0 0 0\n",
        encoding="utf-8",
    )
    output = job / "OUT.ABACUS"
    output.mkdir(exist_ok=True)
    lines = ["E_KohnSham = -1.000000 eV\n", "density error = 1e-9\n"]
    if converged:
        lines.append("charge density convergence is achieved\n")
    lines += [
        "#TOTAL-FORCE (eV/Angstrom)\n",
        "-" * 30 + "\n",
        "  Atoms  Force_x  Force_y  Force_z\n",
        "-" * 30 + "\n",
    ]
    lines += [
        f"  H{index + 1}  {0.1 * (index + 1):.7f}  {0.2 * (index + 1):.7f}  {0.3 * (index + 1):.7f}\n"
        for index in range(natoms)
    ]
    lines += ["-" * 30 + "\n", "Total  Time  : 0 h 0 mins 1 secs\n"]
    (output / "running_scf.log").write_text("".join(lines), encoding="utf-8")
    return job


def _structure(cell: str = "3 0 0\n0 4 0\n0 0 12") -> AbacusSTRU:
    """Return a structure with a controllable cell."""
    rows = [[float(value) for value in line.split()] for line in cell.splitlines()]
    return AbacusSTRU(
        cell=rows,
        atoms=[AbacusATOM(label="H", element="H", coord=(0.0, 0.0, 0.0))],
        metadata={"atom_type": "cartesian"},
    )


def test_validate_positive_float() -> None:
    assert validate_positive_float(0.01, "step") == pytest.approx(0.01)
    assert validate_positive_float(0.0, "temperature", allow_zero=True) == 0.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="must be a positive finite number"):
            validate_positive_float(bad, "step")
    with pytest.raises(ValueError):
        validate_positive_float(-1.0, "temperature", allow_zero=True)


def test_validate_supercell_and_mesh_reject_bad_values() -> None:
    assert validate_supercell([1, 2, 3]) == [1, 2, 3]
    assert validate_mesh([2, 3, 4]) == [2, 3, 4]
    for bad in ([1, 0, 2], [1, 2], [1, 2, 3, 4], [1.5, 2, 2], [True, 2, 2]):
        with pytest.raises(ValueError, match="supercell"):
            validate_supercell(bad)
    for bad in ([1, 2, 0], [1, 2]):
        with pytest.raises(ValueError, match="mesh"):
            validate_mesh(bad)
    with pytest.raises(ValueError, match="must be specified"):
        validate_supercell(None)


def test_automatic_supercell_follows_the_lattice_lengths() -> None:
    assert automatic_supercell(_structure(), 10.0) == [4, 3, 1]
    # A single repetition is kept when the cell is already long enough.
    assert automatic_supercell(_structure("20 0 0\n0 20 0\n0 0 20"), 10.0) == [1, 1, 1]


def test_automatic_supercell_rejects_a_degenerate_cell() -> None:
    with pytest.raises(ValueError, match="finite, non-zero"):
        automatic_supercell(_structure("0 0 0\n0 4 0\n0 0 12"), 10.0)


def test_displacement_task_pads_the_index() -> None:
    assert displacement_task("disp-", 0) == {"task": "disp-0000", "index": 0}
    assert displacement_task("fc3-", 12) == {"task": "fc3-0012", "index": 12}


def test_displacement_tasks_keep_dataset_indices_and_skip_gaps() -> None:
    tasks = displacement_tasks([object(), None, object()], "disp-")

    assert tasks == [
        {"task": "disp-0000", "index": 0},
        {"task": "disp-0002", "index": 2},
    ]


def test_validate_displacement_entries_accepts_extra_keys() -> None:
    entries = validate_displacement_entries(
        [
            {"task": "disp-0000", "index": 0, "atom": 3, "displacement": [0.01, 0, 0]},
            {"task": "disp-0001", "index": 1},
        ],
        2,
        "phonon workflow",
    )

    assert [entry["index"] for entry in entries] == [0, 1]
    assert entries[0]["atom"] == 3  # carried through for reporting


def test_validate_displacement_entries_rejects_bad_input() -> None:
    for bad, message in (
        (None, "no displacement tasks"),
        ([], "no displacement tasks"),
        ([{"index": 0}], "invalid displacement entry"),
        ([{"task": "t"}], "invalid displacement index"),
        ([{"task": "t", "index": "x"}], "invalid displacement index"),
        ([{"task": "t", "index": 5}], "out of range"),
        ([{"task": "t", "index": -1}], "out of range"),
        ([{"task": "t", "index": 0}, {"task": "t", "index": 1}], "twice"),
    ):
        with pytest.raises(RuntimeError, match=message):
            validate_displacement_entries(bad, 2, "phonon workflow")


def test_collect_forces_maps_by_index_and_keeps_gaps(tmp_path: Path) -> None:
    supercells = [object(), None, object()]
    entries = [
        {"task": "d0", "index": 0},
        {"task": "d2", "index": 2},
    ]
    force = np.array([[0.1, 0.2, 0.3]])

    import abacustools.data.phonon as module

    calls = []

    def fake_read_forces(job, version, natoms):
        calls.append((job.name, version, natoms))
        return force

    original = module.read_forces
    module.read_forces = fake_read_forces
    try:
        forces = collect_forces(
            tmp_path, entries, supercells, "auto", 1, workflow="phonon"
        )
    finally:
        module.read_forces = original

    assert calls == [("d0", "auto", 1), ("d2", "auto", 1)]
    assert forces[1] is None  # the symmetry gap stays a gap
    np.testing.assert_allclose(forces[0], force)
    np.testing.assert_allclose(forces[2], force)


def test_collect_forces_reports_a_missing_displacement(tmp_path: Path) -> None:
    import abacustools.data.phonon as module

    original = module.read_forces
    module.read_forces = lambda job, version, natoms: np.zeros((1, 3))
    try:
        with pytest.raises(RuntimeError, match="no forces were collected"):
            collect_forces(
                tmp_path,
                [{"task": "d0", "index": 0}],
                [object(), object()],
                "auto",
                1,
                workflow="phonon",
            )
    finally:
        module.read_forces = original


def test_read_forces_reads_a_converged_force_block(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job", 2)

    forces = read_forces(job, "", 2)

    np.testing.assert_allclose(
        forces, [[0.10, 0.20, 0.30], [0.20, 0.40, 0.60]], atol=1e-6
    )


def test_read_forces_requires_a_converged_calculation(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job", 2, converged=False)

    with pytest.raises(RuntimeError, match="did not converge"):
        read_forces(job, "", 2)


def test_read_forces_requires_a_matching_atom_count(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job", 2)

    with pytest.raises(RuntimeError, match="invalid force array"):
        read_forces(job, "", 3)


def test_jsonable_converts_nested_numpy() -> None:
    value = {
        "array": np.arange(3),
        "scalar": np.float64(1.5),
        "nested": [np.int64(2), {"inner": np.zeros((2, 2))}],
        "plain": "text",
    }

    converted = jsonable(value)

    assert converted == {
        "array": [0, 1, 2],
        "scalar": 1.5,
        "nested": [2, {"inner": [[0.0, 0.0], [0.0, 0.0]]}],
        "plain": "text",
    }


def test_moved_mode_indices_finds_nothing_in_an_unchanged_spectrum() -> None:
    reference = [0.0, 0.0, 0.0, 8.0, 8.26138, 8.39169]

    assert moved_mode_indices(reference, reference) == []


def test_moved_mode_indices_finds_the_mode_a_correction_raised() -> None:
    reference = [0.0, 0.0, 0.0, 8.0, 8.26138, 8.39169]
    corrected = [0.0, 0.0, 0.0, 8.26138, 8.39169, 10.24492]

    # Band by band this looks as if three modes had moved; as sets only the one
    # that has no partner in the reference spectrum did.
    assert moved_mode_indices(corrected, reference) == [5]


def test_moved_mode_indices_survives_degenerate_pairs() -> None:
    """A doubly degenerate mode splits into one that moves and one that stays."""
    reference = [0.0, 0.0, 0.0, 8.00148, 8.26138, 8.26138, 8.39169, 8.39169]
    corrected = [0.0, 0.0, 0.0, 8.00148, 8.26138, 8.39169, 8.39169, 10.26734]

    assert moved_mode_indices(corrected, reference) == [7]


def test_moved_mode_indices_reproduce_the_wurtzite_gamma_point() -> None:
    """The hexagonal ZnS case that exposed the band by band comparison.

    A self consistent calculation along the c axis raises the A1 mode from
    8.00148 THz to 10.24492 THz.  Comparing the two spectra band by band marks
    the two E1 modes, the two E2 modes and the B1 mode as longitudinal as well,
    because the raised mode climbs above all of them.
    """
    reference = [
        0.0, 0.0, 0.0, 2.06995, 2.06995, 5.87288, 8.00148, 8.26138, 8.26138,
        8.39169, 8.39169, 9.77211,
    ]
    corrected = [
        0.0, 0.0, 0.0, 2.06995, 2.06995, 5.87288, 8.26138, 8.26138, 8.39169,
        8.39169, 9.77211, 10.24492,
    ]

    assert moved_mode_indices(corrected, reference) == [11]


def test_moved_mode_indices_rejects_spectra_of_different_size() -> None:
    with pytest.raises(ValueError, match="same number of modes"):
        moved_mode_indices([0.0, 1.0], [0.0])
