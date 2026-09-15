"""Tests for job-level charge-density assembly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.charge import (
    ChargeDensityError,
    DensitySource,
    JobDensity,
    atoms_in_plane,
    combine,
    cube_paths,
    find_density_source,
    integrate,
    planar_profile,
    read_cube_charges,
    read_job_total_density,
    read_restart_charges,
    select_spin,
    slice_plane,
    subtract,
    total_charge,
    validate_same_grid,
)
from abacustools.data.grid import Charge, Grid, RestartCharge
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


BOHR2A = BOHR_TO_ANG

STRU = """\
ATOMIC_SPECIES
Si 28.0855 Si.upf

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
4 0 0
0 4 0
0 0 4

ATOMIC_POSITIONS
Cartesian

Si
0.0
2
0.0 0.0 0.0 1 1 1
2.0 2.0 2.0 1 1 1
"""


def _job(base: Path, *, nspin: int = 1, suffix: str = "ABACUS") -> tuple[Path, Path]:
    """Create a minimal job directory with an INPUT and return it with OUT.*."""
    job = base / "job"
    output = job / f"OUT.{suffix}"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        f"INPUT_PARAMETERS\nsuffix {suffix}\nnspin {nspin}\n", encoding="utf-8"
    )
    return job, output


def _charge(data, *, cell=None, origin=None, charges=(4.0, 4.0)) -> Charge:
    return Charge(
        np.asarray(data, dtype=float),
        np.diag([4.0, 4.0, 4.0]) if cell is None else np.asarray(cell, dtype=float),
        np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        [14, 14],
        list(charges),
        np.zeros(3) if origin is None else np.asarray(origin, dtype=float),
    )


def _write_cube(path: Path, data, **kwargs) -> Charge:
    charge = _charge(data, **kwargs)
    charge.save_cube(str(path))
    return charge


def _write_restart(
    path: Path, *, nspin: int = 1, shape: tuple[int, int, int] = (2, 2, 2)
) -> Path:
    """Write a small restart file whose reciprocal and real spaces are related."""
    fractions = [np.fft.fftfreq(count) * count for count in shape]
    mesh = np.meshgrid(*fractions, indexing="ij")
    miller = np.stack([entry.ravel() for entry in mesh], axis=1).astype(np.int64)
    reciprocal = np.linalg.inv(np.diag([4.0, 4.0, 4.0]))
    rng = np.random.default_rng(0)
    rhog = rng.standard_normal((nspin, miller.shape[0])) + 1j * np.random.default_rng(
        1
    ).standard_normal((nspin, miller.shape[0]))
    RestartCharge(rhog, miller, reciprocal).write(str(path))
    return path


def test_cubes_keep_the_abacus_units_through_the_charge_class(tmp_path: Path) -> None:
    data = np.linspace(0.5, 1.5, 8).reshape(2, 2, 2)
    path = tmp_path / "SPIN1_CHG.cube"
    written = _write_cube(path, data)

    # The cube writer keeps 12 significant digits per number.
    raw = Grid.from_cube(str(path))
    np.testing.assert_allclose(raw.data, data * BOHR2A**3, rtol=1e-10)
    np.testing.assert_allclose(raw.cell, written.cell / BOHR2A, rtol=1e-10)

    roundtripped = Charge.from_cube(str(path), format="abacus")
    np.testing.assert_allclose(roundtripped.data, data, rtol=1e-10)
    np.testing.assert_allclose(roundtripped.cell, written.cell, rtol=1e-10)


def test_find_density_source_prefers_cube_files(tmp_path: Path) -> None:
    job, output = _job(tmp_path, nspin=2)
    _write_cube(output / "SPIN1_CHG.cube", np.ones((2, 2, 2)))
    _write_cube(output / "SPIN2_CHG.cube", np.ones((2, 2, 2)) * 0.5)
    (output / "ABACUS-CHARGE-DENSITY.restart").write_bytes(b"unused")

    source = find_density_source(job, ReadInput(str(job / "INPUT")))

    assert source.kind == "cube"
    assert [path.name for path in source.paths] == ["SPIN1_CHG.cube", "SPIN2_CHG.cube"]
    assert source.describe() == "cube"


def test_explicit_cube_directory_overrides_the_output_directory(tmp_path: Path) -> None:
    job, output = _job(tmp_path, nspin=1)
    stored = tmp_path / "stored"
    stored.mkdir()
    _write_cube(stored / "SPIN1_CHG.cube", np.ones((2, 2, 2)) * 3.0)
    (output / "SPIN1_CHG.cube").write_text("ignored", encoding="utf-8")
    inputs = ReadInput(str(job / "INPUT"))

    directory = find_density_source(job, inputs, cube=str(stored))
    assert directory.paths == [stored / "SPIN1_CHG.cube"]

    single = find_density_source(job, inputs, cube="OUT.ABACUS/SPIN1_CHG.cube")
    assert single.paths == [job / "OUT.ABACUS/SPIN1_CHG.cube"]

    empty = tmp_path / "empty"
    empty.mkdir()
    assert cube_paths(job, output, 1, cube=str(empty)) is None


def test_find_density_source_falls_back_to_the_restart_file(tmp_path: Path) -> None:
    job, output = _job(tmp_path, nspin=1)
    restart = _write_restart(output / "ABACUS-CHARGE-DENSITY.restart")

    source = find_density_source(job, ReadInput(str(job / "INPUT")))

    assert source.kind == "restart"
    assert source.paths == [restart]
    assert source.describe((2, 2, 2)) == (
        "restart (ABACUS-CHARGE-DENSITY.restart, grid=(2, 2, 2))"
    )


def test_find_density_source_reports_a_missing_density(tmp_path: Path) -> None:
    job, _ = _job(tmp_path, nspin=1)
    with pytest.raises(ChargeDensityError, match="no charge density"):
        find_density_source(job, ReadInput(str(job / "INPUT")))


def test_read_cube_charges_checks_the_channel_count(tmp_path: Path) -> None:
    job, output = _job(tmp_path, nspin=2)
    _write_cube(output / "SPIN1_CHG.cube", np.ones((2, 2, 2)))
    inputs = ReadInput(str(job / "INPUT"))
    source = find_density_source(job, inputs, cube="OUT.ABACUS/SPIN1_CHG.cube")

    with pytest.raises(ChargeDensityError, match="found 1 spin channel"):
        read_cube_charges(source)


def test_read_restart_charges_uses_the_structure_and_converts_units(tmp_path: Path) -> None:
    job, output = _job(tmp_path, nspin=1)
    (job / "STRU").write_text(STRU, encoding="utf-8")
    structure = AbacusSTRU.read(str(job / "STRU"))
    restart = _write_restart(output / "ABACUS-CHARGE-DENSITY.restart")
    source = find_density_source(job, ReadInput(str(job / "INPUT")))

    charges = read_restart_charges(
        source,
        structure=structure,
        valences=[4.0, 4.0],
        grid_shape=(2, 2, 2),
        lat0=1.889726,
    )

    expected = RestartCharge.read(str(restart)).to_real((2, 2, 2))[0]
    assert len(charges) == 1
    np.testing.assert_allclose(charges[0].data, expected / BOHR2A**3, rtol=1e-12)
    np.testing.assert_allclose(charges[0].cell, np.diag([4.0, 4.0, 4.0]), rtol=1e-6)
    assert list(charges[0].atom_types) == [14, 14]
    assert list(charges[0].atom_charges) == [4.0, 4.0]


def test_read_restart_charges_rejects_noncollinear_channels(tmp_path: Path) -> None:
    job, output = _job(tmp_path, nspin=4)
    (job / "STRU").write_text(STRU, encoding="utf-8")
    _write_restart(output / "ABACUS-CHARGE-DENSITY.restart", nspin=4)
    source = find_density_source(job, ReadInput(str(job / "INPUT")))

    with pytest.raises(ChargeDensityError, match="nspin=4 is not supported"):
        read_restart_charges(
            source,
            structure=AbacusSTRU.read(str(job / "STRU")),
            valences=[4.0, 4.0],
            grid_shape=(2, 2, 2),
        )


def test_combine_and_total_charge_do_not_mutate_their_inputs() -> None:
    first = _charge(np.ones((2, 2, 2)))
    second = _charge(np.full((2, 2, 2), 2.0))

    np.testing.assert_allclose(combine(first, second, 1.0).data, 3.0)
    np.testing.assert_allclose(combine(first, second, -1.0).data, -1.0)
    np.testing.assert_allclose(total_charge([first, second]).data, 3.0)
    np.testing.assert_allclose(first.data, 1.0)


def test_grid_checks_reject_incompatible_densities() -> None:
    first = _charge(np.ones((2, 2, 2)))

    with pytest.raises(ChargeDensityError, match="incompatible grid shape"):
        validate_same_grid(first, _charge(np.ones((2, 2, 3))), "test")
    with pytest.raises(ChargeDensityError, match="incompatible grid geometry"):
        validate_same_grid(first, _charge(np.ones((2, 2, 2)), cell=np.diag([5.0] * 3)), "test")
    with pytest.raises(ChargeDensityError, match="no charge-density channel"):
        total_charge([])


def test_read_job_total_density_sums_the_spin_channels(tmp_path: Path) -> None:
    job, output = _job(tmp_path, nspin=2)
    _write_cube(output / "SPIN1_CHG.cube", np.full((2, 2, 2), 1.0))
    _write_cube(output / "SPIN2_CHG.cube", np.full((2, 2, 2), 0.5))

    total = read_job_total_density(job)

    np.testing.assert_allclose(total.data, 1.5)


def test_read_job_total_density_rejects_unusable_jobs(tmp_path: Path) -> None:
    noncollinear, _ = _job(tmp_path / "noncollinear", nspin=4)
    with pytest.raises(ChargeDensityError, match="nspin 1 and 2"):
        read_job_total_density(noncollinear)

    restart_only, output = _job(tmp_path / "restart_only", nspin=1)
    _write_restart(output / "ABACUS-CHARGE-DENSITY.restart")
    with pytest.raises(ChargeDensityError, match=r"SPIN\*_CHG.cube"):
        read_job_total_density(restart_only)


def test_read_job_total_density_can_require_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, output = _job(tmp_path, nspin=1)
    _write_cube(output / "SPIN1_CHG.cube", np.ones((2, 2, 2)))
    monkeypatch.setattr(
        "abacustools.data.charge.get_result_from_job",
        lambda *args, **kwargs: {"converged": False},
    )

    with pytest.raises(ChargeDensityError, match="did not converge"):
        read_job_total_density(job, require_converged=True)


def test_integrate_reports_the_electrons_and_the_valence_check() -> None:
    density = _charge(np.full((2, 2, 2), 0.5))

    report = integrate(density)

    assert report["grid"] == [2, 2, 2]
    assert report["volume_angstrom3"] == pytest.approx(64.0)
    assert report["electrons"] == pytest.approx(0.5 * 64.0)
    assert report["valence_electrons"] == pytest.approx(8.0)
    assert report["deviation"] == pytest.approx(0.5 * 64.0 - 8.0)


def test_integrate_skips_the_valence_check_without_atom_charges() -> None:
    density = _charge(np.full((2, 2, 2), 0.5), charges=(0.0, 0.0))

    report = integrate(density)

    assert report["valence_electrons"] is None
    assert report["deviation"] is None
    assert report["electrons"] == pytest.approx(0.5 * 64.0)


def test_planar_profile_average_and_integral_agree() -> None:
    # The density varies along c only, so the profile reproduces the values.
    values = np.array([1.0, 2.0, 3.0, 4.0])
    density = _charge(np.broadcast_to(values[None, None, :], (2, 2, 4)).copy())

    average, distances = planar_profile(density, "c", kind="average")
    integral, _ = planar_profile(density, "c", kind="integral")

    np.testing.assert_allclose(average, values)
    # A 64 Angstrom**3 cell with 16 grid points has 4 Angstrom**3 per point, so
    # the four points of one plane sum to 16 Angstrom**3 of charge.
    np.testing.assert_allclose(integral, values * 16.0)
    assert integral.sum() == pytest.approx(float(density.data.sum()) * 4.0)
    np.testing.assert_allclose(distances, np.linspace(0.0, 4.0, 4))


def test_planar_profile_validates_its_arguments() -> None:
    density = _charge(np.ones((2, 2, 2)))

    with pytest.raises(ChargeDensityError, match="unknown profile axis"):
        planar_profile(density, "d")
    with pytest.raises(ChargeDensityError, match="unknown profile kind"):
        planar_profile(density, "c", kind="density")


def _density(channels) -> JobDensity:
    charges = [_charge(channel) for channel in channels]
    source = DensitySource("cube", [Path(f"SPIN{index + 1}_CHG.cube") for index in range(len(charges))], len(charges))
    return JobDensity(Path("job"), source, charges)


def test_select_spin_returns_channels_and_their_difference() -> None:
    density = _density([np.full((2, 2, 2), 0.75), np.full((2, 2, 2), 0.25)])

    np.testing.assert_allclose(select_spin(density, "total").data, 1.0)
    np.testing.assert_allclose(select_spin(density, "up").data, 0.75)
    np.testing.assert_allclose(select_spin(density, "down").data, 0.25)
    np.testing.assert_allclose(select_spin(density, "difference").data, 0.5)


def test_select_spin_validates_its_arguments() -> None:
    single = _density([np.ones((2, 2, 2))])

    with pytest.raises(ChargeDensityError, match="unknown spin choice"):
        select_spin(single, "sideways")
    with pytest.raises(ChargeDensityError, match="needs an nspin 2 calculation"):
        select_spin(single, "difference")
    with pytest.raises(ChargeDensityError, match="needs an nspin 2 calculation"):
        select_spin(single, "up")


def test_subtract_checks_the_grid() -> None:
    first = _charge(np.full((2, 2, 2), 3.0))
    second = _charge(np.full((2, 2, 2), 1.0))

    np.testing.assert_allclose(subtract(first, second, "the two jobs").data, 2.0)
    with pytest.raises(ChargeDensityError, match="incompatible grid shape for the two jobs"):
        subtract(first, _charge(np.ones((3, 3, 3))), "the two jobs")


def test_slice_plane_takes_the_closest_grid_plane() -> None:
    values = np.arange(1.0, 25.0).reshape(2, 3, 4)
    density = _charge(values)

    plane = slice_plane(density, "c", position=0.6)

    assert plane.index == 2
    assert plane.position == pytest.approx(0.5)
    assert plane.distance == pytest.approx(2.0)
    np.testing.assert_allclose(plane.values, values[:, :, 2])
    assert plane.labels == ("a", "b")
    np.testing.assert_allclose(plane.coordinates[0], np.linspace(0.0, 4.0, 2, endpoint=False))
    np.testing.assert_allclose(plane.coordinates[1], np.linspace(0.0, 4.0, 3, endpoint=False))


def test_slice_plane_validates_its_arguments() -> None:
    density = _charge(np.ones((2, 2, 2)))

    with pytest.raises(ChargeDensityError, match="unknown slice axis"):
        slice_plane(density, "d")
    with pytest.raises(ChargeDensityError, match="fractional coordinate"):
        slice_plane(density, "c", position=1.0)


def test_atoms_in_plane_marks_the_crossed_atoms(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "STRU").write_text(STRU, encoding="utf-8")
    structure = AbacusSTRU.read(str(job / "STRU"))
    density = _charge(np.ones((2, 2, 2)))

    plane = slice_plane(density, "c", position=0.5)
    atoms = atoms_in_plane(density, plane, structure)

    assert [atom["label"] for atom in atoms] == ["Si"]
    np.testing.assert_allclose(atoms[0]["coordinates"], [2.0, 2.0])
    assert atoms[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_atoms_in_plane_wraps_across_the_boundary(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "STRU").write_text(
        STRU.replace("2.0 2.0 2.0 1 1 1", "0.0 0.0 3.96 1 1 1"), encoding="utf-8"
    )
    structure = AbacusSTRU.read(str(job / "STRU"))
    density = _charge(np.ones((4, 4, 4)))

    plane = slice_plane(density, "c", position=0.01)
    atoms = atoms_in_plane(density, plane, structure)

    # A 4x4x4 grid snaps the requested position 0.01 onto the plane at 0.0. The
    # atom at 0.99 still crosses that plane once the periodic image is used,
    # which is one grid step away from it.
    assert plane.index == 0
    assert [atom["label"] for atom in atoms] == ["Si", "Si"]
    distances = sorted(atom["distance"] for atom in atoms)
    assert distances[0] == pytest.approx(-0.04, abs=1e-6)
    assert distances[1] == pytest.approx(0.0, abs=1e-6)
