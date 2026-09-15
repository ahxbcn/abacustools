"""Tests for the IRI and promolecular weak-interaction analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.charge import Charge
from abacustools.data.weak import (
    IRI_EXPONENT,
    IRI_RHO_CUT,
    delta_g,
    iri,
    promolecular_fields,
    read_atomic_density,
)
from abacustools.io.stru import AbacusSTRU


BOHR2A = BOHR_TO_ANG
LENGTH = 8.0
NPOINTS = 40
VALENCE = 4.0
ALPHA = 0.5


def _gaussian_upf(element: str = "Si", valence: float = VALENCE, alpha: float = ALPHA) -> str:
    """Return a legacy UPF whose PP_RHOATOM is a normalized Gaussian."""
    step = 0.01
    radius = np.arange(0.0, 6.0 + step / 2, step)
    normal = valence * 4.0 * alpha**1.5 / np.sqrt(np.pi)
    radial = normal * radius**2 * np.exp(-alpha * radius**2)
    return f"""<PP_INFO>
3S  3  0  2.00  0.000000  1.20  -0.500000
</PP_INFO>
<PP_HEADER>
  1.0 Version Number
  {element} Element
  {valence:.11f} Z valence
  1 Max angular momentum component
  {radius.size} Number of points in mesh
  1 1 Number of Wavefunctions, Number of Projectors
</PP_HEADER>
<PP_MESH>
  <PP_R>
    {" ".join(f"{value:.6f}" for value in radius)}
  </PP_R>
  <PP_RAB>
    {" ".join(f"{step:.6f}" for _ in radius)}
  </PP_RAB>
</PP_MESH>
<PP_LOCAL>
  {" ".join("-1.0" for _ in radius)}
</PP_LOCAL>
<PP_NLCC>
  {" ".join("0.0" for _ in radius)}
</PP_NLCC>
<PP_NONLOCAL>
  <PP_BETA>
    1 0 Beta L
    2
    1.0 2.0
  </PP_BETA>
  <PP_DIJ>
    1 Number of nonzero Dij
    1 1 1.0
  </PP_DIJ>
</PP_NONLOCAL>
<PP_PSWFC>
3S  0  2.00  Wavefunction
  0.1 0.2
</PP_PSWFC>
<PP_RHOATOM>
  {" ".join(f"{value:.10e}" for value in radial)}
</PP_RHOATOM>
"""


def _job(tmp_path: Path, *, center=(4.0, 4.0, 4.0)) -> Path:
    """Create a one-atom job with a Gaussian pseudoatomic density."""
    job = tmp_path / "job"
    job.mkdir(parents=True, exist_ok=True)
    (job / "Si.upf").write_text(_gaussian_upf(), encoding="utf-8")
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 1\n", encoding="utf-8"
    )
    (job / "STRU").write_text(
        "ATOMIC_SPECIES\n"
        "Si 28.0855 Si.upf\n\n"
        "LATTICE_CONSTANT\n"
        "1.8897261254578281\n\n"
        "LATTICE_VECTORS\n"
        f"{LENGTH} 0 0\n0 {LENGTH} 0\n0 0 {LENGTH}\n\n"
        "ATOMIC_POSITIONS\nCartesian\n\n"
        "Si\n0.0\n1\n"
        f"{center[0]} {center[1]} {center[2]} 1 1 1\n",
        encoding="utf-8",
    )
    return job


def _grid(job: Path, values: np.ndarray | None = None) -> Charge:
    structure = AbacusSTRU.read(str(job / "STRU"))
    data = np.zeros((NPOINTS,) * 3) if values is None else values
    return Charge(
        data,
        np.diag([LENGTH] * 3),
        np.asarray(structure.coords, dtype=float),
        [14],
        [VALENCE],
    )


def test_atomic_density_integrates_to_the_valence_charge(tmp_path: Path) -> None:
    table = read_atomic_density(_write_upf(tmp_path))

    assert table.element == "Si"
    assert table.electron_count == pytest.approx(VALENCE, rel=1e-6)
    assert table.cutoff > 1.0
    # rho(r) = Z alpha^(3/2) / pi^(3/2) * exp(-alpha r^2) for this fixture; the
    # origin is taken from the nearest mesh point, which costs about 1e-5.
    center = VALENCE * ALPHA**1.5 / np.pi**1.5
    assert float(table.values(np.zeros(1))[0]) == pytest.approx(center, rel=1e-3)
    peak = 1.0 / np.sqrt(2.0 * ALPHA)
    expected = 2.0 * ALPHA * peak * center * np.exp(-ALPHA * peak**2)
    assert float(table.gradient_magnitudes(np.array([peak]))[0]) == pytest.approx(
        expected, rel=1e-3
    )


def _write_upf(tmp_path: Path) -> Path:
    path = tmp_path / "Si.upf"
    path.write_text(_gaussian_upf(), encoding="utf-8")
    return path


def test_promolecular_density_and_gradient_of_one_atom(tmp_path: Path) -> None:
    job = _job(tmp_path)
    structure = AbacusSTRU.read(str(job / "STRU"))

    rho, gradient = promolecular_fields(_grid(job), structure, job=job)

    volume = LENGTH**3
    element = volume / NPOINTS**3
    assert rho.shape == (NPOINTS,) * 3
    # The atomic density is truncated outside its cutoff, so the far corners of
    # the cell stay empty.
    assert np.all(rho >= 0.0)
    assert rho.max() > 0.0
    assert rho.sum() * element == pytest.approx(VALENCE, rel=0.03)
    # The atomic density is radially symmetric, so the analytic gradient
    # magnitudes must follow the numerical derivative of the same field.
    spacing = LENGTH / NPOINTS
    numerical = np.sqrt(
        sum(np.gradient(rho, spacing, axis=axis) ** 2 for axis in range(3))
    )
    significant = gradient > 0.2 * gradient.max()
    ratio = numerical[significant] / gradient[significant]
    assert float(np.median(ratio)) == pytest.approx(1.0, rel=0.05)


def test_delta_g_vanishes_for_the_promolecular_density(tmp_path: Path) -> None:
    job = _job(tmp_path)
    structure = AbacusSTRU.read(str(job / "STRU"))
    reference = _grid(job)
    rho, gradient = promolecular_fields(reference, structure, job=job)

    values = delta_g(Charge(rho, reference.cell), gradient)

    significant = gradient > 0.2 * gradient.max()
    assert float(np.median(values[significant])) <= 0.05 * float(
        np.median(gradient[significant])
    )
    assert np.all(values >= 0.0)


def test_delta_g_reports_the_difference_for_two_atoms(tmp_path: Path) -> None:
    job = _job(tmp_path)
    # A density that is not the sum of the atomic densities: delta g must show
    # the interaction region, that is a positive value between the atoms.
    structure = AbacusSTRU.read(str(job / "STRU"))
    reference = _grid(job)
    rho, gradient = promolecular_fields(reference, structure, job=job)
    shifted = Charge(rho, reference.cell)
    # A rigid shift of the density towards one side mimics the density change
    # of a bond, which the promolecular reference cannot follow.
    shifted.data = np.roll(rho, 2, axis=0)

    values = delta_g(shifted, gradient)

    assert values.max() > 0.0
    assert float(values.max()) < float(gradient.max())


def test_iri_matches_the_analytic_formula() -> None:
    shape = (32, 1, 1)
    offset, amplitude = 0.005, 0.002
    coordinate = np.linspace(0.0, LENGTH, shape[0], endpoint=False)
    values = offset + amplitude * np.cos(2.0 * np.pi * coordinate / LENGTH)
    density = Charge(
        (values[:, None, None] * np.ones(shape)),
        np.diag([LENGTH] * 3),
        np.array([[0.0, 0.0, 0.0]]),
        [14],
        [VALENCE],
    )

    indicator = iri(density)

    wave = 2.0 * np.pi / LENGTH
    rho = (offset + amplitude * np.cos(wave * coordinate)) * BOHR2A**3
    norm = BOHR2A**4 * amplitude * wave * np.sin(wave * coordinate)
    expected = np.abs(norm) / rho**IRI_EXPONENT
    np.testing.assert_allclose(indicator[:, 0, 0], expected, rtol=1e-6, atol=1e-12)


def test_iri_uses_the_fill_below_the_density_cut() -> None:
    tiny = 1.0e-7
    density = Charge(
        np.full((4, 4, 4), tiny),
        np.diag([LENGTH] * 3),
        np.array([[0.0, 0.0, 0.0]]),
        [14],
        [VALENCE],
    )
    assert tiny * BOHR2A**3 < IRI_RHO_CUT

    indicator = iri(density, fill=2.5)

    np.testing.assert_allclose(indicator, 2.5)
