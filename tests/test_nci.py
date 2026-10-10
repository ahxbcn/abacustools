"""Tests for the reduced density gradient, Hessian and DORI analyses."""

from __future__ import annotations

import numpy as np
import pytest

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.grid import Charge
from abacustools.data.nci import (
    DORI_REGULARIZER,
    RDG_GRADIENT_CUT,
    analyse,
    density_derivatives,
    dori,
    hessian_eigenvalues,
    nci_scatter_data,
    reduced_density_gradient,
    signed_density_hessian,
)


BOHR2A = BOHR_TO_ANG
LENGTH = 6.0


def _charge(values, *, cell=None) -> Charge:
    values = np.asarray(values, dtype=float)
    cell = np.diag([LENGTH] * 3) if cell is None else np.asarray(cell, dtype=float)
    return Charge(
        values,
        cell,
        np.array([[0.0, 0.0, 0.0]]),
        [14],
        [4.0],
    )


def _cosine_density(shape=(32, 1, 1), amplitude=0.002, offset=0.005) -> Charge:
    """Density that varies along a only: offset + amplitude * cos(2 pi a / L)."""
    coordinate = np.linspace(0.0, LENGTH, shape[0], endpoint=False)
    values = offset + amplitude * np.cos(2.0 * np.pi * coordinate / LENGTH)
    return _charge(values[:, None, None] * np.ones(shape))


def test_derivatives_match_the_analytic_gradient_and_hessian() -> None:
    # rho = A + sum of cosines on a cubic cell, whose derivatives are known.
    shape = (24, 24, 24)
    axes = [np.linspace(0.0, LENGTH, count, endpoint=False) for count in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    terms = [
        (0.004, np.array([1.0, 0.0, 0.0])),
        (0.003, np.array([0.0, 2.0, 0.0])),
        (0.002, np.array([1.0, 0.0, 1.0])),
    ]
    values = 0.02 + sum(
        amplitude * np.cos(2 * np.pi * (g[0] * x + g[1] * y + g[2] * z) / LENGTH)
        for amplitude, g in terms
    )
    density = _charge(values)

    gradient, hessian = density_derivatives(density)

    length_au = LENGTH / BOHR2A
    expected_gradient = np.zeros(shape + (3,))
    expected_hessian = np.zeros(shape + (3, 3))
    for amplitude, g in terms:
        wave = 2.0 * np.pi * g / length_au
        phase = 2.0 * np.pi * (g[0] * x + g[1] * y + g[2] * z) / LENGTH
        amplitude_au = amplitude * BOHR2A**3
        expected_gradient += -amplitude_au * np.sin(phase)[..., None] * wave
        expected_hessian += (
            -amplitude_au * np.cos(phase)[..., None, None] * np.outer(wave, wave)
        )
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-9, atol=1e-12)


def test_derivatives_follow_a_triclinic_cell() -> None:
    # The same density on a sheared lattice: the Cartesian gradient has to
    # follow the sheared reciprocal vectors, which a finite difference of the
    # analytic function checks directly.
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 5.0, 0.0], [0.5, 0.5, 6.0]])
    shape = (20, 20, 20)
    fractional = [
        np.linspace(0.0, 1.0, count, endpoint=False) for count in shape
    ]
    f1, f2, f3 = np.meshgrid(*fractional, indexing="ij")
    values = 0.03 + 0.004 * np.cos(2 * np.pi * (f1 + 2 * f2 - f3))
    density = _charge(values, cell=cell)

    step = 1.0e-4
    gradient, _ = density_derivatives(density)
    for axis in range(3):
        direction = np.zeros(3)
        direction[axis] = 1.0
        plus = _sheared_density(f1, f2, f3, cell, direction * step)
        minus = _sheared_density(f1, f2, f3, cell, -direction * step)
        # d/dx_Bohr of a density in e/Bohr^3 from a slope in e/Angstrom^3/Angstrom
        expected = (plus - minus) / (2.0 * step) * BOHR2A**4
        np.testing.assert_allclose(
            gradient[..., axis], expected, rtol=1e-4, atol=1e-10
        )


def _sheared_density(f1, f2, f3, cell, offset):
    """Evaluate the test density of the sheared cell displaced by ``offset``."""
    cartesian = np.stack([f1, f2, f3], axis=-1) @ cell + offset
    fractional = cartesian @ np.linalg.inv(cell)
    return 0.03 + 0.004 * np.cos(
        2 * np.pi * (fractional[..., 0] + 2 * fractional[..., 1] - fractional[..., 2])
    )


def test_reduced_density_gradient_matches_the_analytic_formula() -> None:
    amplitude, offset = 0.002, 0.005
    density = _cosine_density(amplitude=amplitude, offset=offset)
    coordinate = np.linspace(0.0, LENGTH, 32, endpoint=False)
    wave = 2.0 * np.pi / LENGTH

    rdg = reduced_density_gradient(density)

    rho_au = (offset + amplitude * np.cos(wave * coordinate)) * BOHR2A**3
    gradient_au = (
        BOHR2A**4 * amplitude * wave * np.sin(wave * coordinate)
    )
    factor = 0.5 / (3.0 * np.pi**2) ** (1.0 / 3.0)
    expected = factor * np.abs(gradient_au) / rho_au ** (4.0 / 3.0)
    np.testing.assert_allclose(rdg[:, 0, 0], expected, rtol=1e-6, atol=1e-12)


def test_reduced_density_gradient_uses_the_constant_above_the_cut() -> None:
    # Above rho_cut, pp.x inserts a constant gradient instead of the real one.
    offset, amplitude = 0.5, 0.01
    density = _cosine_density(amplitude=amplitude, offset=offset)
    coordinate = np.linspace(0.0, LENGTH, 32, endpoint=False)

    rdg = reduced_density_gradient(density)

    rho_au = (offset + amplitude * np.cos(2 * np.pi * coordinate / LENGTH)) * BOHR2A**3
    assert np.all(rho_au > 0.05)
    factor = 0.5 / (3.0 * np.pi**2) ** (1.0 / 3.0)
    np.testing.assert_allclose(
        rdg[:, 0, 0], factor * RDG_GRADIENT_CUT / rho_au ** (4.0 / 3.0), rtol=1e-6
    )


def test_signed_density_hessian_uses_the_middle_eigenvalue() -> None:
    # rho = A + B cos(x) + C cos(y) with B > C > 0: the middle eigenvalue of
    # the Hessian is the negative y curvature, so the sign flips with cos(y).
    shape = (16, 16, 1)
    axes = [np.linspace(0.0, LENGTH, count, endpoint=False) for count in shape]
    x, y = np.meshgrid(axes[0], axes[1], indexing="ij")
    values = (
        0.05
        + 0.01 * np.cos(2 * np.pi * x / LENGTH)
        + 0.005 * np.cos(2 * np.pi * y / LENGTH)
    )
    density = _charge(values[:, :, None])

    signed = signed_density_hessian(density)

    assert signed[0, 0, 0] == pytest.approx(-values[0, 0], rel=1e-9)
    assert signed[8, 8, 0] == pytest.approx(values[8, 8], rel=1e-9)
    eigenvalues = hessian_eigenvalues(density)
    assert np.all(eigenvalues[..., 0] <= eigenvalues[..., 1])
    assert np.all(eigenvalues[..., 1] <= eigenvalues[..., 2])


def test_dori_follows_the_quantum_espresso_formula() -> None:
    offset, amplitude = 0.05, 0.01
    density = _cosine_density(amplitude=amplitude, offset=offset)
    coordinate = np.linspace(0.0, LENGTH, 32, endpoint=False)
    wave = 2.0 * np.pi / LENGTH

    indicator = dori(density)

    rho_au = (offset + amplitude * np.cos(wave * coordinate)) * BOHR2A**3
    gradient_au = -BOHR2A**4 * amplitude * wave * np.sin(wave * coordinate)
    hessian_au = -BOHR2A**5 * amplitude * wave**2 * np.cos(wave * coordinate)
    square = gradient_au**2
    theta = 4.0 * (rho_au * gradient_au * hessian_au - gradient_au * square) ** 2 / (
        square + DORI_REGULARIZER
    ) ** 3
    expected = theta / (1.0 + theta)
    np.testing.assert_allclose(indicator[:, 0, 0], expected, rtol=1e-6, atol=1e-30)
    assert np.all((indicator >= 0.0) & (indicator < 1.0))


def test_nci_scatter_data_filters_the_core_region() -> None:
    density = _cosine_density(amplitude=0.002, offset=0.005)

    signed, gradient = nci_scatter_data(density, rho_max=0.01)

    assert signed.shape == gradient.shape
    assert signed.size > 0
    assert np.all(np.abs(signed) <= 0.01 + 1e-12)
    assert np.all(np.isfinite(gradient))


def test_analyse_dispatch_and_unknown_quantity() -> None:
    density = _cosine_density()

    np.testing.assert_allclose(analyse(density, "rdg"), reduced_density_gradient(density))
    np.testing.assert_allclose(
        analyse(density, "sl2rho"), signed_density_hessian(density)
    )
    np.testing.assert_allclose(analyse(density, "dori"), dori(density))
    with pytest.raises(ValueError, match="unknown NCI quantity"):
        analyse(density, "elf")
