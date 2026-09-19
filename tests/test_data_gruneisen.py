"""Tests for the Grueneisen parameter data layer."""

from __future__ import annotations

import numpy as np
import pytest

from abacustools.core.constant import BOLTZMANN_CONSTANT_EV_PER_K
from abacustools.data.gruneisen import (
    DEFAULT_FREQUENCY_CUTOFF,
    MAXIMUM_STRAIN,
    MINIMUM_STRAIN,
    cell_volume,
    gruneisen_temperature,
    mode_heat_capacity,
    mode_mask,
    scaled_cell,
    summarize,
    validate_strain,
    volume_strain,
)
from abacustools.io.stru import AbacusATOM, AbacusSTRU


def _structure(cell: float = 4.0) -> AbacusSTRU:
    """Return a one atom cubic cell."""
    return AbacusSTRU(
        cell=[[cell, 0.0, 0.0], [0.0, cell, 0.0], [0.0, 0.0, cell]],
        atoms=[AbacusATOM(label="H", element="H", coord=(0.0, 0.0, 0.0))],
        metadata={"atom_type": "cartesian"},
    )


def test_validate_strain_accepts_the_working_range() -> None:
    assert validate_strain(0.01) == pytest.approx(0.01)
    assert validate_strain("0.005") == pytest.approx(0.005)


@pytest.mark.parametrize("value", [0.0, -0.01, "x", np.nan, MINIMUM_STRAIN / 10, MAXIMUM_STRAIN * 2])
def test_validate_strain_rejects_a_useless_value(value) -> None:
    with pytest.raises(ValueError):
        validate_strain(value)


def test_scaled_cell_changes_the_volume_and_keeps_the_positions() -> None:
    structure = _structure()

    expanded = scaled_cell(structure, 0.03)
    compressed = scaled_cell(structure, -0.03)

    reference = cell_volume(structure.cell)
    assert cell_volume(expanded.cell) == pytest.approx(1.03 * reference)
    assert cell_volume(compressed.cell) == pytest.approx(0.97 * reference)
    # The scaling is isotropic and the fractional coordinates are untouched.
    np.testing.assert_allclose(
        np.asarray(expanded.cell, dtype=float),
        np.asarray(structure.cell, dtype=float) * 1.03 ** (1.0 / 3.0),
    )
    np.testing.assert_allclose(
        np.asarray(expanded.coords_direct, dtype=float),
        np.asarray(structure.coords_direct, dtype=float),
    )


def test_volume_strain_reports_the_relative_change() -> None:
    structure = _structure()

    assert volume_strain(structure.cell, scaled_cell(structure, 0.02).cell) == pytest.approx(
        0.02
    )
    assert volume_strain(structure.cell, structure.cell) == pytest.approx(0.0)


def test_mode_heat_capacity_follows_the_two_limits() -> None:
    frequencies = np.array([[0.0, 1.0, 10.0, 100.0]])

    cold = mode_heat_capacity(frequencies, 10.0)
    hot = mode_heat_capacity(frequencies, 100000.0)

    # A frozen mode carries nothing, and the classical limit is one k_B per
    # mode; the zero frequency mode is dropped by the cutoff.
    assert cold[0, 0] == 0.0
    assert cold[0, 3] < 1.0e-4 * BOLTZMANN_CONSTANT_EV_PER_K
    for index in (1, 2, 3):
        assert hot[0, index] == pytest.approx(BOLTZMANN_CONSTANT_EV_PER_K, rel=1e-3)


def test_mode_heat_capacity_rejects_a_zero_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        mode_heat_capacity(np.array([[1.0]]), 0.0)


def test_mode_mask_leaves_out_the_translations_and_the_gaps() -> None:
    frequencies = np.array([[0.0, 1.0, 2.0, np.nan]])
    parameters = np.array([[1.0, 2.0, np.inf, 1.0]])

    mask = mode_mask(frequencies, parameters)

    assert mask.tolist() == [[False, True, False, False]]
    assert DEFAULT_FREQUENCY_CUTOFF > 0.0


def test_mode_mask_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mode_mask(np.zeros((2, 3)), np.zeros((2, 4)))


def test_gruneisen_temperature_returns_a_constant_parameter() -> None:
    frequencies = np.array([[1.0, 2.0], [1.5, 2.5]])
    weights = np.array([1.0, 3.0])
    parameters = np.full((2, 2), 1.5)

    values = gruneisen_temperature([0.0, 100.0, 300.0], frequencies, weights, parameters)

    # At zero temperature the average is undefined, above it every mode carries
    # the same parameter whatever its heat capacity is.
    assert values[0] is None
    assert values[1] == pytest.approx(1.5)
    assert values[2] == pytest.approx(1.5)


def test_gruneisen_temperature_weights_the_modes_by_their_heat_capacity() -> None:
    """A soft mode dominates the average while it is the only one excited.

    The average is the heat capacity weighted mean, so at a temperature far
    below the stiff mode only the soft mode contributes and the value is its
    own parameter, while at a high temperature the two modes weigh equally.
    """
    soft, stiff = 1.0, 300.0
    frequencies = np.array([[soft, stiff]])
    weights = np.array([1.0])
    parameters = np.array([[4.0, -2.0]])

    low, high = gruneisen_temperature([1.0, 1.0e7], frequencies, weights, parameters)

    assert low == pytest.approx(4.0, abs=1e-6)
    assert high == pytest.approx(1.0, rel=1e-3)


def test_summarize_reports_the_kept_statistics() -> None:
    frequencies = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    weights = np.array([1.0, 1.0])
    parameters = np.array([[5.0, 1.0, 2.0], [3.0, 4.0, 6.0]])

    summary = summarize(frequencies, weights, parameters)

    assert summary["modes"] == 6
    assert summary["modes_left_out"] == 1
    assert summary["minimum"] == pytest.approx(1.0)
    assert summary["maximum"] == pytest.approx(6.0)
    # The unweighted mean of the five kept values is 3.2, and every q point
    # weighs the same here.
    assert summary["mean"] == pytest.approx(3.2)
