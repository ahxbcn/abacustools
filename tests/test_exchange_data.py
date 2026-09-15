"""Tests for the four-state exchange coupling analysis."""

from __future__ import annotations

import unittest

import numpy as np

from abacustools.data.exchange import (
    CASES,
    ExchangeError,
    angle_between,
    fit_exchange_coupling,
    four_state_energy,
    moment_vector,
    tilted_moments,
)


def _moment(magnitude: float, theta: float, phi: float = 0.0) -> np.ndarray:
    """Return a moment of the given magnitude at the given polar angles."""
    theta_radians = np.radians(theta)
    phi_radians = np.radians(phi)
    return magnitude * np.array(
        [
            np.sin(theta_radians) * np.cos(phi_radians),
            np.sin(theta_radians) * np.sin(phi_radians),
            np.cos(theta_radians),
        ]
    )


class TestMomentHelpers(unittest.TestCase):
    def test_scalar_moment_points_along_z(self) -> None:
        np.testing.assert_allclose(moment_vector(2.5), [0.0, 0.0, 2.5])

    def test_angle_between_known_moments(self) -> None:
        self.assertAlmostEqual(angle_between([0, 0, 1], [0, 0, 1]), 0.0)
        self.assertAlmostEqual(angle_between([0, 0, 1], [0, 0, -1]), 180.0)
        self.assertAlmostEqual(angle_between([0, 0, 1], [1, 0, 0]), 90.0)
        self.assertAlmostEqual(angle_between([0, 0, 2], [3, 0, 3]), 45.0)

    def test_invalid_moments_are_rejected(self) -> None:
        with self.assertRaises(ExchangeError):
            moment_vector(None)
        with self.assertRaises(ExchangeError):
            moment_vector([1.0, 2.0])
        with self.assertRaises(ExchangeError):
            moment_vector(["a", "b", "c"])
        with self.assertRaises(ExchangeError):
            angle_between([0, 0, 0], [0, 0, 1])


class TestTiltedMoments(unittest.TestCase):
    def test_every_case_shares_the_pair_angle(self) -> None:
        for theta0 in (0.0, 30.0, 90.0, 150.0, 180.0):
            for azimuth in (0.0, 35.0, 200.0):
                first = _moment(2.0, theta0, azimuth)
                second = _moment(1.5, 0.0)
                for tilt in (1.0, 25.0):
                    for case in CASES:
                        rotated1, rotated2 = tilted_moments(first, second, tilt, case)
                        achieved = angle_between(rotated1, rotated2)
                        # The reported angle wraps at 180 degrees, but the
                        # cosine that enters the fit does not.
                        self.assertAlmostEqual(
                            np.cos(np.radians(achieved)),
                            np.cos(np.radians(theta0 + tilt)),
                            places=10,
                        )
                        self.assertLessEqual(achieved, 180.0)
                        self.assertAlmostEqual(np.linalg.norm(rotated1), 2.0, places=10)
                        self.assertAlmostEqual(np.linalg.norm(rotated2), 1.5, places=10)
                        self.assertTrue(np.all(np.isfinite(rotated1)))
                        self.assertTrue(np.all(np.isfinite(rotated2)))

    def test_antiparallel_references_are_handled(self) -> None:
        antiparallel = ([0.0, 0.0, 2.0], [0.0, 0.0, -2.0])
        for case in CASES:
            rotated1, rotated2 = tilted_moments(antiparallel[0], antiparallel[1], 10.0, case)
            self.assertAlmostEqual(angle_between(rotated1, rotated2), 170.0, places=8)

    def test_zero_tilt_returns_the_reference_moments(self) -> None:
        first = _moment(2.0, 60.0, 20.0)
        second = _moment(1.0, 25.0, 130.0)
        for case in CASES:
            rotated1, rotated2 = tilted_moments(first, second, 0.0, case)
            np.testing.assert_allclose(rotated1, first, atol=1e-12)
            np.testing.assert_allclose(rotated2, second, atol=1e-12)

    def test_invalid_tilts_are_rejected(self) -> None:
        with self.assertRaises(ExchangeError):
            tilted_moments([0, 0, 1], [0, 0, 1], -1.0, "atom1")
        with self.assertRaises(ExchangeError):
            tilted_moments([0, 0, 1], [0, 0, 1], 1.0, "neither")
        with self.assertRaises(ExchangeError):
            tilted_moments([0, 0, 0], [0, 0, 1], 1.0, "atom1")


class TestFourStateEnergy(unittest.TestCase):
    def test_combination_subtracts_the_single_references(self) -> None:
        self.assertAlmostEqual(four_state_energy(-10.0, -10.5, -10.25, -11.0), -0.25)

    def _bilinear_fit(self, coefficient, first, second, tilts=(5.0, 10.0, 15.0, 20.0, 25.0)):
        """Fit four-state energies of the model E = c * (m1 . m2)."""
        original = coefficient * float(np.dot(first, second))
        angles, energies = [], []
        for tilt in tilts:
            states = {case: tilted_moments(first, second, tilt, case) for case in CASES}
            energies.append(
                four_state_energy(
                    original,
                    coefficient * float(np.dot(*states["atom1"])),
                    coefficient * float(np.dot(*states["atom2"])),
                    coefficient * float(np.dot(*states["both"])),
                )
            )
            angles.append(angle_between(*states["both"]))
        return fit_exchange_coupling(angles, energies)

    def test_fit_recovers_the_coefficient_of_the_moment_coupling(self) -> None:
        # The fit is defined on the unit directions of the two moments, so an
        # energy E = c * (m1 . m2) is recovered as c times the product of the
        # two moment magnitudes.
        coefficient = 0.012
        first = _moment(2.0, 65.0, 10.0)
        second = _moment(1.5, 46.0, 250.0)
        magnitude_product = float(np.linalg.norm(first) * np.linalg.norm(second))

        fit = self._bilinear_fit(coefficient, first, second)
        self.assertAlmostEqual(
            fit.coupling_mev, coefficient * magnitude_product * 1000.0, places=8
        )
        self.assertAlmostEqual(fit.r_squared, 1.0, places=10)
        self.assertLess(fit.residual_ev, 1e-12)
        self.assertEqual(fit.to_dict()["points"], 5)

    def test_fit_returns_the_unit_direction_coefficient(self) -> None:
        coefficient = 0.02
        first = _moment(1.0, 0.0)
        second = _moment(1.0, 70.0)

        fit = self._bilinear_fit(coefficient, first, second)
        self.assertAlmostEqual(fit.coupling_mev, coefficient * 1000.0, places=8)

    def test_fit_reports_the_reference_angle_offset(self) -> None:
        # The intercept absorbs the reference angle, so the slope is unchanged
        # when the pair does not start from a parallel configuration.
        coefficient = -0.008
        first = _moment(1.0, 0.0)
        second = _moment(1.0, 120.0)
        original = coefficient * float(np.dot(first, second))
        angles, energies = [], []
        for index in range(1, 4):
            tilt = 2.0 * index
            states = {case: tilted_moments(first, second, tilt, case) for case in CASES}
            both = coefficient * float(np.dot(*states["both"]))
            atom1 = coefficient * float(np.dot(*states["atom1"]))
            atom2 = coefficient * float(np.dot(*states["atom2"]))
            angles.append(angle_between(*states["both"]))
            energies.append(four_state_energy(original, atom1, atom2, both))

        fit = fit_exchange_coupling(angles, energies)
        self.assertAlmostEqual(fit.coupling_mev, coefficient * 1000.0, places=8)
        # The intercept absorbs the reference angle: dE = c x - c (1 - cos(theta0)).
        self.assertAlmostEqual(
            fit.intercept_ev, -coefficient * (1.0 - np.cos(np.radians(120.0))), places=10
        )

    def test_fit_validates_its_input(self) -> None:
        with self.assertRaises(ExchangeError):
            fit_exchange_coupling([1.0, 2.0], [1.0])
        with self.assertRaises(ExchangeError):
            fit_exchange_coupling([1.0], [1.0])


if __name__ == "__main__":
    unittest.main()
