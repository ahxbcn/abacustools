"""Tests for the built-in harmonic vibration analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from abacustools.core.constant import (
    AMU_TO_KG,
    ANGSTROM_TO_METRE,
    BOLTZMANN_CONSTANT_EV_PER_K,
    ELEMENTARY_CHARGE,
    HBAR,
    INV_CM_TO_EV,
    SPEED_OF_LIGHT,
)
from abacustools.data.vibration import (
    HarmonicVibration,
    frequency_values,
    harmonic_thermo,
)
from abacustools.io.stru import AbacusSTRU


_H2_MASS = 1.008
_H2_FORCE_CONSTANT = 20.0


def diatomic_hessian(force_constant: float = _H2_FORCE_CONSTANT) -> np.ndarray:
    """Return the Hessian of a diatomic molecule along all three directions."""
    hessian = np.zeros((2, 3, 2, 3))
    for axis in range(3):
        hessian[0, axis, 0, axis] = force_constant
        hessian[1, axis, 1, axis] = force_constant
        hessian[0, axis, 1, axis] = -force_constant
        hessian[1, axis, 0, axis] = -force_constant
    return hessian


class TestHarmonicVibration(unittest.TestCase):
    def test_diatomic_frequency_matches_analytic_solution(self) -> None:
        vibration = HarmonicVibration(
            diatomic_hessian(),
            [_H2_MASS, _H2_MASS],
        )
        frequencies = np.sort(np.abs(vibration.signed_frequencies))
        # Three translations are zero, three stretching modes are degenerate.
        for frequency in frequencies[:3]:
            self.assertAlmostEqual(frequency, 0.0, places=6)

        force_constant_si = _H2_FORCE_CONSTANT * ELEMENTARY_CHARGE / ANGSTROM_TO_METRE ** 2
        reduced_mass_si = 0.5 * _H2_MASS * AMU_TO_KG
        expected = np.sqrt(force_constant_si / reduced_mass_si)
        expected /= 2.0 * np.pi * SPEED_OF_LIGHT * 100.0
        for frequency in frequencies[3:]:
            self.assertAlmostEqual(frequency, expected, places=6)

    def test_modes_are_mass_normalized_and_diagonalize_the_hessian(self) -> None:
        vibration = HarmonicVibration(diatomic_hessian(), [_H2_MASS, _H2_MASS])
        weights = np.repeat(vibration.masses, 3).reshape(-1, 3)
        for mode_index, mode in enumerate(vibration.modes):
            vector = mode.reshape(-1)
            self.assertAlmostEqual(float(np.sum(weights * mode ** 2)), 1.0, places=9)
            expected = vibration.eigenvalues[mode_index] * weights.reshape(-1) * vector
            np.testing.assert_allclose(vibration.hessian @ vector, expected, atol=1e-9)
        # The modes of a symmetric Hessian are orthonormal in mass-weighted space.
        overlap = np.einsum("ij,kij,kij->k", weights, vibration.modes, vibration.modes)
        np.testing.assert_allclose(overlap, 1.0, atol=1e-9)

    def test_reduced_mass_and_force_constant_describe_the_mode(self) -> None:
        vibration = HarmonicVibration(diatomic_hessian(), [_H2_MASS, _H2_MASS])
        reduced = vibration.reduced_masses
        # The mode vectors move both atoms with equal amplitude, so the reduced
        # mass of every mode of equal masses is the atomic mass itself.
        np.testing.assert_allclose(reduced, _H2_MASS, rtol=1e-12)
        np.testing.assert_allclose(
            vibration.force_constants,
            vibration.eigenvalues * reduced,
            rtol=1e-12,
        )
        np.testing.assert_allclose(vibration.force_constants[3:], 2.0 * _H2_FORCE_CONSTANT)
        # The mode energy follows from the reduced mass and the force constant.
        frequencies = np.abs(vibration.frequencies[3:].real)
        expected = np.sqrt(
            vibration.force_constants[3:]
            * ELEMENTARY_CHARGE
            / ANGSTROM_TO_METRE ** 2
            / (reduced[3:] * AMU_TO_KG)
        )
        expected /= 2.0 * np.pi * SPEED_OF_LIGHT * 100.0
        np.testing.assert_allclose(frequencies, expected, rtol=1e-12)
        # Each Cartesian component contributes 1 / m to the sum of 1 / mu.
        self.assertAlmostEqual(
            float(np.sum(1.0 / reduced)),
            float(np.sum(3.0 / vibration.masses)),
            places=12,
        )

    def test_mode_energy_matches_the_analytic_photon_energy(self) -> None:
        vibration = HarmonicVibration(diatomic_hessian(), [_H2_MASS, _H2_MASS])
        energy = float(np.sort(np.abs(vibration.energies))[-1])
        force_constant_si = _H2_FORCE_CONSTANT * ELEMENTARY_CHARGE / ANGSTROM_TO_METRE ** 2
        reduced_mass_si = 0.5 * _H2_MASS * AMU_TO_KG
        expected = np.sqrt(force_constant_si / reduced_mass_si) * HBAR / ELEMENTARY_CHARGE
        self.assertAlmostEqual(energy, expected, places=9)

    def test_zero_point_energy_can_skip_unstable_modes(self) -> None:
        hessian = diatomic_hessian(-_H2_FORCE_CONSTANT)
        vibration = HarmonicVibration(hessian, [_H2_MASS, _H2_MASS])
        self.assertEqual(vibration.imaginary_modes.tolist(), [0, 1, 2])
        self.assertAlmostEqual(vibration.zero_point_energy(stable_only=True), 0.0, places=12)
        self.assertGreater(vibration.zero_point_energy(), 0.0)

    def test_modes_all_atoms_pads_unselected_atoms(self) -> None:
        vibration = HarmonicVibration(
            diatomic_hessian(),
            [_H2_MASS, _H2_MASS],
            indices=[0, 2],
            natoms=3,
        )
        modes = vibration.modes_all_atoms()
        self.assertEqual(modes.shape, (6, 3, 3))
        np.testing.assert_allclose(modes[:, 1, :], 0.0)
        np.testing.assert_allclose(modes[:, 0, :], vibration.modes[:, 0, :])
        np.testing.assert_allclose(modes[:, 2, :], vibration.modes[:, 1, :])

    def test_hessian_is_symmetrized(self) -> None:
        asymmetric = diatomic_hessian().reshape(6, 6)
        asymmetric[0, 1] += 0.4
        vibration = HarmonicVibration(asymmetric, [_H2_MASS, _H2_MASS])
        np.testing.assert_allclose(vibration.hessian, vibration.hessian.T)
        np.testing.assert_allclose(
            vibration.hessian, 0.5 * (asymmetric + asymmetric.T)
        )

    def test_summary_lists_mode_properties(self) -> None:
        vibration = HarmonicVibration(diatomic_hessian(), [_H2_MASS, _H2_MASS])
        summary = vibration.summary()
        self.assertEqual(len(summary), 6)
        self.assertEqual(summary[0]["mode"], 1)
        self.assertAlmostEqual(
            summary[3]["frequency"],
            float(vibration.frequencies[3].real),
            places=9,
        )
        self.assertGreater(summary[3]["force_constant"], 0.0)
        self.assertIn("reduced_mass", summary[3])

    def test_from_structure_uses_structure_masses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(
                """ATOMIC_SPECIES
H 1.5 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Cartesian

H
0.0
2
0 0 0
0 0 0.74
""",
                encoding="utf-8",
            )
            structure = AbacusSTRU.read(path)
            self.assertIsNotNone(structure)
            vibration = HarmonicVibration.from_structure(
                structure,
                np.diag([_H2_FORCE_CONSTANT] * 3),
                indices=[0],
                masses=[_H2_MASS, _H2_MASS],
            )
            self.assertEqual(vibration.n_atoms, 1)
            self.assertEqual(vibration.natoms, 2)
            self.assertEqual(vibration.modes_all_atoms().shape, (3, 2, 3))
            np.testing.assert_allclose(vibration.masses, [_H2_MASS])

            default = HarmonicVibration.from_structure(structure, diatomic_hessian())
            np.testing.assert_allclose(default.masses, [1.5, 1.5])
            overridden = HarmonicVibration.from_structure(
                structure,
                diatomic_hessian(),
                masses=[2.014, 2.014],
            )
            np.testing.assert_allclose(overridden.masses, [2.014, 2.014])

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HarmonicVibration(np.zeros((2, 2)), [1.0])
        with self.assertRaises(ValueError):
            HarmonicVibration(diatomic_hessian(), [1.0])
        with self.assertRaises(ValueError):
            HarmonicVibration(diatomic_hessian(), [1.0, 0.0])
        with self.assertRaises(ValueError):
            HarmonicVibration(diatomic_hessian(), [1.0, -1.0])
        with self.assertRaises(ValueError):
            HarmonicVibration(diatomic_hessian(), [1.0, 1.0], indices=[0, 0])
        with self.assertRaises(ValueError):
            HarmonicVibration(diatomic_hessian(), [1.0, 1.0], indices=[0, 2], natoms=2)
        with self.assertRaises(ValueError):
            HarmonicVibration(np.full((6, 6), np.nan), [1.0, 1.0])


class TestHarmonicThermo(unittest.TestCase):
    def test_thermo_follows_the_harmonic_oscillator_formulas(self) -> None:
        energies = np.array([0.05, 0.2, 0.4])
        temperature = 350.0
        thermo = harmonic_thermo(energies, temperature)

        kT = BOLTZMANN_CONSTANT_EV_PER_K * temperature
        x = energies / kT
        occupation = 1.0 / np.expm1(x)
        expected_entropy = BOLTZMANN_CONSTANT_EV_PER_K * np.sum(
            x * occupation - np.log1p(-np.exp(-x))
        )
        expected_thermal = float(np.sum(energies * occupation))
        expected_heat_capacity = BOLTZMANN_CONSTANT_EV_PER_K * float(
            np.sum(x ** 2 * occupation ** 2 * np.exp(x))
        )

        self.assertAlmostEqual(thermo.zero_point_energy, 0.5 * float(energies.sum()))
        self.assertAlmostEqual(thermo.thermal_energy, expected_thermal, places=12)
        self.assertAlmostEqual(
            thermo.internal_energy,
            thermo.zero_point_energy + expected_thermal,
            places=12,
        )
        self.assertAlmostEqual(thermo.entropy, expected_entropy, places=12)
        self.assertAlmostEqual(thermo.heat_capacity, expected_heat_capacity, places=12)
        self.assertAlmostEqual(
            thermo.free_energy,
            thermo.internal_energy - temperature * thermo.entropy,
            places=12,
        )

    def test_thermo_matches_the_ase_reference(self) -> None:
        from ase.thermochemistry import HarmonicThermo

        energies = [0.05, 0.13, 0.2038, 0.4362]
        temperature = 298.15
        reference = HarmonicThermo(energies)
        thermo = harmonic_thermo(energies, temperature)
        self.assertAlmostEqual(
            thermo.entropy, float(reference.get_entropy(temperature)), places=7
        )
        self.assertAlmostEqual(
            thermo.free_energy,
            float(reference.get_helmholtz_energy(temperature)),
            places=5,
        )

    def test_heat_capacity_reaches_the_classical_limit(self) -> None:
        # A mode far below kT behaves classically: one k_B per mode.
        temperature = 1000.0
        energies = np.array([0.001, 0.002])
        thermo = harmonic_thermo(energies, temperature)
        x = energies / (BOLTZMANN_CONSTANT_EV_PER_K * temperature)
        classical_entropy = BOLTZMANN_CONSTANT_EV_PER_K * float(np.sum(1.0 - np.log(x)))
        classical_heat_capacity = 2.0 * BOLTZMANN_CONSTANT_EV_PER_K
        self.assertAlmostEqual(
            thermo.heat_capacity, classical_heat_capacity, delta=0.01 * classical_heat_capacity
        )
        self.assertAlmostEqual(
            thermo.entropy, classical_entropy, delta=0.01 * classical_entropy
        )

    def test_low_temperature_freezes_the_modes(self) -> None:
        thermo = harmonic_thermo([0.2, 0.4], 1.0)
        self.assertAlmostEqual(thermo.thermal_energy, 0.0, places=30)
        self.assertAlmostEqual(thermo.entropy, 0.0, places=30)
        self.assertAlmostEqual(thermo.heat_capacity, 0.0, places=30)
        self.assertAlmostEqual(thermo.free_energy, thermo.zero_point_energy, places=30)

    def test_thermo_skips_or_rejects_imaginary_modes(self) -> None:
        vibration = HarmonicVibration(
            diatomic_hessian(-_H2_FORCE_CONSTANT),
            [_H2_MASS, _H2_MASS],
        )
        thermo = vibration.thermo(300.0)
        self.assertEqual(thermo.n_imaginary_modes, 3)
        self.assertAlmostEqual(thermo.zero_point_energy, 0.0, places=12)
        with self.assertRaises(RuntimeError):
            vibration.thermo(300.0, ignore_imaginary=False)
        with self.assertRaises(ValueError):
            harmonic_thermo([0.0], 300.0)
        with self.assertRaises(ValueError):
            harmonic_thermo([0.1], 0.0)
        with self.assertRaises(ValueError):
            harmonic_thermo([0.1], float("nan"))

    def test_stable_thermo_matches_the_mode_energies(self) -> None:
        vibration = HarmonicVibration(diatomic_hessian(), [_H2_MASS, _H2_MASS])
        thermo = vibration.thermo(298.15)
        self.assertEqual(thermo.n_imaginary_modes, 0)
        stable = np.abs(vibration.energies)
        self.assertAlmostEqual(thermo.zero_point_energy, 0.5 * float(stable.sum()))
        self.assertAlmostEqual(
            thermo.internal_energy - 298.15 * thermo.entropy,
            thermo.free_energy,
            places=12,
        )


class TestFrequencyValues(unittest.TestCase):
    def test_imaginary_frequencies_become_negative_values(self) -> None:
        self.assertEqual(
            frequency_values(np.array([100.0 + 0j, 20.0j, -30.0j, -5.0 + 0j])),
            [100.0, -20.0, -30.0, -5.0],
        )

    def test_mixed_frequency_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            frequency_values(np.array([1.0 + 1.0j]))

    def test_energy_unit_conversion_round_trip(self) -> None:
        vibration = HarmonicVibration(diatomic_hessian(), [_H2_MASS, _H2_MASS])
        np.testing.assert_allclose(
            np.abs(vibration.energies),
            np.abs(np.asarray(vibration.signed_frequencies)) * INV_CM_TO_EV,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
