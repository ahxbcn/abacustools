"""Tests for the ASE harmonic vibration adapter."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from abacustools.data.vibration import HarmonicVibration
from abacustools.integrations.ase_vibration import AseVibrationData
from abacustools.io.stru import AbacusSTRU


def hydrogen_structure(directory: Path) -> AbacusSTRU:
    """Write and read a two-atom hydrogen STRU."""
    path = directory / "STRU"
    path.write_text(
        """ATOMIC_SPECIES
H 1.0079 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
15 0 0
0 15 0
0 0 15

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
    return AbacusSTRU.read(path)


def hydrogen_hessian(
    force_constants: tuple[float, float, float] = (20.0, 20.0, 20.0),
) -> np.ndarray:
    """Return the Hessian of a diatomic molecule along all three directions."""
    hessian = np.zeros((2, 3, 2, 3))
    for axis, force_constant in enumerate(force_constants):
        hessian[0, axis, 0, axis] = force_constant
        hessian[1, axis, 1, axis] = force_constant
        hessian[0, axis, 1, axis] = -force_constant
        hessian[1, axis, 0, axis] = -force_constant
    return hessian


class TestAseVibrationData(unittest.TestCase):
    def test_frequencies_match_the_builtin_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structure = hydrogen_structure(Path(temporary))
            ase_vibration = AseVibrationData(structure, hydrogen_hessian())
            # The masses of the STRU reach the ASE atoms.
            np.testing.assert_allclose(ase_vibration.atoms.get_masses(), [1.0079, 1.0079])
            builtin = HarmonicVibration(
                hydrogen_hessian(),
                ase_vibration.atoms.get_masses(),
            )
            np.testing.assert_allclose(
                np.sort(ase_vibration.signed_frequencies),
                np.sort(builtin.signed_frequencies),
                rtol=1e-6,
            )
            self.assertAlmostEqual(
                ase_vibration.zero_point_energy(),
                builtin.zero_point_energy(),
                delta=1e-6,
            )

    def test_mass_overrides_reach_the_ase_atoms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structure = hydrogen_structure(Path(temporary))
            light = AseVibrationData(structure, hydrogen_hessian())
            heavy = AseVibrationData(
                structure,
                hydrogen_hessian(),
                masses=[2.014, 2.014],
            )
            np.testing.assert_allclose(heavy.atoms.get_masses(), [2.014, 2.014])
            ratio = abs(heavy.signed_frequencies[-1]) / abs(light.signed_frequencies[-1])
            self.assertAlmostEqual(ratio, np.sqrt(1.0079 / 2.014), places=6)
            with self.assertRaises(ValueError):
                AseVibrationData(structure, hydrogen_hessian(), masses=[1.0])

    def test_thermo_matches_ase_harmonic_thermo(self) -> None:
        from ase.thermochemistry import HarmonicThermo

        with tempfile.TemporaryDirectory() as temporary:
            structure = hydrogen_structure(Path(temporary))
            ase_vibration = AseVibrationData(structure, hydrogen_hessian())
            reference = HarmonicThermo(
                ase_vibration.energies,
                ignore_imag_modes=True,
            )
            thermo = ase_vibration.thermo(298.15)
            self.assertAlmostEqual(
                thermo["entropy"],
                float(reference.get_entropy(298.15)),
                places=12,
            )
            self.assertAlmostEqual(
                thermo["free_energy"],
                float(reference.get_helmholtz_energy(298.15)),
                places=10,
            )

    def test_unstable_modes_are_skipped_by_the_thermochemistry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structure = hydrogen_structure(Path(temporary))
            ase_vibration = AseVibrationData(
                structure,
                hydrogen_hessian((20.0, 20.0, -20.0)),
            )
            self.assertEqual(len(ase_vibration.signed_frequencies), 6)
            self.assertTrue(any(value < 0.0 for value in ase_vibration.signed_frequencies))
            self.assertTrue(any(value > 0.0 for value in ase_vibration.signed_frequencies))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                thermo = ase_vibration.thermo(300.0)
            self.assertGreater(thermo["entropy"], 0.0)

    def test_invalid_hessian_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structure = hydrogen_structure(Path(temporary))
            with self.assertRaises(ValueError):
                AseVibrationData(structure, np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
