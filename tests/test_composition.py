"""Tests for the composition, mass and density summary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from abacustools.data.composition import composition_summary
from abacustools.io.stru import AbacusSTRU


DIAMOND = """ATOMIC_SPECIES
Si 28.0855 Si.upf

NUMERICAL_ORBITAL
Si.orb

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
0 5.1306 5.1306
5.1306 0 5.1306
5.1306 5.1306 0

ATOMIC_POSITIONS
Direct

Si
0.0
2
0.0 0.0 0.0
0.25 0.25 0.25
"""

ROCK_SALT = """ATOMIC_SPECIES
Si 28.0855 Si.upf
O 15.999 O.upf

NUMERICAL_ORBITAL
Si.orb
O.orb

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
9.448 0 0
0 9.448 0
0 0 9.448

ATOMIC_POSITIONS
Direct

Si
0.0
1
0.0 0.0 0.0

O
0.0
1
0.5 0.5 0.5
"""


class TestCompositionSummary(unittest.TestCase):
    """Cover the formula unit, prototype, mass and density."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _read(self, name: str, text: str) -> AbacusSTRU:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return AbacusSTRU.read(path)

    def test_diamond_silicon(self) -> None:
        result = composition_summary(self._read("STRU", DIAMOND))

        self.assertTrue(result["available"])
        self.assertEqual(result["formula_unit"], "Si")
        self.assertEqual(result["formula_units_per_cell"], 2)
        self.assertEqual(result["prototype"], "A")
        self.assertAlmostEqual(result["mass_amu"], 2 * 28.0855, places=4)
        # Two silicon atoms in 40.03 Angstrom^3, the density of the cell.
        self.assertAlmostEqual(result["density_g_cm3"], 2.330, places=2)

    def test_two_element_cell(self) -> None:
        result = composition_summary(self._read("STRU", ROCK_SALT))

        self.assertEqual(result["formula_unit"], "SiO")
        self.assertEqual(result["formula_units_per_cell"], 1)
        self.assertEqual(result["prototype"], "AB")
        self.assertGreater(result["density_g_cm3"], 0)

    def test_a_cell_less_structure_keeps_its_composition(self) -> None:
        path = self.directory / "molecule.xyz"
        path.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
        result = composition_summary(AbacusSTRU.read(path))

        self.assertTrue(result["available"])
        self.assertEqual(result["formula_unit"], "H2")
        self.assertIsNone(result["density_g_cm3"])


if __name__ == "__main__":
    unittest.main()
