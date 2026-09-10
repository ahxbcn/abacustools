"""Tests for structure information reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from abacustools.commands.file.structure_info import run, structure_information


STRU = """ATOMIC_SPECIES
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


class TestStructureInfo(unittest.TestCase):
    def test_reports_cell_species_symmetry_wyckoff_and_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            result = structure_information(path)

        self.assertEqual(result["natoms"], 2)
        self.assertEqual(result["element_counts"], {"Si": 2})
        self.assertAlmostEqual(result["cell"]["volume_angstrom3"], 5.43**3 / 4, places=2)
        self.assertTrue(result["symmetry"]["available"])
        self.assertEqual(result["symmetry"]["space_group_number"], 227)
        self.assertEqual(result["symmetry"]["space_group_symbol"], "Fd-3m")
        self.assertEqual(len(result["symmetry"]["wyckoff_positions"]), 2)
        self.assertEqual({atom["wyckoff"] for atom in result["atoms"]}, {"b"})
        self.assertEqual(result["symmetry"]["equivalent_atoms"], [1, 1])
        self.assertEqual(len(result["inequivalent_positions"]), 1)
        self.assertEqual(result["inequivalent_positions"][0]["equivalent_indices"], [1, 2])
        self.assertEqual(result["inequivalent_positions"][0]["multiplicity"], 2)
        self.assertEqual(result["resources"]["pseudopotentials"]["Si"], ["Si.upf"])
        self.assertEqual(result["resources"]["orbitals"]["Si"], ["Si.orb"])
        self.assertNotIn("paw", result["resources"])
        for atom in result["atoms"]:
            self.assertNotIn("pseudopotential", atom)
            self.assertNotIn("orbital", atom)
            self.assertNotIn("paw", atom)

    def test_atom_table_lists_resources_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    run(Namespace(
                        filename=path,
                        input_format=None,
                        cell=None,
                        symprec=1e-5,
                        angle_tolerance=5.0,
                        json=False,
                    )),
                    0,
                )
            report = output.getvalue()

        self.assertIn("resources:", report)
        self.assertIn("label pseudopotential orbital", report)
        self.assertIn("Si.upf", report)
        self.assertIn("Si.orb", report)
        self.assertNotIn("paw", report.lower())
        header = next(line for line in report.splitlines() if "index label element" in line)
        self.assertNotIn("pseudopotential", header)
        self.assertNotIn("orbital", header)
        self.assertNotIn("magmom", header)
        self.assertNotIn("move", header)

    def test_atom_table_shows_moments_constraints_and_velocities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(
                STRU.replace(
                    "0.0 0.0 0.0\n",
                    "0.0 0.0 0.0 0 0 0 mag 2.0 angle1 90.0 angle2 45.0\n",
                ).replace(
                    "0.25 0.25 0.25\n",
                    "0.25 0.25 0.25 1 1 1 v 0.1 0.2 0.3 mag 0.5\n",
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                run(Namespace(
                    filename=path,
                    input_format=None,
                    cell=None,
                    symprec=1e-5,
                    angle_tolerance=5.0,
                    json=False,
                ))
            report = output.getvalue()

        header = next(line for line in report.splitlines() if "index label element" in line)
        self.assertIn("magmom", header)
        self.assertIn("move", header)
        self.assertIn("velocity", header)
        self.assertIn("2.0 (90.0, 45.0)", report)
        self.assertIn("0 0 0", report)
        self.assertIn("0.1 0.2 0.3", report)

    def test_json_cli_output_and_xyz_without_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "molecule.xyz"
            path.write_text("1\nH molecule\nH 0 0 0\n", encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    run(Namespace(
                        filename=path,
                        input_format=None,
                        cell=None,
                        symprec=1e-5,
                        angle_tolerance=5.0,
                        json=True,
                    )),
                    0,
                )
            result = json.loads(output.getvalue())

        self.assertFalse(result["cell"]["periodic"])
        self.assertFalse(result["symmetry"]["available"])
        self.assertIn("periodic cell", result["symmetry"]["error"])

    def test_rejects_invalid_symmetry_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            with self.assertRaises(ValueError):
                structure_information(path, symprec=0)


if __name__ == "__main__":
    unittest.main()
