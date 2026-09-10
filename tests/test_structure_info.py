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
from abacustools.data.structure import build_slab
from abacustools.io.stru import AbacusSTRU


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


def _arguments(
    path: Path,
    *,
    json: bool = False,
    layer_direction=None,
    coordination=None,
    min_vacuum: float = 5.0,
) -> Namespace:
    """Build the namespace the ``file info`` handler expects."""
    return Namespace(
        filename=path,
        input_format=None,
        cell=None,
        symprec=1e-5,
        angle_tolerance=5.0,
        layer_direction=layer_direction,
        coordination=coordination,
        min_vacuum=min_vacuum,
        json=json,
    )


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
        self.assertEqual({atom["wyckoff_multiplicity"] for atom in result["atoms"]}, {8})
        self.assertEqual(result["symmetry"]["equivalent_atoms"], [1, 1])
        self.assertEqual(len(result["inequivalent_positions"]), 1)
        self.assertEqual(result["inequivalent_positions"][0]["equivalent_indices"], [1, 2])
        self.assertEqual(result["inequivalent_positions"][0]["multiplicity"], 2)
        self.assertEqual(result["inequivalent_positions"][0]["wyckoff_multiplicity"], 8)
        self.assertEqual(result["resources"]["pseudopotentials"]["Si"], ["Si.upf"])
        self.assertEqual(result["resources"]["orbitals"]["Si"], ["Si.orb"])
        self.assertNotIn("paw", result["resources"])
        self.assertEqual(result["composition"]["formula_unit"], "Si")
        self.assertEqual(result["composition"]["formula_units_per_cell"], 2)
        self.assertEqual(result["composition"]["prototype"], "A")
        self.assertAlmostEqual(result["composition"]["density_g_cm3"], 2.330, places=2)
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
                self.assertEqual(run(_arguments(path)), 0)
            report = output.getvalue()

        self.assertIn("resources:", report)
        self.assertIn("label pseudopotential orbital", report)
        self.assertIn("Si.upf", report)
        self.assertIn("Si.orb", report)
        self.assertNotIn("paw", report.lower())
        header = next(line for line in report.splitlines() if "index label element" in line)
        self.assertIn("site_sym", header)
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
                run(_arguments(path))
            report = output.getvalue()

        header = next(line for line in report.splitlines() if "index label element" in line)
        self.assertIn("magmom", header)
        self.assertIn("move", header)
        self.assertIn("velocity", header)
        self.assertIn("2.0 (90.0, 45.0)", report)
        self.assertIn("0 0 0", report)
        self.assertIn("0.1 0.2 0.3", report)

    def test_inequivalent_table_ends_with_equivalent_atoms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path))
            report = output.getvalue()

        header = next(line for line in report.splitlines() if "representative" in line)
        self.assertEqual(
            header.split(),
            [
                "representative",
                "element",
                "wyckoff",
                "site_sym",
                "fractional",
                "equivalent",
                "atoms",
            ],
        )
        row = next(
            line for line in report.splitlines()
            if line.strip().startswith("1 ") and "Si" in line
        )
        self.assertEqual(row.split()[-2:], ["1", "2"])

    def test_wyckoff_symbols_carry_their_multiplicity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path))
            report = output.getvalue()

        # The eightfold Wyckoff position of diamond, as the International
        # Tables write it, in the inequivalent and in the per-atom table.
        self.assertEqual(report.count("8b"), 3)

    def test_a_p1_structure_does_not_repeat_the_atom_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(
                STRU.replace(
                    "2\n0.0 0.0 0.0\n0.25 0.25 0.25\n",
                    "3\n0.1 0.2 0.3\n0.23 0.31 0.17\n0.4 0.15 0.28\n",
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path))
            report = output.getvalue()
            result = structure_information(path)

        self.assertEqual(result["symmetry"]["space_group_number"], 1)
        # Every atom is its own orbit, so the table only repeats the per-atom
        # table, but the positions stay available in the JSON output.
        self.assertEqual(len(result["inequivalent_positions"]), 3)
        self.assertNotIn("symmetry-inequivalent positions:", report)
        self.assertIn("1a", report)
        self.assertIn("resources:", report)

    def test_resources_table_aligns_long_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(
                STRU.replace("Si.upf", "Si_ONCV_PBE-1.0.upf").replace(
                    "Si.orb", "Si_gga_7au_100Ry_2s2p1d.orb"
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path))
            report = output.getvalue()

        lines = report.splitlines()
        index = lines.index("resources:")
        header, row = lines[index + 1], lines[index + 2]
        for column, value in (
            ("pseudopotential", "Si_ONCV_PBE-1.0.upf"),
            ("orbital", "Si_gga_7au_100Ry_2s2p1d.orb"),
        ):
            end = header.index(column) + len(column)
            self.assertEqual(row[end - len(value) : end].strip(), value)
        self.assertEqual(len(header), len(row))

    def test_json_cli_output_and_xyz_without_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "molecule.xyz"
            path.write_text("1\nH molecule\nH 0 0 0\n", encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(run(_arguments(path, json=True)), 0)
            result = json.loads(output.getvalue())

        self.assertFalse(result["cell"]["periodic"])
        self.assertFalse(result["symmetry"]["available"])
        self.assertIn("periodic cell", result["symmetry"]["error"])

    def test_reports_magnetic_and_layer_symmetry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path, layer_direction="c"))
            report = output.getvalue()

        self.assertIn("magnetic symmetry:", report)
        self.assertIn("type II (grey)", report)
        self.assertIn("magnetic operations:", report)
        self.assertIn("layer symmetry:", report)
        self.assertIn("along c", report)
        self.assertIn("-43m", report)

    def test_reports_the_summary_items_of_a_symmetry_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path))
            report = output.getvalue()

        for line in (
            "formula unit: Si, 2 per cell",
            "prototype: A",
            "density (g/cm^3):",
            "symmetry: Fd-3m (No. 227), cubic",
            "symmetry operations: 48",
            "point group: m-3m (Oh)",
            "Bravais lattice: cF (Pearson symbol cF8)",
            "inversion symmetry: yes",
            "polar point group: no",
            "symmetry accuracy: symprec 1e-05 Angstrom, angle tolerance 5 degrees",
        ):
            with self.subTest(line=line):
                self.assertIn(line, report)

    def test_bulk_and_non_magnetic_structures_stay_compact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path))
            report = output.getvalue()

        self.assertNotIn("layer symmetry:", report)
        self.assertNotIn("magnetic ordering:", report)
        self.assertIn("dimensionality: 3D bulk, no vacuum direction", report)
        self.assertNotIn("coordination:", report)
        header = next(line for line in report.splitlines() if "index label element" in line)
        self.assertNotIn("cn", header)

    def test_reports_the_structure_dimensionality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            slab_path = Path(temporary) / "SLAB"
            build_slab(
                AbacusSTRU.read(path),
                miller_indices=(1, 0, 0),
                layers=3,
                vacuum=12.0,
            ).write(str(slab_path))
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(slab_path, json=True))
            result = json.loads(output.getvalue())

        self.assertEqual(result["dimensionality"]["dimensionality"], "slab")
        self.assertEqual(result["dimensionality"]["vacuum_directions"], ["c"])
        self.assertGreater(result["dimensionality"]["vacuum_thickness"], 5.0)
        self.assertIsNone(result["coordination"])

    def test_coordination_analysis_follows_the_dimensionality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path, coordination="crystalnn"))
            report = output.getvalue()

        self.assertIn("coordination: crystalnn for a 3D bulk", report)
        section = report.split("atoms:\n", 1)[1]
        header = section.splitlines()[0]
        self.assertIn("cn", header)
        self.assertNotIn("geometry", header)
        self.assertEqual(section.splitlines()[1].split()[-1], "4")

    def test_coordination_geometries_are_reported_for_a_bulk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path, coordination="auto", json=True))
            result = json.loads(output.getvalue())

        coordination = result["coordination"]
        self.assertTrue(coordination["available"])
        self.assertEqual(coordination["method"], "chemenv")
        self.assertEqual(coordination["dimensionality"]["dimensionality"], "bulk")
        self.assertEqual([site["geometry"] for site in coordination["sites"]], ["T:4", "T:4"])
        self.assertEqual([site["coordination_number"] for site in coordination["sites"]], [4, 4])

    def test_rejects_an_invalid_vacuum_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            with self.assertRaises(ValueError):
                structure_information(path, min_vacuum=0.0)

    def test_reports_the_magnetic_ordering_of_a_magnetic_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(
                STRU.replace("0.0 0.0 0.0\n", "0.0 0.0 0.0 mag 1.0\n").replace(
                    "0.25 0.25 0.25\n", "0.25 0.25 0.25 mag -1.0\n"
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                run(_arguments(path, json=True))
            result = json.loads(output.getvalue())

        ordering = result["magnetic_ordering"]
        self.assertTrue(ordering["available"])
        self.assertEqual(ordering["ordering"], "AFM")
        self.assertEqual(ordering["magnetic_sites"], 2)
        self.assertEqual(result["magnetic_symmetry"]["type"], "type III")
        self.assertIsNone(result["layer_symmetry"]["requested_direction"])

    def test_rejects_invalid_symmetry_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "STRU"
            path.write_text(STRU, encoding="utf-8")
            with self.assertRaises(ValueError):
                structure_information(path, symprec=0)


if __name__ == "__main__":
    unittest.main()
