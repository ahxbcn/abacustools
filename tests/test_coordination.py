"""Tests for the vacuum, dimensionality and coordination analyses."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.coordination import coordination_analysis
from abacustools.data.dimensionality import (
    classify_dimensionality,
    largest_vacuum,
    vacuum_gaps,
)
from abacustools.data.structure import build_slab
from abacustools.data.symmetry import detect_aperiodic_direction
from abacustools.io.stru import AbacusSTRU


DIAMOND_CELL = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
DIAMOND_POSITIONS = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
SLAB_CELL = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 20.0]]
SLAB_POSITIONS = [
    [0.25, 0.25, 0.30],
    [0.75, 0.75, 0.30],
    [0.25, 0.75, 0.70],
    [0.75, 0.25, 0.70],
]
WIRE_CELL = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 4.0]]
WIRE_POSITIONS = [[0.5, 0.5, 0.05], [0.5, 0.5, 0.35], [0.5, 0.5, 0.65], [0.5, 0.5, 0.95]]
MOLECULE_CELL = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
MOLECULE_POSITIONS = [[0.5, 0.5, 0.5], [0.52, 0.5, 0.5], [0.54, 0.5, 0.5]]


def _stru_text(cell: list[list[float]], positions: list[list[float]], element: str = "Si") -> str:
    """Build a minimal STRU file with the given cell and positions."""
    lines = [
        "ATOMIC_SPECIES",
        f"{element} 28.0855 {element}.upf",
        "",
        "NUMERICAL_ORBITAL",
        f"{element}.orb",
        "",
        "LATTICE_CONSTANT",
        "1.0",
        "",
        "LATTICE_VECTORS",
    ]
    lines += [" ".join(str(component) for component in vector) for vector in cell]
    lines += ["", "ATOMIC_POSITIONS", "Direct", "", element, "0.0", str(len(positions))]
    lines += [" ".join(str(component) for component in position) for position in positions]
    return "\n".join(lines) + "\n"


class TestDimensionality(unittest.TestCase):
    """Cover the vacuum gaps and the dimensionality classification."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _structure(
        self,
        name: str,
        cell: list[list[float]],
        positions: list[list[float]],
    ) -> AbacusSTRU:
        path = self.directory / name
        path.write_text(_stru_text(cell, positions), encoding="utf-8")
        return AbacusSTRU.read(path)

    def test_vacuum_gaps_of_a_slab(self) -> None:
        slab = self._structure("slab.STRU", SLAB_CELL, SLAB_POSITIONS)
        gaps = vacuum_gaps(slab)

        self.assertEqual([gap["direction"] for gap in gaps], ["a", "b", "c"])
        self.assertAlmostEqual(gaps[0]["thickness"], 8.0 * BOHR_TO_ANG * 0.5, places=6)
        self.assertAlmostEqual(gaps[2]["thickness"], 20.0 * BOHR_TO_ANG * 0.6, places=6)
        # The gap starts at the highest atom and ends at the lowest one,
        # because it contains the boundary of the periodic cell.
        self.assertAlmostEqual(gaps[2]["bottom"], 20.0 * BOHR_TO_ANG * 0.7)
        self.assertAlmostEqual(gaps[2]["top"], 20.0 * BOHR_TO_ANG * 0.3)
        self.assertEqual(largest_vacuum(slab)["direction"], "c")

    def test_classifies_bulk_slab_wire_and_molecule(self) -> None:
        cases = {
            "bulk": (DIAMOND_CELL, DIAMOND_POSITIONS, []),
            "slab": (SLAB_CELL, SLAB_POSITIONS, ["c"]),
            "wire": (WIRE_CELL, WIRE_POSITIONS, ["a", "b"]),
            "molecule": (MOLECULE_CELL, MOLECULE_POSITIONS, ["a", "b", "c"]),
        }
        for name, (cell, positions, vacuum_directions) in cases.items():
            with self.subTest(dimensionality=name):
                structure = self._structure(f"{name}.STRU", cell, positions)
                result = classify_dimensionality(structure)

                self.assertEqual(result["dimensionality"], name)
                self.assertEqual(result["vacuum_directions"], vacuum_directions)
                self.assertEqual(result["periodic_dimensions"], 3 - len(vacuum_directions))
                self.assertTrue(result["periodic"])

    def test_classifies_a_cell_less_molecule(self) -> None:
        path = self.directory / "molecule.xyz"
        path.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
        result = classify_dimensionality(AbacusSTRU.read(path))

        self.assertEqual(result["dimensionality"], "molecule")
        self.assertFalse(result["periodic"])
        self.assertEqual(result["periodic_dimensions"], 0)

    def test_min_vacuum_changes_the_classification(self) -> None:
        slab = self._structure("slab.STRU", SLAB_CELL, SLAB_POSITIONS)

        self.assertEqual(classify_dimensionality(slab, min_vacuum=5.0)["dimensionality"], "slab")
        self.assertEqual(classify_dimensionality(slab, min_vacuum=9.0)["dimensionality"], "bulk")
        with self.assertRaises(ValueError):
            classify_dimensionality(slab, min_vacuum=0.0)

    def test_a_wire_and_a_molecule_have_no_single_vacuum_direction(self) -> None:
        wire = self._structure("wire.STRU", WIRE_CELL, WIRE_POSITIONS)
        molecule = self._structure("molecule.STRU", MOLECULE_CELL, MOLECULE_POSITIONS)

        self.assertEqual(detect_aperiodic_direction(wire), None)
        self.assertEqual(detect_aperiodic_direction(molecule), None)


class TestCoordinationAnalysis(unittest.TestCase):
    """Cover the coordination analysis and the choice of its method."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _structure(
        self,
        name: str,
        cell: list[list[float]],
        positions: list[list[float]],
    ) -> AbacusSTRU:
        path = self.directory / name
        path.write_text(_stru_text(cell, positions), encoding="utf-8")
        return AbacusSTRU.read(path)

    def test_bulk_structure_uses_the_coordination_environments(self) -> None:
        structure = self._structure("bulk.STRU", DIAMOND_CELL, DIAMOND_POSITIONS)
        result = coordination_analysis(structure)

        self.assertTrue(result["available"])
        self.assertEqual(result["requested_method"], "auto")
        self.assertEqual(result["method"], "chemenv")
        self.assertEqual(result["dimensionality"]["dimensionality"], "bulk")
        self.assertEqual([site["geometry"] for site in result["sites"]], ["T:4", "T:4"])
        self.assertEqual([site["coordination_number"] for site in result["sites"]], [4, 4])
        self.assertLess(result["sites"][0]["csm"], 1e-6)
        self.assertEqual(result["notes"], [])

    def test_slab_uses_nearest_neighbours_and_reports_surface_sites(self) -> None:
        bulk = self._structure("bulk.STRU", DIAMOND_CELL, DIAMOND_POSITIONS)
        slab = build_slab(bulk, miller_indices=(1, 0, 0), layers=3, vacuum=12.0)
        result = coordination_analysis(slab)

        self.assertTrue(result["available"])
        self.assertEqual(result["method"], "crystalnn")
        self.assertEqual(result["dimensionality"]["dimensionality"], "slab")
        self.assertTrue(any("chemenv needs a three-dimensional" in note for note in result["notes"]))
        numbers = [site["coordination_number"] for site in result["sites"]]
        # Every atom of the two outer layers loses one bond of the four.
        self.assertEqual(min(numbers), 3)
        self.assertEqual(max(numbers), 4)
        self.assertTrue(
            all(site["neighbors"] == {"Si": site["coordination_number"]} for site in result["sites"])
        )

    def test_an_explicit_method_overrides_the_dimensionality(self) -> None:
        slab = self._structure("slab.STRU", SLAB_CELL, SLAB_POSITIONS)
        automatic = coordination_analysis(slab)
        voronoi = coordination_analysis(slab, method="voronoi")

        self.assertEqual(voronoi["requested_method"], "voronoi")
        self.assertEqual(voronoi["method"], "voronoi")
        self.assertEqual(voronoi["notes"], [])
        self.assertNotEqual(
            [site["coordination_number"] for site in voronoi["sites"]],
            [site["coordination_number"] for site in automatic["sites"]],
        )

    def test_a_low_dimensional_structure_is_not_routed_to_chemenv(self) -> None:
        for name, cell, positions in (
            ("wire", WIRE_CELL, WIRE_POSITIONS),
            ("molecule", MOLECULE_CELL, MOLECULE_POSITIONS),
        ):
            with self.subTest(dimensionality=name):
                structure = self._structure(f"{name}.STRU", cell, positions)
                result = coordination_analysis(structure)

                self.assertTrue(result["available"])
                self.assertEqual(result["method"], "crystalnn")
                self.assertEqual(result["dimensionality"]["dimensionality"], name)
                self.assertTrue(any("chemenv needs a three-dimensional" in note for note in result["notes"]))

    def test_a_periodic_chain_keeps_both_of_its_neighbours(self) -> None:
        wire = self._structure("wire.STRU", WIRE_CELL, WIRE_POSITIONS)
        result = coordination_analysis(wire, method="crystalnn")

        self.assertEqual([site["coordination_number"] for site in result["sites"]], [2, 2, 2, 2])

    def test_a_cluster_keeps_its_terminal_atoms(self) -> None:
        molecule = self._structure("molecule.STRU", MOLECULE_CELL, MOLECULE_POSITIONS)
        result = coordination_analysis(molecule, method="crystalnn")

        self.assertEqual([site["coordination_number"] for site in result["sites"]], [1, 2, 1])

    def test_a_cell_less_structure_cannot_be_analysed(self) -> None:
        path = self.directory / "molecule.xyz"
        path.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
        result = coordination_analysis(AbacusSTRU.read(path))

        self.assertFalse(result["available"])
        self.assertIn("periodic cell", result["error"])
        self.assertEqual(result["sites"], [])

    def test_rejects_an_unknown_method(self) -> None:
        structure = self._structure("bulk.STRU", DIAMOND_CELL, DIAMOND_POSITIONS)
        with self.assertRaises(ValueError):
            coordination_analysis(structure, method="crystalball")


if __name__ == "__main__":
    unittest.main()
