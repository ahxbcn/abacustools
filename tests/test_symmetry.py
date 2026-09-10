"""Tests for the symmetry analysis helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from abacustools.data.symmetry import (
    crystallographic_symmetry,
    detect_aperiodic_direction,
    layer_symmetry,
    magnetic_ordering,
    magnetic_symmetry,
    site_symmetry_symbols,
)
from abacustools.data.structure import make_supercell
from abacustools.io.stru import AbacusSTRU


DIAMOND_CELL = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
DIAMOND_POSITIONS = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
CUBIC_CELL = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
SLAB_CELL = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 20.0]]
SLAB_POSITIONS = [
    [0.25, 0.25, 0.30],
    [0.75, 0.75, 0.30],
    [0.25, 0.75, 0.70],
    [0.75, 0.25, 0.70],
]

# Carbon and oxygen stacked along the fourfold axis: the two species differ,
# so the cell keeps the mirror planes containing that axis and P4mm survives.
POLAR_STRU = """ATOMIC_SPECIES
C 12.011 C.upf
O 15.999 O.upf

NUMERICAL_ORBITAL
C.orb
O.orb

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
5.669 0 0
0 5.669 0
0 0 9.448

ATOMIC_POSITIONS
Direct

C
0.0
1
0.0 0.0 0.0

O
0.0
1
0.0 0.0 0.4
"""


def _stru_text(
    cell: list[list[float]],
    positions: list[list[float]],
    moments: list[float] | None = None,
    element: str = "Si",
) -> str:
    """Build a minimal STRU file with the given cell, positions and moments."""
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
    for index, position in enumerate(positions):
        line = " ".join(str(component) for component in position)
        if moments is not None:
            line += f" mag {moments[index]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


class TestSymmetryAnalysis(unittest.TestCase):
    """Cover the space group, magnetic, layer and Wyckoff analyses."""

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
        moments: list[float] | None = None,
    ) -> AbacusSTRU:
        path = self.directory / name
        path.write_text(_stru_text(cell, positions, moments), encoding="utf-8")
        return AbacusSTRU.read(path)

    def test_space_group_wyckoff_and_site_symmetry(self) -> None:
        structure = self._structure("diamond.STRU", DIAMOND_CELL, DIAMOND_POSITIONS)
        symmetry = crystallographic_symmetry(structure)

        self.assertTrue(symmetry["available"])
        self.assertEqual(symmetry["space_group_number"], 227)
        self.assertEqual(symmetry["space_group_symbol"], "Fd-3m")
        self.assertEqual(symmetry["wyckoff_positions"], ["b", "b"])
        self.assertEqual(symmetry["wyckoff_multiplicities"], [8, 8])
        self.assertEqual(symmetry["equivalent_atoms"], [1, 1])
        self.assertEqual(symmetry["schoenflies"], "Oh")
        self.assertEqual(symmetry["bravais_lattice"], "cF")
        self.assertEqual(symmetry["pearson_symbol"], "cF8")
        self.assertTrue(symmetry["inversion_symmetry"])
        self.assertFalse(symmetry["polar_point_group"])
        self.assertEqual(site_symmetry_symbols(structure), ["-43m", "-43m"])

    def test_a_polar_structure_keeps_one_invariant_direction(self) -> None:
        path = self.directory / "polar.STRU"
        path.write_text(POLAR_STRU, encoding="utf-8")
        symmetry = crystallographic_symmetry(AbacusSTRU.read(path))

        self.assertEqual(symmetry["space_group_symbol"], "P4mm")
        self.assertEqual(symmetry["point_group"], "4mm")
        self.assertEqual(symmetry["schoenflies"], "C4v")
        self.assertEqual(symmetry["bravais_lattice"], "tP")
        self.assertFalse(symmetry["inversion_symmetry"])
        self.assertTrue(symmetry["polar_point_group"])

    def test_a_wyckoff_multiplicity_does_not_follow_the_cell(self) -> None:
        structure = self._structure("diamond.STRU", DIAMOND_CELL, DIAMOND_POSITIONS)
        primitive = crystallographic_symmetry(structure)
        supercell = crystallographic_symmetry(make_supercell(structure, (2, 2, 2)))

        # Both cells describe the eightfold Wyckoff position of the crystal,
        # even though the supercell holds twice as many equivalent atoms.
        self.assertEqual(primitive["wyckoff_multiplicities"], [8, 8])
        self.assertEqual(set(supercell["wyckoff_multiplicities"]), {8})
        self.assertEqual(len(supercell["equivalent_atoms"]), 16)
        self.assertEqual(set(supercell["equivalent_atoms"]), {1})

    def test_a_general_position_keeps_a_single_atom_orbit(self) -> None:
        structure = self._structure(
            "p1.STRU",
            DIAMOND_CELL,
            [[0.1, 0.2, 0.3], [0.23, 0.31, 0.17], [0.4, 0.15, 0.28]],
        )
        symmetry = crystallographic_symmetry(structure)

        self.assertEqual(symmetry["space_group_number"], 1)
        self.assertEqual(symmetry["wyckoff_positions"], ["a", "a", "a"])
        # The general position of P1 carries one atom per orbit, not one atom
        # per letter.
        self.assertEqual(symmetry["wyckoff_multiplicities"], [1, 1, 1])
        self.assertEqual(symmetry["equivalent_atoms"], [1, 2, 3])
        self.assertEqual(symmetry["bravais_lattice"], "aP")
        self.assertFalse(symmetry["inversion_symmetry"])
        self.assertTrue(symmetry["polar_point_group"])

    def test_symmetry_is_unavailable_without_a_periodic_cell(self) -> None:
        path = self.directory / "molecule.xyz"
        path.write_text("1\nH molecule\nH 0 0 0\n", encoding="utf-8")
        structure = AbacusSTRU.read(path)

        self.assertIsNone(site_symmetry_symbols(structure))
        self.assertFalse(crystallographic_symmetry(structure)["available"])
        self.assertFalse(magnetic_symmetry(structure)["available"])
        self.assertFalse(layer_symmetry(structure)["available"])

    def test_non_magnetic_structure_keeps_the_grey_group(self) -> None:
        structure = self._structure("diamond.STRU", DIAMOND_CELL, DIAMOND_POSITIONS)
        symmetry = magnetic_symmetry(structure)

        self.assertTrue(symmetry["available"])
        self.assertTrue(symmetry["grey"])
        self.assertEqual(symmetry["type"], "type II (grey)")
        self.assertEqual(symmetry["msg_type"], 2)
        self.assertTrue(symmetry["label"].endswith("1'"))
        self.assertEqual(symmetry["operations_with_time_reversal"], symmetry["operations"] // 2)
        self.assertFalse(magnetic_ordering(structure)["available"])

    def test_ferromagnetic_structure_loses_time_reversal(self) -> None:
        structure = self._structure("fm.STRU", CUBIC_CELL, DIAMOND_POSITIONS, moments=[1.0, 1.0])
        symmetry = magnetic_symmetry(structure)

        self.assertTrue(symmetry["available"])
        self.assertFalse(symmetry["grey"])
        self.assertEqual(symmetry["type"], "type I")
        self.assertEqual(symmetry["operations_with_time_reversal"], 0)

    def test_antiferromagnetic_structure_keeps_time_reversal_operators(self) -> None:
        structure = self._structure("afm.STRU", CUBIC_CELL, DIAMOND_POSITIONS, moments=[1.0, -1.0])
        symmetry = magnetic_symmetry(structure)

        self.assertTrue(symmetry["available"])
        self.assertEqual(symmetry["type"], "type III")
        self.assertGreater(symmetry["operations_with_time_reversal"], 0)
        self.assertLess(symmetry["operations_with_time_reversal"], symmetry["operations"])
        self.assertIn("'", symmetry["label"])

    def test_magnetic_ordering_classification(self) -> None:
        positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]
        cases = {
            "fm": ([1.0, 1.0, 1.0], "FM", 3.0),
            "ferrimagnetic": ([1.0, -1.0, -1.0], "FiM", -1.0),
            "compensated": ([2.0, -1.0, -1.0], "AFM", 0.0),
        }
        for name, (moments, ordering, net_moment) in cases.items():
            with self.subTest(ordering=ordering):
                structure = self._structure(
                    f"{name}.STRU", CUBIC_CELL, positions, moments=moments
                )
                result = magnetic_ordering(structure)

                self.assertTrue(result["available"])
                self.assertEqual(result["ordering"], ordering)
                self.assertAlmostEqual(result["net_moment"], net_moment)
                self.assertEqual(result["magnetic_sites"], 3)

    def test_detects_the_vacuum_direction_of_a_slab(self) -> None:
        slab = self._structure("slab.STRU", SLAB_CELL, SLAB_POSITIONS)
        bulk = self._structure("diamond.STRU", DIAMOND_CELL, DIAMOND_POSITIONS)

        self.assertEqual(detect_aperiodic_direction(slab), "c")
        self.assertIsNone(detect_aperiodic_direction(bulk))

    def test_layer_symmetry_of_a_slab(self) -> None:
        slab = self._structure("slab.STRU", SLAB_CELL, SLAB_POSITIONS)
        result = layer_symmetry(slab)

        self.assertTrue(result["available"])
        self.assertEqual(result["direction"], "c")
        self.assertGreater(result["operations"], 0)
        self.assertEqual(result["symbol"], result["symbol"].lower())

    def test_layer_symmetry_uses_an_explicit_or_detected_direction(self) -> None:
        bulk = self._structure("diamond.STRU", DIAMOND_CELL, DIAMOND_POSITIONS)

        detected = layer_symmetry(bulk)
        self.assertFalse(detected["available"])
        self.assertIn("no vacuum direction", detected["error"])
        self.assertIsNone(detected["requested_direction"])

        explicit = layer_symmetry(bulk, direction="a")
        self.assertTrue(explicit["available"])
        self.assertEqual(explicit["direction"], "a")
        self.assertEqual(explicit["requested_direction"], "a")

        detected_only = layer_symmetry(bulk, detect=False)
        self.assertFalse(detected_only["available"])
        self.assertIn("no vacuum direction", detected_only["error"])

    def test_rejects_an_unknown_layer_direction(self) -> None:
        slab = self._structure("slab.STRU", SLAB_CELL, SLAB_POSITIONS)
        result = layer_symmetry(slab, direction="q")

        self.assertFalse(result["available"])
        self.assertIn("unknown layer direction", result["error"])


if __name__ == "__main__":
    unittest.main()
