"""Regression tests for cube-file geometry precision and data layout."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.grid import Charge


class TestCubeGeometryPrecision(unittest.TestCase):
    """Regression guard: low-precision cube geometry corrupted 688 e by ~3e-3 e."""

    def setUp(self) -> None:
        self.cell = np.diag([14.62912486, 14.62912486, 28.99255424])

    def test_cube_roundtrip_preserves_cell_volume(self) -> None:
        charge = Charge(
            np.ones((6, 6, 8)), self.cell, np.zeros((1, 3)), [6], [4.0]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "charge.cube")
            charge.save_cube(path, format="abacus")
            restored = Charge.from_cube(path, format="abacus")

        volume_ratio = abs(np.linalg.det(restored.cell)) / abs(np.linalg.det(self.cell))
        self.assertAlmostEqual(volume_ratio, 1.0, places=10)

    def test_cube_roundtrip_preserves_integrated_charge(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.random((6, 6, 8))
        charge = Charge(data, self.cell, np.zeros((1, 3)), [6], [4.0])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "charge.cube")
            charge.save_cube(path, format="abacus")
            restored = Charge.from_cube(path, format="abacus")

        expected = np.sum(data) * abs(np.linalg.det(self.cell)) / data.size
        actual = (
            np.sum(restored.data)
            * abs(np.linalg.det(restored.cell))
            / restored.data.size
        )
        self.assertAlmostEqual(actual, expected, places=8)


    def test_every_row_of_the_inner_axis_starts_a_new_line(self) -> None:
        """Chargemol reads one (i, j) row per READ, so a row has to end a line."""
        data = np.arange(2 * 2 * 7, dtype=float).reshape(2, 2, 7)
        charge = Charge(data, np.diag([4.0, 4.0, 4.0]), np.zeros((1, 3)), [6], [4.0])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "charge.cube")
            charge.save_cube(path, format="abacus")
            with open(path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()

        rows = lines[6 + 1:]
        self.assertEqual(len(rows), 8)  # 2 x 2 rows, each of them 6 + 1 values
        self.assertEqual(
            [len(row.split()) for row in rows], [6, 1, 6, 1, 6, 1, 6, 1]
        )

    def test_row_wise_reading_recovers_every_value(self) -> None:
        """Emulate the row-wise reads of Chargemol's cube reader."""
        rng = np.random.default_rng(11)
        data = rng.random((3, 2, 7))
        charge = Charge(data, np.diag([4.0, 4.0, 4.0]), np.zeros((1, 3)), [6], [4.0])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "charge.cube")
            charge.save_cube(path, format="abacus")
            with open(path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            restored = Charge.from_cube(path, format="abacus")

        records = [line.split() for line in lines[6 + 1:]]
        values = []
        record = 0
        for _ in range(3 * 2):  # one READ of the inner dimension per row
            wanted = 7
            while wanted > 0:
                fields = records[record]
                position = 0
                while wanted > 0 and position < len(fields):
                    values.append(float(fields[position]))
                    position += 1
                    wanted -= 1
                # A list-directed READ leaves the rest of a record behind.
                record += 1
        # The file holds e/Bohr^3 while a Charge stores e/Angstrom^3.
        expected = restored.data.reshape(-1) * BOHR_TO_ANG**3
        self.assertEqual(len(values), data.size)
        self.assertTrue(np.allclose(values, expected, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
