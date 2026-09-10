"""Regression tests for cube-file geometry precision."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
