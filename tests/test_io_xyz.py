"""Tests for the extended XYZ writer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from abacustools.io.xyz import write_extxyz


def simple_frame(**overrides) -> dict:
    """Return one small frame, with optional overrides."""
    frame = {
        "elements": ["H", "O"],
        "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]],
        "cell": [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        "pbc": (True, True, True),
    }
    frame.update(overrides)
    return frame


class TestWriteExtxyz(unittest.TestCase):
    def test_writes_a_frame_with_cell_and_pbc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.xyz"
            count = write_extxyz(path, simple_frame())
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(count, 1)
            self.assertEqual(lines[0], "2")
            self.assertIn('Lattice="8.0 0.0', lines[1])
            self.assertIn("Properties=species:S:1:pos:R:3", lines[1])
            self.assertIn('pbc="T T T"', lines[1])
            self.assertTrue(lines[2].startswith("H "))
            self.assertEqual(len(lines), 4)

    def test_writes_momenta_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.xyz"
            write_extxyz(
                path,
                simple_frame(
                    momenta=[[0.0, 0.0, 0.1], [0.0, 0.0, -0.2]],
                    magmoms=[0.0, 1.5],
                    info={"pp": {"H": "H.upf"}, "dpks": None, "energy": -12.5},
                ),
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertIn(
                "Properties=species:S:1:pos:R:3:initial_magmoms:R:1:momenta:R:3",
                lines[1],
            )
            self.assertIn('pp="_JSON {\\"H\\": \\"H.upf\\"}"', lines[1])
            self.assertIn(" dpks ", lines[1])
            self.assertIn("energy=-12.5", lines[1])
            self.assertEqual(len(lines[2].split()), 8)

    def test_multiple_frames_and_directory_creation(self) -> None:
        from ase.io import read

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "trajectory.extxyz"
            frames = [
                simple_frame(positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9 + 0.1 * index]])
                for index in range(3)
            ]
            self.assertEqual(write_extxyz(path, frames), 3)
            self.assertEqual(len(read(path, index=":")), 3)
            self.assertEqual(
                [line for line in path.read_text(encoding="utf-8").splitlines() if line == "2"],
                ["2", "2", "2"],
            )

    def test_vector_magnetic_moments_use_three_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.xyz"
            write_extxyz(
                path,
                simple_frame(magmoms=[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertIn("initial_magmoms:R:3", lines[1])
            self.assertEqual(len(lines[2].split()), 7)

    def test_ase_reads_the_written_file_back(self) -> None:
        from ase.io import read

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.xyz"
            write_extxyz(
                path,
                simple_frame(
                    momenta=[[0.0, 0.0, 0.1], [0.0, 0.0, -0.2]],
                    magmoms=[0.0, 1.5],
                ),
            )
            atoms = read(path)
            np.testing.assert_allclose(atoms.positions[1], [0.0, 0.0, 0.96])
            np.testing.assert_allclose(atoms.cell.array.diagonal(), 8.0)
            np.testing.assert_allclose(atoms.get_initial_magnetic_moments(), [0.0, 1.5])
            masses = atoms.get_masses()
            np.testing.assert_allclose(
                atoms.get_velocities(),
                np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.2]]) / masses[:, None],
            )

    def test_invalid_frames_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.xyz"
            with self.assertRaises(ValueError):
                write_extxyz(path, {"elements": ["H"]})
            with self.assertRaises(ValueError):
                write_extxyz(path, {"positions": [[0.0, 0.0, 0.0]]})
            with self.assertRaises(ValueError):
                write_extxyz(path, {"elements": ["H"], "positions": [[0.0, 0.0]]})
            with self.assertRaises(ValueError):
                write_extxyz(path, simple_frame(momenta=[[0.0, 0.0, 0.1]]))
            with self.assertRaises(ValueError):
                write_extxyz(path, simple_frame(cell=[[1.0, 0.0, 0.0]]))
            with self.assertRaises(ValueError):
                write_extxyz(path, simple_frame(info="not a mapping"))


if __name__ == "__main__":
    unittest.main()
