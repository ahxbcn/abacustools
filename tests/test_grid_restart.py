"""Tests for ABACUS restart charge-density I/O and FFT conversion."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest

import numpy as np

from abacustools.data.grid import (
    Charge,
    RestartCharge,
    miller_indices_within_cutoff,
)


def _full_grid_miller(shape):
    fractions = [np.fft.fftfreq(size) * size for size in shape]
    mesh = np.meshgrid(*fractions, indexing="ij")
    return np.stack([component.ravel() for component in mesh], axis=1).astype(np.int64)


class TestRestartChargeBinary(unittest.TestCase):
    def setUp(self) -> None:
        self.reciprocal = np.array(
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
        )
        rng = np.random.default_rng(1)
        self.miller = rng.integers(-3, 4, size=(24, 3)).astype(np.int64)
        self.rhog = rng.standard_normal((2, 24)) + 1j * rng.standard_normal((2, 24))

    def test_binary_roundtrip_preserves_data(self) -> None:
        restart = RestartCharge(self.rhog, self.miller, self.reciprocal)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "chg.restart")
            restart.write(path)
            loaded = RestartCharge.read(path)

        np.testing.assert_array_equal(loaded.miller, self.miller)
        np.testing.assert_allclose(loaded.rhog, self.rhog)
        np.testing.assert_allclose(loaded.reciprocal_lattice, self.reciprocal)
        self.assertEqual(loaded.nspin, 2)
        self.assertEqual(loaded.ngm, 24)
        self.assertFalse(loaded.gamma_only)

    def test_file_size_matches_documented_layout(self) -> None:
        restart = RestartCharge(self.rhog, self.miller, self.reciprocal)
        ngm, nspin = 24, 2
        expected = 20 + (4 + 72 + 4) + (4 + 3 * ngm * 4 + 4) + nspin * (4 + 16 * ngm + 4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "chg.restart")
            restart.write(path)
            self.assertEqual(os.path.getsize(path), expected)

    def test_header_markers_match_abacus_layout(self) -> None:
        restart = RestartCharge(self.rhog, self.miller, self.reciprocal)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "chg.restart")
            restart.write(path)
            with open(path, "rb") as handle:
                header = struct.unpack("<5i", handle.read(20))
            self.assertEqual(header, (3, 0, 24, 2, 3))

    def test_read_rejects_truncated_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "broken.restart")
            with open(path, "wb") as handle:
                handle.write(struct.pack("<5i", 3, 0, 24, 2, 3))
            with self.assertRaises(ValueError):
                RestartCharge.read(path)

    def test_read_rejects_wrong_leading_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "broken.restart")
            with open(path, "wb") as handle:
                handle.write(struct.pack("<5i", 4, 0, 24, 2, 3))
            with self.assertRaises(ValueError):
                RestartCharge.read(path)


class TestRestartChargeFFT(unittest.TestCase):
    def test_real_reciprocal_roundtrip_on_full_grid(self) -> None:
        rng = np.random.default_rng(2)
        shape = (8, 8, 8)
        rho = rng.standard_normal(shape)
        miller = _full_grid_miller(shape)
        reciprocal = np.eye(3)

        restart = RestartCharge.from_real(rho, reciprocal, miller)
        recovered = restart.to_real(shape)[0]

        np.testing.assert_allclose(recovered, rho, atol=1e-12)

    def test_real_reciprocal_roundtrip_with_cutoff_sphere(self) -> None:
        rng = np.random.default_rng(3)
        shape = (10, 10, 10)
        cell = np.diag([8.0, 8.0, 8.0])
        reciprocal = np.linalg.inv(cell)
        miller = miller_indices_within_cutoff(2 * np.pi * reciprocal, cutoff=1.0)

        coefficients = rng.standard_normal(miller.shape[0]) + 1j * rng.standard_normal(
            miller.shape[0]
        )
        full = np.zeros(shape, dtype=np.complex128)
        index = np.mod(miller, np.array(shape))
        full[index[:, 0], index[:, 1], index[:, 2]] = coefficients
        rho = (np.fft.ifftn(full) * int(np.prod(shape))).real

        restart = RestartCharge.from_real(rho, reciprocal, miller)
        recovered = restart.to_real(shape)[0]

        np.testing.assert_allclose(recovered, rho, atol=1e-12)

    def test_gamma_only_conversion_is_not_implemented(self) -> None:
        restart = RestartCharge(np.zeros((1, 1), dtype=complex), np.zeros((1, 3)), np.eye(3), gamma_only=True)
        with self.assertRaises(NotImplementedError):
            restart.to_real((4, 4, 4))
        with self.assertRaises(NotImplementedError):
            RestartCharge.from_real(np.zeros((4, 4, 4)), np.eye(3), np.zeros((1, 3)), gamma_only=True)

    def test_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            RestartCharge(np.zeros((2, 3), dtype=complex), np.zeros((4, 3)), np.eye(3))
        with self.assertRaises(ValueError):
            RestartCharge(np.zeros((1, 3), dtype=complex), np.zeros((3, 3)), np.eye(2))


class TestMillerIndices(unittest.TestCase):
    def test_contains_origin_and_respects_cutoff(self) -> None:
        reciprocal = 2 * np.pi * np.eye(3)
        cutoff = 0.5
        miller = miller_indices_within_cutoff(reciprocal, cutoff)

        self.assertTrue(np.any(np.all(miller == 0, axis=1)))
        squared = np.sum((miller @ reciprocal) ** 2, axis=1)
        self.assertTrue(np.all(squared <= cutoff * (1 + 1e-12)))

    def test_output_is_sorted_and_unique(self) -> None:
        reciprocal = 2 * np.pi * np.eye(3)
        miller = miller_indices_within_cutoff(reciprocal, cutoff=2.0)
        squared = np.sum((miller @ reciprocal) ** 2, axis=1)
        self.assertTrue(np.all(np.diff(squared) >= -1e-12))
        self.assertEqual(len(np.unique(miller, axis=0)), miller.shape[0])

    def test_rejects_non_positive_cutoff(self) -> None:
        with self.assertRaises(ValueError):
            miller_indices_within_cutoff(np.eye(3), cutoff=0.0)


class TestChargeRestart(unittest.TestCase):
    def test_charge_save_and_from_restart_roundtrip(self) -> None:
        rng = np.random.default_rng(4)
        shape = (8, 8, 8)
        cell = np.diag([5.0, 5.0, 5.0])
        charge = Charge(rng.standard_normal(shape), cell, np.zeros((0, 3)))
        miller = _full_grid_miller(shape)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "charge.restart")
            charge.save_restart(path, miller)
            restored = Charge.from_restart(path, shape, cell=cell)

        np.testing.assert_allclose(restored.data, charge.data, atol=1e-12)
        np.testing.assert_allclose(restored.cell, cell)

    def test_from_restart_reconstructs_cell_from_reciprocal(self) -> None:
        rng = np.random.default_rng(5)
        shape = (6, 6, 6)
        cell = np.array([[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
        charge = Charge(rng.standard_normal(shape), cell, np.zeros((0, 3)))
        miller = _full_grid_miller(shape)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "charge.restart")
            charge.save_restart(path, miller)
            restored = Charge.from_restart(path, shape)

        np.testing.assert_allclose(restored.cell, cell, atol=1e-9)

    def test_from_restart_rejects_out_of_range_spin(self) -> None:
        shape = (4, 4, 4)
        charge = Charge(np.zeros(shape), np.diag([5.0, 5.0, 5.0]), np.zeros((0, 3)))
        miller = _full_grid_miller(shape)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "charge.restart")
            charge.save_restart(path, miller)
            with self.assertRaises(IndexError):
                Charge.from_restart(path, shape, spin=1)


if __name__ == "__main__":
    unittest.main()
