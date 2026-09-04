"""Tests for unit-cell geometry and coordinate conversions."""

from __future__ import annotations

import unittest

import numpy as np

from abacustools.data.unitcell import Unitcell


class TestUnitcell(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = Unitcell(
            [
                [2.0, 0.0, 0.0],
                [1.0, np.sqrt(3.0), 0.0],
                [0.0, 0.0, 3.0],
            ]
        )

    def test_cell_parameters_are_in_degrees(self) -> None:
        np.testing.assert_allclose(
            self.cell.get_cell_param(), [2.0, 2.0, 3.0, 90.0, 90.0, 60.0]
        )
        self.assertAlmostEqual(self.cell.get_cell_volume(), 6.0 * np.sqrt(3.0))

    def test_coordinate_conversions_are_inverse_without_wrapping(self) -> None:
        fractional = [[0.25, -0.5, 1.5], [1.2, 0.1, -0.3]]
        cartesian = self.cell.frac_to_cart(fractional, wrap=False)

        np.testing.assert_allclose(
            self.cell.cart_to_frac(cartesian, wrap=False), fractional
        )

    def test_cart_to_frac_wraps_each_component_to_unit_interval(self) -> None:
        cartesian = self.cell.frac_to_cart([[1.25, -0.5, 2.0]], wrap=False)

        np.testing.assert_allclose(self.cell.cart_to_frac(cartesian), [[0.25, 0.5, 0.0]])

    def test_frac_to_cart_wraps_fractional_coordinates_by_default(self) -> None:
        wrapped = self.cell.frac_to_cart([[1.25, -0.5, 2.0]])
        unwrapped_equivalent = self.cell.frac_to_cart([[0.25, 0.5, 0.0]], wrap=False)

        np.testing.assert_allclose(wrapped, unwrapped_equivalent)

    def test_reciprocal_vectors_satisfy_definition_for_nonorthogonal_cell(self) -> None:
        reciprocal = self.cell.reciprocal()

        np.testing.assert_allclose(
            reciprocal.cell_vector @ self.cell.cell_vector.T,
            np.eye(3),
            atol=1e-12,
        )

    def test_empty_coordinate_list_is_supported(self) -> None:
        self.assertEqual(self.cell.cart_to_frac([]), [])
        self.assertEqual(self.cell.frac_to_cart([]), [])

    def test_rejects_invalid_cell_shapes_and_singular_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            Unitcell([[1.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "non-singular"):
            Unitcell(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            )

    def test_rejects_invalid_coordinate_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "cart_coords"):
            self.cell.cart_to_frac([[1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "frac_coords"):
            self.cell.frac_to_cart([[1.0, 2.0, np.nan]])

    def test_left_handed_cell_warns_and_has_positive_volume(self) -> None:
        with self.assertWarnsRegex(UserWarning, "left-handed"):
            cell = Unitcell(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            )

        self.assertEqual(cell.get_cell_volume(), 1.0)


if __name__ == "__main__":
    unittest.main()
