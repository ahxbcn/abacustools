"""Tests for the stress-strain elastic workflow helpers."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from abacustools.commands.workflow.elastic import (
    _elastic_moduli,
    _fit_elastic_tensor,
    _pymatgen_deformations,
)


def _stress_from_voigt(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [values[0], values[5], values[4]],
            [values[5], values[1], values[3]],
            [values[4], values[3], values[2]],
        ]
    )


class TestElasticWorkflow(unittest.TestCase):
    def test_generates_24_independent_strain_states(self) -> None:
        from pymatgen.core import Structure

        structure = Structure(
            lattice=np.eye(3), species=["H"], coords=[[0.0, 0.0, 0.0]]
        )
        structure_wrapper = Mock()
        structure_wrapper.to.return_value = structure
        states = _pymatgen_deformations(structure_wrapper, 0.01, 0.02)

        self.assertEqual(len(states), 24)
        structure_wrapper.to.assert_called_once_with("pymatgen")
        for deformed, strain in states:
            self.assertEqual(deformed.lattice.matrix.shape, (3, 3))
            self.assertEqual(strain.shape, (3, 3))
        np.testing.assert_allclose(
            states[0][0].lattice.matrix,
            np.diag([np.sqrt(0.98), 1.0, 1.0]),
        )
        np.testing.assert_allclose(states[0][1].voigt, [-0.01, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(
            states[12][1].voigt,
            [0, 0, 0, 0, 0, -0.04],
            atol=1e-12,
        )

    def test_fits_tensor_from_synthetic_stresses(self) -> None:
        expected = np.array(
            [
                [100.0, 20.0, 30.0, 0.0, 0.0, 0.0],
                [20.0, 110.0, 25.0, 0.0, 0.0, 0.0],
                [30.0, 25.0, 120.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 40.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 45.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
            ]
        )
        from pymatgen.core import Structure

        structure = Structure(
            lattice=np.eye(3), species=["H"], coords=[[0.0, 0.0, 0.0]]
        )
        structure_wrapper = Mock()
        structure_wrapper.to.return_value = structure
        states = _pymatgen_deformations(structure_wrapper, 0.01, 0.01)
        strains = [strain.as_dict() for _, strain in states]
        equilibrium = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        stresses = [
            equilibrium + _stress_from_voigt(np.asarray(strain.voigt) @ expected)
            for _, strain in states
        ]

        fitted = _fit_elastic_tensor(strains, stresses, equilibrium)

        np.testing.assert_allclose(fitted, expected, atol=1e-10)

    def test_calculates_voigt_moduli(self) -> None:
        tensor = np.array(
            [
                [100.0, 20.0, 30.0, 0.0, 0.0, 0.0],
                [20.0, 110.0, 25.0, 0.0, 0.0, 0.0],
                [30.0, 25.0, 120.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 40.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 45.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
            ]
        )

        result = _elastic_moduli(tensor)

        self.assertAlmostEqual(result["bulk_modulus"], 53.333333333333336)
        self.assertAlmostEqual(result["shear_modulus"], 44.0)
        self.assertAlmostEqual(
            result["young_modulus"], 9.0 * result["bulk_modulus"] * 44.0 / (3.0 * result["bulk_modulus"] + 44.0)
        )
        self.assertAlmostEqual(
            result["poisson_ratio"],
            (3.0 * result["bulk_modulus"] - 2.0 * 44.0) / (2.0 * (3.0 * result["bulk_modulus"] + 44.0)),
        )


if __name__ == "__main__":
    unittest.main()
