"""Tests for the stress-strain elastic workflow helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from abacustools.commands.workflow.elastic import (
    _fit_tensor,
    _pymatgen_deformations,
    prepare,
)


STRU = """ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
4 0 0
0 4 0
0 0 4

ATOMIC_POSITIONS
Direct

H
0.0
1
0 0 0
"""


#: A graphene sheet with 20 Angstrom of vacuum along c.
SLAB_STRU = """ATOMIC_SPECIES
C 12.011 C.upf

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
2.46 0 0
-1.23 2.1304 0
0 0 20

ATOMIC_POSITIONS
Direct

C
0.0
2
0.333333 0.666667 0.5
0.666667 0.333333 0.5
"""


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

        fitted = _fit_tensor(strains, stresses, equilibrium)

        np.testing.assert_allclose(fitted, expected, atol=1e-10)

    def test_prepare_records_the_symmetry_of_the_reference_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
            )
            (job / "STRU").write_text(STRU, encoding="utf-8")
            (job / "H.upf").write_text("pseudo", encoding="utf-8")

            exit_code = prepare(
                Namespace(
                    job=job,
                    norm=0.01,
                    shear=0.01,
                    norelax=False,
                    override=False,
                    strains="full",
                )
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads(
                (job / "workflow_elastic.json").read_text(encoding="utf-8")
            )
            symmetry = manifest["symmetry"]
            self.assertEqual(symmetry["point_group"], "m-3m")
            self.assertEqual(symmetry["space_group_number"], 221)
            self.assertEqual(symmetry["crystal_system"], "cubic")
            self.assertEqual(symmetry["independent_constants"], 3)
            self.assertEqual(symmetry["operations"], 48)
            self.assertEqual(len(manifest["tasks"]), 25)
            self.assertTrue((job / "org" / "STRU").is_file())

    def test_prepare_can_restrict_the_strains_to_the_independent_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
            )
            (job / "STRU").write_text(STRU, encoding="utf-8")
            (job / "H.upf").write_text("pseudo", encoding="utf-8")

            exit_code = prepare(
                Namespace(
                    job=job,
                    norm=0.01,
                    shear=0.01,
                    norelax=True,
                    override=False,
                    strains="independent",
                )
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads(
                (job / "workflow_elastic.json").read_text(encoding="utf-8")
            )
            # A cubic cell needs the xx normal strain and the yz shear only.
            self.assertEqual(manifest["strain_modes"], [0, 3])
            self.assertEqual(manifest["strains_mode"], "independent")
            self.assertEqual(len(manifest["strains"]), 8)
            self.assertEqual(len(manifest["tasks"]), 9)
            self.assertEqual(
                manifest["deformed_paths"],
                [f"deformed_{index:02d}" for index in range(8)],
            )
            self.assertTrue((job / "deformed_07" / "STRU").is_file())
            self.assertFalse((job / "deformed_08").exists())

    def test_prepare_restricts_a_slab_to_the_in_plane_strains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
            )
            (job / "STRU").write_text(SLAB_STRU, encoding="utf-8")
            (job / "C.upf").write_text("pseudo", encoding="utf-8")

            exit_code = prepare(
                Namespace(
                    job=job,
                    norm=0.01,
                    shear=0.01,
                    norelax=True,
                    override=False,
                    strains="independent",
                    dimension="auto",
                )
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads(
                (job / "workflow_elastic.json").read_text(encoding="utf-8")
            )
            dimension = manifest["dimension"]
            self.assertEqual(dimension["dimensionality"], "slab")
            self.assertEqual(dimension["vacuum_direction"], "c")
            self.assertEqual(dimension["vacuum_axis"], 2)
            self.assertEqual(dimension["in_plane_modes"], [0, 1, 5])
            self.assertAlmostEqual(dimension["cell_height"], 20.0, places=6)
            # In the plane a hexagonal sheet has two independent constants and
            # one strain direction determines both of them.
            self.assertEqual(manifest["symmetry"]["independent_constants"], 2)
            self.assertEqual(manifest["strain_modes"], [0])
            self.assertEqual(len(manifest["tasks"]), 5)

    def test_prepare_reports_an_unusable_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
            )
            (job / "STRU").write_text(
                STRU.replace("0 0 4", "0 0 0"), encoding="utf-8"
            )
            (job / "H.upf").write_text("pseudo", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                prepare(
                    Namespace(
                        job=job,
                        norm=0.01,
                        shear=0.01,
                        norelax=False,
                        override=False,
                        strains="full",
                    )
                )


if __name__ == "__main__":
    unittest.main()
