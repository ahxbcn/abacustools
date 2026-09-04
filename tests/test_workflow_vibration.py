"""Tests for the finite-difference molecular vibration workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from abacustools.commands.workflow.vibration import (
    _frequency_values,
    _hessian_from_forces,
    postprocess,
    _selected_atoms,
    _temperatures,
    prepare,
)


class TestVibrationWorkflow(unittest.TestCase):
    def test_selected_atoms_and_temperatures(self) -> None:
        self.assertEqual(_selected_atoms([3, 1], 3), [0, 2])
        self.assertEqual(_temperatures([100, 300, 3]), [100.0, 200.0, 300.0])
        self.assertEqual(_frequency_values(np.array([1 + 0j, 2j])), [1.0, -2.0])
        with self.assertRaises(ValueError):
            _selected_atoms([1, 1], 3)
        with self.assertRaises(ValueError):
            _temperatures([100, 300])

    def test_hessian_from_central_force_difference(self) -> None:
        displacement_tasks = [
            {
                "task": "vib/SCF/disp_1_x+",
                "atom": 1,
                "direction_index": 0,
                "sign": "+",
            },
            {
                "task": "vib/SCF/disp_1_x-",
                "atom": 1,
                "direction_index": 0,
                "sign": "-",
            },
        ]
        forces = {
            "vib/SCF/disp_1_x+": np.array([[-2.0, 0.0, 0.0]]),
            "vib/SCF/disp_1_x-": np.array([[2.0, 0.0, 0.0]]),
        }
        hessian = _hessian_from_forces(forces, displacement_tasks, [0], 0.1)
        self.assertAlmostEqual(hessian[0, 0], 20.0)

    def test_prepare_writes_equilibrium_and_displacements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation relax\ngamma_only 1\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(
                """ATOMIC_SPECIES
H 1.0

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Cartesian

H
0.0
1
1 2 3
""",
                encoding="utf-8",
            )
            result = prepare(
                Namespace(
                    job=job,
                    stepsize=0.01,
                    selected_atoms=[1],
                    override=False,
                )
            )
            self.assertEqual(result, 0)
            manifest = json.loads(
                (job / "workflow_vibration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["workflow"], "vibration")
            self.assertEqual(manifest["selected_atoms"], [1])
            self.assertEqual(len(manifest["tasks"]), 7)
            self.assertTrue((job / "vib/SCF/eq/STRU").is_file())
            self.assertTrue((job / "vib/SCF/disp_1_x+/STRU").is_file())
            self.assertIn("calculation", (job / "vib/SCF/eq/INPUT").read_text())
            self.assertIn("scf", (job / "vib/SCF/eq/INPUT").read_text())

    def test_postprocess_builds_frequencies_from_forces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(
                """ATOMIC_SPECIES
H 1.0

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Cartesian

H
0.0
1
1 2 3
""",
                encoding="utf-8",
            )
            prepare(Namespace(job=job, stepsize=0.01, selected_atoms=[1], override=False))

            def force_for(job_path, version, natoms):
                force = np.zeros((natoms, 3))
                if job_path.name != "eq":
                    _, atom, direction = job_path.name.split("_")
                    sign = 1.0 if direction.endswith("+") else -1.0
                    axis = {"x+": 0, "y+": 1, "z+": 2,
                            "x-": 0, "y-": 1, "z-": 2}[direction]
                    force[0, axis] = -20.0 * sign * 0.01
                return force

            with patch(
                "abacustools.commands.workflow.vibration._read_forces",
                side_effect=force_for,
            ):
                result = postprocess(
                    Namespace(
                        job=job,
                        version="LTS3.10.1",
                        temperature=[298.15],
                        traj=False,
                        traj_format="extxyz",
                        frames=30,
                        output_stru=False,
                        stru_format="extxyz",
                        output="vibration_results.json",
                    )
                )
            self.assertEqual(result, 0)
            vibration_result = json.loads(
                (job / "vibration_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(vibration_result["frequencies"]), 3)
            self.assertTrue(all(value > 2000 for value in vibration_result["frequencies"]))
            self.assertIn("298.15K", vibration_result["thermo_corr"])


if __name__ == "__main__":
    unittest.main()
