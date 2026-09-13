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
    _animation_velocity_scale,
    _frequency_label,
    _frequency_values,
    _hessian_from_forces,
    _write_modes,
    postprocess,
    _selected_atoms,
    _temperatures,
    prepare,
)
from abacustools.core.constant import BOLTZMANN_CONSTANT_EV_PER_K, INV_CM_TO_EV
from abacustools.core.submission import generate_workflow_submission, resolve_submission


class TestVibrationWorkflow(unittest.TestCase):
    def test_submission_scripts_honor_equilibrium_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            result = generate_workflow_submission(
                job,
                "vibration",
                ["vib/SCF/eq", "vib/SCF/disp_1_x+", "vib/SCF/disp_1_x-"],
                submission_type="local",
                generate=True,
                abacus_command="mpirun -np 2 abacus",
            )
            self.assertEqual(result["type"], "local")
            self.assertTrue((job / "vib/SCF/eq/run.sh").stat().st_mode & 0o111)
            launcher = (job / "submit_vibration.sh").read_text(encoding="utf-8")
            self.assertIn("./run.sh)", launcher)
            self.assertIn("pids+=(\"$!\")", launcher)
            self.assertLess(launcher.index("/eq"), launcher.index("/disp_1_x+"))
            self.assertIn("mpirun -np 2 abacus", (job / "vib/SCF/eq/run.sh").read_text())

    def test_scheduler_submission_uses_configured_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            generate_workflow_submission(
                job,
                "vibration",
                ["vib/SCF/eq", "vib/SCF/disp_1_x+"],
                submission_type="slurm",
                generate=True,
            )
            launcher = (job / "submit_vibration.sh").read_text(encoding="utf-8")
            self.assertIn("sbatch --parsable submit.slurm", launcher)
            self.assertIn("--dependency=afterok:${equilibrium_id}", launcher)
            self.assertTrue((job / "vib/SCF/eq/submit.slurm").stat().st_mode & 0o111)

    def test_custom_submission_template_can_be_resolved(self) -> None:
        settings = resolve_submission(
            config={
                "submission": {
                    "generate": True,
                    "default": "custom",
                    "templates": {
                        "custom": {
                            "filename": "submit.sh",
                            "template": "#!/bin/sh\n{abacus_command}\n",
                            "launcher": {"mode": "local"},
                        }
                    },
                }
            }
        )
        self.assertEqual(settings["type"], "custom")
        self.assertEqual(settings["filename"], "submit.sh")

    def test_submission_scripts_are_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(
                generate_workflow_submission(Path(temporary), "vibration", ["vib/SCF/eq"])
            )

    def test_prepare_generates_configured_submission_scripts(self) -> None:
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
            self.assertEqual(
                prepare(
                    Namespace(
                        job=job,
                        stepsize=0.01,
                        selected_atoms=[1],
                        override=False,
                        generate_scripts=True,
                        submission_type="local",
                        abacus_command="abacus",
                    )
                ),
                0,
            )
            manifest = json.loads((job / "workflow_vibration.json").read_text())
            self.assertEqual(manifest["submission"]["workflow_script"], "submit_vibration.sh")
            self.assertTrue((job / "vib/SCF/disp_1_z-/run.sh").is_file())
            self.assertTrue((job / "submit_vibration.sh").stat().st_mode & 0o111)

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

    def test_frequency_label_marks_imaginary_modes(self) -> None:
        self.assertEqual(_frequency_label(3517.91 + 0j), "3517.91")
        self.assertEqual(_frequency_label(-43.06 + 0j), "43.06")
        self.assertEqual(_frequency_label(43.06j), "43.06i")

    def test_animation_velocity_scale_uses_reference_mode(self) -> None:
        kT = BOLTZMANN_CONSTANT_EV_PER_K * 300.0
        scale = _animation_velocity_scale(kT)
        reference_energy = 2500.0 * INV_CM_TO_EV
        displacement = np.sqrt(kT / reference_energy)
        self.assertAlmostEqual(scale * 2500.0 * displacement, 0.5, places=6)
        stiff_energy = 3500.0 * INV_CM_TO_EV
        self.assertLess(scale * 3500.0 * np.sqrt(kT / stiff_energy), 0.65)

    def test_write_modes_scales_velocities_and_labels_modes(self) -> None:
        from ase import Atoms
        from ase.io import read
        from ase.vibrations.data import VibrationsData

        atoms = Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        )
        hessian = np.zeros((2, 3, 2, 3))
        for axis in range(3):
            hessian[0, axis, 0, axis] = 20.0
            hessian[1, axis, 1, axis] = 20.0
            hessian[0, axis, 1, axis] = -20.0
            hessian[1, axis, 0, axis] = -20.0
        vibration_data = VibrationsData(atoms, hessian)

        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            _write_modes(
                vibration_data,
                work_dir,
                output_traj=True,
                traj_format="extxyz",
                frames=5,
                output_stru=True,
                stru_format="extxyz",
            )
            structures = sorted((work_dir / "vib/modes").glob("mode_*.xyz"))
            trajectories = sorted((work_dir / "vib/mode_trajectories").glob("mode_*.extxyz"))
            self.assertTrue(structures)
            self.assertEqual(len(structures), len(trajectories))
            for path in structures:
                velocities = read(path, format="extxyz").get_velocities()
                self.assertLess(np.abs(velocities).max(), 2.0)
            frames = read(trajectories[0], index=":", format="extxyz")
            self.assertEqual(len(frames), 5)
            self.assertLess(
                np.abs(np.array([image.get_velocities() for image in frames])).max(), 2.0
            )


if __name__ == "__main__":
    unittest.main()
