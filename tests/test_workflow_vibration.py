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
    _element_mass_overrides,
    _frequency_label,
    _hessian_from_forces,
    _write_modes_ase,
    _write_modes_builtin,
    postprocess,
    _selected_atoms,
    _temperatures,
    prepare,
)
from abacustools.core.constant import BOLTZMANN_CONSTANT_EV_PER_K, INV_CM_TO_EV
from abacustools.core.submission import generate_workflow_submission, resolve_submission
from abacustools.data.vibration import HarmonicVibration, frequency_values
from abacustools.integrations.ase_vibration import AseVibrationData
from abacustools.io.stru import AbacusSTRU


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
        self.assertEqual(frequency_values(np.array([1 + 0j, 2j])), [1.0, -2.0])
        with self.assertRaises(ValueError):
            _selected_atoms([1, 1], 3)
        with self.assertRaises(ValueError):
            _temperatures([100, 300])

    def test_element_mass_overrides_are_validated(self) -> None:
        self.assertEqual(_element_mass_overrides(None), {})
        self.assertEqual(
            _element_mass_overrides(["H=2.014", "O=18.0"]),
            {"H": 2.014, "O": 18.0},
        )
        for values in (["H"], ["=2.0"], ["H="], ["H=abc"], ["H=0"], ["H=-1.0"]):
            with self.assertRaises(ValueError):
                _element_mass_overrides(values)
        with self.assertRaises(ValueError):
            _element_mass_overrides(["H=1.0", "H=2.0"])
        with self.assertRaises(ValueError):
            _element_mass_overrides(["H=inf"])

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

    def postprocess_arguments(self, job: Path, **overrides) -> Namespace:
        arguments = dict(
            job=job,
            version="LTS3.10.1",
            temperature=[298.15],
            traj=False,
            traj_format="extxyz",
            frames=30,
            output_stru=False,
            stru_format="extxyz",
            backend="ase",
            element_masses=None,
            output="vibration_results.json",
        )
        arguments.update(overrides)
        return Namespace(**arguments)

    def test_postprocess_builds_frequencies_with_both_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(
                """ATOMIC_SPECIES
H 1.008

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
                frequencies = {}
                for backend in ("ase", "builtin"):
                    self.assertEqual(
                        postprocess(self.postprocess_arguments(job, backend=backend)), 0
                    )
                    vibration_result = json.loads(
                        (job / "vibration_results.json").read_text(encoding="utf-8")
                    )
                    frequencies[backend] = vibration_result["frequencies"]
                    self.assertEqual(len(vibration_result["frequencies"]), 3)
                    self.assertTrue(
                        all(value > 2000 for value in vibration_result["frequencies"])
                    )
                    self.assertIn("298.15K", vibration_result["thermo_corr"])
                    thermo = vibration_result["thermo_corr"]["298.15K"]
                    self.assertIn("entropy", thermo)
                    self.assertIn("free_energy", thermo)
                    if backend == "ase":
                        # The ASE backend keeps the original report.
                        self.assertNotIn("modes", vibration_result)
                        self.assertEqual(
                            sorted(thermo),
                            ["entropy", "free_energy"],
                        )
                    else:
                        self.assertEqual(len(vibration_result["modes"]), 3)
                        self.assertIn("heat_capacity", thermo)
                        self.assertIn("internal_energy", thermo)
            # Both backends run on the masses of the STRU and only differ by
            # their implementation.
            np.testing.assert_allclose(
                frequencies["ase"], frequencies["builtin"], rtol=1e-6
            )

    def test_mass_option_supports_both_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(
                """ATOMIC_SPECIES
H 1.008

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
                    sign = 1.0 if job_path.name.endswith("+") else -1.0
                    axis = {"x": 0, "y": 1, "z": 2}[job_path.name[-2]]
                    force[0, axis] = -20.0 * sign * 0.01
                return force

            with patch(
                "abacustools.commands.workflow.vibration._read_forces",
                side_effect=force_for,
            ):
                for backend in ("ase", "builtin"):
                    results = {}
                    for case, masses in (("default", None), ("isotope", ["H=4.0"])):
                        self.assertEqual(
                            postprocess(
                                self.postprocess_arguments(
                                    job,
                                    backend=backend,
                                    element_masses=masses,
                                )
                            ),
                            0,
                        )
                        results[case] = json.loads(
                            (job / "vibration_results.json").read_text(encoding="utf-8")
                        )
                        self.assertEqual(
                            results[case]["masses"],
                            [4.0 if case == "isotope" else 1.008],
                        )
                    # The mass option of a single atom scales its frequency.
                    self.assertAlmostEqual(
                        results["isotope"]["frequencies"][0]
                        / results["default"]["frequencies"][0],
                        np.sqrt(1.008 / 4.0),
                        places=6,
                    )

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

    def write_hydrogen_structure(self, directory: Path) -> AbacusSTRU:
        path = directory / "STRU"
        path.write_text(
            """ATOMIC_SPECIES
H 1.0079 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
15 0 0
0 15 0
0 0 15

ATOMIC_POSITIONS
Cartesian

H
0.0
2
0 0 0
0 0 0.74
""",
            encoding="utf-8",
        )
        return AbacusSTRU.read(path)

    def hydrogen_hessian(self) -> np.ndarray:
        hessian = np.zeros((2, 3, 2, 3))
        for axis in range(3):
            hessian[0, axis, 0, axis] = 20.0
            hessian[1, axis, 1, axis] = 20.0
            hessian[0, axis, 1, axis] = -20.0
            hessian[1, axis, 0, axis] = -20.0
        return hessian

    def test_write_modes_builtin_scales_velocities_and_labels_modes(self) -> None:
        from ase.io import read

        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            structure = self.write_hydrogen_structure(work_dir)
            vibration = HarmonicVibration.from_structure(structure, self.hydrogen_hessian())
            _write_modes_builtin(
                vibration,
                structure,
                work_dir,
                output_traj=True,
                traj_format="extxyz",
                frames=5,
                output_stru=True,
                stru_format="extxyz",
            )
            structures = sorted((work_dir / "vib/modes").glob("mode_*.xyz"))
            trajectories = sorted((work_dir / "vib/mode_trajectories").glob("mode_*.extxyz"))
            # The three translations have no mode, the three stretches do.
            self.assertEqual(len(structures), 3)
            self.assertEqual(len(structures), len(trajectories))
            for path in structures:
                structure_image = read(path, format="extxyz")
                velocities = structure_image.get_velocities()
                self.assertLess(np.abs(velocities).max(), 2.0)
                # Magnetic moments and pseudopotentials survive the writer.
                self.assertEqual(structure_image.info["pp"], {"H": "H.upf"})
                self.assertEqual(len(structure_image.get_initial_magnetic_moments()), 2)
            frames = read(trajectories[0], index=":", format="extxyz")
            self.assertEqual(len(frames), 5)
            self.assertLess(
                np.abs(np.array([image.get_velocities() for image in frames])).max(), 2.0
            )

    def test_write_modes_builtin_supports_poscar_and_traj_output(self) -> None:
        from ase.io import read

        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            structure = self.write_hydrogen_structure(work_dir)
            vibration = HarmonicVibration.from_structure(structure, self.hydrogen_hessian())
            _write_modes_builtin(
                vibration,
                structure,
                work_dir,
                output_traj=True,
                traj_format="traj",
                frames=4,
                output_stru=True,
                stru_format="poscar",
            )
            poscars = sorted((work_dir / "vib/modes").glob("mode_*.poscar"))
            trajectories = sorted((work_dir / "vib/mode_trajectories").glob("mode_*.traj"))
            self.assertEqual(len(poscars), 3)
            self.assertEqual(len(trajectories), 3)
            text = poscars[0].read_text(encoding="utf-8")
            self.assertEqual(text.count("Cartesian"), 2)
            images = read(trajectories[0], index=":")
            self.assertEqual(len(images), 4)
            self.assertLess(np.abs(images[0].get_velocities()).max(), 2.0)

    def test_write_modes_ase_writes_trajectories_and_structures(self) -> None:
        from ase.io import read

        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            structure = self.write_hydrogen_structure(work_dir)
            vibration = AseVibrationData(structure, self.hydrogen_hessian())
            _write_modes_ase(
                vibration,
                work_dir,
                output_traj=True,
                traj_format="extxyz",
                frames=4,
                output_stru=True,
                stru_format="extxyz",
            )
            structures = sorted((work_dir / "vib/modes").glob("mode_*.xyz"))
            trajectories = sorted((work_dir / "vib/mode_trajectories").glob("mode_*.extxyz"))
            self.assertEqual(len(structures), 3)
            self.assertEqual(len(trajectories), 3)
            self.assertLess(np.abs(read(structures[0]).get_velocities()).max(), 2.0)
            self.assertEqual(len(read(trajectories[0], index=":")), 4)


if __name__ == "__main__":
    unittest.main()
