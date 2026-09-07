"""Tests for the Berry-phase Born effective charge workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from abacustools.commands.workflow.bec import (
    polarization_cartesian,
    polarization_delta,
    postprocess,
    prepare,
    read_berry_polarization,
)
from abacustools.core.constant import BOHR_TO_ANG


STRU = """ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
4 0 0
0 4 0
0 0 4

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0
"""


class TestBECWorkflow(unittest.TestCase):
    @staticmethod
    def _write_berry_task(task: Path, values: list[float]) -> None:
        """Write minimal synthetic SCF and Berry phase output for one task."""
        output = task / "OUT.ABACUS"
        output.mkdir(parents=True)
        (task / "INPUT").write_text(
            "INPUT_PARAMETERS\ncalculation scf\nsuffix ABACUS\ngamma_only 1\n",
            encoding="utf-8",
        )
        (output / "running_scf.log").write_text(
            "Final Etot = -1.0 eV\n", encoding="utf-8"
        )
        for index, value in enumerate(values, start=1):
            log_value = value / BOHR_TO_ANG
            (output / f"running_nscf{index}.log").write_text(
                "Volume (A^3) = 64.0\n"
                f"The calculated polarization direction is in R{index} direction\n"
                f"P = {log_value} (mod 100.0) (0.0 0.0 0.0) (e/Omega).bohr\n"
                "P = 0.0 (mod 1.0) (0.0 0.0 0.0) C/m^2\n",
                encoding="utf-8",
            )

    def test_polarization_helpers(self) -> None:
        cell = np.eye(3) * 4.0
        np.testing.assert_allclose(
            polarization_cartesian([1.0, 2.0, 3.0], cell), [1.0, 2.0, 3.0]
        )
        np.testing.assert_allclose(
            polarization_delta(
                [0.0, 0.0, 0.0], [0.9, -0.9, 0.1], [1.0, 1.0, 1.0]
            ),
            [-0.1, 0.1, 0.1],
        )

    def test_reads_berry_phase_polarization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "running_nscf1.log"
            log.write_text(
                "Volume (A^3) = 64.0\n"
                "The calculated polarization direction is in R1 direction\n"
                "P = 0.0\n"
                "P = 2.0 (mod 3.0) (0.0 0.0 0.0) (e/Omega).bohr\n"
                "P = 0.0\n"
                "P = 1.5 (mod 2.0) (0.0 0.0 0.0) C/m^2\n",
                encoding="utf-8",
            )
            result = read_berry_polarization(log)

        self.assertEqual(result["direction"], 1)
        self.assertAlmostEqual(result["p_vec"], 2.0 * BOHR_TO_ANG)
        self.assertAlmostEqual(result["mod"], 3.0 * BOHR_TO_ANG)
        self.assertEqual(result["polarization_cm2"], 1.5)
        self.assertEqual(result["volume"], 64.0)

    def test_prepare_writes_bec_tasks_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(STRU, encoding="utf-8")
            (job / "H.upf").write_text("pseudo", encoding="utf-8")

            result = prepare(
                Namespace(
                    job=job,
                    stepsize=0.01,
                    index=[1],
                    directions=["x", "z"],
                    disp_type="c",
                    use_k_continuity=False,
                    abacus_command="abacus",
                    override=False,
                )
            )

            self.assertEqual(result, 0)
            manifest = json.loads(
                (job / "workflow_bec.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["tasks"],
                [
                    "bec_disp_atom0_x",
                    "bec_disp_atom0_x_back",
                    "bec_disp_atom0_z",
                    "bec_disp_atom0_z_back",
                ],
            )
            task = job / "bec_disp_atom0_x"
            self.assertTrue((task / "INPUT.scf").is_file())
            self.assertTrue((task / "INPUT.nscf1").is_file())
            self.assertTrue((task / "KPT.nscf1").is_file())
            self.assertTrue((task / "run_bec.sh").stat().st_mode & 0o111)
            self.assertIn("calculation         nscf", (task / "INPUT.nscf1").read_text())

    def test_postprocess_calculates_central_difference_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(STRU, encoding="utf-8")
            negative = "bec_disp_atom0_x_back"
            positive = "bec_disp_atom0_x"
            self._write_berry_task(job / negative, [0.0, 0.0, 0.0])
            self._write_berry_task(job / positive, [0.2, 0.0, 0.0])
            (job / "workflow_bec.json").write_text(
                json.dumps(
                    {
                        "format": 1,
                        "workflow": "bec",
                        "tasks": [positive, negative],
                        "atom_indices": [1],
                        "directions": ["x"],
                        "stepsize": 0.1,
                        "displacement_type": "c",
                        "suffix": "ABACUS",
                    }
                ),
                encoding="utf-8",
            )

            result = postprocess(
                Namespace(job=job, version="", output="bec_results.json")
            )

            self.assertEqual(result, 0)
            output = json.loads((job / "bec_results.json").read_text())
            tensor = output["atoms"][0]["bec_tensor"]
            np.testing.assert_allclose(tensor[0], [1.0, 0.0, 0.0])
            self.assertIsNone(tensor[1])


if __name__ == "__main__":
    unittest.main()
