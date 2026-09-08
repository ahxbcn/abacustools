"""Tests for the finite-strain piezoelectric workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.commands.workflow.piezoelectric import (
    _ELECTRON_ANGSTROM_SQUARED_TO_CM2,
    postprocess,
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
Cartesian

H
0.0
1
0 0 0
"""


class TestPiezoelectricWorkflow(unittest.TestCase):
    @staticmethod
    def _write_berry_task(task: Path, values: list[float], volume: float = 64.0) -> None:
        output = task / "OUT.ABACUS"
        output.mkdir(parents=True, exist_ok=True)
        for index, value in enumerate(values, start=1):
            (output / f"running_nscf{index}.log").write_text(
                f"Volume (A^3) = {volume}\n"
                f"The calculated polarization direction is in R{index} direction\n"
                f"P = {value} (mod 100.0) (0.0 0.0 0.0) (e/Omega).bohr\n"
                "P = 0.0 (mod 1.0) (0.0 0.0 0.0) C/m^2\n",
                encoding="utf-8",
            )

    @staticmethod
    def _write_job(path: Path) -> None:
        (path / "INPUT").write_text(
            "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
        )
        (path / "STRU").write_text(STRU, encoding="utf-8")
        (path / "H.upf").write_text("pseudo", encoding="utf-8")

    def test_prepare_generates_six_central_strain_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)

            self.assertEqual(
                prepare(
                    Namespace(
                        job=job,
                        strain=0.01,
                        disp_type="c",
                        use_k_continuity=False,
                        relax=False,
                        abacus_command="abacus",
                        override=False,
                    )
                ),
                0,
            )
            manifest = json.loads(
                (job / "workflow_piezoelectric.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["tasks"]), 12)
            self.assertEqual(
                manifest["tasks"][:3],
                [
                    "piezoelectric_xx",
                    "piezoelectric_xx_back",
                    "piezoelectric_yy",
                ],
            )
            self.assertIn(
                "berry_phase         1",
                (job / "piezoelectric_xx" / "INPUT.nscf1").read_text(),
            )
            self.assertTrue((job / "piezoelectric_xx" / "run.sh").is_symlink())

            from abacustools.io.stru import AbacusSTRU

            strained = AbacusSTRU.read(job / "piezoelectric_xx" / "STRU")
            np.testing.assert_allclose(
                strained.cell,
                [[4.04 * BOHR_TO_ANG, 0, 0], [0, 4 * BOHR_TO_ANG, 0], [0, 0, 4 * BOHR_TO_ANG]],
            )

    def test_postprocess_converts_polarization_change_to_c_per_square_meter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)
            prepare(
                Namespace(
                    job=job,
                    strain=0.1,
                    disp_type="c",
                    use_k_continuity=False,
                    relax=False,
                    abacus_command="abacus",
                    override=False,
                )
            )
            for name in json.loads(
                (job / "workflow_piezoelectric.json").read_text()
            )["tasks"]:
                self._write_berry_task(job / name, [0.0, 0.0, 0.0])

            # For xx, choose a dipole change that corresponds to 1 C/m^2
            # after a central difference of 2 * 0.1.
            dipole = 1.0 * 0.2 * 64.0 / _ELECTRON_ANGSTROM_SQUARED_TO_CM2
            self._write_berry_task(
                job / "piezoelectric_xx", [dipole / BOHR_TO_ANG, 0.0, 0.0]
            )
            status = postprocess(Namespace(job=job, version="LTS3.10.1", output="result.json"))

            self.assertEqual(status, 0)
            result = json.loads((job / "result.json").read_text())
            self.assertAlmostEqual(result["piezoelectric_tensor"][0][0], 1.0)

    def test_prepare_relax_adds_relaxation_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)
            prepare(
                Namespace(
                    job=job,
                    strain=0.01,
                    disp_type="f",
                    use_k_continuity=True,
                    relax=True,
                    abacus_command="abacus",
                    override=False,
                )
            )
            task = job / "piezoelectric_xx"
            self.assertTrue((task / "INPUT.relax").is_file())
            runner = (task / "run.sh").read_text()
            self.assertIn("cp INPUT.relax INPUT", runner)
            self.assertIn("cp OUT.ABACUS/STRU_ION_D STRU", runner)


if __name__ == "__main__":
    unittest.main()
