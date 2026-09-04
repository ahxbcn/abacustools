"""Tests for the electrostatic-potential work-function workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from abacustools.commands.workflow.workfunc import (
    _largest_vacuum_direction,
    calculate_work_functions,
    postprocess,
    prepare,
)


class TestWorkfuncWorkflow(unittest.TestCase):
    def test_detects_largest_vacuum_direction(self) -> None:
        from abacustools.io.stru import AbacusATOM, AbacusSTRU

        structure = AbacusSTRU(
            cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]],
            atoms=[
                AbacusATOM(label="H", element="H", coord=(1.0, 1.0, 2.0)),
                AbacusATOM(label="H", element="H", coord=(1.0, 1.0, 4.0)),
            ],
        )
        direction, vacuum_size, _, _ = _largest_vacuum_direction(structure)
        self.assertEqual(direction, "c")
        self.assertAlmostEqual(vacuum_size, 18.0)

    def test_calculates_work_function_without_plateau(self) -> None:
        results = calculate_work_functions(
            [0.0, 1.0, 2.0, 1.0, 0.0],
            fermi_energy=0.5,
            cell_length=10.0,
            threshold=0.001,
        )
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["vacuum_level"], 2.0)
        self.assertAlmostEqual(results[0]["work_function"], 1.5)

    def test_prepare_writes_workfunc_job_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nntype 1\n",
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
0 0 20

ATOMIC_POSITIONS
Cartesian

H
0.0
2
1 1 2
1 1 4
""",
                encoding="utf-8",
            )
            result = prepare(
                Namespace(
                    job=job,
                    vacuum="auto",
                    dipole_corr=True,
                    use_empty_atom=False,
                    empty_atom_elem=None,
                    empty_atom_height=2.0,
                    empty_atom_dist=2.0,
                    override=False,
                )
            )
            self.assertEqual(result, 0)
            manifest = json.loads(
                (job / "workflow_workfunc.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["workflow"], "workfunc")
            self.assertEqual(manifest["vacuum_direction"], "c")
            self.assertTrue((job / "workfunc_job/STRU").is_file())
            input_text = (job / "workfunc_job/INPUT").read_text(encoding="utf-8")
            self.assertIn("out_pot", input_text)
            self.assertIn("dip_cor_flag", input_text)

    def test_postprocess_writes_results_from_potential(self) -> None:
        class FakePotential:
            cell = np.diag([20.0, 10.0, 10.0])

            def profile1d(self, axis="a", average=True):
                coordinate = np.linspace(0.0, 1.0, 20, endpoint=False)
                potential = np.where(
                    (coordinate > 0.55) & (coordinate < 0.9),
                    -5.0,
                    -2.0,
                )
                return potential, coordinate

        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nntype 1\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(
                """ATOMIC_SPECIES
H 1.0

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
20 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Cartesian

H
0.0
2
1 1 1
3 1 1
""",
                encoding="utf-8",
            )
            prepare(
                Namespace(
                    job=job,
                    vacuum="a",
                    dipole_corr=False,
                    use_empty_atom=False,
                    empty_atom_elem=None,
                    empty_atom_height=2.0,
                    empty_atom_dist=2.0,
                    override=False,
                )
            )
            output_dir = job / "workfunc_job/OUT.ABACUS"
            output_dir.mkdir()
            (output_dir / "potes.cube").touch()
            with patch(
                "abacustools.data.abacus_result.get_result_from_job",
                return_value={"converged": True, "efermi": 1.0},
            ), patch(
                "abacustools.data.grid.Potential.from_cube",
                return_value=FakePotential(),
            ):
                result = postprocess(
                    Namespace(
                        job=job,
                        version="LTS3.10.1",
                        vacuum="a",
                        threshold=0.01,
                        output="workfunc_results.json",
                    )
                )
            self.assertEqual(result, 0)
            data = json.loads((job / "workfunc_results.json").read_text(encoding="utf-8"))
            self.assertEqual(data["vacuum_direction"], "a")
            self.assertEqual(len(data["work_functions"]), 2)
            self.assertAlmostEqual(data["work_functions"][1]["work_function"], 4.0)
            self.assertTrue((job / "workfunc_job/workfunc_profile.dat").is_file())
            self.assertTrue((job / "workfunc_job/workfunc_potential.png").is_file())


if __name__ == "__main__":
    unittest.main()
