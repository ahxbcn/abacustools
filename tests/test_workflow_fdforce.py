"""Tests for finite-difference force validation."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from abacustools.commands.workflow.fdforce import (
    _finite_difference_force,
    _read_info,
    postprocess,
    prepare,
)


def _write_job(path: Path) -> None:
    (path / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
    )
    (path / "STRU").write_text(
        """ATOMIC_SPECIES
C 12.0
H 1.0

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Cartesian

C
0.0
1
1 2 3

H
0.0
1
4 5 6
""",
        encoding="utf-8",
    )


class TestFDForceWorkflow(unittest.TestCase):
    def test_formula_and_info_parser(self) -> None:
        self.assertAlmostEqual(_finite_difference_force(1.0, 0.0, 0.5), 1.0)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as info:
            info.write("C 1 x\nH 1 y z\n")
            info.flush()
            self.assertEqual(_read_info(Path(info.name)), [("C", 1, ["x"]), ("H", 1, ["y", "z"])])

    def test_prepare_honors_info_selection_and_displacements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            _write_job(job)
            (job / "info.txt").write_text("C 1 x\nH 1 y\n", encoding="utf-8")
            prepare(Namespace(job=job, stepsize=0.1, number=2, selected_atoms=None,
                              directions=["x", "y", "z"], info=Path("info.txt"), override=False))
            manifest = json.loads((job / "workflow_fdforce.json").read_text())
            self.assertEqual(manifest["selections"], [{"atom": 1, "directions": ["x"]}, {"atom": 2, "directions": ["y"]}])
            self.assertEqual(len(manifest["tasks"]), 1 + 2 * 2 * 2)
            self.assertTrue((job / "fdforce/atom_1_x_+1/STRU").is_file())
            self.assertTrue((job / "fdforce/atom_2_y_-2/STRU").is_file())

    def test_postprocess_compares_force_at_each_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            _write_job(job)
            prepare(Namespace(job=job, stepsize=0.1, number=2, selected_atoms=[1],
                              directions=["x"], info=None, override=False))

            def result_for(job_path, version, natoms):
                name = job_path.name
                if name == "equilibrium":
                    position = 0.0
                else:
                    match = re.search(r"([+-])(\d+)$", name)
                    sign = 1 if match.group(1) == "+" else -1
                    position = sign * int(match.group(2)) * 0.1
                force = np.zeros((natoms, 3))
                force[0, 0] = -2.0 * position
                return position * position, force

            with patch("abacustools.commands.workflow.fdforce._read_force_energy", side_effect=result_for):
                postprocess(Namespace(job=job, version="LTS3.10.1", output="results.json"))
            result = json.loads((job / "results.json").read_text())
            case = result["cases"]["atom_1_x"]
            self.assertAlmostEqual(case["equilibrium_force"], 0.0)
            self.assertTrue(all(abs(value) < 1e-12 for value in case["deviation"]))


if __name__ == "__main__":
    unittest.main()
