"""Tests for finite-difference stress validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from abacustools.commands.workflow.fdstress import (
    _effective_step,
    _finite_difference_stress,
    postprocess,
    prepare,
)


def _write_job(path: Path) -> None:
    (path / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
    )
    (path / "STRU").write_text(
        """ATOMIC_SPECIES
Si 28.0

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
5 0 0
0 5 0
0 0 5

ATOMIC_POSITIONS
Direct

Si
0.0
1
0 0 0
""",
        encoding="utf-8",
    )


class TestFDStressWorkflow(unittest.TestCase):
    def test_effective_step_and_formula(self) -> None:
        self.assertAlmostEqual(_effective_step("11", 2, 0.1), 0.1 / 1.2)
        self.assertAlmostEqual(_effective_step("12", 2, 0.1), 0.1)
        self.assertAlmostEqual(_finite_difference_stress(1.0, 0.0, 10.0, 0.1), 801.0878861194773)

    def test_prepare_deforms_cell_and_keeps_fractional_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            _write_job(job)
            prepare(Namespace(job=job, step=0.1, number=1, components=["11", "12"], all_components=False, override=False))
            manifest = json.loads((job / "workflow_fdstress.json").read_text())
            self.assertEqual(manifest["tasks"], ["fdstress/equilibrium", "fdstress/component_11_minus1", "fdstress/component_11_plus1", "fdstress/component_12_minus1", "fdstress/component_12_plus1"])
            self.assertEqual(len(manifest["deformations"]), 6)
            deformed = (job / "fdstress/component_11_plus1/STRU").read_text()
            self.assertIn("5.5000000000", deformed)

    def test_postprocess_uses_analytic_stress_and_energy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            _write_job(job)
            prepare(Namespace(job=job, step=0.1, number=1, components=["11"], all_components=False, override=False))

            def energy_for(job_path, version):
                name = job_path.name
                if name == "equilibrium":
                    return 0.0
                sign = 1 if "plus" in name else -1
                return sign * 1.0

            with patch("abacustools.commands.workflow.fdstress._read_energy", side_effect=energy_for), patch(
                "abacustools.commands.workflow.fdstress._read_stress", return_value=np.zeros((3, 3))
            ):
                postprocess(Namespace(job=job, version="LTS3.10.1", output="results.json"))
            result = json.loads((job / "results.json").read_text())
            self.assertEqual(result["stress_unit"], "kBar")
            self.assertEqual(result["cases"]["11"]["equilibrium_stress"], 0.0)


if __name__ == "__main__":
    unittest.main()
