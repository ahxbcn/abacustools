"""Tests for batch ABACUS job monitoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from abacustools.commands.job.monitor_many import _summary, run
from abacustools.core.job import JobStatus, JobValidation


class TestJobMonitorMany(unittest.TestCase):
    def test_summary_uses_latest_geometry_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "relax"
            job.mkdir()
            log = job / "running_relax.log"
            log.write_text(
                "STEP OF RELAXATION : 4\n"
                "final etot is -3.0 eV\n"
                "Largest gradient in force is 0.02 eV/A.\n"
                "Largest gradient in stress is 1.0 kBar.\n",
                encoding="utf-8",
            )
            validation = JobValidation(True, [], {"calculation": "relax"})
            status = JobStatus("running", {}, log)
            with patch("abacustools.commands.job.monitor_many.validate_job", return_value=validation), patch(
                "abacustools.commands.job.monitor_many.status_job", return_value=status
            ):
                result = _summary(job, [job])
            self.assertEqual(result["step"], 4)
            self.assertEqual(result["relaxation_steps"], 1)
            self.assertAlmostEqual(result["max_force"], 0.02)
            self.assertAlmostEqual(result["max_stress"], 1.0)

    def test_json_once_reports_all_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [root / "one", root / "two"]
            for job in jobs:
                job.mkdir()
            items = [
                {"name": "one", "state": "running", "calculation": "scf", "step": None,
                 "energy": -1.0, "energy_change": None, "max_force": None, "max_stress": None,
                 "relaxation_steps": None, "job": str(jobs[0]), "log": None},
                {"name": "two", "state": "converged", "calculation": "relax", "step": 3,
                 "energy": -2.0, "energy_change": -0.1, "max_force": 0.01, "max_stress": 0.2,
                 "relaxation_steps": 3, "job": str(jobs[1]), "log": None},
            ]
            args = Namespace(job=jobs, interval=1.0, once=True, json=True)
            with patch("abacustools.commands.job.monitor_many._summary", side_effect=items):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(run(args), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(len(result["jobs"]), 2)
            self.assertEqual(result["jobs"][1]["step"], 3)


if __name__ == "__main__":
    unittest.main()
