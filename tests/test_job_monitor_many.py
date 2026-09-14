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
    def test_text_table_aligns_columns_and_prints_full_precision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [root / "relax", root / "scf"]
            for job in jobs:
                job.mkdir()
            items = [
                {"name": "relax", "state": "running", "calculation": "relax", "step": 28,
                 "energy": -347.52879, "energy_change": -1.5e-05, "max_force": 0.031,
                 "max_stress": None, "relaxation_steps": 28, "job": str(jobs[0]), "log": None},
                {"name": "scf", "state": "converged", "calculation": "scf", "step": None,
                 "energy": -3220.2521267278, "energy_change": None, "max_force": None,
                 "max_stress": None, "relaxation_steps": None, "job": str(jobs[1]), "log": None},
            ]
            args = Namespace(job=jobs, json=False)
            with patch("abacustools.commands.job.monitor_many._summary", side_effect=items):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(run(args), 0)

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(len(line) == len(lines[0]) for line in lines))
        self.assertIn("-347.52879000", output.getvalue())
        self.assertIn("-1.500000e-05", output.getvalue())
        self.assertIn("-3220.25212673", output.getvalue())

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
            args = Namespace(job=jobs, json=True)
            with patch("abacustools.commands.job.monitor_many._summary", side_effect=items):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(run(args), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(len(result["jobs"]), 2)
            self.assertEqual(result["jobs"][1]["step"], 3)

    def test_running_jobs_report_once_without_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [root / "one", root / "two"]
            for job in jobs:
                job.mkdir()
            items = [
                {"name": "one", "state": "running", "calculation": "relax", "step": 1,
                 "energy": -1.0, "energy_change": None, "max_force": 0.1, "max_stress": None,
                 "relaxation_steps": 1, "job": str(jobs[0]), "log": None},
                {"name": "two", "state": "running", "calculation": "scf", "step": None,
                 "energy": None, "energy_change": None, "max_force": None, "max_stress": None,
                 "relaxation_steps": None, "job": str(jobs[1]), "log": None},
            ]
            args = Namespace(job=jobs, json=False)
            with patch("abacustools.commands.job.monitor_many._summary", side_effect=items), patch(
                "time.sleep", side_effect=AssertionError("the monitor must not wait")
            ):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(run(args), 0)
            self.assertIn("one", output.getvalue())


if __name__ == "__main__":
    unittest.main()
