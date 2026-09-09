"""Tests for geometry-optimization monitoring."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from abacustools.commands.job.monitor import _run_relaxation
from abacustools.data.abacus_result import read_relaxation_history


LOG = """
STEP OF RELAXATION : 1
final etot is -10.000000 eV
 Largest gradient in force is 0.50 eV/A.
 Largest gradient in stress is 12.0 kBar.
 Relaxation is not converged yet!

STEP OF RELAXATION : 2
final etot is -10.250000 eV
 Largest gradient in force is 0.05 eV/A.
 Largest gradient in stress is 1.5 kBar.
 Relaxation is converged!
"""


class TestJobMonitor(unittest.TestCase):
    def test_reads_relaxation_history_and_energy_change(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(LOG)
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["step"], 1)
        self.assertAlmostEqual(history[0]["max_force"], 0.5)
        self.assertAlmostEqual(history[0]["max_stress"], 12.0)
        self.assertIsNone(history[0]["energy_change"])
        self.assertAlmostEqual(history[1]["energy_change"], -0.25)
        self.assertTrue(history[1]["converged"])

    def test_keeps_incomplete_last_step(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(
                "STEP OF RELAXATION : 3\n"
                "final etot is -11.0 eV\n"
                "Largest gradient in force is 0.2 eV/A.\n"
            )
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertEqual(history[-1]["step"], 3)
        self.assertFalse(history[-1]["converged"])
        self.assertAlmostEqual(history[-1]["max_force"], 0.2)

    def test_json_and_csv_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            log = temporary / "running_relax.log"
            log.write_text(LOG, encoding="utf-8")
            csv_path = temporary / "history.csv"
            args = Namespace(
                job=temporary, interval=1.0, once=True, relaxation=True,
                json=False, csv=csv_path, plot=None,
            )
            with patch(
                "abacustools.commands.job.monitor._relax_log",
                return_value=("converged", log),
            ):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(_run_relaxation(args), 0)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["step"], "2")
            self.assertIn("geometry history:", output.getvalue())

    def test_json_output_contains_units_and_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            log = temporary / "running_relax.log"
            log.write_text(LOG, encoding="utf-8")
            args = Namespace(
                job=temporary, interval=1.0, once=False, relaxation=True,
                json=True, csv=None, plot=None,
            )
            with patch(
                "abacustools.commands.job.monitor._relax_log",
                return_value=("converged", log),
            ):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(_run_relaxation(args), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["force_unit"], "eV/Angstrom")
            self.assertEqual(result["stress_unit"], "kBar")
            self.assertEqual(len(result["steps"]), 2)


if __name__ == "__main__":
    unittest.main()
