"""Tests for task-aware ABACUS job monitoring."""

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

from abacustools.commands.job.monitor import (
    DEFAULT_TAIL,
    _format_force_site,
    _unconverged_counts,
    run,
)
from abacustools.core.constant import BOHR_TO_ANG, RY_TO_EV
from abacustools.core.job import JobStatus, JobValidation
from abacustools.data.abacus_result import (
    find_job_log,
    read_convergence_thresholds,
    read_md_history,
    read_normal_end,
    read_relaxation_history,
    read_scf_history,
)
from abacustools.io.abacus import ReadInput
from abacustools.main import _create_parser


RELAX_LOG = """
 STEP OF RELAXATION : 1
------------------------------------------------------------------------------------------
 TOTAL-FORCE (eV/Angstrom)
------------------------------------------------------------------------------------------
                       Ga1        -0.0319780969        -0.0319780969         0.0668876704
                        N1         0.0319780969         0.0319780969        -0.0668876704
------------------------------------------------------------------------------------------
 final etot is -10.000000 eV
 Largest gradient in force is 0.0668877 eV/A.
 Threshold is 0.0257112 eV/A.
 Relaxation is not converged yet!

 STEP OF RELAXATION : 2
 final etot is -10.250000 eV
 Largest gradient in force is 0.005 eV/A.
 Threshold is 0.0257112 eV/A.
 Relaxation is converged!
 Total  Time  : 0 h 0 mins 9 secs
"""


COORDINATE_RELAX_LOG = """
CARTESIAN COORDINATES ( UNIT = 1.0 Bohr ).
    atom                   x                   y                   z
H1                      0.0                 0.0                 0.0
H2                      1.0                 0.0                 0.0

 STEP OF RELAXATION : 1
 final etot is -1.000000 eV
CARTESIAN COORDINATES ( UNIT = 1.0 Bohr ).
    atom                   x                   y                   z
H1                      0.1                 0.0                 0.0
H2                      1.0                 0.2                 0.0

 STEP OF RELAXATION : 2
 final etot is -1.100000 eV
CARTESIAN COORDINATES ( UNIT = 1.0 Bohr ).
    atom                   x                   y                   z
H1                      0.15                0.0                 0.0
H2                      1.3                 0.2                 0.0
"""


CELL_RELAX_LOG = """
 STEP OF RELAXATION : 1
----------------------------------------------------------------
 TOTAL-STRESS (KBAR)
----------------------------------------------------------------
        22.9522602291         0.0664736340        -0.9065846651
         0.0664736340        10.5587173880         0.0228792697
        -0.9065846651         0.0228792697        -0.7318319446
----------------------------------------------------------------
 final etot is -10.000000 eV
 Largest gradient in force is 0.20 eV/A.
 Largest gradient in stress is 22.95226 kBar.
 Threshold is 0.020000 eV/A.
 Threshold is 0.500000 kBar.
 Relaxation is not converged yet!
"""


SCF_LOG = """
 LCAO ALGORITHM --------------- ION=   1  ELEC=   1--------------------------------
 Density error is 0.0491537746297
 E_KohnSham     -236.6840809437      -3220.2521267278

 LCAO ALGORITHM --------------- ION=   1  ELEC=   2--------------------------------
 Density error is 0.0187015006713
 E_KohnSham     -236.6965515963      -3220.4217986602
 Total  Time  : 0 h 0 mins 3 secs
"""


MD_LOG = """
 STEP OF MOLECULAR DYNAMICS : 0
 ------------------------------------------------------------------------------------------------
 Energy              Potential           Kinetic             Temperature         Pressure (KBAR)
 -347.52879          -347.53216          +0.0033726584       +10                 -0.0017223133
 ------------------------------------------------------------------------------------------------

 STEP OF MOLECULAR DYNAMICS : 1
 ------------------------------------------------------------------------------------------------
 Energy              Potential           Kinetic             Temperature         Pressure (KBAR)
 -347.52502          -347.53213          +0.0071090036       +21.078339          0.14316058
 ------------------------------------------------------------------------------------------------
"""


ION_RELAX_LOG = """
 STEP OF ION RELAXATION : 1
 final etot is -20.000000 eV
 TOTAL-FORCE (eV/Angstrom)
------------------------------------------------------------------------------------------
                        O1         0.0004021890        -0.0004021890        -0.0004021890
                        O2        -0.0004021890         0.0004021890         0.0004021890
------------------------------------------------------------------------------------------
 Largest gradient in force is 0.000402 eV/A.
 Threshold is 0.0100000 eV/A.
 Ion relaxation is converged!
 end of geometry optimization
"""


ION_ONLY_LOG = """
 PW ALGORITHM --------------- ION=   1  ELEC=   1--------------------------------
 E_KohnSham     -1.0000   -13.6057
 final etot is -13.6057 eV
 TOTAL-FORCE (eV/Angstrom)
----------------------------------------------------------------
                        O1         0.5000000000        -0.5000000000         0.5000000000
----------------------------------------------------------------
 Largest gradient in force is 0.866025 eV/A.
 Threshold is 0.010000 eV/A.

 PW ALGORITHM --------------- ION=   2  ELEC=   1--------------------------------
 E_KohnSham     -1.1000   -14.9662
 final etot is -14.9662 eV
"""


MD_RY_LOG = """
 STEP OF MOLECULAR DYNAMICS : 0
 ------------------------------------------------------------------------------------------------
 Energy (Ry)         Potential (Ry)      Kinetic (Ry)        Temperature (K)     Pressure (kbar)
 -1427.2279          -1427.7666          0.53867530          900.00000           14.334424
 ------------------------------------------------------------------------------------------------
"""


MD_WITHOUT_PRESSURE_LOG = """
 STEP OF MOLECULAR DYNAMICS : 0
 ------------------------------------------------------------------------------------------------
 Energy (Ry)         Potential (Ry)      Kinetic (Ry)        Temperature (K)
 -2.3317484          -2.3317484          0.0000000           0.0000000
 ------------------------------------------------------------------------------------------------
"""


def _job_with_log(
    root: Path,
    name: str,
    text: str,
    *,
    calculation: str,
    inputs: dict | None = None,
) -> tuple[Path, Path]:
    """Create a minimal job whose INPUT and OUT.* layout resolve to one log."""
    job = root / "job"
    job.mkdir(exist_ok=True)
    lines = ["INPUT_PARAMETERS", f"calculation {calculation}", "suffix ABACUS"]
    lines.extend(f"{key} {value}" for key, value in (inputs or {}).items())
    (job / "INPUT").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = job / "OUT.ABACUS"
    output.mkdir(exist_ok=True)
    log = output / name
    log.write_text(text, encoding="utf-8")
    return job, log


def _monitor(job: Path, calculation: str, *, progress: dict | None = None, **overrides):
    """Run ``job monitor`` against a canned job state."""
    inputs = dict(ReadInput(job / "INPUT"))
    inputs["calculation"] = calculation
    validation = JobValidation(True, [], inputs)
    status = JobStatus("running", progress or {}, None)
    args = Namespace(
        job=job,
        interval=None,
        once=False,
        tail=DEFAULT_TAIL,
        relaxation=False,
        scf_steps=False,
        json=False,
        csv=None,
        plot=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    output = StringIO()
    with patch(
        "abacustools.commands.job.monitor.validate_job", return_value=validation
    ), patch(
        "abacustools.commands.job.monitor.status_job", return_value=status
    ), redirect_stdout(output):
        code = run(args)
    return code, output.getvalue()


def _long_relax_log(steps: int) -> str:
    """Build a relaxation log with ``steps`` ionic steps."""
    return "".join(
        f"STEP OF RELAXATION : {step}\n"
        f"final etot is {-10.0 - step * 0.01} eV\n"
        f"Largest gradient in force is {0.1 / step} eV/A.\n"
        for step in range(1, steps + 1)
    )


def _table_rows(output: str) -> list[str]:
    """Return the rows of the step table, which all share the header width."""
    lines = output.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("step"))
    return [line for line in lines[start + 1:] if len(line) == len(lines[start])]


class TestStepHistories(unittest.TestCase):
    def test_reads_relaxation_history_and_energy_change(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(RELAX_LOG)
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["step"], 1)
        self.assertAlmostEqual(history[0]["max_force"], 0.0668877)
        self.assertIsNone(history[0]["energy_change"])
        self.assertAlmostEqual(history[1]["energy_change"], -0.25)
        self.assertTrue(history[1]["converged"])

    def test_names_the_atom_and_component_of_the_largest_force(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(RELAX_LOG)
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertEqual(history[0]["force_atom"], 1)
        self.assertEqual(history[0]["force_atom_label"], "Ga1")
        self.assertEqual(history[0]["force_component"], "z")
        self.assertEqual(
            history[0]["forces"],
            [
                [-0.0319780969, -0.0319780969, 0.0668876704],
                [0.0319780969, 0.0319780969, -0.0668876704],
            ],
        )
        self.assertIsNone(history[1]["forces"])
        self.assertIsNone(history[0]["stress_component"])

    def test_calculates_displacement_from_consecutive_coordinate_blocks(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(COORDINATE_RELAX_LOG)
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertAlmostEqual(
            history[0]["rms_displacement"],
            (0.025**0.5) * BOHR_TO_ANG,
        )
        self.assertAlmostEqual(history[0]["max_displacement"], 0.2 * BOHR_TO_ANG)
        self.assertAlmostEqual(
            history[1]["rms_displacement"],
            (((0.05**2 + 0.3**2) / 2) ** 0.5) * BOHR_TO_ANG,
        )

    def test_names_the_component_of_the_largest_stress(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(CELL_RELAX_LOG)
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertAlmostEqual(history[0]["max_stress"], 22.95226)
        self.assertEqual(history[0]["stress_component"], "xx")

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

    def test_prefers_the_full_precision_energy_of_the_ionic_step(self) -> None:
        log = (
            " STEP OF RELAXATION : 1\n"
            " LCAO ALGORITHM --------------- ION=   1  ELEC=   1---------------\n"
            "     Energy           Rydberg                 eV\n"
            " E_KohnSham     -1900.0000000000    -25850.1234567890\n"
            " charge density convergence is achieved\n"
            " final etot is -25850.12346 eV\n"
            " STEP OF RELAXATION : 2\n"
            " E_KohnSham     -1900.0001000000    -25850.1245678901\n"
            " final etot is -25850.12457 eV\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(log)
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertAlmostEqual(history[0]["energy"], -25850.1234567890)
        self.assertAlmostEqual(history[1]["energy"], -25850.1245678901)
        self.assertAlmostEqual(
            history[1]["energy_change"],
            -25850.1245678901 + 25850.1234567890,
        )

    def test_reads_thresholds_and_normal_end_from_the_log(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(CELL_RELAX_LOG)
            stream.flush()
            thresholds = read_convergence_thresholds(stream.name)
            self.assertAlmostEqual(thresholds["force_thr_ev"], 0.02)
            self.assertAlmostEqual(thresholds["stress_thr"], 0.5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(RELAX_LOG)
            stream.flush()
            self.assertTrue(read_normal_end(stream.name))

    def test_reads_scf_iterations(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(SCF_LOG)
            stream.flush()
            history = read_scf_history(stream.name)

        self.assertEqual([item["step"] for item in history], [1, 2])
        self.assertAlmostEqual(history[0]["energy"], -3220.2521267278)
        self.assertAlmostEqual(history[0]["drho"], 0.0491537746297)
        self.assertIsNone(history[0]["energy_change"])
        self.assertAlmostEqual(history[1]["energy_change"], -3220.4217986602 + 3220.2521267278)

    def test_reads_md_steps(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(MD_LOG)
            stream.flush()
            history = read_md_history(stream.name)

        self.assertEqual([item["step"] for item in history], [0, 1])
        self.assertAlmostEqual(history[0]["energy"], -347.52879)
        self.assertAlmostEqual(history[0]["kinetic"], 0.0033726584)
        self.assertAlmostEqual(history[0]["temperature"], 10.0)
        self.assertAlmostEqual(history[0]["pressure"], -0.0017223133)
        self.assertAlmostEqual(history[1]["temperature"], 21.078339)

    def test_reads_the_ion_relaxation_dialect(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(ION_RELAX_LOG)
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["step"], 1)
        self.assertAlmostEqual(history[0]["max_force"], 0.000402)
        self.assertEqual(history[0]["force_atom"], 1)
        self.assertEqual(history[0]["force_atom_label"], "O1")
        self.assertTrue(history[0]["converged"])

    def test_falls_back_to_the_electronic_ion_number(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(ION_ONLY_LOG)
            stream.flush()
            history = read_relaxation_history(stream.name)

        self.assertEqual([item["step"] for item in history], [1, 2])
        self.assertAlmostEqual(history[0]["energy"], -13.6057)
        self.assertAlmostEqual(history[0]["max_force"], 0.866025)
        self.assertEqual(history[0]["force_atom"], 1)
        self.assertEqual(history[0]["force_component"], "x")
        self.assertIsNone(history[0]["energy_change"])
        self.assertAlmostEqual(history[1]["energy_change"], -14.9662 + 13.6057)

    def test_md_energies_are_converted_to_electronvolts(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(MD_RY_LOG)
            stream.flush()
            history = read_md_history(stream.name)

        self.assertAlmostEqual(history[0]["energy"], -1427.2279 * RY_TO_EV)
        self.assertAlmostEqual(history[0]["kinetic"], 0.53867530 * RY_TO_EV)
        self.assertAlmostEqual(history[0]["temperature"], 900.0)
        self.assertAlmostEqual(history[0]["pressure"], 14.334424)

    def test_md_without_a_pressure_column_reports_no_pressure(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            stream.write(MD_WITHOUT_PRESSURE_LOG)
            stream.flush()
            history = read_md_history(stream.name)

        self.assertAlmostEqual(history[0]["energy"], -2.3317484 * RY_TO_EV)
        self.assertAlmostEqual(history[0]["temperature"], 0.0)
        self.assertIsNone(history[0]["pressure"])


class TestLogSelection(unittest.TestCase):
    def test_prefers_the_scf_log_over_an_nscf_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, scf_log = _job_with_log(
                Path(temporary), "running_scf.log", SCF_LOG, calculation="nscf"
            )
            (job / "OUT.ABACUS" / "running_nscf.log").write_text(
                "LCAO ALGORITHM --------------- ION=   1  ELEC=   1----\n",
                encoding="utf-8",
            )
            self.assertEqual(find_job_log(job), scf_log)

    def test_uses_the_ionic_log_of_a_relaxation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, relax_log = _job_with_log(
                Path(temporary), "running_relax.log", RELAX_LOG, calculation="relax"
            )
            (job / "OUT.ABACUS" / "running_scf.log").write_text("scf\n", encoding="utf-8")
            self.assertEqual(find_job_log(job, ionic=True), relax_log)

    def test_reports_no_log_for_an_unstarted_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "job"
            job.mkdir()
            (job / "INPUT").write_text("INPUT_PARAMETERS\nsuffix ABACUS\n", encoding="utf-8")
            self.assertIsNone(find_job_log(job))


class TestMonitorReports(unittest.TestCase):
    def test_long_relaxation_prints_only_the_last_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary),
                "running_relax.log",
                _long_relax_log(40),
                calculation="relax",
            )
            code, output = _monitor(job, "relax")

        self.assertEqual(code, 0)
        self.assertIn(f"showing the last {DEFAULT_TAIL} of 40 steps", output)
        rows = _table_rows(output)
        self.assertEqual(len(rows), DEFAULT_TAIL)
        self.assertEqual(rows[0].split()[0], str(40 - DEFAULT_TAIL + 1))
        self.assertEqual(rows[-1].split()[0], "40")

    def test_tail_zero_prints_every_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary),
                "running_relax.log",
                _long_relax_log(40),
                calculation="relax",
            )
            code, output = _monitor(job, "relax", tail=0)

        self.assertEqual(code, 0)
        self.assertNotIn("showing the last", output)
        self.assertEqual(len(_table_rows(output)), 40)

    def test_relax_table_aligns_columns_and_prints_full_precision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary), "running_relax.log", RELAX_LOG, calculation="relax"
            )
            code, output = _monitor(job, "relax")

        self.assertEqual(code, 0)
        lines = output.splitlines()
        start = next(index for index, line in enumerate(lines) if line.startswith("step"))
        table = lines[start:start + 3]
        self.assertEqual(len(table), 3)
        self.assertTrue(all(len(line) == len(table[0]) for line in table))
        self.assertIn("-10.25000000", output)
        self.assertIn("-2.500000e-01", output)

    def test_running_job_reports_one_update_without_waiting(self) -> None:
        unfinished = "\n".join(
            line for line in RELAX_LOG.splitlines() if "Total  Time" not in line
        )
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary), "running_relax.log", unfinished, calculation="relax"
            )
            with patch("time.sleep", side_effect=AssertionError("the monitor must not wait")):
                code, output = _monitor(job, "relax")

        self.assertEqual(code, 0)
        self.assertIn("max_force(eV/A)", output)

    def test_scf_default_reports_status_and_option_details_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary), "running_scf.log", SCF_LOG, calculation="scf"
            )
            progress = {"energy": -3220.42, "scf_steps": 2}

            code, output = _monitor(job, "scf", progress=progress)
            self.assertEqual(code, 0)
            self.assertIn("energy=-3220.42", output)
            self.assertNotIn("dE(eV)", output)

            code, output = _monitor(job, "scf", progress=progress, scf_steps=True)
            self.assertIn("dE(eV)", output)
            self.assertIn("0.049153775", output)

    def test_scf_json_carries_every_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary), "running_scf.log", SCF_LOG, calculation="scf"
            )
            code, output = _monitor(job, "scf", json=True)

            self.assertEqual(code, 0)
            result = json.loads(output)
            self.assertEqual(result["task"], "scf")
            self.assertEqual(len(result["steps"]), 2)

    def test_relax_reports_criteria_and_force_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary),
                "running_relax.log",
                RELAX_LOG,
                calculation="relax",
                inputs={"force_thr_ev": 0.02, "relax_method": "cg", "relax_nmax": 60},
            )
            code, output = _monitor(job, "relax")

            self.assertEqual(code, 0)
            self.assertIn("convergence criteria:", output)
            self.assertIn("force_thr_ev: 0.02 eV/Angstrom", output)
            self.assertIn("relax_method: cg", output)
            self.assertIn("max_force(eV/A)", output)
            self.assertIn("force_atom/component", output)
            self.assertIn("rms_displacement(A)", output)
            self.assertIn("max_displacement(A)", output)
            self.assertIn("Ga1z", output)

    def test_relax_criteria_fall_back_to_the_log_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary), "running_relax.log", RELAX_LOG, calculation="relax"
            )
            code, output = _monitor(job, "relax")

            self.assertEqual(code, 0)
            self.assertIn("0.0257112", output)

    def test_cell_relax_reports_stress_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary),
                "running_cell-relax.log",
                CELL_RELAX_LOG,
                calculation="cell-relax",
                inputs={"stress_thr": 0.5, "force_thr_ev": 0.02},
            )
            code, output = _monitor(job, "cell-relax")

            self.assertEqual(code, 0)
            self.assertIn("stress_thr: 0.5 kBar", output)
            self.assertIn("max_stress(kBar)", output)
            self.assertIn("stress_component", output)
            self.assertIn("xx", output)

    def test_md_reports_target_values_of_its_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary),
                "running_md.log",
                MD_LOG,
                calculation="md",
                inputs={
                    "md_type": "npt",
                    "md_nstep": 100,
                    "md_dt": 1.0,
                    "md_tfirst": 300,
                    "md_tlast": 300,
                    "md_pfirst": 1.0,
                },
            )
            code, output = _monitor(job, "md")

            self.assertEqual(code, 0)
            self.assertIn("md_type: npt", output)
            self.assertIn("md_tfirst: 300", output)
            self.assertIn("md_pfirst: 1.0", output)
            self.assertIn("temperature(K)", output)
            self.assertIn("-347.52879", output)

    def test_md_of_a_non_thermostat_type_omits_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary),
                "running_md.log",
                MD_LOG,
                calculation="md",
                inputs={"md_type": "nve", "md_tfirst": 300, "md_pfirst": 1.0},
            )
            code, output = _monitor(job, "md")

            self.assertEqual(code, 0)
            self.assertIn("md_type: nve", output)
            self.assertNotIn("md_tfirst", output)
            self.assertNotIn("md_pfirst", output)


class TestMonitorExports(unittest.TestCase):
    def test_csv_holds_the_task_specific_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, _ = _job_with_log(
                root, "running_relax.log", RELAX_LOG, calculation="relax"
            )
            csv_path = root / "history.csv"

            code, output = _monitor(job, "relax", csv=csv_path)
            self.assertEqual(code, 0)
            self.assertIn("step history:", output)
            self.assertIn("force_atom/component", output)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]), [
                "step", "energy", "energy_change",
                "max_force", "force_atom", "force_component",
                "rms_displacement", "max_displacement", "converged",
            ])
            self.assertEqual(rows[0]["force_atom"], "1")
            self.assertEqual(rows[0]["force_component"], "z")

    def test_csv_of_an_md_run_holds_the_md_quantities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, _ = _job_with_log(
                root, "running_md.log", MD_LOG, calculation="md", inputs={"md_type": "nvt"}
            )
            csv_path = root / "md.csv"

            code, _ = _monitor(job, "md", csv=csv_path)
            self.assertEqual(code, 0)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]), [
                "step", "energy", "potential", "kinetic", "temperature", "pressure",
            ])
            self.assertEqual(rows[0]["step"], "0")

    def test_relax_plot_marks_the_convergence_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, _ = _job_with_log(
                root,
                "running_relax.log",
                RELAX_LOG,
                calculation="relax",
                inputs={"force_thr_ev": 0.02},
            )
            plot_path = root / "relax.png"

            code, output = _monitor(job, "relax", plot=plot_path)
            self.assertEqual(code, 0)
            self.assertIn("step history plot:", output)
            self.assertTrue(plot_path.is_file())
            self.assertIn("force_thr_ev: 0.02 eV/Angstrom", output)
            self.assertIn("-10.25000000", output)
            self.assertIn("Ga1z", output)

    def test_md_plot_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, _ = _job_with_log(
                root, "running_md.log", MD_LOG, calculation="md", inputs={"md_type": "nvt"}
            )
            plot_path = root / "md.png"

            code, output = _monitor(job, "md", plot=plot_path)
            self.assertEqual(code, 0)
            self.assertIn("step history plot:", output)
            self.assertTrue(plot_path.is_file())
            self.assertIn("temperature(K)", output)
            self.assertIn("-347.52879000", output)

    def test_json_stays_machine_readable_beside_a_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, _ = _job_with_log(
                root, "running_relax.log", RELAX_LOG, calculation="relax"
            )
            plot_path = root / "relax.png"

            code, output = _monitor(job, "relax", json=True, plot=plot_path)

            self.assertEqual(json.loads(output)["displacement_unit"], "Angstrom")
            self.assertEqual(code, 0)
            self.assertTrue(plot_path.is_file())
            steps = json.loads(output)["steps"]
            self.assertEqual(json.loads(output)["task"], "relax")
            # The raw force and stress arrays only feed the plot.
            self.assertNotIn("forces", steps[0])
            self.assertNotIn("stress", steps[0])
            self.assertIn("force_atom_label", steps[0])

    def test_bare_plot_uses_the_name_of_the_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, _ = _job_with_log(
                root, "running_relax.log", RELAX_LOG, calculation="relax"
            )
            code, output = _monitor(job, "relax", plot="auto")

            self.assertEqual(code, 0)
            self.assertTrue((job / "monitor_relax.png").is_file())
            self.assertIn("monitor_relax.png", output)

    def test_bare_plot_of_md_uses_the_md_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, _ = _job_with_log(
                root, "running_md.log", MD_LOG, calculation="md", inputs={"md_type": "nvt"}
            )
            code, _ = _monitor(job, "md", plot="auto")

            self.assertEqual(code, 0)
            self.assertTrue((job / "monitor_md.png").is_file())

    def test_scf_plot_follows_the_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, _ = _job_with_log(
                Path(temporary), "running_scf.log", SCF_LOG, calculation="scf"
            )
            code, output = _monitor(job, "scf", plot="auto")

            self.assertEqual(code, 0)
            self.assertTrue((job / "monitor_scf.png").is_file())
            self.assertIn("monitor_scf.png", output)


class TestPlotArgument(unittest.TestCase):
    def test_plot_accepts_an_optional_file_name(self) -> None:
        parser = _create_parser("abacustools")
        with tempfile.TemporaryDirectory() as temporary:
            bare = parser.parse_args(["job", "monitor", temporary, "--plot"])
            named = parser.parse_args(["job", "monitor", temporary, "--plot", "out.png"])
            absent = parser.parse_args(["job", "monitor", temporary])

        self.assertEqual(bare.plot, "auto")
        self.assertEqual(named.plot, "out.png")
        self.assertIsNone(absent.plot)


class TestTailArgument(unittest.TestCase):
    def test_tail_defaults_to_thirty_steps_and_accepts_another_limit(self) -> None:
        parser = _create_parser("abacustools")
        with tempfile.TemporaryDirectory() as temporary:
            default = parser.parse_args(["job", "monitor", temporary])
            limited = parser.parse_args(["job", "monitor", temporary, "--tail", "5"])

        self.assertEqual(default.tail, DEFAULT_TAIL)
        self.assertEqual(limited.tail, 5)


class TestForceSiteFormat(unittest.TestCase):
    def test_uses_the_log_label_and_falls_back_to_the_atom_index(self) -> None:
        self.assertEqual(_format_force_site(1, "x", "H1"), "H1x")
        self.assertEqual(_format_force_site(12, "z", "Fe12"), "Fe12z")
        self.assertEqual(_format_force_site(2, "x", None), "2x")
        self.assertEqual(_format_force_site(2, None, "H2"), "H2")
        self.assertEqual(_format_force_site(None, None, None), "-")


class TestUnconvergedCounts(unittest.TestCase):
    def test_counts_only_the_components_allowed_to_move(self) -> None:
        history = [
            {"step": 1, "forces": [[0.5, 0.5, 0.0], [0.0, 0.5, 0.0]]},
            {"step": 2, "forces": None},
        ]
        moves = [(True, False, True), (False, False, False)]

        self.assertEqual(_unconverged_counts(history, "forces", 0.1, moves), [(1, 1)])
        self.assertEqual(_unconverged_counts(history, "forces", 0.1, None), [(1, 3)])
        self.assertEqual(_unconverged_counts(history, "forces", None, moves), [])

    def test_counts_stress_components_of_every_step(self) -> None:
        history = [
            {
                "step": 3,
                "stress": [[0.5, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
            },
            {"step": 4, "stress": None},
        ]

        self.assertEqual(_unconverged_counts(history, "stress", 0.1, None), [(3, 2)])


if __name__ == "__main__":
    unittest.main()
