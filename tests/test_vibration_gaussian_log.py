"""Tests for the fake Gaussian frequency log of the vibration workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from abacustools.commands.workflow.vibration import postprocess, prepare
from abacustools.core.constant import AMU_TO_KG
from abacustools.data.vibration import (
    _gaussian_orientation_lines,
    _gaussian_value_line,
    gaussian_frequency_log,
    write_gaussian_frequency_log,
)
from abacustools.io.stru import AbacusATOM, AbacusSTRU


def _diatomic() -> AbacusSTRU:
    return AbacusSTRU(
        cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        atoms=[
            AbacusATOM(label="C", element="C", coord=(0.0, 0.0, -0.599918), pp="C.upf"),
            AbacusATOM(label="H", element="H", coord=(0.0, 0.0, -1.661268), pp="H.upf"),
        ],
        metadata={"atom_type": "cartesian"},
    )


def _parse_frequency_blocks(text: str):
    """Parse the frequency block into (frequencies, reduced, constants, modes)."""
    lines = text.splitlines()
    frequencies: list[float] = []
    reduced: list[float] = []
    constants: list[float] = []
    modes: list[list[list[float]]] = []
    for index, line in enumerate(lines):
        if not line.startswith(" Frequencies --"):
            continue
        values = [float(token) for token in line.split()[2:]]
        reduced_values = [float(token) for token in lines[index + 1].split()[3:]]
        constant_values = [float(token) for token in lines[index + 2].split()[3:]]
        atom_rows = []
        row_index = index + 5  # Frequencies, Red. masses, Frc consts, IR, Atom AN
        while row_index < len(lines):
            row = lines[row_index]
            if not (row[:6].strip().isdigit() and row[6:10].strip().isdigit()):
                break
            atom_rows.append(row)
            row_index += 1
        block_modes = []
        for row in atom_rows:
            xyz = []
            for column in range(len(values)):
                base = 12 + 23 * column
                xyz.append(
                    [float(row[base + 7 * axis : base + 7 * axis + 7]) for axis in range(3)]
                )
            block_modes.append(xyz)
        frequencies.extend(values)
        reduced.extend(reduced_values)
        constants.extend(constant_values)
        for column in range(len(values)):
            modes.append([block_modes[atom][column] for atom in range(len(block_modes))])
    return frequencies, reduced, constants, modes


class TestGaussianFrequencyLog(unittest.TestCase):
    def test_value_line_matches_gaussian_columns(self) -> None:
        line = _gaussian_value_line(" Frequencies --", [55.7133, 83.2762, 152.7943])
        self.assertEqual(
            line,
            " Frequencies --     55.7133                83.2762               152.7943",
        )
        self.assertEqual(
            _gaussian_value_line(" Red. masses --", [2.2219, 2.3658, 2.1191]),
            " Red. masses --      2.2219                 2.3658                 2.1191",
        )

    def test_orientation_block_matches_gaussian_columns(self) -> None:
        lines = _gaussian_orientation_lines(_diatomic(), periodic=False)
        self.assertEqual(
            lines[1],
            "                         Standard orientation:                         ",
        )
        self.assertEqual(lines[2], " " + "-" * 69)
        self.assertEqual(
            lines[6],
            "      1          6           0        0.000000    0.000000   -0.599918",
        )
        self.assertEqual(
            lines[7],
            "      2          1           0        0.000000    0.000000   -1.661268",
        )
        self.assertEqual(lines[8], " " + "-" * 69)

    def test_orientation_block_writes_the_cell_as_translation_vectors(self) -> None:
        lines = _gaussian_orientation_lines(_diatomic())
        self.assertEqual(
            lines[1], "                          Input orientation:                          "
        )
        self.assertEqual(
            lines[6], "      1          6           0        0.000000    0.000000   -0.599918"
        )
        self.assertEqual(
            lines[7], "      2          1           0        0.000000    0.000000   -1.661268"
        )
        # The three translation vectors follow the atoms as atomic number -2.
        self.assertEqual(
            lines[8], "      3         -2           0       10.000000    0.000000    0.000000"
        )
        self.assertEqual(
            lines[9], "      4         -2           0        0.000000   10.000000    0.000000"
        )
        self.assertEqual(
            lines[10], "      5         -2           0        0.000000    0.000000   10.000000"
        )
        self.assertEqual(lines[11], " " + "-" * 69)
        self.assertEqual(
            lines[12],
            " Lengths of translation vectors:     10.000000   10.000000   10.000000",
        )
        self.assertEqual(
            lines[13],
            "  Angles of translation vectors:     90.000000   90.000000   90.000000",
        )
        self.assertEqual(lines[14], " " + "-" * 69)

    def test_gaussian_log_carries_normalized_modes(self) -> None:
        structure = _diatomic()
        frequencies = [500.0, 1500.0, 2500.0, 3500.0]
        modes = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.3], [0.0, 0.0, -0.3]],
            ]
        )
        text = gaussian_frequency_log(structure, frequencies, modes)
        self.assertIn(" Normal termination of Gaussian", text)
        parsed_frequencies, reduced, constants, parsed_modes = _parse_frequency_blocks(text)
        self.assertEqual(parsed_frequencies, frequencies)
        masses = np.asarray(structure.masses, dtype=float)
        for index, mode in enumerate(np.asarray(modes, dtype=float)):
            squared = float(np.sum(mode ** 2))
            # The printed modes are normalized to sum |u|^2 = 1.
            printed = np.asarray(parsed_modes[index])
            # The printed columns are rounded to two decimals.
            self.assertAlmostEqual(float(np.sum(printed ** 2)), 1.0, delta=0.02)
            expected_reduced = float(np.sum(mode ** 2 * masses[:, np.newaxis])) / squared
            self.assertAlmostEqual(reduced[index], expected_reduced, places=3)
            omega = 2.0 * np.pi * 2.99792458e10 * frequencies[index]
            expected_constant = expected_reduced * AMU_TO_KG * omega ** 2 / 100.0
            self.assertAlmostEqual(constants[index], expected_constant, delta=5e-4)

    def test_gaussian_log_groups_modes_in_threes(self) -> None:
        structure = _diatomic()
        frequencies = [100.0, 200.0, 300.0, 400.0]
        modes = np.zeros((4, 2, 3))
        for index in range(4):
            modes[index, 1, index % 3] = 1.0
        text = gaussian_frequency_log(structure, frequencies, modes)
        lines = text.splitlines()
        headers = [line for line in lines if line.startswith(" Frequencies --")]
        self.assertEqual(len(headers), 2)
        self.assertEqual(
            lines[lines.index(headers[0]) - 2].split(),
            ["1", "2", "3"],
        )
        self.assertEqual(lines[lines.index(headers[1]) - 2].split(), ["4"])

    def test_gaussian_log_thermal_section(self) -> None:
        structure = _diatomic()
        modes = np.zeros((1, 2, 3))
        modes[0, 1, 2] = 1.0
        text = gaussian_frequency_log(
            structure,
            [1000.0],
            modes,
            temperature=298.15,
            electronic_energy=-100.0,
            zero_point_energy=0.5,
            thermo={"internal_energy": 0.6, "free_energy": 0.4},
        )
        self.assertIn(" Temperature   298.150 Kelvin.  Pressure   1.00000 Atm.", text)
        self.assertIn(" Zero-point correction=", text)
        self.assertIn(" Thermal correction to Energy=", text)
        self.assertIn(" Thermal correction to Enthalpy=", text)
        self.assertIn(" Thermal correction to Gibbs Free Energy=", text)
        self.assertIn(" Electronic energy=", text)
        self.assertIn(" Sum of electronic and thermal Free Energies=", text)

    def test_write_gaussian_frequency_log_creates_file(self) -> None:
        structure = _diatomic()
        modes = np.zeros((1, 2, 3))
        modes[0, 1, 2] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "gaussian_fake.log"
            written = write_gaussian_frequency_log(path, structure, [1000.0], modes)
            self.assertEqual(written, path)
            self.assertTrue(path.is_file())
            self.assertIn("Normal termination", path.read_text(encoding="utf-8"))

    def test_gaussian_log_rejects_mismatched_modes(self) -> None:
        structure = _diatomic()
        with self.assertRaises(ValueError):
            gaussian_frequency_log(structure, [1000.0], np.zeros((2, 2, 3)))

    def test_postprocess_writes_the_gaussian_log(self) -> None:
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
                    direction = job_path.name.split("_")[2]
                    sign = 1.0 if direction.endswith("+") else -1.0
                    axis = {"x": 0, "y": 1, "z": 2}[direction[0]]
                    force[0, axis] = -20.0 * sign * 0.01
                return force

            arguments = Namespace(
                job=job,
                version="LTS3.10.1",
                temperature=[298.15],
                traj=False,
                traj_format="extxyz",
                frames=30,
                output_stru=False,
                stru_format="extxyz",
                backend="builtin",
                element_masses=None,
                output="vibration_results.json",
                gaussian_log="gaussian_fake.log",
            )
            with patch(
                "abacustools.commands.workflow.vibration.read_forces",
                side_effect=force_for,
            ):
                self.assertEqual(postprocess(arguments), 0)
            text = (job / "gaussian_fake.log").read_text(encoding="utf-8")
            self.assertIn(" Input orientation:", text)
            self.assertIn(" Frequencies --", text)
            self.assertIn(" Normal termination of Gaussian", text)
            self.assertEqual(len(json.loads((job / "vibration_results.json").read_text())["frequencies"]), 3)


if __name__ == "__main__":
    unittest.main()
