"""Tests for the optional ABACUS ASE interface adapters."""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from abacustools.integrations.abacuslite import (
    AbacusLiteUnavailableError,
    calculator_from_job,
    calculator_from_structure,
    structure_to_atoms,
    write_result,
)


def _structure(moves=None):
    from abacustools.io.stru import AbacusATOM, AbacusSTRU

    return AbacusSTRU(
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        atoms=[
            AbacusATOM(
                label="H",
                element="H",
                coord=(0.0, 0.0, 0.0),
                pp="H.upf",
                orb="H.orb",
                move=moves[0] if moves else (True, True, True),
            ),
            AbacusATOM(
                label="H",
                element="H",
                coord=(1.0, 1.0, 1.0),
                pp="H.upf",
                orb="H.orb",
                move=moves[1] if moves else (True, True, True),
            ),
        ],
        metadata={"atom_type": "cartesian"},
    )


class TestAbacusLiteIntegration(unittest.TestCase):
    def test_missing_optional_dependency_has_actionable_error(self):
        with patch.dict(sys.modules, {"abacuslite": None}):
            from abacustools.integrations.abacuslite import load_abacuslite

            with self.assertRaises(AbacusLiteUnavailableError):
                load_abacuslite()

    def test_structure_conversion_keeps_constraints_and_files(self):
        atoms = structure_to_atoms(
            _structure(moves=[(False, False, False), (True, False, True)])
        )

        self.assertEqual(atoms.info["pp"], {"H": "H.upf"})
        self.assertEqual(atoms.info["orb"], {"H": "H.orb"})
        self.assertEqual(len(atoms.constraints), 2)
        self.assertTrue(any(type(c).__name__ == "FixAtoms" for c in atoms.constraints))
        self.assertTrue(
            any(type(c).__name__ == "FixCartesian" for c in atoms.constraints)
        )

    def test_calculator_uses_structure_files_and_does_not_require_abacuslite_at_import(self):
        class FakeAbacus:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake = types.ModuleType("abacuslite")
        fake.Abacus = FakeAbacus
        fake.AbacusProfile = object
        with patch.dict(sys.modules, {"abacuslite": fake}):
            calculator = calculator_from_structure(
                _structure(), object(), directory="calc", inp={"calculation": "scf"}
            )

        self.assertEqual(calculator.kwargs["pseudopotentials"], {"H": "H.upf"})
        self.assertEqual(calculator.kwargs["basissets"], {"H": "H.orb"})
        self.assertEqual(calculator.kwargs["directory"], Path("calc"))

    def test_calculator_from_job_reuses_input_and_structure(self):
        class FakeAbacus:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake = types.ModuleType("abacuslite")
        fake.Abacus = FakeAbacus
        fake.AbacusProfile = object
        with TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
                encoding="utf-8",
            )
            self.assertTrue(_structure().write(job / "STRU"))
            with patch.dict(sys.modules, {"abacuslite": fake}):
                calculator = calculator_from_job(
                    job, object(), inp={"cal_force": 1}
                )

        self.assertEqual(calculator.kwargs["inp"]["calculation"], "scf")
        self.assertEqual(calculator.kwargs["inp"]["cal_force"], 1)
        self.assertEqual(calculator.kwargs["inp"]["stru_file"], "STRU")

    def test_result_conversion_and_write(self):
        calculator = types.SimpleNamespace(
            results={
                "energy": -1.25,
                "forces": np.array([[1.0, 2.0, 3.0]]),
                "stress": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            },
            last_scf_converged=True,
        )
        with TemporaryDirectory() as temporary:
            output = write_result(
                Path(temporary) / "result.json",
                calculator,
                trajectory="trajectory.traj",
            )
            self.assertEqual(output["free_energy"], -1.25)
            self.assertEqual(output["force"], [[1.0, 2.0, 3.0]])
            self.assertTrue(output["converged"])
            self.assertEqual(
                json.loads((Path(temporary) / "result.json").read_text()), output
            )


if __name__ == "__main__":
    unittest.main()
