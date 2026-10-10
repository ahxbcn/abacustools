"""Tests for the energy-strain elastic workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from abacustools.commands.workflow import energy_strain
from abacustools.commands.workflow.energy_strain import postprocess, prepare
from abacustools.data.eos import EV_PER_ANGSTROM3_TO_GPA
from abacustools.io.stru import AbacusSTRU


STRU = """ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
4 0 0
0 4 0
0 0 4

ATOMIC_POSITIONS
Direct

H
0.0
1
0 0 0
"""


def _source_job(job: Path) -> Path:
    """Write the reference cell the preparation stage starts from."""
    job.mkdir(parents=True, exist_ok=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
    )
    (job / "STRU").write_text(STRU, encoding="utf-8")
    (job / "H.upf").write_text("pseudo", encoding="utf-8")
    return job


def _prepare_args(job: Path, **overrides) -> Namespace:
    """Return the preparation arguments of the energy-strain workflow."""
    values = dict(job=job, strain=0.01, norelax=True, override=False)
    values.update(overrides)
    return Namespace(**values)


class TestEnergyStrainWorkflow(unittest.TestCase):
    def test_prepare_writes_the_reference_and_the_strained_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = _source_job(Path(temporary) / "job")

            self.assertEqual(prepare(_prepare_args(job)), 0)

            manifest = json.loads(
                (job / "workflow_energy-strain.json").read_text(encoding="utf-8")
            )
            # A cubic cell needs three patterns of four amplitudes each.
            self.assertEqual(len(manifest["patterns"]), 3)
            self.assertEqual(len(manifest["strains"]), 12)
            self.assertEqual(len(manifest["tasks"]), 13)
            self.assertEqual(manifest["symmetry"]["point_group"], "m-3m")
            self.assertEqual(manifest["symmetry"]["independent_constants"], 3)
            self.assertTrue((job / "org" / "STRU").is_file())
            self.assertTrue((job / "strained_11" / "STRU").is_file())
            self.assertFalse((job / "strained_12").exists())

            reference = AbacusSTRU.read(job / "org" / "STRU")
            strained = AbacusSTRU.read(job / "strained_00" / "STRU")
            ratio = abs(np.linalg.det(strained.cell) / np.linalg.det(reference.cell))
            # The first pattern is a uniaxial strain of -1% along x.
            self.assertAlmostEqual(ratio, np.sqrt(0.98), places=8)
            # A homogeneous strain leaves the fractional coordinates alone.
            np.testing.assert_allclose(
                strained.coords_direct, reference.coords_direct, atol=1e-8
            )

    def test_postprocess_recovers_a_known_tensor_from_the_energies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = _source_job(Path(temporary) / "job")
            prepare(_prepare_args(job))
            manifest = json.loads(
                (job / "workflow_energy-strain.json").read_text(encoding="utf-8")
            )
            volume = float(
                abs(np.linalg.det(AbacusSTRU.read(job / "STRU").cell))
            )
            tensor = np.zeros((6, 6), dtype=float)
            tensor[:3, :3] = 20.0
            np.fill_diagonal(tensor[:3, :3], 100.0)
            tensor[3, 3] = tensor[4, 4] = tensor[5, 5] = 40.0
            reference_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.05])
            energies = {}
            for name, strain in zip(manifest["strained_paths"], manifest["strains"]):
                vector = np.asarray(strain, dtype=float)
                energies[name] = (
                    volume
                    * float(reference_stress @ vector)
                    / EV_PER_ANGSTROM3_TO_GPA
                    + 0.5
                    * volume
                    * float(vector @ tensor @ vector)
                    / EV_PER_ANGSTROM3_TO_GPA
                )

            original = energy_strain._read_energy
            energy_strain._read_energy = (  # type: ignore[assignment]
                lambda job_dir, version, require: energies[Path(job_dir).name]
            )
            try:
                exit_code = postprocess(
                    Namespace(
                        job=job,
                        version="",
                        output="energy_strain_results.json",
                        symprec=1e-2,
                    )
                )
            finally:
                energy_strain._read_energy = original  # type: ignore[assignment]

            self.assertEqual(exit_code, 0)
            result = json.loads(
                (job / "energy_strain_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["fit"]["method"], "energy-strain")
            self.assertLess(result["fit"]["energy_residual"], 1e-9)
            constants = result["independent_constants"]
            self.assertAlmostEqual(constants["C11"], 100.0, places=6)
            self.assertAlmostEqual(constants["C12"], 20.0, places=6)
            self.assertAlmostEqual(constants["C44"], 40.0, places=6)


if __name__ == "__main__":
    unittest.main()
