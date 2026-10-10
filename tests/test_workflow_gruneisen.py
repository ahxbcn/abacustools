"""Tests for the Grueneisen workflow.

The postprocessing stage is driven on a model whose frequencies are known to
scale as ``omega ∝ V ** -gamma``: the atoms of the reference cell sit in
independent harmonic wells and the spring constant of each volume is scaled
accordingly, so every mode of every volume has the same, prescribed parameter
and the workflow has to recover it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from abacustools.commands.workflow.gruneisen import postprocess, prepare
from abacustools.data.gruneisen import cell_volume
from abacustools.io.stru import AbacusSTRU

from test_workflow_phonon_postprocess import _build_synthetic_phonon_job


#: Parameter the synthetic model is built with.
GAMMA = 1.5

#: Volume strain of the two strained volumes.
STRAIN = 0.01

#: Edge of the cubic reference cell, in Angstrom.
BASE_CELL = 3.0

_STRU = """ATOMIC_SPECIES
Si 28.0855 Si.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
4 0 0
0 4 0
0 0 4

ATOMIC_POSITIONS
Cartesian

Si
0.0
1
0.0 0.0 0.0
"""


def _source_job(job: Path) -> Path:
    """Write the reference cell the preparation stage starts from."""
    job.mkdir(parents=True, exist_ok=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nscf_thr 1e-8\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(_STRU, encoding="utf-8")
    (job / "Si.upf").write_text("pseudo", encoding="utf-8")
    return job


def _prepare_args(job: Path, **overrides) -> Namespace:
    """Return the preparation arguments of the Grueneisen workflow."""
    values = dict(
        job=job,
        strain=STRAIN,
        supercell=[1, 1, 1],
        min_supercell_length=10.0,
        displacement_stepsize=0.01,
        override=False,
    )
    values.update(overrides)
    return Namespace(**values)


def _postprocess_args(job: Path, **overrides) -> Namespace:
    """Return the postprocessing arguments of the Grueneisen workflow."""
    values = dict(
        job=job,
        version="",
        mesh=[4, 4, 4],
        temperature=300.0,
        tmin=0.0,
        tmax=100.0,
        tstep=10.0,
        qpath=None,
        high_symm_points=None,
        npoints=11,
        symmetrize=True,
        output="gruneisen_results.json",
        mesh_yaml="gruneisen_mesh.yaml",
        band_yaml="gruneisen_band.yaml",
        mesh_plot="gruneisen_mesh.png",
        band_plot="gruneisen_band.png",
    )
    values.update(overrides)
    return Namespace(**values)


def _model_job(job: Path) -> Path:
    """Write a three volume job whose frequencies scale as V ** -gamma.

    Every atom sits in an independent harmonic well with spring constant
    ``k``, so the frequency of every mode follows ``sqrt(k)``.  Scaling the
    spring constant of a volume by ``(1 + strain) ** (-2 gamma)`` therefore
    makes every mode of that volume follow ``V ** -gamma`` exactly, which is
    what the workflow has to measure.
    """
    tasks = ("gruneisen_v0", "gruneisen_vp", "gruneisen_vm")
    volumes = {}
    for name, volume_strain in zip(tasks, (0.0, STRAIN, -STRAIN)):
        factor = (1.0 + volume_strain) ** (1.0 / 3.0)
        cell = BASE_CELL * factor
        _build_synthetic_phonon_job(
            job / name,
            cell=cell,
            supercell=[2, 2, 2],
            force_constant=10.0 * (1.0 + volume_strain) ** (-2.0 * GAMMA),
        )
        volumes[name] = cell_volume(
            [[cell, 0.0, 0.0], [0.0, cell, 0.0], [0.0, 0.0, cell]]
        )
    # The band structure is taken on the reference cell, which the workflow
    # reads from the job directory itself.
    (job / "STRU").write_text(
        (job / "gruneisen_v0" / "STRU").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
    )
    (job / "workflow_gruneisen.json").write_text(
        json.dumps(
            {
                "format": 1,
                "workflow": "gruneisen",
                "tasks": list(tasks),
                "strain": STRAIN,
                "volumes": volumes,
                "supercell": [2, 2, 2],
                "displacement_stepsize": 0.01,
                "stru_filename": "STRU",
            }
        ),
        encoding="utf-8",
    )
    return job


class TestGrueneisenWorkflow(unittest.TestCase):
    def test_prepare_writes_three_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = _source_job(Path(temporary) / "job")

            self.assertEqual(prepare(_prepare_args(job)), 0)

            manifest = json.loads((job / "workflow_gruneisen.json").read_text())
            self.assertEqual(
                manifest["tasks"], ["gruneisen_v0", "gruneisen_vp", "gruneisen_vm"]
            )
            self.assertEqual(manifest["strain"], STRAIN)
            for name, volume_strain in zip(
                manifest["tasks"], (0.0, STRAIN, -STRAIN)
            ):
                volume_job = job / name
                self.assertTrue((volume_job / "INPUT").is_file())
                self.assertTrue((volume_job / "workflow_phonon.json").is_file())
                structure = volume_job / "STRU"
                self.assertTrue(structure.is_file())
                self.assertAlmostEqual(
                    manifest["volumes"][name],
                    cell_volume(AbacusSTRU.read(structure).cell),
                    places=5,
                )
            reference = manifest["volumes"]["gruneisen_v0"]
            self.assertAlmostEqual(
                manifest["volumes"]["gruneisen_vp"] / reference, 1.0 + STRAIN
            )
            self.assertAlmostEqual(
                manifest["volumes"]["gruneisen_vm"] / reference, 1.0 - STRAIN
            )

    def test_prepare_rejects_a_strain_outside_the_working_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = _source_job(Path(temporary) / "job")

            with self.assertRaises(ValueError):
                prepare(_prepare_args(job, strain=0.5))

    def test_postprocess_recovers_a_known_mode_gruneisen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = _model_job(Path(temporary) / "job")

            self.assertEqual(postprocess(_postprocess_args(job)), 0)

            result = json.loads((job / "gruneisen_results.json").read_text())
            self.assertAlmostEqual(result["strain"], STRAIN)
            self.assertAlmostEqual(
                result["mode_gruneisen"]["mean"], GAMMA, delta=0.02 * GAMMA
            )
            # Every mode of the model scales with the volume, so none of them
            # is dropped: the mesh of a one atom cell does not carry the zero
            # frequency translations of Gamma either.
            self.assertEqual(result["mode_gruneisen"]["modes_left_out"], 0)
            self.assertAlmostEqual(
                result["gruneisen_at_temperature"], GAMMA, delta=0.02 * GAMMA
            )
            curve = result["gruneisen_temperature"]
            self.assertIsNone(curve["values"][0])
            for value in curve["values"][1:]:
                self.assertAlmostEqual(value, GAMMA, delta=0.02 * GAMMA)
            self.assertEqual(result["units"]["mode_gruneisen"], "dimensionless")
            self.assertTrue((job / "gruneisen_mesh.yaml").is_file())
            self.assertTrue((job / "gruneisen_band.yaml").is_file())
            self.assertTrue((job / "gruneisen_mesh.png").is_file())
            self.assertTrue((job / "gruneisen_band.png").is_file())

    def test_postprocess_reports_a_manifest_without_three_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = _model_job(Path(temporary) / "job")
            manifest = json.loads((job / "workflow_gruneisen.json").read_text())
            manifest["tasks"] = manifest["tasks"][:2]
            (job / "workflow_gruneisen.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaises(RuntimeError, msg="three volumes"):
                postprocess(_postprocess_args(job))
