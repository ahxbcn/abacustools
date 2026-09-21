"""Tests for the finite-strain piezoelectric workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.commands.workflow.piezoelectric import (
    _ELECTRON_ANGSTROM_SQUARED_TO_CM2,
    _VOIGT_MODES,
    _deformed_structure,
    _strain_matrix,
    postprocess,
    prepare,
)


STRU = """ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
4 0 0
0 4 0
0 0 4

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0
"""


#: A face centred primitive cell, whose lattice vectors are not orthogonal and
#: not aligned with the Cartesian axes, so that a strain has to be applied to
#: the Cartesian components of every lattice vector.
NON_ORTHOGONAL_STRU = """ATOMIC_SPECIES
Zn 65.38 Zn.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
0.0 5.1323 5.1323
5.1323 0.0 5.1323
5.1323 5.1323 0.0

ATOMIC_POSITIONS
Direct

Zn
0.0
1
0.0 0.0 0.0
"""


class TestPiezoelectricWorkflow(unittest.TestCase):
    #: A wurtzite cell, whose point group 6mm leaves three independent
    #: piezoelectric components.
    HEX_STRU = """ATOMIC_SPECIES
Zn 65.38 Zn.upf
S 32.06 S.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
3.82 0 0
-1.91 3.3082 0
0 0 6.26

ATOMIC_POSITIONS
Direct

Zn
0.0
2
0.333333 0.666667 0.0
0.666667 0.333333 0.5

S
0.0
2
0.333333 0.666667 0.375
0.666667 0.333333 0.875
"""

    @staticmethod
    def _write_berry_task(
        task: Path,
        values: list[float],
        volume: float = 64.0,
        modulus: float = 100.0,
    ) -> None:
        output = task / "OUT.ABACUS"
        output.mkdir(parents=True, exist_ok=True)
        for index, value in enumerate(values, start=1):
            (output / f"running_nscf{index}.log").write_text(
                f"Volume (A^3) = {volume}\n"
                f"The calculated polarization direction is in R{index} direction\n"
                f"P = {value} (mod {modulus}) (0.0 0.0 0.0) (e/Omega).bohr\n"
                "P = 0.0 (mod 1.0) (0.0 0.0 0.0) C/m^2\n",
                encoding="utf-8",
            )

    @staticmethod
    def _write_job(path: Path) -> None:
        (path / "INPUT").write_text(
            "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
        )
        (path / "STRU").write_text(STRU, encoding="utf-8")
        (path / "H.upf").write_text("pseudo", encoding="utf-8")

    def test_prepare_generates_six_central_strain_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)

            self.assertEqual(
                prepare(
                    Namespace(
                        job=job,
                        strain=0.01,
                        disp_type="c",
                        use_k_continuity=False,
                        relax=False,
                        abacus_command="abacus",
                        override=False,
                    )
                ),
                0,
            )
            manifest = json.loads(
                (job / "workflow_piezoelectric.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["tasks"]), 12)
            self.assertEqual(
                manifest["tasks"][:3],
                [
                    "piezoelectric_xx",
                    "piezoelectric_xx_back",
                    "piezoelectric_yy",
                ],
            )
            self.assertIn(
                "berry_phase         1",
                (job / "piezoelectric_xx" / "INPUT.nscf1").read_text(),
            )
            self.assertTrue((job / "piezoelectric_xx" / "run.sh").is_symlink())

            from abacustools.io.stru import AbacusSTRU

            strained = AbacusSTRU.read(job / "piezoelectric_xx" / "STRU")
            np.testing.assert_allclose(
                strained.cell,
                [[4.04 * BOHR_TO_ANG, 0, 0], [0, 4 * BOHR_TO_ANG, 0], [0, 0, 4 * BOHR_TO_ANG]],
            )

    def test_postprocess_converts_polarization_change_to_c_per_square_meter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)
            prepare(
                Namespace(
                    job=job,
                    strain=0.1,
                    disp_type="c",
                    use_k_continuity=False,
                    relax=False,
                    abacus_command="abacus",
                    override=False,
                )
            )
            for name in json.loads(
                (job / "workflow_piezoelectric.json").read_text()
            )["tasks"]:
                self._write_berry_task(job / name, [0.0, 0.0, 0.0])

            # For xx, choose a phase change that corresponds to 1 C/m^2 after a
            # central difference of 2 * 0.1: the response is the phase change
            # times the quantum of the reference cell, turned into a density
            # with the reference volume.
            phase = 1.0 * 0.2 * 64.0 / (100.0 * BOHR_TO_ANG * _ELECTRON_ANGSTROM_SQUARED_TO_CM2)
            self._write_berry_task(
                job / "piezoelectric_xx", [phase * 100.0, 0.0, 0.0]
            )
            status = postprocess(
                Namespace(
                    job=job,
                    version="LTS3.10.1",
                    output="result.json",
                    symmetrize=False,
                    fit="full",
                )
            )

            self.assertEqual(status, 0)
            result = json.loads((job / "result.json").read_text())
            self.assertAlmostEqual(result["piezoelectric_tensor"][0][0], 1.0)

    def test_postprocess_ignores_the_branch_point_of_a_centrosymmetric_crystal(
        self,
    ) -> None:
        """A polarization on the branch point must give a vanishing tensor.

        A centrosymmetric crystal has its polarization exactly on the branch
        point, where ABACUS prints either sign of half the quantum.  The two
        signs are the same physical state -- they differ by one period -- so
        the response has to come out as zero and not as half the quantum, which
        is what comparing the printed values directly would give.
        """
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)
            prepare(
                Namespace(
                    job=job,
                    strain=0.01,
                    disp_type="c",
                    relax=False,
                    abacus_command="abacus",
                    override=False,
                )
            )
            for name in json.loads(
                (job / "workflow_piezoelectric.json").read_text()
            )["tasks"]:
                self._write_berry_task(job / name, [0.0, 0.0, 0.0])

            # Both strained cells sit on the branch point, printed with
            # opposite signs and in cells whose quantum differs slightly.
            self._write_berry_task(
                job / "piezoelectric_xx_back", [50.0, 50.0, 50.0], modulus=100.0
            )
            self._write_berry_task(
                job / "piezoelectric_xx",
                [-49.5, -49.5, -49.5],
                modulus=99.0,
                volume=63.5,
            )
            status = postprocess(
                Namespace(
                    job=job,
                    version="LTS3.10.1",
                    output="result.json",
                    symmetrize=False,
                    fit="full",
                )
            )

            self.assertEqual(status, 0)
            tensor = np.asarray(
                json.loads((job / "result.json").read_text())["piezoelectric_tensor"],
                dtype=float,
            )
            np.testing.assert_allclose(tensor, 0.0, atol=1e-9)

    def test_prepare_relax_adds_relaxation_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)
            prepare(
                Namespace(
                    job=job,
                    strain=0.01,
                    disp_type="f",
                    use_k_continuity=True,
                    relax=True,
                    abacus_command="abacus",
                    override=False,
                )
            )
            task = job / "piezoelectric_xx"
            self.assertTrue((task / "INPUT.relax").is_file())
            runner = (task / "run.sh").read_text()
            self.assertIn("cp INPUT.relax INPUT", runner)
            self.assertIn("cp OUT.ABACUS/STRU_ION_D STRU", runner)

    @staticmethod
    def _non_orthogonal_structure(job: Path):
        """Return a face centred primitive cell read from a written STRU."""
        from abacustools.io.stru import AbacusSTRU

        job.mkdir(parents=True, exist_ok=True)
        (job / "STRU").write_text(NON_ORTHOGONAL_STRU, encoding="utf-8")
        return AbacusSTRU.read(job / "STRU")

    def test_deformed_cell_follows_the_cartesian_deformation_gradient(self) -> None:
        """Every mode must deform the lattice vectors by ``I + strain``."""
        with tempfile.TemporaryDirectory() as temporary:
            structure = self._non_orthogonal_structure(Path(temporary))
            cell = np.asarray(structure.cell, dtype=float)
            for label, index, other_index in _VOIGT_MODES:
                with self.subTest(mode=label):
                    strain = _strain_matrix(index, other_index, 0.01)
                    deformed = np.asarray(
                        _deformed_structure(structure, strain).cell, dtype=float
                    )
                    np.testing.assert_allclose(
                        deformed, cell @ (np.eye(3) + strain).T, atol=1e-12
                    )

    def test_normal_strain_scales_the_cartesian_components(self) -> None:
        """A normal strain must act on every lattice vector's x components.

        The first lattice vector of the face centred cell has no x component,
        so its length is unchanged, while the other two are stretched; applying
        the strain by multiplying on the left would stretch the first vector
        instead, which is a different deformation.
        """
        with tempfile.TemporaryDirectory() as temporary:
            structure = self._non_orthogonal_structure(Path(temporary))
            cell = np.asarray(structure.cell, dtype=float)
            strain = _strain_matrix(0, 0, 0.01)
            deformed = np.asarray(
                _deformed_structure(structure, strain).cell, dtype=float
            )

            np.testing.assert_allclose(deformed[:, 0], 1.01 * cell[:, 0], atol=1e-12)
            np.testing.assert_allclose(deformed[:, 1:], cell[:, 1:], atol=1e-12)
            self.assertAlmostEqual(
                float(np.linalg.norm(deformed[0])), float(np.linalg.norm(cell[0]))
            )
            self.assertGreater(
                float(np.linalg.norm(deformed[1])), float(np.linalg.norm(cell[1]))
            )

    def test_shear_modes_use_the_engineering_strain(self) -> None:
        """A requested shear must be the Voigt engineering shear.

        The strain tensor holds half of the requested magnitude in each off
        diagonal element, so that ``S_4 = 2 eps_yz`` and the polarization
        change divided by the requested magnitude is the piezoelectric tensor
        DFPT codes report.
        """
        strain = _strain_matrix(1, 2, 0.01)

        np.testing.assert_allclose(
            strain, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.005], [0.0, 0.005, 0.0]]
        )
        self.assertAlmostEqual(2.0 * strain[1, 2], 0.01)

        with tempfile.TemporaryDirectory() as temporary:
            structure = self._non_orthogonal_structure(Path(temporary))
            cell = np.asarray(structure.cell, dtype=float)
            deformed = np.asarray(
                _deformed_structure(structure, strain).cell, dtype=float
            )
            # A shear in the yz plane stretches the lattice vector that lies
            # along [011] by the strain tensor component, and leaves the one
            # along [110] untouched to first order.
            self.assertAlmostEqual(
                float(np.linalg.norm(deformed[0]) / np.linalg.norm(cell[0])),
                1.005,
                places=6,
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(deformed[2]) / np.linalg.norm(cell[2])), 1.0, places=4
            )

    def test_prepare_defaults_to_no_k_continuity(self) -> None:
        """ABACUS refuses k continuity for the Berry phase steps."""
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)
            prepare(
                Namespace(
                    job=job,
                    strain=0.01,
                    disp_type="f",
                    relax=False,
                    abacus_command="abacus",
                    override=False,
                )
            )
            manifest = json.loads((job / "workflow_piezoelectric.json").read_text())

            self.assertFalse(manifest["use_k_continuity"])
            self.assertNotIn(
                "use_k_continuity", (job / "piezoelectric_xx" / "INPUT.nscf1").read_text()
            )

    def test_prepare_keeps_only_the_independent_strain_modes(self) -> None:
        """A 6mm cell needs three of the six strain modes."""
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n", encoding="utf-8"
            )
            (job / "STRU").write_text(self.HEX_STRU, encoding="utf-8")
            (job / "Zn.upf").write_text("pseudo", encoding="utf-8")
            (job / "S.upf").write_text("pseudo", encoding="utf-8")

            self.assertEqual(
                prepare(
                    Namespace(
                        job=job,
                        strain=0.01,
                        disp_type="c",
                        use_k_continuity=False,
                        relax=False,
                        abacus_command="abacus",
                        override=False,
                        strains="independent",
                    )
                ),
                0,
            )
            manifest = json.loads(
                (job / "workflow_piezoelectric.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["strain_modes"], [0, 2, 3])
            self.assertEqual(manifest["strains_mode"], "independent")
            self.assertEqual(manifest["symmetry"]["point_group"], "6mm")
            self.assertEqual(manifest["symmetry"]["independent_components"], 3)
            self.assertEqual(len(manifest["tasks"]), 6)
            self.assertTrue((job / "piezoelectric_xx" / "INPUT.nscf1").is_file())
            self.assertTrue(
                (job / "piezoelectric_yz_back" / "INPUT.nscf1").is_file()
            )
            self.assertFalse((job / "piezoelectric_yy").exists())

    def test_postprocess_symmetrizes_with_the_crystal_symmetry(self) -> None:
        """A centrosymmetric cell gives a vanishing tensor, the raw fit not."""
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            self._write_job(job)
            prepare(
                Namespace(
                    job=job,
                    strain=0.1,
                    disp_type="c",
                    use_k_continuity=False,
                    relax=False,
                    abacus_command="abacus",
                    override=False,
                )
            )
            for name in json.loads(
                (job / "workflow_piezoelectric.json").read_text()
            )["tasks"]:
                self._write_berry_task(job / name, [0.0, 0.0, 0.0])
            phase = 1.0 * 0.2 * 64.0 / (
                100.0 * BOHR_TO_ANG * _ELECTRON_ANGSTROM_SQUARED_TO_CM2
            )
            self._write_berry_task(
                job / "piezoelectric_xx", [phase * 100.0, 0.0, 0.0]
            )

            status = postprocess(
                Namespace(job=job, version="LTS3.10.1", output="symmetrized.json")
            )

            self.assertEqual(status, 0)
            result = json.loads((job / "symmetrized.json").read_text())
            self.assertAlmostEqual(result["piezoelectric_tensor_raw"][0][0], 1.0)
            for row in result["piezoelectric_tensor"]:
                for value in row:
                    self.assertAlmostEqual(value, 0.0, places=9)
            self.assertEqual(result["independent_components"], {})
            self.assertGreater(result["symmetrization_residual"], 0.9)


if __name__ == "__main__":
    unittest.main()
