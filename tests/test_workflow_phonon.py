"""Tests for the finite-difference phonon workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from abacustools.commands.workflow.phonon import (
    _automatic_supercell,
    _custom_band_path,
    _initialize_phonopy,
    _validate_mesh,
    _validate_supercell,
    prepare,
)


class TestPhononWorkflow(unittest.TestCase):
    def test_supercell_helpers(self) -> None:
        structure = type(
            "Structure",
            (),
            {"cell": [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]]},
        )()
        self.assertEqual(_automatic_supercell(structure, 10.0), [4, 3, 1])
        self.assertEqual(_validate_supercell([1, 2, 3]), [1, 2, 3])
        self.assertEqual(_validate_mesh([2, 3, 4]), [2, 3, 4])
        with self.assertRaises(ValueError):
            _validate_supercell([1, 0, 2])
        with self.assertRaises(ValueError):
            _validate_mesh([1, 2, 0])

    def test_phonopy_generates_displacements(self) -> None:
        from abacustools.io.stru import AbacusATOM, AbacusSTRU

        structure = AbacusSTRU(
            cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            atoms=[AbacusATOM(label="H", element="H", coord=(0.0, 0.0, 0.0))],
            metadata={"atom_type": "cartesian"},
        )
        phonon = _initialize_phonopy(structure, [1, 1, 1])
        phonon.generate_displacements(distance=0.01)
        self.assertEqual(len(phonon.supercells_with_displacements), 1)
        self.assertEqual(phonon.dataset["first_atoms"][0]["number"], 0)

    def test_prepare_writes_manifest_and_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nscf_thr 1e-6\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(
                """ATOMIC_SPECIES
H 1.0

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
3 0 0
0 3 0
0 0 3

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0
""",
                encoding="utf-8",
            )
            result = prepare(
                Namespace(
                    job=job,
                    supercell=[1, 1, 1],
                    displacement_stepsize=0.01,
                    min_supercell_length=10.0,
                    override=False,
                )
            )
            self.assertEqual(result, 0)
            manifest = json.loads((job / "workflow.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["workflow"], "phonon")
            self.assertEqual(manifest["supercell"], [1, 1, 1])
            self.assertEqual(len(manifest["tasks"]), 1)
            self.assertEqual(manifest["displacements"][0]["number"], 0)
            self.assertTrue((job / "disp-1" / "STRU").is_file())
            self.assertIn("scf_thr             1e-07", (job / "disp-1" / "INPUT").read_text())

    def test_custom_band_path(self) -> None:
        (paths, connections), labels = _custom_band_path(
            ["G", "X", "G"],
            {"G": [0, 0, 0], "X": [0.5, 0, 0]},
            5,
        )
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(len(path) == 5 for path in paths))
        self.assertEqual(connections, [True, False])
        self.assertEqual(labels, [r"$\Gamma$", "X", r"$\Gamma$"])


if __name__ == "__main__":
    unittest.main()
