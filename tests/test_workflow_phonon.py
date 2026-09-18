"""Tests for the finite-difference phonon workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from abacustools.commands.workflow.phonon import (
    _custom_band_path,
    _initialize_phonopy,
    prepare,
)
from abacustools.data.phonon import (
    automatic_supercell,
    phonopy_supercell_structure,
    validate_mesh,
    validate_supercell,
)


class TestPhononWorkflow(unittest.TestCase):
    def test_supercell_helpers(self) -> None:
        structure = type(
            "Structure",
            (),
            {"cell": [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]]},
        )()
        self.assertEqual(automatic_supercell(structure, 10.0), [4, 3, 1])
        self.assertEqual(validate_supercell([1, 2, 3]), [1, 2, 3])
        self.assertEqual(validate_mesh([2, 3, 4]), [2, 3, 4])
        with self.assertRaises(ValueError):
            validate_supercell([1, 0, 2])
        with self.assertRaises(ValueError):
            validate_mesh([1, 2, 0])

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
            manifest = json.loads(
                (job / "workflow_phonon.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["workflow"], "phonon")
            self.assertEqual(manifest["supercell"], [1, 1, 1])
            self.assertEqual(len(manifest["tasks"]), 1)
            self.assertEqual(manifest["displacements"][0]["index"], 0)
            self.assertEqual(manifest["displacements"][0]["atom"], 0)
            self.assertEqual(manifest["displacements"][0]["displacement"], [0.01, 0.0, 0.0])
            self.assertEqual(manifest["dataset"][0]["number"], 0)
            self.assertEqual(manifest["tasks"], ["disp-0000"])
            self.assertTrue((job / "disp-0000" / "STRU").is_file())
            self.assertIn("scf_thr             1e-07", (job / "disp-0000" / "INPUT").read_text())

    def test_manifest_displacement_validation(self) -> None:
        from abacustools.commands.workflow.phonon import _manifest_displacements
        from abacustools.io.stru import AbacusATOM, AbacusSTRU

        structure = AbacusSTRU(
            cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            atoms=[
                AbacusATOM(label="H", element="H", coord=(0.0, 0.0, 0.0)),
                AbacusATOM(label="H", element="H", coord=(1.5, 1.5, 1.5)),
            ],
            metadata={"atom_type": "cartesian"},
        )
        phonon = _initialize_phonopy(structure, [1, 1, 1])

        # A well-formed dataset round-trips.
        dataset = _manifest_displacements(
            phonon,
            {"dataset": [{"number": 1, "displacement": [0.0, 0.01, 0.0]}]},
        )
        self.assertEqual(dataset, [{"number": 1, "displacement": [0.0, 0.01, 0.0]}])

        # A displaced atom outside the supercell, a wrong shape and a missing
        # dataset are all rejected rather than silently mis-fitting forces.
        for bad in (
            {"dataset": [{"number": 2, "displacement": [0.0, 0.0, 0.0]}]},
            {"dataset": [{"number": 0, "displacement": [0.0, 0.0]}]},
            {"dataset": [{"number": 0, "displacement": [0.0, float("nan"), 0.0]}]},
            {"dataset": []},
            {},
        ):
            with self.assertRaises(RuntimeError):
                _manifest_displacements(phonon, bad)

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

    def test_supercell_structure_follows_phonopy_atom_order(self) -> None:
        import numpy as np
        from phonopy.structure.atoms import PhonopyAtoms
        from phonopy.structure.cells import get_supercell

        from abacustools.io.stru import AbacusATOM, AbacusSTRU

        structure = AbacusSTRU(
            cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            atoms=[
                AbacusATOM(label="Si", element="Si", coord=(0.0, 0.0, 0.0)),
                AbacusATOM(label="Ge", element="Ge", coord=(1.0, 1.0, 1.0)),
            ],
            metadata={"atom_type": "cartesian"},
        )
        unitcell = PhonopyAtoms(
            symbols=["Si", "Ge"],
            cell=structure.cell,
            scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        phonopy_supercell = get_supercell(unitcell, np.diag([2, 1, 1]))

        supercell = phonopy_supercell_structure(structure, phonopy_supercell)

        self.assertEqual(supercell.elements, list(phonopy_supercell.symbols))
        np.testing.assert_allclose(
            np.asarray(supercell.coords, dtype=float),
            np.asarray(phonopy_supercell.positions, dtype=float),
        )

    def test_prepare_writes_supercells_in_phonopy_order(self) -> None:
        import numpy as np
        from phonopy.structure.atoms import PhonopyAtoms
        from phonopy.structure.cells import get_supercell

        from abacustools.io.stru import AbacusSTRU

        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "INPUT").write_text(
                "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\n",
                encoding="utf-8",
            )
            (job / "STRU").write_text(
                """ATOMIC_SPECIES
Si 28.0855
Ge 72.63

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
0 0 0

Ge
0.0
1
1 1 1
""",
                encoding="utf-8",
            )
            prepare(
                Namespace(
                    job=job,
                    supercell=[2, 1, 1],
                    displacement_stepsize=0.01,
                    min_supercell_length=10.0,
                    override=False,
                )
            )

            parsed = AbacusSTRU.read(str(job / "STRU"))
            unitcell = PhonopyAtoms(
                symbols=parsed.elements,
                cell=np.asarray(parsed.cell, dtype=float),
                scaled_positions=np.asarray(parsed.coords_direct, dtype=float),
            )
            expected = get_supercell(unitcell, np.diag([2, 1, 1]))
            written = AbacusSTRU.read(str(job / "disp-0000" / "STRU"))

            self.assertEqual(written.elements, list(expected.symbols))
            np.testing.assert_allclose(
                np.asarray(written.coords_direct, dtype=float),
                np.asarray(expected.scaled_positions, dtype=float),
                atol=0.01,
            )


if __name__ == "__main__":
    unittest.main()
