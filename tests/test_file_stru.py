"""Tests for structure format conversion and loss warnings."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

from abacustools.commands.file.stru import run
from abacustools.io.stru import (
    AbacusATOM,
    AbacusSTRU,
    StructureConversionWarning,
    convert_structure,
)


def _stru_with_metadata() -> AbacusSTRU:
    return AbacusSTRU(
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        atoms=[
            AbacusATOM(
                label="H",
                element="H",
                coord=(0.0, 0.0, 0.0),
                pp="H.upf",
                orb="H.orb",
                move=(False, True, True),
                velocity=(1.0, 2.0, 3.0),
                mag=1.0,
            )
        ],
        metadata={"atom_type": "cartesian"},
    )


class TestStructureConversion(unittest.TestCase):
    def test_poscar_roundtrip_preserves_selective_dynamics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _stru_with_metadata().write(directory / "POSCAR")
            messages = [str(item.message) for item in caught]
            self.assertTrue(any("pseudopotential" in message for message in messages))

            converted = AbacusSTRU.read(directory / "POSCAR")
            self.assertEqual(converted.moves, [(False, True, True)])

    def test_xyz_requires_or_accepts_an_explicit_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            xyz = directory / "structure.xyz"
            xyz.write_text("1\nH molecule\nH 0 0 0\n", encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                converted = AbacusSTRU.read(xyz)
            self.assertEqual(len(converted.cell), 3)
            self.assertTrue(any("periodic cell" in str(item.message) for item in caught))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                converted = AbacusSTRU.read(
                    xyz,
                    cell=[[4, 0, 0], [0, 5, 0], [0, 0, 6]],
                )
            self.assertEqual(converted.cell, [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])
            self.assertFalse(any("periodic cell" in str(item.message) for item in caught))

    def test_extxyz_extra_per_atom_fields_warn_when_reading_stru(self):
        with tempfile.TemporaryDirectory() as temporary:
            extxyz = Path(temporary) / "structure.extxyz"
            extxyz.write_text(
                "2\n"
                "Properties=species:S:1:pos:R:3:charge:R:1 "
                "Lattice=\"3 0 0 0 3 0 0 0 3\" pbc=\"T T T\"\n"
                "H 0 0 0 1\nH 1 1 1 -1\n",
                encoding="utf-8",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                converted = AbacusSTRU.read(extxyz)
            self.assertIsNotNone(converted)
            self.assertTrue(any("charge" in str(item.message) for item in caught))

    def test_common_ase_formats_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for suffix in ("cif", "xyz", "extxyz", "xsf"):
                output = directory / f"structure.{suffix}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", StructureConversionWarning)
                    self.assertTrue(_stru_with_metadata().write(output))
                self.assertTrue(output.is_file())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", StructureConversionWarning)
                    self.assertIsNotNone(AbacusSTRU.read(output))

    def test_direct_ase_conversion_warns_about_custom_fields(self):
        from ase import Atoms
        import numpy as np

        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
        atoms.arrays["charge"] = np.array([1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AbacusSTRU.from_ase(atoms)
        self.assertTrue(any("charge" in str(item.message) for item in caught))

    def test_convert_structure_and_cli_entrypoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source.stru"
            poscar = directory / "POSCAR"
            output = directory / "converted.stru"
            self.assertTrue(_stru_with_metadata().write(source))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                convert_structure(source, poscar)
            self.assertTrue(poscar.is_file())
            self.assertTrue(any(isinstance(item.message, StructureConversionWarning) for item in caught))

            status = run(type("Args", (), {
                "filename": poscar,
                "output": output,
                "input_format": None,
                "output_format": None,
                "cell": None,
                "direct": False,
                "cartesian": False,
                "empty2x": False,
            })())
            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
