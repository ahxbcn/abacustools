"""Tests for structure format conversion and loss warnings."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pytest

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


def test_from_pymatgen_conversion_writes_a_stru() -> None:
    from pymatgen.core import Lattice, Structure

    pymatgen_structure = Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )

    structure = AbacusSTRU.from_pymatgen(pymatgen_structure)

    assert structure.natoms == 2
    assert structure.labels == ["Si", "Si"]
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "STRU"
        assert structure.write(str(destination))
        assert "ATOMIC_SPECIES" in destination.read_text()


def test_from_phonopy_conversion_keeps_symbols_masses_and_moments() -> None:
    import numpy as np
    from phonopy.structure.atoms import PhonopyAtoms

    phonopy_structure = PhonopyAtoms(
        symbols=["Na", "Cl"],
        cell=np.eye(3) * 5.6,
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        magnetic_moments=[1.0, -1.0],
    )

    structure = AbacusSTRU.from_phonopy(phonopy_structure)

    assert structure.labels == ["Na", "Cl"]
    assert structure.elements == ["Na", "Cl"]
    assert structure.masses == pytest.approx(list(phonopy_structure.masses))
    assert structure.atom_mags == [1.0, -1.0]
    assert structure.cell[0] == [5.6, 0.0, 0.0]
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "STRU"
        assert structure.write(str(destination))
        assert "ATOMIC_SPECIES" in destination.read_text()


def test_from_phonopy_without_magnetic_moments_leaves_them_unset() -> None:
    import numpy as np
    from phonopy.structure.atoms import PhonopyAtoms

    phonopy_structure = PhonopyAtoms(
        symbols=["Si"], cell=np.eye(3) * 5.43, positions=[[0.0, 0.0, 0.0]]
    )

    structure = AbacusSTRU.from_phonopy(phonopy_structure)

    assert structure.atom_mags == [0.0]


def test_phonopy_roundtrip_keeps_the_lattice_and_species() -> None:
    import numpy as np
    from phonopy.structure.atoms import PhonopyAtoms

    phonopy_structure = PhonopyAtoms(
        symbols=["Na", "Cl"],
        cell=np.eye(3) * 5.6,
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    structure = AbacusSTRU.from_phonopy(phonopy_structure)
    roundtripped = AbacusSTRU.from_phonopy(structure.to("phonopy"))

    assert roundtripped.labels == structure.labels
    assert roundtripped.cell == structure.cell


def test_str_summarises_composition_and_cell() -> None:
    structure = AbacusSTRU(
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        atoms=[
            AbacusATOM(label="Si", element="Si", coord=(0.0, 0.0, 0.0)),
            AbacusATOM(label="O", element="O", coord=(1.0, 1.0, 1.0)),
            AbacusATOM(label="Si", element="Si", coord=(2.0, 2.0, 2.0)),
        ],
        metadata={"atom_type": "cartesian"},
    )

    text = str(structure)

    assert text.splitlines()[0] == "ABACUS STRU object:Si2O1"
    assert "NAtoms: 3" in text
    assert "Cell vectors (Angstrom):" in text
    assert "3.0000000" in text


VECTOR_MOMENT_STRU = """\
ATOMIC_SPECIES
Fe 55.845 Fe.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
5 0 0
0 5 0
0 0 5

ATOMIC_POSITIONS
Cartesian

Fe
0.0
2
0 0 0 mag 0 0 2.5
2 2 2 mag 2.5 0 0
"""


def test_vector_magnetic_moments_are_read_and_written(tmp_path: Path) -> None:
    structure_file = tmp_path / "STRU"
    structure_file.write_text(VECTOR_MOMENT_STRU, encoding="utf-8")

    structure = AbacusSTRU.read(str(structure_file))

    assert structure.atom_mags == [(0.0, 0.0, 2.5), (2.5, 0.0, 0.0)]
    assert structure.atoms[0].atommag_magnitude == pytest.approx(2.5)
    assert structure.atoms[0].atommag == (0.0, 0.0, 2.5)

    output = tmp_path / "OUT.STRU"
    assert structure.write(str(output))
    roundtripped = AbacusSTRU.read(str(output))
    assert roundtripped.atom_mags == [(0.0, 0.0, 2.5), (2.5, 0.0, 0.0)]


if __name__ == "__main__":
    unittest.main()
