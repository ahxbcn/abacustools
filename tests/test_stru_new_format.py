"""Tests for the newer ABACUS STRU conventions.

The develop ABACUS writer emits ``Cartesian_angstrom`` coordinates, per-atom
force fields (``f``), comment annotations after the block keywords and a
``pp_type`` column in ``ATOMIC_SPECIES``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacustools.io.stru import BOHR2A, AbacusSTRU


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_cartesian_angstrom_with_forces(tmp_path: Path) -> None:
    stru = _write(
        tmp_path / "STRU",
        """# ABACUS version: v3.11.0-beta10
ATOMIC_SPECIES
Si 28.0855 Si.upf upf201

NUMERICAL_ORBITAL
Si.orb

LATTICE_CONSTANT
1.8897268778 # in Bohr (= 1 Angstrom); lattice vectors below are in Angstrom

LATTICE_VECTORS # in Angstrom
5.43 0.0 0.0
0.0 5.43 0.0
0.0 0.0 5.43

ATOMIC_POSITIONS
Cartesian_angstrom # positions in Angstrom, forces in eV/Angstrom

Si #label
0.0 #magnetism (default, overridden by per-atom mag below)
1 #number of atoms
0.1 0.2 0.3 m 1 1 1 f 0.01 -0.02 0.03 mag 0.5
""",
    )

    structure = AbacusSTRU.read(str(stru))

    assert structure.natoms == 1
    # ABACUS's Bohr-to-Angstrom constant differs from abacustools' by ~5e-7.
    np.testing.assert_allclose(np.diag(structure.cell), [5.43, 5.43, 5.43], atol=1e-4)
    atom = structure.atoms[0]
    np.testing.assert_allclose(atom.coord, [0.1, 0.2, 0.3], atol=1e-9)
    assert atom.force == pytest.approx((0.01, -0.02, 0.03))
    assert atom.mag == pytest.approx(0.5)
    assert tuple(atom.move) == (True, True, True)
    assert atom.pp_type == "upf201"


def test_cartesian_is_in_lattice_constant_units(tmp_path: Path) -> None:
    stru = _write(
        tmp_path / "STRU",
        """ATOMIC_SPECIES
H 1.008 H.upf

LATTICE_CONSTANT
2.0

LATTICE_VECTORS
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0

ATOMIC_POSITIONS
Cartesian

H
0.0
1
1.0 0.0 0.0
""",
    )

    structure = AbacusSTRU.read(str(stru))
    # Classic Cartesian values are multiples of lat0 (Bohr).
    np.testing.assert_allclose(structure.atoms[0].coord, [2.0 * BOHR2A, 0.0, 0.0], atol=1e-12)


def test_cartesian_au_is_bohr(tmp_path: Path) -> None:
    stru = _write(
        tmp_path / "STRU",
        """ATOMIC_SPECIES
H 1.008 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0

ATOMIC_POSITIONS
Cartesian_au

H
0.0
1
1.0 0.0 0.0
""",
    )

    structure = AbacusSTRU.read(str(stru))
    np.testing.assert_allclose(structure.atoms[0].coord, [BOHR2A, 0.0, 0.0], atol=1e-12)


def test_cartesian_angstrom_center_xyz(tmp_path: Path) -> None:
    stru = _write(
        tmp_path / "STRU",
        """ATOMIC_SPECIES
H 1.008 H.upf

LATTICE_CONSTANT
1.889726125457828

LATTICE_VECTORS
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0

ATOMIC_POSITIONS
Cartesian_angstrom_center_xyz

H
0.0
1
0.0 0.0 0.0
""",
    )

    structure = AbacusSTRU.read(str(stru))
    np.testing.assert_allclose(structure.atoms[0].coord, [2.5, 2.5, 2.5], atol=1e-6)


def test_round_trip_preserves_forces(tmp_path: Path) -> None:
    stru = _write(
        tmp_path / "STRU",
        """ATOMIC_SPECIES
C 12.011 C.upf upf201
H 1.008 H.upf upf201

NUMERICAL_ORBITAL
C.orb
H.orb

LATTICE_CONSTANT
1.889726125457828

LATTICE_VECTORS
8.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 8.0

ATOMIC_POSITIONS
Cartesian_angstrom # positions in Angstrom, forces in eV/Angstrom

C #label
0.0 #magnetism (default, overridden by per-atom mag below)
2 #number of atoms
0.0 0.0 0.0 m 1 1 1 f 0.1 0.2 0.3 mag 0.0
1.0 1.0 1.0 m 0 0 1 f -0.1 -0.2 -0.3 mag 0.0
""",
    )

    first = AbacusSTRU.read(str(stru))
    out = tmp_path / "STRU_out"
    assert first.write(str(out), fmt="stru") is True
    second = AbacusSTRU.read(str(out))

    np.testing.assert_allclose(
        np.array([atom.force for atom in first.atoms]),
        np.array([atom.force for atom in second.atoms]),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.array(first.coords), np.array(second.coords), atol=1e-6
    )
    assert tuple(second.atoms[1].move) == (False, False, True)
    assert {atom.pp_type for atom in second.atoms} == {"upf201"}
