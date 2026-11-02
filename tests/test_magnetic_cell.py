"""Tests for building and reducing commensurate magnetic cells."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from abacustools.data.magnetism import (
    build_magnetic_supercell,
    magnetic_primitive,
    magnetic_standard_cell,
    magnetic_translation_lattice,
    rationalize_propagation_vector,
)
from abacustools.data.structure import StructureEditError
from abacustools.io.stru import AbacusATOM, AbacusSTRU
from abacustools.main import main

A = 4.28911407


def _rock_salt() -> AbacusSTRU:
    """Return an eight-atom rocksalt cell with AFM-II-ready Co moments."""
    cell = np.eye(3) * A
    cobalt = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    oxygen = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    atoms = [
        AbacusATOM(
            label="Co",
            element="Co",
            coord=tuple(np.asarray(position) @ cell),
            mag=3.0,
        )
        for position in cobalt
    ]
    atoms += [
        AbacusATOM(
            label="O",
            element="O",
            coord=tuple(np.asarray(position) @ cell),
            mag=0.0,
        )
        for position in oxygen
    ]
    return AbacusSTRU(cell=cell.tolist(), atoms=atoms)


def test_rationalize_propagation_vector() -> None:
    """Half-integer k has index 2; zero k keeps the cell."""
    n, numerator = rationalize_propagation_vector([0.5, 0.5, 0.5])
    assert n == 2
    assert list(numerator) == [1, 1, 1]
    n, numerator = rationalize_propagation_vector([0.0, 0.0, 0.0])
    assert n == 1
    assert list(numerator) == [0, 0, 0]


def test_incommensurate_k_is_rejected() -> None:
    """A k that needs a larger denominator has no finite magnetic cell."""
    with pytest.raises(StructureEditError, match="incommensurate"):
        rationalize_propagation_vector([0.37, 0.0, 0.0], max_denominator=12)


@pytest.mark.parametrize("k", [[0.5, 0.5, 0.5], [1 / 3, 0, 0], [0.25, 0.5, 0]])
def test_magnetic_translation_lattice_has_the_expected_index(k) -> None:
    """The supercell matrix is an integer basis with determinant n."""
    n, numerator = rationalize_propagation_vector(k)
    matrix = magnetic_translation_lattice(numerator, n)
    assert abs(round(np.linalg.det(matrix))) == n
    assert np.all((matrix @ numerator) % n == 0)


def test_afm2_supercell_of_the_conventional_cell() -> None:
    """k = (1/2, 1/2, 1/2) doubles the conventional cell."""
    structure = _rock_salt()
    magnetic, summary = build_magnetic_supercell(
        structure, [0.5, 0.5, 0.5], elements=["Co"], moment=3.0
    )
    assert summary["index"] == 2
    assert magnetic.natoms == 16
    assert np.linalg.det(np.asarray(magnetic.cell)) > 0
    moments = [atom.mag for atom in magnetic.atoms if atom.element == "Co"]
    assert moments.count(3.0) == 4
    assert moments.count(-3.0) == 4
    assert sum(moments) == pytest.approx(0.0)
    assert abs(np.linalg.det(np.asarray(magnetic.cell))) == pytest.approx(2 * A**3)


def test_standard_cell_is_the_hexagonal_afm2_cell() -> None:
    """The standardized cell is Co6O6 with a = a_c/sqrt(2), c = 2*sqrt(3)*a_c."""
    structure = _rock_salt()
    magnetic, _ = build_magnetic_supercell(
        structure, [0.5, 0.5, 0.5], elements=["Co"], moment=3.0
    )
    standard, summary = magnetic_standard_cell(magnetic)
    assert standard.natoms == 12
    assert summary["volume_factor"] == pytest.approx(4 / 3)
    lengths = np.linalg.norm(np.asarray(standard.cell), axis=1)
    assert lengths[0] == pytest.approx(A / np.sqrt(2), rel=1e-6)
    assert lengths[2] == pytest.approx(2 * np.sqrt(3) * A, rel=1e-6)

    planes = sorted(
        (fractional[2], atom.mag)
        for atom, fractional in zip(standard.atoms, standard.coords_direct)
        if atom.element == "Co"
    )
    moments = [moment for _, moment in planes]
    assert moments.count(3.0) == 3
    assert moments.count(-3.0) == 3
    # Every consecutive (111) plane is antiferromagnetic.
    assert all(left * right < 0 for left, right in zip(moments, moments[1:]))


def test_primitive_cell_is_the_smallest_magnetic_cell() -> None:
    """The primitive magnetic cell has two Co and two O."""
    structure = _rock_salt()
    magnetic, _ = build_magnetic_supercell(
        structure, [0.5, 0.5, 0.5], elements=["Co"], moment=3.0
    )
    primitive, summary = magnetic_primitive(magnetic)
    assert primitive.natoms == 4
    assert summary["volume_factor"] == 4
    assert abs(np.linalg.det(np.asarray(primitive.cell))) == pytest.approx(A**3 / 2)
    assert sorted(
        atom.mag for atom in primitive.atoms if atom.element == "Co"
    ) == [-3.0, 3.0]


def test_non_collinear_phase_needs_an_explicit_projection() -> None:
    """k = (1/4, 0, 0) produces phases 0, 90, 180, 270 degrees."""
    structure = AbacusSTRU(
        cell=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
        atoms=[AbacusATOM(label="Co", element="Co", coord=(0.0, 0.0, 0.0), mag=3.0)],
    )
    with pytest.raises(StructureEditError, match="allow-projection"):
        build_magnetic_supercell(structure, [0.25, 0.0, 0.0], elements=["Co"])
    projected, _ = build_magnetic_supercell(
        structure,
        [0.25, 0.0, 0.0],
        elements=["Co"],
        allow_projection=True,
    )
    assert projected.natoms == 4
    assert [round(atom.mag, 6) for atom in projected.atoms] == pytest.approx(
        [3.0, 0.0, -3.0, 0.0], abs=1e-6
    )


def test_editstru_magnetic_cli(tmp_path: Path) -> None:
    """The CLI writes the commensurate cell and reports the magnetic group."""
    source = tmp_path / "CoO.STRU"
    output = tmp_path / "CoO_AFM2.STRU"
    _rock_salt().write(str(source))

    sys.argv = [
        "abacustools",
        "file",
        "editstru",
        "magnetic",
        str(source),
        "-o",
        str(output),
        "--k",
        "0.5",
        "0.5",
        "0.5",
        "--elements",
        "Co",
        "--moment",
        "3",
        "--primitive",
    ]
    assert main() == 0
    result = AbacusSTRU.read(str(output))
    assert result.natoms == 4
