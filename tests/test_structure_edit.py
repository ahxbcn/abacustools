"""Tests for structure editing recipes and the ``file editstru`` command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from abacustools.data.structure import (
    StructureEditError,
    fix_atoms,
    make_supercell,
    select_atoms,
    select_indices,
    with_vacuum,
)
from abacustools.io.stru import AbacusATOM, AbacusSTRU
from abacustools.main import main


def _structure() -> AbacusSTRU:
    """Return a three-atom structure with every attribute filled."""
    return AbacusSTRU(
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]],
        atoms=[
            AbacusATOM(
                label="Si",
                element="Si",
                coord=(0.0, 0.0, 0.0),
                pp="Si.upf",
                orb="Si.orb",
                move=(False, True, True),
                mag=1.5,
            ),
            AbacusATOM(
                label="O",
                element="O",
                coord=(1.0, 1.0, 1.0),
                pp="O.upf",
                orb="O.orb",
            ),
            AbacusATOM(
                label="O",
                element="O",
                coord=(1.0, 1.0, 5.0),
                pp="O.upf",
                orb="O.orb",
            ),
        ],
        metadata={"atom_type": "cartesian"},
    )


def test_make_supercell_replicates_and_keeps_attributes() -> None:
    structure = _structure()

    supercell = make_supercell(structure, [2, 1, 1])

    assert supercell.natoms == 6
    assert supercell.cell[0][0] == pytest.approx(8.0)
    assert supercell.cell[2][2] == pytest.approx(8.0)
    assert supercell.pps[:3] == ["Si.upf", "O.upf", "O.upf"]
    assert supercell.orbs[:3] == ["Si.orb", "O.orb", "O.orb"]
    assert supercell.moves[0] == (False, True, True)
    assert supercell.atom_mags[0] == pytest.approx(1.5)
    # The source is untouched.
    assert structure.natoms == 3 and structure.cell[0][0] == pytest.approx(4.0)


@pytest.mark.parametrize("repeats", [[0, 1, 1], [1, 2], [1, 1, 1.5]])
def test_make_supercell_rejects_bad_factors(repeats) -> None:
    with pytest.raises(StructureEditError):
        make_supercell(_structure(), repeats)


def test_with_vacuum_extends_one_lattice_vector() -> None:
    structure = _structure()

    slab = with_vacuum(structure, 6.0, direction="c")

    lengths = np.linalg.norm(np.asarray(slab.cell, dtype=float), axis=1)
    assert lengths[0] == pytest.approx(4.0)
    assert lengths[1] == pytest.approx(4.0)
    assert lengths[2] == pytest.approx(14.0)
    np.testing.assert_allclose(np.asarray(slab.coords), np.asarray(structure.coords))


def test_with_vacuum_center_moves_the_atoms() -> None:
    structure = _structure()

    slab = with_vacuum(structure, 6.0, direction="c", center=True)

    # Half the added vacuum ends up below the atoms, so they move 3 Angstrom.
    before = np.asarray(structure.coords, dtype=float)[:, 2]
    after = np.asarray(slab.coords, dtype=float)[:, 2]
    np.testing.assert_allclose(after - before, 3.0)


def test_with_vacuum_rejects_a_bad_thickness() -> None:
    with pytest.raises(StructureEditError):
        with_vacuum(_structure(), 0.0)


def test_select_indices_filters() -> None:
    structure = _structure()

    assert select_indices(structure, elements=["O"]) == [1, 2]
    assert select_indices(structure, indices=[0]) == [0]
    # Cartesian z from 0 to 2 Angstrom holds the first two atoms.
    assert (
        select_indices(
            structure, coordinate_range=(0.0, 2.0), direction="z", cartesian=True
        )
        == [0, 1]
    )
    # Filters combine: the last oxygen is not the selected index.
    assert select_indices(structure, indices=[1, 2], elements=["O"]) == [1, 2]
    assert select_indices(structure, indices=[0], elements=["O"]) == []


def test_select_indices_rejects_bad_input() -> None:
    structure = _structure()
    with pytest.raises(StructureEditError, match="needs atom indices"):
        select_indices(structure)
    with pytest.raises(StructureEditError, match="not present"):
        select_indices(structure, elements=["Fe"])
    with pytest.raises(StructureEditError, match="outside 1..3"):
        select_indices(structure, indices=[9])
    with pytest.raises(StructureEditError, match="unknown direction"):
        select_indices(structure, coordinate_range=(0.0, 1.0), direction="q")
    with pytest.raises(StructureEditError, match="larger than maximum"):
        select_indices(structure, coordinate_range=(2.0, 1.0))


def test_select_atoms_keeps_or_drops() -> None:
    structure = _structure()

    oxygens = select_atoms(structure, elements=["O"])
    assert oxygens.natoms == 2
    assert oxygens.elements == ["O", "O"]
    assert oxygens.cell == structure.cell

    without_silicon = select_atoms(structure, indices=[0], remove=True)
    assert without_silicon.elements == ["O", "O"]

    assert structure.natoms == 3


def test_select_atoms_rejects_an_empty_result() -> None:
    with pytest.raises(StructureEditError, match="no atoms"):
        # The index and the element filter cannot both match.
        select_atoms(_structure(), indices=[0], elements=["O"])


def test_fix_atoms_by_index_and_range() -> None:
    structure = _structure()

    fixed = fix_atoms(structure, indices=[0])
    assert fixed.moves[0] == (False, False, False)
    assert fixed.moves[1] == (True, True, True)

    # The oxygen at fractional z = 0.125 sits below the window.
    windowed = fix_atoms(
        structure, coordinate_range=(0.0, 0.1), direction="c", cartesian=False
    )
    assert windowed.moves[0] == (False, False, False)
    assert windowed.moves[1] == (True, True, True)

    partial = fix_atoms(structure, elements=["O"], move=(True, False, False))
    assert partial.moves[1] == (True, False, False)
    assert partial.moves[2] == (True, False, False)
    assert partial.moves[0] == (False, True, True)

    freed = fix_atoms(structure, elements=["O"], free_others=True)
    assert freed.moves[1] == (False, False, False)
    assert freed.moves[0] == (True, True, True)
    assert structure.moves[1] == (True, True, True)


def test_fix_atoms_rejects_an_empty_selection() -> None:
    with pytest.raises(StructureEditError, match="no atom matches"):
        fix_atoms(_structure(), indices=[], elements=None)


def _write_structure(tmp_path: Path) -> Path:
    source = tmp_path / "source.STRU"
    assert _structure().write(str(source))
    return source


def test_editstru_supercell_writes_a_new_file(tmp_path: Path) -> None:
    source = _write_structure(tmp_path)
    output = tmp_path / "super.STRU"

    assert main(["file", "editstru", "supercell", str(source), "-o", str(output), "-n", "2", "1", "1"]) == 0

    result = AbacusSTRU.read(str(output))
    assert result is not None
    assert result.natoms == 6
    assert result.pps[:3] == ["Si.upf", "O.upf", "O.upf"]


def test_editstru_refuses_to_overwrite(tmp_path: Path) -> None:
    source = _write_structure(tmp_path)
    output = tmp_path / "super.STRU"
    output.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists"):
        main(["file", "editstru", "supercell", str(source), "-o", str(output), "-n", "2", "1", "1"])

    assert output.read_text(encoding="utf-8") == "keep me"

    assert main([
        "file", "editstru", "supercell", str(source),
        "-o", str(output), "-n", "2", "1", "1", "--override",
    ]) == 0


def test_editstru_vacuum_reports_json(tmp_path: Path, capsys) -> None:
    source = _write_structure(tmp_path)
    output = tmp_path / "slab.STRU"

    assert main([
        "file", "editstru", "vacuum", str(source),
        "-o", str(output), "-t", "6", "--direction", "c", "--json",
    ]) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout[stdout.index("{") :])
    assert payload["action"] == "vacuum"
    assert payload["atoms_after"] == 3
    assert payload["cell_lengths"][2] == pytest.approx(14.0)
