"""Tests for structure editing recipes and the ``file editstru`` command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from abacustools.data.structure import (
    StructureEditError,
    build_slab,
    fix_atoms,
    fix_slab_bottom,
    make_supercell,
    select_atoms,
    select_indices,
    set_coordinate_mode,
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


def test_stru_reading_uses_the_declared_atomic_masses(tmp_path: Path) -> None:
    """The masses of ATOMIC_SPECIES reach the atoms, whatever the block order."""
    path = tmp_path / "STRU"
    path.write_text(
        """ATOMIC_SPECIES
H 2.014 D.upf
O 15.9994 O.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Cartesian

O
0.0
1
0 0 0
H
0.0
2
0 0 1
0 1 0
""",
        encoding="utf-8",
    )
    structure = AbacusSTRU.read(path)
    assert structure is not None
    assert structure.labels == ["O", "H", "H"]
    assert structure.masses == [15.9994, 2.014, 2.014]

    written = tmp_path / "roundtrip.STRU"
    assert structure.write(written)
    assert "H 2.014 D.upf" in written.read_text(encoding="utf-8")
    reread = AbacusSTRU.read(written)
    assert reread is not None
    assert reread.masses == structure.masses


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


def test_set_coordinate_mode_switches_the_representation() -> None:
    structure = _structure()

    direct = set_coordinate_mode(structure, "direct")
    assert direct.metadata["atom_type"] == "direct"
    # The positions themselves do not move.
    np.testing.assert_allclose(np.asarray(direct.coords), np.asarray(structure.coords))
    assert structure.metadata["atom_type"] == "cartesian"

    assert set_coordinate_mode(direct, "fractional").metadata["atom_type"] == "direct"
    assert set_coordinate_mode(direct, "cartesian").metadata["atom_type"] == "cartesian"
    assert set_coordinate_mode(structure, "cart").metadata["atom_type"] == "cartesian"


def test_set_coordinate_mode_rejects_an_unknown_mode() -> None:
    with pytest.raises(StructureEditError, match="unknown coordinate mode"):
        set_coordinate_mode(_structure(), "polar")


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


def test_build_slab_cuts_a_surface_with_vacuum_along_c() -> None:
    structure = _structure()

    slab = build_slab(structure, miller_indices=(1, 0, 0), layers=2, vacuum=15.0)

    lengths = np.linalg.norm(np.asarray(slab.cell, dtype=float), axis=1)
    assert slab.natoms > 0
    assert lengths[2] >= 15.0
    assert lengths[2] > lengths[0]
    # The pseudopotential data survives the ASE round trip.
    assert set(slab.pps) == {"Si.upf", "O.upf"}
    assert set(slab.orbs) == {"Si.orb", "O.orb"}
    # Only the element that had a moment keeps one; ASE's zero fill is removed.
    assert {atom.mag for atom in slab.atoms} == {1.5, None}
    assert structure.natoms == 3


def test_build_slab_moves_the_vacuum_to_the_requested_direction() -> None:
    structure = _structure()

    slab = build_slab(
        structure,
        miller_indices=(1, 0, 0),
        layers=1,
        vacuum=12.0,
        vacuum_direction="a",
    )

    lengths = np.linalg.norm(np.asarray(slab.cell, dtype=float), axis=1)
    assert lengths[0] >= 12.0
    assert lengths[0] > lengths[1]


def test_build_slab_applies_an_in_plane_supercell() -> None:
    structure = _structure()

    plain = build_slab(structure, layers=1, vacuum=5.0)
    wider = build_slab(structure, layers=1, vacuum=5.0, surface_supercell=(2, 1))

    assert wider.natoms == 2 * plain.natoms
    assert set(wider.pps) == {"Si.upf", "O.upf"}


def test_build_slab_adds_atoms_with_more_layers() -> None:
    structure = _structure()

    one = build_slab(structure, layers=1, vacuum=5.0)
    three = build_slab(structure, layers=3, vacuum=5.0)

    assert three.natoms > one.natoms


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"miller_indices": (0, 0, 0)}, "must not all be zero"),
        ({"miller_indices": (1, 0)}, "three integers"),
        ({"layers": 0}, "must be positive"),
        ({"vacuum": -1.0}, "non-negative"),
        ({"vacuum_direction": "d"}, "unknown vacuum direction"),
        ({"surface_supercell": (1,)}, "two factors"),
    ],
)
def test_build_slab_rejects_bad_parameters(kwargs, message: str) -> None:
    with pytest.raises(StructureEditError, match=message):
        build_slab(_structure(), **kwargs)


def test_build_slab_rejects_empty_atoms() -> None:
    structure = _structure()
    structure.atoms[0].label = "Si_empty"

    with pytest.raises(StructureEditError, match="empty atoms"):
        build_slab(structure)


def test_fix_slab_bottom_fixes_the_lower_half() -> None:
    # Fractional c is 0, 0.125 and 0.625, so the cut sits at 0.3125.
    fixed = fix_slab_bottom(_structure(), direction="c")

    assert fixed.moves[0] == (False, False, False)
    assert fixed.moves[1] == (False, False, False)
    assert fixed.moves[2] == (True, True, True)


def test_fix_slab_bottom_honours_the_fraction() -> None:
    everything = fix_slab_bottom(_structure(), fraction=1.0)
    assert everything.moves == [(False, False, False)] * 3

    lowest_only = fix_slab_bottom(_structure(), fraction=0.0)
    assert lowest_only.moves[0] == (False, False, False)
    assert lowest_only.moves[1] == (True, True, True)


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_fix_slab_bottom_rejects_a_bad_fraction(fraction: float) -> None:
    with pytest.raises(StructureEditError, match="between 0 and 1"):
        fix_slab_bottom(_structure(), fraction=fraction)


def test_build_slab_can_fix_the_bottom_half() -> None:
    slab = build_slab(_structure(), layers=3, vacuum=10.0, fix_fraction=0.5)

    moves = slab.moves
    assert (False, False, False) in moves
    assert (True, True, True) in moves
    # The fixed atoms are the ones lower along the vacuum direction.
    direct = np.asarray(slab.coords_direct, dtype=float)[:, 2]
    fixed = [value for value, move in zip(direct, moves) if move == (False, False, False)]
    free = [value for value, move in zip(direct, moves) if move == (True, True, True)]
    assert max(fixed) < min(free)


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
    assert result.pps[:3] == ["Si.upf", "Si.upf", "O.upf"]


def test_editstru_supercell_groups_atoms_into_one_species_block(tmp_path: Path) -> None:
    source = _write_structure(tmp_path)
    output = tmp_path / "super.STRU"

    assert main(["file", "editstru", "supercell", str(source), "-o", str(output), "-n", "2", "1", "1"]) == 0

    text = output.read_text(encoding="utf-8")
    species = [
        line.split()[0]
        for line in text.split("ATOMIC_SPECIES", 1)[1].split("NUMERICAL_ORBITAL", 1)[0].strip().splitlines()
    ]
    assert species == ["Si", "O"]

    result = AbacusSTRU.read(str(output))
    assert result is not None
    assert result.labels == ["Si", "Si", "O", "O", "O", "O"]


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


def test_editstru_slab_writes_a_slab(tmp_path: Path) -> None:
    source = _write_structure(tmp_path)
    output = tmp_path / "slab.STRU"

    assert main([
        "file", "editstru", "slab", str(source),
        "-o", str(output), "--miller", "1", "0", "0",
        "--layers", "2", "--vacuum", "10", "--vacuum-direction", "c",
    ]) == 0

    slab = AbacusSTRU.read(str(output))
    assert slab is not None
    assert slab.natoms > 0
    assert set(slab.pps) == {"Si.upf", "O.upf"}
    lengths = np.linalg.norm(np.asarray(slab.cell, dtype=float), axis=1)
    assert lengths[2] >= 10.0


def test_editstru_slab_fix_flag_sets_constraints(tmp_path: Path) -> None:
    source = _write_structure(tmp_path)
    output = tmp_path / "fixed.STRU"

    assert main([
        "file", "editstru", "slab", str(source),
        "-o", str(output), "--layers", "3", "--vacuum", "10", "--fix",
    ]) == 0

    slab = AbacusSTRU.read(str(output))
    assert slab is not None
    assert (False, False, False) in slab.moves
    assert (True, True, True) in slab.moves

    free_output = tmp_path / "free.STRU"
    assert main([
        "file", "editstru", "slab", str(source),
        "-o", str(free_output), "--layers", "3", "--vacuum", "10",
    ]) == 0
    free_slab = AbacusSTRU.read(str(free_output))
    assert free_slab is not None
    assert set(free_slab.moves) == {(True, True, True)}


def test_editstru_coordinate_actions_write_each_representation(tmp_path: Path) -> None:
    source = _write_structure(tmp_path)
    original = AbacusSTRU.read(str(source))
    assert original is not None

    direct = tmp_path / "direct.STRU"
    assert main([
        "file", "editstru", "direct", str(source), "-o", str(direct),
    ]) == 0
    assert "ATOMIC_POSITIONS\nDirect" in direct.read_text(encoding="utf-8")
    converted = AbacusSTRU.read(str(direct))
    assert converted is not None
    np.testing.assert_allclose(
        np.asarray(converted.coords_direct, dtype=float),
        np.asarray(original.coords_direct, dtype=float),
        atol=1e-8,
    )

    cartesian = tmp_path / "cartesian.STRU"
    assert main([
        "file", "editstru", "cartesian", str(direct), "-o", str(cartesian),
    ]) == 0
    assert "ATOMIC_POSITIONS\nCartesian" in cartesian.read_text(encoding="utf-8")
    back = AbacusSTRU.read(str(cartesian))
    assert back is not None
    np.testing.assert_allclose(
        np.asarray(back.coords, dtype=float),
        np.asarray(original.coords, dtype=float),
        atol=1e-8,
    )
    assert back.pps == original.pps


def test_editstru_coordinate_actions_are_idempotent(tmp_path: Path) -> None:
    source = _write_structure(tmp_path)
    output = tmp_path / "cartesian.STRU"

    # Asking for the representation the structure already uses changes nothing.
    assert main([
        "file", "editstru", "cartesian", str(source), "-o", str(output),
    ]) == 0

    original = AbacusSTRU.read(str(source))
    converted = AbacusSTRU.read(str(output))
    assert original is not None and converted is not None
    np.testing.assert_allclose(
        np.asarray(converted.coords, dtype=float),
        np.asarray(original.coords, dtype=float),
        atol=1e-8,
    )


# Tests for primitive/standardize/conventional cell functions

def test_find_primitive_reduces_atoms(tmp_path):
    """Primitive cell should have fewer atoms than the input."""
    from abacustools.data.structure import find_primitive
    from abacustools.io.stru import AbacusSTRU

    # Create an FCC structure (4 atoms in conventional cell)
    structure = AbacusSTRU.read("tests/fixtures/FCC_STRU")
    if structure is None:
        pytest.skip("FCC_STRU fixture not available")

    primitive = find_primitive(structure)
    assert primitive.natoms < structure.natoms


def test_find_primitive_preserves_attributes(tmp_path):
    """Primitive cell should preserve pseudopotential and orbital files."""
    from abacustools.data.structure import find_primitive
    from abacustools.io.stru import AbacusSTRU

    # Use NiO structure
    structure = AbacusSTRU.read("/mnt/e/profsoftfiles/abacusfiles/sp/NiO-DFTU/DFTU/STRU")
    if structure is None:
        pytest.skip("NiO STRU not available")

    primitive = find_primitive(structure)

    # Check that pseudopotential files are preserved
    pp_files = {atom.pp for atom in primitive.atoms if atom.pp}
    original_pp = {atom.pp for atom in structure.atoms if atom.pp}
    assert pp_files == original_pp

    # Check that orbital files are preserved
    orb_files = {atom.orb for atom in primitive.atoms if atom.orb}
    original_orb = {atom.orb for atom in structure.atoms if atom.orb}
    assert orb_files == original_orb


def test_find_conventional_increases_atoms(tmp_path):
    """Conventional cell may have more atoms than primitive."""
    from abacustools.data.structure import find_conventional
    from abacustools.io.stru import AbacusSTRU

    # Use NiO structure (rhombohedral setting)
    structure = AbacusSTRU.read("/mnt/e/profsoftfiles/abacusfiles/sp/NiO-DFTU/DFTU/STRU")
    if structure is None:
        pytest.skip("NiO STRU not available")

    conventional = find_conventional(structure)
    # Conventional cell should have at least as many atoms
    assert conventional.natoms >= structure.natoms


def test_standardize_cell_default(tmp_path):
    """Standardize should return a standardized conventional cell."""
    from abacustools.data.structure import standardize_cell
    from abacustools.io.stru import AbacusSTRU

    structure = AbacusSTRU.read("/mnt/e/profsoftfiles/abacusfiles/sp/NiO-DFTU/DFTU/STRU")
    if structure is None:
        pytest.skip("NiO STRU not available")

    standardized = standardize_cell(structure)
    # Should have valid structure
    assert standardized.natoms > 0
    assert len(standardized.cell) == 3


def test_standardize_cell_to_primitive(tmp_path):
    """Standardize with to_primitive should reduce to primitive cell."""
    from abacustools.data.structure import standardize_cell
    from abacustools.io.stru import AbacusSTRU

    structure = AbacusSTRU.read("/mnt/e/profsoftfiles/abacusfiles/sp/NiO-DFTU/DFTU/STRU")
    if structure is None:
        pytest.skip("NiO STRU not available")

    standardized = standardize_cell(structure, to_primitive=True)
    # Should have fewer atoms than original
    assert standardized.natoms <= structure.natoms


def test_editstru_primitive_cli(tmp_path):
    """Test the primitive CLI command."""
    from abacustools.io.stru import AbacusSTRU

    structure = AbacusSTRU.read("/mnt/e/profsoftfiles/abacusfiles/sp/NiO-DFTU/DFTU/STRU")
    if structure is None:
        pytest.skip("NiO STRU not available")

    input_file = tmp_path / "STRU"
    output_file = tmp_path / "STRU_primitive"
    structure.write(str(input_file))

    from abacustools.main import main
    import sys

    sys.argv = [
        "abacustools", "file", "editstru", "primitive",
        str(input_file), "-o", str(output_file)
    ]
    result = main()
    assert result == 0
    assert output_file.exists()

    primitive = AbacusSTRU.read(str(output_file))
    assert primitive.natoms < structure.natoms


def test_editstru_conventional_cli(tmp_path):
    """Test the conventional CLI command."""
    from abacustools.io.stru import AbacusSTRU

    structure = AbacusSTRU.read("/mnt/e/profsoftfiles/abacusfiles/sp/NiO-DFTU/DFTU/STRU")
    if structure is None:
        pytest.skip("NiO STRU not available")

    input_file = tmp_path / "STRU"
    output_file = tmp_path / "STRU_conventional"
    structure.write(str(input_file))

    from abacustools.main import main
    import sys

    sys.argv = [
        "abacustools", "file", "editstru", "conventional",
        str(input_file), "-o", str(output_file)
    ]
    result = main()
    assert result == 0
    assert output_file.exists()


def test_editstru_standardize_cli(tmp_path):
    """Test the standardize CLI command."""
    from abacustools.io.stru import AbacusSTRU

    structure = AbacusSTRU.read("/mnt/e/profsoftfiles/abacusfiles/sp/NiO-DFTU/DFTU/STRU")
    if structure is None:
        pytest.skip("NiO STRU not available")

    input_file = tmp_path / "STRU"
    output_file = tmp_path / "STRU_standardized"
    structure.write(str(input_file))

    from abacustools.main import main
    import sys

    sys.argv = [
        "abacustools", "file", "editstru", "standardize",
        str(input_file), "-o", str(output_file)
    ]
    result = main()
    assert result == 0
    assert output_file.exists()
