"""Tests for the ``file interface`` command and the interface builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abacustools.data.interface import (
    InterfaceError,
    build_interface,
    interface_matches,
)
from abacustools.io.stru import AbacusATOM, AbacusSTRU
from abacustools.main import main


def _cubic(a: float, symbol: str) -> AbacusSTRU:
    """Return a one-atom cubic structure with resources."""
    return AbacusSTRU(
        cell=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        atoms=[
            AbacusATOM(
                label=symbol, element=symbol, coord=(0.0, 0.0, 0.0),
                pp=f"{symbol}.upf", orb=f"{symbol}.orb",
            )
        ],
        metadata={"atom_type": "cartesian"},
    )


def _write(tmp_path: Path, name: str, structure: AbacusSTRU) -> Path:
    path = tmp_path / name
    assert structure.write(str(path))
    return path


def test_interface_matches_finds_the_one_cell_match() -> None:
    candidates = interface_matches(_cubic(4.1, "Ge"), _cubic(4.0, "Si"), max_area=30)

    assert candidates
    best = candidates[0]
    assert best.film_cells == 1
    assert best.substrate_cells == 1
    assert best.length_strain[0] == pytest.approx(0.025, rel=1e-3)
    assert best.angle_mismatch == pytest.approx(0.0, abs=1e-6)


def test_build_interface_stacks_the_film_on_the_substrate() -> None:
    result = build_interface(
        _cubic(4.1, "Ge"), _cubic(4.0, "Si"),
        max_area=30, gap=2.0, vacuum=15.0, film_thickness=2, substrate_thickness=2,
    )

    assert result.structure.natoms == 4
    assert result.terminations == 1
    assert result.termination_index == 0
    assert result.candidate.index == 0
    # The substrate sits at the bottom and the film is stacked on top of it.
    assert result.structure.elements == ["Si", "Si", "Ge", "Ge"]
    parameters = result.structure.get_cell_param()
    # 2 layers of each parent plus the gap and the vacuum.
    assert parameters[:2] == pytest.approx([4.0, 4.0], abs=1e-3)
    assert parameters[2] > 15.0 + 2.0
    assert result.resources["Ge"][0] == "Ge.upf"
    assert result.resources["Ge"][2] == "reused from the film structure"
    assert result.resources["Si"][1] == "Si.orb"
    assert result.resources["Si"][2] == "reused from the substrate structure"
    for atom in result.structure.atoms:
        assert atom.pp == f"{atom.element}.upf"
        assert atom.orb == f"{atom.element}.orb"


def test_build_interface_rejects_an_unreachable_atom_count() -> None:
    with pytest.raises(InterfaceError, match="more than 1 atoms"):
        build_interface(_cubic(4.1, "Ge"), _cubic(4.0, "Si"), max_area=30, max_atoms=1)


def test_build_interface_rejects_an_unknown_termination() -> None:
    with pytest.raises(InterfaceError, match="termination 5"):
        build_interface(_cubic(4.1, "Ge"), _cubic(4.0, "Si"), max_area=30, termination=5)


def test_build_interface_reports_a_missing_lattice_match() -> None:
    film = _cubic(4.1, "Ge")
    substrate = _cubic(4.0, "Si")

    assert interface_matches(film, substrate, max_strain=1e-5, max_area=30) == []
    with pytest.raises(InterfaceError, match="no lattice match"):
        build_interface(film, substrate, max_strain=1e-5, max_area=30)


def test_file_interface_writes_and_reports(tmp_path: Path, capsys) -> None:
    film = _write(tmp_path, "Ge.STRU", _cubic(4.1, "Ge"))
    substrate = _write(tmp_path, "Si.STRU", _cubic(4.0, "Si"))
    output = tmp_path / "HET.STRU"

    assert main([
        "file", "interface", str(film), str(substrate),
        "-o", str(output), "--max-area", "30", "--film-thickness", "2",
        "--substrate-thickness", "2", "--gap", "2.5", "--vacuum", "12", "--json",
    ]) == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout[stdout.index("{"):])

    assert payload["natoms"] == 4
    assert payload["matches"] >= 1
    assert payload["film_miller"] == [0, 0, 1]
    assert payload["substrate_miller"] == [0, 0, 1]
    assert payload["gap"] == 2.5
    assert payload["vacuum"] == 12.0
    assert payload["film_formula"] == "Ge1"
    assert payload["substrate_formula"] == "Si1"
    assert payload["resources"]["Ge"]["reason"] == "reused from the film structure"
    assert payload["resources"]["Si"]["reason"] == "reused from the substrate structure"

    interface = AbacusSTRU.read(str(output))
    assert interface is not None
    assert interface.natoms == 4
    assert interface.pps[:2] == ["Si.upf", "Si.upf"]
    assert interface.orbs[2:] == ["Ge.orb", "Ge.orb"]


def test_file_interface_lists_matches_without_an_output(tmp_path: Path, capsys) -> None:
    film = _write(tmp_path, "Ge.STRU", _cubic(4.1, "Ge"))
    substrate = _write(tmp_path, "Si.STRU", _cubic(4.0, "Si"))

    assert main([
        "file", "interface", str(film), str(substrate), "--list", "--max-area", "30",
    ]) == 0
    stdout = capsys.readouterr().out
    assert "matches:" in stdout
    assert "index" in stdout


def test_file_interface_refuses_to_overwrite(tmp_path: Path) -> None:
    film = _write(tmp_path, "Ge.STRU", _cubic(4.1, "Ge"))
    substrate = _write(tmp_path, "Si.STRU", _cubic(4.0, "Si"))
    output = tmp_path / "HET.STRU"
    output.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists"):
        main([
            "file", "interface", str(film), str(substrate), "-o", str(output),
            "--max-area", "30",
        ])
    assert output.read_text(encoding="utf-8") == "keep me"

    assert main([
        "file", "interface", str(film), str(substrate), "-o", str(output),
        "--max-area", "30", "--override",
    ]) == 0
