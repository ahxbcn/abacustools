"""Render wavefunctions in the Molden file format.

The Molden format stores a Gaussian-orbital basis together with the molecular
orbital coefficients, which is what a wavefunction viewer such as Molden,
Multiwfn or Avogadro reads.  This module only turns an already assembled basis
and set of orbitals into text: expanding ABACUS numerical orbitals into
Gaussian contractions and locating the wavefunction files live in
:mod:`abacustools.data.molden`.

Coordinates of the ``[Atoms]`` block are written in Bohr (``[Atoms] AU``), the
``[Cell]`` block is written in Angstrom, and the ``[GTO]`` primitives are
written as ``exponent coefficient`` pairs, the ordering Molden, Q-Chem and the
ABACUS reference implementation use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from abacustools.core.constant import ANG_TO_BOHR


#: Shell-type labels of the Molden ``[GTO]`` block, indexed by ``l``.
_L_LABELS = ("s", "p", "d", "f", "g", "h", "i")


@dataclass(frozen=True)
class MoldenShell:
    """One contracted Gaussian shell centered on an atom.

    Attributes:
        l: Angular momentum quantum number.
        exponents: Gaussian exponents, in Bohr^-2.
        coefficients: Contraction coefficients that multiply normalized
            primitives, in the same order as ``exponents``.
    """

    l: int
    exponents: Sequence[float]
    coefficients: Sequence[float]


@dataclass(frozen=True)
class MoldenAtom:
    """One atom of the ``[Atoms]`` and ``[GTO]`` blocks.

    Attributes:
        element: Element symbol.
        atomic_number: Nuclear charge.
        coord: Cartesian coordinate in Angstrom.
        shells: Contracted Gaussian shells of this atom, in basis order.
    """

    element: str
    atomic_number: int
    coord: Sequence[float]
    shells: Sequence[MoldenShell] = field(default_factory=tuple)


@dataclass(frozen=True)
class MoldenOrbital:
    """One molecular orbital of the ``[MO]`` block.

    Attributes:
        energy: Orbital energy in Hartree.
        spin: ``"Alpha"`` or ``"Beta"``.
        occupation: Orbital occupation.
        coefficients: Coefficients on the Gaussian basis functions, in the
            order the shells are written.
    """

    energy: float
    spin: str
    occupation: float
    coefficients: Sequence[float]


def basis_size(atoms: Sequence[MoldenAtom]) -> int:
    """Return the number of Gaussian basis functions of the atoms."""

    return sum((2 * shell.l + 1) for atom in atoms for shell in atom.shells)


def _spherical_markers(atoms: Sequence[MoldenAtom]) -> list[str]:
    """Return the ``[5D7F]``/``[9G]`` pure-spherical declarations to write."""

    angular_momenta = {shell.l for atom in atoms for shell in atom.shells}
    markers: list[str] = []
    if 2 in angular_momenta:
        markers.append("[5D7F]" if 3 in angular_momenta else "[5D]")
    elif 3 in angular_momenta:
        markers.append("[7F]")
    if 4 in angular_momenta:
        markers.append("[9G]")
    return markers


def format_molden(
    cell: Sequence[Sequence[float]],
    atoms: Sequence[MoldenAtom],
    valences: dict[str, float],
    orbitals: Sequence[MoldenOrbital],
    *,
    atoms_unit: str = "bohr",
) -> str:
    """Return the text of a Molden file.

    Args:
        cell: Three cell vectors in Angstrom.
        atoms: Atoms with their Gaussian shells, in basis order.
        valences: Number of valence electrons per element for ``[Nval]``.
        orbitals: Molecular orbitals in the order they should be written.
        atoms_unit: Unit of the ``[Atoms]`` block, ``"bohr"`` (``AU``) or
            ``"angstrom"`` (``Angs``).

    Returns:
        The complete Molden file as a string.

    Raises:
        ValueError: If ``atoms_unit`` is unknown or an orbital coefficient
            does not match the basis size.
    """

    unit = atoms_unit.lower()
    if unit in ("bohr", "au", "a.u."):
        atom_header, scale = "[Atoms] AU", ANG_TO_BOHR
    elif unit in ("angstrom", "angs", "a"):
        atom_header, scale = "[Atoms] Angs", 1.0
    else:
        raise ValueError(f"unknown atoms unit: {atoms_unit!r}")

    size = basis_size(atoms)
    lines: list[str] = ["[Molden Format]"]

    vectors = list(cell) if cell is not None else []
    if len(vectors) == 3:
        lines.append("[Cell]")
        for vector in vectors:
            lines.append("  " + " ".join(f"{component:18.12f}" for component in vector))

    lines.append("[Nval]")
    for element, valence in valences.items():
        lines.append(f" {element} {valence:g}")

    lines.append(atom_header)
    for index, atom in enumerate(atoms, start=1):
        coordinates = " ".join(f"{component * scale:18.12f}" for component in atom.coord)
        lines.append(f" {atom.element:<2s} {index:4d} {atom.atomic_number:4d} {coordinates}")

    lines.append("[GTO]")
    for index, atom in enumerate(atoms, start=1):
        lines.append(f"{index} 0")
        for shell in atom.shells:
            label = _L_LABELS[shell.l] if shell.l < len(_L_LABELS) else "?"
            lines.append(f" {label} {len(shell.exponents):4d} 1.000000")
            for exponent, coefficient in zip(shell.exponents, shell.coefficients):
                lines.append(f"  {exponent:20.10E} {coefficient:20.10E}")
        lines.append("")

    lines.extend(_spherical_markers(atoms))

    lines.append("[MO]")
    for orbital in orbitals:
        if len(orbital.coefficients) != size:
            raise ValueError(
                f"orbital with {len(orbital.coefficients)} coefficients does not match "
                f"the {size} basis functions"
            )
        lines.append("Sym= A")
        lines.append(f"Ene= {orbital.energy:.12f}")
        lines.append(f"Spin= {orbital.spin}")
        lines.append(f"Occup= {orbital.occupation:.12f}")
        for index, coefficient in enumerate(orbital.coefficients, start=1):
            lines.append(f"{index:6d} {coefficient:20.12f}")

    return "\n".join(lines) + "\n"


def write_molden(
    path: str | Path,
    cell: Sequence[Sequence[float]],
    atoms: Sequence[MoldenAtom],
    valences: dict[str, float],
    orbitals: Sequence[MoldenOrbital],
    *,
    atoms_unit: str = "bohr",
) -> Path:
    """Write a Molden file and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        format_molden(cell, atoms, valences, orbitals, atoms_unit=atoms_unit),
        encoding="utf-8",
    )
    return output
