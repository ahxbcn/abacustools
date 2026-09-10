"""Structure editing recipes that build a new :class:`AbacusSTRU`.

The functions here compose the primitives of :mod:`abacustools.io.stru`, which
owns the structure model, into reusable edits.  They never modify the structure
they are given and never touch the file system, so a command layer can chain
them and write the result.
"""

from __future__ import annotations

import copy
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from abacustools.io.stru import AbacusSTRU


class StructureEditError(RuntimeError):
    """Raised when a structure edit cannot be applied."""


_DIRECTIONS = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}


def _direction_index(direction: Union[str, int]) -> int:
    """Map ``a``/``b``/``c`` or ``x``/``y``/``z`` to a lattice direction."""
    if isinstance(direction, bool):
        raise StructureEditError(f"unknown direction: {direction}")
    if isinstance(direction, int):
        if direction in (0, 1, 2):
            return direction
        raise StructureEditError(f"unknown direction: {direction}")
    key = str(direction).strip().lower()
    if key in _DIRECTIONS:
        return _DIRECTIONS[key]
    raise StructureEditError(
        f"unknown direction: {direction}; use a, b, c or x, y, z"
    )


def _copy(structure: AbacusSTRU) -> AbacusSTRU:
    """Return an independent copy so edits never touch the input."""
    return copy.deepcopy(structure)


def _validate_coordinate_range(
    coordinate_range: Sequence[float],
) -> Tuple[float, float]:
    """Validate an inclusive ``(minimum, maximum)`` window."""
    if len(coordinate_range) != 2:
        raise StructureEditError("a coordinate range needs a minimum and a maximum")
    try:
        start, end = float(coordinate_range[0]), float(coordinate_range[1])
    except (TypeError, ValueError) as error:
        raise StructureEditError("coordinate range values must be numbers") from error
    if not np.isfinite(start) or not np.isfinite(end):
        raise StructureEditError("coordinate range values must be finite")
    if start > end:
        raise StructureEditError(
            f"coordinate range minimum {start:g} is larger than maximum {end:g}"
        )
    return start, end


def select_indices(
    structure: AbacusSTRU,
    *,
    indices: Optional[Iterable[int]] = None,
    elements: Optional[Iterable[str]] = None,
    coordinate_range: Optional[Sequence[float]] = None,
    direction: Union[str, int] = "c",
    cartesian: bool = False,
) -> list[int]:
    """Return the zero-based indices of the atoms matching every filter.

    Args:
        structure: Structure to inspect.
        indices: Zero-based atom indices.
        elements: Element symbols.
        coordinate_range: Inclusive ``(minimum, maximum)`` window.
        direction: Direction of the window, ``a``/``b``/``c`` or ``x``/``y``/``z``.
        cartesian: Interpret the window in Cartesian instead of fractional
            coordinates.

    Returns:
        list[int]: Matching atom indices in ascending order.
    """
    if indices is None and elements is None and coordinate_range is None:
        raise StructureEditError(
            "selecting atoms needs atom indices, elements or a coordinate range"
        )
    selected = set(range(structure.natoms))
    if indices is not None:
        wanted = set()
        for index in indices:
            try:
                value = int(index)
            except (TypeError, ValueError) as error:
                raise StructureEditError(f"invalid atom index: {index}") from error
            if value < 0 or value >= structure.natoms:
                raise StructureEditError(
                    f"atom index {value + 1} is outside 1..{structure.natoms}"
                )
            wanted.add(value)
        selected &= wanted
    if elements is not None:
        wanted = {str(element).capitalize() for element in elements}
        present = set(structure.elements)
        unknown = sorted(wanted - present)
        if unknown:
            raise StructureEditError(
                "element not present in the structure: " + ", ".join(unknown)
            )
        selected &= {
            index
            for index, atom in enumerate(structure.atoms)
            if atom.element in wanted
        }
    if coordinate_range is not None:
        start, end = _validate_coordinate_range(coordinate_range)
        axis = _direction_index(direction)
        values = np.asarray(
            structure.coords if cartesian else structure.coords_direct, dtype=float
        )[:, axis]
        selected &= {
            index for index, value in enumerate(values) if start <= value <= end
        }
    return sorted(selected)


def select_atoms(
    structure: AbacusSTRU,
    *,
    indices: Optional[Iterable[int]] = None,
    elements: Optional[Iterable[str]] = None,
    coordinate_range: Optional[Sequence[float]] = None,
    direction: Union[str, int] = "c",
    cartesian: bool = False,
    remove: bool = False,
) -> AbacusSTRU:
    """Return a new structure holding, or dropping, the selected atoms.

    Args:
        structure: Structure to copy from.
        indices: Zero-based atom indices.
        elements: Element symbols.
        coordinate_range: Inclusive ``(minimum, maximum)`` window.
        direction: Direction of the window.
        cartesian: Interpret the window in Cartesian coordinates.
        remove: Drop the selection instead of keeping it.

    Returns:
        AbacusSTRU: A new structure with the remaining atoms.
    """
    selected = set(
        select_indices(
            structure,
            indices=indices,
            elements=elements,
            coordinate_range=coordinate_range,
            direction=direction,
            cartesian=cartesian,
        )
    )
    kept = [
        index
        for index in range(structure.natoms)
        if (index not in selected if remove else index in selected)
    ]
    if not kept:
        raise StructureEditError("the selection would leave no atoms behind")
    edited = _copy(structure)
    edited._atoms = [edited._atoms[index] for index in kept]
    return edited


def fix_atoms(
    structure: AbacusSTRU,
    *,
    indices: Optional[Iterable[int]] = None,
    elements: Optional[Iterable[str]] = None,
    coordinate_range: Optional[Sequence[float]] = None,
    direction: Union[str, int] = "c",
    cartesian: bool = False,
    move: Sequence[bool] = (False, False, False),
    free_others: bool = False,
) -> AbacusSTRU:
    """Return a copy with the movement constraints of the selected atoms set.

    Args:
        structure: Structure to copy from.
        indices: Zero-based atom indices.
        elements: Element symbols.
        coordinate_range: Inclusive ``(minimum, maximum)`` window.
        direction: Direction of the window.
        cartesian: Interpret the window in Cartesian coordinates.
        move: Movement allowed along each lattice direction; ``(False, False,
            False)`` fixes the selected atoms completely.
        free_others: Allow every unselected atom to move.

    Returns:
        AbacusSTRU: A new structure with updated ``move`` flags.
    """
    if len(move) != 3:
        raise StructureEditError("move must contain three flags")
    selected = set(
        select_indices(
            structure,
            indices=indices,
            elements=elements,
            coordinate_range=coordinate_range,
            direction=direction,
            cartesian=cartesian,
        )
    )
    if not selected:
        raise StructureEditError("no atom matches the fix selection")
    flags = tuple(bool(flag) for flag in move)
    edited = _copy(structure)
    for index, atom in enumerate(edited.atoms):
        if index in selected:
            atom.move = flags
        elif free_others:
            atom.move = (True, True, True)
    return edited


def with_vacuum(
    structure: AbacusSTRU,
    thickness: float,
    *,
    direction: Union[str, int] = "c",
    center: bool = False,
) -> AbacusSTRU:
    """Return a copy with vacuum added along one lattice direction.

    Args:
        structure: Structure to copy from.
        thickness: Vacuum to add in Angstrom.
        direction: Lattice direction to extend.
        center: Shift the atoms by half the added vacuum, so the extra space is
            split between both sides of the cell instead of staying on one side.

    Returns:
        AbacusSTRU: A new structure with the extended cell.
    """
    try:
        value = float(thickness)
    except (TypeError, ValueError) as error:
        raise StructureEditError("vacuum thickness must be a number") from error
    if not np.isfinite(value) or value <= 0:
        raise StructureEditError("vacuum thickness must be a positive number")
    axis = _direction_index(direction)
    cell = np.asarray(structure.cell, dtype=float)
    length = float(np.linalg.norm(cell[axis]))
    if length <= 0:
        raise StructureEditError("the lattice vector to extend is degenerate")

    edited = _copy(structure)
    new_cell = cell.copy()
    new_cell[axis] = cell[axis] * (length + value) / length
    edited.cell = new_cell.tolist()
    if center:
        direct = np.asarray(edited.coords_direct, dtype=float)
        direct[:, axis] += 0.5 * value / (length + value)
        edited.coords_direct = direct.tolist()
    return edited


def make_supercell(structure: AbacusSTRU, repeats: Sequence[int]) -> AbacusSTRU:
    """Return a diagonal supercell of the structure.

    Args:
        structure: Structure to replicate.
        repeats: Positive repetitions along the three lattice vectors.

    Returns:
        AbacusSTRU: The supercell, with every atom attribute preserved.
    """
    if len(repeats) != 3:
        raise StructureEditError("a supercell needs three repetition factors")
    factors = []
    for repeat in repeats:
        try:
            number = float(repeat)
        except (TypeError, ValueError) as error:
            raise StructureEditError(f"invalid repetition factor: {repeat}") from error
        if not np.isfinite(number) or number != int(number):
            raise StructureEditError(
                f"repetition factors must be whole numbers, got {repeat}"
            )
        factor = int(number)
        if factor <= 0:
            raise StructureEditError(
                f"repetition factors must be positive, got {factor}"
            )
        factors.append(factor)
    return structure.supercell(factors)


__all__ = [
    "StructureEditError",
    "fix_atoms",
    "make_supercell",
    "select_atoms",
    "select_indices",
    "with_vacuum",
]
