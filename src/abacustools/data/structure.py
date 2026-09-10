"""Structure editing recipes that build a new :class:`AbacusSTRU`.

The functions here compose the primitives of :mod:`abacustools.io.stru`, which
owns the structure model, into reusable edits.  They never modify the structure
they are given and never touch the file system, so a command layer can chain
them and write the result.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from abacustools.core.constant import ANG_TO_BOHR
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


def _positive_int(value: Any, name: str) -> int:
    """Validate a positive whole number."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise StructureEditError(f"{name} must be a whole number, got {value!r}") from error
    if not np.isfinite(number) or number != int(number):
        raise StructureEditError(f"{name} must be a whole number, got {value!r}")
    if int(number) <= 0:
        raise StructureEditError(f"{name} must be positive, got {value!r}")
    return int(number)


def _finite_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise StructureEditError(f"{name} must be a number, got {value!r}") from error
    if not np.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise StructureEditError(f"{name} must be a {qualifier} number, got {value!r}")
    return number


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
    factors = [_positive_int(repeat, "repetition factor") for repeat in repeats]
    return structure.supercell(factors)


def _restore_element_attributes(edited: AbacusSTRU, source: AbacusSTRU) -> None:
    """Copy per-element attributes that an ASE round trip may have dropped.

    ``ase.build.make_supercell`` rebuilds the atoms and loses ``Atoms.info``,
    which is where the pseudopotential and orbital file names travel, and ASE
    fills in zero magnetic moments and velocities that the source never had.
    """
    templates: dict[Optional[str], Any] = {}
    uniform_mag: dict[Optional[str], Optional[Any]] = {}
    has_moment: dict[Optional[str], bool] = {}
    for atom in source.atoms:
        templates.setdefault(atom.element, atom)
        siblings = [other for other in source.atoms if other.element == atom.element]
        values = {other.mag for other in siblings}
        uniform_mag[atom.element] = values.pop() if len(values) == 1 else None
        has_moment[atom.element] = any(
            other.mag is not None or other.type_mag for other in siblings
        )
    keeps_velocity = any(atom.velocity is not None for atom in source.atoms)
    for atom in edited.atoms:
        template = templates.get(atom.element)
        if template is None:
            continue
        atom.pp = atom.pp or template.pp
        atom.orb = atom.orb or template.orb
        atom.paw = atom.paw or template.paw
        if not has_moment.get(atom.element, False):
            atom.mag = None
        elif atom.mag is None and uniform_mag.get(atom.element) is not None:
            atom.mag = uniform_mag[atom.element]
        if not keeps_velocity:
            atom.velocity = None


def build_slab(
    structure: AbacusSTRU,
    *,
    miller_indices: Sequence[int] = (1, 0, 0),
    layers: int = 3,
    surface_supercell: Sequence[int] = (1, 1),
    vacuum: float = 15.0,
    vacuum_direction: str = "c",
    fix_fraction: Optional[float] = None,
) -> AbacusSTRU:
    """Cut a surface slab out of a bulk structure.

    The surface is created along the third lattice vector and can be moved to
    another direction afterwards.  Movement constraints are not carried over,
    because the cut removes atoms: apply them to the slab with :func:`fix_atoms`.

    Args:
        structure: Bulk structure to cut.
        miller_indices: Three Miller indices of the surface, such as ``(1, 0, 0)``.
        layers: Number of repeating units along the surface normal.
        surface_supercell: Repetitions along the two in-plane directions.
        vacuum: Empty space between the slab and its periodic image, in Angstrom.
        vacuum_direction: Lattice direction that receives the vacuum, ``a``,
            ``b`` or ``c``.
        fix_fraction: Fraction of the slab thickness to fix, counted from the
            lowest atom along ``vacuum_direction``; ``0.5`` fixes the bottom
            half and frees the rest, while ``None`` leaves every atom free.

    Returns:
        AbacusSTRU: The slab, with pseudopotential, orbital and magnetic data
        restored per element.

    Raises:
        StructureEditError: If a parameter is invalid or the structure cannot be
            passed through ASE.
    """
    if len(miller_indices) != 3:
        raise StructureEditError("miller_indices needs three integers")
    miller = tuple(int(value) for value in miller_indices)
    if all(value == 0 for value in miller):
        raise StructureEditError("miller_indices must not all be zero")
    layer_count = _positive_int(layers, "layers")
    if len(surface_supercell) != 2:
        raise StructureEditError("surface_supercell needs two factors")
    repeats = tuple(
        _positive_int(value, "surface supercell factor") for value in surface_supercell
    )
    thickness = _finite_float(vacuum, "vacuum", allow_zero=True)
    direction = str(vacuum_direction).strip().lower()
    if direction not in {"a", "b", "c"}:
        raise StructureEditError(
            f"unknown vacuum direction: {vacuum_direction}; use a, b or c"
        )
    empty_labels = sorted(
        {atom.label for atom in structure.atoms if atom.label and "empty" in atom.label}
    )
    if empty_labels:
        raise StructureEditError(
            "the structure contains empty atoms (" + ", ".join(empty_labels) + "); "
            "ASE cannot represent them, so cut the slab from a structure without them"
        )

    from ase.build import make_supercell as ase_make_supercell
    from ase.build import surface as ase_surface

    atoms = structure.to("ase")
    slab = ase_surface(
        atoms, miller, layer_count, vacuum=thickness / 2, periodic=True
    )
    if repeats != (1, 1):
        slab = ase_make_supercell(
            slab, [[repeats[0], 0, 0], [0, repeats[1], 0], [0, 0, 1]]
        )
    # ase.build.surface labels the atoms with layer tags, which ABACUS has no
    # field for; dropping them keeps the conversion warning meaningful.
    slab.arrays.pop("tags", None)

    metadata = dict(structure.metadata)
    metadata["atom_type"] = "cartesian"
    metadata.setdefault("lattice_constant", ANG_TO_BOHR)
    edited = AbacusSTRU.from_ase(slab, metadata=metadata)
    _restore_element_attributes(edited, structure)
    edited.sort()
    if direction == "a":
        edited.permute_lat_vec(mode="cab", rotate_cart_coord=True)
    elif direction == "b":
        edited.permute_lat_vec(mode="bca", rotate_cart_coord=True)
    if fix_fraction is not None:
        edited = fix_slab_bottom(edited, direction=direction, fraction=fix_fraction)
    return edited


def fix_slab_bottom(
    structure: AbacusSTRU,
    *,
    direction: Union[str, int] = "c",
    fraction: float = 0.5,
) -> AbacusSTRU:
    """Fix the lower part of a slab and free the remaining atoms.

    The fixed window spans the atoms' own extent along ``direction``: it starts
    at the lowest atom and reaches ``fraction`` of the slab thickness above it,
    so it does not depend on where the slab sits inside the cell.  With the
    default fraction this is the bottom half of the slab.

    Args:
        structure: Slab to constrain.
        direction: Direction of the slab normal, ``a``/``b``/``c``.
        fraction: Part of the slab thickness to fix, between 0 and 1.

    Returns:
        AbacusSTRU: A new structure with updated ``move`` flags.
    """
    axis = _direction_index(direction)
    try:
        value = float(fraction)
    except (TypeError, ValueError) as error:
        raise StructureEditError(f"invalid fix fraction: {fraction!r}") from error
    if not np.isfinite(value) or value < 0 or value > 1:
        raise StructureEditError(
            f"the fix fraction must be between 0 and 1, got {fraction!r}"
        )
    direct = np.asarray(structure.coords_direct, dtype=float)[:, axis]
    lower = float(direct.min())
    thickness = float(direct.max()) - lower
    cutoff = lower + value * thickness
    return fix_atoms(
        structure,
        coordinate_range=(lower, cutoff),
        direction=axis,
        cartesian=False,
        move=(False, False, False),
        free_others=True,
    )


__all__ = [
    "StructureEditError",
    "build_slab",
    "fix_atoms",
    "fix_slab_bottom",
    "make_supercell",
    "select_atoms",
    "select_indices",
    "with_vacuum",
]
