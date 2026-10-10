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
_COORDINATE_MODES = {
    "direct": "direct",
    "fractional": "direct",
    "cartesian": "cartesian",
    "cart": "cartesian",
}

def _spglib_cell(structure: AbacusSTRU):
    """Return the ``(lattice, positions, numbers)`` triple spglib expects."""
    from ase.data import atomic_numbers

    numbers = []
    for atom in structure.atoms:
        element = atom.element or atom.label
        key = str(element).strip().capitalize()
        if key not in atomic_numbers:
            raise StructureEditError(f"unknown element: {element}")
        numbers.append(int(atomic_numbers[key]))
    return (
        np.asarray(structure.cell, dtype=float),
        np.asarray(structure.coords_direct, dtype=float),
        numbers,
    )


def _spglib_to_stru(
    structure: AbacusSTRU,
    cell: np.ndarray,
    positions: np.ndarray,
    numbers: list[int],
) -> AbacusSTRU:
    """Convert spglib output back to an AbacusSTRU, preserving attributes."""
    from ase.data import chemical_symbols, atomic_numbers

    # Build a mapping from atomic number to template atoms
    templates: dict[int, list] = {}
    for atom in structure.atoms:
        element = atom.element or atom.label
        key = str(element).strip().capitalize()
        num = atomic_numbers.get(key, 0)
        templates.setdefault(num, []).append(atom)

    # Track which template atoms have been used for each element
    template_indices: dict[int, int] = {num: 0 for num in templates}

    atoms = []
    for num in numbers:
        symbol = chemical_symbols[num]
        template_list = templates.get(num, [])
        if not template_list:
            raise StructureEditError(
                f"spglib returned element {symbol} not found in original structure"
            )
        idx = template_indices[num] % len(template_list)
        template = template_list[idx]
        template_indices[num] += 1

        atom = copy.deepcopy(template)
        atom.element = symbol
        atoms.append(atom)

    # Set fractional coordinates from spglib output
    positions_list = positions.tolist()
    for atom, pos in zip(atoms, positions_list):
        atom.coord = tuple(pos)

    metadata = copy.deepcopy(structure.metadata)
    metadata["atom_type"] = "direct"

    return AbacusSTRU(cell=cell.tolist(), atoms=atoms, metadata=metadata)


def find_primitive(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
) -> AbacusSTRU:
    """Return the primitive cell of the structure.

    Uses spglib to find the smallest cell that preserves the crystallographic
    symmetry. The primitive cell has the minimum number of atoms.

    Args:
        structure: Structure to reduce.
        symprec: Symmetry tolerance in Angstrom.
        angle_tolerance: Angle tolerance in degrees.

    Returns:
        AbacusSTRU: The primitive cell.

    Raises:
        StructureEditError: If spglib fails or the structure has no symmetry.
    """
    try:
        import spglib
    except ImportError as error:
        raise StructureEditError("spglib is not installed") from error

    cell = _spglib_cell(structure)
    primitive = spglib.find_primitive(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    if primitive is None:
        raise StructureEditError("spglib could not find a primitive cell")

    lattice, positions, numbers = primitive
    return _spglib_to_stru(structure, np.array(lattice), np.array(positions), list(numbers))


def standardize_cell(
    structure: AbacusSTRU,
    *,
    to_primitive: bool = False,
    no_idealize: bool = False,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
) -> AbacusSTRU:
    """Return a standardized version of the structure.

    Uses spglib to standardize the cell according to the International Tables
    for Crystallography. The standardized cell has a unique orientation and
    origin.

    Args:
        structure: Structure to standardize.
        to_primitive: If True, also reduce to primitive cell.
        no_idealize: If True, do not idealize the cell (keep small distortions).
        symprec: Symmetry tolerance in Angstrom.
        angle_tolerance: Angle tolerance in degrees.

    Returns:
        AbacusSTRU: The standardized cell.

    Raises:
        StructureEditError: If spglib fails.
    """
    try:
        import spglib
    except ImportError as error:
        raise StructureEditError("spglib is not installed") from error

    cell = _spglib_cell(structure)
    standardized = spglib.standardize_cell(
        cell,
        to_primitive=to_primitive,
        no_idealize=no_idealize,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if standardized is None:
        raise StructureEditError("spglib could not standardize the cell")

    lattice, positions, numbers = standardized
    return _spglib_to_stru(structure, np.array(lattice), np.array(positions), list(numbers))


def find_conventional(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
) -> AbacusSTRU:
    """Return the conventional cell of the structure.

    Uses spglib to find the conventional (standard) cell, which follows the
    crystallographic conventions for the space group. The conventional cell
    may have more atoms than the primitive cell but has higher symmetry.

    Args:
        structure: Structure to convert.
        symprec: Symmetry tolerance in Angstrom.
        angle_tolerance: Angle tolerance in degrees.

    Returns:
        AbacusSTRU: The conventional cell.

    Raises:
        StructureEditError: If spglib fails.
    """
    try:
        import spglib
    except ImportError as error:
        raise StructureEditError("spglib is not installed") from error

    cell = _spglib_cell(structure)
    conventional = spglib.refine_cell(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    if conventional is None:
        raise StructureEditError("spglib could not find a conventional cell")

    lattice, positions, numbers = conventional
    return _spglib_to_stru(structure, np.array(lattice), np.array(positions), list(numbers))




def _matrix_square_root(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric positive semi-definite square root of a matrix."""
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def _idealized_lattice(cell: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Return the cell closest to ``cell`` whose metric obeys the rotations.

    The metric tensor of a lattice that obeys a symmetry operation ``R``
    satisfies ``R^T G R = G``, so the average of ``R^T G R`` over the space
    group is its invariant part. A pure strain then maps the input cell onto a
    cell with that ideal metric, which keeps the orientation of the input.
    """
    metric = cell @ cell.T
    invariant = np.mean([rotation.T @ metric @ rotation for rotation in rotations], axis=0)
    root = _matrix_square_root(metric)
    inverse_root = np.linalg.inv(root)
    strain = inverse_root @ _matrix_square_root(root @ invariant @ root) @ inverse_root
    return strain @ cell


def _symmetrized_positions(
    fractional: np.ndarray,
    metric: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    symprec: float,
) -> np.ndarray:
    """Average every position over the images its symmetry partners give it.

    For an operation ``g`` the atom nearest to ``g(x_i)`` is the image of
    ``x_i`` under ``g``, so ``g^-1`` of that atom is another estimate of
    ``x_i``. Averaging those estimates removes the noise along the orbit.
    """
    count = len(fractional)
    accumulated = np.zeros_like(fractional)
    weights = np.zeros(count)
    for rotation, translation in zip(rotations, translations):
        images = (fractional @ rotation.T + translation) % 1.0
        delta = images[:, None, :] - fractional[None, :, :]
        delta -= np.rint(delta)
        distances = np.einsum("ijk,kl,ijl->ij", delta, metric, delta)
        nearest = np.argmin(distances, axis=1)
        matched = distances[np.arange(count), nearest] <= symprec**2
        inverse = np.rint(np.linalg.inv(rotation)).astype(int)
        back = (fractional[nearest] - translation) @ inverse.T
        back -= np.rint(back - fractional)
        accumulated[matched] += back[matched]
        weights[matched] += 1
    return (accumulated / weights[:, None]) % 1.0


def symmetrize_structure(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
    keep_cell: bool = False,
) -> AbacusSTRU:
    """Return a copy snapped onto the symmetry found in its own cell.

    spglib determines the space group of the structure. The atomic positions
    are then averaged with their symmetry images and the lattice metric is
    projected onto the metric that the space group requires, so small numerical
    errors disappear and the symmetry of the result is exact. The cell setting,
    the number and the order of the atoms, and every atom attribute are
    preserved; use :func:`standardize_cell` or :func:`find_conventional` to
    rewrite the cell into the standard setting instead.

    Args:
        structure: Structure to clean.
        symprec: Distance tolerance in Angstrom, with the meaning it has in
            spglib: deviations have to stay below it to count as noise.
        angle_tolerance: Angle tolerance in degrees, passed to spglib.
        keep_cell: Idealize the positions only and leave the lattice vectors
            untouched, which keeps the symmetry approximate in the cell.

    Returns:
        AbacusSTRU: A new structure whose symmetry is exact.

    Raises:
        StructureEditError: If the structure has no periodic cell, spglib is
            not installed, or the symmetry analysis fails.
    """
    try:
        import spglib
    except ImportError as error:
        raise StructureEditError("spglib is not installed") from error

    tolerance = _finite_float(symprec, "symprec")
    angle = _finite_float(angle_tolerance, "angle tolerance", allow_zero=True)
    cell = np.asarray(structure.cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise StructureEditError("symmetrize needs a finite three-dimensional cell")
    if abs(float(np.linalg.det(cell))) < 1e-12:
        raise StructureEditError("symmetrize needs a periodic cell with a volume")

    lattice, positions, numbers = _spglib_cell(structure)
    dataset = spglib.get_symmetry_dataset(
        (lattice, positions, numbers),
        symprec=tolerance,
        angle_tolerance=angle,
    )
    if dataset is None:
        raise StructureEditError("spglib could not determine the symmetry of the structure")

    rotations = np.asarray(dataset.rotations, dtype=int)
    translations = np.asarray(dataset.translations, dtype=float)
    fractional = np.asarray(positions, dtype=float) % 1.0
    idealized = _symmetrized_positions(
        fractional, cell @ cell.T, rotations, translations, tolerance
    )

    edited = _copy(structure)
    if not keep_cell:
        edited.cell = _idealized_lattice(cell, rotations).tolist()
    edited.coords_direct = idealized.tolist()
    return edited


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


def set_coordinate_mode(structure: AbacusSTRU, mode: str) -> AbacusSTRU:
    """Return a copy that writes its coordinates in the requested form.

    The atoms keep their positions; only the representation written to the
    structure file changes, so ``direct``/``fractional`` produces an
    ``ATOMIC_POSITIONS Direct`` block and ``cartesian`` the Cartesian one.

    Args:
        structure: Structure to copy from.
        mode: ``direct``/``fractional`` or ``cartesian``.

    Returns:
        AbacusSTRU: A new structure carrying the requested coordinate mode.
    """
    key = str(mode).strip().lower()
    if key not in _COORDINATE_MODES:
        raise StructureEditError(
            f"unknown coordinate mode: {mode}; use direct (fractional) or cartesian"
        )
    edited = _copy(structure)
    edited.metadata["atom_type"] = _COORDINATE_MODES[key]
    return edited


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




def generate_all_slabs(
    structure: AbacusSTRU,
    miller_indices: Sequence[int],
    *,
    min_slab_size: float = 3.0,
    min_vacuum_size: float = 10.0,
    center_slab: bool = True,
    in_unit_planes: bool = False,
    primitive: bool = False,
    max_normal_search: Optional[int] = None,
    symmetrize: bool = False,
    repair: bool = False,
    tol: float = 0.1,
    ftol: float = 0.1,
    max_broken_bonds: int = 0,
    filter_out_sym_slabs: bool = True,
) -> list[AbacusSTRU]:
    """Generate all possible surface terminations for given Miller indices.

    Uses pymatgen's SlabGenerator to find all symmetrically distinct surface
    terminations. This is particularly useful for polar surfaces or surfaces
    with multiple possible terminations (e.g., TiO2(110) can have Ti or O
    terminations).

    Args:
        structure: Bulk structure to cut surfaces from.
        miller_indices: Three Miller indices of the surface, such as ``(1, 1, 0)``.
        min_slab_size: Minimum slab thickness in number of atomic layers or
            Angstroms (depending on in_unit_planes). Default: 3.0.
        min_vacuum_size: Minimum vacuum thickness in Angstrom. Default: 10.0.
        center_slab: Whether to center the slab in the cell. Default: True.
        in_unit_planes: If True, min_slab_size is in number of unit cells;
            if False, it's in Angstroms. Default: False.
        primitive: Whether to reduce the slab to primitive cell. Default: False.
        max_normal_search: Maximum index to search for the surface normal.
            Default: None.
        symmetrize: Whether to symmetrize the slab. Default: False.
        repair: Whether to repair the slab structure. Default: False.
        tol: Tolerance for comparing sites. Default: 0.1.
        ftol: Fractional tolerance for comparing sites. Default: 0.1.
        max_broken_bonds: Maximum number of broken bonds allowed. Default: 0.
        filter_out_sym_slabs: Whether to filter out symmetric slabs. Default: True.

    Returns:
        list[AbacusSTRU]: List of all possible slab terminations, each as an
        AbacusSTRU with pseudopotential and orbital data preserved.

    Raises:
        StructureEditError: If pymatgen fails or the structure is invalid.

    Example:
        >>> # Generate all (110) terminations of TiO2
        >>> slabs = generate_all_slabs(structure, [1, 1, 0], min_slab_size=5)
        >>> for i, slab in enumerate(slabs):
        ...     slab.write(f"slab_{i}.STRU")
    """
    if len(miller_indices) != 3:
        raise StructureEditError("miller_indices needs three integers")
    miller = tuple(int(value) for value in miller_indices)
    if all(value == 0 for value in miller):
        raise StructureEditError("miller_indices must not all be zero")

    try:
        from pymatgen.core.surface import SlabGenerator
    except ImportError as error:
        raise StructureEditError("pymatgen is not installed") from error

    # Convert to pymatgen Structure
    try:
        pmg_structure = structure.to("pymatgen")
    except Exception as error:
        raise StructureEditError(
            f"failed to convert structure to pymatgen format: {error}"
        ) from error

    # Generate all possible slabs
    try:
        slabgen = SlabGenerator(
            initial_structure=pmg_structure,
            miller_index=miller,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=center_slab,
            in_unit_planes=in_unit_planes,
            primitive=primitive,
            max_normal_search=max_normal_search,
        )
        
        # Get all symmetrically distinct slabs
        all_slabs = slabgen.get_slabs(
            symmetrize=symmetrize,
            repair=repair,
            tol=tol,
            ftol=ftol,
            max_broken_bonds=max_broken_bonds,
            filter_out_sym_slabs=filter_out_sym_slabs,
        )
    except Exception as error:
        raise StructureEditError(
            f"pymatgen failed to generate slabs: {error}"
        ) from error

    if not all_slabs:
        raise StructureEditError(
            f"no slabs found for Miller indices {miller}"
        )

    # Convert each slab back to AbacusSTRU
    result = []
    for slab in all_slabs:
        metadata = dict(structure.metadata)
        metadata["atom_type"] = "cartesian"
        metadata.setdefault("lattice_constant", ANG_TO_BOHR)
        
        edited = AbacusSTRU.from_ase(slab.to_ase_atoms(), metadata=metadata)
        _restore_element_attributes(edited, structure)
        edited.sort()
        result.append(edited)

    return result


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
    "find_conventional",
    "find_primitive",
    "fix_atoms",
    "fix_slab_bottom",
    "generate_all_slabs",
    "make_supercell",
    "standardize_cell",
    "symmetrize_structure",
    "set_coordinate_mode",
    "select_atoms",
    "select_indices",
    "with_vacuum",
]
