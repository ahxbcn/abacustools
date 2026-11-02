"""Construction and reduction of commensurate magnetic cells.

For one propagation vector ``k`` a magnetic structure is invariant under every
lattice translation ``T`` that satisfies ``exp(2*pi*i k.T) = 1``.  Those
translations form a sublattice of the crystal lattice, the magnetic translation
group, and its index is the order of the phase factor.  Written with the
smallest common denominator,

    k = (p1/n, p2/n, p3/n),   n = lcm(q1, q2, q3)

the magnetic cell is ``n`` times the input cell.  Its volume is fixed by ``k``;
the basis is not, so any integer matrix with determinant ``n`` whose rows are in
the kernel is a valid choice.

``build_magnetic_supercell`` builds one such cell and assigns the collinear
moment ``m cos(2*pi*k.r + phi)``.  ``magnetic_primitive`` goes the other way:
given a structure whose atoms already carry moments, it uses the magnetic
symmetry to reduce it to the primitive cell of the magnetic translation group.
"""

from __future__ import annotations

import copy
import math
from fractions import Fraction
from itertools import product
from typing import Any, Optional, Sequence

import numpy as np

from abacustools.data.structure import StructureEditError
from abacustools.io.stru import AbacusSTRU


def rationalize_propagation_vector(
    k: Sequence[float],
    *,
    max_denominator: int = 12,
    tol: float = 1e-6,
) -> tuple[int, np.ndarray]:
    """Return ``(n, p)`` such that ``k = p / n`` and ``n`` is minimal.

    Components of ``k`` are taken modulo an integer reciprocal lattice vector,
    because ``exp(2*pi*i k.T)`` is unchanged by ``k -> k + G``.

    Raises:
        StructureEditError: If a component cannot be written as a fraction with
            a denominator of at most ``max_denominator``.  Such a ``k`` is
            incommensurate and has no finite magnetic cell.
    """
    values = np.asarray(k, dtype=float).reshape(3)
    values = values % 1.0
    values[np.isclose(values, 1.0, atol=tol)] = 0.0
    values[np.isclose(values, 0.0, atol=tol)] = 0.0

    fractions: list[Fraction] = []
    for index, value in enumerate(values):
        fraction = Fraction(float(value)).limit_denominator(max_denominator)
        if abs(float(fraction) - float(value)) > tol:
            raise StructureEditError(
                f"k[{index}] = {value:g} is not commensurate with a denominator "
                f"of at most {max_denominator}; an incommensurate magnetic "
                "structure has no finite cell"
            )
        fractions.append(fraction)

    n = 1
    for fraction in fractions:
        n = math.lcm(n, fraction.denominator)
    numerator = np.array([int(fraction * n) for fraction in fractions], dtype=int)
    return n, numerator


def magnetic_translation_lattice(numerator: np.ndarray, n: int) -> np.ndarray:
    """Return an integer basis of ``{h in Z^3 : numerator . h = 0 (mod n)}``.

    The rows of the returned matrix are the magnetic supercell vectors in units
    of the input cell vectors.  Its determinant is ``n``, the index of the
    magnetic translation group.
    """
    if n == 1:
        return np.eye(3, dtype=int)

    candidates = [
        vector
        for vector in product(range(-n, n + 1), repeat=3)
        if vector != (0, 0, 0) and int(np.dot(numerator, vector)) % n == 0
    ]
    if not candidates:
        raise StructureEditError(f"no magnetic translation vector found for n = {n}")
    vectors = np.asarray(candidates, dtype=int)
    order = np.argsort(np.einsum("ij,ij->i", vectors, vectors), kind="stable")
    vectors = vectors[order]

    # The shortest vectors span the sublattice in practice; widen the search
    # only when no triple with determinant n is found among them.
    first_limits = (12, 60, len(vectors))
    second_limits = (200, 800, len(vectors))
    for first_limit, second_limit in zip(first_limits, second_limits):
        for first in range(min(first_limit, len(vectors))):
            v1 = vectors[first]
            for second in range(first + 1, min(second_limit, len(vectors))):
                v2 = vectors[second]
                if np.linalg.norm(np.cross(v1, v2)) < 1e-9:
                    continue
                determinants = np.abs(vectors @ np.cross(v1, v2))
                hits = np.nonzero(determinants == n)[0]
                if len(hits):
                    v3 = vectors[hits[0]]
                    basis = np.asarray([v1, v2, v3], dtype=int)
                    # Keep the cell right-handed; negating a row keeps the
                    # same lattice and the same magnetic group.
                    if np.linalg.det(basis) < 0:
                        basis[2] = -basis[2]
                    return basis
    raise StructureEditError(
        f"could not find a basis with determinant {n} for the magnetic cell"
    )


def _element(atom) -> str:
    """Return the element symbol of an atom, falling back to its label."""
    return str(atom.element or atom.label).strip().capitalize()


def _moment(atom) -> Optional[float]:
    """Return the collinear moment of an atom, or ``None`` when it has none."""
    value = atom.mag if atom.mag is not None else atom.type_mag
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.linalg.norm(np.asarray(value, dtype=float)))
    return float(value)


def _unique_sites(atoms, positions, lattice, tolerance):
    """Drop repeated sites, comparing periodic distances instead of rounded text.

    Coordinates read from a file are usually rounded to a few decimals, so two
    equivalent images can differ by more than their textual precision.  A
    periodic distance tolerance is the only safe test.
    """
    kept_atoms = []
    kept_positions = []
    lattice = np.asarray(lattice, dtype=float)
    for atom, position in zip(atoms, positions):
        duplicate = False
        for kept_atom, kept_position in zip(kept_atoms, kept_positions):
            if kept_atom.label != atom.label:
                continue
            delta = np.asarray(position, dtype=float) - np.asarray(
                kept_position, dtype=float
            )
            delta -= np.rint(delta)
            if np.linalg.norm(delta @ lattice) < tolerance:
                duplicate = True
                break
        if duplicate:
            continue
        kept_atoms.append(atom)
        kept_positions.append(position)
    return kept_atoms, kept_positions


def build_magnetic_supercell(
    structure: AbacusSTRU,
    k: Sequence[float],
    *,
    elements: Optional[Sequence[str]] = None,
    moment: Optional[float] = None,
    phase: float = 0.0,
    max_denominator: int = 12,
    tol: float = 1e-6,
    allow_projection: bool = False,
) -> tuple[AbacusSTRU, dict[str, Any]]:
    """Build the commensurate magnetic cell of ``k`` from ``structure``.

    Args:
        structure: Input structure; ``k`` is given in its reciprocal basis.
        k: Propagation vector in fractional reciprocal coordinates.
        elements: Elements that carry a moment.  Default: every element that
            already has a non-zero moment in ``structure``.
        moment: Moment amplitude in Bohr magnetons.  Default: the amplitude
            already stored on each magnetic atom.
        phase: Phase of the spin wave at the origin, in degrees.
        max_denominator: Largest denominator accepted for ``k``.
        tol: Tolerance used to recognize rational ``k`` components and
            0/180 degree phases.
        allow_projection: Write ``m cos(2*pi*k.r + phase)`` even when the phase
            is not 0 or 180 degrees.  Such a structure is a spin-density wave,
            not a collinear +/- arrangement.

    Returns:
        tuple: The magnetic cell and a summary dictionary.
    """
    lattice = np.asarray(structure.cell, dtype=float)
    n, numerator = rationalize_propagation_vector(
        k, max_denominator=max_denominator, tol=tol
    )
    supercell_matrix = magnetic_translation_lattice(numerator, n)
    new_cell = supercell_matrix @ lattice
    inverse_matrix = np.linalg.inv(supercell_matrix)

    if elements is None:
        magnetic_elements = {
            _element(atom)
            for atom in structure.atoms
            if (_moment(atom) or 0.0) != 0.0
        }
        if not magnetic_elements:
            raise StructureEditError(
                "the structure has no magnetic moment; pass --elements to name "
                "the magnetic elements"
            )
    else:
        magnetic_elements = {str(element).strip().capitalize() for element in elements}

    present = {_element(atom) for atom in structure.atoms}
    missing = magnetic_elements - present
    if missing:
        raise StructureEditError(
            "magnetic element(s) not present in the structure: "
            + ", ".join(sorted(missing))
        )

    amplitudes: dict[str, float] = {}
    for element in magnetic_elements:
        values = []
        for atom in structure.atoms:
            if _element(atom) != element:
                continue
            value = _moment(atom)
            if value is not None and value != 0.0:
                values.append(abs(value))
        if moment is not None:
            amplitudes[element] = float(moment)
        elif values:
            amplitudes[element] = max(values)
        else:
            raise StructureEditError(
                f"element {element} has no moment in the structure; give --moment"
            )

    phase_origin = math.radians(float(phase))
    candidates = []
    candidate_positions = []
    worst_projection = 0.0

    direct = np.asarray(structure.coords_direct, dtype=float)
    for atom, fractional in zip(structure.atoms, direct):
        amplitude = amplitudes.get(_element(atom))
        for shift in product(range(n), repeat=3):
            shift_vector = np.asarray(shift, dtype=float)
            position = (fractional + shift_vector) @ inverse_matrix
            position = position % 1.0
            position[np.isclose(position, 1.0, atol=1e-10)] = 0.0

            new_atom = copy.deepcopy(atom)
            new_atom.coord = tuple(float(value) for value in position @ new_cell)
            new_atom.angle1 = None
            new_atom.angle2 = None
            if amplitude is None:
                new_atom.mag = 0.0
            else:
                angle = (
                    2.0 * math.pi * float(np.dot(numerator, fractional + shift_vector)) / n
                    + phase_origin
                )
                worst_projection = max(worst_projection, abs(math.sin(angle)))
                new_atom.mag = round(float(amplitude * math.cos(angle)), 8)
            candidates.append(new_atom)
            candidate_positions.append(position)

    new_atoms, _ = _unique_sites(
        candidates, candidate_positions, new_cell, max(tol, 1e-4)
    )

    if worst_projection > tol and not allow_projection:
        raise StructureEditError(
            "k puts magnetic atoms at phases other than 0 or 180 degrees, so a "
            "collinear +/- moment cannot represent it; use --allow-projection "
            "to write the collinear projection of the spin-density wave"
        )

    metadata = dict(structure.metadata)
    metadata["atom_type"] = "direct"
    edited = AbacusSTRU(
        cell=[tuple(float(value) for value in vector) for vector in new_cell],
        atoms=new_atoms,
        metadata=metadata,
    )

    summary = {
        "k": [float(value) for value in np.asarray(k, dtype=float).reshape(3)],
        "reduced_k": [float(value) for value in numerator / n],
        "index": int(n),
        "supercell_matrix": [[int(value) for value in row] for row in supercell_matrix],
        "volume_factor": int(abs(round(np.linalg.det(supercell_matrix)))),
        "magnetic_elements": sorted(magnetic_elements),
        "amplitudes": {element: float(value) for element, value in amplitudes.items()},
        "phase": float(phase),
        "atoms_before": structure.natoms,
        "atoms_after": edited.natoms,
    }
    return edited, summary


def _spglib_moments(structure: AbacusSTRU) -> list[Any]:
    """Return moments in the shape spglib expects."""
    vectors = any(
        isinstance(atom.mag, (list, tuple, np.ndarray)) for atom in structure.atoms
    )
    moments: list[Any] = []
    for atom in structure.atoms:
        value = atom.mag if atom.mag is not None else atom.type_mag
        if isinstance(value, (list, tuple, np.ndarray)):
            moments.append([float(component) for component in value])
        elif value is None:
            moments.append([0.0, 0.0, 0.0] if vectors else 0.0)
        else:
            moments.append([0.0, 0.0, float(value)] if vectors else float(value))
    return moments


def _magnetic_dataset(
    structure: AbacusSTRU,
    *,
    symprec: float,
    angle_tolerance: float,
    mag_symprec: float,
):
    """Return the spglib magnetic dataset of ``structure``."""
    try:
        import spglib
    except ImportError as error:  # pragma: no cover - spglib is a dependency
        raise StructureEditError("spglib is not installed") from error
    from ase.data import atomic_numbers

    lattice = np.asarray(structure.cell, dtype=float)
    positions = np.asarray(structure.coords_direct, dtype=float)
    numbers = []
    for atom in structure.atoms:
        key = _element(atom)
        if key not in atomic_numbers:
            raise StructureEditError(f"unknown element: {key}")
        numbers.append(int(atomic_numbers[key]))
    moments = _spglib_moments(structure)

    try:
        dataset = spglib.get_magnetic_symmetry_dataset(
            (lattice, positions, numbers, moments),
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            mag_symprec=mag_symprec,
        )
    except (ValueError, OSError, spglib.SpglibError) as error:
        raise StructureEditError(f"magnetic symmetry failed: {error}") from error
    if dataset is None:
        raise StructureEditError("spglib found no magnetic symmetry")
    return dataset


def _right_handed(
    lattice: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flip the last lattice vector and its fractional coordinate if needed."""
    if np.linalg.det(lattice) >= 0:
        return lattice, positions
    lattice = np.array(lattice, dtype=float, copy=True)
    positions = np.array(positions, dtype=float, copy=True)
    lattice[2] = -lattice[2]
    positions[:, 2] = -positions[:, 2]
    return lattice, positions


def magnetic_primitive(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
    mag_symprec: float = -1.0,
) -> tuple[AbacusSTRU, dict[str, Any]]:
    """Reduce a magnetic structure to the primitive cell of its magnetic group.

    The magnetic symmetry is evaluated by spglib from the species and the
    moments, so the result is the primitive cell of the *magnetic* translation
    group rather than of the crystal alone.

    Args:
        structure: Structure whose atoms carry the moments.
        symprec: Position tolerance in Angstrom.
        angle_tolerance: Lattice angle tolerance in degrees; negative keeps the
            spglib default.
        mag_symprec: Moment tolerance; negative keeps the spglib default.

    Returns:
        tuple: The primitive magnetic cell and a summary dictionary.
    """
    if not any((_moment(atom) or 0.0) != 0.0 for atom in structure.atoms):
        raise StructureEditError(
            "the structure carries no magnetic moment; magnetic_primitive "
            "needs a decorated structure"
        )
    lattice = np.asarray(structure.cell, dtype=float)
    positions = np.asarray(structure.coords_direct, dtype=float)
    dataset = _magnetic_dataset(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        mag_symprec=mag_symprec,
    )
    primitive_lattice = np.asarray(dataset.primitive_lattice, dtype=float)
    if np.linalg.det(primitive_lattice) < 0:
        primitive_lattice = primitive_lattice.copy()
        primitive_lattice[2] = -primitive_lattice[2]
    volume_factor = abs(np.linalg.det(lattice)) / abs(np.linalg.det(primitive_lattice))
    factor = int(round(volume_factor))
    if factor < 1 or abs(volume_factor - factor) > 1e-4:
        raise StructureEditError(
            f"magnetic primitive cell has a non-integer volume ratio {volume_factor}"
        )

    transform = lattice @ np.linalg.inv(primitive_lattice)
    new_positions = (positions @ transform) % 1.0
    new_positions[np.isclose(new_positions, 1.0, atol=1e-10)] = 0.0

    candidates = []
    for atom, position in zip(structure.atoms, new_positions):
        new_atom = copy.deepcopy(atom)
        new_atom.coord = tuple(float(value) for value in position @ primitive_lattice)
        candidates.append(new_atom)
    new_atoms, _ = _unique_sites(
        candidates, new_positions, primitive_lattice, max(symprec, 1e-4)
    )

    expected = structure.natoms / factor
    if abs(len(new_atoms) - expected) > 1e-6:
        raise StructureEditError(
            "the magnetic primitive cell kept "
            f"{len(new_atoms)} atoms, expected {expected:g}; the moments may "
            "break the translation symmetry inconsistently"
        )

    metadata = dict(structure.metadata)
    metadata["atom_type"] = "direct"
    edited = AbacusSTRU(
        cell=[tuple(float(value) for value in vector) for vector in primitive_lattice],
        atoms=new_atoms,
        metadata=metadata,
    )
    summary = {
        "volume_factor": factor,
        "atoms_before": structure.natoms,
        "atoms_after": edited.natoms,
        "uni_number": int(dataset.uni_number),
        "msg_type": int(dataset.msg_type),
    }
    return edited, summary


def magnetic_standard_cell(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
    mag_symprec: float = -1.0,
) -> tuple[AbacusSTRU, dict[str, Any]]:
    """Return the standardized conventional cell of the magnetic group.

    spglib's standardized cell keeps the magnetic translation group and gives
    the conventional setting of the magnetic space group (for rock-salt
    AFM-II, the 12-atom hexagonal cell).
    """
    from ase.data import chemical_symbols

    dataset = _magnetic_dataset(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        mag_symprec=mag_symprec,
    )
    lattice = np.asarray(dataset.std_lattice, dtype=float)
    positions = np.asarray(dataset.std_positions, dtype=float)
    lattice, positions = _right_handed(lattice, positions)
    positions = positions % 1.0
    positions[np.isclose(positions, 1.0, atol=1e-10)] = 0.0
    tensors = np.asarray(dataset.std_tensors)

    templates: dict[str, Any] = {}
    for atom in structure.atoms:
        templates.setdefault(_element(atom), atom)

    new_atoms = []
    for position, number, tensor in zip(positions, dataset.std_types, tensors):
        symbol = chemical_symbols[int(number)]
        template = templates.get(symbol)
        if template is None:
            raise StructureEditError(
                f"spglib returned element {symbol} not found in the structure"
            )
        new_atom = copy.deepcopy(template)
        new_atom.element = symbol
        new_atom.coord = tuple(float(value) for value in position @ lattice)
        if np.ndim(tensor) == 0:
            new_atom.mag = float(tensor)
            new_atom.angle1 = None
            new_atom.angle2 = None
        else:
            new_atom.mag = tuple(float(value) for value in np.ravel(tensor))
            new_atom.angle1 = None
            new_atom.angle2 = None
        new_atoms.append(new_atom)

    metadata = dict(structure.metadata)
    metadata["atom_type"] = "direct"
    edited = AbacusSTRU(
        cell=[tuple(float(value) for value in vector) for vector in lattice],
        atoms=new_atoms,
        metadata=metadata,
    )
    summary = {
        "volume_factor": float(
            abs(np.linalg.det(np.asarray(structure.cell, dtype=float)))
            / abs(np.linalg.det(lattice))
        ),
        "atoms_before": structure.natoms,
        "atoms_after": edited.natoms,
        "uni_number": int(dataset.uni_number),
        "msg_type": int(dataset.msg_type),
    }
    return edited, summary


__all__ = [
    "build_magnetic_supercell",
    "magnetic_primitive",
    "magnetic_standard_cell",
    "magnetic_translation_lattice",
    "rationalize_propagation_vector",
]
