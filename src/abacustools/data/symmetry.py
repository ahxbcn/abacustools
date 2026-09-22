"""Symmetry analysis of ABACUS structures.

The crystallographic space group comes from pymatgen, which wraps spglib.  The
magnetic space group, the magnetic ordering, the layer group and the site
symmetry symbols come from spglib directly; spglib is installed together with
pymatgen and phonopy, so no additional dependency is required.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from abacustools.data.dimensionality import classify_dimensionality
from abacustools.io.stru import AbacusSTRU


@dataclass(frozen=True)
class SpaceGroupOperation:
    """One space-group operation of a crystal.

    Attributes:
        rotation: Integer 3x3 matrix acting on fractional coordinates.
        translation: Fractional translation vector of the operation.
        permutation: Atom permutation, ``permutation[a]`` being the atom that
            atom ``a`` is moved onto.
    """

    rotation: np.ndarray
    translation: np.ndarray
    permutation: tuple[int, ...]

    def rotate_kpoint(self, direct: Sequence[float]) -> np.ndarray:
        """Return the k-point that this operation maps ``direct`` onto.

        The reciprocal-space rotation is the transpose of the direct-space
        one, which is what ABACUS uses when it reduces a k-point mesh.
        """
        return np.asarray(direct, dtype=float) @ self.rotation


def atom_permutation(
    structure: AbacusSTRU,
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
    *,
    tolerance: float = 0.05,
) -> Optional[tuple[int, ...]]:
    """Return the atom permutation of one operation, or None.

    Args:
        structure: Structure the operation acts on.
        rotation: Integer 3x3 matrix in fractional coordinates.
        translation: Fractional translation vector.
        tolerance: Distance in Angstrom within which two positions count as
            the same atom.

    Returns:
        ``permutation[a]`` with the atom that atom ``a`` is moved onto, or
        None when the operation maps an atom onto no atom of the cell or maps
        two atoms onto the same one.
    """
    cell = np.asarray(structure.cell, dtype=float)
    positions = np.asarray(structure.coords_direct, dtype=float)
    moved = positions @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)
    permutation = []
    for target in moved:
        delta = positions - target
        delta -= np.round(delta)
        distances = np.linalg.norm(delta @ cell, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > tolerance:
            return None
        permutation.append(index)
    if len(set(permutation)) != len(permutation):
        return None
    return tuple(permutation)


def space_group_operations(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-4,
    tolerance: float = 0.05,
) -> list[SpaceGroupOperation]:
    """Return the space-group operations of a structure.

    Args:
        structure: Structure to analyse.
        symprec: Distance tolerance in Angstrom handed to spglib.
        tolerance: Distance in Angstrom within which an operation has to map
            atoms onto atoms.

    Returns:
        The operations, each with the atom permutation it induces. Operations
        that differ only in the translation part are kept apart, because their
        permutations differ.

    Raises:
        ValueError: If the cell is not periodic or spglib finds no symmetry.
    """
    import spglib

    if _periodic_cell(structure) is None:
        raise ValueError("a non-zero three-dimensional periodic cell is required")
    dataset = spglib.get_symmetry_dataset(_spglib_cell(structure), symprec=symprec)
    if dataset is None:
        raise ValueError("spglib found no symmetry for the structure")
    operations: list[SpaceGroupOperation] = []
    seen: set[tuple[Any, ...]] = set()
    for rotation, translation in zip(dataset.rotations, dataset.translations):
        matrix = np.asarray(rotation, dtype=int)
        shift = np.asarray(translation, dtype=float)
        permutation = atom_permutation(structure, matrix, shift, tolerance=tolerance)
        key = (tuple(matrix.flatten()), permutation)
        if permutation is None or key in seen:
            continue
        seen.add(key)
        operations.append(SpaceGroupOperation(matrix, shift, permutation))
    return operations


_DIRECTION_AXES = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}
_DIRECTION_NAMES = ("a", "b", "c")
_BRAVAIS_LETTERS = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "rhombohedral": "h",
    "hexagonal": "h",
    "cubic": "c",
}
_CENTERING_LETTERS = ("P", "A", "B", "C", "I", "F", "R")
_MAGNETIC_TYPES = {
    1: "type I",
    2: "type II (grey)",
    3: "type III",
    4: "type IV",
}


def _periodic_cell(structure: AbacusSTRU) -> Optional[np.ndarray]:
    """Return the cell when it can carry a symmetry analysis."""
    cell = np.asarray(structure.cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        return None
    if abs(float(np.linalg.det(cell))) <= 1e-12:
        return None
    return cell


def _atomic_numbers(structure: AbacusSTRU) -> list[int]:
    """Return the atomic number of every atom."""
    from ase.data import atomic_numbers

    numbers = []
    for index, atom in enumerate(structure.atoms, 1):
        element = atom.element or atom.label
        key = str(element).strip().capitalize()
        if key not in atomic_numbers:
            raise ValueError(f"atom {index} has an unknown element: {element}")
        numbers.append(int(atomic_numbers[key]))
    return numbers


def _spglib_cell(structure: AbacusSTRU):
    """Return the ``(lattice, positions, numbers)`` triple spglib expects."""
    return (
        np.asarray(structure.cell, dtype=float),
        np.asarray(structure.coords_direct, dtype=float),
        _atomic_numbers(structure),
    )


def _moments(structure: AbacusSTRU) -> Optional[list[Any]]:
    """Return the per-atom magnetic moments, or ``None`` when there is none."""
    vectors = any(
        isinstance(atom.mag, (list, tuple)) for atom in structure.atoms
    )
    moments: list[Any] = []
    for index, atom in enumerate(structure.atoms, 1):
        value = atom.mag if atom.mag is not None else (atom.type_mag or None)
        if value is None:
            moments.append([0.0, 0.0, 0.0] if vectors else 0.0)
            continue
        if isinstance(value, (list, tuple)):
            if len(value) != 3:
                raise ValueError(f"atom {index} has a moment that is not a 3-vector")
            moments.append([float(component) for component in value])
        elif vectors:
            # A collinear moment inside a non-collinear structure points along z.
            moments.append([0.0, 0.0, float(value)])
        else:
            moments.append(float(value))
    if all(np.all(np.asarray(moment, dtype=float) == 0.0) for moment in moments):
        return None
    return moments


def _dataset_field(dataset: Any, name: str) -> Any:
    """Read one field of a pymatgen symmetry dataset, old or new style."""
    if hasattr(dataset, name):
        return getattr(dataset, name)
    return dataset[name]


def bravais_lattice(
    crystal_system: str,
    space_group_symbol: str,
) -> Optional[str]:
    """Return the Bravais lattice type, such as ``cF`` for face-centred cubic.

    Args:
        crystal_system: Crystal system, for example ``cubic``.
        space_group_symbol: International symbol of the space group.

    Returns:
        Optional[str]: The crystal system letter followed by the centring
        letter, or ``None`` when either of them is not recognised.
    """
    letter = _BRAVAIS_LETTERS.get(str(crystal_system).strip().lower())
    centring = str(space_group_symbol).strip()[:1].upper()
    if letter is None or centring not in _CENTERING_LETTERS:
        return None
    return f"{letter}{centring}"


def _polar_point_group(analyzer: Any) -> Optional[bool]:
    """Return whether the point group keeps one direction invariant.

    A polar point group has a direction that every one of its operations
    leaves alone, which is what a spontaneous polarisation needs.
    """
    try:
        differences = np.stack(
            [
                np.asarray(operation.rotation_matrix, dtype=float) - np.eye(3)
                for operation in analyzer.get_point_group_operations(cartesian=True)
            ]
        )
        return bool(np.linalg.matrix_rank(differences.reshape(-1, 3), tol=1e-6) < 3)
    except Exception:
        return None


def _conventional_cell(
    analyzer: Any,
    wyckoffs: list[str],
    symprec: float,
    angle_tolerance: float,
) -> tuple[list[Optional[int]], Optional[int]]:
    """Return the Wyckoff multiplicities and the conventional cell size.

    A multiplicity counts the atoms of the orbit in the conventional cell, so
    it does not depend on the cell that was handed in: a primitive diamond cell
    still reports its sites as ``8a``.  The orbits are counted through the
    equivalent atoms and not through the Wyckoff letters, because the general
    position of ``P1`` carries the same letter for every atom.

    Returns:
        tuple: The multiplicity of every atom and the number of atoms in the
        conventional cell, both ``None`` when that cell could not be built.
    """
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        conventional = analyzer.get_conventional_standard_structure()
        dataset = SpacegroupAnalyzer(
            conventional,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        ).get_symmetry_dataset()
        equivalent_atoms = [int(value) for value in _dataset_field(dataset, "equivalent_atoms")]
        letters = [str(letter) for letter in _dataset_field(dataset, "wyckoffs")]
        sizes = Counter(equivalent_atoms)
        counts: dict[str, int] = {}
        for representative, letter in zip(equivalent_atoms, letters):
            counts[letter] = sizes[representative]
        return [counts.get(letter) for letter in wyckoffs], len(conventional)
    except Exception:
        return [None] * len(wyckoffs), None


def crystallographic_symmetry(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
) -> dict[str, Any]:
    """Analyse the space group and Wyckoff positions of a structure.

    Args:
        structure: Structure to analyse.
        symprec: Symmetry distance tolerance in Angstrom.
        angle_tolerance: Symmetry angle tolerance in degrees.

    Returns:
        dict: Space group, point group with its Schoenflies symbol, Bravais
        lattice, inversion and polarity, Wyckoff positions with their
        multiplicity and the equivalent atoms, or ``available=False`` with a
        reason.
    """
    if _periodic_cell(structure) is None:
        return {
            "available": False,
            "error": "a non-zero three-dimensional periodic cell is required",
        }
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        analyzer = SpacegroupAnalyzer(
            structure.to("pymatgen"),
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        dataset = analyzer.get_symmetry_dataset()
        wyckoffs = [str(letter) for letter in _dataset_field(dataset, "wyckoffs")]
        equivalent_atoms = [
            int(value) + 1 for value in _dataset_field(dataset, "equivalent_atoms")
        ]
        try:
            import spglib

            space_group_type = spglib.get_spacegroup_type(
                int(_dataset_field(dataset, "hall_number"))
            )
        except Exception:
            space_group_type = None
        multiplicities, conventional_atoms = _conventional_cell(
            analyzer,
            wyckoffs,
            symprec,
            angle_tolerance,
        )
        symbol = analyzer.get_space_group_symbol()
        lattice_type = bravais_lattice(analyzer.get_crystal_system(), symbol)
        return {
            "available": True,
            "space_group_symbol": symbol,
            "space_group_number": int(analyzer.get_space_group_number()),
            "crystal_system": analyzer.get_crystal_system(),
            "lattice_type": analyzer.get_lattice_type(),
            "point_group": analyzer.get_point_group_symbol(),
            "schoenflies": (
                None if space_group_type is None else space_group_type.pointgroup_schoenflies
            ),
            "bravais_lattice": lattice_type,
            "pearson_symbol": (
                None
                if lattice_type is None or conventional_atoms is None
                else f"{lattice_type}{conventional_atoms}"
            ),
            "inversion_symmetry": bool(analyzer.is_laue()),
            "polar_point_group": _polar_point_group(analyzer),
            "symmetry_operations": len(analyzer.get_symmetry_operations()),
            "wyckoff_positions": wyckoffs,
            "wyckoff_multiplicities": multiplicities,
            "equivalent_atoms": equivalent_atoms,
            "symprec": symprec,
            "angle_tolerance": angle_tolerance,
        }
    except Exception as error:
        return {
            "available": False,
            "error": f"symmetry analysis failed: {error}",
            "symprec": symprec,
            "angle_tolerance": angle_tolerance,
        }


def space_group_summary(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
) -> dict[str, Any]:
    """Return the space group and crystal system of a structure.

    This is the part of :func:`crystallographic_symmetry` that a caller listing
    the main fields of many structures needs. It skips the Wyckoff positions,
    the point group, the Bravais lattice and the conventional cell, which cost
    the most and are only read by the full report.

    Args:
        structure: Structure to analyse.
        symprec: Symmetry distance tolerance in Angstrom.
        angle_tolerance: Symmetry angle tolerance in degrees.

    Returns:
        dict: Space group symbol and number with the crystal system, or
        ``available=False`` with a reason.
    """
    unavailable = {
        "available": False,
        "space_group_symbol": None,
        "space_group_number": None,
        "crystal_system": None,
    }
    if _periodic_cell(structure) is None:
        return {
            **unavailable,
            "error": "a non-zero three-dimensional periodic cell is required",
        }
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        analyzer = SpacegroupAnalyzer(
            structure.to("pymatgen"),
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        return {
            "available": True,
            "error": None,
            "space_group_symbol": analyzer.get_space_group_symbol(),
            "space_group_number": int(analyzer.get_space_group_number()),
            "crystal_system": analyzer.get_crystal_system(),
        }
    except Exception as error:
        return {**unavailable, "error": f"symmetry analysis failed: {error}"}


def site_symmetry_symbols(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
) -> Optional[list[str]]:
    """Return the site symmetry symbol of every atom, when available."""
    if _periodic_cell(structure) is None:
        return None
    try:
        import spglib
    except ImportError:
        return None
    try:
        dataset = spglib.get_symmetry_dataset(
            _spglib_cell(structure),
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
    except (ValueError, OSError, spglib.SpglibError):
        return None
    if dataset is None:
        return None
    symbols = list(getattr(dataset, "site_symmetry_symbols", []) or [])
    return symbols or None


def _magnetic_space_group_label(bns_number: Optional[str]) -> Optional[str]:
    """Look up a magnetic space group label in the table shipped by pymatgen."""
    if not bns_number or "." not in str(bns_number):
        return None
    try:
        from pymatgen.symmetry.maggroups import MAGSYMM_DATA

        first, second = str(bns_number).split(".", 1)
        connection = sqlite3.connect(MAGSYMM_DATA)
        try:
            row = connection.execute(
                "SELECT BNS_label, OG_label FROM space_groups WHERE BNS1=? AND BNS2=?",
                (int(first), int(second)),
            ).fetchone()
        finally:
            connection.close()
    except (ImportError, OSError, sqlite3.Error, ValueError):
        return None
    if not row:
        return None
    return str(row[0] or row[1] or "") or None


def magnetic_symmetry(
    structure: AbacusSTRU,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
    mag_symprec: float = -1.0,
) -> dict[str, Any]:
    """Analyse the magnetic space group of a structure.

    A structure without magnetic moments keeps the grey group, in which every
    spatial operation appears together with time reversal.

    Args:
        structure: Structure to analyse.
        symprec: Symmetry distance tolerance in Angstrom.
        angle_tolerance: Symmetry angle tolerance in degrees.
        mag_symprec: Tolerance for the magnetic moments.

    Returns:
        dict: Magnetic space group label, BNS/OG/UNI numbers, type and the
        operation counts, or ``available=False`` with a reason.
    """
    if _periodic_cell(structure) is None:
        return {
            "available": False,
            "error": "a non-zero three-dimensional periodic cell is required",
        }
    try:
        import spglib
    except ImportError:
        return {"available": False, "error": "spglib is not installed"}
    try:
        moments = _moments(structure)
        lattice, positions, numbers = _spglib_cell(structure)
        cell = (
            lattice,
            positions,
            numbers,
            [0.0] * len(numbers) if moments is None else moments,
        )
        dataset = spglib.get_magnetic_symmetry_dataset(
            cell,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            mag_symprec=mag_symprec,
        )
        if dataset is None:
            return {"available": False, "error": "spglib found no magnetic symmetry"}
        space_group_type = spglib.get_magnetic_spacegroup_type(dataset.uni_number)
    except (ValueError, OSError, spglib.SpglibError) as error:
        return {"available": False, "error": f"magnetic symmetry failed: {error}"}
    if space_group_type is None:
        return {
            "available": False,
            "error": f"unknown magnetic space group type: UNI {dataset.uni_number}",
        }

    operations_with_time_reversal = int(np.sum(dataset.time_reversals))
    return {
        "available": True,
        "grey": moments is None,
        "label": _magnetic_space_group_label(space_group_type.bns_number),
        "bns_number": space_group_type.bns_number,
        "og_number": space_group_type.og_number,
        "uni_number": int(dataset.uni_number),
        "msg_type": int(dataset.msg_type),
        "type": _MAGNETIC_TYPES.get(int(dataset.msg_type), "unknown"),
        "operations": int(len(dataset.rotations)),
        "operations_with_time_reversal": operations_with_time_reversal,
    }


def magnetic_ordering(structure: AbacusSTRU) -> dict[str, Any]:
    """Classify the collinear magnetic ordering and report the net moment.

    Args:
        structure: Structure to inspect.

    Returns:
        dict: ``ordering`` (FM, AFM, FiM or NM), the net moment and the number
        of magnetic sites, or ``available=False`` with a reason.
    """
    try:
        moments = _moments(structure)
    except ValueError as error:
        return {"available": False, "error": str(error)}
    if moments is None:
        return {
            "available": False,
            "error": "the structure has no magnetic moments",
        }
    if any(isinstance(moment, list) for moment in moments):
        return {
            "available": False,
            "error": "non-collinear moments are not classified",
        }
    try:
        from pymatgen.analysis.magnetism.analyzer import (
            CollinearMagneticStructureAnalyzer,
        )

        pymatgen_structure = structure.to("pymatgen")
        pymatgen_structure.add_site_property("magmom", [float(m) for m in moments])
        # Keep the cell as it was read, so that the magnetic sites and the net
        # moment always refer to the same atoms.
        analyzer = CollinearMagneticStructureAnalyzer(pymatgen_structure, make_primitive=False)
    except (ImportError, ValueError, OSError) as error:
        return {"available": False, "error": f"magnetic ordering failed: {error}"}
    return {
        "available": True,
        "ordering": str(analyzer.ordering).split(".")[-1],
        "net_moment": float(analyzer.total_magmoms),
        "magnetic_sites": int(analyzer.number_of_magnetic_sites),
        "unique_magnetic_sites": int(analyzer.number_of_unique_magnetic_sites()),
    }


def detect_aperiodic_direction(
    structure: AbacusSTRU,
    *,
    min_vacuum: float = 5.0,
) -> Optional[str]:
    """Return the vacuum direction of a slab.

    Args:
        structure: Structure to inspect.
        min_vacuum: Empty span in Angstrom that counts as vacuum.

    Returns:
        Optional[str]: ``a``, ``b`` or ``c``, or ``None`` when the structure is
        not a slab, because a bulk has no vacuum and a wire or a molecule has
        more than one vacuum direction.
    """
    if _periodic_cell(structure) is None:
        return None
    dimension = classify_dimensionality(structure, min_vacuum=min_vacuum)
    if dimension["dimensionality"] != "slab":
        return None
    return dimension["vacuum_directions"][0]


def layer_symmetry(
    structure: AbacusSTRU,
    *,
    direction: Optional[Union[str, int]] = None,
    symprec: float = 1e-5,
    min_vacuum: float = 5.0,
    detect: bool = True,
) -> dict[str, Any]:
    """Analyse the layer group of a slab.

    Layer groups describe the symmetry of a slab, for which the three
    dimensional space group carries no useful information because the cell is
    periodic along the surface normal only through its vacuum.  spglib takes
    no angle tolerance for this analysis.

    Args:
        structure: Structure to analyse.
        direction: Aperiodic direction, ``a``/``b``/``c``; detected by default.
        symprec: Symmetry distance tolerance in Angstrom.
        min_vacuum: Empty span in Angstrom that counts as vacuum.
        detect: Look for a vacuum direction when none is given.

    Returns:
        dict: The layer group number, symbol and operation count, or
        ``available=False`` with a reason.  The requested direction is always
        reported, so that callers can tell a bulk structure from a slab whose
        direction was given explicitly.
    """
    requested = None if direction is None else str(direction)
    if _periodic_cell(structure) is None:
        return {
            "available": False,
            "error": "a non-zero three-dimensional periodic cell is required",
            "requested_direction": requested,
        }
    axis: Optional[int] = None
    reason = "no vacuum direction detected; pass --layer-direction"
    if direction is not None:
        key = str(direction).strip().lower()
        if key not in _DIRECTION_AXES:
            return {
                "available": False,
                "error": f"unknown layer direction: {direction}",
                "requested_direction": requested,
            }
        axis = _DIRECTION_AXES[key]
    elif detect:
        dimension = classify_dimensionality(structure, min_vacuum=min_vacuum)
        vacuum_directions = dimension["vacuum_directions"]
        if dimension["dimensionality"] == "slab":
            axis = _DIRECTION_AXES[vacuum_directions[0]]
        elif dimension["dimensionality"] != "bulk":
            reason = (
                f"a {dimension['label']} has {len(vacuum_directions)} vacuum directions; "
                "pass --layer-direction to pick one"
            )
    if axis is None:
        return {
            "available": False,
            "error": reason,
            "requested_direction": requested,
        }
    try:
        import spglib
    except ImportError:
        return {
            "available": False,
            "error": "spglib is not installed",
            "requested_direction": requested,
        }
    try:
        dataset = spglib.get_symmetry_layerdataset(
            _spglib_cell(structure),
            aperiodic_dir=axis,
            symprec=symprec,
        )
    except (ValueError, OSError, spglib.SpglibError) as error:
        return {
            "available": False,
            "error": f"layer symmetry failed: {error}",
            "requested_direction": requested,
        }
    if dataset is None:
        return {
            "available": False,
            "error": "spglib found no layer symmetry",
            "requested_direction": requested,
        }
    return {
        "available": True,
        "direction": _DIRECTION_NAMES[axis],
        "requested_direction": requested,
        "number": int(dataset.number),
        "symbol": dataset.international,
        "operations": int(len(dataset.rotations)),
    }


__all__ = [
    "SpaceGroupOperation",
    "atom_permutation",
    "crystallographic_symmetry",
    "detect_aperiodic_direction",
    "layer_symmetry",
    "magnetic_ordering",
    "magnetic_symmetry",
    "site_symmetry_symbols",
    "space_group_operations",
]
