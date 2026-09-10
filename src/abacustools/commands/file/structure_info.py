"""Display crystallographic and ABACUS metadata for a structure file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from abacustools.io.stru import AbacusSTRU


def _structure_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"structure file does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the structure information command."""
    parser = subparsers.add_parser(
        "info",
        aliases=["stru-info", "structure-info"],
        help="Show basic crystallographic and ABACUS structure information.",
    )
    parser.add_argument("filename", type=_structure_file, metavar="STRUCTURE")
    parser.add_argument(
        "--input-format",
        default=None,
        help="Input format (STRU, POSCAR, CIF, XYZ, EXTXYZ, or XSF).",
    )
    parser.add_argument(
        "--cell",
        type=float,
        nargs=9,
        metavar="CELL",
        help="Nine cell-vector components for formats without a cell, such as XYZ.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-5,
        help="Symmetry distance tolerance in Angstrom, default: 1e-5.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=5.0,
        help="Symmetry angle tolerance in degrees, default: 5.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a formatted report.")
    parser.set_defaults(handler=run)


def _validate_tolerances(symprec: float, angle_tolerance: float) -> None:
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be a positive finite number")
    if not np.isfinite(angle_tolerance) or angle_tolerance < 0:
        raise ValueError("angle-tolerance must be a non-negative finite number")


def _round_values(values: Any, digits: int = 8) -> Any:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim == 0:
        return float(array)
    return np.round(array.astype(float), digits).tolist()


def _resource_by_label(structure, attribute: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for atom in structure.atoms:
        value = getattr(atom, attribute)
        values.setdefault(atom.label, [])
        if value not in values[atom.label]:
            values[atom.label].append(value)
    return values


def _atom_moment(atom) -> Any:
    """Return the magnetic moment of an atom, falling back to its type moment."""
    if atom.mag is not None:
        return _round_values(atom.mag) if isinstance(atom.mag, (list, tuple)) else float(atom.mag)
    if atom.type_mag:
        return float(atom.type_mag)
    return None


def _atom_moment_angles(atom) -> Optional[list[Any]]:
    """Return the polar angles of a magnetic moment, when they are set."""
    if atom.angle1 is None and atom.angle2 is None:
        return None
    return [None if atom.angle1 is None else float(atom.angle1),
            None if atom.angle2 is None else float(atom.angle2)]


def _symmetry_info(structure, symprec: float, angle_tolerance: float) -> dict[str, Any]:
    cell = np.asarray(structure.cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) <= 1e-12:
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
        if hasattr(dataset, "wyckoffs"):
            wyckoffs = list(dataset.wyckoffs)
        else:
            wyckoffs = list(dataset["wyckoffs"])
        if hasattr(dataset, "equivalent_atoms"):
            equivalent_atoms = [int(value) + 1 for value in dataset.equivalent_atoms]
        else:
            equivalent_atoms = [int(value) + 1 for value in dataset["equivalent_atoms"]]
        return {
            "available": True,
            "space_group_symbol": analyzer.get_space_group_symbol(),
            "space_group_number": int(analyzer.get_space_group_number()),
            "crystal_system": analyzer.get_crystal_system(),
            "lattice_type": analyzer.get_lattice_type(),
            "point_group": analyzer.get_point_group_symbol(),
            "symmetry_operations": len(analyzer.get_symmetry_operations()),
            "wyckoff_positions": wyckoffs,
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


def structure_information(
    filename: Path,
    *,
    input_format: Optional[str] = None,
    cell: Optional[list[float]] = None,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
) -> dict[str, Any]:
    """Read a structure and return JSON-compatible basic information."""
    _validate_tolerances(symprec, angle_tolerance)
    structure = AbacusSTRU.read(
        filename,
        fmt=input_format,
        cell=None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3),
    )
    if structure is None:
        raise RuntimeError(f"failed to read structure: {filename}")

    cell_vectors = np.asarray(structure.cell, dtype=float)
    volume = None
    if cell_vectors.shape == (3, 3) and np.all(np.isfinite(cell_vectors)):
        volume = abs(float(np.linalg.det(cell_vectors)))
    lengths_angles = structure.get_cell_param() if volume and volume > 1e-12 else None
    elements = [atom.element or atom.label for atom in structure.atoms]
    labels = [atom.label for atom in structure.atoms]
    symmetry = _symmetry_info(structure, symprec, angle_tolerance)
    wyckoffs = symmetry.get("wyckoff_positions", [None] * structure.natoms)

    fractional_coordinates = None
    if volume is not None and volume > 1e-12:
        fractional_coordinates = structure.coords_direct
    atoms = []
    for index, atom in enumerate(structure.atoms):
        atoms.append(
            {
                "index": index + 1,
                "label": atom.label,
                "element": atom.element or atom.label,
                "cartesian": _round_values(atom.coord),
                "fractional": None if fractional_coordinates is None else _round_values(fractional_coordinates[index]),
                "wyckoff": wyckoffs[index] if index < len(wyckoffs) else None,
                "mass": None if atom.mass is None else float(atom.mass),
                "magmom": _atom_moment(atom),
                "magmom_angles": _atom_moment_angles(atom),
                "move": list(atom.move) if atom.move is not None else None,
                "velocity": None if atom.velocity is None else _round_values(atom.velocity),
            }
        )

    equivalent_atoms = symmetry.get("equivalent_atoms")
    if not isinstance(equivalent_atoms, list) or len(equivalent_atoms) != structure.natoms:
        equivalent_atoms = list(range(1, structure.natoms + 1))
    groups: dict[int, list[int]] = {}
    for atom_index, representative in enumerate(equivalent_atoms, 1):
        groups.setdefault(representative, []).append(atom_index)
    inequivalent_positions = []
    for equivalent_indices in groups.values():
        representative = atoms[equivalent_indices[0] - 1]
        inequivalent_positions.append(
            {
                "representative_index": representative["index"],
                "equivalent_indices": equivalent_indices,
                "multiplicity": len(equivalent_indices),
                "label": representative["label"],
                "element": representative["element"],
                "fractional": representative["fractional"],
                "cartesian": representative["cartesian"],
                "wyckoff": representative["wyckoff"],
            }
        )

    return {
        "file": str(Path(filename).absolute()),
        "format": input_format,
        "natoms": structure.natoms,
        "formula": " ".join(f"{element}{count if count != 1 else ''}" for element, count in Counter(elements).items()),
        "element_counts": dict(Counter(elements)),
        "label_counts": dict(Counter(labels)),
        "cell": {
            "vectors_angstrom": _round_values(cell_vectors),
            "volume_angstrom3": volume,
            "lengths_angstrom": None if lengths_angles is None else _round_values(lengths_angles[:3]),
            "angles_degree": None if lengths_angles is None else _round_values(lengths_angles[3:]),
            "periodic": volume is not None and volume > 1e-12,
        },
        "resources": {
            "pseudopotentials": _resource_by_label(structure, "pp"),
            "orbitals": _resource_by_label(structure, "orb"),
        },
        "symmetry": symmetry,
        "atoms": atoms,
        "inequivalent_positions": inequivalent_positions,
    }


def _display_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return " ".join(_display_value(item) for item in value)
    return str(value)


def _moment_text(atom: dict[str, Any]) -> str:
    """Format one atom's magnetic moment and its angles, if any."""
    text = _display_value(atom["magmom"])
    angles = atom.get("magmom_angles")
    if angles:
        text += " (" + ", ".join(_display_value(angle) for angle in angles) + ")"
    return text


def _move_text(atom: dict[str, Any]) -> str:
    """Format one atom's movement flags the way a STRU file writes them."""
    move = atom.get("move")
    if move is None:
        return "-"
    return " ".join("1" if flag else "0" for flag in move)


def _print_table(columns: list[str], rows: list[list[str]]) -> None:
    """Print a right-aligned table whose columns fit their content."""
    widths = [
        max([len(header)] + [len(row[index]) for row in rows])
        for index, header in enumerate(columns)
    ]
    print("  " + " ".join(header.rjust(width) for header, width in zip(columns, widths)))
    for row in rows:
        print("  " + " ".join(value.rjust(width) for value, width in zip(row, widths)))


def _atom_table(atoms: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    """Build the per-atom table from the columns the structure actually has."""
    columns = ["index", "label", "element", "fractional", "cartesian", "wyckoff"]
    optional = [
        name
        for name, present in (
            ("magmom", any(atom["magmom"] is not None for atom in atoms)),
            (
                "move",
                any(
                    atom["move"] is not None and tuple(atom["move"]) != (True, True, True)
                    for atom in atoms
                ),
            ),
            ("velocity", any(atom["velocity"] is not None for atom in atoms)),
        )
        if present
    ]
    columns += optional

    rows = []
    for atom in atoms:
        values = [
            str(atom["index"]),
            str(atom["label"]),
            str(atom["element"]),
            _display_value(atom["fractional"]),
            _display_value(atom["cartesian"]),
            _display_value(atom["wyckoff"]),
        ]
        if "magmom" in optional:
            values.append(_moment_text(atom))
        if "move" in optional:
            values.append(_move_text(atom))
        if "velocity" in optional:
            values.append(_display_value(atom["velocity"]))
        rows.append(values)

    return columns, rows


def _inequivalent_table(positions: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    """Build the table of symmetry-inequivalent positions."""
    columns = ["representative", "element", "wyckoff", "fractional", "equivalent atoms"]
    rows = [
        [
            str(position["representative_index"]),
            str(position["element"]),
            str(position["wyckoff"] or "-"),
            _display_value(position["fractional"]),
            _display_value(position["equivalent_indices"]),
        ]
        for position in positions
    ]
    return columns, rows


def _print_report(result: dict[str, Any]) -> None:
    cell = result["cell"]
    symmetry = result["symmetry"]
    print(f"file: {result['file']}")
    print(f"atoms: {result['natoms']} ({result['formula']})")
    print("element counts: " + ", ".join(f"{key}={value}" for key, value in result["element_counts"].items()))
    print("label counts: " + ", ".join(f"{key}={value}" for key, value in result["label_counts"].items()))
    print("cell vectors (Angstrom):")
    for vector in cell["vectors_angstrom"]:
        print("  " + " ".join(f"{float(value): .8f}" for value in vector))
    print(f"cell lengths (Angstrom): {_display_value(cell['lengths_angstrom'])}")
    print(f"cell angles (degree): {_display_value(cell['angles_degree'])}")
    print(f"cell volume (Angstrom^3): {_display_value(cell['volume_angstrom3'])}")
    if symmetry.get("available"):
        print(
            "symmetry: "
            f"{symmetry['space_group_symbol']} (No. {symmetry['space_group_number']}), "
            f"point group {symmetry['point_group']}, {symmetry['crystal_system']}"
        )
        print(f"symmetry operations: {symmetry['symmetry_operations']}")
    else:
        print(f"symmetry: unavailable ({symmetry.get('error', 'unknown reason')})")
    print("symmetry-inequivalent positions:")
    _print_table(*_inequivalent_table(result["inequivalent_positions"]))
    print("resources:")
    _print_table(
        ["label", "pseudopotential", "orbital"],
        [
            [
                str(label),
                _display_value(result["resources"]["pseudopotentials"].get(label)),
                _display_value(result["resources"]["orbitals"].get(label)),
            ]
            for label in result["label_counts"]
        ],
    )
    print("atoms:")
    _print_table(*_atom_table(result["atoms"]))


def run(args: argparse.Namespace) -> int:
    """Read a structure and print its basic information."""
    result = structure_information(
        args.filename,
        input_format=args.input_format,
        cell=args.cell,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_report(result)
    return 0
