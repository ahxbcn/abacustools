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
                "pseudopotential": atom.pp,
                "orbital": atom.orb,
                "paw": atom.paw,
                "move": list(atom.move) if atom.move is not None else None,
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
            "paw": _resource_by_label(structure, "paw"),
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
    print("  representative equivalent atoms element wyckoff fractional")
    for position in result["inequivalent_positions"]:
        print(
            f"  {position['representative_index']:>13} "
            f"{_display_value(position['equivalent_indices']):>17} "
            f"{position['element']:>7} {position['wyckoff'] or '-':>7} "
            f"{_display_value(position['fractional']):>28}"
        )
    print("atoms:")
    print("  index label element fractional cartesian wyckoff pseudopotential orbital paw")
    for atom in result["atoms"]:
        print(
            f"  {atom['index']:5d} {atom['label']:>5} {atom['element']:>7} "
            f"{_display_value(atom['fractional']):>28} "
            f"{_display_value(atom['cartesian']):>28} "
            f"{_display_value(atom['wyckoff']):>8} "
            f"{_display_value(atom['pseudopotential']):>18} "
            f"{_display_value(atom['orbital']):>18} "
            f"{_display_value(atom['paw']):>18}"
        )


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
