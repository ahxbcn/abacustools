"""Display crystallographic and ABACUS metadata for a structure file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from abacustools.data.coordination import coordination_analysis
from abacustools.data.composition import composition_summary
from abacustools.data.dimensionality import classify_dimensionality
from abacustools.data.structure_summary import structure_summary
from abacustools.data.symmetry import (
    crystallographic_symmetry,
    layer_symmetry,
    magnetic_ordering,
    magnetic_symmetry,
    site_symmetry_symbols,
)
from abacustools.io.stru import AbacusSTRU, normalize_structure_format


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
        help="Show structure information; several files are listed as a table.",
    )
    parser.add_argument(
        "filename",
        type=_structure_file,
        nargs="+",
        metavar="STRUCTURE",
        help="Structure file; several files are listed as a summary table.",
    )
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
    parser.add_argument(
        "--layer-direction",
        choices=["a", "b", "c"],
        default=None,
        help="Aperiodic direction of a slab used for the layer group, detected from the vacuum by default.",
    )
    parser.add_argument(
        "--coordination",
        nargs="?",
        const="auto",
        choices=["auto", "crystalnn", "chemenv", "voronoi", "minimum-distance"],
        default=None,
        metavar="METHOD",
        help=(
            "Analyse coordination numbers and coordination environments; METHOD defaults to auto, "
            "which follows the dimensionality of the structure."
        ),
    )
    parser.add_argument(
        "--min-vacuum",
        type=float,
        default=5.0,
        help="Empty span in Angstrom that counts as vacuum, default: 5.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="List the main fields instead of the full report, also for a single structure.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a formatted report.")
    parser.set_defaults(handler=run)


def _validate_inputs(symprec: float, angle_tolerance: float, min_vacuum: float) -> None:
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be a positive finite number")
    if not np.isfinite(angle_tolerance) or angle_tolerance < 0:
        raise ValueError("angle-tolerance must be a non-negative finite number")
    if not np.isfinite(min_vacuum) or min_vacuum <= 0:
        raise ValueError("min-vacuum must be a positive finite number")


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


def structure_information(
    filename: Path,
    *,
    input_format: Optional[str] = None,
    cell: Optional[list[float]] = None,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
    layer_direction: Optional[str] = None,
    coordination: Optional[str] = None,
    min_vacuum: float = 5.0,
) -> dict[str, Any]:
    """Read a structure and return JSON-compatible basic information."""
    _validate_inputs(symprec, angle_tolerance, min_vacuum)
    resolved_format = normalize_structure_format(input_format, str(filename))
    structure = AbacusSTRU.read(
        filename,
        fmt=resolved_format,
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
    symmetry = crystallographic_symmetry(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    wyckoffs = symmetry.get("wyckoff_positions", [None] * structure.natoms)
    multiplicities = symmetry.get("wyckoff_multiplicities") or [None] * structure.natoms
    site_symmetries = site_symmetry_symbols(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    ) or [None] * structure.natoms
    magnetic = magnetic_symmetry(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    ordering = magnetic_ordering(structure)
    dimension = classify_dimensionality(structure, min_vacuum=min_vacuum)
    layer = layer_symmetry(
        structure,
        direction=layer_direction,
        symprec=symprec,
        min_vacuum=min_vacuum,
    )
    coordination_result = (
        None
        if coordination is None
        else coordination_analysis(structure, method=coordination, min_vacuum=min_vacuum)
    )

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
                "wyckoff_multiplicity": (
                    multiplicities[index] if index < len(multiplicities) else None
                ),
                "site_symmetry": site_symmetries[index] if index < len(site_symmetries) else None,
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
                "wyckoff_multiplicity": representative["wyckoff_multiplicity"],
                "site_symmetry": representative["site_symmetry"],
            }
        )

    return {
        "file": str(Path(filename).absolute()),
        "format": resolved_format,
        "natoms": structure.natoms,
        "formula": " ".join(f"{element}{count if count != 1 else ''}" for element, count in Counter(elements).items()),
        "element_counts": dict(Counter(elements)),
        "label_counts": dict(Counter(labels)),
        "composition": composition_summary(structure),
        "cell": {
            "vectors_angstrom": _round_values(cell_vectors),
            "volume_angstrom3": volume,
            "lengths_angstrom": None if lengths_angles is None else _round_values(lengths_angles[:3]),
            "angles_degree": None if lengths_angles is None else _round_values(lengths_angles[3:]),
            "periodic": volume is not None and volume > 1e-12,
        },
        # The pseudopotential and orbital file names live in the ATOMIC_SPECIES
        # and NUMERICAL_ORBITAL blocks, which only the ABACUS STRU format has.
        "resources": (
            {
                "pseudopotentials": _resource_by_label(structure, "pp"),
                "orbitals": _resource_by_label(structure, "orb"),
            }
            if resolved_format == "stru"
            else None
        ),
        "symmetry": symmetry,
        "magnetic_symmetry": magnetic,
        "magnetic_ordering": ordering,
        "layer_symmetry": layer,
        "dimensionality": dimension,
        "coordination": coordination_result,
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


def _wyckoff_text(entry: dict[str, Any]) -> str:
    """Format a Wyckoff position the way the International Tables write it."""
    letter = entry.get("wyckoff")
    if not letter:
        return "-"
    multiplicity = entry.get("wyckoff_multiplicity")
    return f"{multiplicity}{letter}" if multiplicity else str(letter)


def _print_table(columns: list[str], rows: list[list[str]]) -> None:
    """Print a right-aligned table whose columns fit their content."""
    widths = [
        max([len(header)] + [len(row[index]) for row in rows])
        for index, header in enumerate(columns)
    ]
    print("  " + " ".join(header.rjust(width) for header, width in zip(columns, widths)))
    for row in rows:
        print("  " + " ".join(value.rjust(width) for value, width in zip(row, widths)))


def _dimensionality_text(dimension: dict[str, Any]) -> str:
    """Describe the dimensionality of a structure in one line."""
    if not dimension["periodic"]:
        return f"{dimension['label']} (no periodic cell)"
    if not dimension["vacuum_directions"]:
        return f"{dimension['label']}, no vacuum direction"
    gaps = {gap["direction"]: gap["thickness"] for gap in dimension["gaps"]}
    spans = ", ".join(f"{name} = {gaps[name]:.6f}" for name in dimension["vacuum_directions"])
    return f"{dimension['label']}, vacuum along {spans} Angstrom"


def _yes_no(value: Optional[bool]) -> str:
    """Format a symmetry flag."""
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _composition_lines(composition: dict[str, Any]) -> list[str]:
    """Describe the formula unit and prototype of a structure."""
    if not composition["available"]:
        return [f"composition: unavailable ({composition.get('error', 'unknown reason')})"]
    lines = []
    if composition["formula_unit"]:
        units = composition["formula_units_per_cell"]
        suffix = f", {units} per cell" if units else ""
        lines.append(f"formula unit: {composition['formula_unit']}{suffix}")
    if composition["prototype"]:
        lines.append(f"prototype: {composition['prototype']}")
    return lines


def _coordination_lines(coordination: dict[str, Any]) -> list[str]:
    """Describe the coordination analysis and how its method was chosen."""
    if not coordination["available"]:
        return [f"coordination: unavailable ({coordination.get('error', 'unknown reason')})"]
    lines = [
        f"coordination: {coordination['method']} "
        f"for a {coordination['dimensionality']['label']}"
    ]
    lines += [f"  note: {note}" for note in coordination["notes"]]
    return lines


def _coordination_by_index(coordination: Optional[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return the coordination of every atom, keyed by its index."""
    if not coordination or not coordination.get("available"):
        return {}
    return {site["index"]: site for site in coordination["sites"]}


def _atom_table(
    atoms: list[dict[str, Any]],
    coordination: Optional[dict[str, Any]] = None,
) -> tuple[list[str], list[list[str]]]:
    """Build the per-atom table from the columns the structure actually has."""
    columns = ["index", "label", "element", "fractional", "cartesian", "wyckoff", "site_sym"]
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
    sites = _coordination_by_index(coordination)
    if sites:
        columns.append("cn")
        if any(site["geometry"] for site in sites.values()):
            columns.append("geometry")

    rows = []
    for atom in atoms:
        values = [
            str(atom["index"]),
            str(atom["label"]),
            str(atom["element"]),
            _display_value(atom["fractional"]),
            _display_value(atom["cartesian"]),
            _wyckoff_text(atom),
            _display_value(atom["site_symmetry"]),
        ]
        if "magmom" in optional:
            values.append(_moment_text(atom))
        if "move" in optional:
            values.append(_move_text(atom))
        if "velocity" in optional:
            values.append(_display_value(atom["velocity"]))
        site = sites.get(atom["index"])
        if "cn" in columns:
            values.append("-" if site is None else _display_value(site["coordination_number"]))
        if "geometry" in columns:
            values.append("-" if site is None else _display_value(site["geometry"]))
        rows.append(values)

    return columns, rows


def _inequivalent_table(positions: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    """Build the table of symmetry-inequivalent positions."""
    columns = ["representative", "element", "wyckoff", "site_sym", "fractional", "equivalent atoms"]
    rows = [
        [
            str(position["representative_index"]),
            str(position["element"]),
            _wyckoff_text(position),
            _display_value(position["site_symmetry"]),
            _display_value(position["fractional"]),
            _display_value(position["equivalent_indices"]),
        ]
        for position in positions
    ]
    return columns, rows


def _print_report(result: dict[str, Any]) -> None:
    cell = result["cell"]
    symmetry = result["symmetry"]
    magnetic = result["magnetic_symmetry"]
    ordering = result["magnetic_ordering"]
    layer = result["layer_symmetry"]
    composition = result["composition"]
    print(f"file: {result['file']}")
    print(f"atoms: {result['natoms']} ({result['formula']})")
    for line in _composition_lines(composition):
        print(line)
    print("element counts: " + ", ".join(f"{key}={value}" for key, value in result["element_counts"].items()))
    print("label counts: " + ", ".join(f"{key}={value}" for key, value in result["label_counts"].items()))
    print("cell vectors (Angstrom):")
    for vector in cell["vectors_angstrom"]:
        print("  " + " ".join(f"{float(value): .8f}" for value in vector))
    print(f"cell lengths (Angstrom): {_display_value(cell['lengths_angstrom'])}")
    print(f"cell angles (degree): {_display_value(cell['angles_degree'])}")
    print(f"cell volume (Angstrom^3): {_display_value(cell['volume_angstrom3'])}")
    density = composition.get("density_g_cm3")
    print(f"density (g/cm^3): {'-' if density is None else f'{density:.6f}'}")
    print(f"dimensionality: {_dimensionality_text(result['dimensionality'])}")
    if symmetry.get("available"):
        print(
            "symmetry: "
            f"{symmetry['space_group_symbol']} (No. {symmetry['space_group_number']}), "
            f"{symmetry['crystal_system']}"
        )
        print(f"symmetry operations: {symmetry['symmetry_operations']}")
        schoenflies = symmetry.get("schoenflies")
        print(
            f"point group: {symmetry['point_group']}"
            + (f" ({schoenflies})" if schoenflies else "")
        )
        if symmetry.get("bravais_lattice"):
            pearson = symmetry.get("pearson_symbol")
            suffix = f" (Pearson symbol {pearson})" if pearson else ""
            print(f"Bravais lattice: {symmetry['bravais_lattice']}{suffix}")
        print(f"inversion symmetry: {_yes_no(symmetry.get('inversion_symmetry'))}")
        print(f"polar point group: {_yes_no(symmetry.get('polar_point_group'))}")
        print(
            "symmetry accuracy: "
            f"symprec {symmetry['symprec']:g} Angstrom, "
            f"angle tolerance {symmetry['angle_tolerance']:g} degrees"
        )
    else:
        print(f"symmetry: unavailable ({symmetry.get('error', 'unknown reason')})")
    if magnetic.get("available"):
        label = magnetic["label"] or "-"
        print(
            "magnetic symmetry: "
            f"{label} (BNS {magnetic['bns_number']}, UNI {magnetic['uni_number']}), {magnetic['type']}"
        )
        print(
            "magnetic operations: "
            f"{magnetic['operations']} "
            f"({magnetic['operations_with_time_reversal']} with time reversal)"
        )
    else:
        print(f"magnetic symmetry: unavailable ({magnetic.get('error', 'unknown reason')})")
    if ordering.get("available"):
        print(
            f"magnetic ordering: {ordering['ordering']}, net moment {ordering['net_moment']}, "
            f"{ordering['magnetic_sites']} magnetic sites "
            f"({ordering['unique_magnetic_sites']} unique)"
        )
    if layer.get("available"):
        print(
            f"layer symmetry: {layer['symbol']} (No. {layer['number']}) "
            f"along {layer['direction']}, {layer['operations']} operations"
        )
    elif layer.get("requested_direction"):
        print(f"layer symmetry: unavailable ({layer.get('error', 'unknown reason')})")
    coordination = result.get("coordination")
    if coordination is not None:
        for line in _coordination_lines(coordination):
            print(line)
    # Without symmetry every atom is its own orbit and the table would only
    # repeat the per-atom table, which is the case for P1 and for structures
    # whose symmetry could not be analysed.
    positions = result["inequivalent_positions"]
    if len(positions) != result["natoms"]:
        print("symmetry-inequivalent positions:")
        _print_table(*_inequivalent_table(positions))
    resources = result.get("resources")
    if resources:
        print("resources:")
        _print_table(
            ["label", "pseudopotential", "orbital"],
            [
                [
                    str(label),
                    _display_value(resources["pseudopotentials"].get(label)),
                    _display_value(resources["orbitals"].get(label)),
                ]
                for label in result["label_counts"]
            ],
        )
    print("atoms:")
    _print_table(*_atom_table(result["atoms"], result.get("coordination")))


_SUMMARY_COLUMNS = (
    "file",
    "formula",
    "atoms",
    "space group",
    "crystal system",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "volume",
)


def _number_text(value: Optional[float], digits: int) -> str:
    """Format a cell parameter, or a dash when the structure has none."""
    return "-" if value is None else f"{float(value):.{digits}f}"


def _summary_row(summary: dict[str, Any]) -> list[str]:
    """Turn one structure summary into a row of the batch table."""
    cell = summary["cell"]
    lengths = cell["lengths_angstrom"] or [None, None, None]
    angles = cell["angles_degree"] or [None, None, None]
    space_group = summary["space_group"] or "-"
    if summary["space_group"] and summary["space_group_number"] is not None:
        space_group = f"{space_group} ({summary['space_group_number']})"
    return [
        Path(summary["file"]).name,
        summary["formula"],
        str(summary["natoms"]),
        space_group,
        summary["crystal_system"] or "-",
        *(_number_text(value, 4) for value in lengths),
        *(_number_text(value, 3) for value in angles),
        _number_text(cell["volume_angstrom3"], 3),
    ]


def _print_summary(summaries: list[dict[str, Any]]) -> None:
    """Print one row per structure with its main fields."""
    print("cell lengths in Angstrom, angles in degree, volume in Angstrom^3")
    _print_table(
        list(_SUMMARY_COLUMNS),
        [_summary_row(summary) for summary in summaries],
    )


def run(args: argparse.Namespace) -> int:
    """Read one or more structures and print their information."""
    paths = list(args.filename)
    _validate_inputs(args.symprec, args.angle_tolerance, args.min_vacuum)
    if args.summary or len(paths) > 1:
        if args.coordination is not None or args.layer_direction is not None:
            raise ValueError(
                "the coordination and layer analyses belong to the full report; "
                "run file info on a single structure without --summary"
            )
        summaries = [
            structure_summary(
                path,
                input_format=args.input_format,
                cell=args.cell,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
            )
            for path in paths
        ]
        if args.json:
            print(json.dumps(summaries, indent=2, ensure_ascii=False))
        else:
            _print_summary(summaries)
        return 0
    result = structure_information(
        paths[0],
        input_format=args.input_format,
        cell=args.cell,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        layer_direction=args.layer_direction,
        coordination=args.coordination,
        min_vacuum=args.min_vacuum,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_report(result)
    return 0
