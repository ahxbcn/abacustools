"""Implementation of the ``abacustools file editstru`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.data.structure import (
    build_slab,
    find_conventional,
    find_primitive,
    fix_atoms,
    generate_all_slabs,
    make_supercell,
    select_atoms,
    set_coordinate_mode,
    standardize_cell,
    symmetrize_structure,
    with_vacuum,
)
from abacustools.data.symmetry import crystallographic_symmetry
from abacustools.io.stru import AbacusSTRU


def _structure_file(value: str) -> Path:
    """Return an existing structure file or raise a parser error."""
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"structure file does not exist: {value}")
    return path


def _positive_int(value: str) -> int:
    """Return a positive integer or raise a parser error."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an integer: {value}") from error
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def _positive_float(value: str) -> float:
    """Return a positive float or raise a parser error."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a number: {value}") from error
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def _non_negative_float(value: str) -> float:
    """Return a non-negative float or raise a parser error."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a number: {value}") from error
    if number < 0:
        raise argparse.ArgumentTypeError(f"must not be negative: {value}")
    return number


def _fraction(value: str) -> float:
    """Return a fraction between 0 and 1 or raise a parser error."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a number: {value}") from error
    if number < 0 or number > 1:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1: {value}")
    return number


def _add_common_arguments(parser: argparse.ArgumentParser, action: str) -> None:
    """Register the arguments every edit action shares."""
    parser.add_argument("filename", type=_structure_file, metavar="STRUCTURE")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        metavar="OUTPUT",
        help="New structure file to write.",
    )
    parser.add_argument(
        "--input-format",
        default=None,
        help="Input format (STRU, POSCAR, CIF, XYZ, EXTXYZ, or XSF); inferred by default.",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help="Output format; inferred from OUTPUT when omitted.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace OUTPUT when it already exists.",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    parser.set_defaults(handler=_run_action, action=action)


def _add_selection_arguments(
    parser: argparse.ArgumentParser, *, allow_remove: bool = False
) -> None:
    """Register the shared atom-selection arguments."""
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        metavar="INDEX",
        help="One-based atom indices, as in the other abacustools commands.",
    )
    parser.add_argument("--elements", nargs="+", metavar="ELEMENT", help="Element symbols.")
    parser.add_argument(
        "--coords",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Inclusive coordinate window used together with --direction.",
    )
    parser.add_argument(
        "--direction",
        default="c",
        choices=("a", "b", "c", "x", "y", "z"),
        help="Direction of the coordinate window, default: c.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Interpret --coords as fractional coordinates; Cartesian by default.",
    )
    if allow_remove:
        parser.add_argument(
            "--remove",
            action="store_true",
            help="Drop the selected atoms instead of keeping them.",
        )


def _selection(args: argparse.Namespace) -> dict[str, Any]:
    """Turn the parsed selection arguments into keyword arguments."""
    indices = None
    if args.indices is not None:
        indices = [index - 1 for index in args.indices]
    return {
        "indices": indices,
        "elements": args.elements,
        "coordinate_range": args.coords,
        "direction": args.direction,
        "cartesian": not args.direct,
    }


def _register_supercell(subparsers) -> None:
    """Register ``file editstru supercell``."""
    parser = subparsers.add_parser(
        "supercell",
        help="Replicate the structure along the three lattice vectors.",
    )
    _add_common_arguments(parser, "supercell")
    parser.add_argument(
        "-n", "--repeat",
        type=_positive_int,
        nargs=3,
        required=True,
        metavar=("A", "B", "C"),
        help="Positive repetitions along a, b and c.",
    )


def _register_vacuum(subparsers) -> None:
    """Register ``file editstru vacuum``."""
    parser = subparsers.add_parser(
        "vacuum",
        help="Add vacuum along one lattice direction.",
    )
    _add_common_arguments(parser, "vacuum")
    parser.add_argument(
        "-t", "--thickness",
        type=_positive_float,
        required=True,
        metavar="ANGSTROM",
        help="Vacuum to add in Angstrom.",
    )
    parser.add_argument(
        "--direction",
        default="c",
        choices=("a", "b", "c", "x", "y", "z"),
        help="Lattice direction to extend, default: c.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Move the atoms to the middle of the extended cell.",
    )


def _register_select(subparsers) -> None:
    """Register ``file editstru select``."""
    parser = subparsers.add_parser(
        "select",
        help="Keep or drop the atoms matching the given filters.",
    )
    _add_common_arguments(parser, "select")
    _add_selection_arguments(parser, allow_remove=True)


def _register_fix(subparsers) -> None:
    """Register ``file editstru fix``."""
    parser = subparsers.add_parser(
        "fix",
        help="Set the movement constraints of the selected atoms.",
    )
    _add_common_arguments(parser, "fix")
    _add_selection_arguments(parser)
    parser.add_argument(
        "--move",
        nargs="+",
        choices=("x", "y", "z"),
        default=(),
        metavar="AXIS",
        help="Axes the selected atoms may still move along; fully fixed by default.",
    )
    parser.add_argument(
        "--free-others",
        action="store_true",
        help="Allow every unselected atom to move.",
    )


def _register_slab(subparsers) -> None:
    """Register ``file editstru slab``."""
    parser = subparsers.add_parser(
        "slab",
        help="Cut a surface slab out of a bulk structure.",
    )
    _add_common_arguments(parser, "slab")
    parser.add_argument(
        "--miller",
        type=int,
        nargs=3,
        default=(1, 0, 0),
        metavar=("H", "K", "L"),
        help="Miller indices of the surface, default: 1 0 0.",
    )
    parser.add_argument(
        "--layers",
        type=_positive_int,
        default=3,
        metavar="N",
        help="Repeating units along the surface normal, default: 3.",
    )
    parser.add_argument(
        "--surface-supercell",
        type=_positive_int,
        nargs=2,
        default=(1, 1),
        metavar=("A", "B"),
        help="Repetitions along the two in-plane directions, default: 1 1.",
    )
    parser.add_argument(
        "--vacuum",
        type=_non_negative_float,
        default=15.0,
        metavar="ANGSTROM",
        help="Empty space between the slab and its periodic image, default: 15.",
    )
    parser.add_argument(
        "--vacuum-direction",
        default="c",
        choices=("a", "b", "c"),
        help="Lattice direction that receives the vacuum, default: c.",
    )
    parser.add_argument(
        "--fix",
        type=_fraction,
        nargs="?",
        const=0.5,
        default=None,
        metavar="FRACTION",
        help=(
            "Fix the atoms in the lower part of the slab along the vacuum "
            "direction and free the rest; FRACTION of the slab thickness is "
            "fixed, 0.5 (the bottom half) by default."
        ),
    )


def _register_coordinate_actions(subparsers) -> None:
    """Register ``file editstru direct`` and ``file editstru cartesian``."""
    for name, help_text in (
        ("direct", "Write the atomic positions as fractional (direct) coordinates."),
        ("cartesian", "Write the atomic positions as Cartesian coordinates."),
    ):
        parser = subparsers.add_parser(name, help=help_text)
        _add_common_arguments(parser, name)


def _register_primitive(subparsers) -> None:
    """Register ``file editstru primitive``."""
    parser = subparsers.add_parser(
        "primitive",
        help="Reduce the structure to its primitive cell.",
    )
    _add_common_arguments(parser, "primitive")
    parser.add_argument(
        "--symprec",
        type=_non_negative_float,
        default=1e-5,
        metavar="ANGSTROM",
        help="Symmetry tolerance in Angstrom, default: 1e-5.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=_non_negative_float,
        default=5.0,
        metavar="DEGREES",
        help="Angle tolerance in degrees, default: 5.",
    )


def _register_standardize(subparsers) -> None:
    """Register ``file editstru standardize``."""
    parser = subparsers.add_parser(
        "standardize",
        help="Standardize the cell according to crystallographic conventions.",
    )
    _add_common_arguments(parser, "standardize")
    parser.add_argument(
        "--to-primitive",
        action="store_true",
        help="Also reduce to primitive cell.",
    )
    parser.add_argument(
        "--no-idealize",
        action="store_true",
        help="Do not idealize the cell (keep small distortions).",
    )
    parser.add_argument(
        "--symprec",
        type=_non_negative_float,
        default=1e-5,
        metavar="ANGSTROM",
        help="Symmetry tolerance in Angstrom, default: 1e-5.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=_non_negative_float,
        default=5.0,
        metavar="DEGREES",
        help="Angle tolerance in degrees, default: 5.",
    )


def _register_conventional(subparsers) -> None:
    """Register ``file editstru conventional``."""
    parser = subparsers.add_parser(
        "conventional",
        help="Find the conventional (standard) cell of the structure.",
    )
    _add_common_arguments(parser, "conventional")
    parser.add_argument(
        "--symprec",
        type=_non_negative_float,
        default=1e-5,
        metavar="ANGSTROM",
        help="Symmetry tolerance in Angstrom, default: 1e-5.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=_non_negative_float,
        default=5.0,
        metavar="DEGREES",
        help="Angle tolerance in degrees, default: 5.",
    )




def _register_symmetrize(subparsers) -> None:
    """Register ``file editstru symmetrize``."""
    parser = subparsers.add_parser(
        "symmetrize",
        help="Remove small numerical errors and make the symmetry of the structure exact.",
    )
    _add_common_arguments(parser, "symmetrize")
    parser.add_argument(
        "--symprec",
        type=_non_negative_float,
        default=1e-5,
        metavar="ANGSTROM",
        help=(
            "Distance tolerance in Angstrom, default: 1e-5. Deviations larger than "
            "this are not treated as noise, so raise it to clean a rougher file."
        ),
    )
    parser.add_argument(
        "--angle-tolerance",
        type=_non_negative_float,
        default=5.0,
        metavar="DEGREES",
        help="Angle tolerance in degrees, default: 5.",
    )
    parser.add_argument(
        "--keep-cell",
        action="store_true",
        help="Idealize the atomic positions only and leave the lattice as it is.",
    )


def _register_all_slabs(subparsers) -> None:
    """Register ``file editstru all-slabs``."""
    parser = subparsers.add_parser(
        "all-slabs",
        help="Generate all possible surface terminations for given Miller indices.",
    )
    # Add arguments manually without -o (output is generated automatically)
    parser.add_argument("filename", type=_structure_file, metavar="STRUCTURE")
    parser.add_argument(
        "--input-format",
        default=None,
        help="Input format (STRU, POSCAR, CIF, XYZ, EXTXYZ, or XSF); inferred by default.",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help="Output format; inferred from output file when omitted.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace output files when they already exist.",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    parser.set_defaults(handler=_run_action, action="all-slabs")
    parser.add_argument(
        "--miller",
        type=int,
        nargs=3,
        required=True,
        metavar=("H", "K", "L"),
        help="Miller indices of the surface, e.g. 1 1 0.",
    )
    parser.add_argument(
        "--min-slab-size",
        type=_positive_float,
        default=3.0,
        metavar="SIZE",
        help="Minimum slab thickness (in layers or Angstrom), default: 3.0.",
    )
    parser.add_argument(
        "--min-vacuum-size",
        type=_positive_float,
        default=10.0,
        metavar="ANGSTROM",
        help="Minimum vacuum thickness in Angstrom, default: 10.0.",
    )
    parser.add_argument(
        "--in-unit-planes",
        action="store_true",
        help="Interpret min-slab-size as number of unit planes instead of Angstrom.",
    )
    parser.add_argument(
        "--center-slab",
        action="store_true",
        default=True,
        help="Center the slab in the cell (default: True).",
    )
    parser.add_argument(
        "--no-center-slab",
        action="store_false",
        dest="center_slab",
        help="Do not center the slab in the cell.",
    )
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help="Symmetrize the slab structure.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair the slab structure.",
    )
    parser.add_argument(
        "--tol",
        type=_positive_float,
        default=0.1,
        metavar="TOL",
        help="Tolerance for comparing sites, default: 0.1.",
    )
    parser.add_argument(
        "--max-broken-bonds",
        type=int,
        default=0,
        metavar="N",
        help="Maximum number of broken bonds allowed, default: 0.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="slab",
        metavar="PREFIX",
        help="Prefix for output files (suffix _0, _1, ... added), default: slab.",
    )


def register_parser(subparsers) -> None:
    """Register the ``file editstru`` parser and its actions."""
    parser = subparsers.add_parser(
        "editstru",
        help="Edit a structure file and write a new one.",
    )
    actions = parser.add_subparsers(
        dest="editstru_action",
        metavar="ACTION",
        title="editstru actions",
        required=True,
    )
    _register_supercell(actions)
    _register_vacuum(actions)
    _register_slab(actions)
    _register_select(actions)
    _register_fix(actions)
    _register_coordinate_actions(actions)
    _register_primitive(actions)
    _register_standardize(actions)
    _register_conventional(actions)
    _register_symmetrize(actions)
    _register_all_slabs(actions)


def _formula(structure: AbacusSTRU) -> str:
    """Return a composition string such as ``Si2O1``."""
    counts: dict[str, int] = {}
    for element in structure.elements:
        counts[element] = counts.get(element, 0) + 1
    return "".join(f"{element}{count}" for element, count in counts.items())


def _cell_lengths(structure: AbacusSTRU) -> list[float]:
    """Return the three lattice-vector lengths in Angstrom."""
    return [float(value) for value in np.linalg.norm(structure.cell, axis=1)]


def _max_displacement(before: AbacusSTRU, after: AbacusSTRU) -> float:
    """Return the largest periodic atom shift between two structures in Angstrom."""
    delta = np.asarray(after.coords_direct, dtype=float) - np.asarray(
        before.coords_direct, dtype=float
    )
    delta -= np.rint(delta)
    return float(np.max(np.linalg.norm(delta @ np.asarray(after.cell, dtype=float), axis=1)))


def _edited_structure(
    args: argparse.Namespace, structure: AbacusSTRU
) -> AbacusSTRU:
    """Apply the requested action to the structure."""
    if args.action == "supercell":
        return make_supercell(structure, args.repeat)
    if args.action == "vacuum":
        return with_vacuum(
            structure, args.thickness, direction=args.direction, center=args.center
        )
    if args.action == "slab":
        return build_slab(
            structure,
            miller_indices=args.miller,
            layers=args.layers,
            surface_supercell=args.surface_supercell,
            vacuum=args.vacuum,
            vacuum_direction=args.vacuum_direction,
            fix_fraction=args.fix,
        )
    if args.action == "select":
        return select_atoms(structure, remove=args.remove, **_selection(args))
    if args.action == "fix":
        move = tuple(axis in set(args.move) for axis in ("x", "y", "z"))
        return fix_atoms(
            structure, move=move, free_others=args.free_others, **_selection(args)
        )
    if args.action in {"direct", "cartesian"}:
        return set_coordinate_mode(structure, args.action)
    if args.action == "primitive":
        return find_primitive(
            structure,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
        )
    if args.action == "standardize":
        return standardize_cell(
            structure,
            to_primitive=args.to_primitive,
            no_idealize=args.no_idealize,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
        )
    if args.action == "conventional":
        return find_conventional(
            structure,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
        )
    if args.action == "symmetrize":
        return symmetrize_structure(
            structure,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            keep_cell=args.keep_cell,
        )
    if args.action == "all-slabs":
        # This is handled specially in _run_action
        raise RuntimeError("all-slabs is handled separately")
    raise RuntimeError(f"unknown editstru action: {args.action}")


def _run_action(args: argparse.Namespace) -> int:
    """Run one ``file editstru`` action."""
    structure = AbacusSTRU.read(str(args.filename), fmt=args.input_format)
    if structure is None:
        raise RuntimeError(f"failed to read structure: {args.filename}")
    
    # Special handling for all-slabs
    if args.action == "all-slabs":
        slabs = generate_all_slabs(
            structure,
            args.miller,
            min_slab_size=args.min_slab_size,
            min_vacuum_size=args.min_vacuum_size,
            in_unit_planes=args.in_unit_planes,
            center_slab=args.center_slab,
            symmetrize=args.symmetrize,
            repair=args.repair,
            tol=args.tol,
            max_broken_bonds=args.max_broken_bonds,
        )
        
        # Write all slabs
        output_prefix = Path(args.output_prefix)
        output_dir = output_prefix.parent if output_prefix.parent != Path(".") else Path(".")
        output_base = output_prefix.name
        
        written_files = []
        for i, slab in enumerate(slabs):
            output = output_dir / f"{output_base}_{i}.STRU"
            if output.exists() and not args.override:
                raise RuntimeError(
                    f"output already exists: {output}; use --override to replace it"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            if not slab.write(str(output), fmt=args.output_format):
                raise RuntimeError(f"failed to write structure: {output}")
            written_files.append(str(output))
        
        if args.json:
            payload = {
                "action": "all-slabs",
                "input": str(args.filename),
                "miller_indices": args.miller,
                "num_slabs": len(slabs),
                "output_files": written_files,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"  all-slabs: {args.filename}")
            print(f"  Miller indices: {tuple(args.miller)}")
            print(f"  Generated {len(slabs)} slab(s):")
            for i, f in enumerate(written_files):
                print(f"    [{i}] {f}")
        return 0
    
    edited = _edited_structure(args, structure)

    symmetry = None
    if args.action == "symmetrize":
        symmetry = crystallographic_symmetry(
            edited, symprec=args.symprec, angle_tolerance=args.angle_tolerance
        )

    output = Path(args.output).expanduser()
    if output.exists() and not args.override:
        raise RuntimeError(
            f"output already exists: {output}; use --override to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not edited.write(str(output), fmt=args.output_format):
        raise RuntimeError(f"failed to write structure: {output}")

    payload: dict[str, Any] = {
        "action": args.action,
        "input": str(args.filename),
        "output": str(output),
        "atoms_before": structure.natoms,
        "atoms_after": edited.natoms,
        "formula": _formula(edited),
        "cell_lengths": _cell_lengths(edited),
        "fixed_atoms": sum(
            1 for atom in edited.atoms if tuple(atom.move or ()) == (False, False, False)
        ),
    }
    if args.action in {"direct", "cartesian"}:
        payload["coordinates"] = args.action
    if args.action == "symmetrize":
        payload["space_group"] = symmetry.get("space_group_symbol")
        payload["space_group_number"] = symmetry.get("space_group_number")
        payload["max_displacement_angstrom"] = _max_displacement(structure, edited)
        payload["cell_idealized"] = not args.keep_cell
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"  {args.action}: {args.filename} -> {output}")
    print(f"  atoms: {payload['atoms_before']} -> {payload['atoms_after']}")
    print(f"  formula: {payload['formula']}")
    if args.action in {"direct", "cartesian"}:
        print(f"  coordinates: {args.action}")
    if args.action == "symmetrize":
        number = payload["space_group_number"]
        symbol = payload["space_group"] or "unavailable"
        label = symbol if number is None else f"{symbol} (No. {number})"
        print(f"  symmetry: {label}")
        print(f"  max displacement: {payload['max_displacement_angstrom']:.6g} Angstrom")
        if not payload["cell_idealized"]:
            print("  lattice: kept")
        if number == 1:
            print(
                "  note: no symmetry beyond P1 was found; "
                "raise --symprec if the file should be more symmetric"
            )
    if payload["fixed_atoms"]:
        print(f"  fixed atoms: {payload['fixed_atoms']} of {payload['atoms_after']}")
    print(
        "  cell: "
        + " ".join(f"{value:.6f}" for value in payload["cell_lengths"])
        + " Angstrom"
    )
    return 0


__all__ = ["register_parser"]
