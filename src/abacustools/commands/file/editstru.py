"""Implementation of the ``abacustools file editstru`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.data.structure import (
    fix_atoms,
    make_supercell,
    select_atoms,
    with_vacuum,
)
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
    _register_select(actions)
    _register_fix(actions)


def _formula(structure: AbacusSTRU) -> str:
    """Return a composition string such as ``Si2O1``."""
    counts: dict[str, int] = {}
    for element in structure.elements:
        counts[element] = counts.get(element, 0) + 1
    return "".join(f"{element}{count}" for element, count in counts.items())


def _cell_lengths(structure: AbacusSTRU) -> list[float]:
    """Return the three lattice-vector lengths in Angstrom."""
    return [float(value) for value in np.linalg.norm(structure.cell, axis=1)]


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
    if args.action == "select":
        return select_atoms(structure, remove=args.remove, **_selection(args))
    if args.action == "fix":
        move = tuple(axis in set(args.move) for axis in ("x", "y", "z"))
        return fix_atoms(
            structure, move=move, free_others=args.free_others, **_selection(args)
        )
    raise RuntimeError(f"unknown editstru action: {args.action}")


def _run_action(args: argparse.Namespace) -> int:
    """Run one ``file editstru`` action."""
    structure = AbacusSTRU.read(str(args.filename), fmt=args.input_format)
    if structure is None:
        raise RuntimeError(f"failed to read structure: {args.filename}")
    edited = _edited_structure(args, structure)

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
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"  {args.action}: {args.filename} -> {output}")
    print(f"  atoms: {payload['atoms_before']} -> {payload['atoms_after']}")
    print(f"  formula: {payload['formula']}")
    print(
        "  cell: "
        + " ".join(f"{value:.6f}" for value in payload["cell_lengths"])
        + " Angstrom"
    )
    return 0


__all__ = ["register_parser"]
