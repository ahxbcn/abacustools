"""Implementation of the ``abacustools file interface`` command."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from abacustools.data.interface import (
    InterfaceCandidate,
    build_interface,
    interface_matches,
)
from abacustools.io.stru import AbacusSTRU


def _structure_file(value: str) -> Path:
    """Return an existing structure file or raise a parser error."""
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"structure file does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the ``file interface`` parser and its arguments."""
    parser = subparsers.add_parser(
        "interface",
        help="Build a lattice-matched heterojunction interface of two structures.",
    )
    parser.add_argument(
        "film", type=_structure_file, metavar="FILM",
        help="Structure the film is cut from; it is placed on top.",
    )
    parser.add_argument(
        "substrate", type=_structure_file, metavar="SUBSTRATE",
        help="Structure the substrate is cut from.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, metavar="OUTPUT",
        help="Interface file to write.",
    )
    parser.add_argument(
        "--input-format", default=None,
        help="Format of both input files (STRU, POSCAR, CIF, XYZ, EXTXYZ, or XSF); inferred by default.",
    )
    parser.add_argument(
        "--output-format", default=None,
        help="Output format; inferred from OUTPUT when omitted.",
    )
    parser.add_argument(
        "--film-miller", type=int, nargs=3, default=[0, 0, 1], metavar=("H", "K", "L"),
        help="Miller indices of the film surface, default: 0 0 1.",
    )
    parser.add_argument(
        "--substrate-miller", type=int, nargs=3, default=[0, 0, 1], metavar=("H", "K", "L"),
        help="Miller indices of the substrate surface, default: 0 0 1.",
    )
    parser.add_argument(
        "--film-thickness", type=float, default=1.0,
        help="Film thickness in layers, default: 1.",
    )
    parser.add_argument(
        "--substrate-thickness", type=float, default=1.0,
        help="Substrate thickness in layers, default: 1.",
    )
    parser.add_argument(
        "--in-angstrom", action="store_true",
        help="Interpret the two thicknesses in Angstrom instead of layers.",
    )
    parser.add_argument(
        "-g", "--gap", type=float, default=2.0,
        help="Distance between the film and the substrate in Angstrom, default: 2.",
    )
    parser.add_argument(
        "--vacuum", type=float, default=15.0,
        help="Vacuum above the film in Angstrom, default: 15.",
    )
    parser.add_argument(
        "--termination", type=int, default=0,
        help="Which surface termination pair to use, default: 0 (the first).",
    )
    parser.add_argument(
        "--max-strain", type=float, default=0.03,
        help="Largest relative length mismatch accepted by the lattice match, default: 0.03 (3%%).",
    )
    parser.add_argument(
        "--max-angle", type=float, default=0.6, metavar="DEGREES",
        help="Largest angle mismatch accepted by the lattice match, default: 0.6.",
    )
    parser.add_argument(
        "--max-area", type=float, default=200.0, metavar="ANGSTROM^2",
        help="Largest supercell area of the lattice match, default: 200.",
    )
    parser.add_argument(
        "--max-atoms", type=int, default=None, metavar="N",
        help="Skip lattice matches whose interface has more than N atoms.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List the lattice matches instead of writing an interface.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace OUTPUT when it already exists.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.set_defaults(handler=run)


def _read_structure(path: Path, input_format: str | None) -> AbacusSTRU:
    """Read one input structure."""
    structure = AbacusSTRU.read(str(path), fmt=input_format)
    if structure is None:
        raise RuntimeError(f"failed to read structure: {path}")
    return structure


def _formula(structure: AbacusSTRU) -> str:
    """Return a composition string such as ``Si2O1``."""
    counts = Counter(atom.element or atom.label for atom in structure.atoms)
    return "".join(f"{element}{count}" for element, count in counts.items())


def _match_row(candidate: InterfaceCandidate) -> list[str]:
    """Format one lattice match as a table row."""
    strain = " ".join(f"{value * 100:+.2f}%" for value in candidate.length_strain)
    return [
        str(candidate.index),
        f"{candidate.area:.2f}",
        strain,
        f"{candidate.angle_mismatch:+.3f}",
        f"{candidate.film_cells}/{candidate.substrate_cells}",
    ]


def _print_matches(
    film_label: str,
    substrate_label: str,
    candidates: list[InterfaceCandidate],
    as_json: bool,
) -> None:
    """Print the lattice matches of two surfaces."""
    if as_json:
        print(
            json.dumps(
                {
                    "film": film_label,
                    "substrate": substrate_label,
                    "matches": [candidate.__dict__ for candidate in candidates],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(f"  interface: {film_label} on {substrate_label}")
    print(f"  matches: {len(candidates)}")
    print("  index      area(Angstrom^2)  length strain      angle(deg)  cells")
    for candidate in candidates:
        index, area, strain, angle, cells = _match_row(candidate)
        print(f"  {index:>5}  {area:>17}  {strain:<18}{angle:>10}  {cells}")


def _print_result(payload: dict[str, Any]) -> None:
    """Print the report of a built interface."""
    candidate = payload
    print(f"  interface: {payload['film']} on {payload['substrate']}")
    print(
        f"  film: {payload['film_formula']} ({' '.join(map(str, payload['film_miller']))}), "
        f"{payload['film_thickness']:g} {'Angstrom' if payload['in_angstrom'] else 'layers'}"
    )
    print(
        f"  substrate: {payload['substrate_formula']} "
        f"({' '.join(map(str, payload['substrate_miller']))}), "
        f"{payload['substrate_thickness']:g} {'Angstrom' if payload['in_angstrom'] else 'layers'}"
    )
    termination = "/".join(payload["termination"])
    print(
        f"  termination: {payload['termination_index'] + 1} of {payload['terminations']}: "
        f"{termination}"
    )
    strain = " ".join(f"{value * 100:+.2f}%" for value in candidate["length_strain"])
    print(
        f"  lattice match: #{candidate['selected_match'] + 1} of {candidate['matches']}, "
        f"area {candidate['match_area']:.2f} Angstrom^2, "
        f"length strain {strain}, angle mismatch {candidate['angle_mismatch_degree']:+.3f} deg"
    )
    print(f"  gap: {candidate['gap']:g} Angstrom, vacuum: {candidate['vacuum']:g} Angstrom")
    print(f"  atoms: {candidate['natoms']}")
    lengths = " ".join(f"{value:.6f}" for value in candidate["cell_lengths"])
    angles = " ".join(f"{value:.3f}" for value in candidate["cell_angles"])
    print(f"  cell: {lengths} Angstrom, angles {angles} deg")
    print("  resources:")
    for element, choice in candidate["resources"].items():
        files = " ".join(
            str(value) for value in (choice["pseudopotential"], choice["orbital"]) if value
        )
        print(f"    {element}: {files or '-'} ({choice['reason']})")
    print(f"  output: {candidate['output']}")


def run(args: argparse.Namespace) -> int:
    """Build a heterojunction interface, or list its lattice matches."""
    film = _read_structure(args.film, args.input_format)
    substrate = _read_structure(args.substrate, args.input_format)
    film_label = f"{_formula(film)} ({' '.join(map(str, args.film_miller))})"
    substrate_label = f"{_formula(substrate)} ({' '.join(map(str, args.substrate_miller))})"

    if args.list:
        candidates = interface_matches(
            film,
            substrate,
            film_miller=args.film_miller,
            substrate_miller=args.substrate_miller,
            max_strain=args.max_strain,
            max_angle=args.max_angle,
            max_area=args.max_area,
        )
        _print_matches(film_label, substrate_label, candidates, args.json)
        return 0

    if args.output is None:
        raise ValueError("--output is required unless --list is used")

    result = build_interface(
        film,
        substrate,
        film_miller=args.film_miller,
        substrate_miller=args.substrate_miller,
        film_thickness=args.film_thickness,
        substrate_thickness=args.substrate_thickness,
        in_layers=not args.in_angstrom,
        gap=args.gap,
        vacuum=args.vacuum,
        termination=args.termination,
        max_strain=args.max_strain,
        max_angle=args.max_angle,
        max_area=args.max_area,
        max_atoms=args.max_atoms,
    )

    output = Path(args.output).expanduser()
    if output.exists() and not args.override:
        raise RuntimeError(f"output already exists: {output}; use --override to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not result.structure.write(str(output), fmt=args.output_format):
        raise RuntimeError(f"failed to write structure: {output}")

    parameters = result.structure.get_cell_param()
    payload: dict[str, Any] = {
        "film": str(args.film),
        "substrate": str(args.substrate),
        "film_formula": _formula(film),
        "substrate_formula": _formula(substrate),
        "film_miller": list(result.film_miller),
        "substrate_miller": list(result.substrate_miller),
        "film_thickness": args.film_thickness,
        "substrate_thickness": args.substrate_thickness,
        "in_angstrom": bool(args.in_angstrom),
        "termination": list(result.termination),
        "termination_index": result.termination_index,
        "terminations": result.terminations,
        "matches": result.matches,
        "selected_match": result.candidate.index,
        "match_area": result.candidate.area,
        "length_strain": result.candidate.length_strain,
        "angle_mismatch_degree": result.candidate.angle_mismatch,
        "film_cells": result.candidate.film_cells,
        "substrate_cells": result.candidate.substrate_cells,
        "gap": result.gap,
        "vacuum": result.vacuum,
        "natoms": result.structure.natoms,
        "cell_lengths": [float(value) for value in parameters[:3]],
        "cell_angles": [float(value) for value in parameters[3:]],
        "resources": {
            element: {
                "pseudopotential": pseudo,
                "orbital": orbital,
                "reason": reason,
            }
            for element, (pseudo, orbital, reason) in result.resources.items()
        },
        "output": str(output),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_result(payload)
    return 0
