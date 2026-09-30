"""Implementation of the ``abacustools file kpt`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.data.kpt import (
    band_path,
    mesh_from_spacing,
    model_report,
    read_kpt,
    spacing_from_mesh,
)
from abacustools.io.abacus import WriteKpt
from abacustools.io.stru import AbacusSTRU


def _existing_file(value: str, what: str):
    """Return an existing file path or raise a parser error."""
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{what} does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the ``file kpt`` parser and its arguments."""
    parser = subparsers.add_parser(
        "kpt",
        help="Inspect a KPT file, or generate a mesh or band path as one.",
    )
    parser.add_argument(
        "filename",
        type=lambda value: _existing_file(value, "KPT file"),
        nargs="?",
        metavar="KPT",
        help="KPT file to inspect.",
    )
    parser.add_argument(
        "--structure",
        type=lambda value: _existing_file(value, "structure file"),
        metavar="STRU",
        help=(
            "Structure file: required by --spacing and --path, and used to "
            "report the k-spacing an inspected or generated mesh realizes."
        ),
    )
    parser.add_argument(
        "-o", "--output", type=Path, metavar="KPT",
        help="Write a generated KPT file to this path.",
    )
    parser.add_argument(
        "--mesh", type=int, nargs=3, metavar=("N1", "N2", "N3"),
        help="Mesh subdivisions to write.",
    )
    parser.add_argument(
        "--spacing", type=float, nargs="+", metavar="1/ANGSTROM",
        help="Target k-spacing in 1/Angstrom, one value or three.",
    )
    parser.add_argument(
        "--path", action="store_true",
        help="Write the high-symmetry path of --structure in line mode.",
    )
    parser.add_argument(
        "--path-mode", choices=["auto", "bulk", "slab", "wire"], default="auto",
        help="How --path picks the path: follow the dimensionality of the structure, or force one; default: auto.",
    )
    parser.add_argument(
        "--min-vacuum", type=float, default=5.0, metavar="ANGSTROM",
        help="Empty span that counts as vacuum when --path-mode is auto, default: 5.",
    )
    parser.add_argument(
        "--npoints", type=int, default=20,
        help="Points sampled in every band-path segment, default: 20.",
    )
    parser.add_argument(
        "--model", choices=["gamma", "mp"], default="gamma",
        help="Mesh model written for --mesh/--spacing, default: gamma.",
    )
    parser.add_argument(
        "--symprec", type=float, default=1e-5, metavar="ANGSTROM",
        help="Symmetry tolerance of the band path in Angstrom, default: 1e-5.",
    )
    parser.add_argument(
        "--angle-tolerance", type=float, default=-1.0, metavar="DEGREES",
        help="Angle tolerance of the band path in degrees; a negative value lets seekpath estimate it.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace OUTPUT when it already exists.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.set_defaults(handler=run)


def _read_structure(path: Path) -> AbacusSTRU:
    """Read a structure file or raise when it cannot be parsed."""
    structure = AbacusSTRU.read(str(path))
    if structure is None:
        raise RuntimeError(f"failed to read structure: {path}")
    return structure


def _write(values: Any, model: str, output: Path, override: bool) -> Path:
    """Write one KPT file, refusing to replace an existing output."""
    output = Path(output).expanduser()
    if output.exists() and not override:
        raise RuntimeError(f"output already exists: {output}; use --override to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    WriteKpt(values, str(output), model)
    return output


def _print_report(report: dict[str, Any], as_json: bool) -> None:
    """Print one inspection or generation report."""
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"  kpt: {report['file']}")
    print(f"  model: {report['model']}")
    if report.get("dimensionality_label"):
        method = report.get("path_method")
        suffix = "" if method is None else f", {method}"
        print(f"  dimensionality: {report['dimensionality_label']}{suffix}")
    if "mesh" in report:
        print(
            f"  mesh: {' '.join(str(value) for value in report['mesh'])} "
            f"({report['mesh_points']} grid points)"
        )
        print(f"  shifts: {' '.join(str(value) for value in report['shifts'])}")
    else:
        print(f"  nodes: {report['nodes']} ({report['points_total']} points)")
        labels = [label for label in report.get("labels", []) if label]
        if labels:
            print(f"  labels: {' '.join(str(label) for label in labels)}")
    spacing = report.get("kspacing_angstrom")
    if spacing is not None:
        text = " ".join("inf" if value is None else f"{value:g}" for value in spacing)
        print(f"  k spacing (1/Angstrom): {text}")
    if not report.get("valid", True):
        print(f"  warning: {report.get('error', 'the KPT values are invalid')}")


def _describe(args: argparse.Namespace) -> int:
    """Inspect one KPT file."""
    structure = None if args.structure is None else _read_structure(args.structure)
    values, model = read_kpt(args.filename)
    report = {
        "file": str(Path(args.filename).absolute()),
        **model_report(values, model),
    }
    if model in ("direct", "cartesian"):
        report["points_total"] = report["nodes"]
    elif "mesh" not in report:
        report["points_total"] = int(sum(report.get("point_counts", [])))
    if structure is not None and "mesh" in report and report["valid"]:
        report["kspacing_angstrom"] = spacing_from_mesh(structure, report["mesh"])
    _print_report(report, args.json)
    return 0


def _generate(args: argparse.Namespace) -> int:
    """Write a mesh or band-path KPT file."""
    structure = None if args.structure is None else _read_structure(args.structure)

    if args.path:
        if structure is None:
            raise ValueError("--path needs --structure")
        path = band_path(
            structure,
            npoints=args.npoints,
            min_vacuum=args.min_vacuum,
            path_mode=args.path_mode,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
        )
        output = _write(path.nodes, "line", args.output, args.override)
        report = {
            "file": str(output.absolute()),
            "model": "line",
            "nodes": len(path.nodes),
            "points_total": path.points,
            "labels": path.labels,
            "segments": path.segments,
            "npoints": args.npoints,
            "dimensionality": path.dimensionality,
            "dimensionality_label": path.label,
            "path_method": path.method,
            "periodic_directions": path.periodic_directions,
            "natoms": structure.natoms,
            "valid": True,
        }
        _print_report(report, args.json)
        return 0

    if args.mesh is not None and args.spacing is not None:
        raise ValueError("give either --mesh or --spacing, not both")
    if args.mesh is not None:
        mesh = [int(value) for value in args.mesh]
    elif args.spacing is not None:
        if structure is None:
            raise ValueError("--spacing needs --structure")
        mesh = mesh_from_spacing(structure, args.spacing)
    else:
        raise ValueError("--output needs --mesh, --spacing or --path")

    values = [*mesh, 0, 0, 0]
    output = _write(values, args.model, args.output, args.override)
    report = {
        "file": str(output.absolute()),
        "model": args.model,
        "mesh": mesh,
        "shifts": [0.0, 0.0, 0.0],
        "mesh_points": int(np.prod(mesh)),
        "valid": True,
    }
    if structure is not None:
        report["kspacing_angstrom"] = spacing_from_mesh(structure, mesh)
    _print_report(report, args.json)
    return 0


def run(args: argparse.Namespace) -> int:
    """Inspect a KPT file, or write a generated one."""
    if args.output is None:
        if args.filename is None:
            raise ValueError("give a KPT file to inspect, or --output to write a new one")
        return _describe(args)
    if args.filename is not None:
        raise ValueError("--output writes a new file; drop the KPT argument")
    return _generate(args)
