"""Implementation of the ``abacustools job checkinput`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abacustools.core.job import as_json, check_input


def _job_directory(value: str) -> Path:
    """Return an existing job directory."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _print_summary(summary: dict) -> None:
    """Print the main calculation settings in a stable, readable form."""
    for name in (
        "calculation",
        "basis_type",
        "esolver_type",
        "nspin",
        "ecutwfc",
        "scf_thr",
        "smearing_method",
        "smearing_sigma",
    ):
        if name in summary:
            print(f"{name}: {summary[name]}")

    kpoints = summary.get("kpoints")
    if kpoints:
        details = ", ".join(f"{key}={value}" for key, value in kpoints.items())
        print(f"kpoints: {details}")

    structure = summary.get("structure")
    if structure:
        species = ", ".join(f"{name}:{count}" for name, count in structure["species"].items())
        parameters = " ".join(f"{value:.6g}" for value in structure["cell_parameters_ang_deg"])
        print(f"structure: natoms={structure['natoms']}, species={species}")
        print(f"cell (Angstrom/degrees): {parameters}")
        print(f"cell volume (Angstrom^3): {structure['cell_volume_ang3']:.6g}")

    resources = summary.get("resources", {})
    if resources:
        print("resources:")
        for name, values in resources.items():
            print(f"  {name}: {', '.join(values)}")


def register_parser(subparsers) -> None:
    """Register the ``job checkinput`` parser."""
    parser = subparsers.add_parser(
        "checkinput",
        help="Check an ABACUS INPUT and its referenced input files.",
    )
    parser.add_argument("job", type=_job_directory, metavar="JOB")
    parser.add_argument("--strict", action="store_true", help="Treat unknown INPUT keywords as errors.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Check only input files, without reading ABACUS output files."""
    report = check_input(args.job, strict=args.strict)
    if args.json:
        print(json.dumps(as_json(report), indent=2, sort_keys=True))
        return 0 if report.valid else 1

    print(f"job: {Path(args.job).absolute()}")
    print("input check:")
    _print_summary(report.summary)
    print(f"valid: {'yes' if report.valid else 'no'}")
    for issue in report.issues:
        print(f"{issue.level}: {issue.code}: {issue.message}")
    return 0 if report.valid else 1
