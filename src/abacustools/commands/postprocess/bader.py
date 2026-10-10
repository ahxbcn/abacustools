"""Implementation of the ``abacustools postprocess bader`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from abacustools.data.bader import BaderAnalysis, BaderError, analyze_bader
from abacustools.integrations.baderkit import DEFAULT_METHOD, METHODS, analyze_baderkit


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _output_path(job: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job / path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-j", "--job", required=True, type=_job_directory, help="ABACUS job directory.")
    parser.add_argument("-o", "--output", default=None, help="Write a JSON report, relative to JOB by default.")
    parser.add_argument(
        "--backend",
        choices=("bader", "baderkit"),
        default="bader",
        help="Bader implementation: the external Henkelman program or the baderkit library.",
    )
    parser.add_argument(
        "--baderkit-method",
        choices=METHODS,
        default=DEFAULT_METHOD,
        help="Partitioning method of the baderkit backend (default matches the external program).",
    )
    parser.add_argument(
        "--bader-exe",
        default=None,
        help="Bader executable of the bader backend (default: BADER_EXE env var or config).",
    )
    parser.add_argument("--cube", default=None, help="Explicit charge-density cube file or directory, relative to JOB.")
    parser.add_argument("--reference", default=None, help="Reference charge cube passed to bader -ref.")
    parser.add_argument("--grid", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=None, help="FFT grid for restart input when it cannot be read from the log.")
    parser.add_argument("--lat0", type=float, default=None, help="ABACUS LATTICE_CONSTANT in Bohr (default: the value in STRU).")
    parser.add_argument("--vacuum", default=None, help="Vacuum handling: 'off', 'auto' (1e-3 e/Angstrom^3) or a density value.")
    parser.add_argument("--workdir", default=None, help="Keep the generated cubes and bader output in this directory.")
    parser.add_argument("--keep-cubes", action="store_true", help="Keep the temporary cubes and the partition output.")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON report to stdout.")


def _print_table(analysis: BaderAnalysis) -> None:
    console = Console()
    table = Table(
        title=(
            f"Bader charges: {analysis.job.name} "
            f"(nspin={analysis.nspin}, {analysis.backend})"
        )
    )
    table.add_column("#", justify="right")
    table.add_column("Element")
    table.add_column("Z_valence", justify="right")
    table.add_column("Bader charge (e)", justify="right")
    table.add_column("Net charge (e)", justify="right")
    if analysis.nspin == 2:
        table.add_column("Spin moment (uB)", justify="right")
    for atom in analysis.atoms:
        row = [
            str(atom.index),
            atom.element,
            f"{atom.z_valence:.4f}",
            f"{atom.bader_charge:.4f}",
            f"{atom.net_charge:+.4f}",
        ]
        if analysis.nspin == 2:
            moment = "-" if atom.spin_moment is None else f"{atom.spin_moment:+.4f}"
            row.append(moment)
        table.add_row(*row)
    console.print(table)
    console.print(
        f"Number of electrons: {analysis.number_of_electrons:.4f}  "
        f"Total net charge: {analysis.total_net_charge:+.4f}  "
        f"Vacuum charge: {analysis.vacuum_charge:.4f}  "
        f"Source: {analysis.charge_source}"
    )


def run(args: argparse.Namespace) -> int:
    job = Path(args.job)
    common = dict(
        cube=args.cube,
        reference=args.reference,
        grid_shape=tuple(args.grid) if args.grid else None,
        lat0=args.lat0,
        vacuum=args.vacuum,
        workdir=args.workdir,
        keep=args.keep_cubes,
    )
    try:
        if args.backend == "baderkit":
            analysis = analyze_baderkit(job, method=args.baderkit_method, **common)
        else:
            analysis = analyze_bader(job, exe=args.bader_exe, **common)
    except (BaderError, ImportError) as error:
        print(f"Bader analysis failed: {error}")
        return 1

    report = analysis.to_dict()
    if args.output:
        output = _output_path(job, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.json:
            print(f"Wrote Bader report: {output}")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    _print_table(analysis)
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess bader``."""
    parser = subparsers.add_parser(
        "bader",
        help="Calculate Bader charges with the external bader program.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
