"""Mulliken, Loewdin and Wiberg population analysis from an ABACUS job."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from abacustools.data.population import PopulationAnalysis, analyze_population


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _output_path(job: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job / path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j",
        "--job",
        required=True,
        type=_job_directory,
        help="ABACUS LCAO job directory.",
    )
    parser.add_argument(
        "--method",
        choices=("mulliken", "mulliken-overlap", "lowdin", "wiberg"),
        default="mulliken",
        help=(
            "mulliken reports Mulliken populations and the quadratic "
            "Mulliken/Mayer bond order; mulliken-overlap reports the linear "
            "overlap population; lowdin and wiberg report Loewdin populations "
            "and Wiberg bond orders."
        ),
    )
    parser.add_argument(
        "--spin",
        choices=("total", "up", "down", "difference"),
        default="total",
        help=(
            "Spin channel to report. up, down and difference require nspin=2; "
            "difference is spin-up minus spin-down."
        ),
    )
    parser.add_argument(
        "--spin-factor",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Multiply nspin=2 bond orders. The default 1 is the direct "
            "spin-channel sum; use 2 to reproduce the legacy Mayer convention."
        ),
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "-c",
        "--cutoff",
        type=float,
        help="Only analyze atom pairs within this distance in Angstrom.",
    )
    selector.add_argument(
        "-p",
        "--pairs",
        help="One-based atom pairs, for example 1-2,1-3.",
    )
    selector.add_argument(
        "--pairs-file",
        help="File containing one-based atom pairs, one pair per line.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.2,
        help="Only print bond values at or above this threshold (default: 0.2).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write a JSON report, relative to JOB by default.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete JSON report to stdout.",
    )


def _report(analysis: PopulationAnalysis, threshold: float | None) -> dict:
    report = asdict(analysis)
    report["bonds"] = [
        bond for bond in report["bonds"] if threshold is None or bond["value"] >= threshold
    ]
    report["bond_count"] = len(report["bonds"])
    report["total_bond_count"] = len(analysis.bonds)
    return report


def run(args: argparse.Namespace) -> int:
    analysis = analyze_population(
        args.job,
        method=args.method,
        spin=args.spin,
        cutoff=args.cutoff,
        pairs=args.pairs,
        pairs_file=args.pairs_file,
        spin_factor=args.spin_factor,
    )
    explicit_pairs = args.pairs is not None or args.pairs_file is not None
    report = _report(
        analysis,
        None if args.json or explicit_pairs else args.threshold,
    )
    if args.output:
        output = _output_path(Path(args.job), args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_report(analysis, None), indent=2), encoding="utf-8")
        if not args.json:
            print(f"Wrote bond population report: {output}")
    if args.json:
        print(json.dumps(_report(analysis, None), indent=2))
        return 0

    label = {
        "mulliken": "Mulliken bond order",
        "mulliken-overlap": "Mulliken overlap population",
        "lowdin": "Wiberg bond order (Loewdin)",
        "wiberg": "Wiberg bond order",
    }[analysis.method]
    print(
        f"{label}: {analysis.job} "
        f"({'gamma-only' if analysis.gamma_only else 'multi-k'}, "
        f"nspin={analysis.nspin}, spin={analysis.spin})"
    )
    expected = (
        "unknown" if analysis.expected_electrons is None else f"{analysis.expected_electrons:.8f}"
    )
    print(
        f"  electrons={analysis.total_electrons:.8f} expected={expected} "
        f"kpoints={analysis.kpoint_count} basis={analysis.basis_functions}"
    )
    print(f"  occupation convention: {analysis.occupancy_convention}")
    for atom in analysis.atoms:
        charge = "" if atom.charge is None else f" charge={atom.charge: .8f} e"
        magnetization = (
            "" if atom.magnetization is None else f" magnetization={atom.magnetization: .8f}"
        )
        print(
            f"  atom {atom.index} {atom.element}: "
            f"population={atom.population:.8f} e{charge}{magnetization}"
        )
    for bond in report["bonds"]:
        print(
            f"  {bond['atom1']}({bond['element1']}) - "
            f"{bond['atom2']}({bond['element2']}): "
            f"{bond['value']:.8f}  distance={bond['distance']:.6f} A"
        )
    return 0


def register_parser(subparsers) -> None:
    """Register the ``bond`` command."""

    parser = subparsers.add_parser(
        "bond",
        help=(
            "Analyze Mulliken/Loewdin populations and Wiberg bond orders; "
            "multi-k nspin=1 and nspin=2 outputs are supported."
        ),
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
