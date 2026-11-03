"""Process approximate Periodic NBO results from an ABACUS LCAO job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from abacustools.core.render import cli_output
from abacustools.data.nbo import (
    PeriodicNBOAnalysis,
    PeriodicNBOError,
    PeriodicNBOOptions,
    analyze_periodic_nbo,
)


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
        "-c",
        "--cutoff",
        type=float,
        default=3.0,
        help="Only search two-centre bonds within this periodic distance in Angstrom.",
    )
    parser.add_argument(
        "--r-shells",
        type=int,
        default=1,
        help="Include real-space lattice images from -N to N along each axis.",
    )
    parser.add_argument(
        "--threshold-one",
        type=float,
        default=1.8,
        help="One-centre occupancy threshold, default: 1.8.",
    )
    parser.add_argument(
        "--threshold-two",
        type=float,
        default=1.8,
        help="Two-centre bond occupancy threshold, default: 1.8.",
    )
    parser.add_argument(
        "--rydberg-threshold",
        type=float,
        default=1.0e-3,
        help="Minimum residual one-centre occupancy for Rydberg output.",
    )
    parser.add_argument(
        "--donor-acceptor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate periodic second-order donor-acceptor interactions.",
    )
    parser.add_argument(
        "--e2-threshold",
        type=float,
        default=0.05,
        help="Minimum donor-acceptor stabilization in eV.",
    )
    parser.add_argument(
        "--max-e2",
        type=int,
        default=100,
        help="Maximum number of donor-acceptor interactions to retain.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=20,
        help="Maximum number of NBOs shown in the terminal.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the complete report as JSON, relative to JOB by default.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete JSON report to stdout.",
    )


def _print_analysis(
    analysis: PeriodicNBOAnalysis,
    maximum: int,
) -> None:
    console = Console()
    console.print(
        f"Periodic NBO: [bold]{Path(analysis.job).name}[/bold] "
        f"({analysis.kpoint_count} k-points, nspin={analysis.nspin}, "
        f"{analysis.basis_functions} basis functions)"
    )
    console.print(
        f"Electrons: {analysis.total_electrons:.8f}; "
        f"population: {analysis.population_method}; "
        f"occupancy: {analysis.occupancy_method}; "
        f"occupation convention: {analysis.occupancy_convention}"
    )

    atoms = Table(title="Atomic populations (Mulliken)")
    atoms.add_column("#", justify="right")
    atoms.add_column("Element")
    atoms.add_column("Population (e)", justify="right")
    atoms.add_column("Charge (e)", justify="right")
    for atom in analysis.atoms:
        atoms.add_row(
            str(atom.index),
            atom.element,
            f"{atom.population:.6f}",
            f"{atom.charge:+.6f}",
        )
    console.print(atoms)

    counts: dict[str, int] = {}
    for orbital in analysis.wavefunctions:
        counts[orbital.kind] = counts.get(orbital.kind, 0) + 1
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    console.print(f"NBO orbitals: {summary or 'none'}")

    table = Table(title=f"Localised orbitals (top {maximum})")
    table.add_column("Index", justify="right")
    table.add_column("Kind")
    table.add_column("Centres")
    table.add_column("Occupancy", justify="right")
    table.add_column("Distance (A)", justify="right")
    table.add_column("Hybridisation")
    shown = sorted(
        analysis.wavefunctions,
        key=lambda item: item.occupancy,
        reverse=True,
    )[: max(0, maximum)]
    for orbital in shown:
        centres = ", ".join(
            f"{atom + 1}{'' if image == (0, 0, 0) else f'@{image}'}"
            for atom, image in orbital.centers
        )
        table.add_row(
            str(orbital.index),
            orbital.kind,
            centres,
            f"{orbital.occupancy:.6f}",
            "-" if orbital.distance is None else f"{orbital.distance:.4f}",
            " / ".join(orbital.hybridizations),
        )
    console.print(table)

    if analysis.donor_acceptor:
        interactions = Table(title="Periodic donor-acceptor interactions")
        interactions.add_column("Donor")
        interactions.add_column("Acceptor")
        interactions.add_column("E(2) eV", justify="right")
        interactions.add_column("E(2) kcal/mol", justify="right")
        for item in analysis.donor_acceptor[:maximum]:
            interactions.add_row(
                item.donor_label,
                item.acceptor_label,
                f"{item.stabilization_ev:.6f}",
                f"{item.stabilization_kcal_mol:.3f}",
            )
        console.print(interactions)

    for warning in analysis.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def run(args: argparse.Namespace) -> int:
    try:
        analysis = analyze_periodic_nbo(
            args.job,
            options=PeriodicNBOOptions(
                cutoff=args.cutoff,
                r_shells=args.r_shells,
                one_center_threshold=args.threshold_one,
                two_center_threshold=args.threshold_two,
                rydberg_threshold=args.rydberg_threshold,
                donor_acceptor=args.donor_acceptor,
                e2_threshold_ev=args.e2_threshold,
                maximum_donor_acceptor=args.max_e2,
            ),
        )
    except (PeriodicNBOError, OSError, ValueError) as error:
        cli_output(f"Periodic NBO analysis failed: {error}")
        return 1

    report = analysis.to_dict()
    if args.output:
        output = _output_path(Path(args.job), args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not args.json:
            cli_output(f"Wrote Periodic NBO report: {output}")

    if args.json:
        cli_output(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_analysis(analysis, args.max_print)
    return 0


def register_parser(subparsers) -> None:
    """Register the postprocess nbo command."""

    parser = subparsers.add_parser(
        "nbo",
        help=(
            "Analyze approximate Periodic NBOs, Mulliken populations, lone pairs, "
            "bonds, antibonds and donor-acceptor interactions from ABACUS LCAO output."
        ),
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
