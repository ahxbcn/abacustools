"""Search approximate QTAIM bond critical points outside atomic cores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from abacustools.core.render import cli_output
from abacustools.data.qtaim import QtaimBcpResult, analyze_qtaim_bcp


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
        help="ABACUS job directory with a converged total charge density.",
    )
    parser.add_argument(
        "--density",
        default=None,
        metavar="CUBE",
        help=(
            "Use an explicit total-density cube instead of the density files of "
            "the job. Relative paths are resolved below JOB."
        ),
    )
    parser.add_argument(
        "--core-radius",
        default=0.3,
        metavar="AUTO|ANGSTROM",
        help=(
            "Exclude this radius around every nucleus from the BCP search "
            "(default: 0.3 Angstrom). 'auto' uses the largest UPF non-local "
            "projector cutoff radius as a more conservative heuristic."
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=4.0,
        metavar="ANGSTROM",
        help="Maximum atom-pair distance used to seed BCP searches (default: 4.0).",
    )
    parser.add_argument(
        "--line-samples",
        type=int,
        default=31,
        help="Density samples along each candidate atom pair (default: 31).",
    )
    parser.add_argument(
        "--seeds-per-pair",
        type=int,
        default=3,
        help="Density minima refined per atom pair (default: 3).",
    )
    parser.add_argument(
        "--gradient-tol",
        type=float,
        default=1.0e-3,
        metavar="A.U.",
        help="Gradient norm accepted as a stationary point, in e/Bohr^4.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=60,
        help="Maximum Newton iterations for each seed (default: 60).",
    )
    parser.add_argument(
        "--step-limit",
        type=float,
        default=0.25,
        metavar="ANGSTROM",
        help="Maximum Newton displacement per iteration (default: 0.25).",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=0.08,
        metavar="ANGSTROM",
        help="Merge BCPs closer than this periodic distance (default: 0.08).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="FFT grid for restart input when the job log does not provide it.",
    )
    parser.add_argument(
        "--lat0",
        type=float,
        default=None,
        help="ABACUS LATTICE_CONSTANT in Bohr for restart input.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the JSON report, relative to JOB by default.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete JSON report.",
    )


def _report(result: QtaimBcpResult) -> dict[str, Any]:
    return {
        "method": "qtaim-bcp-search",
        "job": result.job,
        "density_source": result.density_source,
        "grid": list(result.grid),
        "cell_angstrom": [list(row) for row in result.cell],
        "core_radius_source": result.core_radius_source,
        "core_radii_angstrom": [
            {
                "atom": entry.index,
                "element": entry.element,
                "radius": entry.radius,
            }
            for entry in result.core_radii
        ],
        "search": {
            "pair_cutoff_angstrom": result.pair_cutoff,
            "candidate_pairs": result.candidate_pairs,
            "seeds": result.seeds,
            "converged": result.converged_searches,
            "rejected_core": result.rejected_core,
            "rejected_gradient": result.rejected_gradient,
            "rejected_signature": result.rejected_signature,
            "rejected_assignment": result.rejected_assignment,
            "duplicate_points": result.duplicate_points,
        },
        "critical_points": [
            {
                "index": point.index,
                "type": "BCP",
                "atom1": point.atom1,
                "atom2": point.atom2,
                "element1": point.element1,
                "element2": point.element2,
                "position_angstrom": list(point.position),
                "fractional": list(point.fractional),
                "rho_e_bohr3": point.rho,
                "gradient_e_bohr4": list(point.gradient),
                "gradient_norm_e_bohr4": point.gradient_norm,
                "hessian_e_bohr5": [list(row) for row in point.hessian],
                "hessian_eigenvalues_e_bohr5": list(point.eigenvalues),
                "laplacian_e_bohr5": point.laplacian,
                "ellipticity": point.ellipticity,
                "distance_to_atom1_angstrom": point.distance1,
                "distance_to_atom2_angstrom": point.distance2,
                "core_clearance_angstrom": point.core_clearance,
                "iterations": point.iterations,
            }
            for point in result.points
        ],
        "caveat": result.caveat,
    }


def _print_result(result: QtaimBcpResult) -> None:
    cli_output(
        f"QTAIM BCP search: {result.job} (grid {'x'.join(str(value) for value in result.grid)})"
    )
    cli_output(
        f"  core radii: {result.core_radius_source}; "
        f"pairs={result.candidate_pairs} seeds={result.seeds} "
        f"converged={result.converged_searches}"
    )
    cli_output(
        f"  rejected: core={result.rejected_core} "
        f"gradient={result.rejected_gradient} "
        f"signature={result.rejected_signature}"
    )
    for point in result.points:
        ellipticity = "n/a" if point.ellipticity is None else f"{point.ellipticity:.5f}"
        cli_output(
            f"  BCP {point.index}: {point.atom1}({point.element1})-"
            f"{point.atom2}({point.element2}) "
            f"rho={point.rho:.8f} e/Bohr^3 "
            f"lap={point.laplacian:.8f} e/Bohr^5 "
            f"ellipticity={ellipticity}"
        )
    if not result.points:
        cli_output("  no BCP passed the core, gradient and signature checks")


def run(args: argparse.Namespace) -> int:
    result = analyze_qtaim_bcp(
        args.job,
        density_cube=args.density,
        core_radius=args.core_radius,
        pair_cutoff=args.cutoff,
        line_samples=args.line_samples,
        seeds_per_pair=args.seeds_per_pair,
        gradient_tolerance=args.gradient_tol,
        max_iterations=args.max_iterations,
        step_limit=args.step_limit,
        duplicate_distance=args.merge_distance,
        grid_shape=tuple(args.grid) if args.grid else None,
        lat0=args.lat0,
    )
    report = _report(result)
    if args.output:
        output = _output_path(Path(args.job), args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.json:
            cli_output(f"Wrote QTAIM BCP report: {output}")
    if args.json:
        cli_output(json.dumps(report, indent=2))
    else:
        _print_result(result)
    return 0


def register_parser(subparsers) -> None:
    """Register the approximate QTAIM BCP search command."""
    parser = subparsers.add_parser(
        "qtaim",
        help=(
            "Search QTAIM bond critical points outside atomic cores; the "
            "excluded core density is not reconstructed."
        ),
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
