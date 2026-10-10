"""Compute Hirshfeld, Hirshfeld-I and CM5 atomic charges from an ABACUS job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abacustools.data.hirshfeld import (
    hirshfeld_charges,
    hirshfeld_i_charges,
    read_reference_densities,
)


def _job_directory(value: str) -> Path:
    """Return an existing job directory or raise a parser error."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _reference_directory(value: str) -> Path:
    """Return an existing reference-density directory or raise a parser error."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"reference directory does not exist: {value}")
    return path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j", "--job", required=True, type=_job_directory,
        help="ABACUS job directory with a converged charge density.",
    )
    parser.add_argument(
        "--hirshfeld-i", action="store_true",
        help="Run the iterative Hirshfeld-I scheme instead of the plain Hirshfeld one.",
    )
    parser.add_argument(
        "--references", type=_reference_directory, default=None, metavar="DIR",
        help="Directory of <element>_<population>.dat reference densities for Hirshfeld-I.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=200,
        help="Maximum number of Hirshfeld-I iterations (default 200).",
    )
    parser.add_argument(
        "--tol", type=float, default=5e-4,
        help="Hirshfeld-I convergence threshold on the population change (default 5e-4).",
    )
    parser.add_argument(
        "--mixing", type=float, default=1.0,
        help="Hirshfeld-I population mixing, in (0, 1] (default 1.0, undamped).",
    )
    parser.add_argument(
        "--no-cm5", action="store_true",
        help="Only report the Hirshfeld charges, without the CM5 correction.",
    )
    parser.add_argument(
        "--grid", type=int, nargs=3, default=None, metavar=("NX", "NY", "NZ"),
        help="FFT grid, when the density has to be rebuilt from a restart file.",
    )
    parser.add_argument(
        "--lat0", type=float, default=None,
        help="Lattice constant, when the density has to be rebuilt from a restart file.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the report as JSON.",
    )


def _hirshfeld_report(result) -> dict:
    return {
        "method": "hirshfeld",
        "job": result.job,
        "grid": list(result.grid),
        "atoms": [
            {
                "element": element,
                "charge": float(charge),
                "volume": float(volume),
                "valence": float(valence),
            }
            for element, charge, volume, valence in zip(
                result.elements, result.charges, result.volumes, result.valence
            )
        ],
        "total_charge": float(result.charges.sum()),
        "cm5": None if result.cm5 is None else [float(value) for value in result.cm5],
    }


def _hirshfeld_i_report(result) -> dict:
    return {
        "method": "hirshfeld-i",
        "job": result.job,
        "grid": list(result.grid),
        "reference_source": result.reference_source,
        "iterations": result.iterations,
        "converged": result.converged,
        "atoms": [
            {
                "element": element,
                "charge": float(charge),
                "population": float(population),
                "valence": float(valence),
            }
            for element, charge, population, valence in zip(
                result.elements, result.charges, result.populations, result.valence
            )
        ],
        "total_charge": float(result.charges.sum()),
    }


def run(args: argparse.Namespace) -> int:
    if args.hirshfeld_i:
        references = read_reference_densities(args.references) if args.references else None
        result = hirshfeld_i_charges(
            args.job,
            references=references,
            grid_shape=tuple(args.grid) if args.grid else None,
            lat0=args.lat0,
            max_iter=args.max_iter,
            tol=args.tol,
            mixing=args.mixing,
        )
        if args.json:
            print(json.dumps(_hirshfeld_i_report(result), indent=2))
            return 0
        print(
            f"Hirshfeld-I charges: {result.job} "
            f"(grid {result.grid[0]}x{result.grid[1]}x{result.grid[2]})"
        )
        print(
            f"  references: {result.reference_source}; "
            f"{result.iterations} iterations, "
            f"{'converged' if result.converged else 'NOT converged'}"
        )
        for index, (element, charge, population) in enumerate(
            zip(result.elements, result.charges, result.populations), start=1
        ):
            print(
                f"  atom {index:3d} {element:3s} q={charge:9.5f} e  "
                f"N={population:9.5f} e"
            )
        print(f"  total charge: {result.charges.sum():.5f} e")
        return 0

    result = hirshfeld_charges(
        args.job,
        grid_shape=tuple(args.grid) if args.grid else None,
        lat0=args.lat0,
        cm5=not args.no_cm5,
    )

    if args.json:
        print(json.dumps(_hirshfeld_report(result), indent=2))
        return 0

    print(f"Hirshfeld charges: {result.job} (grid {result.grid[0]}x{result.grid[1]}x{result.grid[2]})")
    for index, (element, charge, volume) in enumerate(
        zip(result.elements, result.charges, result.volumes), start=1
    ):
        line = f"  atom {index:3d} {element:3s} q={charge:9.5f} e  V={volume:9.3f} A^3"
        if result.cm5 is not None:
            line += f"  q_CM5={result.cm5[index - 1]:9.5f} e"
        print(line)
    print(f"  total charge: {result.charges.sum():.5f} e")
    return 0


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "hirshfeld",
        help="Compute Hirshfeld, Hirshfeld-I and CM5 atomic charges from an ABACUS job.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
