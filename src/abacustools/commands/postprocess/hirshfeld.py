"""Compute Hirshfeld (and CM5) atomic charges from an ABACUS job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abacustools.data.hirshfeld import hirshfeld_charges, read_cm5_parameters


def _job_directory(value: str) -> Path:
    """Return an existing job directory or raise a parser error."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j", "--job", required=True, type=_job_directory,
        help="ABACUS job directory with a converged charge density.",
    )
    parser.add_argument(
        "--cm5-params", default=None,
        help="JSON table of CM5 element-pair coefficients; enables CM5 charges.",
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


def _report(result) -> dict:
    return {
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


def run(args: argparse.Namespace) -> int:
    parameters = None
    if args.cm5_params:
        path = Path(args.cm5_params)
        if not path.is_absolute():
            path = Path(args.job) / path
        parameters = read_cm5_parameters(path)

    result = hirshfeld_charges(
        args.job,
        grid_shape=tuple(args.grid) if args.grid else None,
        lat0=args.lat0,
        cm5_parameters=parameters,
    )

    if args.json:
        print(json.dumps(_report(result), indent=2))
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
        help="Compute Hirshfeld (and CM5) atomic charges from an ABACUS job.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
