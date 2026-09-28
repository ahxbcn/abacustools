"""Export an ABACUS LCAO wavefunction as a Molden file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from abacustools.data.molden import convert_wfc_to_molden


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j",
        "--job",
        required=True,
        type=_job_directory,
        help="ABACUS LCAO job directory with WFC_NAO_* files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Molden filename, relative to JOB by default (default: wfc.molden).",
    )
    parser.add_argument(
        "--kpoint",
        type=int,
        default=None,
        help="One-based k-point index to write; defaults to Gamma or the first k-point.",
    )
    parser.add_argument(
        "--gto-primitives",
        type=int,
        default=6,
        help="Number of Gaussian primitives used to expand every NAO (default: 6).",
    )
    parser.add_argument(
        "--atoms-unit",
        choices=("bohr", "angstrom"),
        default="bohr",
        help="Unit of the [Atoms] block (default: bohr).",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")


def run(args: argparse.Namespace) -> int:
    result = convert_wfc_to_molden(
        args.job,
        args.output,
        kpoint=args.kpoint,
        gto_primitives=args.gto_primitives,
        atoms_unit=args.atoms_unit,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return 0
    kpoint_text = "Gamma" if result.gamma_only else f"k-point {result.kpoint}/{result.nkpoints}"
    print(f"Exported {result.job} to {result.output}")
    print(
        f"  {kpoint_text}, nspin={result.nspin}, {result.basis_functions} basis functions, "
        f"{result.nbands} bands per spin, {result.gto_primitives} GTO primitives per NAO"
    )
    print(f"  largest relative radial fit error: {result.max_relative_error:.3e}")
    return 0


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "molden",
        help=(
            "Convert an ABACUS LCAO wavefunction into a Molden file by expanding "
            "the numerical orbitals into contracted Gaussians."
        ),
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
