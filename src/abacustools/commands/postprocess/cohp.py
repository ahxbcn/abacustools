"""Process and plot COOP/COHP from ABACUS LCAO output."""

from __future__ import annotations

import argparse
from pathlib import Path

from abacustools.data.cohp import analyze_cohp


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _output_path(job: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job / path


def _orbital_list(value: str) -> list[int]:
    try:
        orbitals = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("orbital indices must be comma-separated integers") from error
    if not orbitals:
        raise argparse.ArgumentTypeError("at least one orbital index is required")
    return orbitals


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-j", "--job", required=True, type=_job_directory, help="ABACUS LCAO job directory.")
    parser.add_argument("--atom-i-orbs", required=True, type=_orbital_list, help="Zero-based NAO indices for group I, e.g. 0,1,2.")
    parser.add_argument("--atom-j-orbs", required=True, type=_orbital_list, help="Zero-based NAO indices for group J, e.g. 13,14,15.")
    parser.add_argument("--method", choices=("COHP", "COOP"), default="COHP", help="Bonding curve to calculate.")
    parser.add_argument("--spin", choices=("sum", "up", "down"), default="sum", help="Spin channel to calculate.")
    parser.add_argument("--de", type=float, default=0.1, help="Energy-grid spacing in eV (default: 0.1).")
    parser.add_argument("--no-smooth", action="store_true", help="Do not apply Gaussian smoothing.")
    parser.add_argument("--smooth-nstddev", type=float, default=3.0, help="Gaussian width in grid spacings (default: 3).")
    parser.add_argument("--emin", type=float, default=-10.0, help="Lower plot limit relative to the Fermi level, in eV.")
    parser.add_argument("--emax", type=float, default=10.0, help="Upper plot limit relative to the Fermi level, in eV.")
    parser.add_argument("--width", type=float, default=None, help="Half-width of the horizontal plot axis.")
    parser.add_argument("--invert", action="store_true", help="Invert the plotted curve; saved data keeps the computed sign.")
    parser.add_argument("-o", "--output", default=None, help="Plot filename, relative to JOB by default.")
    parser.add_argument("--data-output", default=None, help="Processed data filename, relative to JOB by default.")
    parser.add_argument("--efermi", type=float, default=None, help="Override the Fermi energy in eV.")


def run(args: argparse.Namespace) -> int:
    if args.emin >= args.emax:
        raise ValueError("emin must be smaller than emax")
    job = Path(args.job)
    method = args.method.upper()
    output = _output_path(job, args.output or f"{method.lower()}.png")
    data_output = _output_path(job, args.data_output or f"{method.lower()}.dat")
    result = analyze_cohp(
        job,
        args.atom_i_orbs,
        args.atom_j_orbs,
        method=method,
        spin=args.spin,
        de=args.de,
        smooth=not args.no_smooth,
        smooth_nstddev=args.smooth_nstddev,
        efermi=args.efermi,
    )
    result.plot(output, emin=args.emin, emax=args.emax, width=args.width, invert=args.invert)
    result.write_data(data_output)
    fermi_text = "not found" if result.efermi is None else f"{result.efermi:.8f} eV"
    print(f"Processed {job}: {method}, spin={args.spin}, {len(result.energy)} energy points, efermi={fermi_text}")
    print(f"{result.label} = {result.ico_value:.8f}")
    print(f"Wrote {method} plot: {output}")
    print(f"Wrote {method} data: {data_output}")
    return 0


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser("cohp", help="Process and plot COHP or COOP from an ABACUS LCAO calculation.")
    _register_arguments(parser)
    parser.set_defaults(handler=run)
