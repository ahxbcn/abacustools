"""Process and plot COOP/COHP for atom pairs from ABACUS LCAO output."""

from __future__ import annotations

import argparse
from pathlib import Path

from abacustools.data.cohp import analyze_cohp, analyze_cohp_pairs
from abacustools.core.render import cli_output


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
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "-c", "--cutoff", type=float,
        help="Analyze every atom pair within this periodic distance in Angstrom.",
    )
    selector.add_argument(
        "-p", "--pairs",
        help="One-based atom pairs, for example 1-2,1-3.",
    )
    selector.add_argument(
        "--pairs-file",
        help="File containing one-based atom pairs, one pair per line.",
    )
    parser.add_argument(
        "--atom-i-orbs", type=_orbital_list,
        help="Advanced: zero-based global NAO indices for the left orbital group.",
    )
    parser.add_argument(
        "--atom-j-orbs", type=_orbital_list,
        help="Advanced: zero-based global NAO indices for the right orbital group.",
    )
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


def _pair_output(path: Path, pair: tuple[int, int], multiple: bool) -> Path:
    """Return the per-pair output path for a multi-pair analysis."""
    if not multiple:
        return path
    return path.with_name(f"{path.stem}_{pair[0]}-{pair[1]}{path.suffix}")


def _orbital_text(indices: tuple[int, ...]) -> str:
    """Format a compact range of zero-based NAO indices."""
    if not indices:
        return "-"
    if len(indices) == 1:
        return str(indices[0])
    return f"{indices[0]}-{indices[-1]}"


def run(args: argparse.Namespace) -> int:
    if args.emin >= args.emax:
        raise ValueError("emin must be smaller than emax")
    job = Path(args.job)
    method = args.method.upper()
    output = _output_path(job, args.output or f"{method.lower()}.png")
    data_output = _output_path(job, args.data_output or f"{method.lower()}.dat")

    pair_selector = (args.cutoff, args.pairs, args.pairs_file)
    analyze_pairs = any(value is not None for value in pair_selector)
    raw_orbital_groups = args.atom_i_orbs is not None or args.atom_j_orbs is not None
    if analyze_pairs and raw_orbital_groups:
        raise ValueError(
            "atom-pair selection and explicit NAO orbital groups are mutually exclusive"
        )
    if not analyze_pairs and not raw_orbital_groups:
        raise ValueError(
            "select atom pairs with --pairs, --pairs-file or --cutoff, "
            "or supply both --atom-i-orbs and --atom-j-orbs"
        )

    entries = []
    if analyze_pairs:
        selected_pairs, results = analyze_cohp_pairs(
            job,
            cutoff=args.cutoff,
            pairs=args.pairs,
            pairs_file=args.pairs_file,
            method=method,
            spin=args.spin,
            de=args.de,
            smooth=not args.no_smooth,
            smooth_nstddev=args.smooth_nstddev,
            efermi=args.efermi,
        )
        multiple = len(results) > 1
        entries = [
            (
                (pair[0] + 1, pair[1] + 1),
                result,
                _pair_output(output, (pair[0] + 1, pair[1] + 1), multiple),
                _pair_output(data_output, (pair[0] + 1, pair[1] + 1), multiple),
            )
            for pair, result in zip(selected_pairs, results)
        ]
    else:
        if args.atom_i_orbs is None or args.atom_j_orbs is None:
            raise ValueError("--atom-i-orbs and --atom-j-orbs must be supplied together")
        entries = [
            (
                None,
                analyze_cohp(
                    job,
                    args.atom_i_orbs,
                    args.atom_j_orbs,
                    method=method,
                    spin=args.spin,
                    de=args.de,
                    smooth=not args.no_smooth,
                    smooth_nstddev=args.smooth_nstddev,
                    efermi=args.efermi,
                ),
                output,
                data_output,
            )
        ]

    for pair, result, plot_path, data_path in entries:
        result.plot(
            plot_path,
            emin=args.emin,
            emax=args.emax,
            width=args.width,
            invert=args.invert,
        )
        result.write_data(data_path)
        fermi_text = "not found" if result.efermi is None else f"{result.efermi:.8f} eV"
        pair_text = ""
        integral_label = result.label
        if pair is not None:
            pair_text = (
                f" atoms {pair[0]}-{pair[1]}, "
                f"NAO {_orbital_text(result.orbital_i)} / "
                f"{_orbital_text(result.orbital_j)}"
            )
            integral_label += f"({pair[0]}-{pair[1]})"
        cli_output(
            f"Processed {job}: {method}, spin={args.spin}{pair_text}, "
            f"{len(result.energy)} energy points, efermi={fermi_text}"
        )
        unit = " eV" if result.method == "COHP" else ""
        cli_output(f"{integral_label} = {result.ico_value:.8f}{unit}")
        cli_output(f"Wrote {method} plot: {plot_path}")
        cli_output(f"Wrote {method} data: {data_path}")
    return 0


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser("cohp", help="Process and plot COHP or COOP for selected atom pairs in an ABACUS LCAO calculation.")
    _register_arguments(parser)
    parser.set_defaults(handler=run)
