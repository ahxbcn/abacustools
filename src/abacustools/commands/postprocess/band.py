"""Process and plot band structures from ABACUS NSCF calculations."""

from __future__ import annotations

import argparse
from pathlib import Path

from abacustools.data.band import BandData


def _job_directory(value: str) -> Path:
    """Return an existing ABACUS job directory."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _output_path(job: Path, value: str) -> Path:
    """Resolve a relative output path below the job directory."""
    path = Path(value)
    return path if path.is_absolute() else job / path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the band postprocessing command."""
    parser.add_argument(
        "-j",
        "--job",
        required=True,
        type=_job_directory,
        help="ABACUS job directory containing BANDS_1.dat; NSCF is recommended.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="band.png",
        help="Band plot filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--data-output",
        default="band.dat",
        help="Processed band data filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--kpath-output",
        default="KPATH.txt",
        help="High-symmetry k-path filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=-10.0,
        help="Lower energy limit in eV relative to the Fermi level.",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=10.0,
        help="Upper energy limit in eV relative to the Fermi level.",
    )
    parser.add_argument(
        "--efermi",
        type=float,
        default=None,
        help="Override the Fermi energy in eV. Otherwise read it from the NSCF log.",
    )


def run(args: argparse.Namespace) -> int:
    """Read an ABACUS NSCF band result and write its plot and processed data."""
    if args.emin >= args.emax:
        raise ValueError("emin must be smaller than emax")

    job = Path(args.job)
    band_data = BandData.ReadFromAbacusJob(job, efermi=args.efermi)

    output = _output_path(job, args.output)
    data_output = _output_path(job, args.data_output)
    kpath_output = _output_path(job, args.kpath_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_output.parent.mkdir(parents=True, exist_ok=True)
    kpath_output.parent.mkdir(parents=True, exist_ok=True)

    band_data.plot_band(emin=args.emin, emax=args.emax, fig_name=str(output))
    band_data.write_to_file(str(data_output))
    band_data.write_kpath_info(str(kpath_output))
    if band_data.nspin == 2:
        data_paths = [
            data_output.with_name(f"{data_output.stem}_up{data_output.suffix}"),
            data_output.with_name(f"{data_output.stem}_down{data_output.suffix}"),
        ]
    else:
        data_paths = [data_output]

    print(
        f"Processed {job}: {band_data.nkpts} k-points, "
        f"{band_data.nbands} bands, nspin={band_data.nspin}, "
        f"efermi={band_data.efermi:.8f} eV"
    )
    print(f"Wrote band plot: {output}")
    print(f"Wrote band data: {', '.join(str(path) for path in data_paths)}")
    print(f"Wrote k-path information: {kpath_output}")
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess band``."""
    parser = subparsers.add_parser(
        "band",
        help="Process and plot bands from an ABACUS NSCF calculation.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
