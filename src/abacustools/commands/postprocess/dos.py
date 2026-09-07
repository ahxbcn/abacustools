"""Process and plot DOS/PDOS from ABACUS calculation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from abacustools.data.dos import DOSData, PDOSData


_PDOS_MODES = ("species", "species-shell", "species-orbital", "atoms")


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
    """Register arguments for the DOS postprocessing command."""
    parser.add_argument(
        "-j",
        "--job",
        required=True,
        type=_job_directory,
        help="ABACUS job directory containing DOS*_smearing.dat.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Plot filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--data-output",
        default=None,
        help="Processed data filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=-20.0,
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
        help="Override the Fermi energy in eV.",
    )
    parser.add_argument(
        "--pdos",
        choices=_PDOS_MODES,
        default=None,
        metavar="MODE",
        help="Plot projected DOS by species, species-shell, species-orbital, or atoms.",
    )
    parser.add_argument(
        "--atom-index",
        type=int,
        nargs="+",
        default=None,
        help="One-based atom indices for --pdos atoms.",
    )


def _pdos_handlers(
    pdos: PDOSData, mode: str, atom_indices: Optional[List[int]]
) -> Tuple[Callable, Callable]:
    """Return the existing PDOS plot and export methods for a selected mode."""
    if mode == "species":
        return pdos.plot_species_pdos, pdos.write_species_pdos
    if mode == "species-shell":
        return pdos.plot_species_shell_pdos, pdos.write_species_shell_pdos
    if mode == "species-orbital":
        return pdos.plot_species_orbital_pdos, pdos.write_species_orbital_pdos
    if mode == "atoms":
        if not atom_indices:
            raise ValueError("--atom-index is required when --pdos atoms is selected")
        if any(index < 1 for index in atom_indices):
            raise ValueError("--atom-index values must be positive")
        return (
            lambda emin, emax, pdos_fig_name: pdos.plot_atoms_pdos(
                [index - 1 for index in atom_indices], emin, emax, pdos_fig_name
            ),
            lambda pdos_dat_file: pdos.write_atoms_pdos(
                [index - 1 for index in atom_indices], pdos_dat_file
            ),
        )
    raise ValueError(f"unsupported PDOS mode: {mode}")


def run(args: argparse.Namespace) -> int:
    """Read DOS/PDOS output and write a plot and processed data."""
    if args.emin >= args.emax:
        raise ValueError("emin must be smaller than emax")

    job = Path(args.job)
    mode = getattr(args, "pdos", None)
    output = _output_path(job, args.output or ("PDOS.png" if mode else "DOS.png"))
    data_output = _output_path(
        job, args.data_output or ("PDOS.dat" if mode else "DOS.dat")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data_output.parent.mkdir(parents=True, exist_ok=True)

    if mode is None:
        dos = DOSData.ReadFromAbacusJob(job, efermi=args.efermi)
        dos.plot_dos(args.emin, args.emax, "Density of states", str(output))
        dos.write_dos(str(data_output))
        channels = dos.dosdata.shape[1]
        energy = dos.energy
    else:
        pdos = PDOSData.ReadFromAbacusJob(job, efermi=args.efermi)
        plot, write = _pdos_handlers(pdos, mode, getattr(args, "atom_index", None))
        plot(args.emin, args.emax, str(output))
        write(str(data_output))
        channels = len(pdos.projected_dos)
        energy = pdos.energy

    print(
        f"Processed {job}: {len(energy)} energy points, "
        f"{channels} {'PDOS orbitals' if mode else 'DOS channels'}"
    )
    print(f"Wrote DOS plot: {output}")
    print(f"Wrote DOS data: {data_output}")
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess dos``."""
    parser = subparsers.add_parser(
        "dos",
        help="Process and plot DOS or PDOS from an ABACUS calculation.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
