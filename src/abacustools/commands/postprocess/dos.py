"""Process and plot DOS/PDOS from ABACUS calculation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, cast

import numpy as np

from abacustools.data.dos import DOSData, PDOSData, l_map, plot_dos_pdos, write_dos_pdos


_PDOS_MODES = (
    "species",
    "species-shell",
    "species-orbital",
    "atoms",
    "atom-shell",
    "atom-orbital",
)


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
        help="Plot projected DOS by species, species-shell, species-orbital, atoms, atom-shell, or atom-orbital.",
    )
    parser.add_argument(
        "--atom-index",
        type=int,
        nargs="+",
        default=None,
        help="One-based atom indices for the atom PDOS modes.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Plot the total DOS together with the species-projected DOS.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_metadata",
        help="List the available species, shells, orbitals and atoms.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the --list metadata as JSON.",
    )


def _require_atom_indices(atom_indices: Optional[List[int]]) -> List[int]:
    if not atom_indices:
        raise ValueError("--atom-index is required for the atom PDOS modes")
    if any(index < 1 for index in atom_indices):
        raise ValueError("--atom-index values must be positive")
    return list(atom_indices)


def _atom_shell_handlers(pdos: PDOSData, atom_indices: List[int]) -> Tuple[Callable, Callable]:
    """Return plot/export handlers for per-atom, per-shell PDOS."""

    def collect() -> Tuple[List[List[np.ndarray]], List[List[str]], List[str]]:
        arrays, labels, titles = [], [], []
        for index in atom_indices:
            species = pdos.get_atom_species(index)
            shells = sorted(pdos.get_atom_shell(index))
            arrays.append([pdos.get_pdos_by_atom_shell(index, shell) for shell in shells])
            labels.append([f"{species}{index}-{l_map[shell]}" for shell in shells])
            titles.append(f"Atom {index} ({species})")
        return arrays, labels, titles

    def plot(emin: float, emax: float, fig_name: str) -> None:
        arrays, labels, titles = collect()
        plot_dos_pdos(arrays, labels, titles, pdos.energy, emin, emax, pdos.efermi is not None, fig_name)

    def write(pdos_dat_file: str) -> None:
        arrays, labels, _ = collect()
        flat_arrays = [array for group in arrays for array in group]
        flat_labels = [label for group in labels for label in group]
        write_dos_pdos(flat_arrays, pdos.energy, flat_labels, pdos.efermi is not None, pdos_dat_file)

    return plot, write


def _atom_orbital_handlers(pdos: PDOSData, atom_indices: List[int]) -> Tuple[Callable, Callable]:
    """Return plot/export handlers for per-atom, per-orbital PDOS."""

    def collect() -> Tuple[List[List[np.ndarray]], List[List[str]], List[str]]:
        arrays, labels, titles = [], [], []
        for index in atom_indices:
            species = pdos.get_atom_species(index)
            orbital_arrays, orbital_labels = [], []
            for shell in sorted(pdos.get_atom_shell(index)):
                for m in sorted(pdos.get_atom_shell_orbital(index, shell)):
                    orbital_arrays.append(pdos.get_pdos_by_atom_orbital(index, shell, m))
                    orbital_labels.append(f"{species}{index}-{m}")
            arrays.append(orbital_arrays)
            labels.append(orbital_labels)
            titles.append(f"Atom {index} ({species})")
        return arrays, labels, titles

    def plot(emin: float, emax: float, fig_name: str) -> None:
        arrays, labels, titles = collect()
        plot_dos_pdos(arrays, labels, titles, pdos.energy, emin, emax, pdos.efermi is not None, fig_name)

    def write(pdos_dat_file: str) -> None:
        arrays, labels, _ = collect()
        flat_arrays = [array for group in arrays for array in group]
        flat_labels = [label for group in labels for label in group]
        write_dos_pdos(flat_arrays, pdos.energy, flat_labels, pdos.efermi is not None, pdos_dat_file)

    return plot, write


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
        selected = _require_atom_indices(atom_indices)
        return (
            lambda emin, emax, pdos_fig_name: pdos.plot_atoms_pdos(
                selected, emin, emax, pdos_fig_name
            ),
            lambda pdos_dat_file: pdos.write_atoms_pdos(selected, pdos_dat_file),
        )
    if mode == "atom-shell":
        return _atom_shell_handlers(pdos, _require_atom_indices(atom_indices))
    if mode == "atom-orbital":
        return _atom_orbital_handlers(pdos, _require_atom_indices(atom_indices))
    raise ValueError(f"unsupported PDOS mode: {mode}")


def _list_metadata(job: Path, args: argparse.Namespace) -> int:
    """List the species, shells, orbitals and atoms available in the PDOS data."""
    pdos = PDOSData.ReadFromAbacusJob(str(job), efermi=args.efermi)
    species = pdos.get_species()
    metadata: dict[str, Any] = {
        "efermi": pdos.efermi,
        "nenergy": int(len(pdos.energy)),
        "energy_min": float(pdos.energy.min()),
        "energy_max": float(pdos.energy.max()),
        "species": species,
        "shells": {s: sorted(pdos.get_species_shell(s)) for s in species},
        "orbitals": {
            s: {
                str(shell): sorted(pdos.get_species_shell_orbital(s, shell))
                for shell in sorted(pdos.get_species_shell(s))
            }
            for s in species
        },
        "atoms": sorted(
            {cast(int, orb["atom_index"]) for orb in (pdos.projected_dos or [])}
        ),
    }
    if args.json:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    print(f"Species: {', '.join(species)}")
    for s in species:
        shells = ", ".join(l_map[shell] for shell in sorted(pdos.get_species_shell(s)))
        print(f"  {s}: shells {shells}")
    print(f"Atoms: {', '.join(str(index) for index in metadata['atoms'])}")
    return 0


def _plot_combined(job: Path, args: argparse.Namespace) -> int:
    """Plot the total DOS together with the species-projected DOS."""
    dos = DOSData.ReadFromAbacusJob(str(job), efermi=args.efermi)
    pdos = PDOSData.ReadFromAbacusJob(str(job), efermi=args.efermi)
    species = pdos.get_species()

    total = np.asarray(dos.dosdata)
    if total.ndim == 1:
        total = total.reshape(len(dos.energy), 1)
    arrays = [total] + [pdos.get_pdos_by_species(s, sum_only=True) for s in species]
    labels = ["total"] + species

    output = _output_path(job, args.output or "DOS_PDOS.png")
    data_output = _output_path(job, args.data_output or "DOS_PDOS.dat")
    output.parent.mkdir(parents=True, exist_ok=True)
    data_output.parent.mkdir(parents=True, exist_ok=True)

    plot_dos_pdos(
        [arrays],
        [labels],
        ["Total and projected density of states"],
        dos.energy,
        args.emin,
        args.emax,
        dos.efermi is not None,
        str(output),
    )
    write_dos_pdos(arrays, dos.energy, labels, dos.efermi is not None, str(data_output))

    print(f"Processed {job}: total DOS and {len(species)} species PDOS")
    print(f"Wrote combined DOS/PDOS plot: {output}")
    print(f"Wrote combined DOS/PDOS data: {data_output}")
    return 0


def run(args: argparse.Namespace) -> int:
    """Read DOS/PDOS output and write a plot and processed data."""
    job = Path(args.job)

    if args.list_metadata:
        return _list_metadata(job, args)

    if args.emin >= args.emax:
        raise ValueError("emin must be smaller than emax")

    if args.combined:
        return _plot_combined(job, args)

    mode = getattr(args, "pdos", None)
    output = _output_path(job, args.output or ("PDOS.png" if mode else "DOS.png"))
    data_output = _output_path(
        job, args.data_output or ("PDOS.dat" if mode else "DOS.dat")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data_output.parent.mkdir(parents=True, exist_ok=True)

    if mode is None:
        dos = DOSData.ReadFromAbacusJob(str(job), efermi=args.efermi)
        dos.plot_dos(args.emin, args.emax, "Density of states", str(output))
        dos.write_dos(str(data_output))
        channels = dos.dosdata.shape[1]
        energy = dos.energy
    else:
        pdos = PDOSData.ReadFromAbacusJob(str(job), efermi=args.efermi)
        plot, write = _pdos_handlers(pdos, mode, getattr(args, "atom_index", None))
        plot(args.emin, args.emax, str(output))
        write(str(data_output))
        channels = len(pdos.projected_dos or [])
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
