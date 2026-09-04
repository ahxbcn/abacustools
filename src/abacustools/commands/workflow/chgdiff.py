"""The ``abacustools workflow chgdiff`` workflow."""

from __future__ import annotations

import argparse
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _job_directory(value: str) -> Path:
    """Return an existing ABACUS job directory or raise a parser error."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _atom_index(value: str) -> int:
    """Return a one-based atom index or raise a parser error."""
    try:
        index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"atom index must be an integer: {value}"
        ) from error
    if index < 1:
        raise argparse.ArgumentTypeError(
            f"atom index must be greater than zero: {value}"
        )
    return index


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j", "--job", type=_job_directory, required=True,
        help="ABACUS input directory used to prepare charge-density jobs.",
    )
    parser.add_argument(
        "-i", "--index", type=_atom_index, nargs="+", required=True,
        help="One-based atom indices belonging to subsystem 1; remaining atoms form subsystem 2.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j", "--job", type=_job_directory, required=True,
        help="Directory containing the prepared charge-density jobs.",
    )
    parser.add_argument(
        "-o", "--output", default="charge_density_diff.cube",
        help="Output cube filename. Relative paths are resolved below JOB.",
    )


def _copy_referenced_files(
    filenames: Iterable[Optional[str]], source_dir: Path, destination_dir: Path,
) -> None:
    """Link files referenced by a generated STRU into its job directory."""
    copied = set()
    for filename in filenames:
        if filename is None or filename in copied:
            continue
        copied.add(filename)
        source = source_dir / filename
        if not source.is_file():
            raise RuntimeError(f"referenced file not found: {source}")
        target = destination_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())


def prepare(args: argparse.Namespace) -> int:
    """Prepare full-system and subsystem SCF jobs for charge-density output."""
    from abacustools.io.abacus import ReadInput, WriteInput
    from abacustools.io.stru import AbacusSTRU

    job = Path(args.job).absolute()
    inputs = ReadInput(job / "INPUT")
    stru_filename = inputs.get("stru_file", "STRU")
    stru = AbacusSTRU.read(job / stru_filename)
    if stru is None:
        raise RuntimeError(f"failed to read structure: {job / stru_filename}")

    subsystem1 = sorted(set(index - 1 for index in args.index))
    if any(index >= stru.natoms for index in subsystem1):
        raise ValueError(
            f"atom index exceeds the number of atoms ({stru.natoms}) in {job}"
        )
    subsystem2 = sorted(set(range(stru.natoms)) - set(subsystem1))
    if not subsystem1 or not subsystem2:
        raise ValueError("charge-density difference requires two non-empty subsystems")

    kpoint_filename = inputs.get("kpoint_file", "KPT")
    has_kpoint_setting = inputs.get("kspacing", 0) not in (0, "0")
    has_kpoint_setting = has_kpoint_setting or bool(inputs.get("gamma_only", 0))
    if not has_kpoint_setting and not (job / kpoint_filename).is_file():
        raise RuntimeError(f"could not find KPT file: {job / kpoint_filename}")

    scf_inputs = deepcopy(inputs)
    scf_inputs["calculation"] = "scf"
    scf_inputs["out_chg"] = 1
    generated_names = ("full_system", "subsys1", "subsys2")
    print(f"  job: {job}")
    print(f"  subsystem 1 atom indices: {', '.join(str(i + 1) for i in subsystem1)}")
    print(f"  subsystem 2 atom indices: {', '.join(str(i + 1) for i in subsystem2)}")

    for name in generated_names:
        generated = job / name
        if generated.exists() or generated.is_symlink():
            print(f"  removing old directory: {generated}")
            if generated.is_dir() and not generated.is_symlink():
                shutil.rmtree(generated)
            else:
                generated.unlink()

    structures = {
        "full_system": stru,
        "subsys1": stru.create_subset(subsystem1),
        "subsys2": stru.create_subset(subsystem2),
    }
    for name, structure in structures.items():
        generated = job / name
        generated.mkdir(parents=True)
        WriteInput(scf_inputs, generated / "INPUT")
        structure.write(generated / stru_filename)
        _copy_referenced_files(
            (*structure.pps, *structure.orbs, *structure.paws), job, generated
        )
        if (job / kpoint_filename).is_file():
            _copy_referenced_files((kpoint_filename,), job, generated)
        print(f"  prepared {name}")

    return 0


def _read_total_charge_density(job: Path):
    """Read total charge density, combining spin channels when necessary."""
    from abacustools.data.grid import Grid
    from abacustools.io.abacus import ReadInput

    inputs = ReadInput(job / "INPUT")
    nspin = inputs.get("nspin", 1)
    suffix = inputs.get("suffix", "ABACUS")
    output_dir = job / f"OUT.{suffix}"
    spin1 = Grid.from_cube(output_dir / "SPIN1_CHG.cube")
    if nspin == 1:
        return spin1
    if nspin != 2:
        raise ValueError("charge-density difference supports only nspin=1 and nspin=2")

    spin2 = Grid.from_cube(output_dir / "SPIN2_CHG.cube")
    _validate_grid(spin1, spin2, "spin channels")
    spin1.data = spin1.data + spin2.data
    return spin1


def _validate_grid(reference, other, description: str) -> None:
    """Ensure two cube data sets can be combined point by point."""
    if reference.data.shape != other.data.shape:
        raise ValueError(
            f"incompatible grid shape for {description}: "
            f"{reference.data.shape} != {other.data.shape}"
        )
    if not np.allclose(reference.cell, other.cell) or not np.allclose(
        reference.origin, other.origin
    ):
        raise ValueError(f"incompatible grid geometry for {description}")


def postprocess(args: argparse.Namespace) -> int:
    """Combine the three charge-density cubes into a difference cube."""
    job = Path(args.job).absolute()
    print(f"  job: {job}")
    full = _read_total_charge_density(job / "full_system")
    subsystem1 = _read_total_charge_density(job / "subsys1")
    subsystem2 = _read_total_charge_density(job / "subsys2")
    _validate_grid(full, subsystem1, "full system and subsystem 1")
    _validate_grid(full, subsystem2, "full system and subsystem 2")

    full.data = full.data - subsystem1.data - subsystem2.data
    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    full.save_cube(output)
    print(f"  charge-density difference: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the ``workflow chgdiff`` parser and its stages."""
    parser = subparsers.add_parser(
        "chgdiff",
        aliases=["charge-density-difference", "charge_density_difference"],
        help="Calculate charge-density difference.",
    )
    stages = parser.add_subparsers(
        dest="chgdiff_command", metavar="CHGDIFF_COMMAND",
        title="charge-density difference commands", required=True,
    )
    prepare_parser = stages.add_parser("prepare", help="Prepare charge-density jobs.")
    _register_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=prepare)

    postprocess_parser = stages.add_parser(
        "postprocess", help="Generate the charge-density difference cube."
    )
    _register_postprocess_arguments(postprocess_parser)
    postprocess_parser.set_defaults(handler=postprocess)
