"""The ``abacustools workflow chgdiff`` workflow."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .common import (
    clear_generated_jobs,
    completed_scf_output,
    kpoint_filename,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory used to prepare charge-density jobs.",
    )
    parser.add_argument(
        "-i", "--index", type=int, nargs="+", required=True,
        help="One-based atom indices belonging to subsystem 1; remaining atoms form subsystem 2.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace existing generated workflow directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory containing the prepared charge-density jobs.",
    )
    parser.add_argument(
        "-o", "--output", default="charge_density_diff.cube",
        help="Output cube filename. Relative paths are resolved below JOB.",
    )


def prepare(args: argparse.Namespace) -> int:
    """Prepare full-system and subsystem SCF jobs for charge-density output."""
    from abacustools.io.abacus import ReadInput
    from abacustools.io.stru import AbacusSTRU

    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    inputs = ReadInput(job / "INPUT")
    stru_filename = inputs.get("stru_file", "STRU")
    stru = AbacusSTRU.read(job / stru_filename)
    if stru is None:
        raise RuntimeError(f"failed to read structure: {job / stru_filename}")

    if any(index < 1 for index in args.index):
        raise ValueError("atom indices must be greater than zero")
    subsystem1 = sorted(set(index - 1 for index in args.index))
    if any(index >= stru.natoms for index in subsystem1):
        raise ValueError(
            f"atom index exceeds the number of atoms ({stru.natoms}) in {job}"
        )
    subsystem2 = sorted(set(range(stru.natoms)) - set(subsystem1))
    if not subsystem1 or not subsystem2:
        raise ValueError("charge-density difference requires two non-empty subsystems")

    kpoint_file = kpoint_filename(job, inputs)

    scf_inputs = deepcopy(inputs)
    scf_inputs["calculation"] = "scf"
    scf_inputs["out_chg"] = 1
    generated_names = ("full_system", "subsys1", "subsys2")
    print(f"  job: {job}")
    print(f"  subsystem 1 atom indices: {', '.join(str(i + 1) for i in subsystem1)}")
    print(f"  subsystem 2 atom indices: {', '.join(str(i + 1) for i in subsystem2)}")

    clear_generated_jobs(job, generated_names, override=args.override)

    structures = {
        "full_system": stru,
        "subsys1": stru.create_subset(subsystem1),
        "subsys2": stru.create_subset(subsystem2),
    }
    for name, structure in structures.items():
        generated = job / name
        write_abacus_job(
            scf_inputs,
            structure,
            job,
            generated,
            stru_filename=stru_filename,
            kpoint=kpoint_file,
        )
        print(f"  prepared {name}")

    write_manifest(
        job,
        "chgdiff",
        tasks=list(generated_names),
        subsystem1=[index + 1 for index in subsystem1],
        subsystem2=[index + 1 for index in subsystem2],
        nspin=inputs.get("nspin", 1),
    )

    return 0


def _read_total_charge_density(job: Path):
    """Read total charge density, combining spin channels when necessary."""
    from abacustools.data.grid import Grid

    inputs, output_dir = completed_scf_output(job)
    nspin = inputs.get("nspin", 1)
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
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    print(f"  job: {job}")
    task_names = ("full_system", "subsys1", "subsys2")
    read_manifest(job, "chgdiff", task_names)
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
    register_stages(
        subparsers,
        "chgdiff",
        "Calculate charge-density difference.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
        aliases=["charge-density-difference", "charge_density_difference"],
    )
