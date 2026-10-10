"""The ``abacustools workflow chgdiff`` workflow."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from abacustools.data.charge import (
    combine,
    read_job_total_density,
    validate_same_grid,
)
from abacustools.data.versions import default_version
from abacustools.core.job import read_job_structure

from .common import (
    clear_generated_jobs,
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
        "-v", "--version",
        default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-o", "--output", default="charge_density_diff.cube",
        help="Output cube filename. Relative paths are resolved below JOB.",
    )


def prepare(args: argparse.Namespace) -> int:
    """Prepare full-system and subsystem SCF jobs for charge-density output."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    inputs, stru_filename, stru = read_job_structure(job)

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


def _read_total_charge_density(job: Path, version: str):
    """Read the total charge density of one prepared subsystem."""
    return read_job_total_density(
        job,
        version=version,
        require_converged=True,
        description="charge-density difference",
    )


def postprocess(args: argparse.Namespace) -> int:
    """Combine the three charge-density cubes into a difference cube."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    print(f"  job: {job}")
    task_names = ("full_system", "subsys1", "subsys2")
    read_manifest(job, "chgdiff", task_names)
    full = _read_total_charge_density(job / "full_system", args.version)
    subsystem1 = _read_total_charge_density(job / "subsys1", args.version)
    subsystem2 = _read_total_charge_density(job / "subsys2", args.version)
    validate_same_grid(full, subsystem1, "full system and subsystem 1")
    validate_same_grid(full, subsystem2, "full system and subsystem 2")

    difference = combine(combine(full, subsystem1, -1.0), subsystem2, -1.0)
    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    difference.save_cube(output)
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
    )
