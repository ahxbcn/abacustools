"""The ``abacustools workflow bsse`` workflow."""

from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path

from abacustools.data.versions import default_version

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    read_job_input,
    read_job_structure,
    register_stages,
    write_abacus_job,
    write_manifest,
)


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the BSSE preparation stage."""
    parser.add_argument(
        "-j", "--job",
        action="extend",
        nargs="+",
        type=Path,
        required=True,
        help="Input ABACUS job directory to be prepared for BSSE calculation.",
    )
    parser.add_argument(
        "-i", "--index",
        type=int,
        nargs="+",
        required=True,
        help="Index of atoms (started from 1) to indicate a subsystem of the substructure. The other atoms" \
        " will be considered as the another subsystem, and input files for calculations used to calculate" \
        " the bsse-corrected interaction energy will be generated.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace existing generated workflow directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the BSSE postprocessing stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="ABACUS job directory with prepared files for BSSE calculation.",
    )
    parser.add_argument(
        "-v", "--version",
        default=default_version(),
        help="ABACUS version used for the calculation.",
    )


def prepare(args: argparse.Namespace) -> int:
    """Run the BSSE preparation stage."""
    job_names = [
        "full_system",
        "subsys_a_ghost_b",
        "subsys_ghost_a_b",
        "subsys_a",
        "subsys_b",
    ]
    # The command accepts one-based atom numbers; AbacusSTRU uses zero-based
    # list indices for create_subset() and atoms.
    if any(index < 1 for index in args.index):
        raise ValueError("atom indices must be greater than zero")
    subsys_a_indices = sorted(set(index - 1 for index in args.index))
    print(f"  subsystem atom indices: {', '.join(str(index + 1) for index in subsys_a_indices)}")

    for job in args.job:
        job = Path(job).absolute()
        if not job.is_dir():
            raise RuntimeError(f"job directory does not exist: {job}")
        print(f"  job: {job}")

        inputs, stru_filename, stru = read_job_structure(job)
        if inputs.get("basis_type", "pw") != "lcao":
            print(
                "BSSE correction generally needs the LCAO basis type. "
                f"Current basis type: {inputs.get('basis_type')}"
            )
        scf_inputs = deepcopy(inputs)
        scf_inputs["calculation"] = "scf"
        kpoint_file = kpoint_filename(job, inputs)
        if any(index >= stru.natoms for index in subsys_a_indices):
            raise ValueError(
                f"atom index exceeds the number of atoms ({stru.natoms}) in {job}"
            )
        subsys_b_indices = sorted(set(range(stru.natoms)) - set(subsys_a_indices))
        if not subsys_b_indices:
            raise ValueError("BSSE requires at least one atom in subsystem B")

        clear_generated_jobs(job, job_names, override=args.override)

        def write_job(job_name: str, structure) -> None:
            write_abacus_job(
                scf_inputs,
                structure,
                job,
                job / job_name,
                stru_filename=stru_filename,
                kpoint=kpoint_file,
            )

        print("  writing full-system calculation")
        write_job("full_system", stru)

        print("  writing A + ghost B calculation")
        stru_a_ghost_b = deepcopy(stru)
        for index in subsys_b_indices:
            stru_a_ghost_b.atoms[index].label += "_empty"
        write_job("subsys_a_ghost_b", stru_a_ghost_b)

        print("  writing ghost A + B calculation")
        stru_ghost_a_b = deepcopy(stru)
        for index in subsys_a_indices:
            stru_ghost_a_b.atoms[index].label += "_empty"
        write_job("subsys_ghost_a_b", stru_ghost_a_b)

        print("  writing subsystem A calculation")
        write_job("subsys_a", stru.create_subset(subsys_a_indices))

        print("  writing subsystem B calculation")
        write_job("subsys_b", stru.create_subset(subsys_b_indices))

        write_manifest(
            job,
            "bsse",
            tasks=job_names,
            subsystem_a=[index + 1 for index in subsys_a_indices],
            subsystem_b=[index + 1 for index in subsys_b_indices],
            nspin=inputs.get("nspin", 1),
        )

    return 0


def postprocess(args: argparse.Namespace) -> int:
    """Run the BSSE postprocessing stage."""
    from abacustools.data.abacus_result import get_result_from_job

    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    print(f"  job: {job}")
    job_names = [
        "full_system",
        "subsys_a_ghost_b",
        "subsys_ghost_a_b",
        "subsys_a",
        "subsys_b",
    ]
    read_manifest(job, "bsse", job_names)
    jobs = {
        name: get_result_from_job(
            job / directory,
            param_names=["energy", "converged", "vdw_energy"],
            version=args.version,
        )
        for name, directory in {
            "full system": "full_system",
            "A + ghost B": "subsys_a_ghost_b",
            "ghost A + B": "subsys_ghost_a_b",
            "A": "subsys_a",
            "B": "subsys_b",
        }.items()
    }

    for name, result in jobs.items():
        if not result["converged"]:
            raise RuntimeError(f"{name} calculation did not converge")
        if result["energy"] is None:
            raise RuntimeError(f"{name} calculation has no total energy")

    ghost_a_b_energy = jobs["A + ghost B"]["energy"]
    ghost_b_a_energy = jobs["ghost A + B"]["energy"]
    normalized_version = re.sub(r"[^0-9a-z]", "", args.version.lower())
    dftd_method = str(read_job_input(job).get("vdw_method", "none")).lower()
    uses_dftd = dftd_method not in {"", "none", "off", "false", "0"}
    if normalized_version in {"lts3101", "3101"} and uses_dftd:
        for ghost_name, reference_name in (
            ("A + ghost B", "A"),
            ("ghost A + B", "B"),
        ):
            ghost = jobs[ghost_name]
            reference = jobs[reference_name]
            if ghost["vdw_energy"] is None or reference["vdw_energy"] is None:
                raise RuntimeError(
                    f"cannot correct DFT-D energy for {ghost_name}: "
                    "dispersion energy is missing"
                )
            corrected_energy = (
                ghost["energy"]
                - ghost["vdw_energy"]
                + reference["vdw_energy"]
            )
            print(
                f"  corrected {ghost_name} DFT-D energy: "
                f"{ghost['vdw_energy']:.10f} -> {reference['vdw_energy']:.10f} eV"
            )
            if ghost_name == "A + ghost B":
                ghost_a_b_energy = corrected_energy
            else:
                ghost_b_a_energy = corrected_energy
        print(f"  applied LTS 3.10.1 DFT-D ghost-atom correction")

    e_orig_interaction = (
        jobs["full system"]["energy"]
        - jobs["A"]["energy"]
        - jobs["B"]["energy"]
    )
    e_corr_interaction = (
        jobs["full system"]["energy"] - ghost_a_b_energy - ghost_b_a_energy
    )
    bsse_correction = e_corr_interaction - e_orig_interaction
    print(f"  Original interaction energy: {e_orig_interaction:.8f} eV")
    print(f"  BSSE-corrected interaction energy: {e_corr_interaction:.8f} eV")
    print(f"  BSSE correction: {bsse_correction:.8f} eV")
    return 0

def register_parser(subparsers) -> None:
    """Register the BSSE preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "bsse",
        "Run a basis set superposition error (BSSE) workflow.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
