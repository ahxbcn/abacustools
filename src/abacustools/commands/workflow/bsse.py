"""The ``abacustools workflow bsse`` workflow."""

from __future__ import annotations

import argparse
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional


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
    """Register arguments for the BSSE preparation stage."""
    parser.add_argument(
        "-j", "--job",
        default=[],
        action="extend",
        nargs="*", 
        required=True,
        help="Input ABACUS job directory to be prepared for BSSE calculation.",
    )
    parser.add_argument(
        "-i", "--index",
        type=_atom_index,
        nargs="+",
        required=True,
        help="Index of atoms (started from 1) to indicate a subsystem of the substructure. The other atoms" \
        " will be considered as the another subsystem, and input files for calculations used to calculate" \
        " the bsse-corrected interaction energy will be generated.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the BSSE postprocessing stage."""
    parser.add_argument(
        "-j", "--job",
        type=_job_directory,
        required=True,
        help="ABACUS job directory with prepared files for BSSE calculation.",
    )
    parser.add_argument(
        "-v", "--version",
        default="LTS3.10.1",
        help="ABACUS version used for the calculation.",
    )


def _is_lts_3_10_1(version: str) -> bool:
    """Return whether version has the LTS 3.10.1 DFT-D bug."""
    normalized = re.sub(r"[^0-9a-z]", "", version.lower())
    return normalized in {"lts3101", "3101"}


def _uses_dftd(job: Path) -> bool:
    """Return whether an ABACUS job enables a DFT-D correction."""
    from abacustools.io.abacus import ReadInput

    method = str(ReadInput(job / "INPUT").get("vdw_method", "none")).lower()
    return method not in {"", "none", "off", "false", "0"}


def _result(job: Path, version: str):
    """Read the BSSE quantities needed from one completed calculation."""
    from abacustools.data.abacus_result import get_result_from_job

    return get_result_from_job(
        job,
        param_names=["energy", "converged", "vdw_energy"],
        version=version,
    )


def prepare(args: argparse.Namespace) -> int:
    """Run the BSSE preparation stage."""
    from abacustools.io.abacus import ReadInput, WriteInput
    from abacustools.io.stru import AbacusSTRU

    def copy_pp_orb(
        pp_files: Iterable[Optional[str]],
        orb_files: Iterable[Optional[str]],
        original_dir: Path,
        dest_dir: Path,
        link: bool = True,
    ) -> None:
        """Copy or link the files referenced by a generated STRU file."""
        files = [
            *[(filename, "pseudopotential") for filename in pp_files],
            *[(filename, "orbital") for filename in orb_files],
        ]
        copied = set()
        for filename, file_type in files:
            if filename is None or filename in copied:
                continue
            copied.add(filename)
            source = original_dir / filename
            target = dest_dir / filename
            if not source.is_file():
                raise RuntimeError(f"{file_type} file not found: {source}")
            if target.exists() or target.is_symlink():
                target.unlink()
            if link:
                # Resolve source links so the generated job has a valid link.
                target.symlink_to(source.resolve())
            else:
                shutil.copy2(source, target)

    job_dirnames = [
        "full_system",
        "subsys_a_ghost_b",
        "subsys_ghost_a_b",
        "subsys_a",
        "subsys_b",
    ]
    # The command accepts one-based atom numbers; AbacusSTRU uses zero-based
    # list indices for create_subset() and atoms.
    subsys_a_indices = sorted(set(index - 1 for index in args.index))
    print(f"  subsystem atom indices: {', '.join(str(index + 1) for index in subsys_a_indices)}")

    for job in args.job:
        job = Path(job).absolute()
        print(f"  job: {job}")

        for job_dirname in job_dirnames:
            generated_job = job / job_dirname
            if generated_job.exists():
                print(f"  removing old directory: {generated_job}")
                shutil.rmtree(generated_job)

        inputs = ReadInput(job / "INPUT")
        if inputs.get("basis_type", "pw") != "lcao":
            print(
                "BSSE correction generally needs the LCAO basis type. "
                f"Current basis type: {inputs.get('basis_type')}"
            )
        scf_inputs = deepcopy(inputs)
        scf_inputs["calculation"] = "scf"
        stru_filename = inputs.get("stru_file", "STRU")
        stru = AbacusSTRU.read(job / stru_filename)
        if stru is None:
            raise RuntimeError(f"failed to read structure in job: {job}")
        if any(index >= stru.natoms for index in subsys_a_indices):
            raise ValueError(
                f"atom index exceeds the number of atoms ({stru.natoms}) in {job}"
            )
        subsys_b_indices = sorted(set(range(stru.natoms)) - set(subsys_a_indices))
        if not subsys_b_indices:
            raise ValueError("BSSE requires at least one atom in subsystem B")

        def write_job(job_name: str, structure: AbacusSTRU) -> None:
            generated_job = job / job_name
            generated_job.mkdir(parents=True, exist_ok=True)
            WriteInput(scf_inputs, generated_job / "INPUT")
            structure.write(generated_job / stru_filename)
            copy_pp_orb(structure.pps, structure.orbs, job, generated_job)

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

    return 0


def postprocess(args: argparse.Namespace) -> int:
    """Run the BSSE postprocessing stage."""
    job = Path(args.job)
    print(f"  job: {job}")
    jobs = {
        "full system": _result(job / "full_system", args.version),
        "A + ghost B": _result(job / "subsys_a_ghost_b", args.version),
        "ghost A + B": _result(job / "subsys_ghost_a_b", args.version),
        "A": _result(job / "subsys_a", args.version),
        "B": _result(job / "subsys_b", args.version),
    }

    for name, result in jobs.items():
        if not result["converged"]:
            raise RuntimeError(f"{name} calculation did not converge")
        if result["energy"] is None:
            raise RuntimeError(f"{name} calculation has no total energy")

    ghost_a_b_energy = jobs["A + ghost B"]["energy"]
    ghost_b_a_energy = jobs["ghost A + B"]["energy"]
    if _is_lts_3_10_1(args.version) and _uses_dftd(job):
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
    bsse_parser = subparsers.add_parser(
        "bsse",
        help="Run a basis set superposition error (BSSE) workflow.",
    )
    bsse_subparsers = bsse_parser.add_subparsers(
        dest="bsse_command",
        metavar="BSSE_COMMAND",
        title="BSSE commands",
        required=True,
    )
    prepare_parser = bsse_subparsers.add_parser(
        "prepare",
        help="Prepare ABACUS jobs for a BSSE calculation.",
    )
    _register_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=prepare)

    postprocess_parser = bsse_subparsers.add_parser(
        "postprocess",
        help="Collect and process results from a BSSE calculation.",
    )
    _register_postprocess_arguments(postprocess_parser)
    postprocess_parser.set_defaults(handler=postprocess)
