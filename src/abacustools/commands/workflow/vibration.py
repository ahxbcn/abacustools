"""The ``abacustools workflow vibration`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.core.constant import BOLTZMANN_CONSTANT_EV_PER_K
from abacustools.core.submission import generate_workflow_submission

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    read_job_structure,
    register_stages,
    write_abacus_job,
    write_manifest,
)


_VIBRATION_DIRECTORY = "vib"
_SCF_DIRECTORY = "vib/SCF"
_EQUILIBRIUM_TASK = f"{_SCF_DIRECTORY}/eq"
_DIRECTIONS = ("x", "y", "z")
_SIGNS = (("+", 1), ("-", -1))


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the molecular vibration preparation stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="ABACUS input directory used to prepare vibration calculations.",
    )
    parser.add_argument(
        "-s", "--stepsize", "--displacement-stepsize",
        dest="stepsize",
        type=float,
        default=0.01,
        help="Cartesian displacement in Angstrom, default: 0.01.",
    )
    parser.add_argument(
        "-i", "--index",
        dest="selected_atoms",
        type=int,
        nargs="+",
        help="One-based atom indices to vibrate; by default all atoms are used.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace the existing generated vibration directory.",
    )
    submission = parser.add_mutually_exclusive_group()
    submission.add_argument(
        "--submit-script",
        dest="generate_scripts",
        action="store_true",
        help="Generate configured task and workflow submission scripts.",
    )
    submission.add_argument(
        "--no-submit-script",
        dest="generate_scripts",
        action="store_false",
        help="Do not generate submission scripts, overriding the config default.",
    )
    parser.set_defaults(generate_scripts=None)
    parser.add_argument(
        "--submission-type",
        "--submit-type",
        dest="submission_type",
        help="Submission template type from the config, such as local, slurm, pbs, or lsf.",
    )
    parser.add_argument(
        "--abacus-command",
        help="ABACUS command used in generated scripts; otherwise use the config default.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the molecular vibration postprocessing stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="Directory containing the prepared vibration calculations.",
    )
    parser.add_argument(
        "-v", "--version",
        default="LTS3.10.1",
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        nargs="+",
        default=[298.15],
        help="One temperature, or START END COUNT in Kelvin, default: 298.15.",
    )
    parser.add_argument(
        "--traj",
        action="store_true",
        help="Write trajectory files for non-zero vibration modes.",
    )
    parser.add_argument(
        "--traj-format",
        choices=("extxyz", "traj"),
        default="extxyz",
        help="Trajectory format, default: extxyz.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Frames per vibration period when writing trajectories, default: 30.",
    )
    structure_group = parser.add_mutually_exclusive_group()
    structure_group.add_argument(
        "--output-stru",
        action="store_true",
        dest="output_stru",
        help="Write structures with mode velocities (default).",
    )
    structure_group.add_argument(
        "--no-output-stru",
        action="store_false",
        dest="output_stru",
        help="Do not write structures with mode velocities.",
    )
    parser.set_defaults(output_stru=True)
    parser.add_argument(
        "--stru-format",
        choices=("extxyz", "poscar"),
        default="extxyz",
        help="Structure format, default: extxyz.",
    )
    parser.add_argument(
        "-o", "--output",
        default="vibration_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )


def _validate_stepsize(stepsize: float) -> None:
    """Validate a finite positive Cartesian displacement."""
    if not np.isfinite(stepsize) or stepsize <= 0:
        raise ValueError("stepsize must be a positive finite number")


def _selected_atoms(selected_atoms: Any, natoms: int) -> list[int]:
    """Validate one-based CLI atom indices and return zero-based indices."""
    if selected_atoms is None:
        return list(range(natoms))
    if not selected_atoms:
        raise ValueError("selected atom indices must not be empty")
    if any(isinstance(index, bool) for index in selected_atoms):
        raise ValueError("atom indices must be positive integers")
    indices = [int(index) for index in selected_atoms]
    if any(index < 1 or index > natoms for index in indices):
        raise ValueError(f"atom indices must be between 1 and {natoms}")
    if len(set(indices)) != len(indices):
        raise ValueError("atom indices must not contain duplicates")
    return sorted(index - 1 for index in indices)


def _displacement_tasks(selected_atoms: list[int]) -> list[dict[str, Any]]:
    """Return the task metadata for all central finite differences."""
    tasks = []
    for atom_index in selected_atoms:
        for direction_index, direction in enumerate(_DIRECTIONS):
            for sign, sign_value in _SIGNS:
                tasks.append(
                    {
                        "task": f"{_SCF_DIRECTORY}/disp_{atom_index + 1}_{direction}{sign}",
                        "atom": atom_index + 1,
                        "direction": direction,
                        "direction_index": direction_index,
                        "sign": sign,
                        "sign_value": sign_value,
                    }
                )
    return tasks


def prepare(args: argparse.Namespace) -> int:
    """Prepare equilibrium and displaced ABACUS force calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _validate_stepsize(args.stepsize)

    inputs, stru_filename, structure = read_job_structure(job)
    selected_atoms = _selected_atoms(args.selected_atoms, structure.natoms)
    vibration_inputs = deepcopy(inputs)
    vibration_inputs["calculation"] = "scf"
    vibration_inputs["cal_force"] = 1
    try:
        scf_thr = float(vibration_inputs.get("scf_thr", 1e-7))
    except (TypeError, ValueError):
        scf_thr = 1e-7
    if scf_thr > 1e-7:
        vibration_inputs["scf_thr"] = 1e-7
    kpoint_file = kpoint_filename(job, inputs)

    displacement_tasks = _displacement_tasks(selected_atoms)
    task_names = [_EQUILIBRIUM_TASK] + [item["task"] for item in displacement_tasks]
    clear_generated_jobs(job, [_VIBRATION_DIRECTORY], override=args.override)

    equilibrium_path = job / _EQUILIBRIUM_TASK
    write_abacus_job(
        vibration_inputs,
        structure,
        job,
        equilibrium_path,
        stru_filename=stru_filename,
        kpoint=kpoint_file,
    )
    print(f"  prepared {_EQUILIBRIUM_TASK}")

    original_coords = np.asarray(structure.coords, dtype=float)
    for item in displacement_tasks:
        displaced = deepcopy(structure)
        coords = original_coords.copy()
        coords[item["atom"] - 1, item["direction_index"]] += (
            args.stepsize * item["sign_value"]
        )
        displaced.coords = coords.tolist()
        write_abacus_job(
            vibration_inputs,
            displaced,
            job,
            job / item["task"],
            stru_filename=stru_filename,
            kpoint=kpoint_file,
        )
        print(f"  prepared {item['task']}")

    submission = generate_workflow_submission(
        job,
        "vibration",
        task_names,
        submission_type=getattr(args, "submission_type", None),
        generate=getattr(args, "generate_scripts", None),
        abacus_command=getattr(args, "abacus_command", None),
    )
    if submission is not None:
        print(f"  submission type: {submission['type']}")
        print(f"  workflow script: {submission['workflow_script']}")

    manifest = dict(
        tasks=task_names,
        selected_atoms=[index + 1 for index in selected_atoms],
        stepsize=float(args.stepsize),
        displacements=displacement_tasks,
    )
    if submission is not None:
        manifest["submission"] = submission
    write_manifest(job, "vibration", **manifest)
    print(f"  job: {job}")
    print(f"  selected atoms: {', '.join(str(index + 1) for index in selected_atoms)}")
    print(f"  displacement step: {args.stepsize} Angstrom")
    print(f"  generated calculations: {len(task_names)}")
    return 0


def _read_forces(job: Path, version: str, natoms: int) -> np.ndarray:
    """Read one converged ABACUS force array."""
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(
        job,
        param_names=["force", "converged"],
        version=version,
    )
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    if result["force"] is None:
        raise RuntimeError(f"forces were not found in the output: {job}")
    forces = np.asarray(result["force"], dtype=float)
    if forces.shape != (natoms, 3) or not np.all(np.isfinite(forces)):
        raise RuntimeError(
            f"invalid force array in the output: {job}; "
            f"expected {(natoms, 3)}, got {forces.shape}"
        )
    return forces


def _hessian_from_forces(
    force_sets: dict[str, np.ndarray],
    displacement_tasks: list[dict[str, Any]],
    selected_atoms: list[int],
    stepsize: float,
) -> np.ndarray:
    """Build a symmetrized Cartesian Hessian from central force differences."""
    n_active = len(selected_atoms)
    hessian = np.zeros((3 * n_active, 3 * n_active), dtype=float)
    positions = {atom: index for index, atom in enumerate(selected_atoms)}
    grouped: dict[tuple[int, int], dict[str, str]] = {}
    for item in displacement_tasks:
        key = (item["atom"] - 1, item["direction_index"])
        grouped.setdefault(key, {})[item["sign"]] = item["task"]

    for (atom_index, direction_index), signs in grouped.items():
        if set(signs) != {"+", "-"}:
            raise RuntimeError("each vibration displacement must have plus and minus forces")
        row = 3 * positions[atom_index] + direction_index
        force_difference = (
            force_sets[signs["-"]][selected_atoms]
            - force_sets[signs["+"]][selected_atoms]
        )
        hessian[row] = force_difference.reshape(-1) / (2.0 * stepsize)
    return 0.5 * (hessian + hessian.T)


def _temperatures(values: Any) -> list[float]:
    """Normalize a single temperature or a START/END/COUNT specification."""
    values = list(values)
    if len(values) == 1:
        temperatures = [float(values[0])]
    elif len(values) == 3:
        start, end, count = values
        if not float(count).is_integer():
            raise ValueError("temperature count must be an integer")
        count = int(count)
        if count < 1:
            raise ValueError("temperature count must be positive")
        temperatures = np.linspace(float(start), float(end), count).tolist()
    else:
        raise ValueError("temperature must contain one value or START END COUNT")
    if any(not np.isfinite(value) or value <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive finite numbers")
    return temperatures


def _frequency_values(frequencies: np.ndarray) -> list[float]:
    """Represent imaginary frequencies as negative real values in cm^-1."""
    values = []
    for frequency in np.asarray(frequencies, dtype=complex):
        if abs(frequency.imag) > 1e-8:
            if abs(frequency.real) > 1e-8:
                raise RuntimeError(f"frequency has both real and imaginary parts: {frequency}")
            values.append(-abs(float(frequency.imag)))
        else:
            values.append(float(frequency.real))
    return values


def _write_modes(
    vibration_data,
    work_dir: Path,
    *,
    output_traj: bool,
    traj_format: str,
    frames: int,
    output_stru: bool,
    stru_format: str,
) -> None:
    """Write optional mode trajectories and velocity-bearing structures."""
    if frames < 1:
        raise ValueError("frames must be positive")
    if not output_traj and not output_stru:
        return

    from ase.io import write
    from ase.io.trajectory import Trajectory

    energies = vibration_data.get_energies()
    modes = vibration_data.get_modes(all_atoms=True)
    equilibrium = vibration_data.get_atoms()
    trajectory_dir = work_dir / _VIBRATION_DIRECTORY / "mode_trajectories"
    structure_dir = work_dir / _VIBRATION_DIRECTORY / "modes"
    if output_traj:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    if output_stru:
        structure_dir.mkdir(parents=True, exist_ok=True)

    kT = BOLTZMANN_CONSTANT_EV_PER_K * 300.0
    mode_number = 0
    for mode_index, energy in enumerate(energies):
        if abs(energy) <= 1e-5:
            continue
        mode_number += 1
        mode = modes[mode_index]
        mode_displacement = mode * np.sqrt(kT / abs(energy))
        frequency = abs(complex(vibration_data.get_frequencies()[mode_index]))
        phases = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
        mode_frames = []
        for phase in phases:
            image = equilibrium.copy()
            image.positions += np.sin(phase) * mode_displacement
            image.set_velocities(frequency * np.cos(phase) * mode_displacement)
            mode_frames.append(image)

        if output_traj:
            if traj_format == "traj":
                path = trajectory_dir / f"mode_{mode_number}.traj"
                with Trajectory(path, "w", mode_frames[0]) as trajectory:
                    for image in mode_frames[1:]:
                        trajectory.write(image)
            else:
                path = trajectory_dir / f"mode_{mode_number}.extxyz"
                write(path, mode_frames, format="extxyz")
            print(f"  trajectory: {path}")

        if output_stru:
            if stru_format == "poscar":
                path = structure_dir / f"mode_{mode_number}.poscar"
                write(path, mode_frames[0], format="vasp")
            else:
                path = structure_dir / f"mode_{mode_number}.xyz"
                write(path, mode_frames[0], format="extxyz")
            print(f"  mode structure: {path}")


def postprocess(args: argparse.Namespace) -> int:
    """Calculate molecular vibration frequencies from prepared force jobs."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _temperatures(args.temperature)
    if args.frames < 1:
        raise ValueError("frames must be positive")

    _, _, structure = read_job_structure(job)
    manifest = read_manifest(job, "vibration", [_EQUILIBRIUM_TASK])
    try:
        stepsize = float(manifest["stepsize"])
        selected_atoms = _selected_atoms(manifest["selected_atoms"], structure.natoms)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("vibration workflow manifest has invalid metadata") from error
    _validate_stepsize(stepsize)
    displacement_tasks = manifest.get("displacements")
    if not isinstance(displacement_tasks, list) or len(displacement_tasks) != 6 * len(selected_atoms):
        raise RuntimeError("vibration workflow manifest has invalid displacements")
    task_names = [_EQUILIBRIUM_TASK] + [item.get("task") for item in displacement_tasks]
    read_manifest(job, "vibration", task_names)

    force_sets = {
        task: _read_forces(job / task, args.version, structure.natoms)
        for task in task_names
    }
    hessian = _hessian_from_forces(
        force_sets,
        displacement_tasks,
        selected_atoms,
        stepsize,
    )

    from ase.thermochemistry import HarmonicThermo
    from ase.vibrations.data import VibrationsData

    atoms = structure.to("ase")
    vibration_data = VibrationsData(
        atoms,
        hessian.reshape(len(selected_atoms), 3, len(selected_atoms), 3),
        indices=selected_atoms,
    )
    frequencies = _frequency_values(vibration_data.get_frequencies())
    energies = vibration_data.get_energies()
    thermo = HarmonicThermo(energies, ignore_imag_modes=True)
    thermo_corr = {}
    for temperature in _temperatures(args.temperature):
        thermo_corr[f"{temperature:g}K"] = {
            "entropy": float(thermo.get_entropy(temperature)),
            "free_energy": float(thermo.get_helmholtz_energy(temperature)),
        }

    _write_modes(
        vibration_data,
        job,
        output_traj=args.traj,
        traj_format=args.traj_format,
        frames=args.frames,
        output_stru=args.output_stru,
        stru_format=args.stru_format,
    )
    result = {
        "selected_atoms": [index + 1 for index in selected_atoms],
        "stepsize": stepsize,
        "frequencies": frequencies,
        "frequency_unit": "cm^-1",
        "zero_point_energy": float(sum(abs(energy) for energy in energies) / 2.0),
        "energy_unit": "eV",
        "thermo_corr": thermo_corr,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"  job: {job}")
    print("  frequencies (cm^-1): " + " ".join(f"{value:.6f}" for value in frequencies))
    print(f"  zero-point energy: {result['zero_point_energy']:.8f} eV")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the vibration preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "vibration",
        "Calculate molecular vibration frequencies with finite differences.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
