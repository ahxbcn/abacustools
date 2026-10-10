"""The ``abacustools workflow vibration`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from abacustools.core.constant import (
    AMU_TO_KG,
    ANGSTROM_TO_METRE,
    BOLTZMANN_CONSTANT_EV_PER_K,
    ELEMENTARY_CHARGE,
    INV_CM_TO_EV,
)
from abacustools.core.submission import generate_workflow_submission
from abacustools.data.versions import default_version
from abacustools.data.phonon import read_forces
from abacustools.data.vibration import (
    HarmonicVibration,
    selected_atom_indices,
    validate_stepsize,
    write_gaussian_frequency_log,
)
from abacustools.integrations.ase_vibration import AseVibrationData
from abacustools.io.stru import write_poscar
from abacustools.io.xyz import write_extxyz
from abacustools.core.job import read_job_structure

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    register_stages,
    resolve_output,
    write_abacus_job,
    write_manifest,
)


_VIBRATION_DIRECTORY = "vib"
_SCF_DIRECTORY = "vib/SCF"
_EQUILIBRIUM_TASK = f"{_SCF_DIRECTORY}/eq"
_DIRECTIONS = ("x", "y", "z")
_SIGNS = (("+", 1), ("-", -1))

#: Harmonic analysis backends of the postprocessing stage.  The ASE classes
#: stay the default; the built-in analysis is the extension point for features
#: that ASE does not provide, such as reduced masses and force constants.
_BACKENDS = ("ase", "builtin")

#: Temperature of the harmonic amplitudes used for the mode animations.
_ANIMATION_TEMPERATURE = 300.0

#: Reference mode of the animation velocity scaling: a mode at this wavenumber
#: moves its atoms with ``_ANIMATION_PEAK_VELOCITY`` at the turning point.
_ANIMATION_REFERENCE_FREQUENCY = 2500.0
_ANIMATION_PEAK_VELOCITY = 0.5

#: Length of the velocity unit of the mode animations in femtoseconds.  The
#: display velocities follow the internal unit of ASE, ``sqrt(amu) Angstrom /
#: sqrt(eV)``, which the POSCAR velocity block converts to Angstrom/fs.
_ANIMATION_TIME_UNIT_FS = np.sqrt(
    AMU_TO_KG * ANGSTROM_TO_METRE ** 2 / ELEMENTARY_CHARGE
) * 1.0e15


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
        default=default_version(),
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
        "--backend",
        choices=_BACKENDS,
        default="ase",
        help="Harmonic analysis backend, default: ase. 'builtin' uses the "
        "analysis of abacustools.data.vibration, which also reports reduced "
        "masses and force constants of the modes.",
    )
    parser.add_argument(
        "--mass",
        "--element-mass",
        dest="element_masses",
        nargs="+",
        action="extend",
        metavar="ELEMENT=MASS",
        help="Relative atomic mass of one or more elements, such as "
        "--mass H=2.014 or --mass H=2.014 O=18.0, which accounts for isotope "
        "effects. By default the masses of the ATOMIC_SPECIES block are used.",
    )
    parser.add_argument(
        "--gaussian-log",
        nargs="?",
        const="gaussian_fake.log",
        default=None,
        metavar="FILE",
        help="Write a fake Gaussian frequency log that GaussView can open. "
        "FILE defaults to gaussian_fake.log below JOB; omit the option to skip it.",
    )
    parser.add_argument(
        "--no-cell",
        action="store_true",
        help="Leave the cell out of the fake Gaussian log, which is written as "
        "Gaussian translation vectors by default.",
    )
    parser.add_argument(
        "-o", "--output",
        default="vibration_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )


def _element_mass_overrides(values: Any) -> dict[str, float]:
    """Parse ``ELEMENT=MASS`` assignments into relative atomic mass overrides.

    Args:
        values: Command line assignments, or None when the option is unused.

    Returns:
        dict: Element symbol or atom label and the relative atomic mass in amu
        that replaces the mass read from the structure.

    Raises:
        ValueError: If an assignment is malformed, is given twice, or does not
            hold a positive finite mass.
    """
    if values is None:
        return {}
    overrides: dict[str, float] = {}
    for value in values:
        name, separator, text = str(value).partition("=")
        name = name.strip()
        if not separator or not name or not text.strip():
            raise ValueError(
                f"mass override must be written as ELEMENT=MASS, got {value!r}"
            )
        try:
            mass = float(text)
        except ValueError as error:
            raise ValueError(f"invalid mass in {value!r}") from error
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"mass of {name} must be a positive finite number")
        if name in overrides:
            raise ValueError(f"mass of {name} is set more than once")
        overrides[name] = mass
    return overrides


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
    validate_stepsize(args.stepsize)

    inputs, stru_filename, structure = read_job_structure(job)
    selected_atoms = selected_atom_indices(args.selected_atoms, structure.natoms)
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


def _frequency_label(frequency: complex) -> str:
    """Return a mode label in cm^-1, marking an imaginary mode with ``i``."""
    frequency = complex(frequency)
    if abs(frequency.imag) > 1e-8 and abs(frequency.real) <= 1e-8:
        return f"{abs(frequency.imag):.2f}i"
    return f"{abs(frequency.real):.2f}"


def _animation_velocity_scale(kT: float) -> float:
    """Return the factor that turns a phase velocity into a display velocity.

    The velocities written for the mode animations are not the physical ones.
    They are rescaled so that a reference mode of
    ``_ANIMATION_REFERENCE_FREQUENCY`` wavenumbers reaches
    ``_ANIMATION_PEAK_VELOCITY`` at its turning point, which keeps the
    animation speed of light and heavy modes within a readable range.

    Args:
        kT: Thermal energy of the animation temperature, in eV.

    Returns:
        float: Multiplier applied to the phase velocity of every mode.
    """
    return _ANIMATION_PEAK_VELOCITY / np.sqrt(
        _ANIMATION_REFERENCE_FREQUENCY * kT / INV_CM_TO_EV
    )


def _momenta_note() -> str:
    """Return the note that explains the momenta column of extxyz files."""
    return (
        "Note: extxyz stores the atomic velocities in the 'momenta' property, "
        "which ASE writes as mass-weighted velocities; read them back with "
        "atoms.get_velocities(). In Ovito the 'momenta' column can be shown as "
        "velocity arrows. The velocities are scaled for visualization, not "
        "physical time propagation."
    )


def _structure_metadata(structure) -> dict[str, Any]:
    """Return the ABACUS data written to the comment line of mode structures.

    The pseudopotentials, orbitals, PAW datasets and numerical descriptors of
    the equilibrium structure are carried over, so a mode structure can be
    converted back into a usable ABACUS ``STRU``.
    """
    return {
        "pp": structure.pp_dict(),
        "orb": structure.orb_dict(),
        "paw": structure.paw_dict(),
        "dpks": structure.dpks,
    }


def _mode_animations(
    energies: np.ndarray,
    frequencies: np.ndarray,
    modes: np.ndarray,
    coordinates: np.ndarray,
    frames: int,
) -> Iterator[tuple[int, str, np.ndarray, np.ndarray]]:
    """Yield the animation of every mode with a non-zero energy.

    Args:
        energies: Mode energies in eV, imaginary for unstable modes.
        frequencies: Mode frequencies in cm^-1, imaginary for unstable modes.
        modes: Cartesian displacement modes of all atoms of the structure.
        coordinates: Equilibrium Cartesian coordinates in Angstrom.
        frames: Number of frames of one full vibration period.

    Yields:
        tuple: The one-based mode number, the mode label in cm^-1, the
        ``(frames, natoms, 3)`` positions and the ``(frames, natoms, 3)``
        velocities of the mode.  The first frame is the equilibrium geometry at
        the highest velocity, so its velocities describe the mode itself.
    """
    kT = BOLTZMANN_CONSTANT_EV_PER_K * _ANIMATION_TEMPERATURE
    velocity_scale = _animation_velocity_scale(kT)
    phases = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    mode_number = 0
    for mode_index, energy in enumerate(energies):
        if abs(energy) <= 1e-5:
            continue
        mode_number += 1
        displacement = modes[mode_index] * np.sqrt(kT / abs(energy))
        frequency = abs(complex(frequencies[mode_index]))
        yield (
            mode_number,
            _frequency_label(frequencies[mode_index]),
            coordinates + np.sin(phases)[:, np.newaxis, np.newaxis] * displacement,
            (velocity_scale * frequency * np.cos(phases))[:, np.newaxis, np.newaxis]
            * displacement,
        )


def _mode_output_directories(
    work_dir: Path,
    *,
    output_traj: bool,
    output_stru: bool,
) -> tuple[Path, Path]:
    """Return the mode trajectory and mode structure directories of a job."""
    trajectory_dir = work_dir / _VIBRATION_DIRECTORY / "mode_trajectories"
    structure_dir = work_dir / _VIBRATION_DIRECTORY / "modes"
    if output_traj:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    if output_stru:
        structure_dir.mkdir(parents=True, exist_ok=True)
    return trajectory_dir, structure_dir


def _write_modes_ase(
    vibration: AseVibrationData,
    work_dir: Path,
    *,
    output_traj: bool,
    traj_format: str,
    frames: int,
    output_stru: bool,
    stru_format: str,
) -> None:
    """Write mode trajectories and structures through ASE.

    The ASE backend writes its files with ASE itself, so the output stays the
    one of the original implementation.
    """
    if frames < 1:
        raise ValueError("frames must be positive")
    if not output_traj and not output_stru:
        return

    from ase.io import write
    from ase.io.trajectory import Trajectory

    equilibrium = vibration.atoms
    trajectory_dir, structure_dir = _mode_output_directories(
        work_dir,
        output_traj=output_traj,
        output_stru=output_stru,
    )
    if (output_traj and traj_format == "extxyz") or (output_stru and stru_format == "extxyz"):
        print(f"  {_momenta_note()}")

    animations = _mode_animations(
        vibration.energies,
        vibration.frequencies,
        vibration.modes(),
        np.asarray(equilibrium.positions, dtype=float),
        frames,
    )
    for mode_number, label, positions, velocities in animations:
        if output_traj:
            images = []
            for index in range(frames):
                image = equilibrium.copy()
                image.positions = positions[index]
                image.set_velocities(velocities[index])
                images.append(image)
            if traj_format == "traj":
                path = trajectory_dir / f"mode_{mode_number}.traj"
                with Trajectory(path, "w") as trajectory:
                    for image in images:
                        trajectory.write(image)
            else:
                path = trajectory_dir / f"mode_{mode_number}.extxyz"
                write(path, images, format="extxyz")
            print(f"  trajectory for mode {mode_number} ({label} cm^-1): {path}")

        if output_stru:
            image = equilibrium.copy()
            image.set_velocities(velocities[0])
            if stru_format == "poscar":
                path = structure_dir / f"mode_{mode_number}.poscar"
                write(path, image, format="vasp")
            else:
                path = structure_dir / f"mode_{mode_number}.xyz"
                write(path, image, format="extxyz")
            print(f"  structure for mode {mode_number} ({label} cm^-1): {path}")


def _write_ase_trajectory(
    path: Path,
    frames: list[dict[str, Any]],
    cell: np.ndarray,
) -> None:
    """Write animation frames as an ASE binary trajectory.

    The binary ``traj`` format belongs to ASE, so this optional output keeps a
    lazy ASE import; the built-in analysis itself does not use ASE.
    """
    from ase import Atoms
    from ase.io.trajectory import Trajectory

    images = []
    for frame in frames:
        atoms = Atoms(
            symbols=frame["elements"],
            positions=frame["positions"],
            cell=cell,
            pbc=True,
        )
        atoms.set_velocities(frame["velocities"])
        images.append(atoms)
    with Trajectory(path, "w") as trajectory:
        for image in images:
            trajectory.write(image)


def _frame_dict(
    elements: list[str],
    positions: np.ndarray,
    velocities: np.ndarray,
    cell: np.ndarray,
    masses: np.ndarray,
    magmoms: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Complete one animation frame for the extended XYZ writer.

    The writer stores velocities as momenta, the mass-weighted velocities that
    ASE uses in the ``momenta`` column of an extended XYZ file.  Magnetic
    moments travel along with the geometry, so a mode structure keeps every
    piece of information that is needed to rebuild an ABACUS ``STRU``.
    """
    return {
        "elements": elements,
        "positions": positions,
        "cell": cell,
        "pbc": (True, True, True),
        "magmoms": magmoms,
        "momenta": masses[:, np.newaxis] * np.asarray(velocities, dtype=float),
        "info": metadata,
    }


def _write_modes_builtin(
    vibration: HarmonicVibration,
    structure,
    work_dir: Path,
    *,
    output_traj: bool,
    traj_format: str,
    frames: int,
    output_stru: bool,
    stru_format: str,
) -> None:
    """Write mode trajectories and structures with the built-in writers."""
    if frames < 1:
        raise ValueError("frames must be positive")
    if not output_traj and not output_stru:
        return

    coordinates = np.asarray(structure.coords, dtype=float)
    cell = np.asarray(structure.cell, dtype=float)
    elements = list(structure.elements)
    masses = np.asarray(structure.masses, dtype=float)
    magmoms = np.asarray(structure.atom_mags, dtype=float)
    metadata = _structure_metadata(structure)
    trajectory_dir, structure_dir = _mode_output_directories(
        work_dir,
        output_traj=output_traj,
        output_stru=output_stru,
    )

    if (output_traj and traj_format == "extxyz") or (output_stru and stru_format == "extxyz"):
        print(f"  {_momenta_note()}")

    animations = _mode_animations(
        vibration.energies,
        vibration.frequencies,
        vibration.modes_all_atoms(),
        coordinates,
        frames,
    )
    for mode_number, label, positions, velocities in animations:
        if output_traj:
            if traj_format == "traj":
                path = trajectory_dir / f"mode_{mode_number}.traj"
                _write_ase_trajectory(
                    path,
                    [
                        {
                            "elements": elements,
                            "positions": positions[index],
                            "velocities": velocities[index],
                        }
                        for index in range(frames)
                    ],
                    cell,
                )
            else:
                path = trajectory_dir / f"mode_{mode_number}.extxyz"
                write_extxyz(
                    path,
                    [
                        _frame_dict(
                            elements,
                            positions[index],
                            velocities[index],
                            cell,
                            masses,
                            magmoms,
                            metadata,
                        )
                        for index in range(frames)
                    ],
                )
            print(f"  trajectory for mode {mode_number} ({label} cm^-1): {path}")

        if output_stru:
            if stru_format == "poscar":
                path = structure_dir / f"mode_{mode_number}.poscar"
                write_poscar(
                    cell=cell.tolist(),
                    coord=coordinates.tolist(),
                    label=elements,
                    poscar=str(path),
                    direct=False,
                    velocities=(velocities[0] / _ANIMATION_TIME_UNIT_FS).tolist(),
                )
            else:
                path = structure_dir / f"mode_{mode_number}.xyz"
                write_extxyz(
                    path,
                    _frame_dict(
                        elements,
                        coordinates,
                        velocities[0],
                        cell,
                        masses,
                        magmoms,
                        metadata,
                    ),
                )
            print(f"  structure for mode {mode_number} ({label} cm^-1): {path}")


def _equilibrium_energy(job: Path, version: str) -> float | None:
    """Return the electronic energy of the equilibrium job in eV, or None."""
    from abacustools.data.abacus_result import get_result_from_job

    try:
        result = get_result_from_job(
            str(job / _EQUILIBRIUM_TASK), param_names=["energy"], version=version
        )
    except Exception:
        return None
    energy = result.get("energy")
    return None if energy is None else float(energy)


def postprocess(args: argparse.Namespace) -> int:
    """Calculate harmonic frequencies and thermochemistry of prepared force jobs.

    The harmonic analysis uses the ASE vibration classes by default and the
    built-in analysis of :mod:`abacustools.data.vibration` with
    ``--backend builtin``.  Both backends use the relative atomic masses of the
    structure, which ``--mass ELEMENT=MASS`` can replace for isotope effects.
    """
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    temperatures = _temperatures(args.temperature)
    if args.frames < 1:
        raise ValueError("frames must be positive")
    backend = args.backend
    mass_overrides = _element_mass_overrides(getattr(args, "element_masses", None))

    _, _, structure = read_job_structure(job)
    # Resolve the masses before reading the force sets, so that a typo in
    # --mass fails fast.
    all_masses = (
        structure.masses_with_overrides(mass_overrides) if mass_overrides else None
    )
    manifest = read_manifest(job, "vibration", [_EQUILIBRIUM_TASK])
    try:
        stepsize = float(manifest["stepsize"])
        selected_atoms = selected_atom_indices(manifest["selected_atoms"], structure.natoms)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("vibration workflow manifest has invalid metadata") from error
    validate_stepsize(stepsize)
    displacement_tasks = manifest.get("displacements")
    if not isinstance(displacement_tasks, list) or len(displacement_tasks) != 6 * len(selected_atoms):
        raise RuntimeError("vibration workflow manifest has invalid displacements")
    task_names = [_EQUILIBRIUM_TASK] + [item.get("task") for item in displacement_tasks]
    read_manifest(job, "vibration", task_names)

    force_sets = {
        task: read_forces(job / task, args.version, structure.natoms)
        for task in task_names
    }
    hessian = _hessian_from_forces(
        force_sets,
        displacement_tasks,
        selected_atoms,
        stepsize,
    )
    mode_options = {
        "output_traj": args.traj,
        "traj_format": args.traj_format,
        "frames": args.frames,
        "output_stru": args.output_stru,
        "stru_format": args.stru_format,
    }
    if backend == "ase":
        ase_vibration = AseVibrationData(
            structure,
            hessian,
            indices=selected_atoms,
            masses=all_masses,
        )
        frequencies = ase_vibration.signed_frequencies
        zero_point_energy = ase_vibration.zero_point_energy()
        thermo_corr = {
            f"{temperature:g}K": ase_vibration.thermo(temperature)
            for temperature in temperatures
        }
        _write_modes_ase(ase_vibration, job, **mode_options)
        extra: dict[str, Any] = {}
    else:
        vibration = HarmonicVibration.from_structure(
            structure,
            hessian,
            indices=selected_atoms,
            masses=all_masses,
        )
        frequencies = vibration.signed_frequencies
        zero_point_energy = vibration.zero_point_energy()
        thermo_corr = {}
        for temperature in temperatures:
            thermo = vibration.thermo(temperature)
            thermo_corr[f"{temperature:g}K"] = {
                "entropy": thermo.entropy,
                "free_energy": thermo.free_energy,
                "internal_energy": thermo.internal_energy,
                "heat_capacity": thermo.heat_capacity,
                "n_imaginary_modes": thermo.n_imaginary_modes,
            }
        _write_modes_builtin(vibration, structure, job, **mode_options)
        extra = {"modes": vibration.summary()}

    per_atom_masses = all_masses if all_masses is not None else structure.masses
    if getattr(args, "gaussian_log", None):
        mode_array = (
            ase_vibration.modes() if backend == "ase" else vibration.modes_all_atoms()
        )
        path = write_gaussian_frequency_log(
            resolve_output(job, args.gaussian_log),
            structure,
            frequencies,
            mode_array,
            masses=per_atom_masses,
            temperature=temperatures[0],
            electronic_energy=_equilibrium_energy(job, args.version),
            zero_point_energy=zero_point_energy,
            thermo=thermo_corr.get(f"{temperatures[0]:g}K"),
            periodic=not getattr(args, "no_cell", False),
        )
        print(f"  fake Gaussian log: {path}")
    result = {
        "selected_atoms": [index + 1 for index in selected_atoms],
        "masses": [float(per_atom_masses[index]) for index in selected_atoms],
        "stepsize": stepsize,
        "frequencies": frequencies,
        "frequency_unit": "cm^-1",
        "zero_point_energy": zero_point_energy,
        "energy_unit": "eV",
        "thermo_corr": thermo_corr,
        **extra,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"  job: {job}")
    print("  frequencies (cm^-1): " + " ".join(f"{value:.6f}" for value in frequencies))
    imaginary = [index for index, value in enumerate(frequencies) if value < 0]
    if imaginary:
        print(
            "  imaginary modes: "
            + ", ".join(f"{index + 1} ({frequencies[index]:.2f} cm^-1)" for index in imaginary)
        )
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
