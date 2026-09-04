"""The ``abacustools workflow phonon`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)


_THZ_TO_K = 47.9924
_TASK_PREFIX = "disp-"


def _json_argument(value: str) -> Any:
    """Parse a JSON command-line argument."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error.msg}") from error


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the phonon preparation stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="ABACUS input directory used to prepare phonon calculations.",
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        metavar=("A", "B", "C"),
        help="Positive supercell repetitions along the three lattice vectors.",
    )
    parser.add_argument(
        "--displacement-stepsize",
        type=float,
        default=0.01,
        help="Finite-difference displacement in Angstrom, default: 0.01.",
    )
    parser.add_argument(
        "--min-supercell-length",
        type=float,
        default=10.0,
        help="Minimum lattice-vector length for an automatic supercell, default: 10.0 Angstrom.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace existing generated displacement directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the phonon postprocessing stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="Directory containing the prepared phonon calculations.",
    )
    parser.add_argument(
        "-v", "--version",
        default="LTS3.10.1",
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature for thermal properties in Kelvin, default: 298.15.",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        default=[20, 20, 20],
        metavar=("A", "B", "C"),
        help="Phonon mesh dimensions, default: 20 20 20.",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=101,
        help="Number of points used for each automatic/custom band segment.",
    )
    parser.add_argument(
        "--qpath",
        type=_json_argument,
        help='Custom q-path as JSON, e.g. `["G", "X", "G"]`.',
    )
    parser.add_argument(
        "--high-symm-points",
        type=_json_argument,
        help='Custom high-symmetry points as JSON, e.g. `{"G": [0, 0, 0]}`.',
    )
    parser.add_argument(
        "-o", "--output",
        default="phonon_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )
    parser.add_argument(
        "--plot",
        default="phonon_dispersion_dos.png",
        help="Band and DOS plot filename. Relative paths are resolved below JOB.",
    )


def _validate_positive_float(value: float, name: str, *, allow_zero: bool = False) -> None:
    """Validate a finite positive command parameter."""
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} finite number")


def _validate_supercell(supercell: Any) -> list[int]:
    """Validate and normalize a three-dimensional supercell."""
    if supercell is None:
        raise ValueError("supercell must be specified before validation")
    values = list(supercell)
    if len(values) != 3 or any(isinstance(value, bool) for value in values):
        raise ValueError("supercell must contain three positive integers")
    try:
        normalized = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("supercell must contain three positive integers") from error
    if any(value != original or value <= 0 for value, original in zip(normalized, values)):
        raise ValueError("supercell must contain three positive integers")
    return normalized


def _automatic_supercell(structure, min_supercell_length: float) -> list[int]:
    """Choose diagonal supercell repetitions from the lattice-vector lengths."""
    _validate_positive_float(min_supercell_length, "min_supercell_length")
    lengths = np.linalg.norm(np.asarray(structure.cell, dtype=float), axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("structure must have three finite, non-zero lattice vectors")
    return [max(1, int(np.ceil(min_supercell_length / length))) for length in lengths]


def _phonopy_atoms(structure):
    """Convert an ABACUS structure to PhonopyAtoms in Angstrom units."""
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=structure.elements,
        cell=np.asarray(structure.cell, dtype=float),
        scaled_positions=np.asarray(structure.coords_direct, dtype=float),
    )


def _initialize_phonopy(structure, supercell: list[int]):
    """Initialize Phonopy with a diagonal supercell matrix."""
    from phonopy import Phonopy

    return Phonopy(_phonopy_atoms(structure), supercell_matrix=np.diag(supercell))


def _read_structure(job: Path):
    """Read the structure referenced by an ABACUS INPUT file."""
    from abacustools.io.abacus import ReadInput
    from abacustools.io.stru import AbacusSTRU

    inputs = ReadInput(job / "INPUT")
    stru_filename = str(inputs.get("stru_file", "STRU"))
    structure = AbacusSTRU.read(job / stru_filename)
    if structure is None:
        raise RuntimeError(f"failed to read structure: {job / stru_filename}")
    return inputs, stru_filename, structure


def _displacement_metadata(phonon) -> list[dict[str, Any]]:
    """Return the generated Phonopy displacement dataset in JSON form."""
    dataset = phonon.dataset
    if not dataset or "first_atoms" not in dataset:
        raise RuntimeError("Phonopy did not generate a displacement dataset")
    return [
        {
            "number": int(item["number"]),
            "displacement": np.asarray(item["displacement"], dtype=float).tolist(),
        }
        for item in dataset["first_atoms"]
    ]


def _existing_displacement_names(job: Path) -> list[str]:
    """Find old displacement directories so changed settings cannot leave stale jobs."""
    return sorted(
        path.name
        for path in job.glob(f"{_TASK_PREFIX}*")
        if path.is_dir() or path.is_symlink()
    )


def prepare(args: argparse.Namespace) -> int:
    """Prepare displaced supercell calculations for a finite-difference phonon spectrum."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _validate_positive_float(args.displacement_stepsize, "displacement_stepsize")
    _validate_positive_float(args.min_supercell_length, "min_supercell_length")

    inputs, stru_filename, structure = _read_structure(job)
    supercell = (
        _validate_supercell(args.supercell)
        if args.supercell is not None
        else _automatic_supercell(structure, args.min_supercell_length)
    )
    phonon = _initialize_phonopy(structure, supercell)
    phonon.generate_displacements(distance=args.displacement_stepsize)
    displaced_structures = phonon.supercells_with_displacements
    if not displaced_structures:
        raise RuntimeError("Phonopy generated no displaced structures")

    names = [f"{_TASK_PREFIX}{index}" for index in range(1, len(displaced_structures) + 1)]
    clear_generated_jobs(
        job,
        sorted(set(names + _existing_displacement_names(job))),
        override=args.override,
    )

    phonon_inputs = deepcopy(inputs)
    phonon_inputs["calculation"] = "scf"
    phonon_inputs["cal_force"] = 1
    try:
        scf_thr = float(phonon_inputs.get("scf_thr", 1e-7))
    except (TypeError, ValueError):
        scf_thr = 1e-7
    if scf_thr > 1e-7:
        phonon_inputs["scf_thr"] = 1e-7
    kpoint_file = kpoint_filename(job, inputs)
    supercell_structure = structure.supercell(supercell)
    if supercell_structure.natoms != len(displaced_structures[0]):
        raise RuntimeError("Phonopy and ABACUS generated supercells have different atom counts")

    print(f"  job: {job}")
    print(f"  supercell: {' '.join(str(value) for value in supercell)}")
    print(f"  displacement step: {args.displacement_stepsize} Angstrom")
    print(f"  generated displacements: {len(displaced_structures)}")
    for name, displaced in zip(names, displaced_structures):
        displaced_structure = deepcopy(supercell_structure)
        displaced_structure.cell = np.asarray(displaced.cell, dtype=float).tolist()
        displaced_structure.coords = np.asarray(displaced.positions, dtype=float).tolist()
        write_abacus_job(
            phonon_inputs,
            displaced_structure,
            job,
            job / name,
            stru_filename=stru_filename,
            kpoint=kpoint_file,
        )
        print(f"  prepared {name}")

    write_manifest(
        job,
        "phonon",
        tasks=names,
        supercell=supercell,
        displacement_stepsize=float(args.displacement_stepsize),
        min_supercell_length=float(args.min_supercell_length),
        displacements=_displacement_metadata(phonon),
    )
    return 0


def _manifest_displacements(phonon, manifest: dict[str, Any]) -> None:
    """Restore the exact displacement dataset recorded during preparation."""
    displacements = manifest.get("displacements")
    if not isinstance(displacements, list) or not displacements:
        raise RuntimeError("phonon workflow manifest has no displacement dataset")
    first_atoms = []
    for item in displacements:
        if not isinstance(item, dict) or "number" not in item or "displacement" not in item:
            raise RuntimeError("invalid displacement entry in phonon workflow manifest")
        try:
            number = int(item["number"])
            displacement = np.asarray(item["displacement"], dtype=float)
        except (TypeError, ValueError) as error:
            raise RuntimeError("invalid displacement entry in phonon workflow manifest") from error
        if number < 0 or number >= len(phonon.supercell) or displacement.shape != (3,):
            raise RuntimeError("invalid displacement entry in phonon workflow manifest")
        if not np.all(np.isfinite(displacement)):
            raise RuntimeError("invalid displacement entry in phonon workflow manifest")
        first_atoms.append({"number": number, "displacement": displacement.tolist()})
    phonon.dataset = {"natom": len(phonon.supercell), "first_atoms": first_atoms}


def _read_forces(job: Path, version: str, expected_natoms: int) -> np.ndarray:
    """Read one converged ABACUS force set in eV/Angstrom."""
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
    expected_shape = (expected_natoms, 3)
    if forces.shape != expected_shape or not np.all(np.isfinite(forces)):
        raise RuntimeError(
            f"invalid force array in the output: {job}; "
            f"expected {expected_shape}, got {forces.shape}"
        )
    return forces


def _validate_mesh(mesh: Any) -> list[int]:
    """Validate the reciprocal-space mesh dimensions."""
    values = list(mesh)
    if len(values) != 3 or any(isinstance(value, bool) for value in values):
        raise ValueError("mesh must contain three positive integers")
    normalized = [int(value) for value in values]
    if any(value != original or value <= 0 for value, original in zip(normalized, values)):
        raise ValueError("mesh must contain three positive integers")
    return normalized


def _custom_band_path(qpath: Any, high_symm_points: Any, npoints: int):
    """Convert a JSON q-path and point dictionary to Phonopy band paths."""
    if not isinstance(qpath, list) or not qpath:
        raise ValueError("qpath must be a non-empty list")
    if not isinstance(high_symm_points, dict) or not high_symm_points:
        raise ValueError("high_symm_points must be a non-empty object")
    if not isinstance(npoints, int) or npoints < 2:
        raise ValueError("npoints must be at least 2")

    if all(isinstance(item, str) for item in qpath):
        paths = [qpath]
    elif all(isinstance(item, list) and item and all(isinstance(point, str) for point in item) for item in qpath):
        paths = qpath
    else:
        raise ValueError("qpath must be a list of labels or a list of label lists")

    points = {}
    for label, coordinates in high_symm_points.items():
        values = np.asarray(coordinates, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"high-symmetry point {label!r} must contain three finite coordinates")
        points[label] = values.tolist()

    band_paths = []
    labels = []
    for path in paths:
        converted = []
        for label in path:
            if label not in points:
                raise ValueError(f"qpath label {label!r} is missing from high_symm_points")
            converted.append(points[label])
            labels.append(r"$\Gamma$" if label.lower() in {"g", "gamma"} else label)
        band_paths.append(converted)

    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    return get_band_qpoints_and_path_connections(band_paths, npoints=npoints), labels


def _jsonable(value: Any) -> Any:
    """Convert NumPy values nested in Phonopy result dictionaries to JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _resolve_output(job: Path, filename: str) -> Path:
    """Resolve a workflow output filename against its job directory."""
    output = Path(filename)
    return output if output.is_absolute() else job / output


def postprocess(args: argparse.Namespace) -> int:
    """Build force constants and calculate the phonon spectrum."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _validate_positive_float(args.temperature, "temperature", allow_zero=True)
    mesh = _validate_mesh(args.mesh)
    if not isinstance(args.npoints, int) or args.npoints < 2:
        raise ValueError("npoints must be at least 2")
    if (args.qpath is None) != (args.high_symm_points is None):
        raise ValueError("qpath and high_symm_points must be provided together")

    _, _, structure = _read_structure(job)
    manifest = read_manifest(job, "phonon", [])
    try:
        manifest_supercell = _validate_supercell(manifest.get("supercell"))
    except ValueError as error:
        raise RuntimeError("phonon workflow manifest has an invalid supercell") from error
    try:
        displacement_stepsize = float(manifest["displacement_stepsize"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "phonon workflow manifest has an invalid displacement_stepsize"
        ) from error
    _validate_positive_float(displacement_stepsize, "displacement_stepsize")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
        raise RuntimeError("phonon workflow manifest has invalid tasks")
    read_manifest(job, "phonon", tasks)

    phonon = _initialize_phonopy(structure, manifest_supercell)
    _manifest_displacements(phonon, manifest)
    expected_natoms = len(phonon.supercell)
    forces = [_read_forces(job / task, args.version, expected_natoms) for task in tasks]
    if len(forces) != len(phonon.dataset["first_atoms"]):
        raise RuntimeError("number of force sets does not match the displacement dataset")

    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_thermal_properties(temperatures=[args.temperature])
    phonon.run_total_dos()

    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
    from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

    commensurate_points = get_commensurate_points(phonon.supercell_matrix)
    frequencies = np.asarray(
        [phonon.get_frequencies(point) for point in commensurate_points],
        dtype=float,
    )
    if args.qpath is None:
        band_paths, labels, connections = get_band_qpoints_by_seekpath(
            _phonopy_atoms(structure),
            npoints=args.npoints,
            is_const_interval=True,
        )
    else:
        (band_paths, connections), labels = _custom_band_path(
            args.qpath,
            args.high_symm_points,
            args.npoints,
        )
    phonon.run_band_structure(band_paths, path_connections=connections, labels=labels)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = _resolve_output(job, args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot = phonon.plot_band_structure_and_dos()
    figure = plot.gcf()
    axes = figure.get_axes()
    if axes:
        axes[0].set_ylabel("Frequency (THz)")
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)

    thermal = _jsonable(phonon.get_thermal_properties_dict())
    max_frequency = float(np.max(frequencies))
    result = {
        "supercell": manifest_supercell,
        "displacement_stepsize": displacement_stepsize,
        "temperature": float(args.temperature),
        "thermal_properties": thermal,
        "entropy": float(thermal["entropy"][0]),
        "free_energy": float(thermal["free_energy"][0]),
        "heat_capacity": float(thermal["heat_capacity"][0]),
        "commensurate_frequencies_thz": frequencies.tolist(),
        "max_frequency_thz": max_frequency,
        "max_frequency_K": max_frequency * _THZ_TO_K,
        "band_structure": _jsonable(phonon.get_band_structure_dict()),
        "total_dos": _jsonable(phonon.get_total_dos_dict()),
        "band_dos_plot": str(plot_path),
    }
    output = _resolve_output(job, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"  job: {job}")
    print(f"  max frequency: {max_frequency:.8f} THz")
    print(f"  entropy: {result['entropy']:.8f}")
    print(f"  free energy: {result['free_energy']:.8f}")
    print(f"  heat capacity: {result['heat_capacity']:.8f}")
    print(f"  plot: {plot_path}")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the phonon preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "phonon",
        "Calculate a phonon spectrum with finite differences.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
