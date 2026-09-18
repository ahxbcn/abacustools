"""The ``abacustools workflow phonon`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.core.constant import THZ_TO_K
from abacustools.data.phonon import (
    automatic_supercell,
    displacement_task,
    jsonable,
    phonopy_atoms,
    phonopy_supercell_structure,
    read_forces,
    validate_mesh,
    validate_positive_float,
    validate_supercell,
)
from abacustools.data.versions import default_version

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    read_job_structure,
    register_stages,
    write_abacus_job,
    write_manifest,
)


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
        default=default_version(),
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












def _initialize_phonopy(structure, supercell: list[int]):
    """Initialize Phonopy with a diagonal supercell matrix.

    ``primitive_matrix="P"`` pins the primitive cell to the reference cell.
    Phonopy 4 resolves the ``"auto"`` default with a symmetry search, while
    phonopy 3 used the identity, so leaving it unset would make the dynamical
    matrix depend on the installed phonopy version.
    """
    from phonopy import Phonopy

    return Phonopy(
        phonopy_atoms(structure),
        supercell_matrix=np.diag(supercell),
        primitive_matrix="P",
    )


def _displacement_metadata(phonon) -> list[dict[str, Any]]:
    """Return the generated Phonopy displacement dataset in JSON form.

    The atom offset is recorded next to every displacement, so that
    postprocessing can locate the displaced atom of a task without relying on
    the order of the task list.
    """
    dataset = phonon.dataset
    if not dataset or "first_atoms" not in dataset:
        raise RuntimeError("Phonopy did not generate a displacement dataset")
    natom = int(dataset.get("natom") or len(phonon.supercell))
    if natom != len(phonon.supercell):
        raise RuntimeError(
            f"Phonopy displacement dataset describes {natom} atoms, "
            f"but the supercell holds {len(phonon.supercell)}"
        )
    return [
        {
            "number": int(item["number"]),
            "displacement": np.asarray(item["displacement"], dtype=float).tolist(),
            "atom_offset": int(item["number"]) * natom,
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
    validate_positive_float(args.displacement_stepsize, "displacement_stepsize")
    validate_positive_float(args.min_supercell_length, "min_supercell_length")

    inputs, stru_filename, structure = read_job_structure(job)
    supercell = (
        validate_supercell(args.supercell)
        if args.supercell is not None
        else automatic_supercell(structure, args.min_supercell_length)
    )
    phonon = _initialize_phonopy(structure, supercell)
    phonon.generate_displacements(distance=args.displacement_stepsize)
    displaced_structures = phonon.supercells_with_displacements
    if not displaced_structures:
        raise RuntimeError("Phonopy generated no displaced structures")

    displacements = _displacement_metadata(phonon)
    # The displaced supercells are indexed by displacement order, while
    # ``number`` indexes the displaced atom.  Both are recorded, so the
    # postprocessing stage can map every force set onto its dataset entry.
    entries = [
        {
            "task": displacement_task(_TASK_PREFIX, index)["task"],
            "index": index,
            "atom": item["number"],
            "displacement": item["displacement"],
        }
        for index, item in enumerate(displacements)
    ]
    names = [entry["task"] for entry in entries]
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
    supercell_structure = phonopy_supercell_structure(structure, phonon.supercell)
    if supercell_structure.natoms != len(displaced_structures[0]):
        raise RuntimeError("Phonopy and ABACUS generated supercells have different atom counts")

    print(f"  job: {job}")
    print(f"  supercell: {' '.join(str(value) for value in supercell)}")
    print(f"  displacement step: {args.displacement_stepsize} Angstrom")
    print(f"  generated displacements: {len(displaced_structures)}")
    for entry, displaced in zip(entries, displaced_structures):
        name = entry["task"]
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
        displacements=entries,
        supercell=supercell,
        displacement_stepsize=float(args.displacement_stepsize),
        min_supercell_length=float(args.min_supercell_length),
        dataset=displacements,
    )
    return 0


def _manifest_displacements(phonon, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and return the displacement dataset recorded during preparation.

    Args:
        phonon: Phonopy object used to bound the displaced-atom index.
        manifest: Preparation-time manifest of the workflow.

    Returns:
        The validated dataset entries.

    Raises:
        RuntimeError: When the dataset is missing, malformed, or inconsistent
            with the supercell.
    """
    displacements = manifest.get("dataset")
    if not isinstance(displacements, list) or not displacements:
        raise RuntimeError("phonon workflow manifest has no displacement dataset")
    validated = []
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
        validated.append(
            {"number": number, "displacement": displacement.tolist()}
        )
    return validated


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




def _resolve_output(job: Path, filename: str) -> Path:
    """Resolve a workflow output filename against its job directory."""
    output = Path(filename)
    return output if output.is_absolute() else job / output


def postprocess(args: argparse.Namespace) -> int:
    """Build force constants and calculate the phonon spectrum."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    validate_positive_float(args.temperature, "temperature", allow_zero=True)
    mesh = validate_mesh(args.mesh)
    if not isinstance(args.npoints, int) or args.npoints < 2:
        raise ValueError("npoints must be at least 2")
    if (args.qpath is None) != (args.high_symm_points is None):
        raise ValueError("qpath and high_symm_points must be provided together")

    _, _, structure = read_job_structure(job)
    manifest = read_manifest(job, "phonon", [])
    try:
        manifest_supercell = validate_supercell(manifest.get("supercell"))
    except ValueError as error:
        raise RuntimeError("phonon workflow manifest has an invalid supercell") from error
    try:
        displacement_stepsize = float(manifest["displacement_stepsize"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "phonon workflow manifest has an invalid displacement_stepsize"
        ) from error
    validate_positive_float(displacement_stepsize, "displacement_stepsize")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
        raise RuntimeError("phonon workflow manifest has invalid tasks")
    read_manifest(job, "phonon", tasks)

    phonon = _initialize_phonopy(structure, manifest_supercell)
    displacements = _manifest_displacements(phonon, manifest)
    entries = [
        {
            "task": displacement_task(_TASK_PREFIX, index)["task"],
            "index": index,
            "atom": item["number"],
            "displacement": item["displacement"],
        }
        for index, item in enumerate(displacements)
    ]
    if [entry["task"] for entry in entries] != tasks:
        raise RuntimeError(
            "phonon workflow manifest lists displacement tasks that do not match "
            "its task list"
        )

    phonon.dataset = {
        "natom": len(phonon.supercell),
        "first_atoms": [
            {"number": item["number"], "displacement": item["displacement"]}
            for item in displacements
        ],
    }
    expected_natoms = len(phonon.supercell)
    if len(entries) != len(phonon.dataset["first_atoms"]):
        raise RuntimeError(
            "the number of displaced calculations does not match the displacement "
            "dataset of the phonon workflow manifest"
        )
    # Map every force set onto its displacement through the recorded index, so
    # the fit never depends on the order the tasks happen to be listed in.
    forces = [
        read_forces(job / str(entry["task"]), args.version, expected_natoms)
        for entry in entries
    ]

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
            phonopy_atoms(structure),
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

    thermal = jsonable(phonon.get_thermal_properties_dict())
    max_frequency = float(np.max(frequencies))
    result = {
        "supercell": manifest_supercell,
        "displacement_stepsize": displacement_stepsize,
        "displacements": entries,
        "temperature": float(args.temperature),
        "thermal_properties": thermal,
        "entropy": float(thermal["entropy"][0]),
        "free_energy": float(thermal["free_energy"][0]),
        "heat_capacity": float(thermal["heat_capacity"][0]),
        "commensurate_frequencies_thz": frequencies.tolist(),
        "max_frequency_thz": max_frequency,
        "max_frequency_K": max_frequency * THZ_TO_K,
        "band_structure": jsonable(phonon.get_band_structure_dict()),
        "total_dos": jsonable(phonon.get_total_dos_dict()),
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
