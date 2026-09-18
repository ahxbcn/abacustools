"""The ``abacustools workflow phonon`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from abacustools.core.constant import (
    JOULE_PER_MOL_KELVIN_TO_EV_PER_KELVIN,
    KILOJOULE_PER_MOL_TO_EV,
    THZ_TO_K,
)
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
    parser.add_argument(
        "--pdos",
        action="store_true",
        help="Also report the atom- and Cartesian-projected DOS and plot it.",
    )
    parser.add_argument(
        "--pdos-plot",
        default="phonon_projected_dos.png",
        help="Projected DOS plot filename. Relative paths are resolved below JOB.",
    )
    parser.add_argument(
        "--debye",
        action="store_true",
        help="Also report the Debye frequency fitted to the total DOS.",
    )
    parser.add_argument(
        "--irreps",
        action="store_true",
        help="Also report the space-group irreducible representations of the "
        "phonon modes at the Gamma point.",
    )
    parser.add_argument(
        "--irreps-plot",
        default="phonon_gamma_irreps.png",
        help="Gamma point mode plot filename. Relative paths are resolved below JOB.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-5,
        help="Symmetry tolerance in Angstrom for the Gamma point mode "
        "representation, default: 1e-05.",
    )
    parser.add_argument(
        "--bec-results",
        type=Path,
        default=None,
        help="bec_results.json written by 'workflow bec', used for the Born "
        "effective charges of the non-analytical correction. The dielectric "
        "tensor still has to be given with --dielectric.",
    )
    parser.add_argument(
        "--dielectric",
        type=_json_argument,
        default=None,
        metavar="JSON",
        help="Dielectric tensor for the non-analytical correction, either a "
        'scalar or a 3x3 matrix as JSON, for example "[2.34, 0, 0, 0, 2.34, 0, 0, 0, 2.34]" '
        'or "2.34". Required together with --born unless the tensor is read '
        "with --dielectric-results.",
    )
    parser.add_argument(
        "--dielectric-results",
        type=Path,
        default=None,
        help="dielectric_results.json written by 'workflow dielectric', which "
        "supplies the dielectric tensor of the non-analytical correction and "
        "avoids transcribing it by hand.",
    )
    parser.add_argument(
        "--born",
        type=_json_argument,
        default=None,
        metavar="JSON",
        help="Born effective charges for the non-analytical correction, as a "
        "list holding one 3x3 tensor per atom of the reference cell, in the "
        "atom order of the structure, collinear with its lattice vectors. "
        'For example "[[[1.1,0,0],[0,1.1,0],[0,0,1.1]],'
        '[[-1.1,0,0],[0,-1.1,0],[0,0,-1.1]]]". Required together with --dielectric.',
    )
    parser.add_argument(
        "--nac-direction",
        type=_json_argument,
        default=None,
        metavar="JSON",
        help="Direction, in fractional coordinates of the reciprocal basis, "
        "along which the q to zero limit of the non-analytical correction is "
        "taken for the Gamma point modes and the thermal properties. Defaults "
        "to the first lattice vector.",
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


def _thermal_properties_dict(phonon) -> Dict[str, Any]:
    """Return the thermal properties as a JSON-compatible dictionary.

    Phonopy 3 exposes the result objects directly and deprecates the
    ``get_*_dict`` accessors, so those are preferred when present.

    Phonopy reports the free energy in kJ/mol and the entropy and the heat
    capacity in J/(K mol), that is per mole of unit cells.  Those are converted
    to eV and eV/K per cell, the unit the vibration workflow reports its
    thermochemistry in, so that the two can be compared directly.
    """
    properties = getattr(phonon, "thermal_properties", None)
    if properties is None:
        raw = phonon.get_thermal_properties_dict()
    else:
        raw = {
            "temperatures": properties.temperatures,
            "free_energy": properties.free_energy,
            "entropy": properties.entropy,
            "heat_capacity": properties.heat_capacity,
        }
    return {
        "temperatures": jsonable(raw["temperatures"]),
        "free_energy": jsonable(
            np.asarray(raw["free_energy"], dtype=float) * KILOJOULE_PER_MOL_TO_EV
        ),
        "entropy": jsonable(
            np.asarray(raw["entropy"], dtype=float)
            * JOULE_PER_MOL_KELVIN_TO_EV_PER_KELVIN
        ),
        "heat_capacity": jsonable(
            np.asarray(raw["heat_capacity"], dtype=float)
            * JOULE_PER_MOL_KELVIN_TO_EV_PER_KELVIN
        ),
        "units": {
            "temperature": "K",
            "free_energy": "eV per cell",
            "entropy": "eV/K per cell",
            "heat_capacity": "eV/K per cell",
        },
    }


def _total_dos_dict(phonon) -> Dict[str, Any]:
    """Return the total DOS as a JSON-compatible dictionary."""
    dos = getattr(phonon, "total_dos", None)
    if dos is None:
        return phonon.get_total_dos_dict()
    return {
        "frequency_points": dos.frequency_points,
        "total_dos": dos.dos,
    }


def _band_structure_dict(phonon) -> Dict[str, Any]:
    """Return the band structure as a JSON-compatible dictionary."""
    structure = getattr(phonon, "band_structure", None)
    if structure is None:
        return phonon.get_band_structure_dict()
    return {
        "frequencies": structure.frequencies,
        "distances": structure.distances,
        "qpoints": structure.qpoints,
    }


def _projected_dos_rows(phonon, symbols, *, xyz_projection: bool) -> List[Dict[str, Any]]:
    """Return the projected DOS as one labelled record per projection.

    Args:
        phonon: Phonopy object whose projected DOS has been run.
        symbols: Element symbol of every atom of the primitive cell.
        xyz_projection: Whether the projection separates the Cartesian axes.

    Returns:
        One record per projection, holding its label, the element and atom it
        belongs to, the Cartesian direction when resolved, and the projected
        DOS values.
    """
    values = np.asarray(phonon.projected_dos.projected_dos, dtype=float)
    rows: List[Dict[str, Any]] = []
    for index in range(values.shape[0]):
        if xyz_projection:
            atom, axis = divmod(index, 3)
            direction = ("x", "y", "z")[axis]
            label = f"{symbols[atom]}{atom + 1}:{direction}"
        else:
            atom, direction = index, None
            label = f"{symbols[atom]}{atom + 1}"
        rows.append(
            {
                "label": label,
                "element": symbols[atom],
                "atom": atom + 1,
                "direction": direction,
                "values": values[index].tolist(),
            }
        )
    return rows


#: Coulomb constant ``e^2 / (4 pi eps0)`` in eV Angstrom, the unit phonopy
#: expects for the non-analytical term of a force-constant calculation.
_COULOMB_EV_ANGSTROM = 14.399645

#: Frequencies below this, in THz, count as the translations of a free cell.
_ACOUSTIC_TOLERANCE = 1e-4

#: Frequency shift, in THz, above which the non-analytical correction is taken
#: to have moved a mode and the mode is therefore longitudinal.
_LONGITUDINAL_TOLERANCE = 1e-6


def _dielectric_tensor(value: Any) -> np.ndarray:
    """Return a 3x3 dielectric tensor from a CLI value.

    Args:
        value: Scalar, three diagonal values, or a row-major 3x3 matrix.

    Returns:
        The tensor.

    Raises:
        ValueError: When the value cannot be read as a tensor.
    """
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.shape == (1,):
        tensor = np.eye(3) * float(array.reshape(()))
    elif array.shape == (3,):
        tensor = np.diag(array)
    elif array.shape == (9,):
        # A row-major matrix typed as one flat list.
        tensor = array.reshape(3, 3)
    elif array.shape == (3, 3):
        tensor = array
    else:
        raise ValueError(
            "the dielectric tensor must be a scalar, three diagonal values, a flat "
            f"nine value matrix or a 3x3 matrix, got shape {array.shape}"
        )
    if not np.all(np.isfinite(tensor)):
        raise ValueError("the dielectric tensor holds non-finite values")
    return tensor


def _dielectric_tensor_from_results(path: Path) -> Any:
    """Return the dielectric tensor written by the dielectric workflow.

    Args:
        path: ``dielectric_results.json`` of the dielectric workflow.

    Returns:
        The tensor, in the layout :func:`_dielectric_tensor` accepts.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When it holds no dielectric tensor.
    """
    if not path.is_file():
        raise FileNotFoundError(f"could not find the dielectric results: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid dielectric results file: {path}") from error
    tensor = payload.get("tensor") if isinstance(payload, dict) else None
    if not isinstance(tensor, list) or len(tensor) != 3:
        raise ValueError(f"the dielectric results hold no 3x3 tensor: {path}")
    return tensor


def _born_charges_from_bec_results(path: Path, structure) -> np.ndarray:
    """Return the Born effective charges written by the BEC workflow.

    The BEC workflow stores a tensor per displaced atom whose rows are the
    displacement directions and whose columns are the Cartesian polarization
    directions, which is the layout the non-analytical correction expects, so
    the tensors are passed on as they are.

    Args:
        path: ``bec_results.json`` of the BEC workflow.
        structure: Reference cell the charges are needed for.

    Returns:
        The ``(natoms, 3, 3)`` array.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When the file does not describe every atom of the cell, or
            when one of its tensors is incomplete.
    """
    if not path.is_file():
        raise FileNotFoundError(f"could not find the BEC results: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid BEC results file: {path}") from error

    atoms = payload.get("atoms") if isinstance(payload, dict) else None
    if not isinstance(atoms, list) or not atoms:
        raise ValueError(f"the BEC results hold no atoms: {path}")

    charges = np.full((structure.natoms, 3, 3), np.nan, dtype=float)
    for atom in atoms:
        if not isinstance(atom, dict):
            raise ValueError(f"invalid atom entry in the BEC results: {path}")
        try:
            index = int(atom["index"]) - 1
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid atom index in the BEC results: {path}") from error
        if index < 0 or index >= structure.natoms:
            raise ValueError(
                f"the BEC results describe atom {index + 1}, which the reference "
                f"cell of {structure.natoms} atoms does not hold"
            )
        tensor = atom.get("bec_tensor")
        if not isinstance(tensor, list) or len(tensor) != 3:
            raise ValueError(f"atom {index + 1} of the BEC results has no tensor")
        for row, values in enumerate(tensor):
            if values is None:
                raise ValueError(
                    f"atom {index + 1} of the BEC results is missing the displacement "
                    f"along {'xyz'[row]}; compute every direction of every atom"
                )
            charges[index, row] = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(charges)):
        raise ValueError(
            f"the BEC results do not describe every atom of the reference cell: {path}"
        )
    return charges


def _born_charges(value: Any, natoms: int) -> np.ndarray:
    """Return the Born effective charges from a CLI value.

    Args:
        value: One 3x3 tensor per atom of the reference cell.
        natoms: Number of atoms of the reference cell.

    Returns:
        The ``(natoms, 3, 3)`` array.

    Raises:
        ValueError: When the shape does not match the reference cell.
    """
    array = np.asarray(value, dtype=float)
    if array.shape == (3, 3) and natoms == 1:
        array = array[None, :, :]
    if array.shape != (natoms, 3, 3):
        raise ValueError(
            f"the Born effective charges must hold one 3x3 tensor per atom of the "
            f"reference cell, expected {(natoms, 3, 3)}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("the Born effective charges hold non-finite values")
    return array


def _nac_parameters(
    args: argparse.Namespace,
    structure,
) -> Optional[Dict[str, Any]]:
    """Assemble the non-analytical correction parameters of the workflow.

    Args:
        args: Postprocessing arguments.
        structure: Reference cell the Born charges are given for.

    Returns:
        The payload for ``phonopy.nac_params``, or ``None`` when the correction
        was not requested.

    Raises:
        ValueError: When only one of the two required inputs is given, or when
            an input cannot be read.
    """
    dielectric = getattr(args, "dielectric", None)
    dielectric_results = getattr(args, "dielectric_results", None)
    born = getattr(args, "born", None)
    bec_results = getattr(args, "bec_results", None)
    if (
        dielectric is None
        and dielectric_results is None
        and born is None
        and bec_results is None
    ):
        return None
    if dielectric is not None and dielectric_results is not None:
        raise ValueError(
            "give the dielectric tensor either with --dielectric or with "
            "--dielectric-results, not both"
        )
    if born is None and bec_results is None:
        raise ValueError(
            "the non-analytical correction needs the Born effective charges, "
            "given with --born or read with --bec-results"
        )
    if dielectric is None and dielectric_results is None:
        raise ValueError(
            "the non-analytical correction needs the dielectric tensor, given "
            "with --dielectric or read with --dielectric-results, together with "
            "--born or --bec-results"
        )
    if born is None and bec_results is not None:
        born = _born_charges_from_bec_results(Path(bec_results), structure)
    if dielectric is None:
        dielectric = _dielectric_tensor_from_results(Path(dielectric_results))
    return {
        "born": _born_charges(born, structure.natoms),
        "dielectric": _dielectric_tensor(dielectric),
        "factor": _COULOMB_EV_ANGSTROM,
    }


def _nac_direction(args: argparse.Namespace, structure) -> np.ndarray:
    """Return the direction of the q to zero limit, in Cartesian coordinates.

    Args:
        args: Postprocessing arguments.
        structure: Reference cell whose basis the direction is given in.

    Returns:
        A Cartesian direction vector.

    Raises:
        ValueError: When the direction cannot be read or is zero.
    """
    value = getattr(args, "nac_direction", None)
    cell = np.asarray(structure.cell, dtype=float)
    if value is None:
        # The first lattice vector is a natural longitudinal direction and
        # differs between the sublattices of an ionic crystal, so the limit is
        # not degenerate along it.
        return np.array(cell[0], dtype=float)
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(
            "the non-analytical direction must hold three finite coordinates"
        )
    if np.allclose(array, 0.0):
        raise ValueError("the non-analytical direction must not be zero")
    return np.asarray(array @ cell, dtype=float)


def _plot_projected_dos(
    path: Path,
    frequency_points: np.ndarray,
    rows: List[Dict[str, Any]],
    *,
    total: Optional[np.ndarray] = None,
) -> None:
    """Plot the projected DOS of every atom on a shared frequency axis.

    Args:
        path: File to write the figure to.
        frequency_points: Frequency grid of the projection, in THz.
        rows: Projections returned by :func:`_projected_dos_rows`.
        total: Total DOS to draw on every panel for reference.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(rows)
    figure, axes = plt.subplots(
        count, 1, sharex=True, figsize=(6.0, max(2.0, 1.1 * count)), dpi=300
    )
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, rows):
        if total is not None:
            axis.plot(
                frequency_points, total, color="0.75", linewidth=0.8, label="total"
            )
        axis.plot(frequency_points, row["values"], color="#9467bd", linewidth=1.0)
        axis.set_ylabel(row["label"], rotation=0, ha="right", va="center", fontsize=8)
        axis.set_yticks([])
        axis.set_ylim(bottom=0.0)
    axes[-1].set_xlabel("Frequency (THz)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _plot_gamma_modes(path: Path, modes: List[Dict[str, Any]]) -> None:
    """Plot the Gamma point frequencies labelled by their representation.

    Args:
        path: File to write the figure to.
        modes: Records holding a frequency and an optional representation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequencies = [mode["frequency_thz"] for mode in modes]
    figure, axis = plt.subplots(figsize=(5.0, 0.6 + 0.28 * len(modes)), dpi=300)
    positions = np.arange(len(modes))
    axis.barh(positions, frequencies, color="#1f77b4", height=0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [
            f"mode {mode['band']}: {mode['irrep']}"
            if mode.get("irrep")
            else f"mode {mode['band']}"
            for mode in modes
        ],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Frequency (THz)")
    for position, frequency in zip(positions, frequencies):
        axis.text(frequency, position, f" {frequency:.3f}", va="center", fontsize=7)
    axis.set_xlim(left=0.0)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _gamma_modes(
    phonon,
    *,
    want_irreps: bool,
    symprec: float,
    nac_direction: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Return the Gamma point modes with their degeneracy and representation.

    Args:
        phonon: Phonopy object whose mesh has been run.
        want_irreps: Whether to resolve the space-group representations.
        symprec: Symmetry tolerance in Angstrom.
        nac_direction: Cartesian direction of the q to zero limit of the
            non-analytical correction; omitted when the correction is off.

    Returns:
        One record per mode, holding the one-based band index, the frequency in
        THz, the size of its degenerate set, and the representation when it
        could be resolved.

    Raises:
        RuntimeError: When ``want_irreps`` is set and the symmetry of the
            structure cannot be found.
    """
    from phonopy.phonon.degeneracy import degenerate_sets

    # With the correction on, the q to zero limit has to be taken along a
    # direction, otherwise the longitudinal optical mode keeps the transverse
    # frequency and the two remain degenerate.
    points = phonon.run_qpoints([[0.0, 0.0, 0.0]], nac_q_direction=nac_direction)
    frequencies = np.asarray(points.frequencies[0], dtype=float)
    if nac_direction is not None:
        # A mode is longitudinal when the correction moves it.  Comparing with
        # the uncorrected limit is what separates a longitudinal mode from an
        # optical mode that is simply non-degenerate in a low symmetry crystal.
        # A zero direction only means "no preferred direction", which still
        # applies the correction with the direction of the q point itself, so
        # the correction has to be switched off on a separate object that is
        # thrown away: clearing it on this one would invalidate the mesh, the
        # DOS and the thermal properties that have already been computed.
        from phonopy import Phonopy

        plain_object = Phonopy(
            phonon.unitcell,
            supercell_matrix=phonon.supercell_matrix,
            primitive_matrix=phonon.primitive_matrix,
        )
        plain_object.force_constants = phonon.force_constants
        plain = np.asarray(
            plain_object.run_qpoints([[0.0, 0.0, 0.0]]).frequencies[0], dtype=float
        )
    else:
        plain = None
    modes = [
        {"band": index + 1, "frequency_thz": float(frequency), "degeneracy": 1}
        for index, frequency in enumerate(frequencies)
    ]
    if plain is not None:
        for index, mode in enumerate(modes):
            if abs(mode["frequency_thz"]) < _ACOUSTIC_TOLERANCE:
                mode["character"] = "acoustic"
            elif abs(frequencies[index] - plain[index]) > _LONGITUDINAL_TOLERANCE:
                mode["character"] = "LO"
            else:
                mode["character"] = "TO"
    for group in degenerate_sets(frequencies):
        if len(group) > 1:
            for index in group:
                modes[index]["degeneracy"] = len(group)

    if not want_irreps:
        return modes

    try:
        representations = phonon.run_irreps(
            [0.0, 0.0, 0.0], nac_q_direction=nac_direction
        )
    except Exception as error:  # pragma: no cover - depends on the structure
        raise RuntimeError(
            f"could not find the symmetry needed for the Gamma point irreps: {error}"
        ) from error

    point_group = getattr(representations, "_pointgroup_symbol_at_q", "") or ""
    # Phonopy names the Gamma point representations with their Mulliken
    # symbols. The label is left unset for point groups whose character table
    # it cannot index unequivocally, so an unnamed representation is reported
    # by its dimension alongside the point group instead.
    labels = getattr(representations, "ir_labels", None)
    if labels is None:
        labels = getattr(representations, "_ir_labels", None)
    for position, group in enumerate(representations.band_indices):
        if labels is not None and position < len(labels) and labels[position] is not None:
            symbol = str(labels[position])
        else:
            dimensions = np.shape(representations.irreps[position])
            dimension = int(dimensions[1]) if len(dimensions) > 1 else 1
            symbol = f"{dimension}D" + (f" ({point_group})" if point_group else "")
        for index in group:
            modes[index]["irrep"] = symbol
    return modes


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
    symprec = float(getattr(args, "symprec", 1e-5))
    validate_positive_float(symprec, "symprec")
    pdos = bool(getattr(args, "pdos", False))
    debye = bool(getattr(args, "debye", False))
    irreps = bool(getattr(args, "irreps", False))
    pdos_plot = args.pdos_plot if hasattr(args, "pdos_plot") else None
    irreps_plot = args.irreps_plot if hasattr(args, "irreps_plot") else None

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
    nac = _nac_parameters(args, structure)
    nac_direction = None
    if nac is not None:
        phonon.nac_params = nac
        nac_direction = _nac_direction(args, structure)
        result_nac = {
            "dielectric": np.asarray(nac["dielectric"], dtype=float).tolist(),
            "born": np.asarray(nac["born"], dtype=float).tolist(),
            "direction_cartesian": nac_direction.tolist(),
        }
    else:
        result_nac = None
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    # The Gamma point modes are resolved before the DOS, because the
    # longitudinal character is found by comparing against the limit with the
    # correction switched off, which resets the mesh and the DOS.
    gamma_modes = _gamma_modes(
        phonon,
        want_irreps=irreps,
        symprec=symprec,
        nac_direction=nac_direction,
    )
    # The mesh applies the correction with the direction of each q point of its
    # own, so the mesh, the DOS and the thermal properties need no direction.
    phonon.run_thermal_properties(temperatures=[args.temperature])
    phonon.run_total_dos()

    projected = None
    if pdos:
        phonon.run_projected_dos(use_tetrahedron_method=True, xyz_projection=True)
        projected = _projected_dos_rows(
            phonon, list(structure.elements), xyz_projection=True
        )

    debye_frequency = None
    if debye:
        phonon.total_dos.run_debye_frequency()
        fitted = float(phonon.total_dos.debye_frequency)
        if np.isfinite(fitted):
            debye_frequency = fitted
        else:
            # Phonopy returns an infinite Debye frequency when the DOS does
            # not support the fit, and then fails to draw the marker it adds
            # to the DOS plot, so drop the value from the object as well.
            phonon.total_dos._freq_Debye = None
            print(
                "  warning: the Debye frequency did not converge for this DOS "
                "and is left out of the report"
            )

    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
    from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

    commensurate_points = get_commensurate_points(phonon.supercell_matrix)
    frequencies = np.asarray(
        phonon.run_qpoints(commensurate_points).frequencies, dtype=float
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

    thermal = jsonable(_thermal_properties_dict(phonon))
    # The maximum has to be taken over the dispersion rather than over the
    # commensurate points of the supercell: a polar material reaches its
    # highest frequency in the longitudinal optical mode at Gamma, which the
    # supercell does not carry.
    band_structure = jsonable(_band_structure_dict(phonon))
    max_frequency = float(
        np.max(
            np.concatenate(
                [np.asarray(segment, dtype=float) for segment in band_structure["frequencies"]]
            )
        )
    )
    result = {
        "supercell": manifest_supercell,
        "mesh": mesh,
        "displacement_stepsize": displacement_stepsize,
        "displacements": entries,
        "non_analytical_correction": result_nac,
        "temperature": float(args.temperature),
        "thermal_properties": thermal,
        "entropy": float(thermal["entropy"][0]),
        "free_energy": float(thermal["free_energy"][0]),
        "heat_capacity": float(thermal["heat_capacity"][0]),
        "commensurate_frequencies_thz": frequencies.tolist(),
        "gamma_modes": gamma_modes,
        "max_frequency_thz": max_frequency,
        "max_frequency_K": max_frequency * THZ_TO_K,
        "band_structure": band_structure,
        "total_dos": jsonable(_total_dos_dict(phonon)),
        "band_dos_plot": str(plot_path),
        "units": {
            "temperature": "K",
            "entropy": "eV/K per cell",
            "free_energy": "eV per cell",
            "heat_capacity": "eV/K per cell",
        },
    }
    if debye_frequency is not None:
        result["debye"] = {
            "frequency_thz": float(debye_frequency),
            "temperature_K": float(debye_frequency) * THZ_TO_K,
        }
    if projected is not None:
        frequency_points = np.asarray(
            phonon.projected_dos.frequency_points, dtype=float
        )
        result["projected_dos"] = {
            "xyz_projection": True,
            "frequency_points": frequency_points.tolist(),
            "projections": projected,
        }
        projected_plot = _resolve_output(job, pdos_plot)
        total_dos = np.asarray(phonon.total_dos.dos, dtype=float)
        _plot_projected_dos(
            projected_plot,
            frequency_points,
            projected,
            total=total_dos,
        )
        result["projected_dos"]["plot"] = str(projected_plot)
    if irreps:
        irreps_plot_path = _resolve_output(job, irreps_plot)
        _plot_gamma_modes(irreps_plot_path, gamma_modes)
        result["gamma_irreps_plot"] = str(irreps_plot_path)
    output = _resolve_output(job, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"  job: {job}")
    # The thermal properties are reported in eV and eV/K per cell.
    print(
        f"  max frequency: {max_frequency:.8f} THz "
        f"({result['max_frequency_K']:.4f} K)"
    )
    print(f"  entropy: {result['entropy']:.8f} eV/K")
    print(f"  free energy: {result['free_energy']:.8f} eV")
    print(f"  heat capacity: {result['heat_capacity']:.8f} eV/K")
    if debye_frequency is not None:
        print(
            f"  Debye frequency: {result['debye']['frequency_thz']:.8f} THz "
            f"({result['debye']['temperature_K']:.4f} K)"
        )
    if irreps:
        labels = ", ".join(
            f"{mode['band']}:{mode.get('irrep', '?')}" for mode in gamma_modes
        )
        print(f"  Gamma modes: {labels}")
    if nac is not None:
        for mode in gamma_modes:
            if mode.get("character") == "LO":
                print(
                    f"  Gamma longitudinal mode: {mode['frequency_thz']:.8f} THz"
                )
    if projected is not None:
        print(f"  projected DOS: {result['projected_dos']['plot']}")
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
