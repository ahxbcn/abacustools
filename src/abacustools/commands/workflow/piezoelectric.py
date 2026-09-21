"""The ``abacustools workflow piezoelectric`` workflow."""

from __future__ import annotations

import argparse
import json
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from abacustools.core.constant import ELEMENTARY_CHARGE
from abacustools.data.elastic import point_group_operations
from abacustools.data.piezoelectric import (
    fit_independent_piezoelectric,
    independent_component_count,
    independent_components,
    independent_strain_modes,
    symmetrize_piezoelectric_tensor,
    symmetrization_residual,
)
from abacustools.data.versions import default_version
from abacustools.io.abacus import WriteInput, WriteKpt
from abacustools.io.stru import AbacusSTRU

from abacustools.data.polarization import (
    kpoint_mesh,
    polarization_cartesian,
    read_task_polarization,
    task_metrics,
)
from .common import (
    clear_generated_jobs,
    read_job_structure,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)


_VOIGT_MODES = (
    ("xx", 0, 0),
    ("yy", 1, 1),
    ("zz", 2, 2),
    ("yz", 1, 2),
    ("xz", 0, 2),
    ("xy", 0, 1),
)
_DISP_TYPES = ("f", "b", "c")
_TASK_PREFIX = "piezoelectric_"
_ELECTRON_ANGSTROM_SQUARED_TO_CM2 = ELEMENTARY_CHARGE * 1.0e20

#: Symmetry tolerance of the reference cell, in Angstrom.
DEFAULT_SYMPREC = 1.0e-2


def _selected_modes(modes=None) -> list[tuple[str, int, int]]:
    """Return the Voigt strain modes a preparation or a fit works with."""
    if modes is None:
        return list(_VOIGT_MODES)
    wanted = {int(mode) for mode in modes}
    return [
        entry for position, entry in enumerate(_VOIGT_MODES) if position in wanted
    ]


def _selectable_modes(modes=None) -> list[int]:
    """Return the Voigt indices of the selected strain modes."""
    wanted = (
        set(range(len(_VOIGT_MODES)))
        if modes is None
        else {int(mode) for mode in modes}
    )
    return [position for position in range(len(_VOIGT_MODES)) if position in wanted]


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for piezoelectric preparation."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory used to prepare piezoelectric calculations.",
    )
    parser.add_argument(
        "--strain", type=float, default=0.01,
        help="Finite strain magnitude, default: 0.01.",
    )
    parser.add_argument(
        "--type", dest="disp_type", choices=_DISP_TYPES, default="f",
        help="Difference type: f=forward, b=backward, c=central; default: f.",
    )
    continuity = parser.add_mutually_exclusive_group()
    continuity.add_argument(
        "--use-k-continuity", dest="use_k_continuity", action="store_true",
        help="Enable ABACUS k-point continuity. ABACUS only accepts it for "
        "plane wave calculations and refuses it for the non self consistent "
        "Berry phase steps this workflow runs, so enabling it makes every "
        "generated calculation stop; it is available for future versions that "
        "lift the restriction.",
    )
    continuity.add_argument(
        "--no-k-continuity", dest="use_k_continuity", action="store_false",
        help="Do not enable ABACUS k-point continuity, the default.",
    )
    parser.set_defaults(use_k_continuity=False)
    parser.add_argument(
        "--relax", action="store_true",
        help="Relax ionic positions under each strain before the Berry calculation.",
    )
    parser.add_argument(
        "--strains",
        choices=("full", "independent"),
        default="full",
        help=(
            "Strain modes to prepare. 'full' applies all six Voigt strains, "
            "'independent' applies only the modes that determine the "
            "independent piezoelectric components of the point group, "
            "default: full."
        ),
    )
    parser.add_argument(
        "--abacus-command", "--abacus_command", dest="abacus_command", default="abacus",
        help="ABACUS command written to generated runners, default: abacus.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace existing generated piezoelectric directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for piezoelectric postprocessing."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory containing the prepared piezoelectric calculations.",
    )
    parser.add_argument(
        "-v", "--version", default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-o", "--output", default="piezoelectric_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=DEFAULT_SYMPREC,
        help=f"Symmetry tolerance of the reference cell, default: {DEFAULT_SYMPREC}.",
    )
    parser.add_argument(
        "--symmetrize",
        dest="symmetrize",
        action="store_true",
        default=True,
        help="Symmetrize the tensor with the crystal symmetry, the default.",
    )
    parser.add_argument(
        "--no-symmetrize",
        dest="symmetrize",
        action="store_false",
        help="Keep the fitted components as they are.",
    )
    parser.add_argument(
        "--fit",
        choices=("full", "independent"),
        default="full",
        help=(
            "Fit the six measured columns and symmetrize afterwards ('full'), "
            "or fit only the independent components ('independent'). A job "
            "prepared with --strains independent always uses the independent "
            "fit, default: full."
        ),
    )


def _validate_strain(value: float) -> float:
    """Validate a strain magnitude that can produce a non-singular cell."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError("strain must be a positive finite number")
    if value >= 1.0:
        raise ValueError("strain must be smaller than 1")
    return float(value)


def _strain_matrix(index: int, other_index: int, magnitude: float) -> np.ndarray:
    """Build one normal or shear strain matrix in the Voigt convention.

    A normal mode is the strain tensor component itself, ``eps_xx`` for the
    ``xx`` mode.  A shear mode gets half of the requested magnitude in each
    off-diagonal element, so that the *engineering* shear of the IEEE
    convention, ``S_4 = 2 eps_yz`` and so on, equals the requested magnitude.
    Dividing the polarization change by that magnitude then gives the
    piezoelectric tensor the literature and DFPT codes report, and a request
    for a one percent shear means a one percent shear in the same sense the
    elastic workflow uses.
    """
    matrix = np.zeros((3, 3), dtype=float)
    if index == other_index:
        matrix[index, other_index] = magnitude
    else:
        matrix[index, other_index] = 0.5 * magnitude
        matrix[other_index, index] = 0.5 * magnitude
    return matrix


def _task_names(disp_type: str, modes=None) -> list[str]:
    """Return the deterministic task order of the selected strain components."""
    names: list[str] = []
    if disp_type != "c":
        names.append("piezoelectric_org")
    for label, _, _ in _selected_modes(modes):
        if disp_type in {"f", "c"}:
            names.append(f"piezoelectric_{label}")
        if disp_type in {"b", "c"}:
            names.append(f"piezoelectric_{label}_back")
    return names


def _write_runner(
    path: Path,
    abacus_command: str,
    *,
    relax: bool,
    stru_filename: str,
) -> None:
    """Write the local SCF/NSCF runner used by one generated task."""
    command_parts = shlex.split(abacus_command)
    if not command_parts:
        raise ValueError("abacus_command must not be empty")
    command = shlex.join(command_parts)
    structure_target = shlex.quote(stru_filename)
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    if relax:
        lines.extend(
            [
                "cp INPUT.relax INPUT",
                "cp KPT.scf KPT",
                f"{command} | tee relax.log",
                f"cp OUT.ABACUS/STRU_ION_D {structure_target}",
                "",
            ]
        )
    lines.extend(
        [
            "cp INPUT.scf INPUT",
            "cp KPT.scf KPT",
            f"{command} | tee scf.log",
            "",
            "for direction in 1 2 3; do",
            "    cp INPUT.nscf${direction} INPUT",
            "    cp KPT.nscf${direction} KPT",
            f"    {command} | tee nscf${{direction}}.log",
            "    mv -f OUT.ABACUS/running_nscf.log OUT.ABACUS/running_nscf${direction}.log",
            "    rm -f OUT.ABACUS/*.restart",
            "done",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _write_piezo_inputs(
    source_job: Path,
    structure,
    inputs: dict[str, Any],
    stru_filename: str,
    destination: Path,
    kpoint: list[float],
    kpoint_model: str,
    *,
    use_k_continuity: bool,
    relax: bool,
    abacus_command: str,
) -> None:
    """Write SCF and three Berry-phase NSCF inputs for one strain state."""
    scf_inputs = deepcopy(inputs)
    scf_inputs.update(
        {
            "calculation": "scf",
            "suffix": "ABACUS",
            "kpoint_file": "KPT",
            "out_chg": 1,
            "out_bandgap": 1,
        }
    )
    write_abacus_job(
        scf_inputs,
        structure,
        source_job,
        destination,
        stru_filename=stru_filename,
        kpoint=None,
    )
    WriteInput(scf_inputs, destination / "INPUT.scf")
    WriteKpt(kpoint.copy(), destination / "KPT.scf", kpoint_model)

    if relax:
        relax_inputs = deepcopy(scf_inputs)
        relax_inputs["calculation"] = "relax"
        relax_inputs.pop("out_bandgap", None)
        WriteInput(relax_inputs, destination / "INPUT.relax")

    nscf_inputs = deepcopy(scf_inputs)
    nscf_inputs.update(
        {
            "calculation": "nscf",
            "out_chg": None,
            "init_chg": "file",
            "berry_phase": 1,
        }
    )
    if use_k_continuity:
        nscf_inputs["use_k_continuity"] = True
    for axis in range(3):
        current = deepcopy(nscf_inputs)
        current["gdir"] = axis + 1
        WriteInput(current, destination / f"INPUT.nscf{axis + 1}")
        nscf_kpoint = kpoint.copy()
        nscf_kpoint[axis] = 2 * int(nscf_kpoint[axis])
        WriteKpt(nscf_kpoint, destination / f"KPT.nscf{axis + 1}", kpoint_model)

    _write_runner(
        destination / "run_piezoelectric.sh",
        abacus_command,
        relax=relax,
        stru_filename=stru_filename,
    )
    # Keep the short runner name used by the original abacus-test workflow.
    (destination / "run.sh").symlink_to("run_piezoelectric.sh")


def _deformed_structure(structure, strain: np.ndarray):
    """Apply a finite strain to the cell while preserving fractional positions.

    ``structure.cell`` holds the lattice vectors as its rows, and a homogeneous
    deformation maps every lattice vector by the deformation gradient
    ``F = I + strain``, so the deformed cell is ``C @ F.T``: the strain acts on
    the Cartesian components of each lattice vector.  Multiplying on the left
    instead would combine the lattice vectors with each other, which is a
    different deformation for every cell that is not diagonal.
    """
    result = deepcopy(structure)
    fractional = np.asarray(structure.coords_direct, dtype=float)
    cell = np.asarray(structure.cell, dtype=float)
    result.cell = (cell @ (np.eye(3) + strain).T).tolist()
    result.coords_direct = fractional.tolist()
    return result


def _symmetry_block(structure, symprec: float) -> dict[str, Any]:
    """Return the symmetry block recorded in the workflow manifest."""
    from abacustools.data.symmetry import crystallographic_symmetry

    analysis = crystallographic_symmetry(structure, symprec=symprec)
    if not analysis.get("available"):
        raise RuntimeError(
            "the symmetry of the reference cell could not be determined: "
            f"{analysis.get('error', 'unknown reason')}"
        )
    rotations = point_group_operations(structure, symprec=symprec)
    return {
        "point_group": analysis["point_group"],
        "schoenflies": analysis.get("schoenflies"),
        "space_group_symbol": analysis["space_group_symbol"],
        "space_group_number": analysis["space_group_number"],
        "crystal_system": analysis["crystal_system"],
        "operations": int(len(rotations)),
        "independent_components": independent_component_count(rotations),
        "symprec": float(symprec),
    }


def _prepare_one(
    job: Path,
    strain_magnitude: float,
    disp_type: str,
    *,
    use_k_continuity: bool,
    relax: bool,
    abacus_command: str,
    override: bool,
    modes=None,
    symmetry: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Prepare one job directory and return its generated task paths."""
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    if disp_type not in _DISP_TYPES:
        raise ValueError(f"invalid difference type: {disp_type}")
    strain_magnitude = _validate_strain(strain_magnitude)

    inputs, stru_filename, structure = read_job_structure(job)
    kpoint, kpoint_model = kpoint_mesh(job, inputs, structure)
    task_names = _task_names(disp_type, modes)
    existing = sorted(
        path.name
        for path in job.glob(f"{_TASK_PREFIX}*")
        if path.is_dir() or path.is_symlink()
    )
    clear_generated_jobs(
        job,
        sorted(set(task_names).union(existing)),
        override=override,
    )

    piezo_inputs = deepcopy(inputs)
    piezo_inputs.pop("kspacing", None)
    piezo_inputs["kpoint_file"] = "KPT"
    piezo_inputs["suffix"] = "ABACUS"
    if (job / "OUT.ABACUS" / "SPIN1_CHG.cube").is_file():
        piezo_inputs["init_chg"] = "file"

    if disp_type != "c":
        _write_piezo_inputs(
            job,
            structure,
            piezo_inputs,
            stru_filename,
            job / "piezoelectric_org",
            kpoint,
            kpoint_model,
            use_k_continuity=use_k_continuity,
            relax=relax,
            abacus_command=abacus_command,
        )

    for label, index, other_index in _selected_modes(modes):
        strain_matrix = _strain_matrix(index, other_index, strain_magnitude)
        if disp_type in {"f", "c"}:
            name = f"piezoelectric_{label}"
            destination = job / name
            _write_piezo_inputs(
                job,
                _deformed_structure(structure, strain_matrix),
                piezo_inputs,
                stru_filename,
                destination,
                kpoint,
                kpoint_model,
                use_k_continuity=use_k_continuity,
                relax=relax,
                abacus_command=abacus_command,
            )
            (destination / "strain_metadata.json").write_text(
                json.dumps(
                    {
                        "label": label,
                        "magnitude": strain_magnitude,
                        "strain_3x3": strain_matrix.tolist(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if disp_type in {"b", "c"}:
            name = f"piezoelectric_{label}_back"
            destination = job / name
            negative_matrix = -strain_matrix
            _write_piezo_inputs(
                job,
                _deformed_structure(structure, negative_matrix),
                piezo_inputs,
                stru_filename,
                destination,
                kpoint,
                kpoint_model,
                use_k_continuity=use_k_continuity,
                relax=relax,
                abacus_command=abacus_command,
            )
            (destination / "strain_metadata.json").write_text(
                json.dumps(
                    {
                        "label": label,
                        "magnitude": -strain_magnitude,
                        "strain_3x3": negative_matrix.tolist(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    write_manifest(
        job,
        "piezoelectric",
        tasks=task_names,
        strain=strain_magnitude,
        difference_type=disp_type,
        displacement_type=disp_type,
        strain_modes=_selectable_modes(modes),
        strains_mode="independent" if modes is not None else "full",
        symmetry=symmetry,
        modes=[
            {
                "label": label,
                "strain_3x3": _strain_matrix(
                    index, other_index, strain_magnitude
                ).tolist(),
            }
            for label, index, other_index in _selected_modes(modes)
        ],
        kpoint=kpoint,
        kpoint_model=kpoint_model,
        stru_filename=stru_filename,
        suffix="ABACUS",
        use_k_continuity=bool(use_k_continuity),
        relax=bool(relax),
    )
    return [str(job / name) for name in task_names]


def prepare_piezoelectric_jobs(
    jobs: Iterable[str | Path],
    strain: float,
    disp_type: str = "f",
    k_continuity: bool = False,
    relax: bool = False,
    *,
    abacus_command: str = "abacus",
    override: bool = True,
) -> list[str]:
    """Prepare piezoelectric tasks for one or more jobs.

    This compatibility helper mirrors the function exposed by the legacy
    ``abacustest model piezoelectric`` workflow.
    """
    if isinstance(jobs, (str, Path)):
        jobs = [jobs]
    result: list[str] = []
    for value in jobs:
        result.extend(
            _prepare_one(
                Path(value).absolute(),
                strain,
                disp_type,
                use_k_continuity=k_continuity,
                relax=relax,
                abacus_command=abacus_command,
                override=override,
            )
        )
    return result


def prepare(args: argparse.Namespace) -> int:
    """Prepare equilibrium and finite-strained Berry phase calculations."""
    job = Path(args.job).absolute()
    disp_type = getattr(args, "disp_type", getattr(args, "type", "f"))
    use_k_continuity = getattr(args, "use_k_continuity", False)
    strains_mode = getattr(args, "strains", "full")
    _, _, structure = read_job_structure(job)
    rotations = point_group_operations(
        structure, symprec=getattr(args, "symprec", DEFAULT_SYMPREC)
    )
    symmetry = _symmetry_block(structure, float(getattr(args, "symprec", DEFAULT_SYMPREC)))
    modes = (
        independent_strain_modes(rotations) if strains_mode == "independent" else None
    )
    _prepare_one(
        job,
        args.strain,
        disp_type,
        use_k_continuity=bool(use_k_continuity),
        relax=bool(getattr(args, "relax", False)),
        abacus_command=getattr(args, "abacus_command", "abacus"),
        override=bool(getattr(args, "override", False)),
        modes=modes,
        symmetry=symmetry,
    )
    print(f"  job: {job}")
    print(f"  strain: {float(args.strain):g} ({disp_type})")
    print(
        f"  point group: {symmetry['point_group']} "
        f"({symmetry['space_group_symbol']}, "
        f"space group {symmetry['space_group_number']})"
    )
    print(f"  independent piezoelectric components: {symmetry['independent_components']}")
    print(f"  modes: {len(_selectable_modes(modes))} of {len(_VOIGT_MODES)}")
    print(f"  generated calculations: {len(_task_names(disp_type, modes))}")
    return 0


def _read_task_data(
    task: Path,
    version: str,
    suffix: str,
    stru_filename: str,
) -> dict[str, Any]:
    """Collect SCF diagnostics and Berry phase data from one task."""
    data: dict[str, Any] = {
        "metrics": task_metrics(task, version),
        "polarization": None,
    }
    polarization = read_task_polarization(task, suffix)
    if polarization is None:
        return data
    structure = AbacusSTRU.read(task / stru_filename)
    if structure is None:
        print(f"  warning: skipping polarization from {task}: invalid structure")
        return data
    polarization = dict(polarization)
    polarization["cell"] = np.asarray(structure.cell, dtype=float).tolist()
    polarization["p_cart"] = polarization_cartesian(
        polarization["p_vec"], structure.cell
    )
    volume = polarization.get("volume")
    if volume is None:
        volume = abs(float(np.linalg.det(np.asarray(structure.cell, dtype=float))))
    if not np.isfinite(volume) or volume <= 0:
        print(f"  warning: skipping polarization from {task}: invalid cell volume")
        return data
    polarization["volume"] = float(volume)
    factor = _ELECTRON_ANGSTROM_SQUARED_TO_CM2 / float(volume)
    components = np.asarray(polarization["p_vec"], dtype=float) * factor
    modulus = np.asarray(polarization["mod"], dtype=float) * factor
    if components.shape != (3,) or modulus.shape != (3,):
        print(f"  warning: skipping polarization from {task}: invalid Berry data")
        return data
    polarization["components_cm2"] = components.tolist()
    polarization["modulus_cm2"] = modulus.tolist()
    polarization["p_cart_cm2"] = polarization_cartesian(components, structure.cell)
    data["polarization"] = polarization
    return data


def _read_full_polarization(
    folder: str | Path,
    *,
    suffix: str = "ABACUS",
    stru_filename: str = "STRU",
) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Read the legacy raw Cartesian polarization tuple for one task."""
    task = Path(folder)
    polarization = read_task_polarization(task, suffix)
    if polarization is None:
        raise ValueError(f"complete Berry phase output was not found in {task}")
    structure = AbacusSTRU.read(task / stru_filename)
    if structure is None:
        raise ValueError(f"structure was not found in {task}")
    return (
        np.asarray(polarization_cartesian(polarization["p_vec"], structure.cell)),
        np.asarray(polarization["mod"], dtype=float),
        polarization.get("volume"),
    )


def _read_strain_metadata(job: Path, directory: str) -> Optional[float]:
    """Read the signed strain magnitude stored with a strained task."""
    path = job / directory / "strain_metadata.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["magnitude"]
        value = float(value)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value


def _get_strain_magnitude(job: str | Path, directory: str) -> float:
    """Return the signed strain magnitude recorded for one task."""
    value = _read_strain_metadata(Path(job), directory)
    if value is None:
        raise FileNotFoundError(
            f"strain_metadata.json not found or invalid in {Path(job) / directory}"
        )
    return value


def _mode_difference(
    task_data: dict[str, dict[str, Any]],
    label: str,
    disp_type: str,
    strain: float,
) -> Optional[list[float]]:
    """Calculate one Cartesian polarization response by finite difference.

    ABACUS prints the polarization of each Berry phase direction as a dipole
    per cell referred to the lattice vectors, that is modulo the quantum of a
    lattice translation, and the quantum itself depends on the cell the value
    was computed in.  Comparing the printed values of two differently strained
    cells therefore mixes that geometry into the response: a centrosymmetric
    crystal, whose polarization sits exactly on the branch point, acquires a
    tensor of order of the quantum even though symmetry forces it to vanish,
    and every other crystal acquires symmetry forbidden components of the same
    order.

    The printed value divided by its own quantum is a phase in units of the
    quantum, and that ratio carries no geometry: the branch point is the same
    number in every cell.  The response is therefore taken as the phase
    difference, wrapped to the period of the printed polarization, converted
    back to a dipole with the quantum and the lattice of the *reference* cell
    alone, and only then turned into a Cartesian polarization density.
    """
    positive = task_data.get(f"piezoelectric_{label}")
    negative = task_data.get(f"piezoelectric_{label}_back")
    original = task_data.get("piezoelectric_org")
    if disp_type == "f":
        reference, displaced, denominator = original, positive, strain
    elif disp_type == "b":
        reference, displaced, denominator = original, negative, -strain
    else:
        reference, displaced, denominator = negative, positive, 2.0 * strain
    if (
        not reference
        or not displaced
        or not reference.get("polarization")
        or not displaced.get("polarization")
    ):
        print(f"  warning: insufficient Berry phase data for mode {label}")
        return None

    reference_p = reference["polarization"]
    displaced_p = displaced["polarization"]
    delta_phase = np.asarray(_phase_difference(reference_p, displaced_p), dtype=float)
    factor = _ELECTRON_ANGSTROM_SQUARED_TO_CM2 / float(reference_p["volume"])
    delta_cartesian = np.asarray(
        polarization_cartesian(
            delta_phase * np.asarray(reference_p["mod"], dtype=float) * factor,
            reference_p["cell"],
        ),
        dtype=float,
    )
    return [float(value) / denominator for value in delta_cartesian]


def _phase_difference(
    reference: dict[str, Any], displaced: dict[str, Any]
) -> np.ndarray:
    """Return the change of the polarization phase between two cells.

    ABACUS prints polarizations modulo twice the lattice translation quantum,
    so the printed value divided by the printed modulus runs over one period of
    one, independently of the cell.

    Args:
        reference: Polarization of the reference task.
        displaced: Polarization of the strained task.

    Returns:
        The phase difference of each of the three directions, wrapped to the
        period of the printed polarization.

    Raises:
        ValueError: When the two polarizations do not hold three directions or
            a quantum cannot be read.
    """
    reference_values = np.asarray(reference["p_vec"], dtype=float)
    displaced_values = np.asarray(displaced["p_vec"], dtype=float)
    reference_quantum = np.asarray(reference["mod"], dtype=float)
    displaced_quantum = np.asarray(displaced["mod"], dtype=float)
    for values in (reference_values, displaced_values, reference_quantum, displaced_quantum):
        if values.shape != (3,):
            raise ValueError("polarization vectors must contain three values")
    if np.any(reference_quantum <= 0.0) or np.any(displaced_quantum <= 0.0):
        raise ValueError("polarization quantum must be positive")
    delta = displaced_values / displaced_quantum - reference_values / reference_quantum
    return delta - np.rint(delta)


def _postprocess_one(
    job: Path,
    *,
    version: str,
    tasks: list[str],
    strain: float,
    disp_type: str,
    suffix: str,
    stru_filename: str,
) -> tuple[dict[str, Any], list[list[float]], dict[str, dict[str, Any]]]:
    """Postprocess one job and return diagnostics, tensor, and raw task data."""
    metrics: dict[str, Any] = {}
    task_data: dict[str, dict[str, Any]] = {}
    for name in tasks:
        task = job / name
        if not task.is_dir():
            print(f"  warning: missing piezoelectric task {task}")
            continue
        data = _read_task_data(task, version, suffix, stru_filename)
        task_data[name] = data
        metrics[name] = data["metrics"]

    tensor = [[0.0 for _ in _VOIGT_MODES] for _ in range(3)]
    for mode, (label, _, _) in enumerate(_VOIGT_MODES):
        values = _mode_difference(task_data, label, disp_type, strain)
        if values is not None:
            for direction in range(3):
                tensor[direction][mode] = values[direction]
    return metrics, tensor, task_data


def _infer_legacy_setup(job: Path) -> tuple[list[str], float, str]:
    """Infer task metadata for directories produced before manifests existed."""
    tasks = sorted(
        path.name
        for path in job.glob(f"{_TASK_PREFIX}*")
        if path.is_dir()
    )
    if not tasks:
        raise RuntimeError(f"no piezoelectric task directories found in {job}")
    has_org = "piezoelectric_org" in tasks
    has_positive = any(
        f"piezoelectric_{label}" in tasks for label, _, _ in _VOIGT_MODES
    )
    has_negative = any(
        f"piezoelectric_{label}_back" in tasks for label, _, _ in _VOIGT_MODES
    )
    if has_org and has_positive and not has_negative:
        disp_type = "f"
    elif has_org and has_negative and not has_positive:
        disp_type = "b"
    else:
        disp_type = "c"
    magnitudes = [
        abs(value)
        for name in tasks
        if name != "piezoelectric_org"
        for value in [_read_strain_metadata(job, name)]
        if value is not None and value != 0
    ]
    if not magnitudes:
        raise RuntimeError("piezoelectric task strain metadata was not found")
    strain = magnitudes[0]
    if not np.allclose(magnitudes, strain):
        raise RuntimeError("piezoelectric tasks use inconsistent strain magnitudes")
    return tasks, strain, disp_type


def _result_path(job: Path, filename: str) -> Path:
    """Resolve a workflow output path below JOB unless explicitly absolute."""
    path = Path(filename)
    return path if path.is_absolute() else job / path


def _summary(tensor: list[list[float]]) -> str:
    """Format the tensor in the same compact form as the legacy workflow."""
    text = [
        "Piezoelectric stress tensor e_{i,alpha} (C/m^2):",
        "Rows (i): polarization direction (x, y, z in Cartesian)",
        "Columns (alpha): Voigt strain mode (1=xx 2=yy 3=zz 4=yz 5=xz 6=xy)",
        "",
    ]
    for index, label in enumerate(("e_{x,alpha}", "e_{y,alpha}", "e_{z,alpha}")):
        text.append(
            f"{label}  " + "  ".join(f"{value:9.4f}" for value in tensor[index])
        )
    text.extend(
        [
            "",
            "The tensor is computed by finite differences of the Berry-phase",
            "polarization, taken as a phase so that the branch of each cell",
            "cancels; a centrosymmetric crystal therefore gives a vanishing",
            "tensor instead of one of the order of the polarization quantum.",
            "",
        ]
    )
    return "\n".join(text)


def postprocess_piezoelectric(
    jobs: Iterable[str | Path],
    *,
    version: Optional[str] = None,
) -> tuple[dict[str, Any], list[list[float]]]:
    """Postprocess one or more prepared piezoelectric jobs.

    The return shape mirrors the legacy model helper: task metrics followed by
    the 3x6 tensor.
    """
    if isinstance(jobs, (str, Path)):
        jobs = [jobs]
    all_metrics: dict[str, Any] = {}
    tensor = [[0.0 for _ in _VOIGT_MODES] for _ in range(3)]
    for value in jobs:
        job = Path(value).absolute()
        if not job.is_dir():
            print(f"warning: job path {job} is not a directory, skipping")
            continue
        try:
            manifest = read_manifest(job, "piezoelectric", [])
        except FileNotFoundError:
            tasks, strain, disp_type = _infer_legacy_setup(job)
            suffix = "ABACUS"
            stru_filename = "STRU"
        else:
            tasks = manifest.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                raise RuntimeError("piezoelectric workflow manifest has no tasks")
            strain = _validate_strain(float(manifest.get("strain", 0)))
            disp_type = str(
                manifest.get("difference_type", manifest.get("displacement_type", "f"))
            )
            if disp_type not in _DISP_TYPES:
                raise RuntimeError("piezoelectric manifest has an invalid difference type")
            suffix = str(manifest.get("suffix", "ABACUS"))
            stru_filename = str(manifest.get("stru_filename", "STRU"))
        metrics, job_tensor, _ = _postprocess_one(
            job,
            version=version,
            tasks=tasks,
            strain=strain,
            disp_type=disp_type,
            suffix=suffix,
            stru_filename=stru_filename,
        )
        all_metrics.update({str(job / name): value for name, value in metrics.items()})
        # Multiple jobs are accepted for legacy compatibility. Each job is an
        # independent result; retain the historical last-job tensor behavior.
        tensor = job_tensor
    return all_metrics, tensor


def _symmetry_from_manifest(
    job: Path, recorded: Any, symprec: float
) -> dict[str, Any]:
    """Return the symmetry of the reference cell.

    The preparation stage records it in the manifest; a manifest written
    before that information existed is completed here from the reference
    structure.
    """
    if isinstance(recorded, dict) and recorded.get("point_group"):
        return dict(recorded)
    _, _, structure = read_job_structure(job)
    return _symmetry_block(structure, symprec)


def postprocess(args: argparse.Namespace) -> int:
    """Calculate the piezoelectric stress tensor from Berry phase tasks."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    recorded_symmetry = None
    measured_modes: Optional[list[int]] = None
    try:
        manifest = read_manifest(job, "piezoelectric", [])
    except FileNotFoundError:
        tasks, strain, disp_type = _infer_legacy_setup(job)
        suffix = "ABACUS"
        stru_filename = "STRU"
    else:
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise RuntimeError("piezoelectric workflow manifest has no tasks")
        strain = _validate_strain(float(manifest.get("strain", 0)))
        disp_type = str(
            manifest.get("difference_type", manifest.get("displacement_type", "f"))
        )
        if disp_type not in _DISP_TYPES:
            raise RuntimeError("piezoelectric manifest has an invalid difference type")
        suffix = str(manifest.get("suffix", "ABACUS"))
        stru_filename = str(manifest.get("stru_filename", "STRU"))
        recorded_symmetry = manifest.get("symmetry")
        modes = manifest.get("strain_modes")
        if isinstance(modes, list) and modes:
            measured_modes = [int(mode) for mode in modes]

    metrics, tensor, task_data = _postprocess_one(
        job,
        version=args.version,
        tasks=tasks,
        strain=strain,
        disp_type=disp_type,
        suffix=suffix,
        stru_filename=stru_filename,
    )
    symprec = float(getattr(args, "symprec", DEFAULT_SYMPREC))
    symmetrize = bool(getattr(args, "symmetrize", True))
    symmetry = _symmetry_from_manifest(job, recorded_symmetry, symprec)
    if not measured_modes:
        measured_modes = list(range(len(_VOIGT_MODES)))
    _, _, structure = read_job_structure(job)
    rotations = point_group_operations(structure, symprec=symprec)

    method = str(getattr(args, "fit", "full"))
    if len(measured_modes) < len(_VOIGT_MODES) and method == "full":
        print(
            "  the strain set covers the independent modes only; "
            "fitting the independent components"
        )
        method = "independent"
    if method == "independent":
        tensor = fit_independent_piezoelectric(
            np.asarray(tensor, dtype=float), measured_modes, rotations
        ).tolist()
        raw = tensor
        residual = 0.0
        print("  fit: independent components")
    else:
        raw = tensor
        if symmetrize:
            tensor = symmetrize_piezoelectric_tensor(
                np.asarray(tensor, dtype=float), rotations
            ).tolist()
        residual = symmetrization_residual(np.asarray(raw), np.asarray(tensor))
    result = {
        "workflow": "piezoelectric",
        "difference_type": disp_type,
        "strain": strain,
        "voigt_modes": [label for label, _, _ in _VOIGT_MODES],
        "strain_modes": [int(mode) for mode in measured_modes],
        "tensor_layout": (
            "rows are Cartesian polarization directions; columns are Voigt strain modes"
        ),
        "units": {
            "piezoelectric_tensor": "C/m^2",
            "strain": "dimensionless",
            "symmetrization_residual": "C/m^2",
        },
        "piezoelectric_tensor": tensor,
        "piezoelectric_tensor_raw": raw,
        "symmetrization_residual": residual,
        "independent_components": independent_components(
            np.asarray(tensor, dtype=float), rotations
        ),
        "symmetry": symmetry,
        "fit": {
            "method": method,
            "strain_modes": [int(mode) for mode in measured_modes],
        },
        "tasks": {
            name: {
                "metrics": metrics.get(name, {}),
                "polarization": task_data.get(name, {}).get("polarization"),
            }
            for name in tasks
        },
    }
    output = _result_path(job, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = _summary(tensor)
    (job / "piezoelectric_summary.txt").write_text(summary + "\n", encoding="utf-8")

    print(f"  job: {job}")
    print(summary)
    print(
        f"  point group: {symmetry['point_group']} "
        f"({symmetry['space_group_symbol']}, "
        f"space group {symmetry['space_group_number']})"
    )
    print(
        f"  independent components: {symmetry['independent_components']} "
        f"from {len(measured_modes)} of {len(_VOIGT_MODES)} strain modes"
    )
    if method == "independent":
        print("  piezoelectric tensor (C/m^2, independent fit):")
    else:
        print("  piezoelectric tensor (C/m^2, symmetrized):")
    for row in tensor:
        print("    " + " ".join(f"{value: .8f}" for value in row))
    if method == "full" and symmetrize:
        print(f"  symmetrization residual: {residual:.8f} C/m^2")
    components = result["independent_components"]
    print(
        "  independent components (C/m^2): "
        + ", ".join(f"{name} = {value:.6f}" for name, value in components.items())
    )
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the piezoelectric preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "piezoelectric",
        "Calculate the piezoelectric tensor by finite-strain Berry phase differences.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
