"""The ``abacustools workflow bec`` workflow."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.io.abacus import ReadKpt, WriteInput, WriteKpt, kspacing2kpt

from .common import (
    clear_generated_jobs,
    read_job_structure,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)


_DIRECTIONS = ("x", "y", "z")
_DIRECTION_INDEX = {direction: index for index, direction in enumerate(_DIRECTIONS)}
_DISP_TYPES = ("f", "b", "c")
_BERRY_LOGS = ("running_nscf1.log", "running_nscf2.log", "running_nscf3.log")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the BEC preparation stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory used to prepare BEC calculations.",
    )
    parser.add_argument(
        "--stepsize", type=float, default=0.01,
        help="Finite-difference displacement in Angstrom, default: 0.01.",
    )
    parser.add_argument(
        "--index", type=int, nargs="+", default=[1],
        help="One-based atom indices to displace, default: 1.",
    )
    parser.add_argument(
        "--dir", "--direction", dest="directions", nargs="+",
        choices=_DIRECTIONS, default=list(_DIRECTIONS),
        help="Cartesian displacement directions, default: x y z.",
    )
    parser.add_argument(
        "--type", dest="disp_type", choices=_DISP_TYPES, default="f",
        help="Displacement type: f=forward, b=backward, c=central; default: f.",
    )
    parser.add_argument(
        "--use-k-continuity", action="store_true",
        help="Enable ABACUS k-point continuity for Berry phase calculations.",
    )
    parser.add_argument(
        "--abacus-command", default="abacus",
        help="ABACUS command written to generated run scripts, default: abacus.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace existing BEC calculation directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the BEC postprocessing stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory containing the prepared BEC calculations.",
    )
    parser.add_argument(
        "-v", "--version", default="LTS3.10.1",
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-o", "--output", default="bec_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )


def _validate_stepsize(value: float) -> float:
    """Validate and normalize the displacement size."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError("stepsize must be a positive finite number")
    return float(value)


def _validate_indices(indices: Iterable[int], natoms: int) -> list[int]:
    """Validate one-based atom indices and return zero-based unique indices."""
    values = list(indices)
    if not values or any(index < 1 or index > natoms for index in values):
        raise ValueError(f"atom indices must be between 1 and {natoms}")
    return sorted(set(index - 1 for index in values))


def _validate_directions(directions: Iterable[str]) -> list[str]:
    """Validate directions while preserving their command-line order."""
    result = []
    for direction in directions:
        if direction not in _DIRECTIONS:
            raise ValueError(f"invalid displacement direction: {direction}")
        if direction not in result:
            result.append(direction)
    if not result:
        raise ValueError("at least one displacement direction is required")
    return result


def _kpoint_mesh(job: Path, inputs: dict[str, Any], structure) -> tuple[list[float], str]:
    """Read a regular Gamma/Monkhorst-Pack mesh for BEC calculations."""
    try:
        if float(inputs.get("gamma_only", 0)) > 0:
            return [1, 1, 1, 0.0, 0.0, 0.0], "gamma"
    except (TypeError, ValueError):
        pass

    kspacing = inputs.get("kspacing")
    if kspacing not in (None, 0, "0", "0.0"):
        cell_bohr = np.asarray(structure.cell, dtype=float) * ANG_TO_BOHR
        mesh = kspacing2kpt(kspacing, cell_bohr)
        return [*mesh, 0.0, 0.0, 0.0], "gamma"

    filename = str(inputs.get("kpoint_file", "KPT"))
    kpoint_path = job / filename
    if not kpoint_path.is_file():
        raise FileNotFoundError(f"could not find KPT file: {kpoint_path}")
    try:
        kpoint, model = ReadKpt(kpoint_path)
    except SystemExit as error:
        raise ValueError(f"could not read KPT file: {kpoint_path}") from error
    if model not in {"gamma", "mp"} or len(kpoint) < 3:
        raise ValueError("BEC requires a regular Gamma or Monkhorst-Pack KPT mesh")
    mesh = [int(value) for value in kpoint[:3]]
    if any(value <= 0 for value in mesh):
        raise ValueError("KPT mesh dimensions must be positive")
    shifts = [float(value) for value in (list(kpoint[3:6]) + [0.0, 0.0, 0.0])[:3]]
    return [*mesh, *shifts], model


def _write_runner(path: Path, abacus_command: str) -> None:
    """Write a runner that executes SCF and all Berry phase directions."""
    command = shlex.join(shlex.split(abacus_command))
    script = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            "cp INPUT.scf INPUT",
            "cp KPT.scf KPT",
            f"{command} | tee scf.log",
            "",
            "for direction in 1 2 3; do",
            "    cp INPUT.nscf${direction} INPUT",
            "    cp KPT.nscf${direction} KPT",
            f"    {command} | tee nscf${{direction}}.log",
            "    mv -f OUT.ABACUS/running_nscf.log OUT.ABACUS/running_nscf${direction}.log",
            "done",
            "",
        ]
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _write_bec_inputs(
    job: Path,
    structure,
    inputs: dict[str, Any],
    stru_filename: str,
    destination: Path,
    kpoint: list[float],
    kpoint_model: str,
    use_k_continuity: bool,
    abacus_command: str,
) -> None:
    """Write one BEC task with SCF and three Berry phase input pairs."""
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
        job,
        destination,
        stru_filename=stru_filename,
        kpoint=None,
    )

    WriteInput(scf_inputs, destination / "INPUT.scf")
    WriteKpt(kpoint.copy(), destination / "KPT.scf", kpoint_model)

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

    _write_runner(destination / "run_bec.sh", abacus_command)


def _displacement_names(
    atom_indices: Iterable[int], directions: Iterable[str], disp_type: str
) -> list[str]:
    """Return generated task names in deterministic order."""
    names = []
    if disp_type != "c":
        names.append("bec_org")
    for index in atom_indices:
        for direction in directions:
            if disp_type in {"f", "c"}:
                names.append(f"bec_disp_atom{index}_{direction}")
            if disp_type in {"b", "c"}:
                names.append(f"bec_disp_atom{index}_{direction}_back")
    return names


def _displaced_structure(structure, atom_index: int, direction: str, amount: float):
    """Return a copy with one atom displaced in Cartesian Angstrom units."""
    displaced = deepcopy(structure)
    coordinates = np.asarray(displaced.coords, dtype=float)
    coordinates[atom_index, _DIRECTION_INDEX[direction]] += amount
    displaced.coords = coordinates.tolist()
    return displaced


def prepare(args: argparse.Namespace) -> int:
    """Prepare equilibrium and displaced Berry phase calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    stepsize = _validate_stepsize(args.stepsize)
    inputs, stru_filename, structure = read_job_structure(job)
    atom_indices = _validate_indices(args.index, structure.natoms)
    directions = _validate_directions(args.directions)
    if args.disp_type not in _DISP_TYPES:
        raise ValueError(f"invalid displacement type: {args.disp_type}")
    kpoint, kpoint_model = _kpoint_mesh(job, inputs, structure)

    task_names = _displacement_names(atom_indices, directions, args.disp_type)
    clear_generated_jobs(job, task_names, override=args.override)

    print(f"  job: {job}")
    print(f"  atoms: {', '.join(str(index + 1) for index in atom_indices)}")
    print(f"  directions: {' '.join(directions)}")
    print(f"  displacement: {stepsize} Angstrom ({args.disp_type})")
    print(f"  k-point mesh: {' '.join(str(value) for value in kpoint[:3])}")

    bec_inputs = deepcopy(inputs)
    bec_inputs.pop("kspacing", None)
    bec_inputs["kpoint_file"] = "KPT"
    bec_inputs["suffix"] = "ABACUS"

    if args.disp_type != "c":
        _write_bec_inputs(
            job, structure, bec_inputs, stru_filename, job / "bec_org",
            kpoint, kpoint_model, args.use_k_continuity, args.abacus_command,
        )
        print("  prepared bec_org")

    for index in atom_indices:
        for direction in directions:
            if args.disp_type in {"f", "c"}:
                name = f"bec_disp_atom{index}_{direction}"
                _write_bec_inputs(
                    job,
                    _displaced_structure(structure, index, direction, stepsize),
                    bec_inputs,
                    stru_filename,
                    job / name,
                    kpoint,
                    kpoint_model,
                    args.use_k_continuity,
                    args.abacus_command,
                )
                print(f"  prepared {name}")
            if args.disp_type in {"b", "c"}:
                name = f"bec_disp_atom{index}_{direction}_back"
                _write_bec_inputs(
                    job,
                    _displaced_structure(structure, index, direction, -stepsize),
                    bec_inputs,
                    stru_filename,
                    job / name,
                    kpoint,
                    kpoint_model,
                    args.use_k_continuity,
                    args.abacus_command,
                )
                print(f"  prepared {name}")

    write_manifest(
        job,
        "bec",
        tasks=task_names,
        atom_indices=[index + 1 for index in atom_indices],
        directions=directions,
        stepsize=stepsize,
        displacement_type=args.disp_type,
        kpoint=kpoint,
        kpoint_model=kpoint_model,
        stru_filename=stru_filename,
        suffix="ABACUS",
        use_k_continuity=bool(args.use_k_continuity),
    )
    print(f"  generated calculations: {len(task_names)}")
    return 0


def _numbers(line: str) -> list[float]:
    """Extract floating-point values from an ABACUS log line."""
    return [float(value.replace("D", "E").replace("d", "e")) for value in re.findall(_NUMBER, line)]


def read_berry_polarization(log_path: Path) -> dict[str, Any]:
    """Read Berry phase polarization and its quantum from one ABACUS log."""
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    p_vec: Optional[float] = None
    modulus: Optional[float] = None
    polarization_cm2: Optional[float] = None
    volume: Optional[float] = None
    direction: Optional[int] = None

    for index, line in enumerate(lines):
        if "Volume (A^3)" in line:
            values = _numbers(line)
            if values:
                volume = values[-1]
        if "The calculated polarization direction is" not in line:
            continue

        direction_values = re.findall(r"R\s*([123])", line)
        if direction_values:
            direction = int(direction_values[-1])
        for following in lines[index + 1 : index + 8]:
            values = _numbers(following)
            if "(e/Omega).bohr" in following and len(values) >= 2:
                p_vec = values[0] * BOHR_TO_ANG
                modulus = values[1] * BOHR_TO_ANG
            elif "C/m^2" in following and values:
                polarization_cm2 = values[0]

    if p_vec is None or modulus is None:
        raise ValueError(f"Berry phase polarization was not found in {log_path}")
    return {
        "direction": direction,
        "p_vec": p_vec,
        "mod": modulus,
        "polarization_cm2": polarization_cm2,
        "volume": volume,
    }


def polarization_cartesian(p_vec: Iterable[float], cell: Iterable[Iterable[float]]) -> list[float]:
    """Convert polarization components in lattice-vector basis to Cartesian axes."""
    vectors = np.asarray(list(cell), dtype=float)
    values = np.asarray(list(p_vec), dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    if vectors.shape != (3, 3) or values.shape != (3,) or np.any(lengths <= 0):
        raise ValueError("cell and polarization must contain three valid vectors")
    return np.sum(values[:, None] * vectors / lengths[:, None], axis=0).tolist()


def polarization_delta(
    original: Iterable[float], displaced: Iterable[float], modulus: Iterable[float]
) -> list[float]:
    """Calculate the shortest polarization change across the Berry phase branch."""
    delta = np.asarray(list(displaced), dtype=float) - np.asarray(list(original), dtype=float)
    quantum = np.asarray(list(modulus), dtype=float)
    if delta.shape != (3,) or quantum.shape != (3,):
        raise ValueError("polarization vectors must contain three values")
    valid = quantum > 0
    delta[valid] -= np.rint(delta[valid] / quantum[valid]) * quantum[valid]
    return delta.tolist()


def _read_task_polarization(task: Path, suffix: str) -> Optional[dict[str, Any]]:
    """Read all three Berry phase directions, returning None when incomplete."""
    output = task / f"OUT.{suffix}"
    try:
        values = [read_berry_polarization(output / name) for name in _BERRY_LOGS]
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"  warning: skipping incomplete BEC task {task}: {error}")
        return None
    return {
        "p_vec": [item["p_vec"] for item in values],
        "mod": [item["mod"] for item in values],
        "polarization_cm2": [item["polarization_cm2"] for item in values],
        "volume": values[0]["volume"],
    }


def _task_metrics(task: Path, version: str) -> dict[str, Any]:
    """Collect available SCF metrics without failing incomplete tasks."""
    from abacustools.data.abacus_result import get_result_from_job

    try:
        return get_result_from_job(
            task,
            ["energy", "drho", "denergy", "scf_steps", "converged"],
            version,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        return {"error": str(error)}


def _task_data(task: Path, version: str, suffix: str) -> dict[str, Any]:
    """Collect SCF and Berry phase information for one task."""
    return {
        "metrics": _task_metrics(task, version),
        "polarization": _read_task_polarization(task, suffix),
    }


def _calculate_tensor(
    task_data: dict[str, dict[str, Any]],
    structure,
    atom_index: int,
    directions: Iterable[str],
    stepsize: float,
    disp_type: str,
) -> tuple[list[Optional[list[float]]], list[Optional[list[float]]], list[Optional[list[float]]]]:
    """Calculate one atom's BEC tensor and retain intermediate polarization data."""
    tensor: list[Optional[list[float]]] = [None, None, None]
    displacements: list[Optional[list[float]]] = [None, None, None]
    displaced_polarizations: list[Optional[list[float]]] = [None, None, None]
    equilibrium = task_data.get("bec_org")

    for direction in directions:
        axis = _DIRECTION_INDEX[direction]
        positive = task_data.get(f"bec_disp_atom{atom_index}_{direction}")
        negative = task_data.get(f"bec_disp_atom{atom_index}_{direction}_back")
        if disp_type == "f":
            reference, displaced, signed_step = equilibrium, positive, stepsize
        elif disp_type == "b":
            reference, displaced, signed_step = equilibrium, negative, -stepsize
        else:
            reference, displaced, signed_step = negative, positive, 2.0 * stepsize
        if (
            not reference
            or not displaced
            or not reference.get("polarization")
            or not displaced.get("polarization")
        ):
            continue

        reference_p = reference["polarization"]
        displaced_p = displaced["polarization"]
        delta_lattice = polarization_delta(
            reference_p["p_vec"], displaced_p["p_vec"], displaced_p["mod"]
        )
        delta_cartesian = polarization_cartesian(delta_lattice, structure.cell)
        tensor[axis] = [value / signed_step for value in delta_cartesian]
        displacement = np.zeros(3, dtype=float)
        displacement[axis] = signed_step
        displacements[axis] = displacement.tolist()
        displaced_polarizations[axis] = displaced_p["p_vec"]

    return tensor, displacements, displaced_polarizations


def _result_path(job: Path, filename: str) -> Path:
    """Resolve a workflow output path."""
    path = Path(filename)
    return path if path.is_absolute() else job / path


def postprocess(args: argparse.Namespace) -> int:
    """Calculate BEC tensors from the prepared Berry phase tasks."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    manifest = read_manifest(job, "bec", [])
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("BEC workflow manifest has no tasks")
    manifest = read_manifest(job, "bec", tasks)

    directions = _validate_directions(manifest.get("directions", []))
    stepsize = _validate_stepsize(float(manifest.get("stepsize", 0)))
    disp_type = str(manifest.get("displacement_type", "f"))
    suffix = str(manifest.get("suffix", "ABACUS"))
    _, _, structure = read_job_structure(job)
    try:
        atom_indices = _validate_indices(
            manifest.get("atom_indices", []), structure.natoms
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("BEC workflow manifest contains invalid atom indices") from error
    if any(index < 0 or index >= structure.natoms for index in atom_indices):
        raise RuntimeError("BEC workflow manifest contains invalid atom indices")
    if disp_type not in _DISP_TYPES:
        raise RuntimeError("BEC workflow manifest contains an invalid displacement type")

    collected = {
        name: _task_data(job / name, args.version, suffix)
        for name in tasks
    }
    atoms = []
    for atom_index in atom_indices:
        tensor, displacements, displaced_polarizations = _calculate_tensor(
            collected, structure, atom_index, directions, stepsize, disp_type
        )
        atoms.append(
            {
                "index": atom_index + 1,
                "label": str(structure.labels[atom_index]),
                "bec_tensor": tensor,
                "displacements": displacements,
                "displaced_polarization": displaced_polarizations,
            }
        )

    result = {
        "workflow": "bec",
        "displacement_type": disp_type,
        "stepsize_angstrom": stepsize,
        "directions": directions,
        "tensor_layout": (
            "rows are displacement directions; columns are Cartesian "
            "polarization directions"
        ),
        "units": {
            "bec_tensor": "e",
            "displacement": "Angstrom",
            "polarization": "e Angstrom",
        },
        "atoms": atoms,
        "tasks": collected,
    }
    output = _result_path(job, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"  job: {job}")
    print("  Born effective charge tensors (e):")
    for atom in atoms:
        print(f"    atom {atom['index']} ({atom['label']}):")
        for direction, row in zip(_DIRECTIONS, atom["bec_tensor"]):
            values = "-" if row is None else " ".join(f"{value: .8f}" for value in row)
            print(f"      {direction}: {values}")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the BEC preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "bec",
        "Calculate Born effective charges with Berry phase finite differences.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
