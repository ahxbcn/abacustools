"""Finite-difference validation of ABACUS analytic forces."""

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
    read_job_structure,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)
from .vibration import _selected_atoms, _validate_stepsize


_ROOT = "fdforce"
_EQUILIBRIUM_TASK = f"{_ROOT}/equilibrium"
_DIRECTIONS = ("x", "y", "z")


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-j", "--job", type=Path, required=True)
    parser.add_argument("-s", "--stepsize", type=float, default=0.01)
    parser.add_argument("-n", "--number", type=int, default=5)
    parser.add_argument("-i", "--index", dest="selected_atoms", type=int, nargs="+")
    parser.add_argument("--dir", dest="directions", choices=_DIRECTIONS, nargs="+", default=list(_DIRECTIONS))
    parser.add_argument(
        "--info", type=Path,
        help="Selection file with lines such as 'C 2 x y z'; paths are relative to JOB.",
    )
    parser.add_argument("--override", action="store_true")


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-j", "--job", type=Path, required=True)
    parser.add_argument("-v", "--version", default="LTS3.10.1")
    parser.add_argument("-o", "--output", default="fdforce_results.json")


def _validate_number(number: int) -> None:
    if isinstance(number, bool) or int(number) != number or number < 1:
        raise ValueError("number must be a positive integer")


def _read_info(path: Path) -> list[tuple[str, int, list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"could not find finite-difference selection file: {path}")
    selections = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 3:
            raise ValueError(f"invalid info line {line_number}: {line}")
        try:
            atom_number = int(fields[1])
        except ValueError as error:
            raise ValueError(f"invalid atom number on info line {line_number}: {line}") from error
        directions = [value.lower() for value in fields[2:]]
        if atom_number < 1 or any(value not in _DIRECTIONS for value in directions):
            raise ValueError(f"invalid atom selection on info line {line_number}: {line}")
        if len(set(directions)) != len(directions):
            raise ValueError(f"duplicate direction on info line {line_number}: {line}")
        selections.append((fields[0], atom_number, directions))
    if not selections:
        raise ValueError(f"selection file is empty: {path}")
    return selections


def _resolve_info_selections(structure, selections: list[tuple[str, int, list[str]]]):
    resolved = []
    seen = set()
    for label, ordinal, directions in selections:
        matches = [
            index for index, atom in enumerate(structure.atoms)
            if atom.label == label or atom.element == label
        ]
        if ordinal > len(matches):
            raise ValueError(f"{label} has only {len(matches)} atoms; requested {ordinal}")
        atom = matches[ordinal - 1] + 1
        duplicate = {(atom, direction) for direction in directions} & seen
        if duplicate:
            raise ValueError(f"duplicate atom selection: {sorted(duplicate)}")
        seen.update((atom, direction) for direction in directions)
        resolved.append((atom, directions))
    return resolved


def _selection(structure, selected_atoms: Any, directions: Any, info: Any, job: Path):
    if info is not None and selected_atoms is not None:
        raise ValueError("--info and --index cannot be used together")
    if info is not None:
        return _resolve_info_selections(structure, _read_info(Path(info) if Path(info).is_absolute() else job / info))
    atoms = _selected_atoms(selected_atoms, structure.natoms)
    return [(index + 1, list(directions)) for index in atoms]


def _displacement_metadata(selection, number: int, stepsize: float):
    metadata = []
    for atom, directions in selection:
        for direction in directions:
            for step_number in range(1, number + 1):
                for sign, sign_value in (("+", 1), ("-", -1)):
                    metadata.append({
                        "task": f"{_ROOT}/atom_{atom}_{direction}_{sign}{step_number}",
                        "atom": atom,
                        "direction": direction,
                        "direction_index": _DIRECTIONS.index(direction),
                        "step_number": step_number,
                        "sign": sign,
                        "sign_value": sign_value,
                        "displacement": sign_value * step_number * stepsize,
                    })
    return metadata


def prepare(args: argparse.Namespace) -> int:
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _validate_stepsize(args.stepsize)
    _validate_number(args.number)
    inputs, stru_filename, structure = read_job_structure(job)
    selection = _selection(
        structure, getattr(args, "selected_atoms", None),
        getattr(args, "directions", _DIRECTIONS), getattr(args, "info", None), job,
    )
    displacements = _displacement_metadata(selection, args.number, args.stepsize)
    task_names = [_EQUILIBRIUM_TASK] + [item["task"] for item in displacements]
    clear_generated_jobs(job, [_ROOT], override=getattr(args, "override", False))

    force_inputs = deepcopy(inputs)
    force_inputs["calculation"] = "scf"
    force_inputs["cal_force"] = 1
    try:
        if float(force_inputs.get("scf_thr", 1e-7)) > 1e-7:
            force_inputs["scf_thr"] = 1e-7
    except (TypeError, ValueError):
        force_inputs["scf_thr"] = 1e-7
    kpoint = kpoint_filename(job, inputs)
    write_abacus_job(force_inputs, structure, job, job / _EQUILIBRIUM_TASK, stru_filename=stru_filename, kpoint=kpoint)
    original_coords = np.asarray(structure.coords, dtype=float)
    for item in displacements:
        displaced = deepcopy(structure)
        coords = original_coords.copy()
        coords[item["atom"] - 1, item["direction_index"]] += item["displacement"]
        displaced.coords = coords.tolist()
        write_abacus_job(force_inputs, displaced, job, job / item["task"], stru_filename=stru_filename, kpoint=kpoint)

    write_manifest(
        job, "fdforce", tasks=task_names,
        selected_atoms=sorted({item[0] for item in selection}),
        directions=sorted({direction for _, dirs in selection for direction in dirs}, key=_DIRECTIONS.index),
        selections=[{"atom": atom, "directions": dirs} for atom, dirs in selection],
        stepsize=float(args.stepsize), number=int(args.number), displacements=displacements,
    )
    return 0


def _read_force_energy(job: Path, version: str, natoms: int) -> tuple[float, np.ndarray]:
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(job, ["energy", "force", "converged"], version)
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    if result["energy"] is None or not np.isfinite(float(result["energy"])):
        raise RuntimeError(f"energy was not found in the output: {job}")
    force = np.asarray(result["force"], dtype=float) if result["force"] is not None else None
    if force is None or force.shape != (natoms, 3) or not np.all(np.isfinite(force)):
        raise RuntimeError(f"invalid force array in the output: {job}")
    return float(result["energy"]), force


def _finite_difference_force(energy_minus: float, energy_plus: float, stepsize: float) -> float:
    return (energy_minus - energy_plus) / (2.0 * stepsize)


def _rmsd(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def postprocess(args: argparse.Namespace) -> int:
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _, _, structure = read_job_structure(job)
    manifest = read_manifest(job, "fdforce", [_EQUILIBRIUM_TASK])
    try:
        stepsize = float(manifest["stepsize"])
        number = int(manifest["number"])
        selected_atoms = _selected_atoms(manifest["selected_atoms"], structure.natoms)
        directions = manifest["directions"]
        selections = manifest["selections"]
        displacements = manifest["displacements"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("fdforce workflow manifest has invalid metadata") from error
    _validate_stepsize(stepsize)
    _validate_number(number)
    if not isinstance(directions, list) or any(direction not in _DIRECTIONS for direction in directions):
        raise RuntimeError("fdforce workflow manifest has invalid directions")
    if not isinstance(selections, list) or not selections:
        raise RuntimeError("fdforce workflow manifest has invalid selections")
    if not isinstance(displacements, list):
        raise RuntimeError("fdforce workflow manifest has invalid displacements")
    if any(
        not isinstance(item, dict) or not isinstance(item.get("task"), str)
        for item in displacements
    ):
        raise RuntimeError("fdforce workflow manifest has invalid displacement tasks")
    tasks = [_EQUILIBRIUM_TASK] + [item["task"] for item in displacements]
    read_manifest(job, "fdforce", tasks)
    data = {task: _read_force_energy(job / task, args.version, structure.natoms) for task in tasks}
    cases = {}
    for selection in selections:
        if not isinstance(selection, dict):
            raise RuntimeError("fdforce workflow manifest has invalid selections")
        atom = int(selection.get("atom", 0)) - 1
        if atom not in selected_atoms:
            raise RuntimeError("fdforce workflow manifest has an invalid selected atom")
        for direction in selection.get("directions", []):
            if direction not in directions:
                raise RuntimeError("fdforce workflow manifest has an invalid direction")
            items = [
                item for item in displacements
                if item.get("atom") == atom + 1 and item.get("direction") == direction
            ]
            if len(items) != 2 * number:
                raise RuntimeError(f"missing displacement tasks for atom {atom + 1} {direction}")
            try:
                by_step = {(int(item["step_number"]), item["sign"]): item for item in items}
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("fdforce workflow manifest has invalid displacement metadata") from error
            positions, energies, analytical, numerical, errors = [], [], [], [], []
            for k in range(-number + 1, number):
                task = _EQUILIBRIUM_TASK if k == 0 else by_step[(abs(k), "+" if k > 0 else "-")]["task"]
                plus = by_step[(abs(k + 1), "+")]["task"] if k + 1 > 0 else (_EQUILIBRIUM_TASK if k + 1 == 0 else by_step[(abs(k + 1), "-")]["task"])
                minus = by_step[(abs(k - 1), "+")]["task"] if k - 1 > 0 else (_EQUILIBRIUM_TASK if k - 1 == 0 else by_step[(abs(k - 1), "-")]["task"])
                analytical_value = data[task][1][atom, _DIRECTIONS.index(direction)]
                numerical_value = _finite_difference_force(data[minus][0], data[plus][0], stepsize)
                positions.append(k * stepsize)
                energies.append(data[task][0])
                analytical.append(float(analytical_value))
                numerical.append(float(numerical_value))
                errors.append(float(numerical_value - analytical_value))
            convergence = []
            for k in range(1, number + 1):
                convergence.append({"step": k * stepsize, "finite_difference_force": _finite_difference_force(data[by_step[(k, "-")]["task"]][0], data[by_step[(k, "+")]["task"]][0], k * stepsize)})
            cases[f"atom_{atom + 1}_{direction}"] = {
                "atom": atom + 1, "direction": direction, "position": positions,
                "energy": energies, "analytical_force": analytical,
                "finite_difference_force": numerical, "deviation": errors,
                "rmsd": _rmsd(errors), "equilibrium_force": float(data[_EQUILIBRIUM_TASK][1][atom, _DIRECTIONS.index(direction)]),
                "step_convergence": convergence,
            }
    result = {"stepsize": stepsize, "number": number, "force_unit": "eV/Angstrom", "cases": cases}
    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    register_stages(subparsers, "fdforce", "Validate analytic forces with finite differences.", prepare, postprocess, _register_prepare_arguments, _register_postprocess_arguments)
