"""Finite-difference validation of ABACUS analytic stresses."""

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
_ROOT = "fdstress"
_EQUILIBRIUM_TASK = f"{_ROOT}/equilibrium"
_DEFAULT_COMPONENTS = ("11", "12", "13", "22", "23", "33")
_EV_PER_A3_TO_KBAR = 1602.1757722389546


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-j", "--job", type=Path, required=True)
    parser.add_argument("-s", "--step", type=float, default=0.0001)
    parser.add_argument("-n", "--number", type=int, default=5)
    parser.add_argument("--component", "--comp", dest="components", nargs="+", default=list(_DEFAULT_COMPONENTS))
    parser.add_argument("--all-components", action="store_true")
    parser.add_argument("--override", action="store_true")


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-j", "--job", type=Path, required=True)
    parser.add_argument("-v", "--version", default="LTS3.10.1")
    parser.add_argument("-o", "--output", default="fdstress_results.json")


def _validate_parameters(step: float, number: int) -> None:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a positive finite number")
    if isinstance(number, bool) or int(number) != number or number < 1:
        raise ValueError("number must be a positive integer")
    if number * step >= 1:
        raise ValueError("number * step must be smaller than 1")


def _components(values: Any, all_components: bool = False) -> list[str]:
    if all_components:
        values = [f"{i}{j}" for i in range(1, 4) for j in range(1, 4)]
    if not values:
        raise ValueError("at least one stress component is required")
    normalized = [str(value) for value in values]
    if any(len(value) != 2 or value[0] not in "123" or value[1] not in "123" for value in normalized):
        raise ValueError("stress components must be two digits from 1 to 3")
    if len(set(normalized)) != len(normalized):
        raise ValueError("stress components must not contain duplicates")
    return normalized


def _deformation_metadata(components: list[str], step: float, number: int):
    result = []
    for component in components:
        for index in range(-number, number + 1):
            task = _EQUILIBRIUM_TASK if index == 0 else f"{_ROOT}/component_{component}_{'plus' if index > 0 else 'minus'}{abs(index)}"
            result.append({"task": task, "component": component, "index": index, "strain": index * step})
    return result


def prepare(args: argparse.Namespace) -> int:
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _validate_parameters(args.step, args.number)
    components = _components(getattr(args, "components", _DEFAULT_COMPONENTS), getattr(args, "all_components", False))
    inputs, stru_filename, structure = read_job_structure(job)
    kpoint = kpoint_filename(job, inputs)
    stress_inputs = deepcopy(inputs)
    stress_inputs["calculation"] = "scf"
    stress_inputs["cal_stress"] = 1
    deformations = _deformation_metadata(components, args.step, args.number)
    task_names = [_EQUILIBRIUM_TASK] + [item["task"] for item in deformations if item["index"] != 0]
    clear_generated_jobs(job, [_ROOT], override=getattr(args, "override", False))
    write_abacus_job(stress_inputs, structure, job, job / _EQUILIBRIUM_TASK, stru_filename=stru_filename, kpoint=kpoint)
    fractional = np.asarray(structure.coords_direct, dtype=float)
    original_cell = np.asarray(structure.cell, dtype=float)
    volume = abs(float(np.linalg.det(original_cell)))
    if volume <= 0:
        raise ValueError("stress finite differences require a non-zero cell volume")
    for item in deformations:
        if item["index"] == 0:
            continue
        i, j = int(item["component"][0]) - 1, int(item["component"][1]) - 1
        cell = original_cell.copy()
        cell[:, i] += item["strain"] * original_cell[:, j]
        deformed = deepcopy(structure)
        deformed.cell = cell.tolist()
        deformed.coords_direct = fractional.tolist()
        item["volume"] = abs(float(np.linalg.det(cell)))
        write_abacus_job(stress_inputs, deformed, job, job / item["task"], stru_filename=stru_filename, kpoint=kpoint)
    for item in deformations:
        if item["index"] == 0:
            item["volume"] = volume
    write_manifest(job, "fdstress", tasks=task_names, step=float(args.step), number=int(args.number), components=components, deformations=deformations, stress_unit="kBar", stress_sign_convention="abacus", energy_unit="eV", volume_unit="Angstrom^3")
    return 0


def _read_energy(job: Path, version: str) -> float:
    from abacustools.data.abacus_result import get_result_from_job
    result = get_result_from_job(job, ["energy", "converged"], version)
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    if result["energy"] is None or not np.isfinite(float(result["energy"])):
        raise RuntimeError(f"energy was not found in the output: {job}")
    return float(result["energy"])


def _read_stress(job: Path, version: str) -> np.ndarray:
    """Read one converged ABACUS stress tensor in its native kBar unit."""
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(job, ["stress", "converged"], version)
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    if result["stress"] is None:
        raise RuntimeError(f"stress was not found in the output: {job}")
    stress = np.asarray(result["stress"], dtype=float)
    if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
        raise RuntimeError(f"invalid stress tensor in the output: {job}")
    return stress


def _effective_step(component: str, index: int, step: float) -> float:
    return step / (1.0 + index * step) if component[0] == component[1] else step


def _finite_difference_stress(energy_minus: float, energy_plus: float, volume: float, effective_step: float) -> float:
    return (energy_minus - energy_plus) / (2.0 * effective_step * volume) * _EV_PER_A3_TO_KBAR


def postprocess(args: argparse.Namespace) -> int:
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _, _, structure = read_job_structure(job)
    manifest = read_manifest(job, "fdstress", [_EQUILIBRIUM_TASK])
    try:
        step = float(manifest["step"])
        number = int(manifest["number"])
        components = _components(manifest["components"])
        deformations = manifest["deformations"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("fdstress workflow manifest has invalid metadata") from error
    _validate_parameters(step, number)
    if not isinstance(deformations, list):
        raise RuntimeError("fdstress workflow manifest has invalid deformations")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("task"), str)
        or item.get("index") == 0
        and item.get("task") != _EQUILIBRIUM_TASK
        for item in deformations
    ):
        raise RuntimeError("fdstress workflow manifest has invalid deformation tasks")
    tasks = [_EQUILIBRIUM_TASK] + [
        item["task"] for item in deformations if item["index"] != 0
    ]
    read_manifest(job, "fdstress", tasks)
    energies = {task: _read_energy(job / task, args.version) for task in tasks}
    stresses = {task: _read_stress(job / task, args.version, False) for task in tasks}
    equilibrium_cell = np.asarray(structure.cell, dtype=float)
    equilibrium_volume = abs(float(np.linalg.det(equilibrium_cell)))
    cases = {}
    for component in components:
        states = [item for item in deformations if item.get("component") == component]
        if len(states) != 2 * number + 1:
            raise RuntimeError(f"missing deformation tasks for stress component {component}")
        by_index = {int(item["index"]): item for item in states}
        component_index = (int(component[0]) - 1, int(component[1]) - 1)
        positions, analytical, numerical, deviations = [], [], [], []
        for index in range(-number + 1, number):
            current = by_index[index]
            minus = by_index[index - 1]
            plus = by_index[index + 1]
            analytical_value = float(stresses[current["task"]][component_index])
            numerical_value = _finite_difference_stress(energies[minus["task"]], energies[plus["task"]], float(current["volume"]), _effective_step(component, index, step))
            positions.append(index * step)
            analytical.append(analytical_value)
            numerical.append(numerical_value)
            deviations.append(numerical_value - analytical_value)
        convergence = []
        for magnitude in range(1, number + 1):
            convergence.append({"step": magnitude * step, "finite_difference_stress": _finite_difference_stress(energies[by_index[-magnitude]["task"]], energies[by_index[magnitude]["task"]], equilibrium_volume, magnitude * step)})
        cases[component] = {
            "component": component, "position": positions,
            "analytical_stress": analytical, "finite_difference_stress": numerical,
            "deviation": deviations, "rmsd": float(np.sqrt(np.mean(np.square(deviations)))),
            "equilibrium_stress": float(stresses[_EQUILIBRIUM_TASK][component_index]),
            "step_convergence": convergence,
        }
    result = {"step": step, "number": number, "components": components, "stress_unit": "kBar", "stress_sign_convention": "abacus", "cases": cases}
    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    register_stages(subparsers, "fdstress", "Validate analytic stresses with finite differences.", prepare, postprocess, _register_prepare_arguments, _register_postprocess_arguments)
