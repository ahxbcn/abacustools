"""Shared implementation for parameter convergence-test workflows."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_job_structure,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)


@dataclass(frozen=True)
class ConvergenceSpec:
    """Describe one scalar parameter convergence test."""

    name: str
    label: str
    unit: str
    default_output: str
    default_plot: str
    ascending: bool
    requires_kpt: bool = True
    disable_gamma_only: bool = False


def _register_prepare_arguments(parser: argparse.ArgumentParser, spec: ConvergenceSpec) -> None:
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory used to prepare the convergence test.",
    )
    parser.add_argument(
        "--values", type=float, nargs="+", required=True,
        help=f"Values of {spec.name} to test ({spec.unit}).",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace existing generated convergence-test directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser, spec: ConvergenceSpec) -> None:
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory containing the prepared convergence-test calculations.",
    )
    parser.add_argument(
        "-v", "--version", default="LTS3.10.1",
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "--energy-tol", type=float, default=1.0e-4,
        help="Energy convergence tolerance in eV per atom; default: 1e-4.",
    )
    parser.add_argument(
        "-o", "--output", default=spec.default_output,
        help="JSON report filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--plot", default=spec.default_plot,
        help="Convergence plot filename, relative to JOB by default.",
    )


def _validate_values(values: list[float]) -> list[float]:
    """Validate and deduplicate scan values while preserving user order."""
    if len(values) < 2:
        raise ValueError("at least two convergence-test values are required")
    result = []
    for value in values:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("convergence-test values must be positive finite numbers")
        if value not in result:
            result.append(float(value))
    return result


def _task_names(spec: ConvergenceSpec, values: list[float]) -> list[str]:
    """Return stable generated task names for a scan."""
    return [f"{spec.name}_{index:02d}" for index in range(1, len(values) + 1)]


def prepare(args: argparse.Namespace, spec: ConvergenceSpec) -> int:
    """Prepare SCF jobs for one parameter convergence scan."""
    values = _validate_values(list(args.values))
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    inputs, stru_filename, structure = read_job_structure(job)
    scan_inputs = deepcopy(inputs)
    if str(inputs.get("calculation", "scf")).lower() != "scf":
        print(
            "  warning: convergence tests use SCF jobs; "
            f"replacing calculation={inputs.get('calculation')}"
        )
    scan_inputs["calculation"] = "scf"
    kpoint = kpoint_filename(job, inputs) if spec.requires_kpt else None
    if spec.disable_gamma_only:
        scan_inputs["gamma_only"] = 0

    tasks = _task_names(spec, values)
    clear_generated_jobs(job, tasks, override=args.override)
    print(f"  job: {job}")
    print(f"  parameter: {spec.name} ({spec.unit})")
    print(f"  atoms: {structure.natoms}")
    print(f"  values: {', '.join(f'{value:g}' for value in values)}")

    for task, value in zip(tasks, values):
        task_inputs = deepcopy(scan_inputs)
        task_inputs[spec.name] = value
        write_abacus_job(
            task_inputs,
            structure,
            job,
            job / task,
            stru_filename=stru_filename,
            kpoint=kpoint,
        )
        print(f"  prepared {task}: {spec.name}={value:g}")

    write_manifest(
        job,
        spec.name,
        tasks=tasks,
        values=values,
        parameter=spec.name,
        unit=spec.unit,
        calculation="scf",
        natoms=structure.natoms,
    )
    return 0


def _output_path(job: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job / path


def _read_point(
    job: Path,
    value: float,
    version: str,
    natoms: int,
) -> dict[str, Any]:
    """Read one scan point and retain missing-data diagnostics."""
    from abacustools.data.abacus_result import get_result_from_job

    try:
        result = get_result_from_job(job, ["energy", "converged", "normal_end"], version)
    except Exception as error:
        return {
            "parameter": value,
            "energy": None,
            "energy_per_atom": None,
            "converged": False,
            "normal_end": None,
            "status": "error",
            "message": str(error),
        }

    energy = result.get("energy")
    converged = bool(result.get("converged"))
    normal_end = result.get("normal_end")
    if energy is None:
        status = "missing-energy"
        message = "total energy was not found"
    elif not converged:
        status = "not-converged"
        message = "SCF did not converge"
    elif normal_end is False:
        status = "abnormal-end"
        message = "calculation did not end normally"
    else:
        status = "ok"
        message = None

    point = {
        "parameter": value,
        "energy": energy,
        "energy_per_atom": None if energy is None else float(energy) / natoms,
        "converged": converged,
        "normal_end": normal_end,
        "status": status,
    }
    if message:
        point["message"] = message
    return point


def _ordered_points(points: list[dict[str, Any]], spec: ConvergenceSpec) -> list[dict[str, Any]]:
    """Order points from less converged to more converged."""
    return sorted(points, key=lambda point: point["parameter"], reverse=not spec.ascending)


def _add_deltas(points: list[dict[str, Any]]) -> None:
    previous = None
    for point in points:
        energy = point.get("energy_per_atom")
        if energy is None or previous is None:
            point["delta_energy_per_atom"] = None
        else:
            point["delta_energy_per_atom"] = abs(energy - previous)
        if energy is not None:
            previous = energy


def _select_converged_point(
    points: list[dict[str, Any]], energy_tol: float
) -> Optional[dict[str, Any]]:
    """Select the earliest point whose complete suffix is converged."""
    for index, point in enumerate(points):
        suffix = points[index:]
        if any(item.get("status") != "ok" for item in suffix):
            continue
        deltas = [item.get("delta_energy_per_atom") for item in suffix[1:]]
        if deltas and all(delta is not None and delta <= energy_tol for delta in deltas):
            return point
    return None


def _plot(
    points: list[dict[str, Any]],
    spec: ConvergenceSpec,
    energy_tol: float,
    path: Path,
) -> None:
    """Plot energy per atom relative to the finest scan point."""
    import matplotlib.pyplot as plt

    completed = [point for point in points if point.get("energy_per_atom") is not None]
    reference = completed[-1]["energy_per_atom"]
    x = [point["parameter"] for point in completed]
    y = [point["energy_per_atom"] - reference for point in completed]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, "o-", linewidth=1.2)
    ax.axhline(energy_tol, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-energy_tol, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"{spec.name} ({spec.unit})")
    ax.set_ylabel("Energy difference per atom (eV)")
    ax.set_title(f"{spec.label.capitalize()} convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def postprocess(args: argparse.Namespace, spec: ConvergenceSpec) -> int:
    """Collect scan energies and write a convergence report and plot."""
    if not math.isfinite(args.energy_tol) or args.energy_tol <= 0:
        raise ValueError("energy tolerance must be a positive finite number")
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    manifest = read_manifest(job, spec.name, [])
    values = _validate_values([float(value) for value in manifest.get("values", [])])
    tasks = manifest.get("tasks")
    if tasks != _task_names(spec, values):
        raise RuntimeError(f"{spec.name} workflow manifest has invalid tasks or values")
    natoms = int(manifest.get("natoms", 0))
    if natoms <= 0:
        _, _, structure = read_job_structure(job)
        natoms = structure.natoms

    print(f"  job: {job}")
    print(f"  parameter: {spec.name} ({spec.unit})")
    print(f"  energy tolerance: {args.energy_tol:g} eV/atom")
    points = [
        _read_point(job / task, value, args.version, natoms)
        for task, value in zip(tasks, values)
    ]
    ordered = _ordered_points(points, spec)
    _add_deltas(ordered)
    selected = _select_converged_point(ordered, args.energy_tol)
    completed = [point for point in ordered if point.get("energy_per_atom") is not None]

    for point in ordered:
        message = point.get("message", "")
        delta = point.get("delta_energy_per_atom")
        delta_text = "-" if delta is None else f"{delta:.6g}"
        energy = point.get("energy_per_atom")
        energy_text = "-" if energy is None else f"{energy:.10f}"
        print(
            f"  {spec.name}={point['parameter']:g}: "
            f"status={point['status']}, energy/atom={energy_text}, "
            f"delta={delta_text} {message}"
        )

    result = {
        "parameter": spec.name,
        "unit": spec.unit,
        "energy_tolerance_eV_per_atom": float(args.energy_tol),
        "natoms": natoms,
        "all_tasks_complete": all(point["status"] == "ok" for point in points),
        "points": points,
        "selected": None if selected is None else {
            "parameter": selected["parameter"],
            "energy_per_atom": selected["energy_per_atom"],
            "reason": "successive energy difference is within tolerance",
        },
    }

    output = _output_path(job, args.output)
    plot = _output_path(job, args.plot)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if completed:
        _plot(ordered, spec, args.energy_tol, plot)
    print(f"  results: {output}")
    if completed:
        print(f"  plot: {plot}")
    else:
        print("  warning: no usable total energies were found; plot was skipped")
    if selected is None:
        print("  warning: no successive points meet the energy tolerance")
    else:
        print(f"  selected {spec.name}: {selected['parameter']:g} {spec.unit}")
    if not result["all_tasks_complete"]:
        print("  warning: one or more convergence-test tasks are incomplete")
    return 0


def register_workflow(
    subparsers,
    spec: ConvergenceSpec,
    *,
    aliases: Optional[list[str]] = None,
) -> None:
    """Register one parameter-specific convergence workflow."""
    register_stages(
        subparsers,
        spec.name,
        f"Test {spec.name} convergence with independent SCF calculations.",
        lambda args: prepare(args, spec),
        lambda args: postprocess(args, spec),
        lambda parser: _register_prepare_arguments(parser, spec),
        lambda parser: _register_postprocess_arguments(parser, spec),
        aliases=aliases,
    )
