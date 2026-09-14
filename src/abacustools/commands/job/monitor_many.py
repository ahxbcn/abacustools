"""Implementation of the ``abacustools job monitor-many`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from abacustools.core.job import JobStatus, JobValidation, status_job, validate_job
from abacustools.data.abacus_result import read_relaxation_history

from .monitor import (
    _format_energy,
    _format_energy_change,
    _format_metric,
    _job_directory,
    _print_table,
)


def register_parser(subparsers) -> None:
    """Register batch monitoring for ABACUS jobs."""
    parser = subparsers.add_parser(
        "monitor-many",
        help="Show one update of several ABACUS jobs.",
    )
    parser.add_argument(
        "-j", "--job",
        required=True,
        action="extend",
        nargs="+",
        type=_job_directory,
        metavar="JOB",
        help="Directories of ABACUS jobs to monitor.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Deprecated: the monitor prints one update and exits without waiting.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Deprecated: the monitor always prints one update and exits.",
    )
    parser.add_argument("--json", action="store_true", help="Print the batch status as JSON.")
    parser.set_defaults(handler=run)


def _job_name(job: Path, jobs: list[Path]) -> str:
    """Return a short unique label for a job path."""
    name = job.name or str(job)
    if sum(other.name == name for other in jobs) > 1:
        return f"{job.parent.name}/{name}"
    return name


def _is_relaxation(calculation: str) -> bool:
    return calculation in {"relax", "cell-relax", "cell_relax", "md"}


def _summary(job: Path, jobs: list[Path]) -> dict[str, Any]:
    validation: JobValidation = validate_job(job)
    status: JobStatus = status_job(job, validation)
    calculation = str(validation.inputs.get("calculation", "scf")).lower()
    item: dict[str, Any] = {
        "job": str(job.absolute()),
        "name": _job_name(job, jobs),
        "state": status.state,
        "calculation": calculation,
        "step": None,
        "energy": status.progress.get("energy"),
        "energy_change": status.progress.get("denergy"),
        "max_force": status.progress.get("largest_force"),
        "max_stress": status.progress.get("largest_stress"),
        "relaxation_steps": None,
        "log": None if status.log is None else str(status.log),
    }
    if _is_relaxation(calculation) and status.log is not None:
        history = read_relaxation_history(status.log)
        if history:
            latest = history[-1]
            item.update(
                {
                    "step": latest["step"],
                    "energy": latest["energy"],
                    "energy_change": latest["energy_change"],
                    "max_force": latest["max_force"],
                    "max_stress": latest["max_stress"],
                    "relaxation_steps": len(history),
                }
            )
    return item


def _print_jobs(items: list[dict[str, Any]]) -> None:
    """Print one row per job."""
    header = [
        "job", "state", "calculation", "step",
        "energy(eV)", "dE(eV)", "max_force(eV/A)", "max_stress(kBar)",
    ]
    rows = [
        [
            item["name"],
            item["state"],
            item["calculation"],
            "-" if item["step"] is None else str(item["step"]),
            _format_energy(item["energy"]),
            _format_energy_change(item["energy_change"]),
            _format_metric(item["max_force"]),
            _format_metric(item["max_stress"]),
        ]
        for item in items
    ]
    _print_table(header, rows)


def _print_items(items: list[dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps({"jobs": items}, indent=2, sort_keys=True))
    else:
        _print_jobs(items)


def run(args: argparse.Namespace) -> int:
    """Print one update for several ABACUS jobs."""
    jobs = [Path(job).absolute() for job in args.job]
    if not jobs:
        raise ValueError("at least one job is required")

    items = [_summary(job, jobs) for job in jobs]
    _print_items(items, args.json)
    return 1 if any(item["state"] in {"invalid", "failed"} for item in items) else 0
