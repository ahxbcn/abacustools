"""Implementation of the ``abacustools job monitor`` command."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

from abacustools.core.job import status_job, validate_job
from abacustools.data.abacus_result import read_relaxation_history


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the ``job monitor`` parser."""
    parser = subparsers.add_parser("monitor", help="Monitor one ABACUS job until it finishes.")
    parser.add_argument("job", type=_job_directory, metavar="JOB")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Show one status update and exit.")
    parser.add_argument(
        "--relax", "--geometry",
        dest="relaxation",
        action="store_true",
        help="Monitor geometry-optimization steps and their force/stress trends.",
    )
    parser.add_argument("--json", action="store_true", help="Print geometry history as JSON.")
    parser.add_argument("--csv", type=Path, metavar="FILE", help="Write geometry history to CSV and exit.")
    parser.add_argument("--plot", type=Path, metavar="FILE", help="Write a geometry-history plot and exit.")
    parser.set_defaults(handler=run)


def _print_status(job: Path) -> str:
    validation = validate_job(job)
    status = status_job(job, validation)
    progress = " ".join(f"{key}={value}" for key, value in status.progress.items() if value is not None)
    print(f"{status.state}: {progress}".rstrip())
    return status.state


def _relax_log(job: Path) -> tuple[str, Optional[Path]]:
    validation = validate_job(job)
    status = status_job(job, validation)
    return status.state, status.log


def _history_payload(job: Path, state: str, log: Optional[Path], history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "job": str(job.absolute()),
        "state": state,
        "log": None if log is None else str(log),
        "force_unit": "eV/Angstrom",
        "stress_unit": "kBar",
        "energy_unit": "eV",
        "steps": history,
    }


def _format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.8g}"


def _print_relaxation(job: Path, *, as_json: bool = False) -> str:
    state, log = _relax_log(job)
    history = read_relaxation_history(log) if log is not None else []
    payload = _history_payload(job, state, log, history)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"job: {job.absolute()}")
        print(f"state: {state}")
        if log:
            print(f"log: {log}")
        if not history:
            print("geometry optimization: no relaxation steps found")
        else:
            print("step  energy(eV)  dE(eV)  max_force(eV/A)  max_stress(kBar)  converged")
            for item in history:
                print(
                    f"{item['step']:4d}  {_format_metric(item['energy']):>11}  "
                    f"{_format_metric(item['energy_change']):>8}  "
                    f"{_format_metric(item['max_force']):>16}  "
                    f"{_format_metric(item['max_stress']):>16}  "
                    f"{'yes' if item['converged'] else 'no'}"
                )
    return state


def _write_relax_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "energy", "energy_change", "max_force", "max_stress", "converged"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field) for field in fields} for item in history)


def _write_relax_plot(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        raise RuntimeError("cannot plot geometry optimization history: no relaxation steps found")
    import matplotlib.pyplot as plt

    steps = [item["step"] for item in history]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    energy_steps = [item["step"] for item in history if item["energy"] is not None]
    energies = [item["energy"] for item in history if item["energy"] is not None]
    if energies:
        axes[0].plot(energy_steps, energies, "o-", label="Energy")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].grid(True, alpha=0.3)
    force_steps = [item["step"] for item in history if item["max_force"] is not None]
    force_values = [item["max_force"] for item in history if item["max_force"] is not None]
    stress_steps = [item["step"] for item in history if item["max_stress"] is not None]
    stress_values = [item["max_stress"] for item in history if item["max_stress"] is not None]
    axes[1].plot(force_steps, force_values, "o-", label="Max force (eV/A)")
    if stress_values:
        stress_axis = axes[1].twinx()
        stress_axis.plot(stress_steps, stress_values, "s-", color="tab:red", label="Max stress (kBar)")
        stress_axis.set_ylabel("Max stress (kBar)")
    axes[1].set_xlabel("Relaxation step")
    axes[1].set_ylabel("Max force (eV/A)")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _run_relaxation(args: argparse.Namespace) -> int:
    if args.json and (args.csv or args.plot):
        raise ValueError("--json cannot be combined with --csv or --plot")
    if (args.csv or args.plot) and not args.once:
        raise ValueError("--csv and --plot require --once")
    if args.json:
        args.once = True
    while True:
        state, log = _relax_log(Path(args.job))
        history = read_relaxation_history(log) if log is not None else []
        if args.csv:
            _write_relax_csv(args.csv, history)
        if args.plot:
            _write_relax_plot(args.plot, history)
        if args.json:
            print(json.dumps(_history_payload(Path(args.job), state, log, history), indent=2, sort_keys=True))
        elif args.csv:
            print(f"geometry history: {args.csv}")
        elif args.plot:
            print(f"geometry history plot: {args.plot}")
        else:
            _print_relaxation(Path(args.job), as_json=False)
        if args.once or state in {"invalid", "failed", "converged"}:
            return 0 if state not in {"invalid", "failed"} else 1
        time.sleep(args.interval)


def run(args: argparse.Namespace) -> int:
    """Monitor one ABACUS job."""
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    if args.relaxation:
        return _run_relaxation(args)
    if args.json or args.csv or args.plot:
        raise ValueError("--json, --csv, and --plot require --relax")
    while True:
        state = _print_status(Path(args.job))
        if args.once or state in {"invalid", "failed", "converged"}:
            return 0 if state not in {"invalid", "failed"} else 1
        time.sleep(args.interval)
