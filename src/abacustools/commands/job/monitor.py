"""Implementation of the ``abacustools job monitor`` command."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from abacustools.core.job import status_job, validate_job


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
    parser.set_defaults(handler=run)


def _print_status(job: Path) -> str:
    validation = validate_job(job)
    status = status_job(job, validation)
    progress = " ".join(f"{key}={value}" for key, value in status.progress.items() if value is not None)
    print(f"{status.state}: {progress}".rstrip())
    return status.state


def run(args: argparse.Namespace) -> int:
    """Monitor one ABACUS job."""
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    while True:
        state = _print_status(Path(args.job))
        if args.once or state in {"invalid", "failed", "converged"}:
            return 0 if state not in {"invalid", "failed"} else 1
        time.sleep(args.interval)
