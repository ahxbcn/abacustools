"""Implementation of the ``abacustools job status`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abacustools.core.job import as_json, status_job, validate_job


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the ``job status`` parser."""
    parser = subparsers.add_parser("status", help="Show one ABACUS job's state and progress.")
    parser.add_argument("job", type=_job_directory, metavar="JOB")
    parser.add_argument("--json", action="store_true", help="Print the status as JSON.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Show one ABACUS job's current state and progress."""
    validation = validate_job(args.job)
    status = status_job(args.job, validation)
    result = {"state": status.state, "progress": status.progress, "log": status.log}
    if args.json:
        print(json.dumps(as_json(result), indent=2, sort_keys=True))
    else:
        print(f"job: {Path(args.job).absolute()}")
        print(f"state: {status.state}")
        if status.log:
            print(f"log: {status.log}")
        for name, value in status.progress.items():
            if value is not None:
                print(f"{name}: {value}")
    return 0 if status.state not in {"invalid", "failed"} else 1
