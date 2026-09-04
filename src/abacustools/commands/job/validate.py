"""Implementation of the ``abacustools job validate`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abacustools.core.job import as_json, validate_job


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the ``job validate`` parser."""
    parser = subparsers.add_parser("validate", help="Check one ABACUS job directory.")
    parser.add_argument("job", type=_job_directory, metavar="JOB")
    parser.add_argument("--strict", action="store_true", help="Treat unknown INPUT keywords as errors.")
    parser.add_argument("--json", action="store_true", help="Print the diagnostic report as JSON.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Validate one ABACUS job directory."""
    report = validate_job(args.job, strict=args.strict)
    if args.json:
        print(json.dumps(as_json(report), indent=2, sort_keys=True))
    else:
        print(f"job: {Path(args.job).absolute()}")
        print(f"valid: {'yes' if report.valid else 'no'}")
        for issue in report.issues:
            print(f"{issue.level}: {issue.code}: {issue.message}")
    return 0 if report.valid else 1
