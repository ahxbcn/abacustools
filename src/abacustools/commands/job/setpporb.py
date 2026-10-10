"""Implementation of the ``abacustools job setpporb`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abacustools.core.input_prep import (
    InputPreparationError,
    available_resource_libraries,
)
from abacustools.core.resource_switch import ResourceSwitchResult, switch_job_library


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the ``job setpporb`` parser."""
    parser = subparsers.add_parser(
        "setpporb",
        help="Re-resolve one or more jobs' pseudopotentials/orbitals from another library.",
    )
    parser.add_argument(
        "jobs",
        nargs="*",
        type=_job_directory,
        metavar="JOB",
        help="Job directories to update; the current directory when omitted.",
    )
    parser.add_argument(
        "--library",
        choices=available_resource_libraries(),
        default=None,
        help=(
            "Configured pseudopotential/orbital library to switch to; the configured "
            "default when omitted."
        ),
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Orbital variant such as SZ, DZP or TZDP; defaults to resources.orb_variant.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--copy-resources",
        dest="copy_resources",
        action="store_true",
        default=None,
        help="Copy the resolved files into the job instead of symlinking them.",
    )
    mode.add_argument(
        "--symlink",
        dest="copy_resources",
        action="store_false",
        help="Symlink the resolved files even if the job used copies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the resolved files and the changes without writing anything.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.set_defaults(handler=run)


def _print_result(result: ResourceSwitchResult) -> None:
    print(f"job: {result.job.absolute()}")
    library = result.library or "configured default"
    if result.variant:
        library += f" (variant {result.variant})"
    print(f"library: {library}")
    print(f"basis: {result.basis}")
    for element, filename in sorted(result.pseudopotentials.items()):
        print(f"pseudopotential {element}: {filename}")
    for element, filename in sorted(result.orbitals.items()):
        print(f"orbital {element}: {filename}")
    if result.dry_run:
        print("dry run: nothing was written")
    for name in result.installed:
        print(f"installed: {name}")
    for name in result.removed:
        print(f"removed: {name}")


def run(args: argparse.Namespace) -> int:
    """Switch the resource library of one or more ABACUS jobs."""
    jobs = args.jobs or [Path(".")]
    results = []
    status = 0
    for job in jobs:
        try:
            result = switch_job_library(
                job,
                library=args.library,
                variant=args.variant,
                copy_resources=args.copy_resources,
                dry_run=args.dry_run,
            )
        except InputPreparationError as error:
            status = 1
            if args.json:
                results.append({"job": str(Path(job).absolute()), "error": str(error)})
            else:
                print(f"job: {Path(job).absolute()}")
                print(f"error: {error}")
            continue
        results.append(result)
        if not args.json:
            _print_result(result)
    if args.json:
        payload = [
            result.as_json() if isinstance(result, ResourceSwitchResult) else result
            for result in results
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    return status
