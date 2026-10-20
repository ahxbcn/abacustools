"""Implementation of the ``abacustools postprocess md`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from abacustools.data.md import (
    read_trajectory,
    select_frames,
    summarize_trajectory,
    write_trajectory,
)
from abacustools.data.versions import default_version
from abacustools.core.render import cli_output


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the arguments of the MD trajectory command."""
    parser.add_argument(
        "-j", "--job",
        required=True,
        type=_job_directory,
        help="ABACUS job directory of a molecular-dynamics calculation.",
    )
    parser.add_argument(
        "-o", "--output",
        default="trajectory.extxyz",
        help=(
            "Trajectory file to write, relative to JOB by default; the suffix "
            "selects the format, default: trajectory.extxyz."
        ),
    )
    parser.add_argument(
        "--format",
        default=None,
        help="Explicit ASE output format, such as extxyz, xyz or traj.",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=None,
        metavar="STEP",
        help="Keep frames from this MD step on.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="STEP",
        help="Keep frames up to this MD step.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every N-th frame of the selection, default: 1.",
    )
    parser.add_argument(
        "-v", "--version",
        default=default_version(),
        help="ABACUS version hint for the running log.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")


def _report(frames, path: Path, fmt: Optional[str]) -> Dict[str, Any]:
    """Summarise the written trajectory."""
    report = {
        "job": str(path.parent),
        "output": str(path),
        "format": fmt or path.suffix.lstrip("."),
    }
    report.update(summarize_trajectory(frames))
    return report


def run(args: argparse.Namespace) -> int:
    """Run ``abacustools postprocess md``."""
    job = Path(args.job)
    frames = read_trajectory(job, version=args.version)
    selected = select_frames(
        frames, first=args.first, last=args.last, stride=args.stride
    )
    if not selected:
        cli_output("No frame matches the requested step range.")
        return 1

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    write_trajectory(output, selected, fmt=args.format)
    report = _report(selected, output, args.format)

    if args.json:
        cli_output(json.dumps(report, indent=2))
        return 0
    cli_output(f"  job: {job}")
    cli_output(f"  frames: {report['frames']} of {len(frames)} dumped steps")
    cli_output(f"  atoms: {report['atoms']}, steps {report['steps'][0]} to {report['steps'][1]}")
    content = [
        name
        for name, present in (
            ("forces", report["has_forces"]),
            ("velocities", report["has_velocities"]),
            ("virial", report["has_virial"]),
        )
        if present
    ]
    cli_output(f"  contents: {'positions, cell' + (', ' + ', '.join(content) if content else '')}")
    if report["highest_temperature"] is not None:
        cli_output(f"  highest temperature: {report['highest_temperature']:.2f} K")
    cli_output(f"  trajectory: {report['output']}")
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess md``."""
    parser = subparsers.add_parser(
        "md",
        help="Write the trajectory of an ABACUS molecular-dynamics job to a standard format.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
