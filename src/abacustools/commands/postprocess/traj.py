"""Implementation of the ``abacustools postprocess traj`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from abacustools.data.md import (
    read_relax_trajectory,
    select_frames,
    summarize_trajectory,
    write_trajectory,
)
from abacustools.data.versions import default_version


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the arguments of the relaxation trajectory command."""
    parser.add_argument(
        "-j", "--job",
        required=True,
        type=_job_directory,
        help="ABACUS job directory of a relax or cell-relax calculation.",
    )
    parser.add_argument(
        "-o", "--output",
        default="relax_trajectory.extxyz",
        help=(
            "Trajectory file to write, relative to JOB by default; the suffix "
            "selects the format, default: relax_trajectory.extxyz."
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
        help="Keep frames from this relaxation step on.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="STEP",
        help="Keep frames up to this relaxation step.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every N-th frame of the selection, default: 1.",
    )
    parser.add_argument(
        "--no-energy",
        action="store_true",
        help="Do not read the energy of every step from the running log.",
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
    """Run ``abacustools postprocess traj``."""
    job = Path(args.job)
    try:
        frames = read_relax_trajectory(
            job, version=args.version, with_log=not args.no_energy
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"Relaxation trajectory failed: {error}")
        return 1
    selected = select_frames(
        frames, first=args.first, last=args.last, stride=args.stride
    )
    if not selected:
        print("No frame matches the requested step range.")
        return 1

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    write_trajectory(output, selected, fmt=args.format)
    report = _report(selected, output, args.format)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"  job: {job}")
    print(f"  frames: {report['frames']} of {len(frames)} relaxation steps")
    print(f"  atoms: {report['atoms']}, steps {report['steps'][0]} to {report['steps'][1]}")
    content = [
        name
        for name, present in (
            ("forces", report["has_forces"]),
            ("stress", report["has_stress"]),
            ("velocities", report["has_velocities"]),
            ("energies", report["has_energy"]),
        )
        if present
    ]
    print(f"  contents: {'positions, cell' + (', ' + ', '.join(content) if content else '')}")
    print(f"  trajectory: {report['output']}")
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess traj``."""
    parser = subparsers.add_parser(
        "traj",
        help="Write the trajectory of a relax or cell-relax ABACUS job to a standard format.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
