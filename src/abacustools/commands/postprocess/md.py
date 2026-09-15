"""Implementation of the ``abacustools postprocess md`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from abacustools.data.md import (
    read_trajectory,
    select_frames,
    write_trajectory,
)
from abacustools.data.versions import default_version


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


def _limit(values) -> Optional[float]:
    """Return the largest finite absolute value, or ``None``."""
    import numpy as np

    finite = np.asarray([value for value in values if value is not None], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.max())


def _report(frames, path: Path, fmt: Optional[str]) -> Dict[str, Any]:
    """Summarise the written trajectory."""
    return {
        "job": str(path.parent),
        "output": str(path),
        "format": fmt or path.suffix.lstrip("."),
        "frames": len(frames),
        "atoms": frames[0].natoms if frames else 0,
        "steps": [frames[0].step, frames[-1].step] if frames else [],
        "has_forces": any(frame.forces is not None for frame in frames),
        "has_velocities": any(frame.velocities is not None for frame in frames),
        "has_virial": any(frame.virial is not None for frame in frames),
        "highest_temperature": _limit(frame.temperature for frame in frames),
    }


def run(args: argparse.Namespace) -> int:
    """Run ``abacustools postprocess md``."""
    job = Path(args.job)
    frames = read_trajectory(job, version=args.version)
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
    print(f"  frames: {report['frames']} of {len(frames)} dumped steps")
    print(f"  atoms: {report['atoms']}, steps {report['steps'][0]} to {report['steps'][1]}")
    content = [
        name
        for name, present in (
            ("forces", report["has_forces"]),
            ("velocities", report["has_velocities"]),
            ("virial", report["has_virial"]),
        )
        if present
    ]
    print(f"  contents: {'positions, cell' + (', ' + ', '.join(content) if content else '')}")
    if report["highest_temperature"] is not None:
        print(f"  highest temperature: {report['highest_temperature']:.2f} K")
    print(f"  trajectory: {report['output']}")
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess md``."""
    parser = subparsers.add_parser(
        "md",
        help="Write the trajectory of an ABACUS molecular-dynamics job to a standard format.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
