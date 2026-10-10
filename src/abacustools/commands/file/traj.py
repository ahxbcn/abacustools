"""Implementation of the ``abacustools file traj`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the arguments of the trajectory conversion command."""
    parser.add_argument("input", type=Path, metavar="INPUT", help="Trajectory file to read.")
    parser.add_argument("output", type=Path, metavar="OUTPUT", help="Trajectory file to write.")
    parser.add_argument(
        "--input-format",
        default=None,
        help="ASE input format; inferred from the file name by default.",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help="ASE output format; inferred from the file name by default.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every N-th frame, default: 1.",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")


def run(args: argparse.Namespace) -> int:
    """Run ``abacustools file traj``."""
    from ase.io import read, write

    if not Path(args.input).is_file():
        print(f"Trajectory conversion failed: {args.input} does not exist.")
        return 1
    if isinstance(args.stride, bool) or args.stride < 1:
        print("Trajectory conversion failed: --stride must be a positive integer.")
        return 1

    frames = read(str(args.input), index=":", format=args.input_format)
    if not frames:
        print(f"Trajectory conversion failed: no frame in {args.input}.")
        return 1
    selected = frames[:: args.stride]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(str(output), selected, format=args.output_format)

    report = {
        "input": str(args.input),
        "output": str(output),
        "frames": len(selected),
        "read_frames": len(frames),
        "input_format": args.input_format or Path(args.input).suffix.lstrip("."),
        "output_format": args.output_format or output.suffix.lstrip("."),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"  frames: {report['frames']} of {report['read_frames']}")
    print(f"  formats: {report['input_format']} -> {report['output_format']}")
    print(f"  trajectory: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools file traj``."""
    parser = subparsers.add_parser(
        "traj",
        help="Convert a molecular-dynamics trajectory between file formats.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
