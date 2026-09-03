"""Implementation of the ``abacustools file input`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from abacustools.io.abacus import ReadInput, WriteInput


def _input_file(value: str) -> Path:
    """Return an existing INPUT file path or raise a parser error."""
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"INPUT file does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the ``file input`` parser and its arguments."""
    parser = subparsers.add_parser(
        "input",
        help="Process an ABACUS INPUT file.",
    )
    parser.add_argument("filename", type=_input_file, metavar="INPUT")
    parser.add_argument(
        "--set",
        default=None,
        nargs=2,
        metavar=("PARAMETER", "VALUE"),
        help="Set a parameter in the INPUT file.",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Run the ``file input`` command."""
    print(f"process input file: {args.filename}")
    inputs = ReadInput(args.filename)
    if args.set:
        param, value = args.set
        param = param.lower()
        if param in inputs:
            print(f"Modifying {param} from {inputs[param]} to {value}")
        else:
            print(f"Setting new parameter {param} to {value}")
        inputs[param] = value

    WriteInput(inputs, args.filename)
    return 0
