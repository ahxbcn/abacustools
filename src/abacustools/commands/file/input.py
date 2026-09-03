"""Implementation of the ``abacustools file input`` command."""

from __future__ import annotations

import argparse
from pathlib import Path


def register_parser(subparsers) -> None:
    """Register the ``file input`` parser and its arguments."""
    parser = subparsers.add_parser(
        "input",
        help="Process an ABACUS INPUT file.",
    )
    parser.add_argument("filename", type=Path, metavar="INPUT")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the INPUT file.",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Run the ``file input`` command."""
    # Replace this placeholder with the INPUT-specific workflow.
    print(f"process input file: {args.filename}")
    return 0
