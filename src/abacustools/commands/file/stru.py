"""Implementation of the ``abacustools file stru`` command."""

from __future__ import annotations

import argparse
from pathlib import Path


def register_parser(subparsers) -> None:
    """Register the ``file stru`` parser and its arguments."""
    parser = subparsers.add_parser(
        "stru",
        help="Process an ABACUS STRU file.",
    )
    parser.add_argument("filename", type=Path, metavar="STRU")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Run the ``file stru`` command."""
    # Replace this placeholder with the STRU-specific workflow.
    print(f"process stru file: {args.filename}")
    return 0
