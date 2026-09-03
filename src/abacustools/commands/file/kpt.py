"""Implementation of the ``abacustools file kpt`` command."""

from __future__ import annotations

import argparse
from pathlib import Path


def register_parser(subparsers) -> None:
    """Register the ``file kpt`` parser and its arguments."""
    parser = subparsers.add_parser(
        "kpt",
        help="Process an ABACUS KPT file.",
    )
    parser.add_argument("filename", type=Path, metavar="KPT")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Run the ``file kpt`` command."""
    # Replace this placeholder with the KPT-specific workflow.
    print(f"process kpt file: {args.filename}")
    return 0
