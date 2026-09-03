"""Command-line entry point and subcommand parser for abacustools."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from abacustools import __version__
from abacustools.commands.file import register_parser as register_file_parser
from abacustools.commands.postprocess import (
    register_parser as register_postprocess_parser,
)
from abacustools.commands.workflow import (
    register_parser as register_workflow_parser,
)


def _version_command(args: argparse.Namespace) -> int:
    """Print the installed abacustools version."""
    print(f"{args._prog} {__version__}")
    return 0


def _create_parser(prog: str) -> argparse.ArgumentParser:
    """Create the top-level parser and register built-in subcommands."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Tools for accompanying using ABACUS.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.set_defaults(_prog=prog)

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="subcommands",
    )

    version_parser = subparsers.add_parser(
        "version",
        help="Show the installed abacustools version.",
    )
    version_parser.set_defaults(handler=_version_command)
    register_file_parser(subparsers)
    register_postprocess_parser(subparsers)
    register_workflow_parser(subparsers)

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    prog: str = "abacustools",
) -> int:
    """Run the ``abacustools`` command-line interface."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _create_parser(prog)
    namespace = parser.parse_args(arguments)

    if namespace.command is None:
        parser.print_help()
        return 0

    handler = getattr(namespace, "handler", None)
    if handler is None:
        parser.error(f"no handler registered for command {namespace.command!r}")
    return handler(namespace)


if __name__ == "__main__":
    raise SystemExit(main())
