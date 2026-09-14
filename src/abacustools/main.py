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
from abacustools.commands.job import register_parser as register_job_parser
from abacustools.commands.mp import register_parser as register_mp_parser
from abacustools.commands.workflow import (
    register_parser as register_workflow_parser,
)
from abacustools.menu.runner import run_menu


_BANNER = "\n".join(
    [
        "        _                                  _                    _",
        "  __ _ | |__    __ _   ___   -   -   ___  | |_     ___    ___  | |  ___",
        " / _` || '_ \\  / _` | / __| | | | | / __| | __|   / _ \\  / _ \\ | | / __|",
        "| (_| || |_) || (_| || (__  | |_| | \\__ \\ | |_   | (_) || (_) || | \\__ \\",
        " \\__,_||_.__/  \\__,_| \\___|  \\__,_| |___/  \\__|   \\___/  \\___/ |_| |___/",
    ]
)


def _print_banner() -> None:
    """Print the ASCII-art banner with the program name and version."""
    print()
    print(_BANNER)
    print()
    print(f"  Tools for accompanying using ABACUS    version {__version__}")
    print()


def _is_interactive() -> bool:
    """Return True when both stdin and stdout are attached to a terminal."""
    stdin = getattr(sys.stdin, "isatty", None)
    stdout = getattr(sys.stdout, "isatty", None)
    return bool(stdin and stdin()) and bool(stdout and stdout())


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

    menu_parser = subparsers.add_parser(
        "menu",
        help="Launch the interactive menu.",
    )

    def _menu_handler(namespace: argparse.Namespace) -> int:
        return run_menu(parser, prog=getattr(namespace, "_prog", "abacustools"))

    menu_parser.set_defaults(handler=_menu_handler, _menu_exclude=True)

    register_file_parser(subparsers)
    register_job_parser(subparsers)
    register_mp_parser(subparsers)
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

    _print_banner()

    namespace = parser.parse_args(arguments)

    if namespace.command is None:
        if _is_interactive():
            return run_menu(parser, prog=prog)
        parser.print_help()
        return 0

    handler = getattr(namespace, "handler", None)
    if handler is None:
        parser.error(f"no handler registered for command {namespace.command!r}")
    return handler(namespace)


if __name__ == "__main__":
    raise SystemExit(main())
