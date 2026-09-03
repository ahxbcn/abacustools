"""Command-line entry point and subcommand dispatcher for abacustools.

Subcommands follow the convention used by ``conda``: an executable named
``abacustools-<command>`` on ``PATH`` is invoked as
``abacustools <command>``. This keeps optional tools independent from the
top-level package and lets third-party packages add commands through normal
console-script entry points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Optional, Sequence

from abacustools import __version__


COMMAND_PREFIX = "abacustools-"
_COMMAND_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def find_commands(path: Optional[str] = None) -> list[str]:
    """Return the names of executable ``abacustools-*`` commands on ``PATH``.

    Args:
        path: Search path in the same format as the ``PATH`` environment
            variable. Omitting it searches the current process ``PATH``.

    The result is sorted and de-duplicated so it can be used directly in help
    output. A command occurring in more than one directory remains listed
    only once, just as only the first one would be run by :func:`shutil.which`.
    """
    search_path = os.environ.get("PATH", os.defpath) if path is None else path
    commands: set[str] = set()

    for directory in search_path.split(os.pathsep):
        # An empty PATH entry represents the current working directory.
        command_directory = Path(directory or os.curdir)
        try:
            entries = command_directory.iterdir()
            for entry in entries:
                if (
                    entry.name.startswith(COMMAND_PREFIX)
                    and entry.name != COMMAND_PREFIX
                    and _COMMAND_NAME_PATTERN.fullmatch(
                        entry.name[len(COMMAND_PREFIX) :]
                    )
                    and entry.is_file()
                    and os.access(entry, os.X_OK)
                ):
                    commands.add(entry.name[len(COMMAND_PREFIX) :])
        except OSError:
            # A stale or inaccessible PATH entry should not make --help fail.
            continue

    return sorted(commands)


def find_executable(command: str, path: Optional[str] = None) -> Optional[str]:
    """Find the executable implementing an ``abacustools`` subcommand."""
    if not _COMMAND_NAME_PATTERN.fullmatch(command):
        return None
    return shutil.which(f"{COMMAND_PREFIX}{command}", path=path)


def _create_parser(prog: str, commands: Sequence[str]) -> argparse.ArgumentParser:
    """Create the top-level parser, including dynamically discovered commands."""
    command_list = "\n".join(f"  {command}" for command in commands) or "  (none found)"
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Tools for accompanying using ABACUS.",
        epilog=(
            "Built-in subcommands:\n"
            "  commands  List discovered external subcommands.\n"
            "  help      Show top-level or subcommand help.\n"
            "  version   Show the installed abacustools version.\n\n"
            "Available external subcommands:\n"
            f"{command_list}\n\n"
            "Install an executable named 'abacustools-<command>' to make it "
            "available as 'abacustools <command>'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("command", nargs="?", metavar="COMMAND", help="subcommand to run")
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        metavar="ARGUMENT",
        help="arguments passed unchanged to the subcommand",
    )
    return parser


def _run_subcommand(executable: str, arguments: Sequence[str]) -> int:
    """Run a subcommand and return its exit status."""
    completed_process = subprocess.run([executable, *arguments], check=False)
    return completed_process.returncode


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    prog: str = "abacustools",
    path: Optional[str] = None,
) -> int:
    """Run the ``abacustools`` command-line interface.

    ``argv`` and ``path`` are optional primarily to make embedding and testing
    possible. Normal console-script use obtains both from the process.
    """
    from abacustools.core.config import CONFIG
    
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = find_commands(path)
    parser = _create_parser(prog, commands)
    namespace = parser.parse_args(arguments)

    if namespace.command is None:
        parser.print_help()
        return 0

    command = namespace.command
    if command == "help":
        if not namespace.arguments:
            parser.print_help()
            return 0
        command, *command_arguments = namespace.arguments
        if command == "version":
            print(f"{prog} {__version__}")
            return 0
        executable = find_executable(command, path)
        if executable is None:
            parser.error(f"unrecognized command {command!r}")
        return _run_subcommand(executable, ["--help", *command_arguments])

    if command == "version":
        if namespace.arguments:
            parser.error("the 'version' command does not accept arguments")
        print(f"{prog} {__version__}")
        return 0

    if command == "commands":
        if namespace.arguments:
            parser.error("the 'commands' command does not accept arguments")
        print("\n".join(commands))
        return 0

    executable = find_executable(command, path)
    if executable is None:
        parser.error(f"unrecognized command {command!r}")
    return _run_subcommand(executable, namespace.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
