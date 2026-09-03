"""Command line interface for abacustools."""

import argparse

from abacustools import __version__


def main() -> None:
    """Entry point of the abacustools command line interface."""
    parser = argparse.ArgumentParser(prog="abacustools", description="Tools for accompanying using ABACUS")
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
    main()