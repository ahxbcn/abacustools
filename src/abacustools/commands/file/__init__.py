"""The ``abacustools file`` command family."""

from . import input as input_command
from . import kpt as kpt_command
from . import stru as stru_command
from . import structure_info as structure_info_command


def register_parser(subparsers) -> None:
    """Register ``file`` and its nested subcommands."""
    file_parser = subparsers.add_parser(
        "file",
        help="Process ABACUS files.",
    )
    file_subparsers = file_parser.add_subparsers(
        dest="file_command",
        metavar="FILE_COMMAND",
        title="file subcommands",
        required=True,
    )

    input_command.register_parser(file_subparsers)
    stru_command.register_parser(file_subparsers)
    structure_info_command.register_parser(file_subparsers)
    kpt_command.register_parser(file_subparsers)
