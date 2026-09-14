"""The ``abacustools mp`` command family."""

from . import download as download_command
from . import search as search_command


def register_parser(subparsers) -> None:
    """Register ``mp`` and its nested subcommands."""
    mp_parser = subparsers.add_parser(
        "mp",
        help="Search and download structures from the Materials Project.",
    )
    mp_subparsers = mp_parser.add_subparsers(
        dest="mp_command",
        metavar="MP_COMMAND",
        title="mp subcommands",
        required=True,
    )

    search_command.register_parser(mp_subparsers)
    download_command.register_parser(mp_subparsers)
