"""The ``abacustools database`` command family."""

from . import download as download_command
from . import fields as fields_command
from . import listing as listing_command
from . import providers as providers_command
from . import search as search_command


def register_parser(subparsers) -> None:
    """Register ``database`` and its nested subcommands."""
    database_parser = subparsers.add_parser(
        "database",
        aliases=["db"],
        help="Search and download structures from many databases.",
    )
    database_subparsers = database_parser.add_subparsers(
        dest="database_command",
        metavar="DATABASE_COMMAND",
        title="database subcommands",
        required=True,
    )

    listing_command.register_parser(database_subparsers)
    fields_command.register_parser(database_subparsers)
    providers_command.register_parser(database_subparsers)
    search_command.register_parser(database_subparsers)
    download_command.register_parser(database_subparsers)
