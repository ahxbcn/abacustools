"""The ``abacustools mp`` command family.

The Materials Project is one of the databases of the ``database`` family;
this family is the short spelling of ``database --database mp`` that keeps the
original command names working.
"""

from __future__ import annotations

from abacustools.commands.database import download as download_command
from abacustools.commands.database import search as search_command


def register_parser(subparsers) -> None:
    """Register ``mp`` and its nested subcommands."""
    mp_parser = subparsers.add_parser(
        "mp",
        help="Search and download Materials Project structures.",
    )
    mp_subparsers = mp_parser.add_subparsers(
        dest="mp_command",
        metavar="MP_COMMAND",
        title="mp subcommands",
        required=True,
    )

    search_command.register_parser(mp_subparsers, default_database="mp")
    download_command.register_parser(mp_subparsers, default_database="mp")
