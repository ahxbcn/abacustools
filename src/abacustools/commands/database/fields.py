"""Implementation of the ``abacustools database fields`` command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from .common import DATABASE_ERRORS, selected_database


def register_parser(subparsers) -> None:
    """Register the ``database fields`` parser and its arguments."""
    parser = subparsers.add_parser(
        "fields",
        aliases=["keys"],
        help="List the property keys a database documents.",
    )
    parser.add_argument(
        "-d",
        "--database",
        default=None,
        metavar="NAME",
        help="Database to describe, default: mp; see 'abacustools database list'.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """List the property keys of a database."""
    try:
        database = selected_database(args)
        fields = database.fields()
    except DATABASE_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    records = [{"key": key, "description": description} for key, description in fields]
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    if not records:
        print(f"database {database.name!r} does not document property keys")
        return 0
    table = Table(header_style="bold", title=f"{database.name}: {database.description}")
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("description")
    for record in records:
        table.add_row(record["key"], record["description"])
    Console().print(table)
    print(f"{len(records)} keys; filter on them with 'database search -d {database.name} --where KEY OP VALUE'")
    return 0
