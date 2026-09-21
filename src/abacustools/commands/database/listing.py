"""Implementation of the ``abacustools database list`` command."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from abacustools.integrations.databases import describe_databases

_STATUS_STYLES = {"ready": "green", "needs-api-key": "yellow", "unavailable": "red"}


def register_parser(subparsers) -> None:
    """Register the ``database list`` parser and its arguments."""
    parser = subparsers.add_parser(
        "list",
        help="List the databases abacustools can query.",
    )
    parser.add_argument(
        "--available",
        action="store_true",
        help="Only list databases that are ready to query.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.set_defaults(handler=run)


def _table(records) -> Table:
    """Build the database listing table."""
    table = Table(header_style="bold")
    table.add_column("database", style="cyan", no_wrap=True)
    table.add_column("access", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("selectors")
    table.add_column("description")
    for record in records:
        style = _STATUS_STYLES.get(record["status"], "")
        table.add_row(
            record["name"],
            record["protocol"],
            f"[{style}]{record['status']}[/{style}]" if style else record["status"],
            ", ".join(record["capabilities"]) or "-",
            record["description"],
        )
    return table


def run(args: argparse.Namespace) -> int:
    """List the registered structure databases."""
    records = describe_databases()
    if args.available:
        records = [record for record in records if record["status"] == "ready"]
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    elif records:
        Console().print(_table(records))
    else:
        print("no database is ready to query")
    return 0
