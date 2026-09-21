"""Implementation of the ``abacustools database providers`` command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from abacustools.integrations.databases import (
    load_live_providers,
    optimade_catalogue,
    optimade_catalogue_source,
)

from .common import DATABASE_ERRORS


def register_parser(subparsers) -> None:
    """Register the ``database providers`` parser and its arguments."""
    parser = subparsers.add_parser(
        "providers",
        help="List the OPTIMADE providers behind the 'optimade' database.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Read the live OPTIMADE index instead of the bundled catalogue.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.set_defaults(handler=run)


def _bundled_records():
    """Return one record per catalogued OPTIMADE provider."""
    return [
        {
            "name": provider.name,
            "endpoint": provider.base_url,
            "homepage": provider.homepage,
            "description": provider.description,
            "note": provider.note,
            "requires_api_key": provider.requires_api_key,
        }
        for provider in optimade_catalogue()
    ]


def _refresh_records():
    """Return one record per provider published by the live index."""
    return [
        {
            "name": record["name"],
            "endpoint": record["base_url"] or "-",
            "homepage": record["homepage"] or "-",
            "description": record["description"] or "",
            "note": "index meta-database",
            "requires_api_key": False,
        }
        for record in load_live_providers()
    ]


def _table(records) -> Table:
    """Build the OPTIMADE provider table."""
    table = Table(header_style="bold")
    table.add_column("database", style="cyan", no_wrap=True)
    table.add_column("endpoint")
    table.add_column("key", justify="right", no_wrap=True)
    table.add_column("description")
    for record in records:
        description = record["description"]
        if record["note"]:
            description = f"{description} ({record['note']})"
        table.add_row(
            record["name"],
            record["endpoint"],
            "yes" if record["requires_api_key"] else "no",
            escape(description),
        )
    return table


def run(args: argparse.Namespace) -> int:
    """List the OPTIMADE providers, from the bundle or from the live index."""
    try:
        if args.refresh:
            records = _refresh_records()
        else:
            records = _bundled_records()
    except DATABASE_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    source = optimade_catalogue_source()
    Console().print(_table(records))
    if args.refresh:
        print("live index: https://providers.optimade.org/providers.json")
        print("index entries are meta-databases; use --base-url for their implementations")
    else:
        print(f"catalogue: {source['source']} (retrieved {source['retrieved']})")
    print("query one with 'abacustools database search -d NAME ...'")
    return 0
