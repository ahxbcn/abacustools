"""Implementation of the ``abacustools database search`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from .common import (
    DATABASE_ERRORS,
    provider_options,
    query_from_args,
    selected_database,
    split_columns,
    summary_table,
)


def register_parser(subparsers, *, default_database=None) -> None:
    """Register the search parser and its arguments.

    Args:
        subparsers: Subparser collection to add the command to.
        default_database: Database used when ``--database`` is omitted.
    """
    parser = subparsers.add_parser(
        "search",
        help="Search a structure database.",
    )
    parser.add_argument(
        "-d",
        "--database",
        default=default_database,
        metavar="NAME",
        help="Database to search, default: mp; see 'abacustools database list'.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        metavar="NAME",
        help="OPTIMADE provider to search, such as 'cod' or 'aflow'.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        metavar="URL",
        help="Query another OPTIMADE endpoint instead of a catalogued one.",
    )
    parser.add_argument(
        "--formula",
        default=None,
        metavar="FORMULA",
        help="Composition such as 'Fe2O3'.",
    )
    parser.add_argument(
        "--chemsys",
        default=None,
        metavar="CHEMSYS",
        help="Chemical system such as 'Li-Fe-O'.",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        metavar="ELEMENT",
        help="Elements that must all be present, such as '--elements Li O'.",
    )
    parser.add_argument(
        "--id",
        "--material-id",
        dest="identifiers",
        action="append",
        default=None,
        metavar="ID",
        help="Restrict the search to an entry id; may be repeated.",
    )
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Only return entries on the convex hull, where supported.",
    )
    parser.add_argument(
        "--theoretical",
        action="store_true",
        help="Only return entries without an experimental counterpart.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of entries to request, default: 20.",
    )
    parser.add_argument(
        "--fields",
        default=None,
        metavar="FIELDS",
        help="Comma-separated summary fields to request, Materials Project only.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=None,
        metavar="EXPRESSION",
        help="Filter term in the language of the database, such as 'gap>1.5'; "
        "may be repeated (C2DB only).",
    )
    parser.add_argument(
        "--show",
        action="append",
        default=None,
        metavar="KEY",
        help="Extra property column to show, such as 'gap_hse' or 'magstate'; "
        "may be repeated (C2DB only).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        metavar="KEY",
        help="API key; defaults to the environment variable of the database.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write the results to FILE as JSON.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Search a structure database and report the matching entries."""
    try:
        database = selected_database(args)
        query = query_from_args(args)
        database.check_query(query)
        query.require_selector()
        options = provider_options(args, database)
        summaries = database.search(query, api_key=args.api_key, **options)
    except DATABASE_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    records = [summary.to_dict() for summary in summaries]
    if args.output is not None:
        args.output.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    elif summaries:
        Console().print(summary_table(summaries, columns=split_columns(args.show) or ()))
    else:
        print("no entries matched the query")
    return 0 if summaries else 1
