"""Implementation of the ``abacustools database download`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from abacustools.integrations.databases import structure_path, write_structure

from .common import (
    DATABASE_ERRORS,
    FATAL_DOWNLOAD_ERRORS,
    download_table,
    provider_options,
    selected_database,
)


def register_parser(subparsers, *, default_database=None) -> None:
    """Register the download parser and its arguments.

    Args:
        subparsers: Subparser collection to add the command to.
        default_database: Database used when ``--database`` is omitted.
    """
    parser = subparsers.add_parser(
        "download",
        help="Download structures from a structure database.",
    )
    parser.add_argument(
        "identifiers",
        nargs="+",
        metavar="ID",
        help="Database entry identifiers such as 'mp-149' or 'aflow:000000f0098a6204'.",
    )
    parser.add_argument(
        "-d",
        "--database",
        default=default_database,
        metavar="NAME",
        help="Database to download from, default: mp; see 'abacustools database list'.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        metavar="NAME",
        help="OPTIMADE provider to download from, such as 'cod' or 'aflow'.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        metavar="URL",
        help="Query another OPTIMADE endpoint instead of a catalogued one.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        metavar="DIRECTORY",
        help="Parent directory of the per-entry directories, default: the current directory.",
    )
    parser.add_argument(
        "-f",
        "--format",
        default="stru",
        metavar="FORMAT",
        help="Output structure format: stru, poscar, cif, xyz, extxyz, or xsf.",
    )
    parser.add_argument(
        "--group-by-database",
        action="store_true",
        help="Write to OUTPUT/DATABASE/ID/ instead of OUTPUT/ID/.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        metavar="KEY",
        help="API key; defaults to the environment variable of the database.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Download one or more structures and report where they were written."""
    try:
        database = selected_database(args)
        options = provider_options(args, database)
    except DATABASE_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    for identifier in args.identifiers:
        try:
            structure = database.fetch(identifier, api_key=args.api_key, **options)
        except FATAL_DOWNLOAD_ERRORS as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        except DATABASE_ERRORS + (LookupError, OSError) as error:
            print(f"error: {identifier}: {error}", file=sys.stderr)
            records.append({"database": database.name, "id": identifier, "error": str(error)})
            continue
        destination = structure_path(
            args.output,
            structure.identifier,
            fmt=args.format,
            database=structure.database if args.group_by_database else None,
        )
        write_structure(structure, destination, fmt=args.format)
        record = structure.summary.to_dict()
        record["file"] = str(destination)
        records.append(record)

    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        Console().print(download_table(records))
    return 0 if all("error" not in record for record in records) else 1
