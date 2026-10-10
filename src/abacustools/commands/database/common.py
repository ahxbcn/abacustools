"""Helpers shared by the ``abacustools database`` commands."""

from __future__ import annotations

import argparse
from typing import Any, Optional, Sequence

from rich.table import Table

from abacustools.integrations.databases import (
    DatabaseApiKeyError,
    DatabaseError,
    DatabaseQuery,
    DatabaseRequestError,
    DatabaseSummary,
    DatabaseUnavailableError,
    StructureDatabase,
    default_database,
    get_database,
)

#: Failures a database command reports as a one-line error.
DATABASE_ERRORS = (DatabaseUnavailableError, DatabaseError, ValueError)

#: Failures that stop a batch download before the next identifier.
FATAL_DOWNLOAD_ERRORS = (DatabaseUnavailableError, DatabaseApiKeyError)


def selected_database(args: argparse.Namespace) -> StructureDatabase:
    """Return the database named by ``--database``, or the default one."""
    name = getattr(args, "database", None)
    return get_database(name) if name else default_database()


def query_from_args(args: argparse.Namespace) -> DatabaseQuery:
    """Build a :class:`DatabaseQuery` from parsed command-line arguments."""
    return DatabaseQuery(
        formula=args.formula,
        chemsys=args.chemsys,
        elements=tuple(args.elements) if args.elements else None,
        identifiers=tuple(args.identifiers) if args.identifiers else None,
        is_stable=True if getattr(args, "stable", False) else None,
        theoretical=True if getattr(args, "theoretical", False) else None,
        limit=args.limit,
        fields=tuple(split_fields(args.fields)) if getattr(args, "fields", None) else None,
        where=tuple(getattr(args, "where", None) or ()) or None,
    )


def split_fields(value: Optional[str]) -> list[str]:
    """Split a comma-separated ``--fields`` value.

    Raises:
        DatabaseRequestError: If the value names no field at all.
    """
    fields = [item.strip() for item in (value or "").split(",") if item.strip()]
    if not fields:
        raise DatabaseRequestError("--fields needs at least one field name")
    return fields


def split_columns(value: Optional[Sequence[str]]) -> Optional[list[str]]:
    """Split repeated comma-separated ``--show`` values into column keys."""
    columns = []
    for item in value or ():
        for key in str(item).split(","):
            if key.strip():
                columns.append(key.strip())
    return columns or None


def provider_options(
    args: argparse.Namespace,
    database: StructureDatabase,
) -> dict[str, Any]:
    """Return the provider-specific options of a command, once validated.

    Raises:
        DatabaseRequestError: If the database does not accept them.
    """
    options = {
        "provider": getattr(args, "provider", None),
        "base_url": getattr(args, "base_url", None),
        "show": split_columns(getattr(args, "show", None)),
    }
    options = {name: value for name, value in options.items() if value}
    database.check_options(**options)
    return options


def number(value: Optional[float], digits: int = 3) -> str:
    """Format an optional float for a table cell."""
    return "-" if value is None else f"{value:.{digits}f}"


def flag(value: Optional[bool]) -> str:
    """Format an optional boolean for a table cell."""
    return "-" if value is None else ("yes" if value else "no")


def summary_table(
    summaries: Sequence[DatabaseSummary],
    columns: Sequence[str] = (),
) -> Table:
    """Build the search result table, with extra property columns."""
    table = Table(header_style="bold")
    table.add_column("database", style="magenta", no_wrap=True)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("formula")
    table.add_column("system", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("E_hull", justify="right")
    table.add_column("gap", justify="right")
    for column in columns:
        table.add_column(column, justify="right")
    table.add_column("stable", justify="right")
    table.add_column("theo", justify="right")
    for summary in summaries:
        row = [
            summary.database,
            summary.identifier,
            summary.formula or "-",
            summary.chemsys or "-",
            "-" if summary.nsites is None else str(summary.nsites),
            number(summary.energy_above_hull),
            number(summary.band_gap),
        ]
        for column in columns:
            value = summary.extra.get(column)
            row.append("-" if value in (None, "") else str(value))
        row.extend([flag(summary.is_stable), flag(summary.theoretical)])
        table.add_row(*row)
    return table


def download_table(records: Sequence[dict[str, Any]]) -> Table:
    """Build the download report table."""
    table = Table(header_style="bold")
    table.add_column("database", style="magenta", no_wrap=True)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("formula")
    table.add_column("N", justify="right")
    table.add_column("file")
    for record in records:
        table.add_row(
            str(record.get("database") or "-"),
            str(record["id"]),
            str(record.get("formula") or "-"),
            "-" if record.get("nsites") is None else str(record["nsites"]),
            str(record.get("file") or record.get("error") or "-"),
        )
    return table
