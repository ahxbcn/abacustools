"""Implementation of the ``abacustools mp search`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console
from rich.table import Table

from abacustools.integrations.materials_project import (
    MaterialSummary,
    MaterialsProjectApiKeyError,
    MaterialsProjectUnavailableError,
    search_materials,
)


def register_parser(subparsers) -> None:
    """Register the ``mp search`` parser and its arguments."""
    parser = subparsers.add_parser(
        "search",
        help="Search the Materials Project for materials.",
    )
    parser.add_argument(
        "--formula",
        default=None,
        metavar="FORMULA",
        help="Composition such as 'Fe2O3'; wildcards such as 'Li*O' are allowed.",
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
        "--material-id",
        dest="material_ids",
        action="append",
        default=None,
        metavar="ID",
        help="Restrict the search to a material id; may be repeated.",
    )
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Only return materials on the convex hull.",
    )
    parser.add_argument(
        "--theoretical",
        action="store_true",
        help="Only return materials without an experimental counterpart.",
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
        help="Comma-separated summary fields to request; defaults to the standard set.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="Materials Project API key; defaults to the MP_API_KEY environment variable.",
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


def _fields(value: Optional[str]) -> Optional[Sequence[str]]:
    """Split a comma-separated ``--fields`` value."""
    if value is None:
        return None
    fields = [item.strip() for item in value.split(",") if item.strip()]
    if not fields:
        raise ValueError("--fields needs at least one field name")
    return fields


def _number(value: Optional[float], digits: int = 3) -> str:
    """Format an optional float for a table cell."""
    return "-" if value is None else f"{value:.{digits}f}"


def _flag(value: Optional[bool]) -> str:
    """Format an optional boolean for a table cell."""
    return "-" if value is None else ("yes" if value else "no")


def _table(summaries: Sequence[MaterialSummary]) -> Table:
    """Build the search result table."""
    table = Table(header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("formula")
    table.add_column("system", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("E_hull", justify="right")
    table.add_column("gap", justify="right")
    table.add_column("stable", justify="right")
    table.add_column("theo", justify="right")
    for summary in summaries:
        table.add_row(
            summary.material_id,
            summary.formula or "-",
            summary.chemsys or "-",
            "-" if summary.nsites is None else str(summary.nsites),
            _number(summary.energy_above_hull),
            _number(summary.band_gap),
            _flag(summary.is_stable),
            _flag(summary.theoretical),
        )
    return table


def run(args: argparse.Namespace) -> int:
    """Search the Materials Project and report the matching materials."""
    try:
        summaries = search_materials(
            formula=args.formula,
            chemsys=args.chemsys,
            elements=args.elements,
            material_ids=args.material_ids,
            is_stable=True if args.stable else None,
            theoretical=True if args.theoretical else None,
            limit=args.limit,
            fields=_fields(args.fields),
            api_key=args.api_key,
        )
    except (
        MaterialsProjectUnavailableError,
        MaterialsProjectApiKeyError,
        ValueError,
    ) as error:
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
        Console().print(_table(summaries))
    else:
        print("no materials matched the query")
    return 0 if summaries else 1
