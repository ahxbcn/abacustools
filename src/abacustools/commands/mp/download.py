"""Implementation of the ``abacustools mp download`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from rich.console import Console
from rich.table import Table

from abacustools.integrations.materials_project import (
    STRUCTURE_FILENAMES,
    MaterialStructure,
    MaterialsProjectApiKeyError,
    MaterialsProjectUnavailableError,
    download_material,
    material_directory,
    write_material_structure,
)


def register_parser(subparsers) -> None:
    """Register the ``mp download`` parser and its arguments."""
    parser = subparsers.add_parser(
        "download",
        help="Download Materials Project structures as ABACUS or common files.",
    )
    parser.add_argument(
        "material_ids",
        nargs="+",
        metavar="MATERIAL_ID",
        help="Materials Project identifiers such as 'mp-149'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        metavar="DIRECTORY",
        help="Parent directory of the per-material directories, default: the current directory.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=sorted(STRUCTURE_FILENAMES),
        default="stru",
        help="Output structure format, default: stru.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="Materials Project API key; defaults to the MP_API_KEY environment variable.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.set_defaults(handler=run)


def _number(value: Optional[float], digits: int = 3) -> str:
    """Format an optional float for a table cell."""
    return "-" if value is None else f"{value:.{digits}f}"


def _table(records: Sequence[dict[str, Any]]) -> Table:
    """Build the download report table."""
    table = Table(header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("formula")
    table.add_column("N", justify="right")
    table.add_column("E_hull", justify="right")
    table.add_column("gap", justify="right")
    table.add_column("file")
    for record in records:
        table.add_row(
            str(record["material_id"]),
            str(record.get("formula") or "-"),
            "-" if record.get("nsites") is None else str(record["nsites"]),
            _number(record.get("energy_above_hull")),
            _number(record.get("band_gap")),
            str(record.get("file") or record.get("error") or "-"),
        )
    return table


def _download_one(material_id: str, args: argparse.Namespace) -> MaterialStructure:
    """Download one structure and write it to its destination."""
    material = download_material(material_id, api_key=args.api_key)
    destination = material_directory(args.output, material.material_id, fmt=args.format)
    write_material_structure(material, destination, fmt=args.format)
    return material


def run(args: argparse.Namespace) -> int:
    """Download one or more Materials Project structures."""
    records: list[dict[str, Any]] = []
    for material_id in args.material_ids:
        try:
            material = _download_one(material_id, args)
        except (MaterialsProjectUnavailableError, MaterialsProjectApiKeyError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        except (LookupError, OSError, ValueError) as error:
            print(f"error: {material_id}: {error}", file=sys.stderr)
            records.append({"material_id": material_id, "error": str(error)})
            continue
        record = material.summary.to_dict()
        record["file"] = str(material_directory(args.output, material.material_id, fmt=args.format))
        records.append(record)

    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        Console().print(_table(records))
    return 0 if all("error" not in record for record in records) else 1
