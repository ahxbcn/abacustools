"""Implementation of the ``abacustools postprocess result`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from abacustools.data.abacus_result import (
    get_result_from_job,
    grouped_params,
)
from abacustools.data.versions import default_version


RESULT_PARAMETERS = tuple(
    parameter
    for parameters in grouped_params.values()
    for parameter in parameters
)
_SUMMARY_OMIT_PARAMETERS = {"force", "stress"}
_HIGH_PRECISION_PARAMETERS = {"energy"}


def _job_directory(value: str) -> Path:
    """Return an existing ABACUS job directory or raise a parser error."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _is_scalar(value) -> bool:
    """Return whether a result can be shown compactly in a table cell."""
    return not isinstance(value, (list, tuple, dict))


def _format_value(parameter: str, value) -> str:
    """Format a scalar result value for a single table cell."""
    if value is None:
        return "-"
    if isinstance(value, float):
        value_text = str(value)
        if parameter in _HIGH_PRECISION_PARAMETERS and "e" not in value_text.lower():
            decimal_places = len(value_text.partition(".")[2])
            if decimal_places < 8:
                return f"{value:.8f}"
        if parameter not in _HIGH_PRECISION_PARAMETERS:
            return f"{value:.8g}"
        return value_text
    return str(value)


def _job_labels(results: dict[str, dict]) -> dict[str, str]:
    """Use short, unique job labels for the result tables."""
    labels = {job: Path(job).name or job for job in results}
    duplicates = {label for label in labels.values() if list(labels.values()).count(label) > 1}
    for job, label in labels.items():
        if label in duplicates:
            labels[job] = f"{Path(job).parent.name}/{label}"

    if len(set(labels.values())) != len(labels):
        for index, job in enumerate(labels, start=1):
            if list(labels.values()).count(labels[job]) > 1:
                labels[job] = f"{labels[job]}#{index}"
    return labels


def _table_groups(
    headers: list[str], rows: list[list[str]], max_width: int
) -> list[list[str]]:
    """Split columns into terminal-sized groups, retaining the job column."""
    widths = {header: len(header) for header in headers}
    for row in rows:
        for index, header in enumerate(headers):
            widths[header] = max(widths[header], len(row[index]))
    job_width = widths["job"] + 2
    groups: list[list[str]] = []
    group = ["job"]
    group_width = job_width
    for header in headers[1:]:
        column_width = widths[header] + 2
        if len(group) > 1 and group_width + column_width > max_width:
            groups.append(group)
            group = ["job"]
            group_width = job_width
        group.append(header)
        group_width += column_width
    groups.append(group)
    return groups


def _format_tables(results: dict[str, dict], max_width: int) -> list[Table]:
    """Build borderless Rich tables split horizontally to fit the terminal."""
    headers = ["job"]
    for result in results.values():
        for parameter in result:
            if (
                parameter not in headers
                and parameter not in _SUMMARY_OMIT_PARAMETERS
                and all(_is_scalar(item.get(parameter)) for item in results.values())
            ):
                headers.append(parameter)

    labels = _job_labels(results)
    rows = [
        [labels[job]]
        + [_format_value(header, result.get(header)) for header in headers[1:]]
        for job, result in results.items()
    ]
    header_groups = _table_groups(headers, rows, max_width)
    tables = []
    for group in header_groups:
        table = Table(
            box=None,
            padding=(0, 1),
            pad_edge=False,
            expand=False,
        )
        for header in group:
            table.add_column(
                header,
                justify="right",
                no_wrap=True,
                overflow="ignore",
            )
        indexes = [headers.index(header) for header in group]
        for row in rows:
            table.add_row(*(row[index] for index in indexes))
        tables.append(table)
    return tables


def register_parser(subparsers) -> None:
    """Register the ``postprocess result`` parser and its arguments."""
    parser = subparsers.add_parser(
        "result",
        help="Process ABACUS result files.",
    )
    parser.add_argument(
        "-j",
        "--job",
        required=True,
        action="extend",
        nargs="+",
        type=_job_directory,
        help="Directories of ABACUS jobs from which to collect results.",
    )
    parser.add_argument(
        "-p",
        "--param",
        default=None,
        action="extend",
        nargs="+",
        type=str.lower,
        choices=RESULT_PARAMETERS,
        metavar="PARAM",
        help="Names of results to collect. Defaults to results for the job type.",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=default_version(),
        help="Version of ABACUS used in the jobs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the collected results as JSON.",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Run the ``postprocess result`` command."""
    results = {}
    for job in args.job:
        results[str(job)] = get_result_from_job(job, args.param, args.version)

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    console = Console()
    for index, table in enumerate(_format_tables(results, console.width)):
        if index:
            console.print()
        console.print(table)
    return 0
