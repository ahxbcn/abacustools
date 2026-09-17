"""Implementation of the ``abacustools postprocess ddec`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from abacustools.core.constant import ANG_TO_BOHR
from abacustools.data.ddec import (
    CHARGE_TYPES,
    DdecAnalysis,
    DdecError,
    analyze_ddec,
    parse_pairs,
    read_pairs_file,
    select_pairs,
)


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _output_path(job: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job / path


def _boolean(value: str) -> bool:
    """Read ``.true.``/``.false.`` style flags of the Chargemol input files."""
    text = str(value).strip().strip(".").lower()
    if text in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if text in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a true/false flag, got {value!r}")


def _core_electron_pair(value: str) -> tuple[int, int]:
    """Read a ``--core-electrons "Z NCORE"`` argument."""
    fields = str(value).replace(":", " ").split()
    if len(fields) != 2:
        raise argparse.ArgumentTypeError(
            f"expected an atomic number and a core electron count, got {value!r}"
        )
    try:
        return int(fields[0]), int(fields[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"cannot read {value!r}: {error}") from error


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j", "--job", required=True, type=_job_directory, help="ABACUS job directory."
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write a JSON report, relative to JOB by default.",
    )
    parser.add_argument(
        "--chargemol-exe", default=None,
        help="Chargemol executable (default: CHARGEMOL_EXE env var or config).",
    )
    parser.add_argument(
        "--charge-type", default="DDEC6", choices=CHARGE_TYPES,
        help="Charge partitioning to run (default: DDEC6).",
    )
    parser.add_argument(
        "--atomic-densities", default=None,
        help="atomic_densities directory of the Chargemol distribution.",
    )
    parser.add_argument(
        "--core-electrons", action="append", type=_core_electron_pair, default=None,
        metavar='"Z NCORE"', help="Override the core electron count of one element.",
    )
    parser.add_argument(
        "--net-charge", type=float, default=None,
        help="Net charge of the unit cell (default: nelec of INPUT, else 0).",
    )
    parser.add_argument(
        "--periodicity", type=_boolean, nargs=3, default=(True, True, True),
        metavar=("A", "B", "C"),
        help="Periodicity along the lattice vectors (default: true true true).",
    )
    spin_group = parser.add_mutually_exclusive_group()
    spin_group.add_argument(
        "--spin", dest="spin", action="store_true", default=None,
        help="Write the spin density of an nspin 2 job so that Chargemol reports "
             "atomic spin moments (default for nspin 2).",
    )
    spin_group.add_argument(
        "--no-spin", dest="spin", action="store_false",
        help="Do not write the spin density.",
    )
    parser.add_argument(
        "--no-bos", dest="bos", action="store_false",
        help="Do not compute bond orders and overlap populations.",
    )
    parser.add_argument(
        "--cube", default=None,
        help="Explicit charge-density cube file or directory, relative to JOB.",
    )
    parser.add_argument(
        "--grid", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=None,
        help="FFT grid for restart input when it cannot be read from the log.",
    )
    parser.add_argument(
        "--lat0", type=float, default=ANG_TO_BOHR,
        help="ABACUS LATTICE_CONSTANT in Bohr (default: 1.889726).",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Value of OMP_NUM_THREADS for the OpenMP binary.",
    )
    parser.add_argument(
        "--workdir", default=None,
        help="Keep the cubes and the Chargemol output in this directory.",
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep the temporary working directory."
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "-c", "--cutoff", type=float,
        help="Only print bonds within this distance in Angstrom.",
    )
    selector.add_argument(
        "-p", "--pairs", help="One-based atom pairs, for example 1-2,1-3."
    )
    selector.add_argument(
        "--pairs-file", help="File containing one-based atom pairs, one pair per line."
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.2,
        help="Only print bond orders >= threshold (default: 0.2).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the complete JSON report to stdout."
    )


def _print_table(analysis: DdecAnalysis, pairs, threshold: float) -> None:
    console = Console()
    title = (
        f"DDEC charges: {Path(analysis.job).name} "
        f"({analysis.charge_type}, nspin={analysis.nspin})"
    )
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("Element")
    table.add_column("Net charge (e)", justify="right")
    if analysis.nspin == 2:
        table.add_column("Spin moment (uB)", justify="right")
    table.add_column("Sum of BOs", justify="right")
    for atom in analysis.atoms:
        row = [str(atom.index), atom.element, f"{atom.net_charge:+.4f}"]
        if analysis.nspin == 2:
            row.append("-" if atom.spin_moment is None else f"{atom.spin_moment:+.4f}")
        row.append("-" if atom.bond_order_sum is None else f"{atom.bond_order_sum:.4f}")
        table.add_row(*row)
    console.print(table)

    if pairs:
        bonds = Table(title=f"Bond orders with |BO| >= {threshold:.3f}")
        bonds.add_column("Atom 1", justify="right")
        bonds.add_column("Atom 2", justify="right")
        bonds.add_column("Image")
        bonds.add_column("BO", justify="right")
        bonds.add_column("Spin polarization", justify="right")
        bonds.add_column("Distance (A)", justify="right")
        for pair in pairs:
            distance = "-" if pair.distance is None else f"{pair.distance:.4f}"
            image = "{} {} {}".format(*pair.image)
            bonds.add_row(
                f"{pair.atom1} {pair.element1}",
                f"{pair.atom2} {pair.element2}",
                image,
                f"{pair.bond_order:.4f}",
                f"{pair.spin_polarization:.4f}",
                distance,
            )
        console.print(bonds)

    accounting = analysis.electron_accounting
    electrons = accounting.get("integrated_valence")
    valence = accounting.get("nvalence")
    detail = ""
    if electrons is not None and valence is not None:
        detail = f", {electrons:.4f} of {valence:.4f} valence electrons"
    console.print(
        f"Net charge {analysis.total_net_charge:+.4f} e{detail}; "
        f"{len(analysis.pairs)} printed bonds; work directory {analysis.workdir}"
    )
    for warning in analysis.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def run(args: argparse.Namespace) -> int:
    job = Path(args.job)
    try:
        selection = parse_pairs(args.pairs)
        if args.pairs_file:
            selection = read_pairs_file(_output_path(job, args.pairs_file))
        analysis = analyze_ddec(
            job,
            charge_type=args.charge_type,
            net_charge=args.net_charge,
            periodicity=tuple(args.periodicity),
            core_electrons=args.core_electrons,
            exe=args.chargemol_exe,
            atomic_densities=args.atomic_densities,
            compute_bond_orders=args.bos,
            spin=args.spin,
            cube=args.cube,
            grid_shape=tuple(args.grid) if args.grid else None,
            lat0=args.lat0,
            workdir=args.workdir,
            keep=args.keep,
            threads=args.threads,
        )
    except DdecError as error:
        print(f"DDEC analysis failed: {error}")
        return 1

    report = analysis.to_dict()
    if args.output:
        output = _output_path(job, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.json:
            print(f"Wrote DDEC report: {output}")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    shown = select_pairs(
        analysis.pairs,
        selection=selection,
        cutoff=args.cutoff,
        threshold=None if selection is not None else args.threshold,
    )
    _print_table(analysis, shown, args.threshold)
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess ddec``."""
    parser = subparsers.add_parser(
        "ddec",
        help="Calculate DDEC charges, spin moments and bond orders with Chargemol.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
