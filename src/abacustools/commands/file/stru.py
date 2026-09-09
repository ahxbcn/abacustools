"""Implementation of the ``abacustools file stru`` command."""

from __future__ import annotations

import argparse
from pathlib import Path


def register_parser(subparsers) -> None:
    """Register the ``file stru`` parser and its arguments."""
    parser = subparsers.add_parser(
        "stru",
        help="Process an ABACUS STRU file.",
    )
    parser.add_argument("filename", type=Path, metavar="INPUT")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        metavar="OUTPUT",
        help="Write a converted structure to OUTPUT.",
    )
    parser.add_argument(
        "--input-format",
        default=None,
        help="Input format (STRU, POSCAR, CIF, XYZ, EXTXYZ, or XSF).",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help="Output format; inferred from OUTPUT when omitted.",
    )
    parser.add_argument(
        "--cell",
        type=float,
        nargs=9,
        metavar="CELL",
        help="Nine cell-vector components for formats without a cell, such as XYZ.",
    )
    coordinates = parser.add_mutually_exclusive_group()
    coordinates.add_argument(
        "--direct",
        action="store_true",
        help="Write coordinates in direct/fractional form where supported.",
    )
    coordinates.add_argument(
        "--cartesian",
        action="store_true",
        help="Write coordinates in Cartesian form where supported.",
    )
    parser.add_argument(
        "--empty2x",
        action="store_true",
        help="Write ABACUS empty labels as the dummy element X.",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Run the ``file stru`` command."""
    if args.output is None:
        from abacustools.io.stru import AbacusSTRU

        structure = AbacusSTRU.read(args.filename, fmt=args.input_format)
        if structure is None:
            raise RuntimeError(f"failed to read structure: {args.filename}")
        print(structure)
        return 0

    from abacustools.io.stru import convert_structure

    direct = True if args.direct else False if args.cartesian else None
    convert_structure(
        str(args.filename),
        str(args.output),
        input_format=args.input_format,
        output_format=args.output_format,
        cell=args.cell,
        empty2x=args.empty2x,
        direct=direct,
    )
    print(f"converted {args.filename} -> {args.output}")
    return 0
