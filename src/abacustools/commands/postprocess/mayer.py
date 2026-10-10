"""Process Mayer bond orders from an ABACUS LCAO calculation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from abacustools.data.mayer import MayerAnalysis, analyze_mayer_bond_order


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _output_path(job: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job / path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-j", "--job", required=True, type=_job_directory, help="ABACUS LCAO job directory.")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("-c", "--cutoff", type=float, help="Only analyze atom pairs within this distance in Angstrom.")
    selector.add_argument("-p", "--pairs", help="One-based atom pairs, for example 1-2,1-3.")
    selector.add_argument("--pairs-file", help="File containing one-based atom pairs, one pair per line.")
    parser.add_argument("-t", "--threshold", type=float, default=0.2, help="Only print bond orders >= threshold (default: 0.2).")
    parser.add_argument("-o", "--output", default=None, help="Write a JSON report, relative to JOB by default.")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON report to stdout.")


def _report(analysis: MayerAnalysis, threshold: float | None) -> dict:
    report = asdict(analysis)
    report["pairs"] = [pair for pair in report["pairs"] if threshold is None or pair["bond_order"] >= threshold]
    report["pair_count"] = len(report["pairs"])
    report["total_pair_count"] = len(analysis.pairs)
    return report


def run(args: argparse.Namespace) -> int:
    analysis = analyze_mayer_bond_order(args.job, cutoff=args.cutoff, pairs=args.pairs, pairs_file=args.pairs_file)
    explicit_pairs = args.pairs is not None or args.pairs_file is not None
    report = _report(analysis, None if args.json or explicit_pairs else args.threshold)
    if args.output:
        output = _output_path(Path(args.job), args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_report(analysis, None), indent=2), encoding="utf-8")
        if not args.json:
            print(f"Wrote Mayer bond-order report: {output}")
    if args.json:
        print(json.dumps(_report(analysis, None), indent=2))
        return 0
    print(f"Mayer bond order: {analysis.job} ({'gamma-only' if analysis.gamma_only else 'multi-k'}, nspin={analysis.nspin})")
    print(f"Task: calculation={analysis.calculation}, basis functions={analysis.basis_functions}, pairs={len(analysis.pairs)}")
    for pair in report["pairs"]:
        print(f"  {pair['atom1']}({pair['element1']}) - {pair['atom2']}({pair['element2']}): {pair['bond_order']:.6f}  distance={pair['distance']:.6f} A")
    return 0


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "mayer",
        help=(
            "Analyze Mayer bond orders from an ABACUS LCAO calculation; "
            "a symmetry-reduced k-point mesh (symmetry=1) is expanded to the "
            "full mesh."
        ),
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
