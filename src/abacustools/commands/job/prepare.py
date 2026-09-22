"""Implementation of the ``abacustools job prepare`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from abacustools.core.input_prep import (
    InputPreparer,
    available_job_types,
    available_resource_libraries,
    parse_input_value,
)
from abacustools.core.submission import write_batch_config


def _pairs(values, *, value_type=float) -> dict:
    result = {}
    for key, value in values or []:
        result[key] = value_type(value)
    return result


def register_parser(subparsers) -> None:
    """Register the ``job prepare`` parser."""
    parser = subparsers.add_parser(
        "prepare",
        help="Prepare complete ABACUS input directories.",
    )
    parser.add_argument(
        "-f", "--file", required=True, action="extend", nargs="+", metavar="STRUCTURE",
        help="Structure files or glob patterns.",
    )
    parser.add_argument("--ftype", default=None, help="Input structure format; inferred by default.")
    parser.add_argument("-o", "--output-dir", default=".", type=Path, help="Directory for generated jobs.")
    parser.add_argument("--job-type", default="scf", choices=available_job_types(), help="ABACUS calculation type.")
    parser.add_argument(
        "--library", choices=available_resource_libraries(), default=None,
        help="Configured pseudopotential/orbital library; uses the configured default by default.",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Orbital variant such as SZ, DZP or TZDP; defaults to resources.orb_variant.",
    )
    parser.add_argument("--input", default=None, type=Path, metavar="INPUT", help="INPUT template.")
    parser.add_argument(
        "--kpt",
        default=None,
        action="append",
        nargs="+",
        type=parse_input_value,
        metavar="VALUE",
        help=(
            "KPT values. Gamma/MP take three or six mesh values; the direct, "
            "cartesian and line models take one group per k-point or node, so "
            "repeat the option for each group."
        ),
    )
    parser.add_argument(
        "--kpt-model", default="gamma",
        choices=("gamma", "mp", "direct", "cartesian", "line", "line_cartesian"),
        help="KPT model used with --kpt, default: gamma.",
    )
    basis = parser.add_mutually_exclusive_group()
    basis.add_argument("--basis", choices=("pw", "lcao"), default=None)
    basis.add_argument("--lcao", dest="basis", action="store_const", const="lcao", help="Use the LCAO basis.")
    parser.add_argument("--nspin", default=1, type=int, choices=(1, 2, 4))
    parser.add_argument("--soc", action="store_true", help="Enable spin-orbit coupling.")
    parser.add_argument("--dftu", action="store_true", help="Enable DFT+U.")
    parser.add_argument(
        "--dftu-param", action="append", nargs=2, metavar=("ELEMENT", "U"),
        help="DFT+U value for an element; repeat for multiple elements.",
    )
    parser.add_argument(
        "--init-mag", action="append", nargs=2, metavar=("ELEMENT", "MAG"),
        help="Initial magnetic moment for an element; repeat for multiple elements.",
    )
    parser.add_argument("--afm", action="store_true", help="Alternate initial moments for magnetic atoms.")
    parser.add_argument(
        "--set", action="append", nargs=2, metavar=("PARAMETER", "VALUE"),
        help="Override an INPUT parameter; repeat for multiple parameters.",
    )
    parser.add_argument(
        "--copy-resources", "--copy-pp-orb", dest="copy_resources", action="store_true",
        help="Copy pseudopotentials/orbitals instead of creating symlinks.",
    )
    parser.add_argument(
        "--folder-syntax",
        default=None,
        help=(
            "Generated folder name as an f-string over {x} (source file name) and "
            "{i} (index), such as {x[:-5]} or {i:03d}."
        ),
    )
    parser.add_argument("--override", "--overwrite", dest="override", action="store_true", help="Replace existing folders.")
    submission = parser.add_mutually_exclusive_group()
    submission.add_argument(
        "--submit-config",
        dest="generate_config",
        action="store_true",
        help="Write the configured batch submission file next to the prepared jobs.",
    )
    submission.add_argument(
        "--no-submit-config",
        dest="generate_config",
        action="store_false",
        help="Do not write the batch submission file, overriding the config default.",
    )
    parser.set_defaults(generate_config=None)
    parser.add_argument(
        "--abacus-command",
        help="ABACUS command used in the batch submission file; otherwise use the config default.",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Prepare one or more complete ABACUS input directories."""
    dftu_param = _pairs(args.dftu_param)
    init_mag = _pairs(args.init_mag)
    set_params = {name.lower(): parse_input_value(value) for name, value in args.set or []}
    jobs = InputPreparer(
        args.file,
        output_dir=args.output_dir,
        filetype=args.ftype,
        job_type=args.job_type,
        library=args.library,
        orb_variant=args.variant,
        input_template=args.input,
        kpt=args.kpt,
        kpt_model=args.kpt_model,
        basis=args.basis,
        nspin=args.nspin,
        soc=args.soc,
        dftu=args.dftu,
        dftu_param=dftu_param,
        init_mag=init_mag,
        afm=args.afm,
        set_params=set_params,
        copy_resources=args.copy_resources,
        folder_syntax=args.folder_syntax,
        overwrite=args.override,
    ).run()
    print("Prepared ABACUS jobs:")
    for job in jobs:
        print(f"  {job.path}  (source: {job.source})")
    if jobs:
        submission = write_batch_config(
            jobs[0].path.parent,
            [job.path for job in jobs],
            job_type=args.job_type,
            generate=args.generate_config,
            abacus_command=args.abacus_command,
        )
        if submission is not None:
            print(
                f"  batch config: {jobs[0].path.parent / submission['config_file']}"
                f"  ({submission['job_count']} jobs)"
            )
    return 0
