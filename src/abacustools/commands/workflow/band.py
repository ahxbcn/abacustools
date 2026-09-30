"""The ``abacustools workflow band`` workflow.

The prepare stage turns one converged-input directory into two jobs: an SCF
that writes the charge density, and an NSCF that reads it back and samples the
seekpath high-symmetry path of the structure with ``out_band 1``. The
postprocess stage reads the band output back and writes the band gap report and
the band plot.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from abacustools.core.job import read_job_structure
from abacustools.data.kpt import band_path, mesh_from_job
from abacustools.io.abacus import ReadInput, WriteKpt

from .common import (
    clear_generated_jobs,
    read_manifest,
    register_stages,
    resolve_output,
    write_abacus_job,
    write_manifest,
)


SCF_DIRECTORY = "band_scf"
NSCF_DIRECTORY = "band_nscf"
SCF_SUFFIX = "scf"
NSCF_SUFFIX = "nscf"


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the band preparation stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS SCF input directory used as the reference for the band calculation.",
    )
    parser.add_argument(
        "--npoints", type=int, default=20,
        help="Points sampled in every band-path segment, default: 20.",
    )
    parser.add_argument(
        "--bands", type=int, default=None, metavar="N",
        help="Number of bands written to the NSCF INPUT as nbands.",
    )
    parser.add_argument(
        "--path-mode", choices=["auto", "bulk", "slab", "wire"], default="auto",
        help="Choose the band path from the dimensionality of the structure, or force one; default: auto.",
    )
    parser.add_argument(
        "--min-vacuum", type=float, default=5.0, metavar="ANGSTROM",
        help="Empty span that counts as vacuum when --path-mode is auto, default: 5.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help=f"Replace existing {SCF_DIRECTORY}/ and {NSCF_DIRECTORY}/ directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the band postprocessing stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory holding the prepared band workflow.",
    )
    parser.add_argument(
        "-o", "--output", default="band_results.json",
        help="JSON report filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--plot", default="band.png",
        help="Band plot filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--emin", type=float, default=-10.0,
        help="Lower energy limit in eV relative to the Fermi level, default: -10.",
    )
    parser.add_argument(
        "--emax", type=float, default=10.0,
        help="Upper energy limit in eV relative to the Fermi level, default: 10.",
    )
    parser.add_argument(
        "--efermi", type=float, default=None,
        help="Override the Fermi energy in eV; it is read from the log by default.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Write the JSON report without the band plot.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the report as JSON.",
    )


def _band_inputs(inputs: dict[str, Any], *, calculation: str, suffix: str) -> dict[str, Any]:
    """Return the INPUT of one band step without the reference k-point setup."""
    step_inputs = deepcopy(inputs)
    for name in ("kspacing", "gamma_only", "out_chg", "out_band", "init_chg", "read_file_dir"):
        step_inputs.pop(name, None)
    step_inputs["gamma_only"] = 0
    step_inputs["kpoint_file"] = "KPT"
    step_inputs["calculation"] = calculation
    step_inputs["suffix"] = suffix
    return step_inputs


def prepare(args: argparse.Namespace) -> int:
    """Prepare the SCF and NSCF jobs of a band structure calculation."""
    if args.npoints < 1:
        raise ValueError("npoints must be a positive integer")
    if args.bands is not None and args.bands <= 0:
        raise ValueError("bands must be a positive integer")

    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    inputs, stru_filename, structure = read_job_structure(job)
    mesh_values, mesh_model = mesh_from_job(job, inputs, structure)
    path = band_path(
        structure,
        npoints=args.npoints,
        min_vacuum=args.min_vacuum,
        path_mode=args.path_mode,
    )
    nodes, segments = path.nodes, path.segments

    scf_inputs = _band_inputs(inputs, calculation="scf", suffix=SCF_SUFFIX)
    scf_inputs["out_chg"] = 1

    nscf_inputs = _band_inputs(inputs, calculation="nscf", suffix=NSCF_SUFFIX)
    nscf_inputs.update(
        {
            "init_chg": "file",
            "read_file_dir": f"../{SCF_DIRECTORY}/OUT.{SCF_SUFFIX}",
            "out_band": 1,
        }
    )
    if args.bands is not None:
        nscf_inputs["nbands"] = args.bands

    clear_generated_jobs(job, [SCF_DIRECTORY, NSCF_DIRECTORY], override=args.override)

    write_abacus_job(
        scf_inputs, structure, job, job / SCF_DIRECTORY, stru_filename=stru_filename
    )
    WriteKpt(mesh_values, str(job / SCF_DIRECTORY / "KPT"), mesh_model)
    write_abacus_job(
        nscf_inputs, structure, job, job / NSCF_DIRECTORY, stru_filename=stru_filename
    )
    WriteKpt(nodes, str(job / NSCF_DIRECTORY / "KPT"), "line")

    labels = path.labels
    write_manifest(
        job,
        "band",
        tasks=[SCF_DIRECTORY, NSCF_DIRECTORY],
        directory=NSCF_DIRECTORY,
        scf_directory=SCF_DIRECTORY,
        scf_suffix=SCF_SUFFIX,
        nscf_suffix=NSCF_SUFFIX,
        stru_filename=stru_filename,
        scf_mesh=[int(value) for value in mesh_values[:3]],
        scf_mesh_model=mesh_model,
        npoints=args.npoints,
        nbands=args.bands,
        natoms=structure.natoms,
        labels=labels,
        segments=segments,
        dimensionality=path.dimensionality,
        dimensionality_label=path.label,
        path_method=path.method,
        periodic_directions=path.periodic_directions,
    )

    print(f"  job: {job}")
    print(f"  atoms: {structure.natoms}")
    print(f"  scf mesh: {' '.join(str(int(value)) for value in mesh_values[:3])} ({mesh_model})")
    print(f"  dimensionality: {path.label}, {path.method}")
    print(f"  path: {' -> '.join(labels)}")
    print(f"  points per segment: {args.npoints}")
    print(f"  prepared: {job / SCF_DIRECTORY} (scf)")
    print(f"  prepared: {job / NSCF_DIRECTORY} (nscf, out_band=1)")
    print("  run the scf job first, then the nscf job that reads its OUT." + SCF_SUFFIX)
    return 0


def _nscf_directory(job: Path, manifest: dict[str, Any]) -> Path:
    """Return the prepared NSCF directory of a band workflow."""
    return job / str(manifest.get("directory", NSCF_DIRECTORY))


def _require_output(directory: Path) -> Path:
    """Return the ABACUS output directory of a finished NSCF job."""
    input_path = directory / "INPUT"
    if not input_path.is_file():
        raise RuntimeError(f"the prepared band job is missing: {input_path}")
    suffix = str(ReadInput(input_path).get("suffix", "ABACUS"))
    output = directory / f"OUT.{suffix}"
    if not output.is_dir():
        raise RuntimeError(
            f"no ABACUS output below {directory}; run the prepared NSCF job first"
        )
    return output


def _label_text(edge: dict[str, Any]) -> str:
    """Return the high-symmetry labels of one band edge."""
    labels = [str(label) for label in edge.get("kpoint_labels", []) if label]
    return ", ".join(labels) if labels else "-"


def _print_report(report: dict[str, Any]) -> None:
    """Print the band gap report of one workflow."""
    gap = report["band_gap"]
    print(f"  band: {report['job']}")
    print(f"  nscf: {report['directory']}")
    if report.get("dimensionality_label"):
        print(f"  dimensionality: {report['dimensionality_label']}, {report.get('path_method')}")
    print(
        f"  k-points: {report['nkpts']}, bands: {report['nbands']}, "
        f"nspin: {report['nspin']}, efermi: {report['efermi']:.6f} eV"
    )
    if gap["is_metal"]:
        print("  band gap: metallic (no gap)")
    else:
        kind = "direct" if gap["direct"] else "indirect"
        print(f"  band gap: {gap['band_gap']:.6f} eV ({kind})")
        print(f"  vbm: {gap['vbm']['energy']:.6f} eV at {_label_text(gap['vbm'])}")
        print(f"  cbm: {gap['cbm']['energy']:.6f} eV at {_label_text(gap['cbm'])}")
    print(f"  results: {report['results']}")
    if report.get("plot"):
        print(f"  plot: {report['plot']}")


def postprocess(args: argparse.Namespace) -> int:
    """Read the band output of a prepared workflow and report the band gap."""
    from abacustools.data.band import BandData, band_gap_report

    if args.emin >= args.emax:
        raise ValueError("emin must be smaller than emax")

    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    manifest = read_manifest(job, "band", [])
    directory = _nscf_directory(job, manifest)
    _require_output(directory)

    band = BandData.ReadFromAbacusJob(str(directory), efermi=args.efermi)
    report: dict[str, Any] = {
        "workflow": "band",
        "job": str(job),
        "directory": str(directory),
        "energy_unit": "eV",
        "nkpts": int(band.nkpts),
        "nbands": int(band.nbands),
        "nspin": int(band.nspin),
        "efermi": float(band.efermi),
        "npoints": manifest.get("npoints"),
        "labels": manifest.get("labels"),
        "segments": manifest.get("segments"),
        "dimensionality": manifest.get("dimensionality"),
        "dimensionality_label": manifest.get("dimensionality_label"),
        "path_method": manifest.get("path_method"),
        "periodic_directions": manifest.get("periodic_directions"),
        "scf_mesh": manifest.get("scf_mesh"),
        "nbands_input": manifest.get("nbands"),
        "band_gap": band_gap_report(band, spin_resolved=True),
    }

    output = resolve_output(job, args.output)
    report["results"] = str(output)
    if not args.no_plot:
        plot = resolve_output(job, args.plot)
        plot.parent.mkdir(parents=True, exist_ok=True)
        band.plot_band(emin=args.emin, emax=args.emax, fig_name=str(plot))
        report["plot"] = str(plot)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


def register_parser(subparsers) -> None:
    """Register the ``abacustools workflow band`` workflow."""
    register_stages(
        subparsers,
        "band",
        "Prepare and postprocess a band structure calculation along the seekpath path.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
