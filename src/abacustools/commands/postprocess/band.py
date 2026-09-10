"""Process and plot band structures from ABACUS NSCF calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from abacustools.data.band import BandData, ProjBandData


_FAT_BAND_MODES = ("species", "species-shell", "species-orbital", "atoms")


def _job_directory(value: str) -> Path:
    """Return an existing ABACUS job directory."""
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _output_path(job: Path, value: str) -> Path:
    """Resolve a relative output path below the job directory."""
    path = Path(value)
    return path if path.is_absolute() else job / path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the band postprocessing command."""
    parser.add_argument(
        "-j",
        "--job",
        required=True,
        type=_job_directory,
        help="ABACUS job directory containing BANDS_1.dat; NSCF is recommended.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Plot filename (band.png or fatband.png by default), relative to JOB.",
    )
    parser.add_argument(
        "--data-output",
        default="band.dat",
        help="Processed band data filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--kpath-output",
        default="KPATH.txt",
        help="High-symmetry k-path filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=-10.0,
        help="Lower energy limit in eV relative to the Fermi level.",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=10.0,
        help="Upper energy limit in eV relative to the Fermi level.",
    )
    parser.add_argument(
        "--efermi",
        type=float,
        default=None,
        help="Override the Fermi energy in eV. Otherwise read it from the NSCF log.",
    )
    parser.add_argument(
        "--gap",
        action="store_true",
        help="Report the band gap, VBM, CBM and metal/insulator character.",
    )
    parser.add_argument(
        "--spin-resolved",
        action="store_true",
        help="Also report spin-resolved band gaps with --gap.",
    )
    parser.add_argument(
        "--effective-mass",
        choices=("cbm", "vbm"),
        default=None,
        metavar="EDGE",
        help="Compute the effective mass at the CBM or VBM (nspin=1 only).",
    )
    parser.add_argument(
        "--direction",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="High-symmetry direction for --effective-mass, e.g. G X.",
    )
    parser.add_argument(
        "--fit-points",
        type=int,
        default=5,
        help="Number of k-points used for the effective-mass fit, default: 5.",
    )
    parser.add_argument(
        "--fat-band",
        choices=_FAT_BAND_MODES,
        default=None,
        metavar="MODE",
        help="Plot projected (fat) bands by species, species-shell, species-orbital, or atoms.",
    )
    parser.add_argument(
        "--atom-index",
        type=int,
        nargs="+",
        default=None,
        help="One-based atom indices for --fat-band atoms.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the band-gap/effective-mass report as JSON.",
    )


def _scalar(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _edge_report(edge: dict[str, Any]) -> dict[str, Any]:
    band_index = edge.get("band_index")
    if isinstance(band_index, dict):
        band_index = {str(spin): list(bands) for spin, bands in band_index.items()}
    return {
        "energy": _scalar(edge.get("energy")),
        "kpoint_index": list(edge.get("kpoint_index", [])),
        "kpoint_labels": list(edge.get("kpoint_labels", [])),
        "kpoint_coord": [list(map(float, coord)) for coord in edge.get("kpoint_coord", [])],
        "band_index": band_index,
    }


def _gap_report(band: BandData, spin_resolved: bool) -> dict[str, Any]:
    vbm = band.get_vbm(spin_resolved=False)
    cbm = band.get_cbm(spin_resolved=False)
    report = {
        "is_metal": bool(band.is_metal()),
        "band_gap": _scalar(band.get_band_gap(spin_resolved=False)),
        "direct": bool(set(vbm.get("kpoint_index", [])) & set(cbm.get("kpoint_index", []))),
        "vbm": _edge_report(vbm),
        "cbm": _edge_report(cbm),
    }
    if spin_resolved:
        report["spin_resolved_band_gap"] = band.get_band_gap(spin_resolved=True)
    return report


def _effective_mass_report(band: BandData, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.direction:
        raise ValueError("--direction START END is required with --effective-mass")
    if band.nspin == 2:
        raise ValueError("effective mass is not supported for nspin=2 calculations")
    results = band.get_effective_mass_at_edge(
        direction_labels=list(args.direction),
        edge_type=args.effective_mass,
        num_fit_points=args.fit_points,
    )
    return [
        {
            "band_index": int(item["band_index"]),
            "effective_mass": float(item["effective_mass"]),
            "curvature": _scalar(item.get("curvature")),
        }
        for item in results
    ]


def _print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    gap = report.get("band_gap")
    if gap is not None:
        if gap["is_metal"]:
            print("Band gap: metallic (no gap)")
        else:
            kind = "direct" if gap["direct"] else "indirect"
            print(f"Band gap: {gap['band_gap']:.6f} eV ({kind})")
        print(
            f"  VBM: {gap['vbm']['energy']:.6f} eV at k-point {gap['vbm']['kpoint_index']}"
        )
        print(
            f"  CBM: {gap['cbm']['energy']:.6f} eV at k-point {gap['cbm']['kpoint_index']}"
        )
        if "spin_resolved_band_gap" in gap:
            print(f"  spin-resolved gaps: {gap['spin_resolved_band_gap']}")

    masses = report.get("effective_mass")
    if masses is not None:
        for item in masses:
            print(
                f"  band {item['band_index']}: m* = {item['effective_mass']:.6f} m_e"
            )


def _plot_fat_band(projected: ProjBandData, args: argparse.Namespace, output: Path) -> None:
    mode = args.fat_band
    if mode == "species":
        projected.plot_proj_band_species(emin=args.emin, emax=args.emax, fig_name=str(output))
    elif mode == "species-shell":
        projected.plot_proj_band_species_shell(
            emin=args.emin, emax=args.emax, fig_name=str(output)
        )
    elif mode == "species-orbital":
        projected.plot_proj_band_species_orbital(
            emin=args.emin, emax=args.emax, fig_name=str(output)
        )
    elif mode == "atoms":
        if not args.atom_index:
            raise ValueError("--atom-index is required for --fat-band atoms")
        projected.plot_proj_band_atoms(
            list(args.atom_index), emin=args.emin, emax=args.emax, fig_name=str(output)
        )


def run(args: argparse.Namespace) -> int:
    """Read an ABACUS band result and run the requested analysis or plot."""
    if args.emin >= args.emax:
        raise ValueError("emin must be smaller than emax")

    job = Path(args.job)

    if args.gap or args.effective_mass is not None:
        band_data = BandData.ReadFromAbacusJob(str(job), efermi=args.efermi)
        report: dict[str, Any] = {}
        if args.gap:
            report["band_gap"] = _gap_report(band_data, args.spin_resolved)
        if args.effective_mass is not None:
            report["effective_mass"] = _effective_mass_report(band_data, args)
        _print_report(report, args.json)
        if args.fat_band is None:
            return 0

    if args.fat_band is not None:
        projected = ProjBandData.ReadFromAbacusJob(str(job), efermi=args.efermi)
        output = _output_path(job, args.output or "fatband.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        _plot_fat_band(projected, args, output)
        print(f"Wrote fat-band plot: {output}")
        return 0

    band_data = BandData.ReadFromAbacusJob(str(job), efermi=args.efermi)
    output = _output_path(job, args.output or "band.png")
    data_output = _output_path(job, args.data_output)
    kpath_output = _output_path(job, args.kpath_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_output.parent.mkdir(parents=True, exist_ok=True)
    kpath_output.parent.mkdir(parents=True, exist_ok=True)

    band_data.plot_band(emin=args.emin, emax=args.emax, fig_name=str(output))
    band_data.write_to_file(str(data_output))
    band_data.write_kpath_info(str(kpath_output))
    if band_data.nspin == 2:
        data_paths = [
            data_output.with_name(f"{data_output.stem}_up{data_output.suffix}"),
            data_output.with_name(f"{data_output.stem}_down{data_output.suffix}"),
        ]
    else:
        data_paths = [data_output]

    print(
        f"Processed {job}: {band_data.nkpts} k-points, "
        f"{band_data.nbands} bands, nspin={band_data.nspin}, "
        f"efermi={band_data.efermi:.8f} eV"
    )
    print(f"Wrote band plot: {output}")
    print(f"Wrote band data: {', '.join(str(path) for path in data_paths)}")
    print(f"Wrote k-path information: {kpath_output}")
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess band``."""
    parser = subparsers.add_parser(
        "band",
        help="Process and plot bands from an ABACUS NSCF calculation.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
