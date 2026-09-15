"""Implementation of the ``abacustools postprocess chg`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from abacustools.data.charge import (
    ChargeDensityError,
    integrate,
    planar_profile,
    read_job_density,
)


_AUTO_PLOT = "auto"
_AXES = ("a", "b", "c")
_KINDS = ("average", "integral")
_UNITS = {"average": "e/Angstrom^3", "integral": "e"}


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def _output_path(job: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job / path


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the arguments of the charge-density command."""
    parser.add_argument(
        "-j", "--job",
        required=True,
        type=_job_directory,
        help="ABACUS job directory with SPIN*_CHG.cube files.",
    )
    parser.add_argument(
        "--cube",
        default=None,
        metavar="FILE",
        help="Write the total density as a cube file, relative to JOB by default.",
    )
    parser.add_argument(
        "--profile",
        choices=_AXES,
        default=None,
        metavar="AXIS",
        help="Write the planar profile along the a, b or c direction.",
    )
    parser.add_argument(
        "--profile-kind",
        choices=_KINDS,
        default="average",
        help=(
            "Profile of the in-plane average in e/Angstrom^3 or of the charge "
            "per plane in e, default: average."
        ),
    )
    parser.add_argument(
        "--data-output",
        default=None,
        metavar="FILE",
        help="Profile data file, relative to JOB by default.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=_AUTO_PLOT,
        default=None,
        metavar="FILE",
        help=(
            "Plot the profile next to the data file. Without FILE the plot is "
            "named after the axis and the profile kind."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")


def _default_data_path(job: Path, axis: str, kind: str) -> Path:
    return job / f"chg_profile_{axis}_{kind}.dat"


def _default_plot_path(job: Path, axis: str, kind: str) -> Path:
    return job / f"chg_profile_{axis}_{kind}.png"


def _write_profile(
    path: Path,
    distances: np.ndarray,
    values: np.ndarray,
    kind: str,
) -> None:
    """Write the planar profile as a two-column text file."""
    lines = [f"# distance (Angstrom) {kind} ({_UNITS[kind]})"]
    lines.extend(
        f"{float(distance):16.8f} {float(value):20.10e}"
        for distance, value in zip(distances, values)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_profile(
    path: Path,
    distances: np.ndarray,
    values: np.ndarray,
    axis: str,
    kind: str,
) -> None:
    """Plot the planar profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.5, 4.0))
    axes.plot(distances, values, linewidth=1.2)
    axes.set_xlabel(f"{axis} (Angstrom)")
    axes.set_ylabel(
        "in-plane average (e/Angstrom^3)"
        if kind == "average"
        else "charge per plane (e)"
    )
    axes.set_title("Planar charge-density profile")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _resolve_plot_path(
    value: Any,
    job: Path,
    axis: str,
    kind: str,
) -> Optional[Path]:
    """Resolve the ``--plot`` value into the file to write."""
    if value is None:
        return None
    if str(value) == _AUTO_PLOT:
        return _default_plot_path(job, axis, kind)
    return _output_path(job, str(value))


def _profile_report(
    job: Path,
    axis: str,
    kind: str,
    values: np.ndarray,
    data_path: Path,
    plot_path: Optional[Path],
) -> Dict[str, Any]:
    return {
        "axis": axis,
        "kind": kind,
        "unit": _UNITS[kind],
        "points": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "sum": float(values.sum()),
        "data_output": str(data_path),
        "plot": None if plot_path is None else str(plot_path),
    }


def _print_report(report: Dict[str, Any]) -> None:
    """Print the summary of the charge-density analysis."""
    print(f"  job: {report['job']}")
    print(f"  density source: {report['source']}")
    print(f"  spin channels: {report['nspin']}")
    grid = " x ".join(str(size) for size in report["grid"])
    print(f"  grid: {grid}, cell volume {report['volume_angstrom3']:.6f} Angstrom^3")
    electrons = f"  electrons: {report['electrons']:.6f} e"
    if report["valence_electrons"] is not None:
        electrons += (
            f" (valence electrons {report['valence_electrons']:.6f} e, "
            f"deviation {report['deviation']:+.6f} e)"
        )
    print(electrons)
    if report.get("cube_output"):
        print(f"  cube: {report['cube_output']}")
    profile = report.get("profile")
    if profile:
        print(
            f"  profile: {profile['data_output']} "
            f"({profile['points']} points, {profile['minimum']:.6g} .. "
            f"{profile['maximum']:.6g} {profile['unit']})"
        )
        if profile["plot"]:
            print(f"  plot: {profile['plot']}")


def _analyse(args: argparse.Namespace, job: Path) -> Dict[str, Any]:
    """Assemble the density and run the requested actions."""
    density = read_job_density(job, description="postprocess chg")
    total = density.total()
    report: Dict[str, Any] = {
        "job": str(job),
        "source": density.source.describe(),
        "cube_files": [str(path) for path in density.source.paths],
        "nspin": density.nspin,
        **integrate(total),
    }

    if args.cube:
        cube_path = _output_path(job, args.cube)
        cube_path.parent.mkdir(parents=True, exist_ok=True)
        total.save_cube(str(cube_path))
        report["cube_output"] = str(cube_path)

    if args.profile:
        axis = str(args.profile)
        kind = str(args.profile_kind)
        values, distances = planar_profile(total, axis, kind=kind)
        data_path = (
            _output_path(job, args.data_output)
            if args.data_output
            else _default_data_path(job, axis, kind)
        )
        _write_profile(data_path, distances, values, kind)
        plot_path = _resolve_plot_path(args.plot, job, axis, kind)
        if plot_path is not None:
            _plot_profile(plot_path, distances, values, axis, kind)
        report["profile"] = _profile_report(job, axis, kind, values, data_path, plot_path)

    return report


def run(args: argparse.Namespace) -> int:
    """Run ``abacustools postprocess chg``."""
    job = Path(args.job)
    try:
        report = _analyse(args, job)
    except ChargeDensityError as error:
        print(f"Charge-density analysis failed: {error}")
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    _print_report(report)
    return 0


def register_parser(subparsers) -> None:
    """Register ``abacustools postprocess chg``."""
    parser = subparsers.add_parser(
        "chg",
        help="Inspect a charge density, export it as a cube or reduce it to a profile.",
    )
    _register_arguments(parser)
    parser.set_defaults(handler=run)
