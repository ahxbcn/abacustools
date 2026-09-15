"""Implementation of the ``abacustools postprocess chg`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from abacustools.data.charge import (
    AXES,
    SPIN_CHOICES,
    ChargeDensityError,
    PlaneSlice,
    atoms_in_plane,
    integrate,
    planar_profile,
    read_job_density,
    select_spin,
    slice_plane,
    subtract,
)
from abacustools.data.nci import (
    QUANTITIES as NCI_QUANTITIES,
    analyse as analyse_nci,
    nci_scatter_data,
)
from abacustools.data.grid import Charge
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


_AUTO_PLOT = "auto"
_KINDS = ("average", "integral")
_UNITS = {"average": "e/Angstrom^3", "integral": "e"}
_DENSITY_UNIT = "e/Angstrom^3"

#: Quantity of the density itself plus the fields derived from it.
_QUANTITIES = ("density",) + tuple(NCI_QUANTITIES)

#: Unit of every quantity, before the profile kind is taken into account.
_QUANTITY_UNITS = {"density": _DENSITY_UNIT, "sl2rho": _DENSITY_UNIT, "rdg": "", "dori": ""}

#: Points a non-covalent interaction plot draws at most.
_NCI_MAX_POINTS = 200_000
_NCI_RHO_MAX = 0.05


def _profile_unit(quantity: str, kind: str) -> str:
    """Return the unit of a profile of the given quantity and kind."""
    unit = _QUANTITY_UNITS[quantity]
    if kind == "average":
        return unit or "dimensionless"
    return "e" if unit else "Angstrom^3"


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
        help="ABACUS job directory with charge-density cubes of either branch.",
    )
    parser.add_argument(
        "--spin",
        choices=SPIN_CHOICES,
        default="total",
        help=(
            "Quantity to analyse: the total density, the up or down channel of an "
            "nspin 2 calculation, or their difference, default: total."
        ),
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help=(
            "FFT grid of the job, needed to convert a charge-density restart "
            "file when no running log reports it."
        ),
    )
    parser.add_argument(
        "--difference",
        default=None,
        metavar="OTHER_JOB",
        help="Subtract the density of another job before the analysis.",
    )
    parser.add_argument(
        "--slice",
        choices=AXES,
        default=None,
        metavar="AXIS",
        help="Cut the plane perpendicular to a, b or c and write it out.",
    )
    parser.add_argument(
        "--slice-index",
        type=float,
        default=0.5,
        metavar="FRACTION",
        help="Fractional position of the slice along its axis, default: 0.5.",
    )
    parser.add_argument(
        "--slice-output",
        default=None,
        metavar="FILE",
        help="Slice data file, relative to JOB by default.",
    )
    parser.add_argument(
        "--slice-plot",
        nargs="?",
        const=_AUTO_PLOT,
        default=None,
        metavar="FILE",
        help=(
            "Plot the slice as a colour map next to its data file. Without FILE "
            "the plot is named after the axis and the slice position."
        ),
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Lower limit of the slice colour scale, default: the data range.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Upper limit of the slice colour scale, default: the data range.",
    )
    parser.add_argument(
        "--no-atoms",
        action="store_true",
        help="Do not mark the atoms that the slice crosses.",
    )
    parser.add_argument(
        "--quantity",
        choices=_QUANTITIES,
        default="density",
        help=(
            "Field to analyse: the density itself, or a field derived from it -- "
            "rdg (reduced density gradient, pp.x plot_num=19), sl2rho "
            "(sign(lambda_2) rho, plot_num=20) or dori (density overlap regions "
            "indicator, plot_num=123)."
        ),
    )
    parser.add_argument(
        "--nci-plot",
        nargs="?",
        const=_AUTO_PLOT,
        default=None,
        metavar="FILE",
        help=(
            "Draw the non-covalent interaction plot of the selected density, "
            "reduced density gradient against sign(lambda_2) rho. Without FILE "
            "the plot is called nci.png."
        ),
    )
    parser.add_argument(
        "--nci-rho-max",
        type=float,
        default=_NCI_RHO_MAX,
        metavar="RHO",
        help=(
            "Largest density in e/Bohr^3 that the non-covalent interaction plot "
            "keeps, default: 0.05, which drops the cores and the bonds."
        ),
    )
    parser.add_argument(
        "--cube",
        default=None,
        metavar="FILE",
        help="Write the total density as a cube file, relative to JOB by default.",
    )
    parser.add_argument(
        "--profile",
        choices=AXES,
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
    unit: str,
) -> None:
    """Write the planar profile as a two-column text file."""
    lines = [f"# distance (Angstrom) {kind} ({unit})"]
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
    unit: str,
) -> None:
    """Plot the planar profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.5, 4.0))
    axes.plot(distances, values, linewidth=1.2)
    axes.set_xlabel(f"{axis} (Angstrom)")
    axes.set_ylabel(
        f"in-plane average ({unit})"
        if kind == "average"
        else f"value per plane ({unit})"
    )
    axes.set_title("Planar profile")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _default_slice_path(job: Path, plane: PlaneSlice) -> Path:
    return job / f"chg_slice_{plane.axis}_{plane.position:.4f}.dat"


def _default_slice_plot_path(job: Path, plane: PlaneSlice) -> Path:
    return job / f"chg_slice_{plane.axis}_{plane.position:.4f}.png"


def _write_slice(path: Path, plane: PlaneSlice, unit: str) -> None:
    """Write a slice as three columns of the two in-plane axes and the value."""
    lines = [
        f"# {plane.axis} slice at fractional {plane.position:.6f} "
        f"(index {plane.index}, {plane.distance:.6f} Angstrom)",
        f"# {plane.labels[0]} (Angstrom) {plane.labels[1]} (Angstrom) value "
        f"({unit})",
    ]
    for first, x_value in enumerate(plane.coordinates[0]):
        for second, y_value in enumerate(plane.coordinates[1]):
            lines.append(
                f"{float(x_value):16.8f} {float(y_value):16.8f} "
                f"{float(plane.values[first, second]):20.10e}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_slice(
    path: Path,
    plane: PlaneSlice,
    atoms: List[Dict[str, Any]],
    vmin: Optional[float],
    vmax: Optional[float],
    unit: str,
) -> None:
    """Plot a slice as a colour map with the atoms that it crosses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.0, 5.0))
    mesh = axes.pcolormesh(
        plane.coordinates[0],
        plane.coordinates[1],
        plane.values.T,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(mesh, ax=axes, label=f"value ({unit})" if unit else "value")
    for atom in atoms:
        x_value, y_value = atom["coordinates"]
        axes.plot(
            x_value,
            y_value,
            "o",
            markersize=7.0,
            markerfacecolor="none",
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        axes.annotate(
            atom["label"],
            (x_value, y_value),
            color="white",
            fontsize=8.0,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
    axes.set_xlabel(f"{plane.labels[0]} (Angstrom)")
    axes.set_ylabel(f"{plane.labels[1]} (Angstrom)")
    axes.set_title(
        f"{plane.axis} slice at {plane.distance:.3f} Angstrom (grid index {plane.index})"
    )
    axes.set_aspect("equal")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _resolve_slice_plot_path(
    value: Any,
    job: Path,
    plane: PlaneSlice,
) -> Optional[Path]:
    """Resolve the ``--slice-plot`` value into the file to write."""
    if value is None:
        return None
    if str(value) == _AUTO_PLOT:
        return _default_slice_plot_path(job, plane)
    return _output_path(job, str(value))


def _slice_atoms(
    job: Path,
    density,
    plane: PlaneSlice,
) -> List[Dict[str, Any]]:
    """Return the atoms that the slice crosses, when the structure is readable.

    A job without a readable STRU simply gets no atom markers, and the reader
    is not called on a missing file, which would print its own error message
    and would pollute a ``--json`` report.
    """
    try:
        inputs = ReadInput(str(job / "INPUT"))
    except (OSError, ValueError):
        return []
    structure_file = job / str(inputs.get("stru_file", "STRU"))
    if not structure_file.is_file():
        return []
    try:
        structure = AbacusSTRU.read(str(structure_file))
    except (OSError, ValueError):
        return []
    if structure is None:
        return []
    return atoms_in_plane(density, plane, structure)


def _nci_plot_path(value: Any, job: Path) -> Optional[Path]:
    """Resolve the ``--nci-plot`` value into the file to write."""
    if value is None:
        return None
    if str(value) == _AUTO_PLOT:
        return job / "nci.png"
    return _output_path(job, str(value))


def _plot_nci(
    path: Path,
    signed: np.ndarray,
    gradient: np.ndarray,
    rho_max: float,
) -> None:
    """Draw the reduced density gradient against sign(lambda_2) rho."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signed = np.asarray(signed, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    finite = np.isfinite(signed) & np.isfinite(gradient)
    signed = signed[finite]
    gradient = gradient[finite]
    if signed.size > _NCI_MAX_POINTS:
        stride = int(np.ceil(signed.size / _NCI_MAX_POINTS))
        signed = signed[::stride]
        gradient = gradient[::stride]

    figure, axes = plt.subplots(figsize=(6.0, 4.5))
    axes.scatter(
        signed,
        gradient,
        c=signed,
        cmap="seismic",
        vmin=-_NCI_RHO_MAX,
        vmax=_NCI_RHO_MAX,
        s=0.6,
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )
    axes.axvline(0.0, color="gray", linewidth=0.6)
    axes.set_xlim(-_NCI_RHO_MAX, _NCI_RHO_MAX)
    axes.set_ylim(0.0, 2.0)
    axes.set_xlabel("sign(lambda_2) rho (e/Bohr^3)")
    axes.set_ylabel("reduced density gradient")
    axes.set_title(
        f"Non-covalent interaction plot (rho <= {rho_max:g} e/Bohr^3, "
        f"{signed.size} points)"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _charge_with_values(field: Charge, values: np.ndarray) -> Charge:
    """Return a field that carries new values on the same grid and cell."""
    return Charge(
        np.asarray(values, dtype=float),
        field.cell,
        field.atom_positions,
        field.atom_types,
        field.atom_charges,
        field.origin,
    )


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
    unit: str,
) -> Dict[str, Any]:
    return {
        "axis": axis,
        "kind": kind,
        "unit": unit,
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
    print(f"  spin selection: {report['spin']}")
    print(f"  analysed quantity: {report['quantity']}")
    if report["difference_of"]:
        print(f"  difference of: {report['difference_of']}")
    if report["magnetization"] is not None:
        print(f"  magnetization: {report['magnetization']:.6f} Bohr magneton")
    grid = " x ".join(str(size) for size in report["grid"])
    print(f"  grid: {grid}, cell volume {report['volume_angstrom3']:.6f} Angstrom^3")
    field = report["field"]
    print(
        f"  field: {field['minimum']:.6g} .. {field['maximum']:.6g} "
        f"(mean {field['mean']:.6g})"
    )
    if "electrons" in report:
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
    slice_report = report.get("slice")
    if slice_report:
        print(
            f"  slice: {slice_report['data_output']} ({slice_report['axis']} axis, "
            f"index {slice_report['index']}, {slice_report['distance']:.6f} Angstrom, "
            f"{slice_report['points']} points, {slice_report['minimum']:.6g} .. "
            f"{slice_report['maximum']:.6g} {slice_report['unit']})"
        )
        if slice_report["plot"]:
            labels = ", ".join(atom["label"] for atom in slice_report["atoms"]) or "none"
            print(f"  slice plot: {slice_report['plot']} (atoms in the plane: {labels})")
    nci = report.get("nci")
    if nci:
        print(
            f"  nci plot: {nci['plot']} ({nci['points']} points with "
            f"rho <= {nci['rho_max']:g} e/Bohr^3)"
        )


def _analyse(args: argparse.Namespace, job: Path) -> Dict[str, Any]:
    """Assemble the density and run the requested actions."""
    grid_shape = tuple(args.grid) if args.grid else None
    density = read_job_density(
        job, description="postprocess chg", grid_shape=grid_shape
    )
    spin = str(args.spin)
    base = select_spin(density, spin)

    difference_of = None
    if args.difference:
        other = Path(args.difference)
        if not other.is_dir():
            raise ChargeDensityError(f"difference job directory does not exist: {other}")
        other_density = read_job_density(
            other, description="postprocess chg --difference"
        )
        base = subtract(base, select_spin(other_density, spin), "the two jobs")
        difference_of = str(other)

    quantity_name = str(args.quantity)
    derived = quantity_name != "density"
    quantity = (
        _charge_with_values(base, analyse_nci(base, quantity_name)) if derived else base
    )
    unit = _QUANTITY_UNITS[quantity_name]

    magnetization = None
    if density.nspin == 2:
        magnetization = integrate(select_spin(density, "difference"))["electrons"]

    values = np.asarray(quantity.data, dtype=float)
    report: Dict[str, Any] = {
        "job": str(job),
        "source": density.source.describe(),
        "cube_files": [str(path) for path in density.source.paths],
        "nspin": density.nspin,
        "spin": spin,
        "quantity": quantity_name,
        "difference_of": difference_of,
        "magnetization": magnetization,
        "grid": [int(size) for size in values.shape],
        "volume_angstrom3": abs(
            float(np.linalg.det(np.asarray(quantity.cell, dtype=float)))
        ),
        "field": {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
        },
    }
    if not derived:
        report.update(
            {
                key: value
                for key, value in integrate(
                    quantity,
                    compare_valence=difference_of is None and spin == "total",
                ).items()
                if key not in ("grid", "volume_angstrom3")
            }
        )

    if args.cube:
        cube_path = _output_path(job, args.cube)
        cube_path.parent.mkdir(parents=True, exist_ok=True)
        quantity.save_cube(str(cube_path))
        report["cube_output"] = str(cube_path)

    if args.profile:
        axis = str(args.profile)
        kind = str(args.profile_kind)
        profile_unit = _profile_unit(quantity_name, kind)
        values, distances = planar_profile(quantity, axis, kind=kind)
        data_path = (
            _output_path(job, args.data_output)
            if args.data_output
            else _default_data_path(job, axis, kind)
        )
        _write_profile(data_path, distances, values, kind, profile_unit)
        plot_path = _resolve_plot_path(args.plot, job, axis, kind)
        if plot_path is not None:
            _plot_profile(plot_path, distances, values, axis, kind, profile_unit)
        report["profile"] = _profile_report(
            job, axis, kind, values, data_path, plot_path, profile_unit
        )

    if args.slice:
        plane = slice_plane(quantity, str(args.slice), position=args.slice_index)
        slice_path = (
            _output_path(job, args.slice_output)
            if args.slice_output
            else _default_slice_path(job, plane)
        )
        _write_slice(slice_path, plane, unit)
        slice_plot = _resolve_slice_plot_path(args.slice_plot, job, plane)
        atoms = [] if args.no_atoms else _slice_atoms(job, quantity, plane)
        if slice_plot is not None:
            _plot_slice(slice_plot, plane, atoms, args.vmin, args.vmax, unit)
        report["slice"] = {
            "axis": plane.axis,
            "index": plane.index,
            "position": plane.position,
            "distance": plane.distance,
            "unit": unit or "dimensionless",
            "points": int(plane.values.size),
            "minimum": float(plane.values.min()),
            "maximum": float(plane.values.max()),
            "sum": float(plane.values.sum()),
            "data_output": str(slice_path),
            "plot": None if slice_plot is None else str(slice_plot),
            "atoms": atoms,
        }

    nci_path = _nci_plot_path(args.nci_plot, job)
    if nci_path is not None:
        signed, gradient = nci_scatter_data(base, rho_max=args.nci_rho_max)
        _plot_nci(nci_path, signed, gradient, args.nci_rho_max)
        report["nci"] = {
            "plot": str(nci_path),
            "points": int(np.size(signed)),
            "rho_max": float(args.nci_rho_max),
        }

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
