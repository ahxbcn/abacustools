"""The ``abacustools workflow gruneisen`` workflow.

The mode Grueneisen parameter ``gamma(q, nu) = -d ln omega / d ln V`` is what
ties a harmonic phonon calculation to the thermal expansion of a crystal: the
thermodynamic parameter of the quasi-harmonic approximation is the heat
capacity weighted average of the mode values.  It is obtained from three
phonon calculations of the same crystal whose volumes differ by a small
isotropic strain, which is the scheme phonopy implements through
``PhonopyGruneisen``, and which the user's own three-volume examples follow.

The prepare stage therefore writes three ordinary phonon workflows — one per
volume — each of which is submitted and postprocessed exactly like a plain
phonon calculation, and the postprocessing stage only has to put the three
back together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from abacustools.data.gruneisen import (
    DEFAULT_FREQUENCY_CUTOFF,
    DEFAULT_MESH,
    DEFAULT_STRAIN,
    DEFAULT_TEMPERATURE,
    cell_volume,
    gruneisen_temperature,
    scaled_cell,
    summarize,
    validate_strain,
)
from abacustools.data.phonon import (
    band_path,
    jsonable,
    load_workflow_phonon,
    validate_mesh,
    validate_supercell,
)
from abacustools.data.versions import default_version

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_job_structure,
    read_manifest,
    register_stages,
    resolve_output,
    write_abacus_job,
    write_manifest,
)
from .phonon import prepare_phonon_jobs


#: Directory names of the three volumes, in the order equilibrium, expanded,
#: compressed.
_REFERENCE_TASK = "gruneisen_v0"
_TASKS = (_REFERENCE_TASK, "gruneisen_vp", "gruneisen_vm")

#: Sign of the volume strain applied to each of them.
_STRAIN_SIGNS = (0.0, 1.0, -1.0)


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the Grueneisen preparation stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory holding the reference cell.",
    )
    parser.add_argument(
        "-s", "--strain", type=float, default=DEFAULT_STRAIN,
        help="Isotropic volume strain of the two strained cells, default: "
             f"{DEFAULT_STRAIN}.",
    )
    parser.add_argument(
        "--supercell", type=int, nargs=3, metavar=("A", "B", "C"),
        help="Supercell repetitions of the three volumes, shared by all of "
             "them. Defaults to a supercell that keeps every lattice vector "
             "at least --min-supercell-length long.",
    )
    parser.add_argument(
        "--min-supercell-length", type=float, default=10.0,
        help="Minimum lattice-vector length of an automatic supercell, "
             "default: 10.0 Angstrom.",
    )
    parser.add_argument(
        "--displacement-stepsize", type=float, default=0.01,
        help="Finite-difference displacement in Angstrom, default: 0.01.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace existing generated volume directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the Grueneisen postprocessing stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory holding the prepared volumes.",
    )
    parser.add_argument(
        "-v", "--version", default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "--mesh", type=int, nargs=3, default=list(DEFAULT_MESH),
        metavar=("NX", "NY", "NZ"),
        help="q mesh the mode parameters are averaged over, default: "
             f"{' '.join(str(value) for value in DEFAULT_MESH)}.",
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Temperature of the reported thermodynamic average in K, "
             f"default: {DEFAULT_TEMPERATURE}.",
    )
    parser.add_argument(
        "--tmin", type=float, default=0.0,
        help="Lowest temperature of the written curve in K, default: 0.",
    )
    parser.add_argument(
        "--tmax", type=float, default=1000.0,
        help="Highest temperature of the written curve in K, default: 1000.",
    )
    parser.add_argument(
        "--tstep", type=float, default=10.0,
        help="Temperature step of the written curve in K, default: 10.",
    )
    parser.add_argument(
        "--qpath", type=_json_argument, default=None, metavar="JSON",
        help="Band path as a JSON list of high-symmetry labels, for the mode "
             "parameter plot. Defaults to the seekpath path.",
    )
    parser.add_argument(
        "--high-symm-points", type=_json_argument, default=None, metavar="JSON",
        help="JSON object mapping the labels of --qpath to fractional "
             "coordinates.",
    )
    parser.add_argument(
        "--npoints", type=int, default=101,
        help="Points per band segment, default: 101.",
    )
    parser.add_argument(
        "--symmetrize", dest="symmetrize", action="store_true",
        help="Symmetrize the force constants of every volume, the default.",
    )
    parser.add_argument(
        "--no-symmetrize", dest="symmetrize", action="store_false",
        help="Keep the force constants as fitted.",
    )
    parser.set_defaults(symmetrize=True)
    parser.add_argument(
        "-o", "--output", default="gruneisen_results.json",
        help="Output JSON filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--mesh-yaml", default="gruneisen_mesh.yaml",
        help="phonopy mesh result filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--band-yaml", default="gruneisen_band.yaml",
        help="phonopy band result filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--mesh-plot", default="gruneisen_mesh.png",
        help="Mode parameter plot filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--band-plot", default="gruneisen_band.png",
        help="Band and mode parameter plot filename, relative to JOB by default.",
    )


def _json_argument(value: str) -> Any:
    """Parse a JSON command line argument."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error}") from error


def _temperature_grid(tmin: float, tmax: float, tstep: float) -> List[float]:
    """Return the temperatures of the written curve."""
    for value, name in ((tmin, "tmin"), (tmax, "tmax"), (tstep, "tstep")):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if tstep <= 0.0:
        raise ValueError("tstep must be positive")
    if tmax < tmin:
        raise ValueError("tmax must not be below tmin")
    count = int(np.floor((tmax - tmin) / tstep + 0.5))
    return [float(tmin + index * tstep) for index in range(count + 1)]


def prepare(args: argparse.Namespace) -> int:
    """Prepare the three volumes and their displaced supercells."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    strain = validate_strain(args.strain)
    supercell = (
        validate_supercell(args.supercell) if args.supercell is not None else None
    )
    if not np.isfinite(args.min_supercell_length) or args.min_supercell_length <= 0.0:
        raise ValueError("min_supercell_length must be a positive finite number")
    if not np.isfinite(args.displacement_stepsize) or args.displacement_stepsize <= 0.0:
        raise ValueError("displacement_stepsize must be a positive finite number")

    inputs, stru_filename, structure = read_job_structure(job)
    kpoint = kpoint_filename(job, inputs)
    clear_generated_jobs(job, list(_TASKS), override=args.override)

    print(f"  job: {job}")
    print(f"  volume strain: ±{strain:g}")
    volumes: Dict[str, float] = {}
    for name, sign in zip(_TASKS, _STRAIN_SIGNS):
        volume_job = job / name
        cell = structure if sign == 0.0 else scaled_cell(structure, sign * strain)
        write_abacus_job(
            inputs,
            cell,
            job,
            volume_job,
            stru_filename=stru_filename,
            kpoint=kpoint,
        )
        prepare_phonon_jobs(
            volume_job,
            supercell=supercell,
            displacement_stepsize=args.displacement_stepsize,
            min_supercell_length=args.min_supercell_length,
            override=True,
        )
        volumes[name] = cell_volume(cell.cell)
        print(f"  volume {name}: {volumes[name]:.4f} Angstrom^3")

    write_manifest(
        job,
        "gruneisen",
        tasks=list(_TASKS),
        strain=strain,
        volumes=volumes,
        supercell=supercell,
        displacement_stepsize=float(args.displacement_stepsize),
        min_supercell_length=float(args.min_supercell_length),
        stru_filename=stru_filename,
    )
    print("  submit the displaced calculations of the three volumes, then run "
          f"'abacustools workflow gruneisen postprocess -j {job}'")
    return 0


def _volume_tasks(manifest: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return the three volume directories recorded by the preparation stage."""
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise RuntimeError(
            "the Grueneisen workflow manifest does not name three volumes"
        )
    return tuple(str(name) for name in tasks)  # type: ignore[return-value]


def _load_volumes(
    job: Path, workflow: Dict[str, Any], version: str, symmetrize: bool
) -> Dict[str, Any]:
    """Return the phonopy object of every volume."""
    reference, expanded, compressed = _volume_tasks(workflow)
    return {
        "reference": load_workflow_phonon(
            job / reference, version=version, symmetrize=symmetrize
        ),
        "expanded": load_workflow_phonon(
            job / expanded, version=version, symmetrize=symmetrize
        ),
        "compressed": load_workflow_phonon(
            job / compressed, version=version, symmetrize=symmetrize
        ),
    }


def _save_figure(plot: Any, path: Path) -> None:
    """Write a pyplot to a file and close it."""
    import matplotlib

    matplotlib.use("Agg")

    path.parent.mkdir(parents=True, exist_ok=True)
    plot.gcf().savefig(path, dpi=300)
    plot.close("all")


def postprocess(args: argparse.Namespace) -> int:
    """Average the mode parameters and report the thermodynamic one."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    manifest = read_manifest(job, "gruneisen", [])
    strain = validate_strain(manifest.get("strain", DEFAULT_STRAIN))
    mesh = validate_mesh(args.mesh)
    temperatures = _temperature_grid(args.tmin, args.tmax, args.tstep)
    if not np.isfinite(args.temperature) or args.temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")

    from phonopy.api_gruneisen import PhonopyGruneisen

    phonons = _load_volumes(job, manifest, args.version, args.symmetrize)
    gruneisen = PhonopyGruneisen(
        phonons["reference"], phonons["expanded"], phonons["compressed"]
    )
    if not gruneisen.set_mesh(mesh):
        raise RuntimeError(
            "the dynamical matrices of the three volumes are not available; "
            "compute every displaced calculation of every volume first"
        )
    qpoints, weights, frequencies, _, mode_parameters = gruneisen.get_mesh()
    frequencies = np.asarray(frequencies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mode_parameters = np.asarray(mode_parameters, dtype=float)

    mesh_yaml = resolve_output(job, args.mesh_yaml)
    mesh_plot = resolve_output(job, args.mesh_plot)
    gruneisen.write_yaml_mesh(str(mesh_yaml))
    _save_figure(
        gruneisen.plot_mesh(cutoff_frequency=DEFAULT_FREQUENCY_CUTOFF), mesh_plot
    )

    statistics = summarize(frequencies, weights, mode_parameters)
    curve = gruneisen_temperature(
        temperatures, frequencies, weights, mode_parameters
    )
    at_temperature = float(
        np.interp(
            args.temperature,
            temperatures,
            [value if value is not None else np.nan for value in curve],
        )
    )

    _, _, structure = read_job_structure(job)
    paths, labels, connections = band_path(
        structure,
        qpath=args.qpath,
        high_symm_points=args.high_symm_points,
        npoints=args.npoints,
    )
    gruneisen.set_band_structure(paths)
    band_yaml = resolve_output(job, args.band_yaml)
    band_plot = resolve_output(job, args.band_plot)
    gruneisen.write_yaml_band_structure(str(band_yaml))
    _save_figure(gruneisen.plot_band_structure(), band_plot)

    output = resolve_output(job, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "workflow": "gruneisen",
        "method": (
            "central difference of the dynamical matrices of three volumes, "
            "gamma = -d ln omega / d ln V"
        ),
        "units": {
            "mode_gruneisen": "dimensionless",
            "thermodynamic_gruneisen": "dimensionless",
            "temperature": "K",
            "frequency": "THz",
        },
        "strain": strain,
        "volumes_angstrom3": manifest.get("volumes"),
        "supercell": manifest.get("supercell"),
        "displacement_stepsize": manifest.get("displacement_stepsize"),
        "mesh": mesh,
        "mode_gruneisen": statistics,
        "temperature": float(args.temperature),
        "gruneisen_at_temperature": None if not np.isfinite(at_temperature) else at_temperature,
        "gruneisen_temperature": {
            "temperatures": temperatures,
            "values": curve,
        },
        "tasks": list(_volume_tasks(manifest)),
        "mesh_results": str(mesh_yaml),
        "band_results": str(band_yaml),
        "mesh_plot": str(mesh_plot),
        "band_plot": str(band_plot),
    }
    output.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")

    print(f"  job: {job}")
    print(
        "  volumes: "
        + ", ".join(
            f"{name} {value:.4f}" for name, value in (manifest.get("volumes") or {}).items()
        )
        + " Angstrom^3"
    )
    print(f"  q mesh: {' '.join(str(value) for value in mesh)}")
    if "mean" in statistics:
        print(
            f"  mode Grueneisen parameters: {statistics['modes']} modes, "
            f"mean {statistics['mean']:.4f}, range "
            f"{statistics['minimum']:.4f} to {statistics['maximum']:.4f} "
            "(dimensionless)"
        )
        print(
            f"  modes left out: {statistics['modes_left_out']} "
            "(zero frequency or non-finite)"
        )
    else:
        print("  warning: no mode parameter could be averaged")
    if result["gruneisen_at_temperature"] is not None:
        print(
            f"  thermodynamic Grueneisen parameter at {args.temperature:g} K: "
            f"{result['gruneisen_at_temperature']:.4f} (dimensionless)"
        )
    print(f"  mesh results: {mesh_yaml}")
    print(f"  band results: {band_yaml}")
    print(f"  plots: {mesh_plot}, {band_plot}")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the Grueneisen preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "gruneisen",
        "Calculate mode and thermodynamic Grueneisen parameters.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
        aliases=["grueneisen"],
    )
