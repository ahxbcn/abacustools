"""The ``abacustools workflow thermal-conductivity`` workflow.

The workflow follows the finite-displacement scheme of phono3py: the
preparation stage expands the cell, generates the second- and third-order
displacement supercells and writes one ABACUS force calculation per
displacement; the postprocessing stage reads their forces, fits ``fc2`` and
``fc3`` and solves the phonon Boltzmann transport equation for the lattice
thermal conductivity.  An independent, usually larger, ``fc2`` supercell can
be requested to obtain a better phonon spectrum than the ``fc3`` supercell
provides.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from abacustools.core.submission import generate_workflow_submission
from abacustools.data.phonon import (
    automatic_supercell,
    collect_forces,
    displacement_tasks,
    jsonable,
    phonopy_atoms,
    phonopy_supercell_structure,
    validate_displacement_entries,
    validate_mesh,
    validate_positive_float,
    validate_supercell,
)
from abacustools.data.versions import default_version

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    read_job_structure,
    register_stages,
    write_abacus_job,
    write_manifest,
)


_WORKFLOW = "thermal_conductivity"
_FC3_PREFIX = "fc3-"
_FC2_PREFIX = "fc2-"
_PHONO3PY_FILE = "phono3py_disp.yaml"
_KAPPA_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")


def _load_phono3py():
    """Import phono3py lazily with an actionable error message."""
    try:
        import phono3py
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "the thermal-conductivity workflow requires phono3py; install it "
            "with 'pip install phono3py' or "
            "'conda install -c conda-forge phono3py'"
        ) from error
    return phono3py


def _temperatures(tmin: float, tmax: float, tstep: float) -> list[float]:
    """Build the requested temperature list in Kelvin."""
    for name, value in (("tmin", tmin), ("tmax", tmax), ("tstep", tstep)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if tmin < 0:
        raise ValueError("tmin must not be negative")
    if tmax < tmin:
        raise ValueError("tmax must not be smaller than tmin")
    if tstep <= 0:
        raise ValueError("tstep must be positive")
    count = int(np.floor((tmax - tmin) / tstep)) + 1
    return [float(tmin + index * tstep) for index in range(count)]


def _mesh_setting(values: Any) -> list[int]:
    """Validate a one- or three-dimensional phono3py mesh."""
    mesh = list(values)
    if len(mesh) == 1:
        mesh = validate_mesh([mesh[0], mesh[0], mesh[0]])[:1]
    else:
        mesh = validate_mesh(mesh)
    return mesh


def _scale_kpoints(kpt: Any, model: Any, supercell: Any) -> list:
    """Scale an explicit Monkhorst-Pack mesh with the supercell size.

    The reciprocal cell of a supercell shrinks by the replication factor, so
    the mesh is divided by it to keep the k-point density unchanged.  Other KPT
    modes (direct, line) and kspacing-driven input are used unchanged.
    """
    values = list(kpt)
    if str(model).lower() not in {"gamma", "mp"} or len(values) < 3:
        return values
    scaled = list(values)
    for axis, multiplier in enumerate(list(supercell)[:3]):
        try:
            current = int(round(float(values[axis])))
        except (TypeError, ValueError):
            return values
        scaled[axis] = max(1, int(round(current / int(multiplier))))
    return scaled




def _existing_task_names(job: Path) -> list[str]:
    """Find old displacement directories so changed settings cannot leave stale jobs."""
    names = set()
    for prefix in (_FC3_PREFIX, _FC2_PREFIX):
        names.update(
            path.name
            for path in job.glob(f"{prefix}*")
            if path.is_dir() or path.is_symlink()
        )
    return sorted(names)


def _displaced_structure(template, phonopy_supercell):
    """Copy an ABACUS supercell and apply one phono3py displacement."""
    displaced = deepcopy(template)
    displaced.cell = np.asarray(phonopy_supercell.cell, dtype=float).tolist()
    displaced.coords = np.asarray(phonopy_supercell.positions, dtype=float).tolist()
    return displaced


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the thermal-conductivity preparation stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="ABACUS input directory used to prepare the force calculations.",
    )
    parser.add_argument(
        "--supercell-fc3",
        type=int,
        nargs=3,
        metavar=("A", "B", "C"),
        help="Third-order supercell repetitions; by default it is chosen from --min-supercell-length.",
    )
    parser.add_argument(
        "--supercell-fc2",
        type=int,
        nargs=3,
        metavar=("A", "B", "C"),
        help="Independent second-order supercell repetitions; by default fc2 reuses the fc3 supercell.",
    )
    parser.add_argument(
        "--displacement-stepsize-fc3",
        type=float,
        default=0.03,
        help="Third-order finite-difference displacement in Angstrom, default: 0.03.",
    )
    parser.add_argument(
        "--displacement-stepsize-fc2",
        type=float,
        default=None,
        help="Second-order displacement in Angstrom, default: the fc3 step.",
    )
    parser.add_argument(
        "--min-supercell-length",
        type=float,
        default=10.0,
        help="Minimum lattice-vector length for an automatic fc3 supercell, default: 10.0 Angstrom.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace existing generated displacement directories.",
    )
    submission = parser.add_mutually_exclusive_group()
    submission.add_argument(
        "--submit-script",
        dest="generate_scripts",
        action="store_true",
        help="Generate configured task and workflow submission scripts.",
    )
    submission.add_argument(
        "--no-submit-script",
        dest="generate_scripts",
        action="store_false",
        help="Do not generate submission scripts, overriding the config default.",
    )
    parser.set_defaults(generate_scripts=None)
    parser.add_argument(
        "--submission-type",
        "--submit-type",
        dest="submission_type",
        help="Submission template type from the config, such as local, slurm, pbs, or lsf.",
    )
    parser.add_argument(
        "--abacus-command",
        help="ABACUS command used in generated scripts; otherwise use the config default.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the thermal-conductivity postprocessing stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="Directory containing the prepared force calculations.",
    )
    parser.add_argument(
        "-v", "--version",
        default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        nargs="+",
        default=[11, 11, 11],
        metavar="N",
        help="q-point mesh: three subdivisions, or one value for an automatic mesh, default: 11 11 11.",
    )
    parser.add_argument(
        "--tmin", type=float, default=100.0, help="Lowest temperature in K, default: 100."
    )
    parser.add_argument(
        "--tmax", type=float, default=500.0, help="Highest temperature in K, default: 500."
    )
    parser.add_argument(
        "--tstep", type=float, default=100.0, help="Temperature step in K, default: 100."
    )
    parser.add_argument(
        "--lbte",
        action="store_true",
        help="Solve the linearized Boltzmann equation directly instead of the RTA.",
    )
    parser.add_argument(
        "-o", "--output",
        default="thermal_conductivity_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )
    parser.add_argument(
        "--plot",
        default="thermal_conductivity.png",
        help="Conductivity plot filename. Relative paths are resolved below JOB.",
    )


def prepare(args: argparse.Namespace) -> int:
    """Prepare second- and third-order displaced supercell calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    validate_positive_float(args.displacement_stepsize_fc3, "displacement_stepsize_fc3")
    validate_positive_float(args.min_supercell_length, "min_supercell_length")
    stepsize_fc2 = (
        args.displacement_stepsize_fc3
        if args.displacement_stepsize_fc2 is None
        else args.displacement_stepsize_fc2
    )
    validate_positive_float(stepsize_fc2, "displacement_stepsize_fc2")

    inputs, stru_filename, structure = read_job_structure(job)
    supercell_fc3 = (
        validate_supercell(args.supercell_fc3)
        if args.supercell_fc3 is not None
        else automatic_supercell(structure, args.min_supercell_length)
    )
    supercell_fc2 = (
        validate_supercell(args.supercell_fc2)
        if args.supercell_fc2 is not None
        else None
    )
    separate_fc2 = supercell_fc2 is not None
    if not separate_fc2 and not np.isclose(stepsize_fc2, args.displacement_stepsize_fc3):
        raise ValueError(
            "fc2 and fc3 displacements share one supercell; set --supercell-fc2 "
            "to use a different displacement step for fc2"
        )

    phono3py = _load_phono3py()
    ph3 = phono3py.Phono3py(
        phonopy_atoms(structure),
        supercell_matrix=np.diag(supercell_fc3),
        primitive_matrix="P",
        phonon_supercell_matrix=np.diag(supercell_fc2) if separate_fc2 else None,
    )
    ph3.generate_displacements(distance=args.displacement_stepsize_fc3)
    if separate_fc2 and not np.isclose(stepsize_fc2, args.displacement_stepsize_fc3):
        # generate_displacements also builds the fc2 dataset with the fc3 step.
        ph3.generate_fc2_displacements(distance=stepsize_fc2)

    fc3_tasks = displacement_tasks(ph3.supercells_with_displacements, _FC3_PREFIX)
    if separate_fc2:
        fc2_supercells = ph3.phonon_supercells_with_displacements
        fc2_tasks = displacement_tasks(fc2_supercells, _FC2_PREFIX)
    else:
        fc2_supercells = []
        fc2_tasks = []
    if not fc3_tasks:
        raise RuntimeError("phono3py generated no fc3 displacement supercells")
    if separate_fc2 and not fc2_tasks:
        raise RuntimeError("phono3py generated no fc2 displacement supercells")

    task_names = [item["task"] for item in fc3_tasks + fc2_tasks]
    clear_generated_jobs(
        job,
        sorted(set(task_names + _existing_task_names(job))),
        override=args.override,
    )
    ph3.save(job / _PHONO3PY_FILE)

    force_inputs = deepcopy(inputs)
    force_inputs["calculation"] = "scf"
    force_inputs["cal_force"] = 1
    try:
        scf_thr = float(force_inputs.get("scf_thr", 1e-7))
    except (TypeError, ValueError):
        scf_thr = 1e-7
    if scf_thr > 1e-7:
        force_inputs["scf_thr"] = 1e-7

    kpoint_file = kpoint_filename(job, inputs)
    kpoint = None
    kpoint_model = None
    if kpoint_file is not None:
        from abacustools.io.abacus import ReadKpt, WriteKpt

        kpoint, kpoint_model = ReadKpt(str(job / kpoint_file))

    datasets = [
        (fc3_tasks, supercell_fc3, ph3.supercell, ph3.supercells_with_displacements)
    ]
    if separate_fc2:
        datasets.append(
            (fc2_tasks, supercell_fc2, ph3.phonon_supercell, fc2_supercells)
        )
    for tasks, supercell, phonopy_supercell, supercells in datasets:
        template = phonopy_supercell_structure(structure, phonopy_supercell)
        for item in tasks:
            displaced = supercells[item["index"]]
            if template.natoms != len(displaced):
                raise RuntimeError(
                    "phono3py and ABACUS generated supercells with different atom counts"
                )
            task_path = job / item["task"]
            write_abacus_job(
                force_inputs,
                _displaced_structure(template, displaced),
                job,
                task_path,
                stru_filename=stru_filename,
            )
            if kpoint is not None:
                WriteKpt(
                    _scale_kpoints(kpoint, kpoint_model, supercell),
                    str(task_path / kpoint_file),
                    kpoint_model,
                )
            print(f"  prepared {item['task']}")

    submission = generate_workflow_submission(
        job,
        _WORKFLOW,
        task_names,
        submission_type=getattr(args, "submission_type", None),
        generate=getattr(args, "generate_scripts", None),
        abacus_command=getattr(args, "abacus_command", None),
    )
    manifest: dict[str, Any] = dict(
        tasks=task_names,
        fc3_supercell=supercell_fc3,
        fc2_supercell=supercell_fc2,
        displacement_stepsize_fc3=float(args.displacement_stepsize_fc3),
        displacement_stepsize_fc2=float(stepsize_fc2),
        fc3_displacements=fc3_tasks,
        fc2_displacements=fc2_tasks,
        fc3_supercell_count=len(ph3.supercells_with_displacements),
        fc2_supercell_count=len(fc2_supercells),
        phono3py_file=_PHONO3PY_FILE,
    )
    if submission is not None:
        manifest["submission"] = submission
    write_manifest(job, _WORKFLOW, **manifest)

    print(f"  job: {job}")
    print(f"  fc3 supercell: {' '.join(str(value) for value in supercell_fc3)}")
    if separate_fc2:
        print(f"  fc2 supercell: {' '.join(str(value) for value in supercell_fc2)}")
    else:
        print("  fc2 supercell: shared with fc3")
    print(f"  fc3 displacement step: {args.displacement_stepsize_fc3} Angstrom")
    print(f"  fc2 displacement step: {stepsize_fc2} Angstrom")
    print(f"  generated calculations: {len(task_names)}")
    print(f"  phono3py dataset: {job / _PHONO3PY_FILE}")
    if submission is not None:
        print(f"  submission type: {submission['type']}")
        print(f"  workflow script: {submission['workflow_script']}")
    return 0




@contextmanager
def _working_directory(directory: Path) -> Iterator[None]:
    """Run a phono3py call inside *directory*.

    phono3py writes its ``kappa-*.hdf5`` output relative to the process working
    directory, which would otherwise land in the caller's directory.
    """
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def _kappa_array(kappa: Any) -> Optional[np.ndarray]:
    """Return the per-temperature kappa table of a phono3py result."""
    values = np.asarray(kappa, dtype=float)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2 or values.shape[1] != len(_KAPPA_COMPONENTS):
        return None
    return values


def _conductivity_components(values: np.ndarray) -> dict[str, list[float]]:
    """Convert a per-temperature kappa table into named Voigt components."""
    return {
        name: values[:, index].tolist()
        for index, name in enumerate(_KAPPA_COMPONENTS)
    }


def _plot_conductivity(
    temperatures: list[float],
    components: dict[str, list[float]],
    plot_path: Path,
) -> None:
    """Plot the diagonal conductivity components against temperature."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    for name in ("xx", "yy", "zz"):
        axis.plot(temperatures, components[name], marker="o", label=rf"$\kappa_{{{name}}}$")
    average = np.mean([components[name] for name in ("xx", "yy", "zz")], axis=0)
    axis.plot(temperatures, average, marker="s", linestyle="--", label=r"$\kappa_{avg}$")
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel("Lattice thermal conductivity (W/m-K)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)


def postprocess(args: argparse.Namespace) -> int:
    """Fit fc2/fc3 from the collected forces and solve the transport equation."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    mesh = _mesh_setting(args.mesh)
    temperatures = _temperatures(args.tmin, args.tmax, args.tstep)

    manifest = read_manifest(job, _WORKFLOW, [])
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
        raise RuntimeError("thermal conductivity manifest has invalid tasks")
    read_manifest(job, _WORKFLOW, tasks)
    try:
        supercell_fc3 = validate_supercell(manifest.get("fc3_supercell"))
    except ValueError as error:
        raise RuntimeError(
            "thermal conductivity manifest has an invalid fc3 supercell"
        ) from error

    phono3py = _load_phono3py()
    yaml_path = job / str(manifest.get("phono3py_file", _PHONO3PY_FILE))
    if not yaml_path.is_file():
        raise RuntimeError(
            f"could not find the phono3py dataset {yaml_path}; run the prepare stage first"
        )
    ph3 = phono3py.load(str(yaml_path), produce_fc=False, log_level=0)
    actual_fc3 = list(np.diag(np.asarray(ph3.supercell_matrix, dtype=int)))
    if actual_fc3 != supercell_fc3:
        raise RuntimeError(
            f"phono3py dataset has fc3 supercell {actual_fc3}, but the manifest "
            f"records {supercell_fc3}"
        )

    fc3_entries = validate_displacement_entries(
        manifest.get("fc3_displacements"),
        len(ph3.supercells_with_displacements),
        "thermal conductivity",
    )
    ph3.forces = collect_forces(
        job,
        fc3_entries,
        ph3.supercells_with_displacements,
        args.version,
        len(ph3.supercell),
        workflow="thermal conductivity",
    )
    ph3.produce_fc3()
    ph3.symmetrize_fc3()

    separate_fc2 = ph3.phonon_supercell_matrix is not None
    if separate_fc2:
        fc2_entries = validate_displacement_entries(
            manifest.get("fc2_displacements"),
            len(ph3.phonon_supercells_with_displacements),
            "thermal conductivity",
        )
        ph3.phonon_forces = collect_forces(
            job,
            fc2_entries,
            ph3.phonon_supercells_with_displacements,
            args.version,
            len(ph3.phonon_supercell),
            workflow="thermal conductivity",
        )
        ph3.produce_fc2()
        ph3.symmetrize_fc2()
    else:
        ph3.symmetrize_fc2()

    ph3.mesh_numbers = mesh if len(mesh) == 3 else mesh[0]
    ph3.init_phph_interaction()
    with _working_directory(job):
        ph3.run_thermal_conductivity(
            temperatures=temperatures,
            is_LBTE=bool(args.lbte),
            write_kappa=True,
        )

    conductivity = ph3.thermal_conductivity
    kappa = _kappa_array(conductivity.kappa)
    if kappa is None:
        raise RuntimeError(
            "unexpected thermal conductivity array shape: "
            f"{np.asarray(conductivity.kappa).shape}"
        )
    components = _conductivity_components(kappa)
    if len(components["xx"]) != len(temperatures):
        raise RuntimeError(
            "thermal conductivity array does not match the requested temperatures"
        )
    average = np.mean([components[name] for name in ("xx", "yy", "zz")], axis=0).tolist()

    rta_components = None
    if args.lbte:
        rta = _kappa_array(getattr(conductivity, "kappa_RTA", None))
        if rta is not None and len(rta) == len(temperatures):
            rta_components = _conductivity_components(rta)

    kappa_file = job / f"kappa-m{''.join(str(value) for value in mesh)}.hdf5"
    result = {
        "job": str(job),
        "method": "LBTE" if args.lbte else "RTA",
        "mesh": mesh,
        "temperatures": temperatures,
        "fc3_supercell": supercell_fc3,
        "fc2_supercell": (
            list(np.diag(np.asarray(ph3.phonon_supercell_matrix, dtype=int)))
            if separate_fc2
            else None
        ),
        "displacement_stepsize_fc3": manifest.get("displacement_stepsize_fc3"),
        "displacement_stepsize_fc2": manifest.get("displacement_stepsize_fc2"),
        "fc3_calculations": len(manifest.get("fc3_displacements", [])),
        "fc2_calculations": len(manifest.get("fc2_displacements", [])),
        "kappa_w_m_k": components,
        "kappa_average_w_m_k": average,
        "kappa_components_w_m_k": [
            [temperature, *values] for temperature, values in zip(temperatures, kappa.tolist())
        ],
        "rta_kappa_w_m_k": rta_components,
        "kappa_hdf5": str(kappa_file) if kappa_file.is_file() else None,
    }

    plot_path = Path(args.plot)
    if not plot_path.is_absolute():
        plot_path = job / plot_path
    _plot_conductivity(temperatures, components, plot_path)
    result["plot"] = str(plot_path)

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")

    header = "  ".join(f"{name:>10}" for name in ("T(K)", *_KAPPA_COMPONENTS))
    print(f"  job: {job}")
    print(f"  method: {result['method']}")
    print(f"  mesh: {' '.join(str(value) for value in mesh)}")
    print(f"  temperatures: {temperatures[0]:g} - {temperatures[-1]:g} K")
    print("  lattice thermal conductivity (W/m-K)")
    print(f"  {header}")
    for index, temperature in enumerate(temperatures):
        values = "  ".join(
            f"{components[name][index]:10.4f}" for name in _KAPPA_COMPONENTS
        )
        print(f"  {temperature:10.1f}  {values}")
    print(f"  plot: {plot_path}")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the thermal-conductivity preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "thermal-conductivity",
        "Calculate the lattice thermal conductivity with phono3py.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
        aliases=["kappa"],
    )
