"""The ``abacustools workflow dielectric`` workflow.

The clamped ion (electronic) dielectric tensor ``epsilon_inf`` is the response
of the electrons alone, which is what a phonon non-analytical correction needs:
the long range field of a longitudinal optical vibration is screened by the
electrons but not by the ions, which are the ones moving.  ABACUS does not
write it, so the workflow prepares the one modification of a self consistent
calculation that makes it readable by ``pyatb``, and later evaluates the
Kubo-Greenwood sum on the matrices ABACUS wrote.

Only the static limit is taken.  ``pyatb`` offers many other response
functions, and this workflow deliberately does not expose them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from abacustools.data.dielectric import (
    DEFAULT_DOMEGA,
    DEFAULT_ETA,
    DEFAULT_GRID,
    DEFAULT_OMEGA,
    MATRIX_KEYWORDS,
    copy_matrices,
    dielectric_job_parameters,
    omega_window_warning,
    prepare_matrix_output,
)
from abacustools.integrations.pyatb import dielectric_tensor
from abacustools.core.job import read_job_structure

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    register_stages,
    workflow_manifest_path,
    write_abacus_job,
    write_manifest,
)


_DEFAULT_NAME = "dielectric"
_DEFAULT_WORKDIR = "pyatb"
_DEFAULT_OUTPUT = "dielectric_results.json"
_TENSOR_UNIT = "relative permittivity (dimensionless)"


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the dielectric preparation stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory holding the converged SCF of the cell.",
    )
    parser.add_argument(
        "-n", "--name", default=_DEFAULT_NAME,
        help=f"Name of the generated calculation, default: {_DEFAULT_NAME}.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace an existing generated calculation directory.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the dielectric postprocessing stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory holding the prepared dielectric calculation.",
    )
    parser.add_argument(
        "-t", "--task", default=None,
        help="Generated calculation to read. Defaults to the one recorded in the "
             "workflow manifest, or to JOB itself when there is none.",
    )
    parser.add_argument(
        "--workdir", default=_DEFAULT_WORKDIR,
        help="pyatb working directory, resolved below the calculation. It holds a "
             "copy of the ABACUS matrices and the pyatb results, "
             f"default: {_DEFAULT_WORKDIR}.",
    )
    parser.add_argument(
        "--grid", type=int, nargs=3, default=list(DEFAULT_GRID),
        metavar=("NX", "NY", "NZ"),
        help="Dense Brillouin zone grid of the Kubo-Greenwood sum, default: "
             f"{' '.join(str(value) for value in DEFAULT_GRID)}.",
    )
    parser.add_argument(
        "--omega", type=float, nargs=2, default=list(DEFAULT_OMEGA),
        metavar=("MIN", "MAX"),
        help="Photon energy window in eV. It has to start at zero, and to end "
             "well above the band gap, because every transition contributes to "
             "the zero frequency tensor, default: "
             f"{' '.join(str(value) for value in DEFAULT_OMEGA)}.",
    )
    parser.add_argument(
        "--domega", type=float, default=DEFAULT_DOMEGA,
        help=f"Photon energy step in eV, default: {DEFAULT_DOMEGA}.",
    )
    parser.add_argument(
        "--eta", type=float, default=DEFAULT_ETA,
        help=f"Gaussian broadening in eV, default: {DEFAULT_ETA}.",
    )
    parser.add_argument(
        "--max-kpoint-num", type=int, default=8000,
        help="Largest number of k points pyatb holds in memory at once, "
             "default: 8000.",
    )
    parser.add_argument(
        "-o", "--output", default=_DEFAULT_OUTPUT,
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )
    parser.add_argument(
        "--pyatb-command", default=None,
        help="Run pyatb through this command instead of in process, for example "
             "'mpirun -np 4 pyatb'.",
    )


def _generated_name(value: str) -> str:
    """Validate the name of a generated calculation directory."""
    name = str(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"the generated calculation needs a plain name, got {value!r}")
    return name


def prepare(args: argparse.Namespace) -> int:
    """Prepare the self consistent calculation that dumps the matrices."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    name = _generated_name(args.name)
    inputs, stru_filename, structure = read_job_structure(job)
    # Fails with an explicit message when the job cannot write the matrices.
    prepared = prepare_matrix_output(inputs)
    kpoint = kpoint_filename(job, inputs)
    clear_generated_jobs(job, [name], override=args.override)
    write_abacus_job(
        prepared,
        structure,
        job,
        job / name,
        stru_filename=stru_filename,
        kpoint=kpoint,
    )
    write_manifest(
        job,
        "dielectric",
        tasks=[name],
        name=name,
        stru_filename=stru_filename,
        kpoint=kpoint,
        matrix_keywords=list(MATRIX_KEYWORDS),
        suffix=str(prepared.get("suffix", "ABACUS")),
        calculation=str(prepared.get("calculation", "scf")),
        basis_type=str(prepared.get("basis_type", "lcao")),
    )

    print(f"  job: {job}")
    print(f"  prepared {name}")
    print(f"  calculation: {prepared.get('calculation', 'scf')}")
    print(f"  matrices: {', '.join(MATRIX_KEYWORDS)} (symmetry 0)")
    print(
        f"  run {name}/INPUT, copy the OUT.* matrices back, then run "
        f"'abacustools workflow dielectric postprocess -j {job}'"
    )
    return 0


def _task_directory(job: Path, name: Optional[str]) -> Path:
    """Return the generated calculation a request refers to."""
    if name:
        task = job / _generated_name(name)
        if not task.is_dir():
            raise RuntimeError(f"generated calculation does not exist: {task}")
        return task
    if not workflow_manifest_path(job, "dielectric").is_file():
        # A job that already dumped its matrices can be read in place.
        return job
    manifest = read_manifest(job, "dielectric", [])
    tasks = manifest.get("tasks") or []
    if len(tasks) != 1:
        raise RuntimeError(
            "the dielectric workflow manifest does not name a single calculation"
        )
    task = job / str(tasks[0])
    if not task.is_dir():
        raise RuntimeError(f"generated calculation does not exist: {task}")
    return task


def _resolve(job: Path, value: str) -> Path:
    """Resolve an output path, which is relative to the job by default."""
    path = Path(value)
    return path if path.is_absolute() else job / path


def _work_directory(task: Path, value: str) -> Path:
    """Resolve the pyatb working directory below the calculation."""
    path = Path(value)
    return path if path.is_absolute() else task / path


def _print_tensor(tensor: np.ndarray) -> None:
    """Print the nine components of a dielectric tensor.

    The rounding is display only: the numerically exhausted off diagonal
    elements of a symmetric crystal are noise around zero, and printing them
    as a signed zero would read as a small anisotropy.
    """
    for row in tensor:
        print("    " + " ".join(f"{round(float(value), 8) + 0.0: .8f}" for value in row))


def postprocess(args: argparse.Namespace) -> int:
    """Evaluate the Kubo-Greenwood sum and report the dielectric tensor."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    task = _task_directory(job, args.task)
    _, _, structure = read_job_structure(task)
    parameters = dielectric_job_parameters(task)
    workdir = _work_directory(task, args.workdir)
    # Copies the matrices and reports the one that is missing.
    copy_matrices(task, workdir)

    omega = (float(args.omega[0]), float(args.omega[1]))
    grid = [int(value) for value in args.grid]
    warning = omega_window_warning(omega)
    if warning:
        print(f"  warning: {warning}")

    summary = dielectric_tensor(
        workdir,
        structure,
        nspin=int(parameters["nspin"]),
        fermi_energy=float(parameters["fermi_energy"]),
        occ_band=int(parameters["occ_band"]),
        grid=grid,
        max_kpoint_num=int(args.max_kpoint_num),
        omega=omega,
        domega=float(args.domega),
        eta=float(args.eta),
        command=args.pyatb_command,
    )

    tensor = np.asarray(summary["tensor"], dtype=float)
    result: Dict[str, Any] = {
        "workflow": "dielectric",
        "method": "pyatb Kubo-Greenwood sum over the ABACUS LCAO matrices",
        "quantity": "clamped ion electronic dielectric tensor",
        "units": {"tensor": _TENSOR_UNIT},
        "task": task.name,
        "tensor": tensor.tolist(),
        "isotropic": bool(summary["isotropic"]),
        "diagonal_mean": float(summary["diagonal_mean"]),
        "diagonal_spread": float(summary["diagonal_spread"]),
        "max_off_diagonal": float(summary["max_off_diagonal"]),
        "grid": grid,
        "omega_ev": [omega[0], omega[1]],
        "domega_ev": float(args.domega),
        "eta_ev": float(args.eta),
        "nspin": int(parameters["nspin"]),
        "occ_band": int(parameters["occ_band"]),
        "fermi_energy_ev": float(parameters["fermi_energy"]),
        "pyatb_version": summary.get("pyatb_version"),
        "source_log": parameters["log"],
        # The tensor is a bulk property, so a cell of the same material at a
        # different volume is still the right reference; the cell is recorded
        # as provenance for the reader rather than checked.
        "source_structure": {
            "natoms": int(structure.natoms),
            "volume_angstrom3": float(
                abs(np.linalg.det(np.asarray(structure.cell, dtype=float)))
            ),
        },
        "workdir": str(workdir),
    }
    output = _resolve(job, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"  job: {job}")
    print(f"  task: {task.name}")
    print(
        f"  occupied bands: {result['occ_band']}, "
        f"spin channels: {result['nspin']}"
    )
    print(f"  grid: {' '.join(str(value) for value in grid)}")
    print(
        f"  photon energy window: {omega[0]:g} - {omega[1]:g} eV "
        f"(step {result['domega_ev']:g} eV, broadening {result['eta_ev']:g} eV)"
    )
    print(f"  dielectric tensor ({_TENSOR_UNIT}):")
    _print_tensor(tensor)
    print(f"  isotropic average: {result['diagonal_mean']:.8f} ({_TENSOR_UNIT})")
    if not result["isotropic"]:
        print(
            f"  anisotropy: diagonal spread {result['diagonal_spread']:.8f}, "
            f"largest off diagonal element {result['max_off_diagonal']:.8f}"
        )
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the dielectric preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "dielectric",
        "Calculate the electronic dielectric tensor with pyatb.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
