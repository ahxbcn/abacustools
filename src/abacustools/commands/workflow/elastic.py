"""The ``abacustools workflow elastic`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

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


_ELASTIC_TASKS = ("org",) + tuple(f"deformed_{index:02d}" for index in range(24))


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the elastic preparation stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="ABACUS input directory used to prepare elastic calculations.",
    )
    parser.add_argument(
        "--norm",
        type=float,
        default=0.01,
        help="Maximum normal strain, default: 0.01.",
    )
    parser.add_argument(
        "--shear",
        type=float,
        default=0.01,
        help="Maximum shear strain, default: 0.01.",
    )
    parser.add_argument(
        "--norelax",
        action="store_true",
        help="Use fixed-ion SCF calculations instead of ionic relaxation.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace existing generated workflow directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the elastic postprocessing stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="Directory containing the prepared elastic calculations.",
    )
    parser.add_argument(
        "-v", "--version",
        default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-o", "--output",
        default="elastic_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )


def _validate_strain_amounts(norm: float, shear: float) -> None:
    """Validate strain amplitudes before constructing deformation matrices."""
    for name, amount in (("normal", norm), ("shear", shear)):
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError(f"{name} strain must be a positive finite number")
    if shear >= 0.5:
        raise ValueError("shear strain must be smaller than 0.5")


def _pymatgen_deformations(structure, norm: float, shear: float):
    """Generate independent deformations with pymatgen's elasticity API."""
    from pymatgen.analysis.elasticity.strain import DeformedStructureSet, Strain

    structure = structure.to("pymatgen")
    deformed_set = DeformedStructureSet(
        structure,
        norm_strains=(-norm, -0.5 * norm, 0.5 * norm, norm),
        shear_strains=(-shear, -0.5 * shear, 0.5 * shear, shear),
        symmetry=False,
    )
    return [
        (deformed, Strain.from_deformation(deformation))
        for deformed, deformation in zip(
            deformed_set, deformed_set.deformations
        )
    ]


def _deformed_structure(structure, pymatgen_structure):
    """Convert a pymatgen-deformed structure while retaining ABACUS metadata."""
    deformed = deepcopy(structure)
    deformed.cell = pymatgen_structure.lattice.matrix.tolist()
    deformed.coords_direct = pymatgen_structure.frac_coords.tolist()
    return deformed


def prepare(args: argparse.Namespace) -> int:
    """Prepare the equilibrium and independently strained calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _validate_strain_amounts(args.norm, args.shear)

    inputs, stru_filename, structure = read_job_structure(job)

    elastic_inputs = deepcopy(inputs)
    elastic_inputs["calculation"] = "scf" if args.norelax else "relax"
    elastic_inputs["cal_stress"] = 1
    kpoint_file = kpoint_filename(job, inputs)

    generated = _pymatgen_deformations(structure, args.norm, args.shear)
    clear_generated_jobs(job, _ELASTIC_TASKS, override=args.override)
    print(f"  job: {job}")
    print(f"  normal strain: {args.norm}")
    print(f"  shear strain: {args.shear}")
    print(f"  calculation: {elastic_inputs['calculation']}")

    write_abacus_job(
        elastic_inputs,
        structure,
        job,
        job / "org",
        stru_filename=stru_filename,
        kpoint=kpoint_file,
    )
    print("  prepared org")

    strain_metadata = []
    deformed_paths = []
    for index, (pymatgen_structure, strain) in enumerate(generated):
        name = f"deformed_{index:02d}"
        write_abacus_job(
            elastic_inputs,
            _deformed_structure(structure, pymatgen_structure),
            job,
            job / name,
            stru_filename=stru_filename,
            kpoint=kpoint_file,
        )
        strain_metadata.append(strain.as_dict())
        deformed_paths.append(name)
        print(f"  prepared {name}")

    write_manifest(
        job,
        "elastic",
        tasks=list(_ELASTIC_TASKS),
        deformed_paths=deformed_paths,
        strains=strain_metadata,
        norm=float(args.norm),
        shear=float(args.shear),
        norelax=bool(args.norelax),
    )
    return 0


def _read_stress(job: Path, version: str, require_relaxation: bool) -> np.ndarray:
    """Read one converged ABACUS stress tensor and convert kBar to GPa."""
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(
        job,
        param_names=["stress", "converged", "relax_converged"],
        version=version,
    )
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    if require_relaxation and not result["relax_converged"]:
        raise RuntimeError(f"ionic relaxation did not converge: {job}")
    if result["stress"] is None:
        raise RuntimeError(f"stress was not found in the output: {job}")

    stress = np.asarray(result["stress"], dtype=float)
    if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
        raise RuntimeError(f"invalid stress tensor in the output: {job}")
    # ABACUS reports kBar with the opposite sign to the convention used here.
    return -0.1 * stress


def _fit_elastic_tensor(
    strains: Iterable[dict[str, Any]],
    stresses: Iterable[np.ndarray],
    equilibrium_stress: np.ndarray,
) -> np.ndarray:
    """Fit the 6x6 stress-strain tensor in GPa."""
    from pymatgen.analysis.elasticity.strain import Strain

    strain_values = np.asarray(
        [Strain.from_dict(item).voigt for item in strains], dtype=float
    )
    stress_values = np.asarray(
        [_stress_voigt(stress - equilibrium_stress) for stress in stresses],
        dtype=float,
    )
    if strain_values.shape != (24, 6) or stress_values.shape != (24, 6):
        raise ValueError("elastic fitting requires 24 six-component strain/stress values")

    tensor = np.zeros((6, 6), dtype=float)
    for component in range(6):
        mask = np.abs(strain_values[:, component]) > 0
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"insufficient strain data for component {component}")
        design = np.column_stack((strain_values[mask, component], np.ones(np.count_nonzero(mask))))
        for stress_component in range(6):
            tensor[component, stress_component] = np.linalg.lstsq(
                design,
                stress_values[mask, stress_component],
                rcond=None,
            )[0][0]
    return tensor


def _stress_voigt(stress: np.ndarray) -> list[float]:
    """Convert a symmetric stress matrix to Voigt notation."""
    return [
        float(stress[0, 0]),
        float(stress[1, 1]),
        float(stress[2, 2]),
        float(stress[1, 2]),
        float(stress[0, 2]),
        float(stress[0, 1]),
    ]


def _elastic_moduli(tensor: np.ndarray) -> dict[str, float]:
    """Calculate Voigt bulk, shear, Young's and Poisson moduli in GPa."""
    diagonal = np.trace(tensor[:3, :3])
    off_diagonal = tensor[0, 1] + tensor[0, 2] + tensor[1, 2]
    shear = tensor[3, 3] + tensor[4, 4] + tensor[5, 5]
    bulk_modulus = (diagonal + 2.0 * off_diagonal) / 9.0
    shear_modulus = (diagonal - off_diagonal + 3.0 * shear) / 15.0
    denominator = 3.0 * bulk_modulus + shear_modulus
    if denominator == 0:
        raise ValueError("cannot calculate Young's modulus from the fitted tensor")
    young_modulus = 9.0 * bulk_modulus * shear_modulus / denominator
    poisson_ratio = (3.0 * bulk_modulus - 2.0 * shear_modulus) / (
        2.0 * denominator
    )
    return {
        "bulk_modulus": float(bulk_modulus),
        "shear_modulus": float(shear_modulus),
        "young_modulus": float(young_modulus),
        "poisson_ratio": float(poisson_ratio),
    }


def postprocess(args: argparse.Namespace) -> int:
    """Fit elastic constants from the prepared stress-strain calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    manifest = read_manifest(job, "elastic", _ELASTIC_TASKS)
    strains = manifest.get("strains")
    deformed_paths = manifest.get("deformed_paths")
    if not isinstance(strains, list) or len(strains) != 24:
        raise RuntimeError("elastic workflow manifest must contain 24 strain states")
    if deformed_paths != list(_ELASTIC_TASKS[1:]):
        raise RuntimeError("elastic workflow manifest has invalid deformed paths")

    require_relaxation = not bool(manifest.get("norelax", False))
    print(f"  job: {job}")
    equilibrium_stress = _read_stress(job / "org", args.version, require_relaxation)
    deformed_stresses = [
        _read_stress(job / name, args.version, require_relaxation)
        for name in deformed_paths
    ]
    tensor = _fit_elastic_tensor(strains, deformed_stresses, equilibrium_stress)
    result = {
        "elastic_tensor": tensor.tolist(),
        **_elastic_moduli(tensor),
        "stress_unit": "GPa",
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("  elastic tensor (GPa):")
    for row in tensor:
        print("    " + " ".join(f"{value: .8f}" for value in row))
    for name in ("bulk_modulus", "shear_modulus", "young_modulus"):
        print(f"  {name}: {result[name]:.8f} GPa")
    # The Poisson ratio is dimensionless, so it carries no unit.
    print(f"  poisson_ratio: {result['poisson_ratio']:.8f}")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the elastic preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "elastic",
        "Calculate elastic constants with the stress-strain method.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
