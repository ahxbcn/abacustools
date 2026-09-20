"""The ``abacustools workflow energy-strain`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.data.elastic import (
    deformation_gradient,
    elastic_moduli,
    energy_strain_patterns,
    fit_energy_strain,
    independent_component_count,
    independent_components,
    point_group_operations,
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


#: Strain amplitudes of every pattern, as multiples of the largest one.
_AMPLITUDES = (-1.0, -0.5, 0.5, 1.0)

#: Largest strain along a pattern, dimensionless.
DEFAULT_STRAIN = 0.01

#: Symmetry tolerance of the reference cell, in Angstrom.
DEFAULT_SYMPREC = 1.0e-2


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the preparation stage."""
    parser.add_argument(
        "-j",
        "--job",
        type=Path,
        required=True,
        help="ABACUS input directory used to prepare the strained cells.",
    )
    parser.add_argument(
        "--strain",
        type=float,
        default=DEFAULT_STRAIN,
        help=f"Largest strain along a pattern, default: {DEFAULT_STRAIN}.",
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
    """Register arguments for the postprocessing stage."""
    parser.add_argument(
        "-j",
        "--job",
        type=Path,
        required=True,
        help="Directory containing the prepared strained calculations.",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="energy_strain_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=DEFAULT_SYMPREC,
        help=f"Symmetry tolerance of the reference cell, default: {DEFAULT_SYMPREC}.",
    )


def _structure_symmetry(structure, symprec: float) -> dict[str, Any]:
    """Return the symmetry block recorded for a reference structure."""
    from abacustools.data.symmetry import crystallographic_symmetry

    analysis = crystallographic_symmetry(structure, symprec=symprec)
    if not analysis.get("available"):
        raise RuntimeError(
            "the symmetry of the reference cell could not be determined: "
            f"{analysis.get('error', 'unknown reason')}"
        )
    rotations = point_group_operations(structure, symprec=symprec)
    return {
        "point_group": analysis["point_group"],
        "schoenflies": analysis.get("schoenflies"),
        "space_group_symbol": analysis["space_group_symbol"],
        "space_group_number": analysis["space_group_number"],
        "crystal_system": analysis["crystal_system"],
        "operations": int(len(rotations)),
        "independent_constants": independent_component_count(rotations),
        "symprec": float(symprec),
    }


def _strained_structure(structure, strain):
    """Return a copy of the structure with a Lagrangian strain applied."""
    deformed = deepcopy(structure)
    gradient = deformation_gradient(strain)
    deformed.cell = (np.asarray(structure.cell, dtype=float) @ gradient.T).tolist()
    deformed.coords_direct = list(structure.coords_direct)
    return deformed


def prepare(args: argparse.Namespace) -> int:
    """Prepare the reference and the strained calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    if not np.isfinite(args.strain) or args.strain <= 0.0:
        raise ValueError("the strain must be a positive finite number")

    inputs, stru_filename, structure = read_job_structure(job)

    strained_inputs = deepcopy(inputs)
    strained_inputs["calculation"] = "scf" if args.norelax else "relax"
    strained_inputs["cal_stress"] = 1
    kpoint_file = kpoint_filename(job, inputs)

    symmetry = _structure_symmetry(structure, DEFAULT_SYMPREC)
    rotations = point_group_operations(structure, symprec=DEFAULT_SYMPREC)
    patterns = energy_strain_patterns(rotations)

    states = []
    for pattern in patterns:
        for amplitude in _AMPLITUDES:
            states.append(np.asarray(pattern, dtype=float) * amplitude * args.strain)
    task_names = ["org"] + [f"strained_{index:02d}" for index in range(len(states))]
    clear_generated_jobs(job, task_names, override=args.override)
    print(f"  job: {job}")
    print(f"  largest strain along a pattern: {args.strain}")
    print(f"  calculation: {strained_inputs['calculation']}")
    print(f"  strain patterns: {len(patterns)}")
    print(
        f"  point group: {symmetry['point_group']} "
        f"({symmetry['space_group_symbol']}, "
        f"space group {symmetry['space_group_number']})"
    )
    print(f"  independent elastic constants: {symmetry['independent_constants']}")

    write_abacus_job(
        strained_inputs,
        structure,
        job,
        job / "org",
        stru_filename=stru_filename,
        kpoint=kpoint_file,
    )
    print("  prepared org")

    strain_metadata = []
    strained_paths = []
    for index, strain in enumerate(states):
        name = f"strained_{index:02d}"
        write_abacus_job(
            strained_inputs,
            _strained_structure(structure, strain),
            job,
            job / name,
            stru_filename=stru_filename,
            kpoint=kpoint_file,
        )
        strain_metadata.append([float(value) for value in strain])
        strained_paths.append(name)
        print(f"  prepared {name}")

    write_manifest(
        job,
        "energy-strain",
        tasks=task_names,
        strained_paths=strained_paths,
        strains=strain_metadata,
        patterns=[[float(value) for value in pattern] for pattern in patterns],
        amplitude=float(args.strain),
        symmetry=symmetry,
        norelax=bool(args.norelax),
    )
    return 0


def _read_energy(
    job: Path, version: str, require_relaxation: bool
) -> float:
    """Read one converged ABACUS total energy, in eV."""
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(
        job,
        param_names=["energy", "converged", "relax_converged"],
        version=version,
    )
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    if require_relaxation and not result["relax_converged"]:
        raise RuntimeError(f"ionic relaxation did not converge: {job}")
    energy = result["energy"]
    if energy is None or not np.isfinite(float(energy)):
        raise RuntimeError(f"the total energy was not found in the output: {job}")
    return float(energy)


def postprocess(args: argparse.Namespace) -> int:
    """Fit elastic constants from the curvature of the total energy."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    manifest = read_manifest(job, "energy-strain", ("org",))
    strains = manifest.get("strains")
    strained_paths = manifest.get("strained_paths")
    if not isinstance(strains, list) or not strains:
        raise RuntimeError("energy-strain manifest must contain strain states")
    if not isinstance(strained_paths, list) or len(strained_paths) != len(strains):
        raise RuntimeError("energy-strain manifest has invalid strained paths")

    require_relaxation = not bool(manifest.get("norelax", False))
    symmetry = _structure_symmetry_from_manifest(job, manifest, args.symprec)
    print(f"  job: {job}")
    print(
        f"  point group: {symmetry['point_group']} "
        f"({symmetry.get('space_group_symbol')}, "
        f"space group {symmetry.get('space_group_number')})"
    )
    print(f"  independent elastic constants: {symmetry['independent_constants']}")
    print(f"  strained cells: {len(strains)}")

    energies = [
        _read_energy(job / name, args.version, require_relaxation)
        for name in strained_paths
    ]
    _, _, structure = read_job_structure(job)
    volume = float(abs(np.linalg.det(np.asarray(structure.cell, dtype=float))))
    rotations = point_group_operations(structure, symprec=args.symprec)
    tensor, pattern_stress, residual = fit_energy_strain(
        np.asarray(strains, dtype=float),
        np.asarray(energies, dtype=float),
        volume,
        rotations,
    )
    components = independent_components(tensor, rotations)
    result = {
        "elastic_tensor": tensor.tolist(),
        "independent_constants": components,
        "symmetry": symmetry,
        "fit": {
            "method": "energy-strain",
            "strained_cells": len(strains),
            "patterns": int(len(manifest.get("patterns", []))),
            "amplitude": manifest.get("amplitude"),
            "energy_residual": residual,
        },
        "reference_stress_on_patterns": pattern_stress.tolist(),
        "reference_volume": volume,
        "energy_unit": "eV",
        "stress_unit": "GPa",
        **elastic_moduli(tensor),
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("  elastic tensor (GPa):")
    for row in tensor:
        print("    " + " ".join(f"{value: .8f}" for value in row))
    print(
        "  independent constants (GPa): "
        + ", ".join(f"{name} = {value:.6f}" for name, value in components.items())
    )
    print(f"  reference stress on the patterns (GPa): {np.round(pattern_stress, 5).tolist()}")
    print(f"  energy residual: {residual:.3e} eV")
    for name in ("bulk_modulus", "shear_modulus", "young_modulus"):
        print(f"  {name}: {result[name]:.8f} GPa")
    # The Poisson ratio is dimensionless, so it carries no unit.
    print(f"  poisson_ratio: {result['poisson_ratio']:.8f}")
    print(f"  results: {output}")
    return 0


def _structure_symmetry_from_manifest(
    job: Path, manifest: dict[str, Any], symprec: float
) -> dict[str, Any]:
    """Return the symmetry of the reference cell.

    The preparation stage records it in the manifest; a manifest written
    without that information is completed here from the reference structure.
    """
    recorded = manifest.get("symmetry")
    if isinstance(recorded, dict) and recorded.get("point_group"):
        return dict(recorded)
    _, _, structure = read_job_structure(job)
    return _structure_symmetry(structure, symprec)


def register_parser(subparsers) -> None:
    """Register the energy-strain preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "energy-strain",
        "Calculate elastic constants from the curvature of the total energy.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
