"""The ``abacustools workflow elastic`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from abacustools.data.elastic import (
    elastic_moduli,
    fit_stress_strain,
    fit_independent_stress_strain,
    independent_component_count,
    independent_components,
    independent_strain_modes,
    point_group_operations,
    stress_voigt,
    symmetrize_elastic_tensor,
    symmetrization_residual,
)
from abacustools.data.symmetry import crystallographic_symmetry
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

#: Names of the six Voigt strain directions.
_VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")

#: Symmetry tolerance used to decide which components the elastic tensor may
#: have, in Angstrom.  A relaxed cell is only symmetric up to the tolerance of
#: the relaxation that produced it, which is several 1e-3 Angstrom, so the
#: default follows the tolerance pymatgen uses for a structure analysis.
DEFAULT_SYMPREC = 1.0e-2


def _voigt_labels(modes: Iterable[int]) -> list[str]:
    """Return the names of a list of Voigt strain directions."""
    return [_VOIGT_LABELS[int(mode)] for mode in modes]


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
        "--strains",
        choices=("full", "independent"),
        default="full",
        help=(
            "Strain set of the generated jobs. 'full' applies all six Voigt "
            "strain directions, 'independent' applies one representative per "
            "symmetry orbit of strain directions, which is enough to "
            "determine the independent constants, default: full."
        ),
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
    parser.add_argument(
        "--symprec",
        type=float,
        default=DEFAULT_SYMPREC,
        help=f"Symmetry tolerance of the reference cell, default: {DEFAULT_SYMPREC}.",
    )
    parser.add_argument(
        "--symmetrize",
        dest="symmetrize",
        action="store_true",
        default=True,
        help="Symmetrize the fitted tensor with the crystal symmetry, the default.",
    )
    parser.add_argument(
        "--no-symmetrize",
        dest="symmetrize",
        action="store_false",
        help="Keep the unconstrained fit of the 36 components.",
    )
    parser.add_argument(
        "--fit",
        choices=("full", "independent"),
        default="full",
        help=(
            "Fit every component and symmetrize afterwards ('full'), or fit "
            "only the independent constants ('independent'). A job prepared "
            "with --strains independent always uses the independent fit, "
            "default: full."
        ),
    )


def _validate_strain_amounts(norm: float, shear: float) -> None:
    """Validate strain amplitudes before constructing deformation matrices."""
    for name, amount in (("normal", norm), ("shear", shear)):
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError(f"{name} strain must be a positive finite number")
    if shear >= 0.5:
        raise ValueError("shear strain must be smaller than 0.5")


def _pymatgen_deformations(structure, norm: float, shear: float, modes=None):
    """Generate single-component deformations with pymatgen's elasticity API.

    Args:
        structure: Structure to deform.
        norm: Largest normal strain.
        shear: Largest shear strain.
        modes: Voigt indices of the strain directions to keep.  ``None`` keeps
            all six, the order of the returned states follows the Voigt index.

    Returns:
        List of ``(deformed structure, strain)`` pairs.
    """
    from pymatgen.analysis.elasticity.strain import DeformedStructureSet, Strain

    structure = structure.to("pymatgen")
    deformed_set = DeformedStructureSet(
        structure,
        norm_strains=(-norm, -0.5 * norm, 0.5 * norm, norm),
        shear_strains=(-shear, -0.5 * shear, 0.5 * shear, shear),
        symmetry=False,
    )
    wanted = None if modes is None else {int(mode) for mode in modes}
    states = []
    for index, (deformed, deformation) in enumerate(
        zip(deformed_set, deformed_set.deformations)
    ):
        # DeformedStructureSet walks the three normal directions and then the
        # three shear directions, four amplitudes each.
        if wanted is not None and index // 4 not in wanted:
            continue
        states.append((deformed, Strain.from_deformation(deformation)))
    return states


def _deformed_structure(structure, pymatgen_structure):
    """Convert a pymatgen-deformed structure while retaining ABACUS metadata."""
    deformed = deepcopy(structure)
    deformed.cell = pymatgen_structure.lattice.matrix.tolist()
    deformed.coords_direct = pymatgen_structure.frac_coords.tolist()
    return deformed


def _structure_symmetry(structure, symprec: float) -> dict[str, Any]:
    """Return the symmetry block recorded for a reference structure."""
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
        "lattice_type": analysis.get("lattice_type"),
        "inversion_symmetry": analysis.get("inversion_symmetry"),
        "operations": int(len(rotations)),
        "independent_constants": independent_component_count(rotations),
        "symprec": float(symprec),
    }


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

    symmetry = _structure_symmetry(structure, DEFAULT_SYMPREC)
    rotations = point_group_operations(structure, symprec=DEFAULT_SYMPREC)
    if args.strains == "independent":
        strain_modes = independent_strain_modes(rotations)
    else:
        strain_modes = list(range(6))
    generated = _pymatgen_deformations(
        structure, args.norm, args.shear, modes=strain_modes
    )
    task_names = ["org"] + [
        f"deformed_{index:02d}" for index in range(len(generated))
    ]
    clear_generated_jobs(job, task_names, override=args.override)
    print(f"  job: {job}")
    print(f"  normal strain: {args.norm}")
    print(f"  shear strain: {args.shear}")
    print(f"  calculation: {elastic_inputs['calculation']}")
    print(
        f"  strain modes: {len(strain_modes)} of 6 "
        f"({', '.join(_voigt_labels(strain_modes))})"
    )
    print(f"  strained jobs: {len(generated)}")
    print(
        f"  point group: {symmetry['point_group']} "
        f"({symmetry['space_group_symbol']}, "
        f"space group {symmetry['space_group_number']})"
    )
    print(f"  independent elastic constants: {symmetry['independent_constants']}")

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
        tasks=task_names,
        deformed_paths=deformed_paths,
        strains=strain_metadata,
        strain_modes=strain_modes,
        strains_mode=args.strains,
        symmetry=symmetry,
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


def _fit_tensor(
    strains: Iterable[dict[str, Any]],
    stresses: Iterable[np.ndarray],
    equilibrium_stress: np.ndarray,
) -> np.ndarray:
    """Fit the unconstrained 6x6 stress-strain tensor in GPa."""
    strain_values, stress_values = _strain_stress_values(
        strains, stresses, equilibrium_stress
    )
    if strain_values.shape[0] < 24:
        raise ValueError(
            "the unconstrained fit needs the full strain set; prepare the job "
            "with --strains full or postprocess it with --fit independent"
        )
    return fit_stress_strain(strain_values, stress_values)


def _strain_stress_values(
    strains: Iterable[dict[str, Any]],
    stresses: Iterable[np.ndarray],
    equilibrium_stress: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the six-component strain and stress values of the workflow."""
    from pymatgen.analysis.elasticity.strain import Strain

    strain_values = np.asarray(
        [Strain.from_dict(item).voigt for item in strains], dtype=float
    )
    stress_values = np.asarray(
        [stress_voigt(stress - equilibrium_stress) for stress in stresses],
        dtype=float,
    )
    if strain_values.ndim != 2 or strain_values.shape[1] != 6:
        raise ValueError("the manifest holds invalid strain states")
    if stress_values.shape != strain_values.shape:
        raise ValueError("the number of stresses does not match the strains")
    return strain_values, stress_values


def _symmetry_block(job: Path, manifest: dict[str, Any], symprec: float) -> dict[str, Any]:
    """Return the symmetry of the reference cell.

    The preparation stage records it in the manifest; a manifest written
    before that information existed is completed here from the reference
    structure.
    """
    recorded = manifest.get("symmetry")
    if isinstance(recorded, dict) and recorded.get("point_group"):
        return dict(recorded)
    _, _, structure = read_job_structure(job)
    return _structure_symmetry(structure, symprec)


def postprocess(args: argparse.Namespace) -> int:
    """Fit elastic constants from the prepared stress-strain calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    # The strained jobs depend on the strain set the preparation stage chose,
    # so only the unstrained cell is required by name.
    manifest = read_manifest(job, "elastic", ("org",))
    strains = manifest.get("strains")
    deformed_paths = manifest.get("deformed_paths")
    if not isinstance(strains, list) or not strains:
        raise RuntimeError("elastic workflow manifest must contain strain states")
    if not isinstance(deformed_paths, list) or len(deformed_paths) != len(strains):
        raise RuntimeError("elastic workflow manifest has invalid deformed paths")
    strain_modes = manifest.get("strain_modes")
    if not isinstance(strain_modes, list) or not strain_modes:
        # Manifests written before the strain selection existed hold all six.
        strain_modes = list(range(6))

    require_relaxation = not bool(manifest.get("norelax", False))
    symmetry = _symmetry_block(job, manifest, args.symprec)
    print(f"  job: {job}")
    print(
        f"  point group: {symmetry['point_group']} "
        f"({symmetry.get('space_group_symbol')}, "
        f"space group {symmetry.get('space_group_number')})"
    )
    print(f"  independent elastic constants: {symmetry['independent_constants']}")
    print(
        f"  strain modes: {len(strain_modes)} of 6 "
        f"({', '.join(_voigt_labels(strain_modes))}), "
        f"{len(strains)} strained jobs"
    )
    equilibrium_stress = _read_stress(job / "org", args.version, require_relaxation)
    deformed_stresses: list[np.ndarray] = [
        _read_stress(job / name, args.version, require_relaxation)
        for name in deformed_paths
    ]
    _, _, structure = read_job_structure(job)
    rotations = point_group_operations(structure, symprec=args.symprec)

    method = args.fit
    if len(strain_modes) < 6 and method == "full":
        print(
            "  the strain set covers the independent directions only; "
            "fitting the independent constants"
        )
        method = "independent"
    if method == "independent":
        strain_values, stress_values = _strain_stress_values(
            strains, deformed_stresses, equilibrium_stress
        )
        tensor = fit_independent_stress_strain(
            strain_values, stress_values, rotations
        )
        raw = tensor
        residual = 0.0
        print("  fit: independent constants")
    else:
        raw = _fit_tensor(strains, deformed_stresses, equilibrium_stress)
        if args.symmetrize:
            tensor = symmetrize_elastic_tensor(raw, rotations)
        else:
            tensor = raw
        residual = symmetrization_residual(raw, tensor)
    result = {
        "elastic_tensor": tensor.tolist(),
        "elastic_tensor_raw": raw.tolist(),
        "symmetrization_residual": residual,
        "independent_constants": independent_components(tensor, rotations),
        "symmetry": symmetry,
        "fit": {
            "method": method,
            "strain_modes": [int(mode) for mode in strain_modes],
            "strained_jobs": len(strains),
            "independent_constants": len(
                independent_components(tensor, rotations)
            ),
        },
        **elastic_moduli(tensor),
        "stress_unit": "GPa",
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    components = result["independent_constants"]
    heading = "fitted tensor (raw, GPa)" if method == "full" else "fitted tensor (GPa)"
    print(f"  {heading}:")
    for row in raw:
        print("    " + " ".join(f"{value: .8f}" for value in row))
    if method == "full" and args.symmetrize:
        print("  symmetrized tensor (GPa):")
        for row in tensor:
            print("    " + " ".join(f"{value: .8f}" for value in row))
        print(f"  symmetrization residual: {residual:.8f} GPa")
    if method == "full" and not args.symmetrize:
        print("  the tensor is kept as fitted; no symmetrization was applied")
    print(
        "  independent constants (GPa): "
        + ", ".join(f"{name} = {value:.6f}" for name, value in components.items())
    )
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
