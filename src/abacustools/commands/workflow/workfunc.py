"""The ``abacustools workflow workfunc`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.data.dimensionality import largest_vacuum, vacuum_gaps
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


_WORKFUNC_DIRECTORY = "workfunc_job"
_DIRECTIONS = ("a", "b", "c")


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for work-function preparation."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="ABACUS input directory used to prepare the work-function calculation.",
    )
    parser.add_argument(
        "--vacuum",
        choices=_DIRECTIONS + ("auto",),
        default="auto",
        help="Vacuum direction (a, b, c, or auto), default: auto.",
    )
    parser.add_argument(
        "--dipole-corr",
        action="store_true",
        help="Enable ABACUS dipole correction along the vacuum direction.",
    )
    parser.add_argument(
        "--use-empty-atom",
        action="store_true",
        help="Add empty atoms in the vacuum region on both sides of the slab.",
    )
    parser.add_argument(
        "--empty-atom-elem",
        help="Element used for empty atoms; by default the first element is used.",
    )
    parser.add_argument(
        "--empty-atom-height",
        type=float,
        default=2.0,
        help="Distance of the empty-atom layer from each surface in Angstrom, default: 2.0.",
    )
    parser.add_argument(
        "--empty-atom-dist",
        type=float,
        default=2.0,
        help="Spacing between empty atoms parallel to the surface, default: 2.0 Angstrom.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace the existing generated work-function directory.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for work-function postprocessing."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="Directory containing the prepared work-function calculation.",
    )
    parser.add_argument(
        "-v", "--version",
        default=default_version(),
        help="ABACUS version used for the calculation.",
    )
    parser.add_argument(
        "--vacuum",
        choices=_DIRECTIONS + ("auto",),
        default="auto",
        help="Vacuum direction (a, b, c, or auto), default: auto.",
    )
    parser.add_argument(
        "--threshold",
        "--thr",
        dest="threshold",
        type=float,
        default=0.01,
        help="Potential slope threshold for identifying plateaus, default: 0.01 eV/Angstrom.",
    )
    parser.add_argument(
        "-o", "--output",
        default="workfunc_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )


def _validate_positive(value: float, name: str) -> None:
    """Validate a finite positive parameter."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _vacuum_for_direction(structure, axis: int) -> tuple[float, float, float]:
    """Find the periodic gap and boundaries along one lattice direction."""
    gap = vacuum_gaps(structure)[axis]
    return gap["thickness"], gap["top"], gap["bottom"]


def _largest_vacuum_direction(structure) -> tuple[str, float, float, float]:
    """Find the largest periodic gap and its two boundaries."""
    gap = largest_vacuum(structure)
    if gap["thickness"] < 4.0:
        print(f"  warning: largest vacuum is only {gap['thickness']:.6f} Angstrom")
    return gap["direction"], gap["thickness"], gap["top"], gap["bottom"]


def _choose_vacuum_direction(requested: str, detected: str, input_direction: str | None = None) -> str:
    """Choose a vacuum direction from the CLI, INPUT and structure."""
    if requested not in _DIRECTIONS + ("auto",):
        raise ValueError(f"invalid vacuum direction: {requested}")
    if requested != "auto":
        if requested != detected:
            print(f"  warning: requested vacuum direction {requested} differs from detected {detected}")
        return requested
    if input_direction is not None:
        if input_direction != detected:
            print(f"  warning: INPUT vacuum direction {input_direction} differs from detected {detected}")
        return input_direction
    return detected


def _add_empty_atom_layer(
    structure,
    direction: str,
    height: float,
    distance: float,
    element: str | None,
):
    """Add empty atoms near both surfaces along the selected direction."""
    _validate_positive(height, "empty_atom_height")
    _validate_positive(distance, "empty_atom_dist")
    from abacustools.io.stru import AbacusATOM

    result = deepcopy(structure)
    if element is None:
        element = result.elements[0]
    if element not in result.elements:
        raise ValueError(f"empty atom element must be present in the structure: {element}")
    source = next(atom for atom in result.atoms if atom.element == element)
    axis = _DIRECTIONS.index(direction)
    parallel = [index for index in range(3) if index != axis]
    cell = np.asarray(result.cell, dtype=float)
    vector = cell[axis]
    unit_vector = vector / np.linalg.norm(vector)
    _, vacuum_top, vacuum_bottom = _vacuum_for_direction(result, axis)
    top_center = unit_vector * (vacuum_top - height)
    bottom_center = unit_vector * (vacuum_bottom + height)
    surface_vectors = [cell[index] for index in parallel]
    n1 = max(1, int(np.linalg.norm(surface_vectors[0]) / distance))
    n2 = max(1, int(np.linalg.norm(surface_vectors[1]) / distance))

    def offsets(vector, count):
        length = np.linalg.norm(vector)
        return [((-count + 1) / 2.0 + index) * vector / length * distance for index in range(count)]

    empty_label = f"{element}_empty"
    for center in (top_center, bottom_center):
        for offset1 in offsets(surface_vectors[0], n1):
            for offset2 in offsets(surface_vectors[1], n2):
                result.append(
                    AbacusATOM(
                        label=empty_label,
                        element=element,
                        coord=tuple(center + offset1 + offset2),
                        mass=source.mass,
                        pp=source.pp,
                        orb=source.orb,
                        paw=source.paw,
                    )
                )
    return result


def prepare(args: argparse.Namespace) -> int:
    """Prepare an ABACUS electrostatic-potential calculation."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    inputs, stru_filename, structure = read_job_structure(job)
    detected, vacuum_size, _, _ = _largest_vacuum_direction(structure)
    direction = _choose_vacuum_direction(args.vacuum, detected)

    workfunc_structure = structure
    workfunc_inputs = deepcopy(inputs)
    if args.use_empty_atom:
        workfunc_structure = _add_empty_atom_layer(
            structure,
            direction,
            args.empty_atom_height,
            args.empty_atom_dist,
            args.empty_atom_elem,
        )
        workfunc_inputs["ntype"] = len({atom.label for atom in workfunc_structure.atoms})
    workfunc_inputs["calculation"] = "scf"
    workfunc_inputs["out_pot"] = 2
    if args.dipole_corr:
        workfunc_inputs["efield_flag"] = 1
        workfunc_inputs["dip_cor_flag"] = 1
        workfunc_inputs["efield_amp"] = 0.0
        workfunc_inputs["efield_pos_max"] = None
        workfunc_inputs["efield_pos_dec"] = None
        workfunc_inputs["efield_dir"] = _DIRECTIONS.index(direction)

    clear_generated_jobs(job, [_WORKFUNC_DIRECTORY], override=args.override)
    kpoint_file = kpoint_filename(job, inputs)
    write_abacus_job(
        workfunc_inputs,
        workfunc_structure,
        job,
        job / _WORKFUNC_DIRECTORY,
        stru_filename=stru_filename,
        kpoint=kpoint_file,
    )
    write_manifest(
        job,
        "workfunc",
        tasks=[_WORKFUNC_DIRECTORY],
        vacuum_direction=direction,
        detected_vacuum_direction=detected,
        vacuum_size=float(vacuum_size),
        dipole_corr=bool(args.dipole_corr),
        use_empty_atom=bool(args.use_empty_atom),
    )
    print(f"  job: {job}")
    print(f"  prepared: {job / _WORKFUNC_DIRECTORY}")
    print(f"  vacuum direction: {direction} (detected: {detected})")
    print(f"  vacuum size: {vacuum_size:.6f} Angstrom")
    print(f"  dipole correction: {'on' if args.dipole_corr else 'off'}")
    return 0


def identify_potential_plateaus(
    averaged_potential: Any,
    cell_length: float,
    threshold: float = 0.01,
) -> list[tuple[int, int]]:
    """Identify contiguous potential plateaus using a periodic derivative."""
    _validate_positive(cell_length, "cell_length")
    _validate_positive(threshold, "threshold")
    values = np.asarray(averaged_potential, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("averaged_potential must contain at least three finite values")
    step = cell_length / len(values)
    derivatives = (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * step)
    is_plateau = np.abs(derivatives) < threshold
    ranges = []
    index = 0
    while index < len(values):
        if not is_plateau[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(values) and is_plateau[index + 1]:
            index += 1
        if index - start + 1 >= 2:
            ranges.append((start, index))
        index += 1
    if len(ranges) > 1 and ranges[0][0] == 0 and ranges[-1][1] == len(values) - 1:
        ranges[0] = (ranges[-1][0] - len(values), ranges[0][1])
        ranges.pop()
    return ranges


def calculate_work_functions(
    averaged_potential: Any,
    fermi_energy: float,
    cell_length: float,
    threshold: float = 0.01,
) -> list[dict[str, float]]:
    """Calculate work functions from vacuum plateaus in an averaged potential."""
    values = np.asarray(averaged_potential, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("averaged_potential must be a non-empty one-dimensional array")
    plateaus = identify_potential_plateaus(values, cell_length, threshold)
    npoints = len(values)
    if not plateaus:
        index = int(np.argmax(values))
        vacuum_level = float(values[index])
        return [{
            "plateau_start_fractional": index / npoints,
            "plateau_end_fractional": (index + 1) / npoints,
            "vacuum_level": vacuum_level,
            "work_function": vacuum_level - float(fermi_energy),
        }]

    results = []
    for start, end in plateaus:
        start_mod = start % npoints
        end_mod = end % npoints
        if start < 0 or end >= npoints or start_mod > end_mod:
            plateau = np.concatenate((values[start_mod:], values[:end_mod]))
        else:
            plateau = values[start_mod:end_mod]
        if len(plateau) == 0:
            plateau = values[start_mod:start_mod + 1]
        vacuum_level = float(np.mean(plateau))
        results.append({
            "plateau_start_fractional": start / npoints,
            "plateau_end_fractional": end / npoints,
            "vacuum_level": vacuum_level,
            "work_function": vacuum_level - float(fermi_energy),
        })
    return results


def _potential_file(workfunc_job: Path, inputs: dict[str, Any]) -> Path:
    """Find the electrostatic-potential cube emitted by ABACUS."""
    suffix = str(inputs.get("suffix", "ABACUS"))
    output_dir = workfunc_job / f"OUT.{suffix}"
    for filename in ("potes.cube", "ElecStaticPot.cube"):
        path = output_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"could not find an electrostatic potential cube in {output_dir}"
    )


def _plot_profile(
    coordinate: np.ndarray,
    potential: np.ndarray,
    fermi_energy: float,
    direction: str,
    results: list[dict[str, float]],
    output: Path,
) -> None:
    """Save the averaged electrostatic-potential profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(coordinate, potential, label="Average electrostatic potential")
    axis.axhline(fermi_energy, linestyle="--", color="gray", alpha=0.5, label="Fermi energy")
    axis.set_xlabel(f"Fractional coordinate along {direction}")
    axis.set_ylabel("Electrostatic potential (eV)")
    for result in results:
        midpoint = (result["plateau_start_fractional"] + result["plateau_end_fractional"]) / 2.0
        midpoint %= 1.0
        index = int(midpoint * (len(coordinate) - 1))
        axis.annotate(
            f"{result['work_function']:.2f} eV",
            xy=(coordinate[index], potential[index]),
            xytext=(coordinate[index], (potential[index] + fermi_energy) / 2.0),
            arrowprops={"arrowstyle": "<->", "color": "black"},
            ha="center",
        )
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def postprocess(args: argparse.Namespace) -> int:
    """Read an ABACUS potential and calculate the work function."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _validate_positive(args.threshold, "threshold")
    manifest = read_manifest(job, "workfunc", [_WORKFUNC_DIRECTORY])
    workfunc_job = job / _WORKFUNC_DIRECTORY
    inputs, _, structure = read_job_structure(workfunc_job)
    detected, _, _, _ = _largest_vacuum_direction(structure)
    manifest_direction = manifest.get("vacuum_direction")
    direction = _choose_vacuum_direction(args.vacuum, detected, manifest_direction)

    from abacustools.data.abacus_result import get_result_from_job
    from abacustools.data.grid import Potential

    result = get_result_from_job(
        workfunc_job,
        ["efermi", "converged"],
        args.version,
    )
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {workfunc_job}")
    if result["efermi"] is None:
        raise RuntimeError(f"Fermi energy was not found in the output: {workfunc_job}")
    potential = Potential.from_cube(str(_potential_file(workfunc_job, inputs)))
    average, coordinate = potential.profile1d(axis=direction, average=True)
    average = -np.asarray(average, dtype=float)
    axis_index = _DIRECTIONS.index(direction)
    cell_length = float(np.linalg.norm(potential.cell[axis_index]))
    work_functions = calculate_work_functions(
        average,
        float(result["efermi"]),
        cell_length,
        args.threshold,
    )

    profile_path = workfunc_job / "workfunc_profile.dat"
    with profile_path.open("w", encoding="utf-8") as stream:
        stream.write("fractional_coordinate average_potential\n")
        for coordinate_value, potential_value in zip(coordinate, average):
            stream.write(f"{coordinate_value:.8f} {potential_value:.8f}\n")
    plot_path = workfunc_job / "workfunc_potential.png"
    _plot_profile(coordinate, average, float(result["efermi"]), direction, work_functions, plot_path)

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "vacuum_direction": direction,
        "fermi_energy": float(result["efermi"]),
        "work_functions": work_functions,
        "potential_unit": "eV",
        "work_function_unit": "eV",
        "profile": str(profile_path),
        "plot": str(plot_path),
    }
    output.write_text(json.dumps(output_data, indent=2) + "\n", encoding="utf-8")
    print(f"  job: {job}")
    print(f"  vacuum direction: {direction}")
    print(f"  fermi energy: {result['efermi']:.8f} eV")
    for index, item in enumerate(work_functions, start=1):
        print(f"  work function {index}: {item['work_function']:.8f} eV")
    print(f"  profile: {profile_path}")
    print(f"  plot: {plot_path}")
    print(f"  results: {output}")
    return 0


def register_parser(subparsers) -> None:
    """Register the work-function preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "workfunc",
        "Calculate a surface work function from the electrostatic potential.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
