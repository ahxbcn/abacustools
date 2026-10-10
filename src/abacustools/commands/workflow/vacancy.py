"""The ``abacustools workflow vacancy`` workflow."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import cast

from abacustools.data.vacancy import (
    build_elemental_crystal,
    set_atom_empty,
    vacancy_formation_energy,
)
from abacustools.data.versions import default_version
from abacustools.io.abacus import WriteInput
from abacustools.io.stru import AbacusATOM, AbacusSTRU
from abacustools.core.job import read_job_structure

from .common import (
    clear_generated_jobs,
    copy_referenced_files,
    kpoint_filename,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)


_ORIGINAL_TASK = "vacancy_original_stru"


def _vacancy_indices(index, index_file) -> list[int]:
    """Return the deduplicated 1-based vacancy indices."""
    if index_file is not None:
        lines = [
            line.split()
            for line in Path(index_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"vacancy index file is empty: {index_file}")
        values = [int(value) for value in lines[0]]
    elif index:
        values = [int(value) for value in index]
    else:
        raise ValueError("no vacancy index specified; use --index or --index-file")

    result: list[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _validate_supercell(supercell) -> tuple[int, int, int]:
    if len(supercell) != 3 or any(int(size) < 1 for size in supercell):
        raise ValueError("supercell must be three positive integers")
    return (int(supercell[0]), int(supercell[1]), int(supercell[2]))


def _resolve(job: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job / path


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the vacancy preparation stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory used to prepare the vacancy calculations.",
    )
    parser.add_argument(
        "-s", "--supercell", type=int, nargs=3, default=[1, 1, 1],
        help="Supercell size, default: 1 1 1.",
    )
    parser.add_argument(
        "-i", "--index", type=int, nargs="+", default=None,
        help="1-based indices of the atoms to remove.",
    )
    parser.add_argument(
        "--index-file", default=None,
        help="File whose first line lists the atom indices to remove (overrides --index).",
    )
    parser.add_argument(
        "--cal-reference", action=argparse.BooleanOptionalAction, default=True,
        help="Calculate reference elemental crystal energies (default: enabled).",
    )
    parser.add_argument(
        "--ref-dir", default="ref_element",
        help="Directory for the reference elemental crystal jobs, relative to JOB.",
    )
    parser.add_argument(
        "--max-step", type=int, default=100,
        help="Maximum number of relaxation steps, default: 100.",
    )
    parser.add_argument(
        "--force-thr-ev", type=float, default=0.01,
        help="Force convergence threshold in eV/Angstrom, default: 0.01.",
    )
    parser.add_argument(
        "--stress-thr-kbar", type=float, default=0.5,
        help="Stress convergence threshold in kBar, default: 0.5.",
    )
    parser.add_argument(
        "--relax-kspacing", type=float, default=None,
        help="Override kspacing for the cell-relaxation jobs.",
    )
    parser.add_argument(
        "--scf-kspacing", type=float, default=None,
        help="Override kspacing for the final SCF jobs.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace existing generated vacancy directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the vacancy postprocessing stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory containing the prepared vacancy calculations.",
    )
    parser.add_argument(
        "-v", "--version", default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "--ref-dir", default=None,
        help="Directory with reference elemental crystal jobs; defaults to the manifest value.",
    )
    parser.add_argument(
        "--ref-file", default="ref_energy.txt",
        help="Reference atom energy file (element energy per atom), relative to JOB.",
    )
    parser.add_argument(
        "-o", "--output", default="vacancy_results.json",
        help="Output JSON filename, relative to JOB by default.",
    )


def _write_final_scf(job: Path, destination: Path, scf_inputs, structure, kpoint_file) -> None:
    """Write a final-SCF job whose STRU is produced from the relaxed structure."""
    destination.mkdir(parents=True, exist_ok=True)
    WriteInput(scf_inputs, str(destination / "INPUT"))
    copy_referenced_files(structure, job, destination, kpoint=kpoint_file)


def prepare(args: argparse.Namespace) -> int:
    """Prepare the pristine, defect, and reference calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    supercell = _validate_supercell(args.supercell)
    indices = _vacancy_indices(args.index, args.index_file)

    inputs, stru_filename, structure = read_job_structure(job)
    for index in indices:
        if index < 1 or index > structure.natoms:
            raise ValueError(
                f"vacancy index {index} is out of range for {structure.natoms} atoms"
            )

    relax_inputs = deepcopy(inputs)
    relax_inputs["calculation"] = "cell-relax"
    relax_inputs["relax_method"] = "cg"
    relax_inputs["symmetry"] = 0
    relax_inputs["relax_nmax"] = args.max_step
    relax_inputs["force_thr_ev"] = args.force_thr_ev
    relax_inputs["stress_thr"] = args.stress_thr_kbar
    if args.relax_kspacing is not None:
        relax_inputs["kspacing"] = args.relax_kspacing

    scf_inputs = deepcopy(inputs)
    scf_inputs["calculation"] = "scf"
    if args.scf_kspacing is not None:
        scf_inputs["kspacing"] = args.scf_kspacing

    kpoint_file = kpoint_filename(job, inputs)

    element_pp: dict[str, str] = {}
    element_orb: dict[str, str] = {}
    for element, pp, orb in zip(structure.elements, structure.pps, structure.orbs):
        if element is None:
            continue
        if pp is not None:
            element_pp.setdefault(element, pp)
        if orb is not None:
            element_orb.setdefault(element, orb)

    supercell_structure = structure.supercell(list(supercell))

    points = []
    for index in indices:
        element = cast(AbacusATOM, structure[index - 1]).element
        points.append(
            {
                "name": f"vacancy_defect_{index}_{element}_{supercell[0]}_{supercell[1]}_{supercell[2]}",
                "index": index,
                "element": element,
            }
        )

    ref_dir = _resolve(job, args.ref_dir)
    names = [_ORIGINAL_TASK] + [point["name"] for point in points]
    if args.cal_reference:
        names += [f"{args.ref_dir}/{element}" for element in sorted({p["element"] for p in points})]
    clear_generated_jobs(job, names, override=args.override)

    print(f"  job: {job}")
    print(f"  supercell: {supercell[0]} {supercell[1]} {supercell[2]}")
    print(f"  vacancy indices: {', '.join(str(index) for index in indices)}")

    write_abacus_job(
        relax_inputs, supercell_structure, job, job / _ORIGINAL_TASK,
        stru_filename=stru_filename, kpoint=kpoint_file,
    )
    _write_final_scf(job, job / _ORIGINAL_TASK / "final_scf", scf_inputs, supercell_structure, kpoint_file)
    print(f"  prepared {_ORIGINAL_TASK}")

    for point in points:
        defect_structure = set_atom_empty(supercell_structure, point["index"] - 1)
        write_abacus_job(
            relax_inputs, defect_structure, job, job / point["name"],
            stru_filename=stru_filename, kpoint=kpoint_file,
        )
        _write_final_scf(job, job / point["name"] / "final_scf", scf_inputs, defect_structure, kpoint_file)
        print(f"  prepared {point['name']}")

    if args.cal_reference:
        for element in sorted({point["element"] for point in points}):
            crystal = build_elemental_crystal(
                element, element_pp.get(element), element_orb.get(element)
            )
            write_abacus_job(
                relax_inputs, crystal, job, ref_dir / element,
                stru_filename=stru_filename, kpoint=kpoint_file,
            )
            print(f"  prepared reference {element}")

    write_manifest(
        job,
        "vacancy",
        tasks=names,
        points=points,
        supercell=list(supercell),
        stru_filename=stru_filename,
        ref_dir=args.ref_dir,
        cal_reference=bool(args.cal_reference),
    )
    return 0


def _read_energy(job: Path, version: str) -> float:
    """Read one total energy in eV from an ABACUS job."""
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(str(job), param_names=["energy"], version=version)
    energy = result.get("energy")
    if energy is None or not math.isfinite(energy):
        raise RuntimeError(f"energy was not found in the output: {job}")
    return float(energy)


def _read_reference_energies(
    job: Path, ref_dir: Path, ref_file: Path, stru_filename: str, version: str
) -> dict[str, float]:
    """Read reference atom energies from a file and reference crystal jobs."""
    energies: dict[str, float] = {}
    if ref_file.is_file():
        for line in ref_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label, value = line.split()
            energies[label] = float(value)

    if ref_dir.is_dir():
        for element_dir in sorted(path for path in ref_dir.iterdir() if path.is_dir()):
            crystal = AbacusSTRU.read(str(element_dir / stru_filename))
            if crystal is None:
                raise RuntimeError(f"failed to read reference structure: {element_dir}")
            element = crystal.elements[0]
            if element is None:
                raise RuntimeError(f"could not determine the element of {element_dir}")
            energies[element] = _read_energy(element_dir, version) / crystal.natoms
    return energies


def postprocess(args: argparse.Namespace) -> int:
    """Compute the vacancy formation energies and write the report."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    manifest = read_manifest(job, "vacancy", required_tasks=[])
    points = manifest.get("points", [])
    if not points:
        raise RuntimeError("the vacancy manifest does not contain any defect tasks")

    supercell = manifest.get("supercell", [1, 1, 1])
    supercell_factor = int(supercell[0]) * int(supercell[1]) * int(supercell[2])
    stru_filename = manifest.get("stru_filename", "STRU")

    ref_dir_value = args.ref_dir if args.ref_dir is not None else manifest.get("ref_dir", "ref_element")
    ref_dir = _resolve(job, ref_dir_value)
    ref_file = _resolve(job, args.ref_file)
    reference_energies = _read_reference_energies(
        job, ref_dir, ref_file, stru_filename, args.version
    )

    original_energy = _read_energy(job / _ORIGINAL_TASK / "final_scf", args.version)
    print(f"  original supercell energy: {original_energy:.8f} eV")

    results = {}
    for point in points:
        name = point["name"]
        element = point["element"]
        if element not in reference_energies:
            raise RuntimeError(f"no reference atom energy for element {element}")
        defect_energy = _read_energy(job / name / "final_scf", args.version)
        formation_energy = vacancy_formation_energy(
            defect_energy, original_energy, supercell_factor, reference_energies[element]
        )
        results[name] = {
            "element": element,
            "index": point["index"],
            "defect_energy": defect_energy,
            "reference_atom_energy": reference_energies[element],
            "vacancy_formation_energy": formation_energy,
        }
        print(f"  {name}: E_f = {formation_energy:.6f} eV")

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "workflow": "vacancy",
        "supercell": list(supercell),
        "original_energy": original_energy,
        "reference_atom_energies": reference_energies,
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    ref_file.parent.mkdir(parents=True, exist_ok=True)
    ref_file.write_text(
        "".join(f"{element} {energy}\n" for element, energy in sorted(reference_energies.items())),
        encoding="utf-8",
    )

    print(f"  results: {output}")
    print(f"  reference atom energies: {ref_file}")
    return 0


def register_parser(subparsers) -> None:
    """Register the vacancy preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "vacancy",
        "Calculate vacancy formation energies.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
