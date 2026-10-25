"""Helpers and CLI stages for the pEDA/NOCV workflow."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from abacustools.core.constant import KILOJOULE_PER_MOL_TO_EV
from abacustools.core.job import read_job_structure
from abacustools.core.render import cli_output
from abacustools.data.abacus_result import get_result_from_job
from abacustools.data.eda import (
    analyze_gamma_peda_nocv,
    analyze_kpoint_peda,
    assemble_lts_peda_state_densities,
    validate_peda_jobs,
)
from abacustools.data.lcao import (
    PEDAOrbitalState,
    build_peda_kpoint_states,
    suggest_peda_dmr_r_blocks,
    write_peda_state_dmr,
    write_peda_state_wfc,
)
from abacustools.data.versions import default_version, resolve_version
from abacustools.io.abacus import WriteInput

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    resolve_output,
    write_abacus_job,
    write_manifest,
)


_PEDA_STATE_TASKS = ("peda/states/p0", "peda/states/p_ts")
_PEDA_DENSITY_TASKS = ("peda/density/p0",)
_PEDA_TERM_FIELDS = (
    "preparation",
    "electrostatic",
    "pauli",
    "orbital",
    "dispersion",
    "interaction_electronic",
    "interaction",
    "bond",
)


def _is_gamma_point(direct: Sequence[float]) -> bool:
    return all(abs((float(value) + 0.5) % 1.0 - 0.5) <= 1e-6 for value in direct)


def _flow_for_version(version: str | None, job: Path) -> tuple[str, str]:
    """Return ``(flow, profile key)`` for a pEDA calculation.

    ``auto`` uses the parent job's log when available.  A job without a
    recognizable version keeps the historical develop-input behavior, while
    an explicit LTS alias selects the wavefunction-to-density conversion
    route.
    """

    requested = None if version is None else str(version).strip().lower()
    if requested and requested != "auto":
        profile = resolve_version(version)
    else:
        profile = resolve_version(version, job_dir=job)
    flow = "lts" if profile.key == "3.10.1LTS" else "develop"
    return flow, profile.key


def _without_lts_incompatible_inputs(inputs: dict) -> dict:
    """Remove develop-only pEDA output keywords before writing an LTS deck."""

    cleaned = deepcopy(inputs)
    for key in ("out_hsk", "out_dmk", "out_dmr", "out_dmr1"):
        cleaned.pop(key, None)
    return cleaned


def prepare_peda_state_job(
    job: str | Path,
    states: Sequence[PEDAOrbitalState],
    destination: str | Path,
    *,
    field: str = "p0",
    r_blocks: Sequence[tuple[int, int, int]] | None = None,
    override: bool = False,
    version: str | None = None,
) -> Path:
    """Prepare a fixed-density pEDA state job.

    Gamma-only reference jobs keep the current single-``R=(0,0,0)`` behavior.
    Multi-k reference jobs preserve the input KPT and symmetry mode
    (``symmetry=-1`` or ``0``) and require explicit R blocks for the DMR
    Fourier transform.

    Args:
        job: Reference full-system ABACUS job directory.
        states: pEDA states for one or more k-points.
        destination: Output directory, relative to ``job`` when not absolute.
        field: ``p0``, ``p_ts`` or ``p_ab``.
        r_blocks: Explicit R blocks for multi-k DMR export.
        override: Replace an existing destination directory.
        version: ABACUS version profile.  The LTS profile writes a cube-based
            fixed-density job; the develop profile keeps the DMR input path.

    Returns:
        The generated job directory.

    Raises:
        ValueError: If the reference job, k-points, symmetry mode or R blocks
            are inconsistent with the requested state job.
        RuntimeError: If the destination exists and ``override`` is false.
    """

    if not states:
        raise ValueError("state preparation requires at least one k-point")
    if field not in {"p0", "p_ts", "p_ab"}:
        raise ValueError(f"unknown pEDA state field: {field}")

    job_path = Path(job).resolve()
    if not job_path.is_dir():
        raise RuntimeError(f"job directory does not exist: {job_path}")
    inputs, stru_filename, structure = read_job_structure(job_path)
    if str(inputs.get("basis_type", "pw")).lower() != "lcao":
        raise ValueError("pEDA state jobs require basis_type=lcao")
    gamma_only = bool(int(inputs.get("gamma_only", 0) or 0))
    if gamma_only:
        if len(states) != 1 or not _is_gamma_point(states[0].direct):
            raise ValueError("Gamma state preparation only accepts one Gamma k-point")
    else:
        symmetry_value = inputs.get("symmetry")
        if symmetry_value is None:
            raise ValueError("multi-k pEDA state jobs require an explicit symmetry=-1 or 0")
        symmetry = int(
            symmetry_value[0] if isinstance(symmetry_value, (list, tuple)) else symmetry_value
        )
        if symmetry not in (-1, 0):
            raise ValueError(
                "multi-k pEDA state jobs currently support symmetry=-1 or 0; "
                f"got symmetry={symmetry}"
            )
        if r_blocks is None:
            r_blocks = suggest_peda_dmr_r_blocks(job_path)

    output = Path(destination)
    if not output.is_absolute():
        output = job_path / output
    if output.exists() or output.is_symlink():
        if not override:
            raise RuntimeError(
                f"state job directory already exists: {output}; use override to replace it"
            )
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        else:
            output.unlink()

    flow, _profile_key = _flow_for_version(version, job_path)
    state_inputs = _without_lts_incompatible_inputs(inputs) if flow == "lts" else deepcopy(inputs)
    if flow == "lts":
        state_inputs.pop("out_dm", None)
        state_inputs.pop("out_dm1", None)
        state_inputs.pop("out_pchg", None)
        state_inputs.pop("nbands_istate", None)
        state_inputs.update(
            {
                "calculation": "nscf",
                "init_chg": "file",
                "read_file_dir": ".",
                "mixing_beta": 0.0,
                "scf_nmax": 1,
                "out_mat_hs": 1,
                "out_chg": 0,
                "out_wfc_lcao": 0,
            }
        )
    else:
        state_inputs.update(
            {
                "calculation": "nscf",
                "init_chg": "dm",
                "read_file_dir": ".",
                "mixing_beta": 0.0,
                "scf_nmax": 1,
                "out_hsk": 1,
                "out_chg": 1,
                "out_dmk": 0,
                "out_dmr": 1,
                "out_wfc_lcao": 0,
            }
        )
    if gamma_only:
        state_inputs["symmetry"] = -1
    kpoint_file = kpoint_filename(job_path, inputs)
    write_abacus_job(
        state_inputs,
        structure,
        job_path,
        output,
        stru_filename=stru_filename,
        kpoint=kpoint_file,
    )
    # The generated job contains symlinks to every resource, but retaining the
    # source job's relative pseudo/orbital directories would make ABACUS look
    # for them below the new state directory.
    state_inputs["pseudo_dir"] = "."
    state_inputs["orbital_dir"] = "."
    WriteInput(state_inputs, output / "INPUT")
    if flow == "develop":
        write_peda_state_dmr(
            states,
            structure,
            output,
            field=field,
            r_blocks=r_blocks,
        )
    return output


def prepare_gamma_state_job(
    job: str | Path,
    states: Sequence[PEDAOrbitalState],
    destination: str | Path,
    *,
    field: str = "p0",
    override: bool = False,
    version: str | None = None,
) -> Path:
    """Prepare one Gamma-point fixed-density ABACUS job."""

    if len(states) != 1:
        raise ValueError("Gamma state preparation requires exactly one k-point")
    if not _is_gamma_point(states[0].direct):
        raise ValueError("Gamma state preparation only accepts a Gamma k-point")
    return prepare_peda_state_job(
        job,
        states,
        destination,
        field=field,
        override=override,
        version=version,
    )


def prepare_lts_p0_density_job(
    job: str | Path,
    states: Sequence[PEDAOrbitalState],
    destination: str | Path,
    *,
    override: bool = False,
) -> Path:
    """Prepare an LTS ``get_pchg`` job that turns P0 orbitals into a cube.

    The generated job writes ``WFC_NAO_*`` files containing the P0 occupied
    orbitals and asks ABACUS to evaluate one band-decomposed charge density
    for each occupied band.  A later density stage sums those cubes and
    combines them with the full-system density when P_TS is needed.

    Args:
        job: Parent full-system ABACUS job.
        states: pEDA states for P0.
        destination: Output directory, relative to ``job`` when not absolute.
        override: Replace an existing destination directory.

    Returns:
        The generated conversion job directory.
    """

    if not states:
        raise ValueError("LTS density preparation requires at least one pEDA state")
    job_path = Path(job).resolve()
    if not job_path.is_dir():
        raise RuntimeError(f"job directory does not exist: {job_path}")
    inputs, stru_filename, structure = read_job_structure(job_path)
    if str(inputs.get("basis_type", "pw")).lower() != "lcao":
        raise ValueError("LTS pEDA density conversion requires basis_type=lcao")
    gamma_only = bool(int(inputs.get("gamma_only", 0) or 0))
    if gamma_only and (len(states) != 1 or not _is_gamma_point(states[0].direct)):
        raise ValueError("Gamma density conversion requires one Gamma pEDA state")

    dimensions = {
        int(state.channels[spin].basis.occupied.shape[0])
        for state in states
        for spin in state.channels
    }
    if len(dimensions) != 1:
        raise ValueError("pEDA states use inconsistent NAO basis dimensions")
    nlocal = dimensions.pop()
    noccupied = max(
        int(state.channels[spin].basis.occupied.shape[1])
        for state in states
        for spin in state.channels
    )
    if noccupied <= 0 or noccupied > nlocal:
        raise ValueError("pEDA P0 has an invalid occupied-band count")

    output = Path(destination)
    if not output.is_absolute():
        output = job_path / output
    if output.exists() or output.is_symlink():
        if not override:
            raise RuntimeError(
                f"density job directory already exists: {output}; use override to replace it"
            )
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        else:
            output.unlink()

    density_inputs = _without_lts_incompatible_inputs(inputs)
    density_inputs.pop("out_dm", None)
    density_inputs.pop("out_dm1", None)
    density_inputs.pop("nbands_istate", None)
    density_inputs.update(
        {
            "calculation": "get_pchg",
            "init_wfc": "file",
            "init_chg": "atomic",
            "read_file_dir": ".",
            "nbands": nlocal,
            "out_pchg": [1] * noccupied + [0] * (nlocal - noccupied),
            "if_separate_k": 0,
            "out_wfc_lcao": 0,
            "out_chg": 0,
            "out_mat_hs": 0,
        }
    )
    if gamma_only:
        density_inputs["symmetry"] = -1
    kpoint_file = kpoint_filename(job_path, inputs)
    write_abacus_job(
        density_inputs,
        structure,
        job_path,
        output,
        stru_filename=stru_filename,
        kpoint=kpoint_file,
    )
    density_inputs["pseudo_dir"] = "."
    density_inputs["orbital_dir"] = "."
    WriteInput(density_inputs, output / "INPUT")
    write_peda_state_wfc(states, output, gamma_only=gamma_only)
    return output


def _resolve_job(value: str | Path, label: str) -> Path:
    """Resolve and validate one job directory from the command line."""
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{label} directory does not exist: {path}")
    return path


def _manifest_path(job: Path, path: Path) -> str:
    """Store paths below the parent job relatively and other paths absolutely."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(job.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _path_from_manifest(job: Path, value: str | None) -> Path | None:
    """Resolve an optional path recorded in the pEDA manifest."""
    if value is None:
        return None
    path = Path(value)
    return (job / path).resolve() if not path.is_absolute() else path.resolve()


def _revalidate_manifest_jobs(
    job: Path,
    fragment_a: Path,
    fragment_b: Path,
    reference_a: Path | None,
    reference_b: Path | None,
    atoms_a: Sequence[int],
    atoms_b: Sequence[int],
) -> None:
    """Recheck pEDA inputs when all referenced jobs are available locally."""

    paths = (
        job,
        fragment_a,
        fragment_b,
        *((reference_a,) if reference_a is not None else ()),
        *((reference_b,) if reference_b is not None else ()),
    )
    if not all((path / "INPUT").is_file() for path in paths):
        return
    validation = validate_peda_jobs(
        job,
        fragment_a,
        fragment_b,
        atoms_a,
        atoms_b,
        reference_a_job=reference_a,
        reference_b_job=reference_b,
    )
    if (
        tuple(atoms_a) != validation.fragment_a_atoms
        or tuple(atoms_b) != validation.fragment_b_atoms
    ):
        raise RuntimeError(
            "pEDA manifest atom maps no longer match the current fragment STRU order; "
            "rerun workflow peda prepare"
        )


def _one_based_atoms(
    values: Sequence[int],
    natoms: int,
    label: str,
) -> list[int]:
    """Validate one-based atom indices and convert them to zero-based indices."""
    if any(index < 1 for index in values):
        raise ValueError(f"{label} atom indices must be greater than zero")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} atom indices contain duplicates")
    if any(index > natoms for index in values):
        raise ValueError(f"{label} atom index exceeds the number of atoms ({natoms})")
    return [index - 1 for index in values]


def _require_converged(job: Path, label: str, version: str) -> float:
    """Return an SCF energy after checking that the job converged."""
    result = get_result_from_job(
        str(job),
        ["energy", "converged"],
        version=version,
    )
    if not result.get("converged"):
        raise RuntimeError(f"{label} calculation did not converge: {job}")
    energy = result.get("energy")
    if energy is None:
        raise RuntimeError(f"{label} calculation has no total energy: {job}")
    return float(energy)


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the pEDA preparation stage."""
    parser.add_argument(
        "-j",
        "--job",
        type=Path,
        required=True,
        help="Full-system Gamma/multi-k LCAO SCF job.",
    )
    parser.add_argument(
        "--fragment-a",
        type=Path,
        required=True,
        help="SCF job for fragment A at the complex geometry.",
    )
    parser.add_argument(
        "--fragment-b",
        type=Path,
        required=True,
        help="SCF job for fragment B at the complex geometry.",
    )
    parser.add_argument(
        "--reference-a",
        type=Path,
        help="Optional isolated reference job for fragment A.",
    )
    parser.add_argument(
        "--reference-b",
        type=Path,
        help="Optional isolated reference job for fragment B.",
    )
    parser.add_argument(
        "--atoms-a",
        nargs="+",
        type=int,
        required=True,
        help="One-based full-system atom indices belonging to fragment A.",
    )
    parser.add_argument(
        "--atoms-b",
        nargs="+",
        type=int,
        required=True,
        help="One-based full-system atom indices belonging to fragment B.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace existing generated pEDA state jobs.",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=default_version(),
        help=(
            "ABACUS version used for the generated jobs. Use an LTS alias "
            "(for example LTS3.10.1) to enable the LTS density conversion route; "
            "the default keeps the develop DMR route unless a job log identifies LTS."
        ),
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the pEDA postprocessing stage."""
    parser.add_argument(
        "-j",
        "--job",
        type=Path,
        required=True,
        help="Parent full-system job containing workflow_peda.json.",
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
        default="peda_results.json",
        help="Result JSON filename, relative to JOB unless absolute.",
    )


def _register_density_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the LTS density conversion stage."""

    parser.add_argument(
        "-j",
        "--job",
        type=Path,
        required=True,
        help="Parent full-system job containing workflow_peda.json.",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=default_version(),
        help="ABACUS version used for the density conversion job.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="peda_density.json",
        help="Density metadata filename, relative to JOB unless absolute.",
    )


def prepare(args: argparse.Namespace) -> int:
    """Prepare P0 and, for Gamma calculations, P_TS fixed-density jobs."""
    job = _resolve_job(args.job, "full-system job")
    fragment_a = _resolve_job(args.fragment_a, "fragment A job")
    fragment_b = _resolve_job(args.fragment_b, "fragment B job")
    reference_a = (
        None if args.reference_a is None else _resolve_job(args.reference_a, "reference A job")
    )
    reference_b = (
        None if args.reference_b is None else _resolve_job(args.reference_b, "reference B job")
    )

    inputs, _stru_filename, structure = read_job_structure(job)
    if str(inputs.get("basis_type", "pw")).lower() != "lcao":
        raise ValueError("pEDA requires basis_type=lcao")
    version = getattr(args, "version", default_version())
    flow, profile_key = _flow_for_version(version, job)

    atoms_a_input = _one_based_atoms(args.atoms_a, structure.natoms, "fragment A")
    atoms_b_input = _one_based_atoms(args.atoms_b, structure.natoms, "fragment B")
    validation = (
        validate_peda_jobs(
            job,
            fragment_a,
            fragment_b,
            atoms_a_input,
            atoms_b_input,
            reference_a_job=reference_a,
            reference_b_job=reference_b,
        )
        if all(
            (path / "INPUT").is_file()
            for path in (
                job,
                fragment_a,
                fragment_b,
                *((reference_a,) if reference_a is not None else ()),
                *((reference_b,) if reference_b is not None else ()),
            )
        )
        else None
    )
    atoms_a = list(validation.fragment_a_atoms) if validation is not None else list(atoms_a_input)
    atoms_b = list(validation.fragment_b_atoms) if validation is not None else list(atoms_b_input)

    states = build_peda_kpoint_states(
        job,
        fragment_a,
        fragment_b,
        atoms_a,
        atoms_b,
        gamma_only=None,
    )
    if not states:
        raise ValueError("pEDA found no k-point states")

    gamma_only = bool(int(inputs.get("gamma_only", 0) or 0))
    nocv = gamma_only and len(states) == 1 and _is_gamma_point(states[0].direct)

    generated_tasks = list(_PEDA_STATE_TASKS)
    if flow == "lts":
        generated_tasks = list(_PEDA_DENSITY_TASKS) + generated_tasks
    clear_generated_jobs(job, generated_tasks, override=args.override)
    if flow == "lts":
        prepare_lts_p0_density_job(
            job,
            states,
            job / _PEDA_DENSITY_TASKS[0],
            override=False,
        )
    p0_job = job / "peda/states/p0"
    prepare_peda_state_job(
        job,
        states,
        p0_job,
        field="p0",
        override=False,
        version=version,
    )
    tasks = [_PEDA_STATE_TASKS[0]]
    if flow == "lts":
        tasks.insert(0, _PEDA_DENSITY_TASKS[0])
    if nocv:
        p_ts_job = job / "peda/states/p_ts"
        prepare_peda_state_job(
            job,
            states,
            p_ts_job,
            field="p_ts",
            override=False,
            version=version,
        )
        tasks.append(_PEDA_STATE_TASKS[1])

    write_manifest(
        job,
        "peda",
        tasks=tasks,
        fragment_a_job=_manifest_path(job, fragment_a),
        fragment_b_job=_manifest_path(job, fragment_b),
        reference_a_job=(None if reference_a is None else _manifest_path(job, reference_a)),
        reference_b_job=(None if reference_b is None else _manifest_path(job, reference_b)),
        fragment_a_atoms=[index + 1 for index in atoms_a],
        fragment_b_atoms=[index + 1 for index in atoms_b],
        fragment_a_atom_map=[index + 1 for index in atoms_a],
        fragment_b_atom_map=[index + 1 for index in atoms_b],
        validated_settings={} if validation is None else validation.settings,
        nspin=int(inputs.get("nspin", 1) or 1),
        gamma_only=gamma_only,
        abacus_version=profile_key,
        density_mode="file" if flow == "lts" else "dm",
        density_task=(
            _PEDA_DENSITY_TASKS[0] if flow == "lts" else None
        ),
        nocv=nocv,
        kpoint_count=len(states),
    )
    cli_output(f"  prepared pEDA state jobs: {', '.join(tasks)}")
    cli_output(f"  NOCV enabled: {'yes' if nocv else 'no'}")
    return 0


def postprocess(args: argparse.Namespace) -> int:
    """Analyze prepared pEDA state jobs and write a result JSON."""
    job = _resolve_job(args.job, "full-system job")
    manifest = read_manifest(job, "peda", ("peda/states/p0",))
    tasks = list(manifest.get("tasks", []))
    nocv = bool(manifest.get("nocv", False))
    if nocv and _PEDA_STATE_TASKS[1] not in tasks:
        raise RuntimeError("pEDA manifest enables NOCV but does not list P_TS")
    if manifest.get("density_mode") == "file":
        required_cubes = ["peda/states/p0/SPIN1_CHG.cube"]
        if int(manifest.get("nspin", 1)) == 2:
            required_cubes.append("peda/states/p0/SPIN2_CHG.cube")
        if nocv:
            required_cubes.extend(
                path.replace("peda/states/p0", "peda/states/p_ts") for path in required_cubes[:]
            )
        missing = [path for path in required_cubes if not (job / path).is_file()]
        if missing:
            raise RuntimeError(
                "LTS pEDA state density cubes are missing; run "
                "`abacustools workflow peda density -j JOB` after the density job: "
                + ", ".join(missing)
            )

    fragment_a = _path_from_manifest(job, manifest.get("fragment_a_job"))
    fragment_b = _path_from_manifest(job, manifest.get("fragment_b_job"))
    if fragment_a is None or fragment_b is None:
        raise RuntimeError("pEDA manifest is missing a fragment job")
    reference_a = _path_from_manifest(job, manifest.get("reference_a_job"))
    reference_b = _path_from_manifest(job, manifest.get("reference_b_job"))
    atom_map_a = manifest.get("fragment_a_atom_map", manifest["fragment_a_atoms"])
    atom_map_b = manifest.get("fragment_b_atom_map", manifest["fragment_b_atoms"])
    atoms_a = [int(index) - 1 for index in atom_map_a]
    atoms_b = [int(index) - 1 for index in atom_map_b]
    _revalidate_manifest_jobs(
        job,
        fragment_a,
        fragment_b,
        reference_a,
        reference_b,
        atoms_a,
        atoms_b,
    )

    energies = {
        "full": _require_converged(job, "full-system", args.version),
        "fragment_a": _require_converged(fragment_a, "fragment A", args.version),
        "fragment_b": _require_converged(fragment_b, "fragment B", args.version),
    }
    if reference_a is not None:
        energies["reference_a"] = _require_converged(reference_a, "reference A", args.version)
    if reference_b is not None:
        energies["reference_b"] = _require_converged(reference_b, "reference B", args.version)

    p0_job = job / _PEDA_STATE_TASKS[0]
    if nocv:
        p_ts_job = job / _PEDA_STATE_TASKS[1]
        result = analyze_gamma_peda_nocv(
            job,
            fragment_a,
            fragment_b,
            p0_job,
            p_ts_job,
            atoms_a,
            atoms_b,
            reference_a_job=reference_a,
            reference_b_job=reference_b,
            version=args.version,
            require_converged_fragments=True,
        )
        analysis = "gamma_peda_nocv"
        kpoint_count = 1
        nspin = len(result.nocv_channels)
        if result.energy_ledger and "full" in result.energy_ledger:
            e0 = result.energy_ledger["full"].electronic_ev - result.terms.orbital
        else:
            e0 = energies["full"] - result.terms.orbital
    else:
        result = analyze_kpoint_peda(
            job,
            fragment_a,
            fragment_b,
            p0_job,
            atoms_a,
            atoms_b,
            reference_a_job=reference_a,
            reference_b_job=reference_b,
            version=args.version,
            require_converged_fragments=True,
        )
        analysis = "kpoint_peda"
        kpoint_count = result.kpoint_count
        nspin = result.nspin
        e0 = result.e0

    term_data = asdict(result.terms)
    terms_ev = {name: float(term_data[name]) for name in _PEDA_TERM_FIELDS}
    payload = {
        "format": 1,
        "workflow": "peda",
        "analysis": analysis,
        "abacus_version": manifest.get("abacus_version"),
        "density_mode": manifest.get("density_mode", "dm"),
        "job": str(job),
        "fragment_a_job": str(fragment_a),
        "fragment_b_job": str(fragment_b),
        "reference_a_job": None if reference_a is None else str(reference_a),
        "reference_b_job": None if reference_b is None else str(reference_b),
        "fragment_a_atoms": [index + 1 for index in atoms_a],
        "fragment_b_atoms": [index + 1 for index in atoms_b],
        "kpoint_count": kpoint_count,
        "nspin": nspin,
        "energies_ev": energies,
        "energy_ledger_ev": (
            {}
            if result.energy_ledger is None
            else {name: asdict(value) for name, value in result.energy_ledger.items()}
        ),
        "e0_ev": float(e0),
        "terms_ev": terms_ev,
        "terms_kj_mol": {name: value / KILOJOULE_PER_MOL_TO_EV for name, value in terms_ev.items()},
        "electrostatic_ev": asdict(result.electrostatic),
        "dmr_validation": result.dmr_validation,
    }
    if nocv:
        payload["nocv_total_ev"] = float(result.nocv_total_ev)
        payload["orbital_error_ev"] = float(result.orbital_error_ev)
        payload["nocv_channels"] = {
            str(spin): asdict(decomposition) for spin, decomposition in result.nocv_channels.items()
        }

    output = resolve_output(job, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    cli_output(f"  pEDA result: {output}")
    cli_output(f"  interaction: {terms_ev['interaction']:.10f} eV")
    cli_output(f"  bond: {terms_ev['bond']:.10f} eV")
    return 0


def density(args: argparse.Namespace) -> int:
    """Assemble LTS P0/P_TS cubes after the ``get_pchg`` job has run."""

    job = _resolve_job(args.job, "full-system job")
    manifest = read_manifest(job, "peda", (_PEDA_DENSITY_TASKS[0],))
    if manifest.get("density_mode") != "file":
        raise RuntimeError(
            "this pEDA manifest uses the develop DMR route; "
            "the density stage is only available for LTS density_mode=file"
        )
    fragment_a = _path_from_manifest(job, manifest.get("fragment_a_job"))
    fragment_b = _path_from_manifest(job, manifest.get("fragment_b_job"))
    if fragment_a is None or fragment_b is None:
        raise RuntimeError("pEDA manifest is missing a fragment job")
    atom_map_a = manifest.get("fragment_a_atom_map", manifest["fragment_a_atoms"])
    atom_map_b = manifest.get("fragment_b_atom_map", manifest["fragment_b_atoms"])
    atoms_a = [int(index) - 1 for index in atom_map_a]
    atoms_b = [int(index) - 1 for index in atom_map_b]
    states = build_peda_kpoint_states(
        job,
        fragment_a,
        fragment_b,
        atoms_a,
        atoms_b,
        nspin=manifest.get("nspin"),
        gamma_only=bool(manifest.get("gamma_only", False)),
    )
    if not states:
        raise ValueError("density stage found no pEDA k-point states")
    tasks = list(manifest.get("tasks", []))
    p_ts_job = job / _PEDA_STATE_TASKS[1] if _PEDA_STATE_TASKS[1] in tasks else None
    result = assemble_lts_peda_state_densities(
        states,
        job / manifest["density_task"],
        job,
        job / _PEDA_STATE_TASKS[0],
        p_ts_job=p_ts_job,
        version=args.version,
    )
    output = resolve_output(job, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    cli_output(f"  pEDA density metadata: {output}")
    cli_output(
        "  integrated P0 electrons: "
        + ", ".join(f"spin {spin}: {value:.6f}" for spin, value in result.electron_numbers.items())
    )
    return 0


def register_parser(subparsers) -> None:
    """Register the pEDA preparation and postprocessing stages."""
    workflow_parser = subparsers.add_parser(
        "peda",
        help="Run periodic energy decomposition analysis (pEDA/NOCV).",
    )
    stages = workflow_parser.add_subparsers(
        dest="peda_command",
        metavar="COMMAND",
        title="peda commands",
        required=True,
    )
    prepare_parser = stages.add_parser("prepare", help="Prepare pEDA jobs.")
    _register_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=prepare)
    density_parser = stages.add_parser(
        "density",
        help="Assemble LTS fixed-density cubes from the P0 get_pchg job.",
    )
    _register_density_arguments(density_parser)
    density_parser.set_defaults(handler=density)
    postprocess_parser = stages.add_parser(
        "postprocess",
        help="Postprocess pEDA workflow results.",
    )
    _register_postprocess_arguments(postprocess_parser)
    postprocess_parser.set_defaults(handler=postprocess)
