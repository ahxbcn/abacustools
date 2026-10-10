"""Adapters for using the optional ABACUS ASE interface.

``abacuslite`` owns the ASE calculator and ABACUS I/O implementation. This
module only connects it to the structures, jobs, and result conventions that
already exist in :mod:`abacustools`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


class AbacusLiteUnavailableError(ImportError):
    """Raised when an ABACUS ASE calculator is requested but unavailable."""


def load_abacuslite():
    """Return the optional ``Abacus`` and ``AbacusProfile`` classes.

    The dependency is imported lazily so normal ABACUSTools workflows do not
    need to install the separate ASE interface.
    """
    try:
        from abacuslite import Abacus, AbacusProfile
    except ImportError as error:
        raise AbacusLiteUnavailableError(
            "The optional 'abacuslite' package is required for ASE workflows. "
            "Install the ABACUS ASE interface and make it importable first."
        ) from error
    return Abacus, AbacusProfile


def abacuslite_available() -> bool:
    """Return whether the optional ABACUS ASE interface can be imported."""
    try:
        load_abacuslite()
    except AbacusLiteUnavailableError:
        return False
    return True


def make_profile(
    command: str = "abacus",
    *,
    pseudo_dir: Optional[str | Path] = None,
    orbital_dir: Optional[str | Path] = None,
    omp_num_threads: Optional[int] = None,
    **kwargs: Any,
):
    """Create an ``abacuslite.AbacusProfile`` without a hard dependency."""
    _, profile_type = load_abacuslite()
    return profile_type(
        command=command,
        pseudo_dir=pseudo_dir,
        orbital_dir=orbital_dir,
        omp_num_threads=omp_num_threads,
        **kwargs,
    )


def _structure_maps(structure) -> tuple[dict[str, str], dict[str, str]]:
    """Build element-to-file maps required by ``abacuslite.Abacus``."""
    pseudopotentials: dict[str, str] = {}
    basissets: dict[str, str] = {}
    for atom in structure.atoms:
        element = atom.element or atom.label
        for target, filename, name in (
            (pseudopotentials, atom.pp, "pseudopotential"),
            (basissets, atom.orb, "orbital"),
        ):
            if filename is None:
                continue
            previous = target.get(element)
            if previous is not None and previous != filename:
                raise ValueError(
                    f"{element} has conflicting {name} files: "
                    f"{previous!r} and {filename!r}"
                )
            target[element] = filename
    return pseudopotentials, basissets


def _apply_constraints(atoms, structure) -> None:
    """Translate ABACUS ``move`` flags to ASE constraints."""
    from ase.constraints import FixAtoms, FixCartesian

    moves = np.asarray(structure.moves, dtype=bool)
    if moves.shape != (len(atoms), 3) or np.all(moves):
        return
    constraints = []
    fixed = np.all(~moves, axis=1)
    if np.any(fixed):
        constraints.append(FixAtoms(indices=np.flatnonzero(fixed)))
    partial = (~fixed) & np.any(~moves, axis=1)
    if np.any(partial):
        constraints.append(
            FixCartesian(
                a=np.flatnonzero(partial),
                mask=~moves[partial],
            )
        )
    atoms.set_constraint(constraints)


def structure_to_atoms(structure):
    """Convert an :class:`AbacusSTRU` while retaining files and constraints."""
    atoms = structure.to("ase")
    _apply_constraints(atoms, structure)
    return atoms


def calculator_from_structure(
    structure,
    profile,
    *,
    directory: str | Path = ".",
    inp: Optional[Mapping[str, Any]] = None,
    pseudopotentials: Optional[Mapping[str, str]] = None,
    basissets: Optional[Mapping[str, str]] = None,
    kpts: Any = None,
    **kwargs: Any,
):
    """Create an ``abacuslite.Abacus`` calculator for an ``AbacusSTRU``.

    This is the common entry point for ASE relaxations, cell relaxations, NEB
    images, and molecular dynamics.
    """
    Abacus, _ = load_abacuslite()
    inferred_pp, inferred_orb = _structure_maps(structure)
    calculator_kwargs: dict[str, Any] = {
        "profile": profile,
        "directory": Path(directory),
        "inp": dict(inp or {}),
    }
    pp_files = dict(inferred_pp if pseudopotentials is None else pseudopotentials)
    orb_files = dict(inferred_orb if basissets is None else basissets)
    if pp_files:
        calculator_kwargs["pseudopotentials"] = pp_files
    if orb_files:
        calculator_kwargs["basissets"] = orb_files
    if kpts is not None:
        calculator_kwargs["kpts"] = kpts
    calculator_kwargs.update(kwargs)
    return Abacus(**calculator_kwargs)


def _read_job_kpts(job: Path, inputs: Mapping[str, Any]):
    """Read an explicit KPT using abacuslite's parser when present."""
    try:
        kspacing = inputs.get("kspacing", 0)
        kspacing_enabled = any(
            float(value) > 0
            for value in (kspacing if isinstance(kspacing, (list, tuple)) else [kspacing])
        )
    except (TypeError, ValueError):
        kspacing_enabled = False
    gamma_only = str(inputs.get("gamma_only", "0")).strip().lower() in {
        "1",
        "true",
    }
    if kspacing_enabled or gamma_only:
        return None
    filename = str(inputs.get("kpoint_file", "KPT"))
    path = job / filename
    if not path.is_file():
        return None
    try:
        from abacuslite.io.generalio import read_kpt
    except ImportError as error:
        raise AbacusLiteUnavailableError(
            "abacuslite is required to read an explicit KPT file"
        ) from error
    return read_kpt(path)


def calculator_from_job(
    job: str | Path,
    profile,
    *,
    directory: Optional[str | Path] = None,
    inp: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
):
    """Create an ASE calculator from an existing ABACUS job directory."""
    from abacustools.core.job import read_job_structure

    job_path = Path(job).absolute()
    inputs, stru_filename, structure = read_job_structure(job_path)
    parameters = dict(inputs)
    parameters.update(inp or {})
    parameters.setdefault("stru_file", stru_filename)
    kpts = _read_job_kpts(job_path, parameters)
    return calculator_from_structure(
        structure,
        profile,
        directory=job_path if directory is None else directory,
        inp=parameters,
        kpts=kpts,
        **kwargs,
    )


def attach_calculator(atoms, calculator):
    """Attach and return a calculator, convenient for ASE workflow builders."""
    atoms.calc = calculator
    return calculator


def _json_value(value: Any) -> Any:
    """Convert numpy values in ASE results to JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def result_to_dict(
    calculator,
    *,
    trajectory: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Normalize an ASE calculator result to the repository's JSON style."""
    results = getattr(calculator, "results", {}) or {}
    output: dict[str, Any] = {}
    for name in (
        "energy",
        "free_energy",
        "stress",
        "magmom",
        "efermi",
        "eigenvalues",
        "occupations",
        "ibzkpts",
    ):
        if name in results:
            output[name] = _json_value(results[name])
    if "forces" in results:
        output["force"] = _json_value(results["forces"])
    if "energy" in output and "free_energy" not in output:
        output["free_energy"] = output["energy"]
    converged = getattr(calculator, "last_scf_converged", None)
    if converged is not None:
        output["converged"] = bool(converged)
    if trajectory is not None:
        output["trajectory"] = str(Path(trajectory))
    return output


def write_result(
    filename: str | Path,
    calculator,
    *,
    trajectory: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Write :func:`result_to_dict` output and return the written mapping."""
    result = result_to_dict(calculator, trajectory=trajectory)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
