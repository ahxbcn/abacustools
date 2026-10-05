"""Re-resolve a prepared job's pseudopotentials and orbitals from another library.

``job prepare`` resolves the resource files of a structure once and writes the
chosen file names into the job's ``STRU``.  This module performs the same
mapping for a job that already exists: it resolves every element again from a
selected resource library (or from another orbital variant of the same
library), rewrites the ``ATOMIC_SPECIES`` pseudopotential names and the
``NUMERICAL_ORBITAL`` entries of the ``STRU``, and replaces the files in the
job directory so the directory stays self-contained.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from abacustools.core.input_prep import (
    InputPreparationError,
    PathLike,
    _install_resources,
    _orbital_cutoff,
    _unique,
    resolve_library_resource,
)
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU

#: Suffixes that a resource switch is allowed to replace in the job directory.
_RESOURCE_SUFFIXES = {".upf", ".orb"}


@dataclass
class ResourceSwitchResult:
    """What switching the resource library of one job changed.

    Attributes:
        job: Job directory that was updated.
        library: Resource library the files were resolved from.
        variant: Orbital variant that was honoured, when one applies.
        basis: Basis type read from the job, ``lcao`` or ``pw``.
        copy_resources: Whether files were copied rather than symlinked.
        dry_run: Whether the report only describes the changes.
        pseudopotentials: Element to resolved pseudopotential file name.
        orbitals: Element to resolved orbital file name (LCAO jobs only).
        installed: Resource file names written into the job directory.
        removed: Resource file names dropped from the job directory.
    """

    job: Path
    library: Optional[str]
    variant: Optional[str]
    basis: str
    copy_resources: bool
    dry_run: bool
    pseudopotentials: dict[str, str] = field(default_factory=dict)
    orbitals: dict[str, str] = field(default_factory=dict)
    installed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        """Return the result as a JSON-compatible dictionary."""
        return {
            "job": str(self.job),
            "library": self.library,
            "variant": self.variant,
            "basis": self.basis,
            "copy_resources": self.copy_resources,
            "dry_run": self.dry_run,
            "pseudopotentials": dict(self.pseudopotentials),
            "orbitals": dict(self.orbitals),
            "installed": list(self.installed),
            "removed": list(self.removed),
        }


def read_job_inputs(job: PathLike) -> dict[str, Any]:
    """Read a job's ``INPUT``, returning an empty mapping when it is unreadable.

    Args:
        job: Job directory.

    Returns:
        The parsed INPUT parameters, or an empty mapping.
    """
    input_path = Path(job) / "INPUT"
    if not input_path.is_file():
        return {}
    try:
        return ReadInput(input_path) or {}
    except Exception:
        return {}


def _basis_from_inputs(inputs: dict[str, Any], structure: AbacusSTRU) -> str:
    """Resolve the basis type of a job from its INPUT and structure."""
    basis = str(inputs.get("basis_type", "")).strip().lower()
    return basis or ("lcao" if any(structure.orbs) else "pw")


def read_job_basis(job: PathLike, structure: AbacusSTRU) -> str:
    """Return the basis type of a job directory.

    The ``basis_type`` of the job's ``INPUT`` wins, so a plane-wave job that
    still carries a ``NUMERICAL_ORBITAL`` block from its source structure is
    treated as plane-wave.  Without a readable ``INPUT`` the structure decides:
    a STRU with orbital files is LCAO.

    Args:
        job: Job directory.
        structure: Structure read from the job's structure file.

    Returns:
        The basis type, lowercase.
    """
    return _basis_from_inputs(read_job_inputs(job), structure)


def _job_uses_copies(job: Path, names: set[str]) -> bool:
    """Infer whether a job materialises its resources as copies or symlinks.

    A job prepared with ``--copy-resources`` holds regular files, while the
    default holds symlinks.  Mixed or absent resources fall back to symlinks,
    matching the default of ``job prepare``.
    """
    copies = 0
    links = 0
    for name in names:
        candidate = job / name
        if candidate.is_symlink():
            links += 1
        elif candidate.is_file():
            copies += 1
    return copies > 0 and links == 0


def _warn_below_orbital_cutoff(job: Path, orbitals: dict[str, str]) -> None:
    """Warn when the job's ``ecutwfc`` is below the new orbital cutoff."""
    cutoffs = [cutoff for name in orbitals.values() if (cutoff := _orbital_cutoff(name))]
    if not cutoffs:
        return
    required = max(cutoffs)
    input_path = job / "INPUT"
    if not input_path.is_file():
        return
    try:
        current = ReadInput(input_path).get("ecutwfc")
    except Exception:
        return
    if current is None:
        return
    try:
        current_value = float(current)
    except (TypeError, ValueError):
        return
    if current_value < required:
        warnings.warn(
            f"ecutwfc {current_value:g} Ry is below the {required:g} Ry cutoff of the "
            "new numerical orbitals; ABACUS needs at least the orbital cutoff",
            stacklevel=3,
        )


def switch_job_library(
    job: PathLike,
    *,
    library: Optional[str] = None,
    variant: Optional[str] = None,
    pp_path: Optional[PathLike] = None,
    orb_path: Optional[PathLike] = None,
    copy_resources: Optional[bool] = None,
    dry_run: bool = False,
) -> ResourceSwitchResult:
    """Switch one job directory to another pseudopotential/orbital library.

    Every element of the job's ``STRU`` is resolved again from the target
    library, the ``ATOMIC_SPECIES`` pseudopotential name and the
    ``NUMERICAL_ORBITAL`` entry of each species are rewritten, and the resolved
    files are installed in the job directory.  Files that the old ``STRU``
    referenced but the new one does not are removed, so no stale resource stays
    behind.

    Args:
        job: Job directory holding ``INPUT`` and the structure file it names
            (``STRU`` by default).
        library: Configured library name; the configured default when omitted.
        variant: Orbital variant such as ``SZ``, ``DZP`` or ``precision``.
        pp_path: Explicit pseudopotential directory, overriding the library.
        orb_path: Explicit orbital directory, overriding the library.
        copy_resources: Copy the files instead of symlinking them.  ``None``
            follows how the existing job materialised its resources.
        dry_run: Report the changes without writing ``STRU`` or touching files.

    Returns:
        ResourceSwitchResult: The resolved names and the files installed and
        removed.

    Raises:
        InputPreparationError: When the job, its ``STRU``, or a resource of an
            element in the target library is missing.
    """
    job = Path(job).expanduser()
    if not job.is_dir():
        raise InputPreparationError(f"job directory does not exist: {job}")
    inputs = read_job_inputs(job)
    stru_filename = str(inputs.get("stru_file", "STRU"))
    stru_path = job / stru_filename
    if not stru_path.is_file():
        raise InputPreparationError(f"no {stru_filename} file in {job}")
    structure = AbacusSTRU.read(str(stru_path), fmt="stru")
    if structure is None:
        raise InputPreparationError(f"failed to read {stru_path}")

    basis = _basis_from_inputs(inputs, structure)
    is_lcao = basis.startswith("lcao")
    elements = [element for element in _unique(structure.elements) if element]

    # Resolve the new files before touching anything, so a missing element
    # leaves the job untouched.
    resolved_pp = {
        element: resolve_library_resource(element, "pp", library=library, pp_path=pp_path)
        for element in elements
    }
    resolved_orb = (
        {
            element: resolve_library_resource(
                element, "orb", library=library, variant=variant, orb_path=orb_path
            )
            for element in elements
        }
        if is_lcao
        else {}
    )
    pseudopotentials = {element: res.filename for element, res in resolved_pp.items()}
    orbitals = {element: res.filename for element, res in resolved_orb.items()}
    # resolve_library_resource falls back to resources.default; report the name
    # that was actually used rather than the unset argument.
    selected_library = library
    if resolved_pp:
        selected_library = resolved_pp[elements[0]].library
    elif resolved_orb:
        selected_library = resolved_orb[elements[0]].library

    # The files the old STRU referenced, to be replaced by the new selection.
    old_names = {name for name in structure.pps if name}
    if is_lcao:
        old_names.update(name for name in structure.orbs if name)
    new_names = set(pseudopotentials.values()) | set(orbitals.values())
    obsolete = sorted(old_names - new_names)

    if copy_resources is None:
        copy_resources = _job_uses_copies(job, old_names)

    removed = [
        name
        for name in obsolete
        if (job / name).suffix.lower() in _RESOURCE_SUFFIXES
        and ((job / name).is_file() or (job / name).is_symlink())
    ]
    installed = sorted(new_names)

    if dry_run:
        return ResourceSwitchResult(
            job=job,
            library=selected_library,
            variant=variant,
            basis=basis,
            copy_resources=copy_resources,
            dry_run=True,
            pseudopotentials=pseudopotentials,
            orbitals=orbitals,
            installed=installed,
            removed=removed,
        )

    # Apply the new names to every atom of the structure.
    for atom in structure.atoms:
        if atom.element in pseudopotentials:
            atom.pp = pseudopotentials[atom.element]
        if is_lcao and atom.element in orbitals:
            atom.orb = orbitals[atom.element]
    if not structure.write(str(stru_path), fmt="stru"):
        raise InputPreparationError(f"failed to write {stru_path}")

    resources = {res.path: res.filename for res in resolved_pp.values()}
    resources.update({res.path: res.filename for res in resolved_orb.values()})
    _install_resources(resources, job, copy_resources)
    for name in removed:
        (job / name).unlink()

    if is_lcao:
        _warn_below_orbital_cutoff(job, orbitals)

    return ResourceSwitchResult(
        job=job,
        library=selected_library,
        variant=variant,
        basis=basis,
        copy_resources=copy_resources,
        dry_run=False,
        pseudopotentials=pseudopotentials,
        orbitals=orbitals,
        installed=installed,
        removed=removed,
    )
