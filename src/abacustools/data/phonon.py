"""Periodic phonon calculations on top of phonopy and phono3py.

This module holds the computation the phonon, vibration and
thermal-conductivity workflows share: turning an ABACUS structure into the
objects phonopy expects, building supercells that follow the phonopy atom
order, reading ABACUS forces, and assembling the displacement manifest of a
prepared workflow.  The command modules only parse arguments, drive these
functions and render the result.

Phonopy orders the atoms of a supercell by the atom of the reference cell,
while :meth:`abacustools.io.stru.AbacusSTRU.supercell` orders them by lattice
point.  Mixing the two orders attaches every calculated force to the wrong
atom, which is silent and ruins the spectrum, so supercells written for a
displaced calculation are always rebuilt from the phonopy object here.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from abacustools.io.stru import AbacusSTRU


def validate_positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite positive command parameter.

    Args:
        value: Value to check.
        name: Parameter name used in the error message.
        allow_zero: Whether zero is accepted in addition to positive values.

    Returns:
        The value as a float.

    Raises:
        ValueError: When the value is not a finite number of the required sign.
    """
    number = float(value)
    if not np.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} finite number")
    return number


def validate_supercell(supercell: Any) -> List[int]:
    """Validate and normalize a three-dimensional supercell.

    Args:
        supercell: Three positive integers.

    Returns:
        The normalized repetitions.

    Raises:
        ValueError: When the value is not three positive integers.
    """
    if supercell is None:
        raise ValueError("supercell must be specified before validation")
    values = list(supercell)
    if len(values) != 3 or any(isinstance(value, bool) for value in values):
        raise ValueError("supercell must contain three positive integers")
    try:
        normalized = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("supercell must contain three positive integers") from error
    if any(value != original or value <= 0 for value, original in zip(normalized, values)):
        raise ValueError("supercell must contain three positive integers")
    return normalized


def validate_mesh(mesh: Any) -> List[int]:
    """Validate the reciprocal-space mesh dimensions.

    Args:
        mesh: Three positive integers.

    Returns:
        The normalized mesh.

    Raises:
        ValueError: When the value is not three positive integers.
    """
    values = list(mesh)
    if len(values) != 3 or any(isinstance(value, bool) for value in values):
        raise ValueError("mesh must contain three positive integers")
    normalized = [int(value) for value in values]
    if any(value != original or value <= 0 for value, original in zip(normalized, values)):
        raise ValueError("mesh must contain three positive integers")
    return normalized


def automatic_supercell(structure: AbacusSTRU, min_supercell_length: float) -> List[int]:
    """Choose diagonal supercell repetitions from the lattice-vector lengths.

    Args:
        structure: Structure to expand.
        min_supercell_length: Minimum lattice-vector length of the supercell,
            in Angstrom.

    Returns:
        One repetition per lattice vector.

    Raises:
        ValueError: When the structure has no usable lattice vectors.
    """
    validate_positive_float(min_supercell_length, "min_supercell_length")
    lengths = np.linalg.norm(np.asarray(structure.cell, dtype=float), axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("structure must have three finite, non-zero lattice vectors")
    return [max(1, int(np.ceil(min_supercell_length / length))) for length in lengths]


def phonopy_atoms(structure: AbacusSTRU):
    """Convert an ABACUS structure to ``PhonopyAtoms`` in Angstrom units.

    Args:
        structure: Structure to convert.

    Returns:
        The equivalent ``phonopy`` ``PhonopyAtoms``.
    """
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=structure.elements,
        cell=np.asarray(structure.cell, dtype=float),
        scaled_positions=np.asarray(structure.coords_direct, dtype=float),
    )


def phonopy_supercell_structure(structure: AbacusSTRU, phonopy_supercell) -> AbacusSTRU:
    """Build an ABACUS supercell that follows the phonopy atom order.

    Args:
        structure: Reference-cell structure supplying the per-element atom
            attributes.
        phonopy_supercell: A ``phonopy`` supercell object.

    Returns:
        The supercell with phonopy's atom order.

    Raises:
        RuntimeError: When the structure defines no element, or the supercell
            holds an element the reference cell does not.
    """
    by_element: Dict[str, Any] = {}
    for atom in structure.atoms:
        if atom.element is not None:
            by_element.setdefault(atom.element, atom)
    if not by_element:
        raise RuntimeError("structure does not define any element")

    atoms = []
    for symbol, position in zip(phonopy_supercell.symbols, phonopy_supercell.positions):
        source = by_element.get(symbol)
        if source is None:
            raise RuntimeError(f"phonopy supercell contains an unknown element: {symbol}")
        atom = deepcopy(source)
        atom.coord = tuple(float(value) for value in position)
        atoms.append(atom)
    return AbacusSTRU(
        cell=np.asarray(phonopy_supercell.cell, dtype=float).tolist(),
        atoms=atoms,
        dpks=structure.dpks,
        metadata=deepcopy(structure.metadata),
    )


def moved_mode_indices(
    frequencies: Sequence[float],
    reference: Sequence[float],
    *,
    tolerance: float = 1.0e-6,
) -> List[int]:
    """Return the modes that a correction shifted off the reference spectrum.

    A non-analytical correction is a rank one perturbation of the dynamical
    matrix, so it moves the modes that couple to it and leaves every other
    frequency where it was.  Finding those modes by matching the two frequency
    sets is what survives the reordering the shift causes: a mode that climbs
    above its neighbours changes the band index of everything above it, so
    comparing the two spectra band by band reports the modes it displaced as
    moved as well.  Matching the values instead finds only the mode that has no
    partner in the reference spectrum.

    Args:
        frequencies: Frequencies of the corrected calculation, in THz.
        reference: Frequencies of the same calculation with the correction off,
            in the same order.
        tolerance: Largest difference, in THz, at which two frequencies count
            as the same mode.

    Returns:
        The zero-based indices of the corrected modes with no partner.

    Raises:
        ValueError: When the two spectra hold a different number of modes.
    """
    values = np.asarray(frequencies, dtype=float)
    if values.shape != np.asarray(reference, dtype=float).shape:
        raise ValueError(
            "the corrected and the reference spectra must hold the same number "
            "of modes"
        )
    unmatched = list(np.asarray(reference, dtype=float))
    moved = []
    for index, frequency in enumerate(values):
        for position, candidate in enumerate(unmatched):
            if abs(frequency - candidate) <= tolerance:
                unmatched.pop(position)
                break
        else:
            moved.append(index)
    return moved


def read_forces(job: Path, version: str, expected_natoms: int) -> np.ndarray:
    """Read one converged ABACUS force array in eV/Angstrom.

    A force set is only meaningful for a fully converged electron density:
    force constants come from the difference of forces, which is far more
    sensitive to SCF noise than a total energy.

    Args:
        job: ABACUS job directory holding the force output.
        version: ABACUS version hint for the running-log profiler.
        expected_natoms: Number of atoms the force array must hold.

    Returns:
        The ``(natoms, 3)`` force array.

    Raises:
        RuntimeError: When the calculation did not converge, wrote no force, or
            wrote an array of the wrong shape.
    """
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(
        job,
        param_names=["force", "converged"],
        version=version,
    )
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    if result["force"] is None:
        raise RuntimeError(f"forces were not found in the output: {job}")
    forces = np.asarray(result["force"], dtype=float)
    expected_shape = (expected_natoms, 3)
    if forces.shape != expected_shape or not np.all(np.isfinite(forces)):
        raise RuntimeError(
            f"invalid force array in the output: {job}; "
            f"expected {expected_shape}, got {forces.shape}"
        )
    return forces


def jsonable(value: Any) -> Any:
    """Convert NumPy values nested in a result dictionary to JSON values.

    Args:
        value: Value, possibly holding NumPy arrays or scalars.

    Returns:
        An equivalent value made only of JSON-compatible types.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def displacement_task(prefix: str, index: int) -> Dict[str, Any]:
    """Return the manifest entry of one displaced supercell.

    The zero-padded number makes the dataset index readable from the
    directory name, so a prepared directory tree explains itself.

    Args:
        prefix: Task prefix, such as ``"disp-"``.
        index: Zero-based index of the displaced supercell in the phonopy
            dataset.

    Returns:
        Mapping with the task name and the dataset index.
    """
    return {"task": f"{prefix}{index:04d}", "index": int(index)}


def displacement_tasks(supercells: Sequence[Any], prefix: str) -> List[Dict[str, Any]]:
    """Return the manifest entries of the displaced supercells that exist.

    A symmetry-aware displacement generation leaves ``None`` in place of the
    supercells it can derive from symmetry.  Those entries are skipped, but
    their index is kept so that the forces can be mapped back onto the full
    dataset.

    Args:
        supercells: The ``supercells_with_displacements`` sequence.
        prefix: Task prefix, such as ``"disp-"``.

    Returns:
        One entry per non-empty supercell, in dataset order.
    """
    return [
        displacement_task(prefix, index)
        for index, supercell in enumerate(supercells)
        if supercell is not None
    ]


def validate_displacement_entries(entries: Any, count: int, workflow: str) -> List[Dict[str, Any]]:
    """Validate the displacement entries of a workflow manifest.

    Args:
        entries: Manifest value to validate.
        count: Number of displaced supercells the dataset holds.
        workflow: Workflow name used in the error messages.

    Returns:
        The validated entries.

    Raises:
        RuntimeError: When an entry is missing, malformed or out of range.
    """
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{workflow} manifest has no displacement tasks")
    validated = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("task"), str):
            raise RuntimeError(f"invalid displacement entry in {workflow} manifest")
        try:
            index = int(entry["index"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid displacement index in {workflow} manifest") from error
        if index < 0 or index >= count:
            raise RuntimeError(
                f"displacement index in {workflow} manifest is out of range"
            )
        # Additional keys, such as the displaced atom and its displacement,
        # ride along for reporting without affecting the mapping.
        validated.append({**entry, "index": index})
    if len({entry["task"] for entry in validated}) != len(validated):
        raise RuntimeError(f"{workflow} manifest names a displacement task twice")
    return validated


def collect_forces(
    job: Path,
    entries: Sequence[Dict[str, Any]],
    supercells: Sequence[Any],
    version: str,
    expected_natoms: int,
    *,
    workflow: str,
) -> List[Optional[np.ndarray]]:
    """Read one force set per displaced supercell, keeping symmetry gaps.

    Args:
        job: Directory holding the displaced calculation directories.
        entries: Validated manifest entries with a task name and dataset index.
        supercells: The ``supercells_with_displacements`` sequence, whose
            ``None`` entries mark symmetry-derived displacements.
        version: ABACUS version hint for the running-log profiler.
        expected_natoms: Number of atoms of the supercell.
        workflow: Workflow name used in the error messages.

    Returns:
        One force array per dataset index, ``None`` where the dataset leaves
        the supercell to symmetry.

    Raises:
        RuntimeError: When a required force set could not be read.
    """
    forces: List[Optional[np.ndarray]] = [None] * len(supercells)
    for entry in entries:
        forces[entry["index"]] = read_forces(
            job / str(entry["task"]), version, expected_natoms
        )
    missing = [
        index
        for index, supercell in enumerate(supercells)
        if supercell is not None and forces[index] is None
    ]
    if missing:
        raise RuntimeError(
            f"no forces were collected for {workflow} displacement indices: "
            + ", ".join(str(index) for index in missing)
        )
    return forces
