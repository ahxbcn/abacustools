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


def initialize_phonopy(structure: AbacusSTRU, supercell: Sequence[int]):
    """Return a Phonopy object for a diagonal supercell.

    ``primitive_matrix="P"`` pins the primitive cell to the reference cell.
    Phonopy 4 resolves the ``"auto"`` default with a symmetry search while
    phonopy 3 used the identity, so leaving it unset would make the dynamical
    matrix depend on the installed phonopy version.

    Args:
        structure: Reference cell of the calculation.
        supercell: Diagonal supercell repetitions.

    Returns:
        The Phonopy object, without force constants.
    """
    from phonopy import Phonopy

    return Phonopy(
        phonopy_atoms(structure),
        supercell_matrix=np.diag([int(value) for value in supercell]),
        primitive_matrix="P",
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


def workflow_displacements(phonon, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate and return the displacement dataset recorded during preparation.

    Args:
        phonon: Phonopy object used to bound the displaced-atom index.
        manifest: Preparation-time manifest of a phonon workflow.

    Returns:
        The validated dataset entries, each with a displaced atom and a
        displacement in Angstrom.

    Raises:
        RuntimeError: When the dataset is missing, malformed, or inconsistent
            with the supercell.
    """
    displacements = manifest.get("dataset")
    if not isinstance(displacements, list) or not displacements:
        raise RuntimeError("phonon workflow manifest has no displacement dataset")
    validated = []
    for item in displacements:
        if not isinstance(item, dict) or "number" not in item or "displacement" not in item:
            raise RuntimeError("invalid displacement entry in phonon workflow manifest")
        try:
            number = int(item["number"])
            displacement = np.asarray(item["displacement"], dtype=float)
        except (TypeError, ValueError) as error:
            raise RuntimeError("invalid displacement entry in phonon workflow manifest") from error
        if number < 0 or number >= len(phonon.supercell) or displacement.shape != (3,):
            raise RuntimeError("invalid displacement entry in phonon workflow manifest")
        if not np.all(np.isfinite(displacement)):
            raise RuntimeError("invalid displacement entry in phonon workflow manifest")
        validated.append({"number": number, "displacement": displacement.tolist()})
    return validated


def band_path(
    structure: AbacusSTRU,
    *,
    qpath: Any = None,
    high_symm_points: Any = None,
    npoints: int = 101,
):
    """Return the band path of a structure as phonopy band q-points.

    Without a ``qpath`` the path is the seekpath band path of the structure;
    with one it is read from the label sequence and the label coordinates.

    Args:
        structure: Reference cell the path is built for.
        qpath: Path of labels, or a list of paths of labels.
        high_symm_points: Mapping from label to fractional coordinates, needed
            together with ``qpath``.
        npoints: Number of points of every segment.

    Returns:
        The tuple ``(qpoints, labels, connections)`` phonopy's band structure
        takes, with ``qpoints`` a list of arrays.

    Raises:
        ValueError: When a custom path is incomplete or malformed.
    """
    if qpath is None:
        from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

        return get_band_qpoints_by_seekpath(
            phonopy_atoms(structure), npoints=npoints, is_const_interval=True
        )
    if not isinstance(qpath, list) or not qpath:
        raise ValueError("qpath must be a non-empty list")
    if not isinstance(high_symm_points, dict) or not high_symm_points:
        raise ValueError("high_symm_points must be a non-empty object")
    if not isinstance(npoints, int) or npoints < 2:
        raise ValueError("npoints must be at least 2")

    if all(isinstance(item, str) for item in qpath):
        paths = [qpath]
    elif all(
        isinstance(item, list) and item and all(isinstance(point, str) for point in item)
        for item in qpath
    ):
        paths = qpath
    else:
        raise ValueError("qpath must be a list of labels or a list of label lists")

    points = {}
    for label, coordinates in high_symm_points.items():
        values = np.asarray(coordinates, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"high-symmetry point {label!r} must contain three finite coordinates"
            )
        points[label] = values.tolist()

    band_paths = []
    labels = []
    for path in paths:
        converted = []
        for label in path:
            if label not in points:
                raise ValueError(f"qpath label {label!r} is missing from high_symm_points")
            converted.append(points[label])
            labels.append(r"$\Gamma$" if label.lower() in {"g", "gamma"} else label)
        band_paths.append(converted)

    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    qpoints, connections = get_band_qpoints_and_path_connections(
        band_paths, npoints=npoints
    )
    return qpoints, labels, connections


def load_workflow_phonon(
    job: Path,
    *,
    version: str = "",
    symmetrize: bool = True,
    nac_params: Optional[Dict[str, Any]] = None,
    manifest_name: str = "workflow_phonon.json",
):
    """Rebuild the phonopy object of a finished phonon workflow.

    The displaced supercells and the force sets they produced are put back
    together exactly as the workflow's own postprocessing stage does, so a
    caller can use phonopy features the workflow does not expose — the
    Grueneisen parameters, for instance — on the same force constants.

    Args:
        job: Directory of a phonon workflow whose displacements are finished.
        version: ABACUS version hint for the running-log profiler.
        symmetrize: Whether to symmetrize the force constants.
        nac_params: Non-analytical correction parameters for the dynamical
            matrix, in phonopy's format.
        manifest_name: Name of the preparation manifest to read.

    Returns:
        The Phonopy object with force constants (and ``nac_params`` when given).

    Raises:
        FileNotFoundError: When the job or its manifest is missing.
        RuntimeError: When the manifest does not describe this job.
    """
    import json

    job = Path(job)
    manifest_file = job / manifest_name
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"could not find the phonon workflow manifest: {manifest_file}"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    stru_filename = str(manifest.get("stru_filename", "STRU"))
    structure = AbacusSTRU.read(job / stru_filename)
    if structure is None:
        raise RuntimeError(f"failed to read structure: {job / stru_filename}")
    supercell = manifest.get("supercell")
    if not isinstance(supercell, list) or len(supercell) != 3:
        raise RuntimeError(f"invalid supercell in {manifest_file}")

    phonon = initialize_phonopy(structure, supercell)
    displacements = workflow_displacements(phonon, manifest)
    phonon.dataset = {
        "natom": len(phonon.supercell),
        "first_atoms": [
            {"number": item["number"], "displacement": item["displacement"]}
            for item in displacements
        ],
    }
    phonon.forces = [
        read_forces(job / displacement_task("disp-", index)["task"], version, len(phonon.supercell))
        for index in range(len(displacements))
    ]
    phonon.produce_force_constants()
    if symmetrize:
        phonon.symmetrize_force_constants()
    if nac_params is not None:
        phonon.nac_params = nac_params
    return phonon
