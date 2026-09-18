"""Berry phase polarization data of ABACUS calculations.

The Born effective charge and piezoelectric workflows both consume the same
quantities: the polarization ABACUS writes to its Berry phase logs, the
shortest change of that polarization across the branch, and the regular
k-point mesh a Berry phase step needs.  Those computations live here so that
the command modules only parse arguments and render results.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.io.abacus import kspacing2kpt


#: Berry phase logs of the three Cartesian directions, in the order ABACUS
#: writes them for one displaced structure.
BERRY_LOGS = ("running_nscf1.log", "running_nscf2.log", "running_nscf3.log")

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _numbers(line: str) -> List[float]:
    """Extract floating-point values from an ABACUS log line."""
    return [float(value.replace("D", "E").replace("d", "e")) for value in re.findall(_NUMBER, line)]


def kpoint_mesh(
    job: Path,
    inputs: Dict[str, Any],
    structure,
) -> Tuple[List[float], str]:
    """Return the regular k-point mesh a Berry phase step needs.

    A Berry phase calculation needs an explicit mesh rather than a spacing,
    so the spacing of the source job is expanded into a mesh here.

    Args:
        job: Job directory holding the ``INPUT`` and any explicit ``KPT`` file.
        inputs: Parsed ``INPUT`` of the source job.
        structure: Structure the mesh is built for.

    Returns:
        The mesh values and the KPT model, ``"gamma"`` or ``"mp"``.

    Raises:
        ValueError: When neither an input mesh nor a valid KPT file is found.
    """
    from abacustools.io.abacus import ReadKpt

    try:
        if float(inputs.get("gamma_only", 0)) > 0:
            return [1, 1, 1, 0.0, 0.0, 0.0], "gamma"
    except (TypeError, ValueError):
        pass

    kspacing = inputs.get("kspacing")
    if kspacing not in (None, 0, "0", "0.0"):
        cell_bohr = np.asarray(structure.cell, dtype=float) * ANG_TO_BOHR
        mesh = kspacing2kpt(kspacing, cell_bohr)
        return [*mesh, 0.0, 0.0, 0.0], "gamma"

    parsed = ReadKpt(str(job))
    if parsed is None:
        raise ValueError(f"could not read a KPT file below {job}")
    kpt_data, model = parsed
    values = [float(value) for value in list(kpt_data)[:6]]
    if model not in {"gamma", "mp"} or len(values) != 6:
        raise ValueError(f"KPT file below {job} does not define a regular mesh")
    return values, model


def read_berry_polarization(log_path: Path) -> Dict[str, Any]:
    """Read Berry phase polarization and its quantum from one ABACUS log.

    Args:
        log_path: Berry phase running log.

    Returns:
        Mapping with the polarization direction, the polarization along the
        lattice vector, its modulus in the same unit, the polarization in
        C/m^2 and the cell volume.

    Raises:
        ValueError: When the log holds no polarization.
    """
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    p_vec: Optional[float] = None
    modulus: Optional[float] = None
    polarization_cm2: Optional[float] = None
    volume: Optional[float] = None
    direction: Optional[int] = None

    for index, line in enumerate(lines):
        if "Volume (A^3)" in line:
            values = _numbers(line)
            if values:
                volume = values[-1]
        if "The calculated polarization direction is" not in line:
            continue

        direction_values = re.findall(r"R\s*([123])", line)
        if direction_values:
            direction = int(direction_values[-1])
        for following in lines[index + 1 : index + 8]:
            values = _numbers(following)
            if "(e/Omega).bohr" in following and len(values) >= 2:
                p_vec = values[0] * BOHR_TO_ANG
                modulus = values[1] * BOHR_TO_ANG
            elif "C/m^2" in following and values:
                polarization_cm2 = values[0]

    if p_vec is None or modulus is None:
        raise ValueError(f"Berry phase polarization was not found in {log_path}")
    return {
        "direction": direction,
        "p_vec": p_vec,
        "mod": modulus,
        "polarization_cm2": polarization_cm2,
        "volume": volume,
    }


def polarization_cartesian(p_vec: Iterable[float], cell: Iterable[Iterable[float]]) -> List[float]:
    """Convert polarization components in lattice-vector basis to Cartesian axes.

    Args:
        p_vec: Polarization along each lattice vector.
        cell: Lattice vectors.

    Returns:
        The Cartesian components.

    Raises:
        ValueError: When the cell or the vector is malformed.
    """
    vectors = np.asarray(list(cell), dtype=float)
    values = np.asarray(list(p_vec), dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    if vectors.shape != (3, 3) or values.shape != (3,) or np.any(lengths <= 0):
        raise ValueError("cell and polarization must contain three valid vectors")
    return np.sum(values[:, None] * vectors / lengths[:, None], axis=0).tolist()


def polarization_delta(
    original: Iterable[float], displaced: Iterable[float], modulus: Iterable[float]
) -> List[float]:
    """Calculate the shortest polarization change across the Berry phase branch.

    Args:
        original: Polarization of the reference structure.
        displaced: Polarization of the displaced structure.
        modulus: Polarization quantum of each direction.

    Returns:
        The wrapped polarization difference.

    Raises:
        ValueError: When a vector does not hold three components.
    """
    delta = np.asarray(list(displaced), dtype=float) - np.asarray(list(original), dtype=float)
    quantum = np.asarray(list(modulus), dtype=float)
    if delta.shape != (3,) or quantum.shape != (3,):
        raise ValueError("polarization vectors must contain three values")
    valid = quantum > 0
    delta[valid] -= np.rint(delta[valid] / quantum[valid]) * quantum[valid]
    return delta.tolist()


def read_task_polarization(
    task: Path,
    suffix: str,
    *,
    logs: Sequence[str] = BERRY_LOGS,
) -> Optional[Dict[str, Any]]:
    """Read all three Berry phase directions of one task.

    Args:
        task: Directory of the displaced calculation.
        suffix: Output suffix of the calculation.
        logs: Log names of the three directions.

    Returns:
        Mapping with the per-direction vectors and the cell volume, or
        ``None`` when the task is incomplete.
    """
    output = task / f"OUT.{suffix}"
    try:
        values = [read_berry_polarization(output / name) for name in logs]
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"  warning: skipping incomplete BEC task {task}: {error}")
        return None
    return {
        "p_vec": [item["p_vec"] for item in values],
        "mod": [item["mod"] for item in values],
        "polarization_cm2": [item["polarization_cm2"] for item in values],
        "volume": values[0]["volume"],
        "directions": [item["direction"] for item in values],
    }


def task_metrics(task: Path, version: str) -> Dict[str, Any]:
    """Collect the available SCF metrics of one task without failing it.

    Args:
        task: Directory of the calculation.
        version: ABACUS version hint for the running-log profiler.

    Returns:
        The collected metrics, or a mapping holding the failure message.
    """
    from abacustools.data.abacus_result import get_result_from_job

    try:
        return get_result_from_job(
            task,
            ["energy", "drho", "denergy", "scf_steps", "converged"],
            version,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        return {"error": str(error)}
