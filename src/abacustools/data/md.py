"""Read ABACUS ionic trajectories (molecular dynamics and relaxation).

ABACUS appends one block per dumped step to ``OUT.<suffix>/MD_dump`` holding
the cell and the atomic positions together with, depending on ``dump_force``,
``dump_vel`` and ``dump_virial``, the forces, the velocities and the virial:

    MDSTEP:  0
    LATTICE_CONSTANT: 1.889726 Angstrom
    LATTICE_VECTORS
      <three rows in units of the lattice constant>
    VIRIAL (kbar)
      <three rows, when the stress is dumped>
    INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
      <index> <label> <x> <y> <z> [<fx> <fy> <fz>] [<vx> <vy> <vz>]

The per-step ``STRU_MD_*`` files that ``out_stru 1`` writes are read as a
fallback: the LTS branch keeps them in the job directory, the develop branch in
a directory per step below ``OUT.<suffix>``. The energy, temperature and
pressure of the running log are attached to the frame of the same step.

Geometry optimizations are read from the per-step structures that
``out_stru`` writes: ``STRU_ION<step>_D`` on the 3.10 LTS branch and
``STRU<step>`` on the develop branch. When those files are missing, the
per-step coordinates and cell printed in the
``running_relax.log``/``running_cell-relax.log`` are used instead. The energy
of every step is attached from the same running log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from abacustools.data.abacus_result import (
    read_md_history,
    read_relax_structures,
    read_relaxation_history,
)
from abacustools.data.charge import output_directory
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


_STEP_PATTERN = re.compile(r"(\d+)\s*$")
_SYMBOL_PATTERN = re.compile(r"[A-Za-z]{1,2}")


@dataclass
class TrajectoryFrame:
    """One dumped step of a molecular-dynamics run.

    Attributes:
        step: MD step number as ABACUS writes it.
        cell: Cell vectors in Angstrom, one row per vector.
        symbols: Element symbol of every atom.
        positions: Cartesian positions in Angstrom, shape ``(natoms, 3)``.
        forces: Forces in eV/Angstrom, or ``None`` when they were not dumped.
        velocities: Velocities in Angstrom/fs, or ``None``.
        virial: Virial in kBar, or ``None``.
        energy: Potential energy of the step in eV, from the running log.
        temperature: Temperature of the step in K, from the running log.
        pressure: Pressure of the step in kBar, from the running log.
    """

    step: int
    cell: np.ndarray
    symbols: List[str]
    positions: np.ndarray
    forces: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    virial: Optional[np.ndarray] = None
    energy: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None

    @property
    def natoms(self) -> int:
        """Number of atoms of the frame."""
        return len(self.symbols)


def _symbol(label: str) -> str:
    """Return the element symbol of an ABACUS label such as ``Si1``."""
    match = _SYMBOL_PATTERN.match(label.strip())
    if match is None:
        return label.strip()
    return match.group(0).capitalize()


def _rows_of_floats(lines: Sequence[str], start: int, count: int) -> np.ndarray:
    """Read ``count`` rows of floats starting at ``start``."""
    values = []
    for offset in range(count):
        fields = lines[start + offset].split()
        values.append([float(field) for field in fields[:3]])
    return np.asarray(values, dtype=float)


def _frame_from_structure(step: int, structure: AbacusSTRU) -> TrajectoryFrame:
    """Build a trajectory frame from one parsed ABACUS structure."""
    velocities = None
    if any(atom.velocity is not None for atom in structure.atoms):
        velocities = np.asarray(
            [
                atom.velocity if atom.velocity is not None else (0.0, 0.0, 0.0)
                for atom in structure.atoms
            ],
            dtype=float,
        )
    forces = None
    if structure.atoms and all(atom.force is not None for atom in structure.atoms):
        forces = np.asarray([atom.force for atom in structure.atoms], dtype=float)
    return TrajectoryFrame(
        step=step,
        cell=np.asarray(structure.cell, dtype=float),
        symbols=[str(element) for element in structure.elements],
        positions=np.asarray(structure.coords, dtype=float),
        forces=forces,
        velocities=velocities,
    )


#: Per-step structures written by the relaxation driver, newest naming first.
_RELAX_STEP_FILES = (
    re.compile(r"^STRU_ION(\d+)_D$"),  # ABACUS 3.10 LTS
    re.compile(r"^STRU(\d+)$"),        # ABACUS develop
)


def read_md_dump(path: Path) -> List[TrajectoryFrame]:
    """Read the trajectory blocks of an ABACUS ``MD_dump`` file.

    Args:
        path: ``MD_dump`` file of an MD calculation.

    Returns:
        One frame per dumped step, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a block is malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"could not find the MD dump: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: List[TrajectoryFrame] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("MDSTEP"):
            index += 1
            continue
        try:
            step = int(line.split(":")[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid MD step line {index + 1}: {line}") from error
        index += 1

        constant = None
        cell = None
        virial = None
        has_force = False
        has_velocity = False
        positions: List[List[float]] = []
        forces: List[List[float]] = []
        velocities: List[List[float]] = []
        labels: List[str] = []

        while index < len(lines):
            line = lines[index].strip()
            if line.startswith("MDSTEP"):
                break
            if line.startswith("LATTICE_CONSTANT"):
                constant = float(line.split(":")[1].split()[0])
                index += 1
                continue
            if line.startswith("LATTICE_VECTORS"):
                cell = _rows_of_floats(lines, index + 1, 3)
                index += 4
                continue
            if line.startswith("VIRIAL"):
                virial = _rows_of_floats(lines, index + 1, 3)
                index += 4
                continue
            if line.startswith("INDEX"):
                has_force = "FORCE" in line
                has_velocity = "VELOCITY" in line
                index += 1
                continue
            fields = line.split()
            if len(fields) >= 5:
                labels.append(_symbol(fields[1]))
                positions.append([float(value) for value in fields[2:5]])
                cursor = 5
                if has_force:
                    forces.append([float(value) for value in fields[cursor : cursor + 3]])
                    cursor += 3
                if has_velocity:
                    velocities.append(
                        [float(value) for value in fields[cursor : cursor + 3]]
                    )
            index += 1

        if not positions or cell is None:
            continue
        frames.append(
            TrajectoryFrame(
                step=step,
                cell=cell * (constant if constant is not None else 1.0),
                symbols=labels,
                positions=np.asarray(positions, dtype=float),
                forces=np.asarray(forces, dtype=float) if forces else None,
                velocities=np.asarray(velocities, dtype=float) if velocities else None,
                virial=virial,
            )
        )
    return frames


def _stru_md_frames(job: Path, outdir: Path) -> List[TrajectoryFrame]:
    """Read the per-step ``STRU_MD_*`` files of a job."""
    candidates: List[Path] = []
    for path in job.glob("STRU_MD_*"):
        if path.is_file():
            candidates.append(path)
    for path in outdir.glob("STRU_MD_*/STRU"):
        if path.is_file():
            candidates.append(path)

    frames = []
    for path in sorted(candidates, key=lambda item: item.parent.name if item.name == "STRU" else item.name):
        name = path.parent.name if path.name == "STRU" else path.name
        match = _STEP_PATTERN.search(name)
        if match is None:
            continue
        structure = AbacusSTRU.read(str(path))
        if structure is None:
            continue
        frames.append(_frame_from_structure(int(match.group(1)), structure))
    return sorted(frames, key=lambda frame: frame.step)


def read_trajectory(
    job: Path,
    *,
    version: Optional[str] = None,
    with_log: bool = True,
) -> List[TrajectoryFrame]:
    """Read the trajectory of an ABACUS molecular-dynamics job.

    Args:
        job: ABACUS job directory.
        version: ABACUS version hint for the running log.
        with_log: Attach the energy, temperature and pressure of every step from
            the running log.

    Returns:
        The frames of the job, ordered by step.

    Raises:
        FileNotFoundError: If the job has neither ``MD_dump`` nor step
            structures.
    """
    job = Path(job)
    inputs = ReadInput(str(job / "INPUT"))
    outdir = output_directory(job, inputs)
    dump = outdir / "MD_dump"
    if dump.is_file():
        frames = read_md_dump(dump)
    else:
        frames = _stru_md_frames(job, outdir)
    if not frames:
        raise FileNotFoundError(
            f"no MD trajectory in {outdir}: expected MD_dump or STRU_MD_* files"
        )
    frames.sort(key=lambda frame: frame.step)

    if with_log:
        log = outdir / "running_md.log"
        if log.is_file():
            try:
                history = {
                    int(record["step"]): record
                    for record in read_md_history(log, version)
                }
            except (OSError, ValueError, KeyError, TypeError):
                history = {}
            for frame in frames:
                record = history.get(frame.step)
                if record is None:
                    continue
                frame.energy = record.get("energy")
                frame.temperature = record.get("temperature")
                frame.pressure = record.get("pressure")
    return frames


def read_relax_trajectory(
    job: Path,
    *,
    version: Optional[str] = None,
    with_log: bool = True,
) -> List[TrajectoryFrame]:
    """Read the trajectory of an ABACUS relax or cell-relax job.

    The frames come from the per-step structures ``out_stru`` writes, in step
    order. When those files are missing the trajectory is rebuilt from the
    coordinates that the running relaxation log prints for every ionic step.
    The energy of each step is attached from the running log.

    Args:
        job: ABACUS job directory of a ``relax`` or ``cell-relax`` calculation.
        version: ABACUS version hint for the running log.
        with_log: Attach the energy of every step from the running log.

    Returns:
        The frames of the job, ordered by step.

    Raises:
        ValueError: If the job is not a relax or cell-relax calculation.
        FileNotFoundError: If neither the per-step structures nor a usable
            running log are available.
    """
    job = Path(job)
    inputs = ReadInput(str(job / "INPUT"))
    calculation = str(inputs.get("calculation", "scf")).lower()
    if calculation not in {"relax", "cell-relax"}:
        raise ValueError(
            f"expected a relax or cell-relax job, got calculation={calculation!r}; "
            "use 'postprocess md' for a molecular-dynamics trajectory"
        )
    outdir = output_directory(job, inputs)
    log = outdir / f"running_{calculation}.log"
    steps: Dict[int, Path] = {}
    for path in sorted(outdir.glob("*")) if outdir.is_dir() else []:
        if not path.is_file():
            continue
        for pattern in _RELAX_STEP_FILES:
            match = pattern.match(path.name)
            if match is not None:
                steps[int(match.group(1))] = path
                break
    if steps:
        frames = []
        for step in sorted(steps):
            structure = AbacusSTRU.read(str(steps[step]), fmt="stru")
            if structure is None:
                continue
            frames.append(_frame_from_structure(step, structure))
        if with_log and frames and log.is_file():
            _attach_relax_energies(frames, log, version)
        return frames

    if not log.is_file():
        raise FileNotFoundError(
            f"no per-step structures in {outdir} and no {log.name}; enable "
            "out_stru in INPUT so ABACUS writes STRU_ION<step>_D (LTS) or "
            "STRU<step> (develop), or keep the running log"
        )
    cell, symbols = _job_structure_hint(job, inputs)
    records = read_relax_structures(log, version, cell=cell, symbols=symbols)
    if not records:
        raise FileNotFoundError(
            f"the running log {log} holds no per-step structure; enable out_stru "
            "in INPUT so ABACUS writes the per-step structure files"
        )
    frames = [
        TrajectoryFrame(
            step=int(record["step"]),
            cell=np.asarray(record["cell"], dtype=float),
            symbols=list(record["symbols"]),
            positions=np.asarray(record["positions"], dtype=float),
            energy=record.get("energy") if with_log else None,
        )
        for record in records
    ]
    return frames


def _attach_relax_energies(
    frames: Sequence[TrajectoryFrame], log: Path, version: Optional[str]
) -> None:
    """Copy the energy of every frame from a running relaxation log."""
    try:
        history = {
            int(record["step"]): record
            for record in read_relaxation_history(log, version)
        }
    except (OSError, ValueError, KeyError, TypeError):
        return
    for frame in frames:
        record = history.get(frame.step)
        if record is not None:
            frame.energy = record.get("energy")


def _job_structure_hint(
    job: Path, inputs: Any
) -> tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Return the cell and symbols of the job's input structure, if it exists."""
    name = str(inputs.get("stru_file", "STRU"))
    path = Path(name)
    if not path.is_absolute():
        path = job / path
    if not path.is_file():
        return None, None
    structure = AbacusSTRU.read(str(path), fmt="stru")
    if structure is None:
        return None, None
    return np.asarray(structure.cell, dtype=float), [
        str(element) for element in structure.elements
    ]


def select_frames(
    frames: Sequence[TrajectoryFrame],
    *,
    first: Optional[int] = None,
    last: Optional[int] = None,
    stride: int = 1,
) -> List[TrajectoryFrame]:
    """Select frames by MD step and stride.

    Args:
        frames: Frames of a trajectory.
        first: Smallest MD step to keep.
        last: Largest MD step to keep.
        stride: Keep every ``stride``-th frame of the selection.

    Returns:
        The selected frames.

    Raises:
        ValueError: If the stride is not a positive integer.
    """
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError("the stride must be a positive integer")
    selected = [
        frame
        for frame in frames
        if (first is None or frame.step >= first) and (last is None or frame.step <= last)
    ]
    return selected[:: int(stride)]


def frames_to_atoms(frames: Sequence[TrajectoryFrame]) -> List[Any]:
    """Convert frames into ASE atoms objects.

    Args:
        frames: Frames of a trajectory.

    Returns:
        One :class:`ase.Atoms` per frame, carrying the velocities and, when the
        log or the dump provides them, the energy and the forces.
    """
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    structures = []
    for frame in frames:
        atoms = Atoms(
            symbols=frame.symbols,
            positions=frame.positions,
            cell=frame.cell,
            pbc=True,
        )
        if frame.velocities is not None and frame.velocities.shape == (frame.natoms, 3):
            atoms.set_velocities(frame.velocities)
        results: Dict[str, Any] = {}
        if frame.energy is not None:
            results["energy"] = float(frame.energy)
        if frame.forces is not None and frame.forces.shape == (frame.natoms, 3):
            results["forces"] = frame.forces
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)
        structures.append(atoms)
    return structures


def write_trajectory(
    path: Path,
    frames: Sequence[TrajectoryFrame],
    *,
    fmt: Optional[str] = None,
) -> Path:
    """Write frames to a trajectory file.

    Args:
        path: Output file, whose suffix selects the format.
        frames: Frames to write.
        fmt: Explicit ASE format, such as ``extxyz`` or ``traj``.

    Returns:
        The written path.

    Raises:
        ValueError: If no frame is given.
    """
    from ase.io import write

    if not frames:
        raise ValueError("the trajectory has no frame to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), frames_to_atoms(frames), format=fmt)
    return path


def _largest_finite(values: Sequence[Optional[float]]) -> Optional[float]:
    """Return the largest finite value, or ``None`` when none qualifies."""
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.max())


def summarize_trajectory(frames: Sequence[TrajectoryFrame]) -> Dict[str, Any]:
    """Summarise the contents of a trajectory.

    Args:
        frames: Frames of a trajectory.

    Returns:
        A report with the frame and atom counts, the first and last step, the
        physical quantities the frames carry and the highest temperature.
    """
    return {
        "frames": len(frames),
        "atoms": frames[0].natoms if frames else 0,
        "steps": [frames[0].step, frames[-1].step] if frames else [],
        "has_forces": any(frame.forces is not None for frame in frames),
        "has_velocities": any(frame.velocities is not None for frame in frames),
        "has_virial": any(frame.virial is not None for frame in frames),
        "has_energy": any(frame.energy is not None for frame in frames),
        "highest_temperature": _largest_finite(
            frame.temperature for frame in frames
        ),
    }
