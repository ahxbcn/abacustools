"""Read ABACUS molecular-dynamics trajectories.

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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from abacustools.data.abacus_result import read_md_history
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
        velocities = None
        if any(atom.velocity is not None for atom in structure.atoms):
            velocities = np.asarray(
                [
                    atom.velocity if atom.velocity is not None else (0.0, 0.0, 0.0)
                    for atom in structure.atoms
                ],
                dtype=float,
            )
        frames.append(
            TrajectoryFrame(
                step=int(match.group(1)),
                cell=np.asarray(structure.cell, dtype=float),
                symbols=[str(element) for element in structure.elements],
                positions=np.asarray(structure.coords, dtype=float),
                velocities=velocities,
            )
        )
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
