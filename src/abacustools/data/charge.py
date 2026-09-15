"""Job-level assembly of ABACUS charge densities.

The grid containers and the file formats live in
:mod:`abacustools.data.grid`; this module answers the job-level questions that
every consumer of a density shares: which files of a job hold the charge
density, how the spin channels combine, and whether two grids may be combined
point by point.

Densities are returned as :class:`~abacustools.data.grid.Charge`, which keeps
the data in e/Angstrom**3 and the cell in Angstrom; the cube writer converts
back to the ABACUS units of e/Bohr**3 and Bohr.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from ase.data import atomic_numbers

from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.data.abacus_result import get_result_from_job
from abacustools.data.grid import Charge, RestartCharge
from abacustools.data.grid_files import (
    LTS,
    GridFile,
    GridFileError,
    grid_files,
    output_directory,
    scan_grid_files,
    spin_channels,
)
from abacustools.io.abacus import ReadInput
from abacustools.io.pseudo import UPF
from abacustools.io.stru import AbacusSTRU


BOHR2A = BOHR_TO_ANG

#: Lattice directions used by the profile and slice helpers.
AXES = ("a", "b", "c")

#: ABACUS LTS logs write "fft grid", develop writes "FFT grid".
_FFT_GRID_PATTERN = re.compile(
    r"fft grid for charge/potential\s*=\s*\[([^\]]+)\]", re.IGNORECASE
)

#: Spin-resolved quantities that :func:`select_spin` can return.
SPIN_CHOICES = ("total", "up", "down", "difference")


class ChargeDensityError(RuntimeError):
    """Raised when the charge density of a job cannot be assembled."""


@dataclass
class DensitySource:
    """Files that hold the spin-resolved density of one ABACUS job.

    Attributes:
        kind: ``"cube"`` for charge-density cube files or ``"restart"`` for a
            ``*-CHARGE-DENSITY.restart`` file.
        paths: One cube file per spin channel, or the single restart file.
        nspin: Number of spin channels that the INPUT of the job declares.
        naming: ``"lts"`` or ``"develop"`` for cube files, which name the same
            quantity differently.
        step: Geometry step of a develop cube file, ``None`` when the file is
            not tied to a geometry step.
        grid: FFT grid a restart file was converted with, when it is known.
    """

    kind: str
    paths: List[Path]
    nspin: int
    naming: str = LTS
    step: Optional[int] = None
    grid: Optional[Tuple[int, int, int]] = None

    def describe(self, grid: Optional[Tuple[int, int, int]] = None) -> str:
        """Return a short description of the source for reports.

        Args:
            grid: FFT grid shape a restart file was converted with, when it is
                known.

        Returns:
            ``"cube"`` for the LTS cube files, the naming convention and file
            names for develop cubes, otherwise ``restart (file, grid=...)``.
        """
        if self.kind == "cube":
            if self.naming == LTS:
                return "cube"
            details = [self.naming]
            if self.step is not None:
                details.append(f"step {self.step}")
            names = ", ".join(path.name for path in self.paths)
            return f"cube ({', '.join(details)}: {names})"
        known = grid if grid is not None else self.grid
        shape = "" if known is None else f", grid={tuple(known)}"
        return f"restart ({self.paths[0].name}{shape})"


def charge_cubes(
    job: Path,
    outdir: Path,
    nspin: int,
    cube: Optional[str] = None,
) -> Optional[List[GridFile]]:
    """Return the charge-density cubes of a job, or ``None`` when missing.

    Both ABACUS naming conventions are recognised: the LTS branch writes
    ``SPIN{index}_CHG.cube`` and the develop branch ``chg.cube`` or
    ``chgs{index}.cube``, optionally with a geometry step of ``out_freq_ion``.

    Args:
        job: ABACUS job directory, used to resolve a relative ``cube``.
        outdir: ``OUT.<suffix>`` directory of the job.
        nspin: Number of spin channels that INPUT declares.
        cube: Explicit cube file or directory, which overrides ``outdir``.

    Returns:
        The cube file of every spin channel, or ``None`` when the job has no
        complete set of charge-density cubes.
    """
    if cube is not None:
        path = Path(cube)
        if not path.is_absolute():
            path = Path(job) / path
        if path.is_dir():
            try:
                found = scan_grid_files(path, quantity="charge")
            except GridFileError:
                return None
            return found or None
        return [GridFile(path, "charge", LTS, spin=1)]

    try:
        return grid_files(outdir, "charge", spins=spin_channels(nspin))
    except GridFileError:
        return None


def cube_paths(
    job: Path,
    outdir: Path,
    nspin: int,
    cube: Optional[str] = None,
) -> Optional[List[Path]]:
    """Return the paths of the charge-density cubes of a job, or ``None``.

    This is :func:`charge_cubes` without the naming details, for callers that
    only need the files.
    """
    found = charge_cubes(job, outdir, nspin, cube)
    if found is None:
        return None
    return [item.path for item in found]


def restart_files(outdir: Path) -> List[Path]:
    """Return the charge-density restart files below an output directory."""
    return sorted(Path(outdir).glob("*-CHARGE-DENSITY.restart"))


def find_density_source(
    job: Path,
    inputs: Mapping[str, Any],
    *,
    cube: Optional[str] = None,
) -> DensitySource:
    """Locate the density of one job, preferring cube files over restarts.

    Args:
        job: ABACUS job directory.
        inputs: Parsed INPUT of the job.
        cube: Explicit cube file or directory, relative to ``job``.

    Returns:
        The :class:`DensitySource` that holds the density.

    Raises:
        ChargeDensityError: If the job has neither cube nor restart files.
    """
    job_path = Path(job)
    nspin = int(inputs.get("nspin", 1))
    outdir = output_directory(job_path, inputs)
    found = charge_cubes(job_path, outdir, nspin, cube)
    if found is not None:
        return DensitySource(
            "cube",
            [item.path for item in found],
            nspin,
            naming=found[0].naming,
            step=found[0].step,
        )
    restarts = restart_files(outdir)
    if not restarts:
        raise ChargeDensityError(
            f"no charge density in {outdir}: expected charge-density cubes "
            "(SPIN*_CHG.cube in the LTS branch, chg*.cube in develop) or "
            "*-CHARGE-DENSITY.restart"
        )
    return DensitySource("restart", [restarts[0]], nspin)


def fft_grid_from_log(log_path: Path) -> Optional[Tuple[int, int, int]]:
    """Read the charge/potential FFT grid dimensions from an ABACUS log.

    Args:
        log_path: Path of a ``running_*.log`` file.

    Returns:
        The grid dimensions, or ``None`` when the file is missing or does not
        report them.
    """
    path = Path(log_path)
    if not path.is_file():
        return None
    match = _FFT_GRID_PATTERN.search(path.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        return None
    values = re.findall(r"\d+", match.group(1))
    if len(values) != 3:
        return None
    return (int(values[0]), int(values[1]), int(values[2]))


def job_fft_grid(job: Path, inputs: Mapping[str, Any]) -> Optional[Tuple[int, int, int]]:
    """Find the FFT grid of a job in the log of its calculation.

    A job can keep several running logs, and an earlier one may describe a
    different grid, such as a relaxation whose cell axes were ordered
    differently. The log of the current ``calculation`` therefore wins, and the
    other logs are only fallbacks.

    Args:
        job: ABACUS job directory.
        inputs: Parsed INPUT of the job.

    Returns:
        The FFT grid, or ``None`` when no log reports one.
    """
    outdir = output_directory(job, inputs)
    calculation = str(inputs.get("calculation", "scf")).lower()
    candidates = [outdir / f"running_{calculation}.log", outdir / "running_scf.log"]
    candidates.extend(sorted(outdir.glob("running_*.log")))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        shape = fft_grid_from_log(candidate)
        if shape is not None:
            return shape
    return None


def valence_electrons(
    structure,
    pseudo_dir: Optional[str],
    job: Path,
) -> List[float]:
    """Return the number of valence electrons of every atom of a structure.

    ABACUS resolves the pseudopotential names of the STRU below ``pseudo_dir``
    when INPUT defines it, and in the job directory otherwise.

    Args:
        structure: Structure whose pseudopotential files are read.
        pseudo_dir: ``pseudo_dir`` of INPUT, or ``None``.
        job: ABACUS job directory.

    Returns:
        One valence charge per atom.

    Raises:
        ChargeDensityError: If a pseudopotential is missing or cannot be read.
    """
    directory = Path(job)
    if pseudo_dir:
        candidate = Path(str(pseudo_dir))
        directory = candidate if candidate.is_absolute() else Path(job) / candidate
    cache: dict = {}
    valences: List[float] = []
    for pp in structure.pps:
        if pp is None:
            raise ChargeDensityError("STRU atom is missing a pseudopotential filename")
        if pp not in cache:
            path = directory / str(pp)
            if not path.is_file():
                raise ChargeDensityError(f"pseudopotential file not found: {path}")
            cache[pp] = float(UPF.read_from_file(path).z_valence)
        valences.append(cache[pp])
    return valences


def element_numbers(elements: Sequence[Optional[str]]) -> List[int]:
    """Return the atomic number of every element of a structure.

    Raises:
        ChargeDensityError: If an element is unknown.
    """
    numbers = []
    for element in elements:
        if element is None or element not in atomic_numbers:
            raise ChargeDensityError(f"unknown element in STRU: {element!r}")
        numbers.append(int(atomic_numbers[element]))
    return numbers


def check_channel_count(source: DensitySource, found: int) -> None:
    """Check that the density files match the ``nspin`` of INPUT.

    Raises:
        ChargeDensityError: If the channel counts differ.
    """
    if found != source.nspin:
        raise ChargeDensityError(
            f"found {found} spin channel(s) but INPUT has nspin={source.nspin}"
        )


def read_cube_charges(source: DensitySource) -> List[Charge]:
    """Read one :class:`Charge` per cube file of a source.

    Args:
        source: Cube density source.

    Returns:
        The spin channels of the density, in cube order.

    Raises:
        ChargeDensityError: If the cube count differs from ``nspin``.
    """
    charges = [Charge.from_cube(str(path), format="abacus") for path in source.paths]
    check_channel_count(source, len(charges))
    return charges


def read_restart_charges(
    source: DensitySource,
    *,
    structure,
    valences: Sequence[float],
    grid_shape: Tuple[int, int, int],
    lat0: float = ANG_TO_BOHR,
) -> List[Charge]:
    """Convert a ``*-CHARGE-DENSITY.restart`` file into real-space densities.

    Args:
        source: Restart density source.
        structure: Structure that provides the coordinates and elements.
        valences: Number of valence electrons per atom, written to the atom
            column of the cube.
        grid_shape: FFT grid the reciprocal density is transformed onto.
        lat0: ``LATTICE_CONSTANT`` of the job in Bohr.

    Returns:
        One :class:`Charge` per spin channel of the restart file.

    Raises:
        ChargeDensityError: If the file stores more than two spin channels or
            contradicts the ``nspin`` of INPUT.
    """
    restart = RestartCharge.read(str(source.paths[0]))
    if restart.nspin > 2:
        raise ChargeDensityError(f"nspin={restart.nspin} is not supported (only 1 and 2)")
    real = restart.to_real(grid_shape)
    cell_angstrom = np.linalg.inv(restart.reciprocal_lattice) * lat0 * BOHR2A
    positions = np.asarray(structure.coords, dtype=float)
    numbers = element_numbers(structure.elements)
    charges = [float(value) for value in valences]
    spin_charges = [
        Charge(real[ispin] / BOHR2A**3, cell_angstrom, positions, numbers, charges)
        for ispin in range(restart.nspin)
    ]
    check_channel_count(source, len(spin_charges))
    return spin_charges


def validate_same_grid(reference: Charge, other: Charge, description: str) -> None:
    """Ensure two densities can be combined point by point.

    Args:
        reference: Density that defines the grid.
        other: Density that is compared against it.
        description: Name of the combination, used in error messages.

    Raises:
        ChargeDensityError: If the grids differ in shape or geometry.
    """
    if reference.data.shape != other.data.shape:
        raise ChargeDensityError(
            f"incompatible grid shape for {description}: "
            f"{reference.data.shape} != {other.data.shape}"
        )
    if not np.allclose(reference.cell, other.cell) or not np.allclose(
        reference.origin, other.origin
    ):
        raise ChargeDensityError(f"incompatible grid geometry for {description}")


def combine(
    first: Charge,
    second: Charge,
    sign: float,
    *,
    description: str = "spin channels",
) -> Charge:
    """Return ``first + sign * second`` on the grid of the first density.

    Args:
        first: Density that defines the grid.
        second: Density that is added or subtracted.
        sign: ``+1`` to add and ``-1`` to subtract.
        description: Name of the combination, used in error messages.
    """
    validate_same_grid(first, second, description)
    return Charge(
        first.data + sign * second.data,
        first.cell,
        first.atom_positions,
        first.atom_types,
        first.atom_charges,
        first.origin,
    )


def total_charge(channels: Sequence[Charge]) -> Charge:
    """Sum the spin channels of a density into the total density.

    Raises:
        ChargeDensityError: If no channel is given.
    """
    if not channels:
        raise ChargeDensityError("no charge-density channel to combine")
    total = channels[0]
    for extra in channels[1:]:
        total = combine(total, extra, 1.0)
    return total


@dataclass
class JobDensity:
    """The assembled density of one ABACUS job.

    Attributes:
        job: Job directory the density was read from.
        source: Files that hold the density.
        channels: One :class:`Charge` per spin channel.
    """

    job: Path
    source: DensitySource
    channels: List[Charge]

    @property
    def nspin(self) -> int:
        """Number of spin channels the density holds."""
        return len(self.channels)

    def total(self) -> Charge:
        """Return the sum of the spin channels."""
        return total_charge(self.channels)


def _read_restart_density(
    job: Path,
    inputs: Mapping[str, Any],
    source: DensitySource,
    *,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    lat0: Optional[float] = None,
) -> JobDensity:
    """Convert a ``*-CHARGE-DENSITY.restart`` file into real-space densities.

    The file stores ``rho(G)`` only, so the conversion needs the FFT grid of the
    calculation and the cell of the job; the valence charges that the cube
    columns carry are read from the pseudopotentials the STRU names.
    """
    structure_file = job / str(inputs.get("stru_file", "STRU"))
    if not structure_file.is_file():
        raise ChargeDensityError(
            f"cannot convert {source.paths[0].name}: the structure "
            f"{structure_file} is missing"
        )
    structure = AbacusSTRU.read(str(structure_file))
    if structure is None:
        raise ChargeDensityError(f"cannot read the structure: {structure_file}")

    shape = grid_shape or job_fft_grid(job, inputs)
    if shape is None:
        raise ChargeDensityError(
            f"the restart file of {job} does not report the FFT grid; keep the "
            "running log of the calculation or pass --grid NX NY NZ"
        )
    constant = lat0
    if constant is None:
        constant = float(structure.metadata.get("lattice_constant", 1.0) or 1.0)
    valences = valence_electrons(structure, inputs.get("pseudo_dir"), job)
    channels = read_restart_charges(
        source,
        structure=structure,
        valences=valences,
        grid_shape=shape,
        lat0=constant,
    )
    source.grid = shape
    return JobDensity(job, source, channels)


def read_job_density(
    job: Path,
    *,
    version: Optional[str] = None,
    require_converged: bool = False,
    description: str = "charge-density assembly",
    grid_shape: Optional[Tuple[int, int, int]] = None,
    lat0: Optional[float] = None,
) -> JobDensity:
    """Assemble the spin-resolved density of one job.

    The density is read from the charge-density cubes of either branch
    (``SPIN*_CHG.cube`` or ``chg*.cube``). A job that only stores a
    ``*-CHARGE-DENSITY.restart`` file is converted from ``rho(G)`` instead,
    which needs its structure and the FFT grid of the calculation.

    Args:
        job: ABACUS job directory.
        version: ABACUS version hint used when checking the SCF convergence.
        require_converged: Refuse a job whose SCF calculation did not converge.
        description: Name of the consumer, used in error messages.
        grid_shape: FFT grid of a restart file, read from the log when omitted.
        lat0: ``LATTICE_CONSTANT`` of the job in Bohr, taken from the STRU when
            omitted.

    Returns:
        The assembled :class:`JobDensity`.

    Raises:
        ChargeDensityError: If the job has no usable charge density.
    """
    job_path = Path(job)
    inputs = ReadInput(str(job_path / "INPUT"))
    nspin = int(inputs.get("nspin", 1))
    if nspin not in (1, 2):
        raise ChargeDensityError(
            f"{description} supports only nspin 1 and 2, but {job_path} declares "
            f"nspin {nspin}"
        )
    if require_converged:
        result = get_result_from_job(str(job_path), ["converged"], version)
        if not result["converged"]:
            raise ChargeDensityError(f"SCF calculation did not converge: {job_path}")
    source = find_density_source(job_path, inputs)
    if source.kind == "cube":
        return JobDensity(job_path, source, read_cube_charges(source))
    return _read_restart_density(
        job_path, inputs, source, grid_shape=grid_shape, lat0=lat0
    )


def read_job_total_density(
    job: Path,
    *,
    version: Optional[str] = None,
    require_converged: bool = False,
    description: str = "charge-density assembly",
    grid_shape: Optional[Tuple[int, int, int]] = None,
    lat0: Optional[float] = None,
) -> Charge:
    """Return the total charge density of one job as a single grid.

    This is :func:`read_job_density` for callers that do not need to know where
    the density came from.

    Args:
        job: ABACUS job directory.
        version: ABACUS version hint used when checking the SCF convergence.
        require_converged: Refuse a job whose SCF calculation did not converge.
        description: Name of the consumer, used in error messages.

    Returns:
        The total density in e/Angstrom**3 on an Angstrom cell.

    Raises:
        ChargeDensityError: If the job has no usable charge density.
    """
    return read_job_density(
        job,
        version=version,
        require_converged=require_converged,
        description=description,
        grid_shape=grid_shape,
        lat0=lat0,
    ).total()


def integrate(density: Charge, *, compare_valence: bool = True) -> dict:
    """Integrate a density and compare it with its valence electrons.

    The atom column of a cube written by ABACUS holds the number of valence
    electrons per atom, so the integral of a neutral cell is that sum. A
    deviation reports either a charged cell or the truncation of the real-space
    grid.

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.
        compare_valence: Report the valence comparison. It is meaningless for a
            derived density such as a difference, which integrates to zero for
            two neutral cells.

    Returns:
        A mapping with the grid shape, the cell volume in Angstrom**3, the
        integrated number of electrons, the sum of the valence charges (``None``
        when the grid carries none or the comparison is off) and their
        difference.
    """
    volume = abs(float(np.linalg.det(np.asarray(density.cell, dtype=float))))
    weight = volume / density.data.size
    electrons = float(np.sum(density.data) * weight)
    valence = [float(value) for value in density.atom_charges]
    expected = (
        float(sum(valence))
        if compare_valence and any(value != 0.0 for value in valence)
        else None
    )
    return {
        "grid": [int(size) for size in density.data.shape],
        "volume_angstrom3": volume,
        "electrons": electrons,
        "valence_electrons": expected,
        "deviation": None if expected is None else electrons - expected,
    }


def select_spin(density: "JobDensity", spin: str = "total") -> Charge:
    """Return one spin-resolved quantity of an assembled density.

    Args:
        density: Assembled job density.
        spin: ``"total"`` for the sum of the channels, ``"up"`` or ``"down"``
            for a single channel, ``"difference"`` for up minus down.

    Returns:
        The selected density.

    Raises:
        ChargeDensityError: If the choice is unknown or needs more channels.
    """
    if spin == "total":
        return density.total()
    if spin not in SPIN_CHOICES:
        raise ChargeDensityError(f"unknown spin choice: {spin!r}")
    if density.nspin < 2:
        raise ChargeDensityError(
            f"the {spin!r} density needs an nspin 2 calculation, but the job has "
            f"{density.nspin} spin channel(s)"
        )
    if spin == "up":
        return density.channels[0]
    if spin == "down":
        return density.channels[1]
    return combine(density.channels[0], density.channels[1], -1.0)


def subtract(first: Charge, second: Charge, description: str) -> Charge:
    """Return ``first - second`` after checking that both share one grid."""
    return combine(first, second, -1.0, description=description)


@dataclass
class PlaneSlice:
    """One plane of a density.

    Attributes:
        axis: Lattice direction the plane is perpendicular to.
        index: Grid index of the plane.
        position: Fractional coordinate of the plane in [0, 1).
        distance: Distance along ``axis`` in Angstrom.
        values: Values of the plane, with the shape of the two other axes.
        coordinates: Angstrom coordinates of the two other axes.
        labels: Names of the two other axes, in the order of ``values``.
    """

    axis: str
    index: int
    position: float
    distance: float
    values: np.ndarray
    coordinates: Tuple[np.ndarray, np.ndarray]
    labels: Tuple[str, str]


def slice_plane(density: Charge, axis: str, position: float = 0.5) -> PlaneSlice:
    """Cut one plane out of a density.

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.
        axis: Lattice direction perpendicular to the plane.
        position: Fractional coordinate of the plane along ``axis``, rounded to
            the closest grid plane.

    Returns:
        The requested :class:`PlaneSlice`.

    Raises:
        ChargeDensityError: If the axis is unknown or the position is not a
            fractional coordinate in [0, 1).
    """
    if axis not in AXES:
        raise ChargeDensityError(f"unknown slice axis: {axis!r}")
    if not 0.0 <= position < 1.0:
        raise ChargeDensityError(
            f"the slice position must be a fractional coordinate in [0, 1): {position}"
        )
    axis_index = AXES.index(axis)
    size = int(density.data.shape[axis_index])
    index = int(round(position * size)) % size
    values = np.asarray(np.take(density.data, index, axis=axis_index), dtype=float)
    labels = tuple(letter for letter in AXES if letter != axis)
    cell = np.asarray(density.cell, dtype=float)
    coordinates = tuple(
        np.linspace(
            0.0,
            float(np.linalg.norm(cell[AXES.index(letter)])),
            values.shape[position_index],
            endpoint=False,
        )
        for position_index, letter in enumerate(labels)
    )
    distance = float(index * np.linalg.norm(cell[axis_index]) / size)
    return PlaneSlice(
        axis=axis,
        index=index,
        position=index / size,
        distance=distance,
        values=values,
        coordinates=coordinates,
        labels=labels,
    )


def atoms_in_plane(
    density: Charge,
    plane: PlaneSlice,
    structure,
    tolerance: Optional[float] = None,
) -> List[dict]:
    """Return the atoms that the given grid plane crosses.

    Args:
        density: Density whose cell defines the fractional coordinates.
        plane: Plane that selects the atoms.
        structure: Structure whose atoms are tested.
        tolerance: Half thickness of the plane in fractional coordinates,
            defaulting to half a grid spacing.

    Returns:
        One mapping per atom with its label, its two in-plane coordinates in
        Angstrom and its distance to the plane in Angstrom.
    """
    cell = np.asarray(density.cell, dtype=float)
    axis_index = AXES.index(plane.axis)
    other = [index for index in range(3) if index != axis_index]
    if tolerance is None:
        tolerance = 0.5 / density.data.shape[axis_index]
    atoms = []
    for atom in structure.atoms:
        fractional = np.linalg.solve(cell.T, np.asarray(atom.coord, dtype=float))
        delta = float(fractional[axis_index]) - plane.position
        delta -= round(delta)
        if abs(delta) > tolerance:
            continue
        atoms.append(
            {
                "label": atom.label,
                "coordinates": [
                    float(fractional[index] * np.linalg.norm(cell[index]))
                    for index in other
                ],
                "distance": float(delta * np.linalg.norm(cell[axis_index])),
            }
        )
    return atoms


def planar_profile(
    density: Charge,
    axis: str,
    *,
    kind: str = "average",
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a density to one value per plane along a lattice direction.

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.
        axis: Lattice direction ``"a"``, ``"b"`` or ``"c"``.
        kind: ``"average"`` for the in-plane average in e/Angstrom**3, or
            ``"integral"`` for the charge of one plane in e, whose sum over the
            planes is the total number of electrons.

    Returns:
        The value of every plane and the distance along ``axis`` in Angstrom.

    Raises:
        ChargeDensityError: If the axis or the kind is unknown.
    """
    if axis not in ("a", "b", "c"):
        raise ChargeDensityError(f"unknown profile axis: {axis!r}")
    if kind not in ("average", "integral"):
        raise ChargeDensityError(f"unknown profile kind: {kind!r}")
    values, distances = density.profile1d(
        axis, average=(kind == "average"), cartesian=True
    )
    values = np.asarray(values, dtype=float)
    if kind == "integral":
        volume = abs(float(np.linalg.det(np.asarray(density.cell, dtype=float))))
        values = values * volume / density.data.size
    return values, np.asarray(distances, dtype=float)
