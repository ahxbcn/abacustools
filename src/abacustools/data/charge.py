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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from ase.data import atomic_numbers

from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.data.abacus_result import get_result_from_job
from abacustools.data.grid import Charge, RestartCharge
from abacustools.io.abacus import ReadInput


BOHR2A = BOHR_TO_ANG


class ChargeDensityError(RuntimeError):
    """Raised when the charge density of a job cannot be assembled."""


@dataclass
class DensitySource:
    """Files that hold the spin-resolved density of one ABACUS job.

    Attributes:
        kind: ``"cube"`` for ``SPIN*_CHG.cube`` files or ``"restart"`` for a
            ``*-CHARGE-DENSITY.restart`` file.
        paths: One cube file per spin channel, or the single restart file.
        nspin: Number of spin channels that the INPUT of the job declares.
    """

    kind: str
    paths: List[Path]
    nspin: int

    def describe(self, grid: Optional[Tuple[int, int, int]] = None) -> str:
        """Return a short description of the source for reports.

        Args:
            grid: FFT grid shape a restart file was converted with, when it is
                known.

        Returns:
            ``"cube"`` for cube files, otherwise ``restart (file, grid=...)``.
        """
        if self.kind == "cube":
            return "cube"
        shape = "" if grid is None else f", grid={tuple(grid)}"
        return f"restart ({self.paths[0].name}{shape})"


def output_directory(job: Path, inputs: Mapping[str, Any]) -> Path:
    """Return the ``OUT.<suffix>`` directory that INPUT points at."""
    return Path(job) / f"OUT.{inputs.get('suffix', 'ABACUS')}"


def cube_paths(
    job: Path,
    outdir: Path,
    nspin: int,
    cube: Optional[str] = None,
) -> Optional[List[Path]]:
    """Return the cube files of a job, or ``None`` when the job has none.

    Args:
        job: ABACUS job directory, used to resolve a relative ``cube``.
        outdir: ``OUT.<suffix>`` directory of the job.
        nspin: Number of spin channels that INPUT declares.
        cube: Explicit cube file or directory, which overrides ``outdir``.

    Returns:
        The cube file of every spin channel, or ``None``.
    """
    if cube is not None:
        path = Path(cube)
        if not path.is_absolute():
            path = Path(job) / path
        if path.is_dir():
            found = sorted(path.glob("SPIN*_CHG.cube"))
            return found or None
        return [path]
    expected = [Path(outdir) / f"SPIN{index + 1}_CHG.cube" for index in range(nspin)]
    if all(path.is_file() for path in expected):
        return expected
    return None


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
    found = cube_paths(job_path, outdir, nspin, cube)
    if found is not None:
        return DensitySource("cube", list(found), nspin)
    restarts = restart_files(outdir)
    if not restarts:
        raise ChargeDensityError(
            f"no charge density in {outdir}: expected SPIN*_CHG.cube or "
            "*-CHARGE-DENSITY.restart"
        )
    return DensitySource("restart", [restarts[0]], nspin)


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


def combine(first: Charge, second: Charge, sign: float) -> Charge:
    """Return ``first + sign * second`` on the grid of the first density."""
    validate_same_grid(first, second, "spin channels")
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


def read_job_total_density(
    job: Path,
    *,
    version: Optional[str] = None,
    require_converged: bool = False,
    description: str = "charge-density assembly",
) -> Charge:
    """Return the total charge density of one job as a single grid.

    The density is read from ``SPIN*_CHG.cube``. A job that only stores a
    ``*-CHARGE-DENSITY.restart`` file is reported as unsupported here, because
    converting it needs the FFT grid of the calculation.

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
    if source.kind != "cube":
        raise ChargeDensityError(
            f"{job_path} has no SPIN*_CHG.cube; only {source.paths[0].name} was "
            "found, which needs the FFT grid to be converted"
        )
    return total_charge(read_cube_charges(source))
