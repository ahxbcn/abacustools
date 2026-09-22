"""Adapter that runs the optional ``baderkit`` library for a Bader partition.

``baderkit`` (https://github.com/SWeavz/baderkit) is a Python reimplementation
of the Henkelman grid-based Bader program: it partitions the same real-space
charge density into Bader volumes and integrates the density over them, so it is
an alternative to the external ``bader`` executable behind
``abacustools postprocess bader --backend baderkit``.

The package is an optional dependency: it is imported inside the function that
runs it, and importing this module never imports it.

``baderkit`` 0.10 reads a Gaussian cube file with its axes reversed, so the
density of an ABACUS cube comes out permuted relative to its cell. This adapter
therefore does not use the file reader: it builds the ``baderkit`` grid from the
assembled :class:`~abacustools.data.grid.Charge`, which is the same density the
external program receives through a cube file.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from abacustools.data.bader import (
    BaderAnalysis,
    BaderAtom,
    BaderError,
    read_bader_density,
)
from abacustools.data.grid import Charge


#: Partitioning methods of ``baderkit``. ``neargrid`` is the algorithm the
#: external program uses by default, so it keeps the two backends comparable.
METHODS = ("neargrid", "neargrid-weight", "ongrid", "weight")

#: Default partitioning method of this backend.
DEFAULT_METHOD = "neargrid"

__all__ = [
    "BaderkitResult",
    "DEFAULT_METHOD",
    "METHODS",
    "analyze_baderkit",
    "baderkit_available",
    "baderkit_version",
    "load_baderkit",
]


def baderkit_available() -> bool:
    """Return whether the optional ``baderkit`` package can be imported."""
    import importlib.util

    return importlib.util.find_spec("baderkit") is not None


def baderkit_version() -> Optional[str]:
    """Return the version of the installed ``baderkit`` package.

    Returns:
        The version, or ``None`` when the package is not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("baderkit")
    except PackageNotFoundError:
        return None


def load_baderkit() -> Any:
    """Import and return the ``baderkit`` package.

    Returns:
        The imported ``baderkit`` module.

    Raises:
        ImportError: When ``baderkit`` is missing, or when its numba kernels
            cannot be loaded in this environment.
    """
    try:
        import baderkit
    except ImportError as error:
        raise ImportError(
            "the `baderkit` backend needs the optional `baderkit` package; "
            "install it with `pip install abacustools[baderkit]`"
        ) from error
    except RuntimeError as error:
        # numba caches the kernels of the package next to the installed files,
        # which fails when site-packages is read only.
        raise ImportError(
            "`baderkit` could not be imported: its numba kernels cache next to "
            "the installed package, which fails when that directory is read "
            f"only. Set NUMBA_CACHE_DIR to a writable directory: {error}"
        ) from error
    return baderkit


def _vacuum_tolerance(vacuum: Optional[object]) -> object:
    """Translate ``--vacuum`` into baderkit's ``vacuum_tol``.

    The external program assigns no vacuum volume by default and takes
    ``-vac auto`` as a 1e-3 e/Angstrom**3 cutoff; ``baderkit`` takes ``False``
    for no vacuum, ``True`` for its own default cutoff and a float otherwise.

    Args:
        vacuum: ``None``, ``"off"``, ``"auto"`` or a density in e/Angstrom**3.

    Returns:
        The value to pass to ``baderkit``.
    """
    if vacuum is None:
        return False
    if isinstance(vacuum, (int, float)):
        return float(vacuum)
    text = str(vacuum)
    if text == "off":
        return False
    if text == "auto":
        return True
    return float(text)


def _grid_from_charge(charge: Charge, baderkit: Any) -> Any:
    """Build a ``baderkit`` grid from an assembled charge density.

    ``baderkit`` integrates ``sum(data) / npoints`` and treats a value below
    ``vacuum_tol * volume`` as vacuum, so the array it expects is the density in
    e/Angstrom**3 multiplied by the cell volume in Angstrom**3, the convention
    of a VASP CHGCAR. A :class:`Charge` holds e/Angstrom**3 on an Angstrom cell,
    which is exactly the pair this needs, and the array keeps the axis order of
    the cell instead of the reversed order the cube reader of ``baderkit``
    produces.
    """
    from pymatgen.core import Lattice, Structure

    cell = np.asarray(charge.cell, dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    structure = Structure(
        Lattice(cell),
        [int(number) for number in charge.atom_types],
        np.asarray(charge.atom_positions, dtype=float),
        coords_are_cartesian=True,
    )
    return baderkit.Grid(
        structure=structure,
        data={"total": np.asarray(charge.data, dtype=float) * volume},
        data_type="charge",
    )


@contextlib.contextmanager
def _captured_output():
    """Collect the ``baderkit`` progress output instead of printing it.

    ``baderkit`` logs through :mod:`logging` and its numba kernels ``print``
    from compiled code, which writes to the process stdout rather than to
    ``sys.stdout``. Both are collected here so that they end up in the report
    instead of in the middle of the table or of the ``--json`` payload.
    """
    records: List[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(self.format(record))

    root = logging.getLogger()
    handler = _Handler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    previous_handlers = root.handlers[:]
    previous_level = root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    saved_stdout = os.dup(1)
    captured = tempfile.TemporaryFile()
    try:
        sys.stdout.flush()
        os.dup2(captured.fileno(), 1)
        yield records
    finally:
        sys.stdout.flush()
        _flush_c_streams()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        root.handlers = previous_handlers
        root.setLevel(previous_level)
        captured.seek(0)
        text = captured.read().decode("utf-8", errors="replace")
        captured.close()
        if text.strip():
            records.append(text.rstrip())


def _flush_c_streams() -> None:
    """Flush the C streams the numba ``print`` writes through."""
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except Exception:  # pragma: no cover - only a best effort flush
        pass


@dataclass
class BaderkitResult:
    """Result of one ``baderkit`` partition.

    Attributes:
        charges: Electrons inside the Bader volume of every atom.
        volumes: Volume of the Bader volume of every atom in Angstrom**3.
        min_distances: Distance from every atom to the nearest point of its
            Bader surface in Angstrom.
        vacuum_charge: Electrons assigned to the vacuum volume.
        vacuum_volume: Volume assigned to the vacuum in Angstrom**3.
        number_of_electrons: Electrons in the atoms and the vacuum.
        log: Progress log of the run.
    """

    charges: List[float]
    volumes: List[float]
    min_distances: List[float]
    vacuum_charge: float
    vacuum_volume: float
    number_of_electrons: float
    log: str = ""


def _partition(
    charge: Charge,
    *,
    reference: Optional[Charge] = None,
    vacuum: Optional[object] = None,
    valence_counts: Optional[Dict[str, float]] = None,
    method: str = DEFAULT_METHOD,
) -> BaderkitResult:
    """Partition one density with ``baderkit``.

    Args:
        charge: Density that is integrated over the Bader volumes.
        reference: Density that defines the volumes, the density itself when
            omitted. The spin run passes the total density here so that the
            magnetization is integrated over the same volumes.
        vacuum: ``--vacuum`` value, translated by :func:`_vacuum_tolerance`.
        valence_counts: Number of valence electrons per element, used by
            ``baderkit`` for its oxidation states.
        method: Partitioning method of ``baderkit``, one of :data:`METHODS`.

    Returns:
        The per-atom result of the partition.
    """
    baderkit = load_baderkit()
    charge_grid = _grid_from_charge(charge, baderkit)
    reference_grid = charge_grid if reference is None else _grid_from_charge(reference, baderkit)
    with _captured_output() as records:
        analysis = baderkit.Bader(
            method=method,
            charge_grid=charge_grid,
            total_charge_grid=reference_grid,
            reference_grid=reference_grid,
            valence_counts=valence_counts,
            vacuum_tol=_vacuum_tolerance(vacuum),
        )
        result = BaderkitResult(
            charges=[float(value) for value in analysis.atom_charges],
            volumes=[float(value) for value in analysis.atom_volumes],
            min_distances=[float(value) for value in analysis.atom_min_surface_distances],
            vacuum_charge=float(analysis.vacuum_charge),
            vacuum_volume=float(analysis.vacuum_volume),
            number_of_electrons=float(analysis.total_electron_number),
        )
    result.log = "\n".join(records)
    return result


def _valence_counts(elements: Sequence[str], valences: Sequence[float]) -> Dict[str, float]:
    """Return the valence electron count of every element of a structure."""
    counts: Dict[str, float] = {}
    for element, valence in zip(elements, valences):
        counts.setdefault(str(element), float(valence))
    return counts


def _build_atoms(
    result: BaderkitResult,
    density: Charge,
    elements: Sequence[str],
    valences: Sequence[float],
) -> List[BaderAtom]:
    """Turn one partition into the atoms of a :class:`BaderAnalysis`."""
    if len(result.charges) != len(elements):
        raise BaderError(
            f"baderkit returned {len(result.charges)} basins for {len(elements)} atoms"
        )
    positions = np.asarray(density.atom_positions, dtype=float)
    atoms = []
    for index, (element, valence, charge, volume, distance) in enumerate(
        zip(elements, valences, result.charges, result.volumes, result.min_distances),
        start=1,
    ):
        atoms.append(
            BaderAtom(
                index=index,
                element=str(element),
                position=tuple(float(value) for value in positions[index - 1]),
                z_valence=float(valence),
                bader_charge=float(charge),
                min_distance=float(distance),
                atomic_volume=float(volume),
            )
        )
    return atoms


def analyze_baderkit(
    job: str | Path,
    *,
    cube: Optional[str] = None,
    reference: Optional[str] = None,
    grid_shape: Optional[tuple] = None,
    lat0: Optional[float] = None,
    vacuum: Optional[object] = None,
    method: str = DEFAULT_METHOD,
    workdir: Optional[str | Path] = None,
    keep: bool = False,
) -> BaderAnalysis:
    """Run a full Bader analysis with the ``baderkit`` library.

    The density is assembled exactly as for the external program, so both
    backends partition the same numbers. The magnetization of an ``nspin 2``
    job is integrated over the Bader volumes of the total density, which gives
    the per-atom spin moments.

    Args:
        job: ABACUS job directory.
        cube: Explicit charge-density cube file or directory, relative to
            ``job``.
        reference: Charge-density cube that defines the Bader volumes instead of
            the density itself, relative to ``job``.
        grid_shape: FFT grid of a restart file, read from the log when omitted.
        lat0: ``LATTICE_CONSTANT`` of the job in Bohr, taken from the STRU when
            omitted.
        vacuum: ``"off"``, ``"auto"`` or a density in e/Angstrom**3.
        method: Partitioning method of ``baderkit``, one of :data:`METHODS`.
            The default matches the partitioning of the external program.
        workdir: Keep the generated cubes and the log in this directory.
        keep: Keep the temporary working directory.

    Returns:
        The Bader analysis of the job.

    Raises:
        BaderError: If the density cannot be assembled or the partition fails.
        ImportError: If the optional ``baderkit`` package is missing.
    """
    job_path = Path(job).expanduser().absolute()
    density = read_bader_density(job_path, cube=cube, grid_shape=grid_shape, lat0=lat0)

    reference_charge = None
    reference_path = None
    if reference is not None:
        reference_path = Path(reference)
        if not reference_path.is_absolute():
            reference_path = job_path / reference_path
        reference_charge = Charge.from_cube(str(reference_path), format="abacus")

    if workdir is not None:
        work = Path(workdir).expanduser().absolute()
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="abacustools-baderkit-"))
        cleanup = not keep

    try:
        total_cube = work / "charge_total.cube"
        density.total.save_cube(str(total_cube), format="abacus")
        counts = _valence_counts(density.elements, density.valences)
        result = _partition(
            density.total,
            reference=reference_charge,
            vacuum=vacuum,
            valence_counts=counts,
            method=method,
        )
        atoms = _build_atoms(result, density.total, density.elements, density.valences)
        if density.magnetization is not None:
            spin = _partition(
                density.magnetization,
                reference=density.total,
                vacuum=vacuum,
                valence_counts=counts,
                method=method,
            )
            for atom, moment in zip(atoms, spin.charges):
                atom.spin_moment = float(moment)
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)

    return BaderAnalysis(
        job=job_path,
        nspin=density.nspin,
        atoms=atoms,
        vacuum_charge=result.vacuum_charge,
        vacuum_volume=result.vacuum_volume,
        number_of_electrons=result.number_of_electrons,
        charge_source=density.charge_source,
        workdir=work,
        bader_stdout=result.log,
        reference=reference_path,
        backend="baderkit",
    )
