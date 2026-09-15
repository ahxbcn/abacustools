"""Interaction region indicator and promolecular weak-interaction analyses.

The definitions follow Multiwfn:

* the interaction region indicator of Lu and Chen is ``|grad rho| / rho**1.1``
  (``function.f90`` of Multiwfn), which shows covalent and non-covalent
  interactions in one function;
* the promolecular density is the superposition of the free pseudoatomic
  densities of the elements, and the independent gradient model compares it
  with the calculated density, ``delta g = sum_A |grad rho_A| - |grad rho|``.

The atomic densities are taken from the ``PP_RHOATOM`` table of the same UPF
files that the calculation used, so the reference density belongs to
pseudopotentials of the job, and the fields are evaluated on the grid of the
calculated density so that they can be written back as cubes. Multiwfn fills
the IRI with a large constant where the density is negligible; this module
returns ``fill`` there instead, which keeps the field usable in data files and
in the statistics of a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.charge import Charge, pseudopotential_files
from abacustools.data.nci import density_derivatives
from abacustools.io.pseudo import UPF


BOHR2A = BOHR_TO_ANG

#: Exponent of the density in the IRI, the Multiwfn default.
IRI_EXPONENT = 1.1

#: Density below which Multiwfn replaces the IRI, in e/Bohr**3.
IRI_RHO_CUT = 5.0e-5

#: Radial density in e/Bohr below which the pseudoatomic density is dropped.
_RADIAL_FLOOR = 1.0e-6


@dataclass
class AtomicDensity:
    """Spherically symmetric pseudoatomic valence density of one element.

    Attributes:
        element: Element symbol the UPF declares, when it declares one.
        radius: Radial mesh in Bohr.
        radial_density: ``4 pi r^2 rho(r)`` in e/Bohr, the UPF convention.
        cutoff: Radius in Bohr beyond which the density is negligible.
    """

    element: Optional[str]
    radius: np.ndarray
    radial_density: np.ndarray
    cutoff: float
    _values: Optional[np.ndarray] = field(default=None, repr=False)
    _gradients: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def electron_count(self) -> float:
        """Number of electrons the radial density integrates to."""
        radius = np.asarray(self.radius, dtype=float)
        density = np.asarray(self.radial_density, dtype=float)
        if radius.size < 2:
            return float(density.sum())
        return float(np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(radius)))

    def _value_table(self) -> np.ndarray:
        """Return ``rho(r)`` of the radial mesh, cached."""
        if self._values is None:
            radius = np.asarray(self.radius, dtype=float)
            radial = np.asarray(self.radial_density, dtype=float)
            table = np.zeros_like(radial)
            positive = radius > 0.0
            table[positive] = radial[positive] / (4.0 * np.pi * radius[positive] ** 2)
            if table.size and not positive[0]:
                # rho(r) is finite at the origin; its neighbours are the best
                # available estimate of the limit.
                table[0] = table[1] if table.size > 1 else 0.0
            self._values = table
        return self._values

    def values(self, distances: np.ndarray) -> np.ndarray:
        """Return ``rho(r)`` in e/Bohr**3 at the given distances in Bohr."""
        table = self._value_table()
        return np.interp(distances, self.radius, table, left=table[0], right=0.0)

    def gradient_magnitudes(self, distances: np.ndarray) -> np.ndarray:
        """Return ``|d rho / dr|`` in e/Bohr**4 at the given distances in Bohr."""
        if self._gradients is None:
            radius = np.asarray(self.radius, dtype=float)
            values = self._value_table()
            if radius.size < 2:
                self._gradients = np.zeros_like(radius)
            else:
                self._gradients = np.abs(np.gradient(values, radius))
        return np.interp(distances, self.radius, self._gradients, left=0.0, right=0.0)


def read_atomic_density(path: Path) -> AtomicDensity:
    """Read the pseudoatomic density of a UPF file.

    Args:
        path: UPF file of one element.

    Returns:
        The radial density on the mesh of the file.
    """
    upf = UPF.read_from_file(path)
    radius = np.asarray(upf.r, dtype=float)
    radial = np.asarray(upf.rhoatom, dtype=float)
    if radius.size != radial.size:
        raise ValueError(
            f"the radial mesh and the atomic density of {path} have different sizes"
        )
    significant = np.nonzero(np.abs(radial) > _RADIAL_FLOOR)[0]
    if significant.size:
        cutoff = float(radius[significant[-1]])
    else:
        cutoff = float(radius[-1]) if radius.size else 0.0
    return AtomicDensity(
        element=getattr(upf, "element", None),
        radius=radius,
        radial_density=radial,
        cutoff=cutoff,
    )


def promolecular_fields(
    density: Charge,
    structure,
    *,
    pseudo_dir: Optional[str] = None,
    job: Optional[Path] = None,
    cache: Optional[Dict[Path, AtomicDensity]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the promolecular density and the atomic gradient magnitudes.

    The fields are placed on the grid and the cell of the calculated density,
    with a minimum image convention, so a cutoff radius smaller than half the
    cell is required for an exact superposition of the periodic images.

    Args:
        density: Calculated density that defines the grid and the cell.
        structure: Structure whose atoms and pseudopotentials are used.
        pseudo_dir: ``pseudo_dir`` of INPUT, or ``None`` for the job directory.
        job: ABACUS job directory, used to resolve the pseudopotential names.
        cache: Optional cache of the atomic densities, keyed by UPF path.

    Returns:
        The promolecular density in e/Angstrom**3 and the sum of the atomic
        gradient magnitudes in e/Angstrom**4, both with the shape of the
        calculated density.
    """
    job = Path(job) if job is not None else Path(".")
    tables: Dict[Path, AtomicDensity] = {} if cache is None else cache
    paths = pseudopotential_files(structure, pseudo_dir, job)
    cell = np.asarray(density.cell, dtype=float)
    inverse = np.linalg.inv(cell)
    shape = tuple(int(size) for size in density.data.shape)
    rho = np.zeros(shape, dtype=float)
    gradient = np.zeros(shape, dtype=float)

    for atom, path in zip(structure.atoms, paths):
        table = tables.get(path)
        if table is None:
            table = read_atomic_density(path)
            tables[path] = table
        if table.cutoff <= 0.0:
            continue
        center = np.asarray(atom.coord, dtype=float)
        fractional = center @ inverse
        radius = table.cutoff * BOHR2A
        widths = radius * np.abs(inverse).sum(axis=0) * np.asarray(shape, dtype=float)
        windows = []
        for axis, size in enumerate(shape):
            half = int(np.ceil(widths[axis]))
            if 2 * half + 1 >= size:
                windows.append(np.arange(size))
            else:
                start = int(round(fractional[axis] * size))
                windows.append(np.mod(np.arange(start - half, start + half + 1), size))
        fraction = np.stack(
            np.meshgrid(
                *(window / size for window, size in zip(windows, shape)),
                indexing="ij",
            ),
            axis=-1,
        )
        displacement = fraction - fractional
        displacement -= np.round(displacement)
        distances = np.linalg.norm(displacement @ cell, axis=-1) / BOHR2A
        index = np.ix_(*windows)
        rho[index] += table.values(distances) / BOHR2A**3
        gradient[index] += table.gradient_magnitudes(distances) / BOHR2A**4
    return rho, gradient


def iri(
    density: Charge,
    *,
    exponent: float = IRI_EXPONENT,
    rho_cut: float = IRI_RHO_CUT,
    fill: float = 0.0,
) -> np.ndarray:
    """Return the interaction region indicator of a density.

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.
        exponent: Exponent of the density, 1.1 by default as in Multiwfn.
        rho_cut: Density in e/Bohr**3 below which the value is replaced.
        fill: Value of the replaced points.

    Returns:
        The dimensionless indicator, with the shape of the density.
    """
    gradient, _ = density_derivatives(density)
    rho = np.asarray(density.data, dtype=float) * BOHR2A**3
    norm = np.linalg.norm(gradient, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = norm / rho**exponent
    replaced = (rho <= rho_cut) | (norm == 0.0)
    return np.where(replaced, fill, values)


def delta_g(density: Charge, atomic_gradient: np.ndarray) -> np.ndarray:
    """Return the independent gradient model function ``delta g``.

    Args:
        density: Calculated density in e/Angstrom**3 on an Angstrom cell.
        atomic_gradient: Sum of the atomic gradient magnitudes in
            e/Angstrom**4, as returned by :func:`promolecular_fields`.

    Returns:
        ``sum_A |grad rho_A| - |grad rho|`` in e/Angstrom**4, clipped at zero,
        with the shape of the density. The gradient of the calculated density
        is scaled from the atomic units of :func:`density_derivatives` so that
        both terms of the difference share the unit of the promolecular field.
    """
    gradient, _ = density_derivatives(density)
    norm = np.linalg.norm(gradient, axis=-1) / BOHR2A**4
    return np.clip(np.asarray(atomic_gradient, dtype=float) - norm, 0.0, None)
