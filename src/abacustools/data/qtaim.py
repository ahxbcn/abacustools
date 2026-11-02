"""Approximate QTAIM bond-critical-point search from a real-space density.

The search is designed for the densities ABACUS writes for norm-conserving
pseudopotentials. Those cubes contain the valence pseudo-density and no core
density, so the region around every nucleus is excluded from the search domain.
This is a deliberate approximation: the density is not altered and the result
must not be presented as an all-electron QTAIM analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.charge import (
    ChargeDensityError,
    pseudopotential_files,
    read_job_total_density,
)
from abacustools.data.grid import Charge
from abacustools.data.nci import density_derivatives
from abacustools.io.abacus import ReadInput
from abacustools.io.pseudo import UPF
from abacustools.io.stru import AbacusSTRU, periodic_lattice


BOHR2A = BOHR_TO_ANG

#: Fallback core radius in Angstrom when a UPF has no non-local projector.
DEFAULT_CORE_RADIUS_ANGSTROM = 0.8

#: Relative eigenvalue below which a Hessian direction counts as zero.
_HESSIAN_ZERO_TOLERANCE = 1.0e-7


@dataclass(frozen=True)
class CoreRadius:
    """Core-exclusion radius of one atom.

    Attributes:
        index: One-based atom index.
        element: Element symbol.
        radius: Exclusion radius in Angstrom.
    """

    index: int
    element: str
    radius: float


@dataclass(frozen=True)
class BondCriticalPoint:
    """One approximate bond critical point.

    Density and its derivatives are reported in atomic units: ``rho`` in
    ``e/Bohr^3``, ``gradient`` in ``e/Bohr^4``, and Hessian eigenvalues and the
    Laplacian in ``e/Bohr^5``. Positions and distances are in Angstrom.
    """

    index: int
    atom1: int
    atom2: int
    element1: str
    element2: str
    position: tuple[float, float, float]
    fractional: tuple[float, float, float]
    rho: float
    gradient: tuple[float, float, float]
    gradient_norm: float
    hessian: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    eigenvalues: tuple[float, float, float]
    laplacian: float
    ellipticity: Optional[float]
    distance1: float
    distance2: float
    core_clearance: float
    iterations: int


@dataclass(frozen=True)
class QtaimBcpResult:
    """Result of an approximate QTAIM BCP search."""

    job: str
    density_source: str
    grid: tuple[int, int, int]
    cell: tuple[tuple[float, float, float], ...]
    core_radii: tuple[CoreRadius, ...]
    core_radius_source: str
    pair_cutoff: float
    candidate_pairs: int
    seeds: int
    converged_searches: int
    rejected_core: int
    rejected_gradient: int
    rejected_signature: int
    rejected_assignment: int
    duplicate_points: int
    points: tuple[BondCriticalPoint, ...]
    caveat: str


def _position_fractional(cartesian: np.ndarray, cell: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Convert Cartesian positions in Angstrom to fractional grid coordinates."""
    return (np.asarray(cartesian, dtype=float) - origin) @ np.linalg.inv(cell)


def _position_cartesian(fractional: np.ndarray, cell: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Convert fractional grid coordinates to Cartesian positions in Angstrom."""
    return np.asarray(fractional, dtype=float) @ cell + origin


def _interpolate_periodic(field: np.ndarray, fractional: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate a periodic grid at fractional coordinates.

    Args:
        field: Grid values whose first three axes are the periodic grid and
            whose remaining axes, if any, are interpolated with the values.
        fractional: ``(n, 3)`` fractional coordinates; they are wrapped into
            ``[0, 1)``.

    Returns:
        An array whose first axis is ``n`` and whose trailing shape is that of
        the input field after its first three axes.
    """
    values = np.asarray(field, dtype=float)
    coordinates = np.asarray(fractional, dtype=float)
    if values.ndim < 3:
        raise ValueError("the interpolated field must have at least three dimensions")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"fractional coordinates must have shape (n, 3), got {coordinates.shape}")
    shape = np.asarray(values.shape[:3], dtype=int)
    coordinates = coordinates % 1.0
    grid = coordinates * shape
    lower = np.floor(grid).astype(int)
    fraction = grid - lower
    upper = (lower + 1) % shape
    result = np.zeros((coordinates.shape[0],) + values.shape[3:], dtype=float)
    for first in (0, 1):
        weight_x = fraction[:, 0] if first else 1.0 - fraction[:, 0]
        index_x = lower[:, 0] if first == 0 else upper[:, 0]
        for second in (0, 1):
            weight_y = fraction[:, 1] if second else 1.0 - fraction[:, 1]
            index_y = lower[:, 1] if second == 0 else upper[:, 1]
            for third in (0, 1):
                weight_z = fraction[:, 2] if third else 1.0 - fraction[:, 2]
                index_z = lower[:, 2] if third == 0 else upper[:, 2]
                weight = weight_x * weight_y * weight_z
                if values.ndim > 3:
                    weight = weight[(slice(None),) + (None,) * (values.ndim - 3)]
                result += weight * values[index_x, index_y, index_z]
    return result


def _core_radii_from_setting(
    job: Path,
    structure: AbacusSTRU,
    inputs: dict[str, Any],
    setting: str | float,
) -> tuple[tuple[CoreRadius, ...], str]:
    """Resolve one core-exclusion radius per atom."""
    elements = [str(atom.element or atom.label) for atom in structure.atoms]
    if isinstance(setting, (int, float)) or (
        isinstance(setting, str) and setting.strip().lower() != "auto"
    ):
        try:
            radius = float(setting)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"core radius must be 'auto' or a positive number, got {setting!r}"
            ) from error
        if radius <= 0.0:
            raise ValueError(f"core radius must be positive, got {radius:g}")
        return (
            tuple(CoreRadius(index + 1, element, radius) for index, element in enumerate(elements)),
            "user-supplied value",
        )

    paths = pseudopotential_files(structure, inputs.get("pseudo_dir"), job)
    cache: dict[Path, float] = {}
    radii = []
    fallback = False
    for index, (element, path) in enumerate(zip(elements, paths), start=1):
        if path not in cache:
            upf = UPF.read_from_file(path)
            radius = float(upf.cutoff_radius) * BOHR2A
            if radius <= 0.0:
                radius = DEFAULT_CORE_RADIUS_ANGSTROM
                fallback = True
            cache[path] = radius
        radii.append(CoreRadius(index, element, cache[path]))
    source = "UPF non-local projector cutoff radius"
    if fallback:
        source += (
            f"; {DEFAULT_CORE_RADIUS_ANGSTROM:g} Angstrom fallback where no projector was present"
        )
    return tuple(radii), source


def resolve_core_radii(
    job: str | Path,
    *,
    setting: str | float = 0.3,
    structure: Optional[AbacusSTRU] = None,
    inputs: Optional[dict[str, Any]] = None,
) -> tuple[tuple[CoreRadius, ...], str]:
    """Resolve core-exclusion radii for a job.

    Args:
        job: ABACUS job directory.
        setting: A positive exclusion radius in Angstrom for every atom, or
            ``"auto"`` to use the largest non-local projector cutoff of each
            UPF as a conservative heuristic.
        structure: Already parsed structure; read from ``job`` when omitted.
        inputs: Already parsed INPUT; read from ``job`` when omitted.

    Returns:
        One :class:`CoreRadius` per atom and a description of their source.
    """
    job_path = Path(job).resolve()
    inputs = inputs if inputs is not None else ReadInput(str(job_path / "INPUT"))
    if not inputs:
        raise ChargeDensityError(f"could not read INPUT in {job_path}")
    structure = structure or AbacusSTRU.read(str(job_path / str(inputs.get("stru_file", "STRU"))))
    if structure is None:
        raise ChargeDensityError(f"cannot read structure in {job_path}")
    return _core_radii_from_setting(job_path, structure, inputs, setting)


def _is_inside_core(
    fractional: np.ndarray,
    fractions: np.ndarray,
    radii: np.ndarray,
    lattice,
) -> tuple[bool, float]:
    """Return whether a point is inside a core and its minimum clearance."""
    clearances = np.asarray(
        [
            lattice.get_distance_and_image(fractional, atom_fraction)[0] - radius
            for atom_fraction, radius in zip(fractions, radii)
        ],
        dtype=float,
    )
    clearance = float(np.min(clearances))
    return clearance < 0.0, clearance


def _newton_step(gradient: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """Solve the stationary-point Newton step with a pseudo-inverse fallback."""
    try:
        return np.linalg.solve(hessian, -gradient)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        inverse = np.zeros_like(eigenvalues)
        nonzero = np.abs(eigenvalues) > _HESSIAN_ZERO_TOLERANCE * scale
        inverse[nonzero] = 1.0 / eigenvalues[nonzero]
        return -eigenvectors @ (inverse * (eigenvectors.T @ gradient))


def _refine_critical_point(
    seed: np.ndarray,
    *,
    cell: np.ndarray,
    origin: np.ndarray,
    density: Charge,
    gradient: np.ndarray,
    hessian: np.ndarray,
    gradient_tolerance: float,
    max_iterations: int,
    step_limit: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int]:
    """Refine a stationary point from a Cartesian seed in Angstrom."""
    point = np.asarray(seed, dtype=float)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        fractional = _position_fractional(point[None, :], cell, origin)
        rho = float(_interpolate_periodic(density.data[..., None], fractional)[0, 0])
        grad_atomic = _interpolate_periodic(gradient, fractional)[0]
        hessian_atomic = _interpolate_periodic(hessian, fractional)[0]
        if float(np.linalg.norm(grad_atomic)) <= gradient_tolerance:
            return point, rho * BOHR2A**3, grad_atomic, hessian_atomic, iterations
        step = _newton_step(grad_atomic / BOHR2A, hessian_atomic / BOHR2A**2)
        length = float(np.linalg.norm(step))
        if length > step_limit:
            step *= step_limit / length
        if length <= 1.0e-12:
            return point, rho * BOHR2A**3, grad_atomic, hessian_atomic, iterations
        point += step
        fractional = _position_fractional(point[None, :], cell, origin) % 1.0
        point = _position_cartesian(fractional[0], cell, origin)
    fractional = _position_fractional(point[None, :], cell, origin)
    rho = float(_interpolate_periodic(density.data[..., None], fractional)[0, 0])
    grad_atomic = _interpolate_periodic(gradient, fractional)[0]
    hessian_atomic = _interpolate_periodic(hessian, fractional)[0]
    return point, rho * BOHR2A**3, grad_atomic, hessian_atomic, iterations


def _critical_point_record(
    *,
    point: np.ndarray,
    rho: float,
    gradient: np.ndarray,
    hessian: np.ndarray,
    fractions: np.ndarray,
    elements: Sequence[str],
    radii: np.ndarray,
    lattice,
    cell: np.ndarray,
    origin: np.ndarray,
    iterations: int,
) -> Optional[BondCriticalPoint]:
    """Convert a refined stationary point into a BCP record."""
    fractional = _position_fractional(point[None, :], cell, origin)[0] % 1.0
    eigenvalues = np.linalg.eigvalsh(hessian)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    negative = eigenvalues < -_HESSIAN_ZERO_TOLERANCE * scale
    positive = eigenvalues > _HESSIAN_ZERO_TOLERANCE * scale
    if not (int(negative.sum()) == 2 and int(positive.sum()) == 1):
        return None
    distances = np.asarray(
        [
            lattice.get_distance_and_image(fractional, atom_fraction)[0]
            for atom_fraction in fractions
        ]
    )
    nearest = np.argsort(distances)[:2]
    if len(nearest) < 2:
        return None
    clearance = float(np.min(distances - radii))
    if clearance < 0.0:
        return None
    first, second = int(nearest[0]), int(nearest[1])
    if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian)) and np.isfinite(rho)):
        return None
    ellipticity = None
    if abs(float(eigenvalues[1])) > _HESSIAN_ZERO_TOLERANCE:
        ellipticity = float(eigenvalues[0] / eigenvalues[1] - 1.0)
    return BondCriticalPoint(
        index=0,
        atom1=first + 1,
        atom2=second + 1,
        element1=str(elements[first]),
        element2=str(elements[second]),
        position=tuple(float(value) for value in point),
        fractional=tuple(float(value) for value in fractional),
        rho=float(rho),
        gradient=tuple(float(value) for value in gradient),
        gradient_norm=float(np.linalg.norm(gradient)),
        hessian=tuple(tuple(float(value) for value in row) for row in hessian),
        eigenvalues=tuple(float(value) for value in eigenvalues),
        laplacian=float(np.sum(eigenvalues)),
        ellipticity=ellipticity,
        distance1=float(distances[first]),
        distance2=float(distances[second]),
        core_clearance=clearance,
        iterations=iterations,
    )


def search_bcp(
    density: Charge,
    structure: AbacusSTRU,
    *,
    core_radii: Sequence[CoreRadius],
    pair_cutoff: float = 4.0,
    line_samples: int = 31,
    seeds_per_pair: int = 3,
    gradient_tolerance: float = 1.0e-3,
    max_iterations: int = 60,
    step_limit: float = 0.25,
    duplicate_distance: float = 0.08,
) -> QtaimBcpResult:
    """Search approximate BCPs outside spherical core-exclusion regions.

    The search samples atom pairs on their periodic minimum-image segment,
    takes density minima along the line as seeds, and refines them with a
    three-dimensional Newton iteration on the density gradient. A point is
    accepted only when it is outside every core sphere and its Hessian has the
    ``(3,-1)`` signature.
    """
    if pair_cutoff <= 0.0:
        raise ValueError("pair cutoff must be positive")
    if line_samples < 3:
        raise ValueError("line_samples must be at least 3")
    if seeds_per_pair < 1:
        raise ValueError("seeds_per_pair must be positive")
    if len(core_radii) != structure.natoms:
        raise ValueError(f"expected {structure.natoms} core radii, got {len(core_radii)}")
    if not np.allclose(density.cell, structure.cell):
        raise ValueError("the density cell and structure cell do not match")

    cell = np.asarray(density.cell, dtype=float)
    origin = np.asarray(density.origin, dtype=float)
    fractions = np.asarray(structure.coords_direct, dtype=float) % 1.0
    elements = [str(atom.element or atom.label) for atom in structure.atoms]
    radii = np.asarray([entry.radius for entry in core_radii], dtype=float)
    lattice = periodic_lattice(structure)
    gradient, hessian = density_derivatives(density)
    points: list[BondCriticalPoint] = []
    candidate_pairs = 0
    seed_count = 0
    converged = 0
    rejected_core = 0
    rejected_gradient = 0
    rejected_signature = 0
    rejected_assignment = 0
    duplicates = 0

    for first in range(structure.natoms):
        for second in range(first + 1, structure.natoms):
            distance, image = lattice.get_distance_and_image(fractions[first], fractions[second])
            distance = float(distance)
            if distance <= 1.0e-8 or distance > pair_cutoff:
                continue
            candidate_pairs += 1
            lower = radii[first] / distance
            upper = 1.0 - radii[second] / distance
            if lower >= upper:
                continue
            start_fractional = fractions[first]
            end_fractional = fractions[second] + np.asarray(image)
            start = _position_cartesian(start_fractional, cell, origin)
            end = _position_cartesian(end_fractional, cell, origin)
            positions = start + np.linspace(lower, upper, line_samples)[:, None] * (end - start)
            line_fractional = _position_fractional(positions, cell, origin)
            line_rho = _interpolate_periodic(density.data[..., None], line_fractional)[:, 0]
            minima = [
                index
                for index in range(1, line_samples - 1)
                if line_rho[index] <= line_rho[index - 1] and line_rho[index] <= line_rho[index + 1]
            ]
            if not minima:
                minima = [int(np.argmin(line_rho))]
            minima = sorted(minima, key=lambda index: line_rho[index])[:seeds_per_pair]
            for candidate in minima:
                seed_count += 1
                point, rho, grad, hess, iterations = _refine_critical_point(
                    positions[candidate],
                    cell=cell,
                    origin=origin,
                    density=density,
                    gradient=gradient,
                    hessian=hessian,
                    gradient_tolerance=gradient_tolerance,
                    max_iterations=max_iterations,
                    step_limit=step_limit,
                )
                inside, _clearance = _is_inside_core(
                    _position_fractional(point[None, :], cell, origin)[0] % 1.0,
                    fractions,
                    radii,
                    lattice,
                )
                if inside:
                    rejected_core += 1
                    continue
                if float(np.linalg.norm(grad)) > gradient_tolerance:
                    rejected_gradient += 1
                    continue
                converged += 1
                record = _critical_point_record(
                    point=point,
                    rho=rho,
                    gradient=grad,
                    hessian=hess,
                    fractions=fractions,
                    elements=elements,
                    radii=radii,
                    lattice=lattice,
                    cell=cell,
                    origin=origin,
                    iterations=iterations,
                )
                if record is None:
                    eigenvalues = np.linalg.eigvalsh(hess)
                    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
                    negative = int(np.sum(eigenvalues < -_HESSIAN_ZERO_TOLERANCE * scale))
                    positive = int(np.sum(eigenvalues > _HESSIAN_ZERO_TOLERANCE * scale))
                    if not (negative == 2 and positive == 1):
                        rejected_signature += 1
                    else:
                        rejected_assignment += 1
                    continue
                duplicate = any(
                    lattice.get_distance_and_image(
                        np.asarray(record.fractional), np.asarray(existing.fractional)
                    )[0]
                    <= duplicate_distance
                    for existing in points
                )
                if duplicate:
                    duplicates += 1
                    continue
                points.append(
                    BondCriticalPoint(
                        index=len(points) + 1,
                        atom1=record.atom1,
                        atom2=record.atom2,
                        element1=record.element1,
                        element2=record.element2,
                        position=record.position,
                        fractional=record.fractional,
                        rho=record.rho,
                        gradient=record.gradient,
                        gradient_norm=record.gradient_norm,
                        hessian=record.hessian,
                        eigenvalues=record.eigenvalues,
                        laplacian=record.laplacian,
                        ellipticity=record.ellipticity,
                        distance1=record.distance1,
                        distance2=record.distance2,
                        core_clearance=record.core_clearance,
                        iterations=record.iterations,
                    )
                )

    return QtaimBcpResult(
        job="",
        density_source="",
        grid=tuple(int(value) for value in density.data.shape),
        cell=tuple(tuple(float(value) for value in row) for row in cell),
        core_radii=tuple(core_radii),
        core_radius_source="",
        pair_cutoff=float(pair_cutoff),
        candidate_pairs=candidate_pairs,
        seeds=seed_count,
        converged_searches=converged,
        rejected_core=rejected_core,
        rejected_gradient=rejected_gradient,
        rejected_signature=rejected_signature,
        rejected_assignment=rejected_assignment,
        duplicate_points=duplicates,
        points=tuple(points),
        caveat=(
            "The density may be valence-only. Core regions are excluded from "
            "the search, not reconstructed; the result is approximate QTAIM."
        ),
    )


def analyze_qtaim_bcp(
    job: str | Path,
    *,
    density_cube: Optional[str | Path] = None,
    core_radius: str | float = 0.3,
    pair_cutoff: float = 4.0,
    line_samples: int = 31,
    seeds_per_pair: int = 3,
    gradient_tolerance: float = 1.0e-3,
    max_iterations: int = 60,
    step_limit: float = 0.25,
    duplicate_distance: float = 0.08,
    grid_shape: Optional[tuple[int, int, int]] = None,
    lat0: Optional[float] = None,
) -> QtaimBcpResult:
    """Search approximate QTAIM BCPs for one ABACUS job."""
    job_path = Path(job).resolve()
    inputs = ReadInput(str(job_path / "INPUT"))
    if not inputs:
        raise ChargeDensityError(f"could not read INPUT in {job_path}")
    structure = AbacusSTRU.read(str(job_path / str(inputs.get("stru_file", "STRU"))))
    if structure is None:
        raise ChargeDensityError(f"cannot read structure in {job_path}")

    if density_cube is not None:
        density_path = Path(density_cube)
        if not density_path.is_absolute():
            density_path = job_path / density_path
        if not density_path.is_file():
            raise FileNotFoundError(f"density cube does not exist: {density_path}")
        density = Charge.from_cube(str(density_path), format="abacus")
        source = str(density_path)
    else:
        density = read_job_total_density(
            job_path,
            description="QTAIM BCP search",
            grid_shape=grid_shape,
            lat0=lat0,
        )
        source = "job charge-density files"

    radii, source_radius = _core_radii_from_setting(job_path, structure, inputs, core_radius)
    result = search_bcp(
        density,
        structure,
        core_radii=radii,
        pair_cutoff=pair_cutoff,
        line_samples=line_samples,
        seeds_per_pair=seeds_per_pair,
        gradient_tolerance=gradient_tolerance,
        max_iterations=max_iterations,
        step_limit=step_limit,
        duplicate_distance=duplicate_distance,
    )
    return QtaimBcpResult(
        job=str(job_path),
        density_source=source,
        grid=result.grid,
        cell=result.cell,
        core_radii=result.core_radii,
        core_radius_source=source_radius,
        pair_cutoff=result.pair_cutoff,
        candidate_pairs=result.candidate_pairs,
        seeds=result.seeds,
        converged_searches=result.converged_searches,
        rejected_core=result.rejected_core,
        rejected_gradient=result.rejected_gradient,
        rejected_signature=result.rejected_signature,
        rejected_assignment=result.rejected_assignment,
        duplicate_points=result.duplicate_points,
        points=result.points,
        caveat=result.caveat,
    )


__all__ = [
    "BondCriticalPoint",
    "CoreRadius",
    "QtaimBcpResult",
    "analyze_qtaim_bcp",
    "resolve_core_radii",
    "search_bcp",
]
