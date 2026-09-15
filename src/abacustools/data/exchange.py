"""Magnetic exchange coupling constants from the four-state method.

The four-state method maps the total energy of a magnetic pair onto the
Heisenberg form ``E = J * (S1 . S2)`` by comparing three magnetization
configurations with a reference one. All three configurations are built so
that the pair angle grows by the same tilt angle, which isolates the bilinear
coupling: a linear fit of the energy difference against ``1 - cos(theta)`` has
the coupling constant as its slope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence, Tuple

import numpy as np


Case = Literal["atom1", "atom2", "both"]

#: The three magnetization configurations that share one tilt angle.
CASES: Tuple[Case, ...] = ("atom1", "atom2", "both")

#: Relative length below which a moment counts as parallel to the first one.
_DEGENERATE_TOLERANCE = 1.0e-8

#: Candidate axes used to define a rotation plane for (anti)parallel moments.
_UNIT_AXES = np.eye(3)

#: Number of points a linear four-state fit needs at least.
MINIMUM_FIT_POINTS = 2


class ExchangeError(RuntimeError):
    """Raised when a four-state magnetization configuration cannot be built."""


def moment_vector(moment: Any) -> np.ndarray:
    """Return one magnetic moment as a Cartesian vector.

    Args:
        moment: A scalar moment, which is placed along ``z``, or three numbers
            describing the moment in Bohr magneton.

    Returns:
        The magnetic moment as a length-three array.

    Raises:
        ExchangeError: If the moment is missing or is not a scalar or a
            three-component vector.
    """
    if moment is None:
        raise ExchangeError("the magnetic moment of a selected atom is not defined")
    if isinstance(moment, (int, float, np.integer, np.floating)):
        return np.array([0.0, 0.0, float(moment)], dtype=float)
    try:
        values = np.asarray(moment, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ExchangeError(f"cannot read the magnetic moment {moment!r}") from error
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ExchangeError(
            f"a magnetic moment must be a scalar or three finite numbers, got {moment!r}"
        )
    return values


def angle_between(first: Any, second: Any) -> float:
    """Return the angle between two magnetic moments in degrees.

    Args:
        first: Magnetic moment of the first atom.
        second: Magnetic moment of the second atom.

    Returns:
        The angle between the two moments in degrees, in the range 0 to 180.

    Raises:
        ExchangeError: If one of the moments is zero.
    """
    first_vector = moment_vector(first)
    second_vector = moment_vector(second)
    lengths = float(np.linalg.norm(first_vector)) * float(np.linalg.norm(second_vector))
    if not np.isfinite(lengths) or lengths <= 0.0:
        raise ExchangeError("the four-state method needs two non-zero magnetic moments")
    cosine = float(np.dot(first_vector, second_vector)) / lengths
    return float(np.degrees(np.arccos(float(np.clip(cosine, -1.0, 1.0)))))


def _tilt_plane(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal basis of the plane in which the moments tilt.

    The first vector is parallel to the first moment. For a pair that is
    neither parallel nor antiparallel the second vector points from the first
    moment towards the second one, so that
    ``second = |second| * (cos(theta) * first + sin(theta) * second)``
    reproduces the second moment. Parallel moments do not define such a plane,
    so the coordinate axis with the smallest overlap is used instead, which
    keeps the generated configurations deterministic.

    Args:
        first: Reference magnetic moment of the first atom.
        second: Reference magnetic moment of the second atom.

    Returns:
        A pair of orthonormal, perpendicular vectors.
    """
    axis = first / np.linalg.norm(first)
    projection = second - float(np.dot(second, axis)) * axis
    length = float(np.linalg.norm(projection))
    if length > _DEGENERATE_TOLERANCE * float(np.linalg.norm(second)):
        return axis, projection / length

    index = int(np.argmin(np.abs(axis)))
    direction = np.cross(axis, _UNIT_AXES[index])
    return axis, direction / np.linalg.norm(direction)


def tilted_moments(
    moment1: Any,
    moment2: Any,
    tilt: float,
    case: Case,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tilt two magnetic moments so that their pair angle grows by ``tilt``.

    ``atom1`` rotates only the first moment, ``atom2`` rotates only the second
    one and ``both`` rotates each moment by half the tilt, so all three
    configurations end up with the same pair angle. The magnitudes of both
    moments and the plane of the two reference moments are kept.

    Args:
        moment1: Reference magnetic moment of the first atom.
        moment2: Reference magnetic moment of the second atom.
        tilt: Rotation angle in degrees, which must not be negative.
        case: One of ``"atom1"``, ``"atom2"`` or ``"both"``.

    Returns:
        The two tilted moments.

    Raises:
        ExchangeError: If a moment is zero, the tilt is negative or the case is
            unknown.
    """
    first = moment_vector(moment1)
    second = moment_vector(moment2)
    magnitude1 = float(np.linalg.norm(first))
    magnitude2 = float(np.linalg.norm(second))
    if magnitude1 <= 0.0 or magnitude2 <= 0.0:
        raise ExchangeError("the four-state method needs two non-zero magnetic moments")
    if not np.isfinite(tilt) or tilt < 0.0:
        raise ExchangeError("the tilt angle must be a non-negative number")
    if case not in CASES:
        raise ExchangeError(f"unknown four-state case: {case!r}")

    axis, direction = _tilt_plane(first, second)
    reference_angle = np.radians(angle_between(first, second))
    delta = np.radians(float(tilt))

    if case == "atom1":
        rotated1 = magnitude1 * (np.cos(delta) * axis - np.sin(delta) * direction)
        return rotated1, second
    if case == "atom2":
        rotated2 = magnitude2 * (
            np.cos(reference_angle + delta) * axis
            + np.sin(reference_angle + delta) * direction
        )
        return first, rotated2

    half = delta / 2.0
    rotated1 = magnitude1 * (np.cos(half) * axis - np.sin(half) * direction)
    rotated2 = magnitude2 * (
        np.cos(reference_angle + half) * axis
        + np.sin(reference_angle + half) * direction
    )
    return rotated1, rotated2


def four_state_energy(
    original: float,
    atom1: float,
    atom2: float,
    both: float,
) -> float:
    """Combine four state energies into the coupling energy difference.

    Args:
        original: Total energy of the reference magnetization in eV.
        atom1: Total energy with only the first moment tilted, in eV.
        atom2: Total energy with only the second moment tilted, in eV.
        both: Total energy with both moments tilted, in eV.

    Returns:
        ``(E_both - E_atom1) - (E_atom2 - E_original)`` in eV.
    """
    return (float(both) - float(atom1)) - (float(atom2) - float(original))


@dataclass
class ExchangeFit:
    """Result of the linear four-state fit ``dE = J * (1 - cos(theta)) + c``.

    Attributes:
        coupling_mev: Fitted exchange coupling constant ``J`` in meV, positive
            for antiferromagnetic coupling.
        intercept_ev: Fitted offset in eV, which absorbs the reference angle.
        r_squared: Coefficient of determination of the linear fit.
        residual_ev: Root-mean-square deviation from the fitted line in eV.
        x: ``1 - cos(theta)`` of every fitted point.
        delta_energies: Energy difference in eV of every fitted point.
    """

    coupling_mev: float
    intercept_ev: float
    r_squared: float
    residual_ev: float
    x: list[float]
    delta_energies: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Return the fit as a JSON-serialisable dictionary."""
        return {
            "j_mev": self.coupling_mev,
            "slope_ev": self.coupling_mev / 1000.0,
            "intercept_ev": self.intercept_ev,
            "r_squared": self.r_squared,
            "rms_residual_ev": self.residual_ev,
            "points": len(self.x),
            "x": self.x,
            "delta_energy_ev": self.delta_energies,
        }


def fit_exchange_coupling(
    angles: Sequence[float],
    delta_energies: Sequence[float],
) -> ExchangeFit:
    """Fit ``dE = J * (1 - cos(theta))`` to four-state points.

    Args:
        angles: Pair angle of every four-state point in degrees.
        delta_energies: Four-state energy difference of every point in eV.

    Returns:
        The fitted :class:`ExchangeFit`, whose ``coupling_mev`` is the slope of
        the fit and therefore the coupling constant.

    Raises:
        ExchangeError: If the inputs differ in length or hold fewer than two
            finite points.
    """
    angle_values = np.asarray(angles, dtype=float)
    energy_values = np.asarray(delta_energies, dtype=float)
    if angle_values.shape != energy_values.shape:
        raise ExchangeError("the angles and energy differences must have the same length")

    mask = np.isfinite(angle_values) & np.isfinite(energy_values)
    angle_values = angle_values[mask]
    energy_values = energy_values[mask]
    if angle_values.size < MINIMUM_FIT_POINTS:
        raise ExchangeError(
            f"at least {MINIMUM_FIT_POINTS} finite four-state points are required for a fit"
        )

    x = 1.0 - np.cos(np.radians(angle_values))
    slope, intercept = np.polyfit(x, energy_values, 1)
    residuals = energy_values - (slope * x + intercept)
    square_sum = float(np.sum(residuals**2))
    total_sum = float(np.sum((energy_values - energy_values.mean()) ** 2))
    if total_sum > 0.0:
        r_squared = 1.0 - square_sum / total_sum
    else:
        r_squared = 1.0 if square_sum <= 0.0 else 0.0

    return ExchangeFit(
        coupling_mev=float(slope) * 1000.0,
        intercept_ev=float(intercept),
        r_squared=float(r_squared),
        residual_ev=float(np.sqrt(square_sum / energy_values.size)),
        x=x.tolist(),
        delta_energies=energy_values.tolist(),
    )
