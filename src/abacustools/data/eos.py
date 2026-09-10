"""Birch-Murnaghan equation-of-state fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


EV_PER_ANGSTROM3_TO_GPA = 160.21766208
_MINIMUM_POINTS = 4


@dataclass
class EosFit:
    """Result of a third-order Birch-Murnaghan fit to an E(V) curve."""

    volume: float
    energy: float
    bulk_modulus: float
    bulk_modulus_derivative: float
    residual: float
    volumes: list[float]
    energies: list[float]
    fit_volumes: list[float]
    fit_energies: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Return the fit summary as a JSON-serialisable dictionary."""
        return {
            "equilibrium_volume": self.volume,
            "equilibrium_energy": self.energy,
            "bulk_modulus": self.bulk_modulus,
            "bulk_modulus_derivative": self.bulk_modulus_derivative,
            "residual": self.residual,
            "fit": {"volumes": self.fit_volumes, "energies": self.fit_energies},
        }


def fit_birch_murnaghan(volumes: Sequence[float], energies: Sequence[float]) -> EosFit:
    """Fit a third-order Birch-Murnaghan EOS to ``(volume, energy)`` points.

    For the third-order Birch-Murnaghan form, the energy is an exact cubic
    polynomial in ``V**(-2/3)``, so a cubic least-squares fit recovers the
    equilibrium volume, bulk modulus (GPa) and its pressure derivative without
    nonlinear optimisation (the DeltaCodesDFT method).

    Args:
        volumes: Cell volumes in Angstrom^3.
        energies: Total energies in eV.

    Returns:
        An :class:`EosFit` with the fitted parameters and a smooth curve.
    """
    volume_array = np.asarray(volumes, dtype=float)
    energy_array = np.asarray(energies, dtype=float)
    if volume_array.shape != energy_array.shape:
        raise ValueError("volumes and energies must have the same length")

    mask = np.isfinite(volume_array) & np.isfinite(energy_array) & (volume_array > 0)
    volume_array = volume_array[mask]
    energy_array = energy_array[mask]
    if volume_array.size < _MINIMUM_POINTS:
        raise ValueError(
            f"at least {_MINIMUM_POINTS} finite (volume, energy) points are required"
        )

    order = np.argsort(volume_array)
    volume_array = volume_array[order]
    energy_array = energy_array[order]

    inverse = volume_array ** (-2.0 / 3.0)
    energy_of = np.poly1d(np.polyfit(inverse, energy_array, 3))
    first = np.polyder(energy_of, 1)
    second = np.polyder(energy_of, 2)
    third = np.polyder(energy_of, 3)

    roots = [
        root.real
        for root in np.roots(first)
        if np.isreal(root) and root.real > 0 and second(root.real) > 0
    ]
    if not roots:
        raise ValueError("could not locate a minimum on the fitted E(V) curve")
    inverse0 = min(roots, key=lambda root: float(energy_of(root)))

    volume0 = inverse0 ** (-3.0 / 2.0)
    energy0 = float(energy_of(inverse0))
    second_derivative = 4.0 / 9.0 * inverse0**5 * second(inverse0)
    third_derivative = (
        -20.0 / 9.0 * inverse0**6.5 * second(inverse0)
        - 8.0 / 27.0 * inverse0**7.5 * third(inverse0)
    )
    bulk_modulus = second_derivative / inverse0**1.5 * EV_PER_ANGSTROM3_TO_GPA
    bulk_modulus_derivative = (
        -1.0 - inverse0 ** (-1.5) * third_derivative / second_derivative
    )
    residual = float(np.sqrt(np.mean((energy_array - energy_of(inverse)) ** 2)))

    fit_volumes = np.linspace(
        min(float(volume_array.min()), volume0),
        max(float(volume_array.max()), volume0),
        100,
    )
    fit_energies = energy_of(fit_volumes ** (-2.0 / 3.0))
    return EosFit(
        volume=float(volume0),
        energy=energy0,
        bulk_modulus=float(bulk_modulus),
        bulk_modulus_derivative=float(bulk_modulus_derivative),
        residual=residual,
        volumes=volume_array.tolist(),
        energies=energy_array.tolist(),
        fit_volumes=fit_volumes.tolist(),
        fit_energies=fit_energies.tolist(),
    )
