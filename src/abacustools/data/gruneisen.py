"""Mode Grueneisen parameters of a crystal from three phonon calculations.

The mode Grueneisen parameter measures how a phonon frequency follows a change
of volume,

    gamma(q, nu) = -d ln omega(q, nu) / d ln V,

and it is what turns a harmonic phonon calculation into a statement about
thermal expansion: the thermodynamic Grueneisen parameter that relates the
volumetric thermal expansion to the heat capacity,

    gamma(T) = beta(T) B(T) V(T) / C_V(T),

is the heat capacity weighted average of the mode values.  The parameters are
evaluated from the dynamical matrices of three cells of the same crystal whose
volumes differ by a small isotropic strain, which is the route phonopy
implements in :class:`phonopy.api_gruneisen.PhonopyGruneisen`: it takes the
central difference

    gamma = -<e|D_+ - D_-|e> / (2 lambda_0) * V_0 / (V_+ - V_-)

of the eigenvalues lambda = omega^2.  This module holds the pure parts around
that call: scaling a cell to a strained volume, weighting the mode values with
the mode heat capacities, and summarising the result.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from abacustools.core.constant import BOLTZMANN_CONSTANT_EV_PER_K, PLANCK_CONSTANT, ELEMENTARY_CHARGE
from abacustools.io.stru import AbacusSTRU


#: Default isotropic volume strain of the two strained calculations.
DEFAULT_STRAIN = 0.01

#: Default q mesh the mode parameters are averaged over.
DEFAULT_MESH = (20, 20, 20)

#: Default temperature of the reported thermodynamic average, in K.
DEFAULT_TEMPERATURE = 298.15

#: Smallest and largest strain that still gives a meaningful central
#: difference: below the first the frequency change is buried in the numerical
#: noise of the force constants, above the second the Taylor expansion of the
#: frequency in the strain is no longer accurate.
MINIMUM_STRAIN = 1.0e-3
MAXIMUM_STRAIN = 0.05

#: Modes below this frequency, in THz, are left out of the averages.  The three
#: translations of a free cell are exactly zero at Gamma, where the logarithmic
#: derivative is undefined.
DEFAULT_FREQUENCY_CUTOFF = 1.0e-4

#: Energy of one THz, in eV, which is what phonopy's mode heat capacity takes.
THZ_TO_EV = PLANCK_CONSTANT * 1.0e12 / ELEMENTARY_CHARGE


def validate_strain(value: Any) -> float:
    """Validate an isotropic volume strain.

    Args:
        value: Strain to check.

    Returns:
        The strain as a float.

    Raises:
        ValueError: When the value is not finite, positive, or inside the range
            a central difference of the frequencies can resolve.
    """
    try:
        strain = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("strain must be a number") from error
    if not np.isfinite(strain) or strain <= 0.0:
        raise ValueError("strain must be a positive finite number")
    if strain < MINIMUM_STRAIN or strain > MAXIMUM_STRAIN:
        raise ValueError(
            f"strain must lie between {MINIMUM_STRAIN:g} and {MAXIMUM_STRAIN:g} "
            "for the central difference of the frequencies to be meaningful"
        )
    return strain


def cell_volume(cell: Iterable[Iterable[float]]) -> float:
    """Return the volume of a cell whose rows are its lattice vectors."""
    return float(abs(np.linalg.det(np.asarray(list(cell), dtype=float))))


def volume_strain(reference: Iterable[Iterable[float]], strained: Iterable[Iterable[float]]) -> float:
    """Return ``(V - V0) / V0`` of two cells."""
    volume_0 = cell_volume(reference)
    if volume_0 <= 0.0:
        raise ValueError("the reference cell has no volume")
    return (cell_volume(strained) - volume_0) / volume_0


def scaled_cell(structure: AbacusSTRU, strain: float) -> AbacusSTRU:
    """Return the cell with its volume changed by an isotropic strain.

    Every lattice vector is multiplied by ``(1 + strain) ** (1/3)`` and the
    fractional coordinates are kept, so the strain is the relative volume
    change and every atom follows the affine deformation.

    Args:
        structure: Reference cell.
        strain: Relative volume change, positive or negative.  The caller
            validates its magnitude.

    Returns:
        The strained cell.
    """
    result = deepcopy(structure)
    factor = (1.0 + float(strain)) ** (1.0 / 3.0)
    result.cell = (np.asarray(structure.cell, dtype=float) * factor).tolist()
    result.coords_direct = np.asarray(structure.coords_direct, dtype=float).tolist()
    return result


def mode_heat_capacity(
    frequencies: np.ndarray,
    temperature: float,
    *,
    cutoff: float = DEFAULT_FREQUENCY_CUTOFF,
) -> np.ndarray:
    """Return the harmonic heat capacity of each mode, in eV/K per cell.

    Args:
        frequencies: Mode frequencies in THz, any shape.
        temperature: Temperature in K, which has to be positive.
        cutoff: Frequencies below this, in THz, are treated as the translations
            of a free cell and get no weight.

    Returns:
        The heat capacity of every mode, in the shape of ``frequencies``.

    Raises:
        ValueError: When the temperature is not positive.
    """
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    energies = np.asarray(frequencies, dtype=float) * THZ_TO_EV
    x = energies / (BOLTZMANN_CONSTANT_EV_PER_K * float(temperature))
    with np.errstate(over="ignore", invalid="ignore"):
        # The Einstein heat capacity, written so that it stays finite for the
        # modes whose energy is far above or below k_B T.
        capacity = BOLTZMANN_CONSTANT_EV_PER_K * x**2 * np.exp(-x) / (1.0 - np.exp(-x)) ** 2
    capacity = np.where(x < 1.0e-8, BOLTZMANN_CONSTANT_EV_PER_K, capacity)
    return np.where(energies > cutoff * THZ_TO_EV, capacity, 0.0)


def mode_mask(
    frequencies: np.ndarray,
    mode_gruneisen: np.ndarray,
    *,
    cutoff: float = DEFAULT_FREQUENCY_CUTOFF,
) -> np.ndarray:
    """Return the modes that take part in an average.

    A mode is left out when its frequency is zero to within the cutoff, where
    the logarithmic derivative of the frequency is undefined, and when the
    parameter itself came out as a non-finite number, which happens for a mode
    whose frequency barely changes with the volume.
    """
    values = np.asarray(mode_gruneisen, dtype=float)
    energies = np.asarray(frequencies, dtype=float)
    if values.shape != energies.shape:
        raise ValueError("frequencies and mode parameters must have the same shape")
    return np.isfinite(values) & (energies > cutoff)


def gruneisen_temperature(
    temperatures: Sequence[float],
    frequencies: np.ndarray,
    weights: np.ndarray,
    mode_gruneisen: np.ndarray,
    *,
    cutoff: float = DEFAULT_FREQUENCY_CUTOFF,
) -> List[Optional[float]]:
    """Return the heat capacity weighted average of the mode parameters.

    Args:
        temperatures: Temperatures in K.
        frequencies: Mode frequencies in THz, shaped ``(nq, nbands)``.
        weights: q point weights, shaped ``(nq,)``.
        mode_gruneisen: Mode parameters, shaped ``(nq, nbands)``.
        cutoff: Frequencies below this, in THz, are left out.

    Returns:
        One value per temperature, in the order given; ``None`` when the
        weighted heat capacity vanishes, which is what happens at zero
        temperature where the average is undefined.

    Raises:
        ValueError: When the arrays do not describe the same sampling.
    """
    values = np.asarray(mode_gruneisen, dtype=float)
    energies = np.asarray(frequencies, dtype=float)
    q_weights = np.asarray(weights, dtype=float)
    if values.shape != energies.shape:
        raise ValueError("frequencies and mode parameters must have the same shape")
    if q_weights.shape != (values.shape[0],):
        raise ValueError("there must be one weight per q point")
    keep = mode_mask(energies, values, cutoff=cutoff)
    total_weights = q_weights[:, None] * keep
    result: List[Optional[float]] = []
    for temperature in temperatures:
        if not np.isfinite(temperature) or temperature <= 0.0:
            # At zero temperature every heat capacity vanishes, so the average
            # is undefined rather than zero.
            result.append(None)
            continue
        capacity = mode_heat_capacity(energies, temperature, cutoff=cutoff)
        weight_sum = float(np.sum(capacity * total_weights))
        if weight_sum <= 0.0:
            result.append(None)
        else:
            weighted = capacity * total_weights * np.where(keep, values, 0.0)
            result.append(float(np.sum(weighted) / weight_sum))
    return result


def summarize(
    frequencies: np.ndarray,
    weights: np.ndarray,
    mode_gruneisen: np.ndarray,
    *,
    cutoff: float = DEFAULT_FREQUENCY_CUTOFF,
) -> Dict[str, Any]:
    """Return a summary of the mode parameters on a mesh.

    Args:
        frequencies: Mode frequencies in THz, shaped ``(nq, nbands)``.
        weights: q point weights, shaped ``(nq,)``.
        mode_gruneisen: Mode parameters, shaped ``(nq, nbands)``.
        cutoff: Frequencies below this, in THz, are left out.

    Returns:
        Mapping with the number of modes, the number left out, and the minimum,
        maximum and q weighted mean of the parameters.
    """
    values = np.asarray(mode_gruneisen, dtype=float)
    energies = np.asarray(frequencies, dtype=float)
    keep = mode_mask(energies, values, cutoff=cutoff)
    kept = values[keep]
    weight = np.broadcast_to(np.asarray(weights, dtype=float)[:, None], values.shape)[keep]
    if kept.size == 0 or float(np.sum(weight)) <= 0.0:
        return {"modes": int(values.size), "modes_left_out": int(values.size - kept.size)}
    return {
        "modes": int(values.size),
        "modes_left_out": int(values.size - kept.size),
        "minimum": float(np.min(kept)),
        "maximum": float(np.max(kept)),
        "mean": float(np.sum(kept * weight) / np.sum(weight)),
    }
