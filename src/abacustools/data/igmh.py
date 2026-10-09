"""Hirshfeld-partitioned independent gradient model (IGMH).

IGMH replaces the frozen pseudoatomic densities of the promolecular IGM by
atomic densities obtained from a Hirshfeld partition of the calculated density,

``rho_A(r) = w_A(r) rho(r)``.

The resulting field is ``sum_A |grad rho_A| - |grad rho|`` and is plotted against
``sign(lambda_2) rho`` in the same way as an NCI or IGM plot.  The Hirshfeld-I
variant obtains ``w_A`` from the self-consistent Hirshfeld-I populations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.grid import Charge
from abacustools.data.hirshfeld import (
    AtomicReference,
    hirshfeld_i_weights,
    hirshfeld_weights,
)
from abacustools.data.nci import density_derivatives
from abacustools.data.weak import delta_g
from abacustools.io.stru import AbacusSTRU


BOHR2A = BOHR_TO_ANG


def atomic_gradient_sum(density: Charge, weights: np.ndarray) -> np.ndarray:
    """Return ``sum_A |grad rho_A|`` in e/Angstrom**4.

    Args:
        density: Calculated density on an Angstrom grid.
        weights: Hirshfeld weights with shape ``(natom, ngrid)`` in the flattened
            grid order.

    Returns:
        The sum of atomic gradient magnitudes, with the shape of the density.
    """
    rho = np.asarray(density.data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != rho.size:
        raise ValueError("weights must have shape (natom, ngrid) matching the density grid")
    result = np.zeros_like(rho)
    for atom in range(weights.shape[0]):
        partial = Charge(
            (weights[atom].reshape(rho.shape) * rho),
            density.cell,
            density.atom_positions,
            density.atom_types,
            density.atom_charges,
            density.origin,
        )
        gradient, _ = density_derivatives(partial)
        result += np.linalg.norm(gradient, axis=-1) / BOHR2A**4
    return result


def igmh(
    density: Charge,
    structure: AbacusSTRU,
    *,
    job: str | Path,
    pseudo_dir: Optional[str] = None,
    images: int = 1,
) -> np.ndarray:
    """Return the IGMH field of a density.

    Args:
        density: Electron density in e/Angstrom**3.
        structure: Structure that defines the atoms and elements.
        job: Job directory used to resolve pseudopotential paths.
        pseudo_dir: Pseudopotential directory from ``INPUT``, or ``None``.
        images: Lattice images of the proatoms to sum over.

    Returns:
        ``delta g`` in e/Angstrom**4, with the shape of the density.
    """
    weights = hirshfeld_weights(
        density,
        structure,
        job=job,
        pseudo_dir=pseudo_dir,
        images=images,
    )
    return delta_g(density, atomic_gradient_sum(density, weights))


def igmh_i(
    density: Charge,
    structure: AbacusSTRU,
    *,
    job: str | Path,
    references: Optional[Mapping[str, AtomicReference]] = None,
    pseudo_dir: Optional[str] = None,
    charges: Sequence[int] = (-2, -1, 0, 1, 2),
    images: int = 1,
    max_iter: int = 200,
    tol: float = 5e-4,
    mixing: float = 1.0,
) -> np.ndarray:
    """Return the Hirshfeld-I variant of the IGMH field.

    Args:
        density: Electron density in e/Angstrom**3.
        structure: Structure that defines the atoms and elements.
        job: Job directory used to resolve pseudopotential paths.
        references: Explicit Hirshfeld-I reference densities, when available.
        pseudo_dir: Pseudopotential directory from ``INPUT``, or ``None``.
        charges: Valence charges ``q = z_valence - N`` to tabulate per element.
        images: Lattice images of the reference atoms to sum over.
        max_iter: Maximum number of self-consistent iterations.
        tol: Convergence threshold on the largest population change.
        mixing: Linear mixing of the population update, in ``(0, 1]``.

    Returns:
        ``delta g`` in e/Angstrom**4, with the shape of the density.
    """
    partition = hirshfeld_i_weights(
        density,
        structure,
        job=job,
        references=references,
        pseudo_dir=pseudo_dir,
        charges=charges,
        images=images,
        max_iter=max_iter,
        tol=tol,
        mixing=mixing,
    )
    return delta_g(density, atomic_gradient_sum(density, partition.weights))


__all__ = ["atomic_gradient_sum", "igmh", "igmh_i"]
