"""Reduced density gradient, Hessian eigenvalues and DORI.

These three fields implement the volumetric analyses that Quantum ESPRESSO
offers as ``pp.x`` options ``plot_num`` 19, 20 and 123, so that a density read
from either code can be treated the same way:

* :func:`reduced_density_gradient` is the reduced density gradient (RDG) of
  the non-covalent interaction analysis, ``s = |grad rho| / (2 (3 pi^2)^(1/3)
  rho^(4/3))``.
* :func:`signed_density_hessian` is ``sign(lambda_2) rho``, with ``lambda_2``
  the middle eigenvalue of the density Hessian, which colours an NCI plot.
* :func:`dori` is the density overlap regions indicator of de Silva and
  Corminboeuf, J. Chem. Theory Comput. 7, 625 (2011).

The derivatives are evaluated in reciprocal space, exactly as QE does, which
is consistent with the periodic grid of a plane-wave calculation and avoids
the finite-difference error near the nuclei. All derivatives are taken in
atomic units, the convention of the formulae, while the returned fields use
the Angstrom-based convention of :class:`~abacustools.data.grid.Charge` so
that they can be written back as cubes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.grid import Charge


BOHR2A = BOHR_TO_ANG

#: Density above which ``pp.x`` replaces the gradient of the RDG by a constant.
ATOMIC_RHO_CUT = 0.05

#: Constant that ``pp.x`` uses for the gradient above :data:`ATOMIC_RHO_CUT`.
RDG_GRADIENT_CUT = 100.0

#: Regularizer that ``pp.x`` adds to ``|grad rho|^2`` in the DORI denominator.
DORI_REGULARIZER = 1.0e-5


def density_derivatives(density: Charge) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Cartesian gradient and Hessian of a density.

    The derivatives are evaluated in reciprocal space on the periodic grid and
    are given in atomic units, that is ``e/Bohr^4`` for the gradient and
    ``e/Bohr^5`` for the Hessian.

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.

    Returns:
        The gradient with shape ``(nx, ny, nz, 3)`` and the Hessian with shape
        ``(nx, ny, nz, 3, 3)``.
    """
    values = np.asarray(density.data, dtype=float) * BOHR2A**3
    cell = np.asarray(density.cell, dtype=float) / BOHR2A
    shape = values.shape
    indices = np.stack(
        np.meshgrid(
            *(np.fft.fftfreq(count) * count for count in shape),
            indexing="ij",
        ),
        axis=-1,
    )
    # G = 2 pi m inv(A)^T, so that G . a_i = 2 pi m_i for the row-major cell;
    # a missing transpose is invisible for an orthogonal cell and wrong for a
    # hexagonal or triclinic one.
    gvectors = 2.0 * np.pi * (indices @ np.linalg.inv(cell).T)
    transformed = np.fft.fftn(values)

    gradient = np.empty(shape + (3,), dtype=float)
    for axis in range(3):
        gradient[..., axis] = np.fft.ifftn(
            1j * gvectors[..., axis] * transformed
        ).real

    hessian = np.empty(shape + (3, 3), dtype=float)
    for first in range(3):
        for second in range(3):
            hessian[..., first, second] = np.fft.ifftn(
                -gvectors[..., first] * gvectors[..., second] * transformed
            ).real
    return gradient, hessian


def hessian_eigenvalues(density: Charge) -> np.ndarray:
    """Return the three Hessian eigenvalues of every grid point.

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.

    Returns:
        The eigenvalues in ``e/Bohr^5``, sorted ascending along the last axis,
        so that ``[..., 1]`` is the middle eigenvalue of the NCI analysis.
    """
    _, hessian = density_derivatives(density)
    return np.linalg.eigvalsh(hessian)


def _reduced_gradient(
    gradient: np.ndarray,
    rho: np.ndarray,
    rho_cut: Optional[float],
) -> np.ndarray:
    """Evaluate the RDG formula for a gradient and a density in atomic units."""
    norm = np.linalg.norm(gradient, axis=-1)
    if rho_cut is not None:
        norm = np.where(rho > rho_cut, RDG_GRADIENT_CUT, norm)
    factor = 0.5 / (3.0 * np.pi**2) ** (1.0 / 3.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return factor * norm / np.abs(rho) ** (4.0 / 3.0)


def reduced_density_gradient(
    density: Charge,
    *,
    rho_cut: Optional[float] = ATOMIC_RHO_CUT,
) -> np.ndarray:
    """Return the reduced density gradient of a density (``plot_num=19``).

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.
        rho_cut: Density in e/Bohr**3 above which ``pp.x`` stops using the real
            gradient and inserts :data:`RDG_GRADIENT_CUT` instead, which keeps
            the core region out of the plot. ``None`` uses the gradient
            everywhere.

    Returns:
        The dimensionless reduced density gradient, with the shape of the
        density.
    """
    gradient, _ = density_derivatives(density)
    rho = np.asarray(density.data, dtype=float) * BOHR2A**3
    return _reduced_gradient(gradient, rho, rho_cut)


def signed_density_hessian(density: Charge) -> np.ndarray:
    """Return ``sign(lambda_2) rho`` of a density (``plot_num=20``).

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.

    Returns:
        The density weighted by the sign of the middle Hessian eigenvalue, in
        e/Angstrom**3, which is the abscissa of a non-covalent interaction
        plot.
    """
    eigenvalues = hessian_eigenvalues(density)
    rho = np.asarray(density.data, dtype=float)
    return np.sign(eigenvalues[..., 1]) * rho


def dori(density: Charge) -> np.ndarray:
    """Return the density overlap regions indicator (``plot_num=123``).

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.

    Returns:
        The dimensionless indicator, which lies between zero and one.
    """
    gradient, hessian = density_derivatives(density)
    rho = np.asarray(density.data, dtype=float) * BOHR2A**3
    square = np.einsum("...i,...i->...", gradient, gradient)
    contraction = np.einsum("...j,...ij->...i", gradient, hessian)
    weighted = rho[..., np.newaxis] * contraction - gradient * square[..., np.newaxis]
    theta = 4.0 * np.einsum("...i,...i->...", weighted, weighted) / (
        square + DORI_REGULARIZER
    ) ** 3
    return theta / (1.0 + theta)


def nci_scatter_data(
    density: Charge,
    *,
    rho_max: float = ATOMIC_RHO_CUT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the points of a non-covalent interaction plot.

    The plot shows the reduced density gradient against ``sign(lambda_2) rho``
    for the low-density region, where non-covalent interactions live.

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.
        rho_max: Only grid points with a density below this value in e/Bohr**3
            are returned, which drops the atomic cores and the bonds.

    Returns:
        ``sign(lambda_2) rho`` in e/Bohr**3 and the reduced density gradient,
        both as one-dimensional arrays.
    """
    gradient, hessian = density_derivatives(density)
    eigenvalues = np.linalg.eigvalsh(hessian)
    rho = np.asarray(density.data, dtype=float) * BOHR2A**3
    signed = np.sign(eigenvalues[..., 1]) * rho
    rdg = _reduced_gradient(gradient, rho, rho_max)
    mask = rho <= rho_max
    return signed[mask], rdg[mask]


#: Analyses that :func:`abacustools.commands.postprocess.chg` can select.
QUANTITIES: Dict[str, str] = {
    "rdg": "reduced density gradient (pp.x plot_num=19)",
    "sl2rho": "sign(lambda_2) rho (pp.x plot_num=20)",
    "dori": "density overlap regions indicator (pp.x plot_num=123)",
}


def analyse(density: Charge, quantity: str) -> np.ndarray:
    """Return one of the NCI fields of a density.

    Args:
        density: Density in e/Angstrom**3 on an Angstrom cell.
        quantity: One of :data:`QUANTITIES`.

    Returns:
        The requested field, with the shape of the density.

    Raises:
        ValueError: If the quantity is unknown.
    """
    if quantity == "rdg":
        return reduced_density_gradient(density)
    if quantity == "sl2rho":
        return signed_density_hessian(density)
    if quantity == "dori":
        return dori(density)
    raise ValueError(f"unknown NCI quantity: {quantity!r}")
