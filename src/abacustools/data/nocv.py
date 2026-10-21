"""Natural orbitals for chemical valence from a pEDA density difference.

The pEDA state builder already transforms the density matrices into an
orthonormal fragment basis.  In that basis the deformation density is a
traceless Hermitian matrix, so its eigenvectors are the NOCVs and its paired
positive and negative eigenvalues measure how strongly each channel deforms
the fragment densities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abacustools.core.constant import RY_TO_EV


@dataclass(frozen=True)
class NOCVChannel:
    """One paired NOCV deformation channel.

    Attributes:
        index: One-based channel index, ordered by decreasing magnitude.
        eigenvalue: Positive eigenvalue ``v_i`` of the channel.
        positive_vector: Eigenvector with eigenvalue ``+v_i``.
        negative_vector: Eigenvector with eigenvalue ``-v_i``.
        positive_index: Column index of ``positive_vector`` in the full matrix.
        negative_index: Column index of ``negative_vector`` in the full matrix.
        mismatch: Residual ``abs(v_positive + v_negative)`` in the pairing.
    """

    index: int
    eigenvalue: float
    positive_vector: np.ndarray
    negative_vector: np.ndarray
    positive_index: int
    negative_index: int
    mismatch: float


@dataclass(frozen=True)
class NOCVDecomposition:
    """Eigenvectors and paired channels of one deformation density.

    Attributes:
        eigenvalues: All eigenvalues of ``delta_p`` in ascending order.
        eigenvectors: Orthonormal eigenvectors, stored as matrix columns.
        channels: Paired positive and negative NOCV channels.
        unpaired_indices: Eigenvector columns not assigned to a pair.
        trace: Trace of the validated deformation matrix.
        zero_tolerance: Threshold used to classify zero eigenvalues.
        pair_tolerance: Threshold used when matching ``+v`` with ``-v``.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    channels: tuple[NOCVChannel, ...]
    unpaired_indices: tuple[int, ...]
    trace: float
    zero_tolerance: float
    pair_tolerance: float


@dataclass(frozen=True)
class NOCVChannelEnergy:
    """Orbital-energy contribution of one paired NOCV channel.

    Attributes:
        index: One-based channel index from the NOCV decomposition.
        eigenvalue: Positive deformation eigenvalue ``v_i``.
        positive_expectation_ry: ``<psi_i|F_TS|psi_i>`` in Ry.
        negative_expectation_ry: ``<psi_-i|F_TS|psi_-i>`` in Ry.
        energy_ev: Channel energy contribution in eV.
    """

    index: int
    eigenvalue: float
    positive_expectation_ry: float
    negative_expectation_ry: float
    energy_ev: float


@dataclass(frozen=True)
class NOCVEnergyDecomposition:
    """NOCV channel energies and their total orbital-energy contribution.

    Attributes:
        channels: Paired NOCV channel energies.
        paired_total_ev: Sum of the paired channel energies in eV.
        unpaired_total_ev: Contribution of unpaired eigenvectors in eV.
        total_ev: Full ``Tr(delta_p @ F_TS)`` contribution in eV.
        unit: Energy unit of the returned energy fields.
    """

    channels: tuple[NOCVChannelEnergy, ...]
    paired_total_ev: float
    unpaired_total_ev: float
    total_ev: float
    unit: str = "eV"


def _validate_tolerance(value: float, name: str) -> float:
    """Return a finite non-negative tolerance."""

    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _match_nocv_channels(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    zero_tolerance: float,
    pair_tolerance: float,
) -> tuple[tuple[NOCVChannel, ...], tuple[int, ...]]:
    """Pair positive and negative NOCV eigenvalues by magnitude."""

    negative = [index for index, value in enumerate(eigenvalues) if value < -zero_tolerance]
    positive = [index for index, value in enumerate(eigenvalues) if value > zero_tolerance]
    negative.sort(key=lambda index: abs(float(eigenvalues[index])), reverse=True)
    positive.sort(key=lambda index: float(eigenvalues[index]), reverse=True)

    unused_positive = set(positive)
    channels: list[NOCVChannel] = []
    for negative_index in negative:
        negative_value = float(eigenvalues[negative_index])
        if not unused_positive:
            break
        positive_index = min(
            unused_positive,
            key=lambda index: (
                abs(abs(negative_value) - float(eigenvalues[index])),
                -float(eigenvalues[index]),
            ),
        )
        positive_value = float(eigenvalues[positive_index])
        mismatch = abs(positive_value + negative_value)
        if mismatch > pair_tolerance:
            continue
        unused_positive.remove(positive_index)
        channels.append(
            NOCVChannel(
                index=len(channels) + 1,
                eigenvalue=0.5 * (positive_value - negative_value),
                positive_vector=eigenvectors[:, positive_index].copy(),
                negative_vector=eigenvectors[:, negative_index].copy(),
                positive_index=positive_index,
                negative_index=negative_index,
                mismatch=mismatch,
            )
        )

    paired = {
        index for channel in channels for index in (channel.positive_index, channel.negative_index)
    }
    unpaired = tuple(index for index in range(eigenvalues.size) if index not in paired)
    return tuple(channels), unpaired


def diagonalize_nocv(
    delta_p: np.ndarray,
    *,
    zero_tolerance: float = 1e-8,
    pair_tolerance: float = 1e-6,
    trace_tolerance: float = 1e-6,
    hermitian_tolerance: float = 1e-8,
) -> NOCVDecomposition:
    """Diagonalize a pEDA deformation density in an orthonormal basis.

    The deformation matrix must be Hermitian and traceless for a closed-shell
    pEDA calculation.  ``numpy.linalg.eigh`` returns its orthonormal
    eigenvectors; opposite-sign eigenvalues are paired into NOCV channels by
    magnitude.  Small unmatched eigenvalues remain available in
    ``unpaired_indices`` rather than being silently discarded.

    Args:
        delta_p: Deformation density ``P_AB - P0`` in an orthonormal basis.
        zero_tolerance: Eigenvalues within this magnitude are treated as zero.
        pair_tolerance: Maximum mismatch allowed when pairing ``+v`` and ``-v``.
        trace_tolerance: Maximum accepted absolute trace of ``delta_p``.
        hermitian_tolerance: Absolute tolerance for the Hermiticity check.

    Returns:
        The full eigensystem and the paired NOCV channels.

    Raises:
        ValueError: If the matrix is not a finite square Hermitian matrix,
            has a non-zero trace, or the tolerances are invalid.
    """

    zero = _validate_tolerance(zero_tolerance, "zero_tolerance")
    pair = _validate_tolerance(pair_tolerance, "pair_tolerance")
    trace_limit = _validate_tolerance(trace_tolerance, "trace_tolerance")
    hermitian_limit = _validate_tolerance(hermitian_tolerance, "hermitian_tolerance")

    matrix = np.asarray(delta_p, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("delta_p must be a square two-dimensional matrix")
    if matrix.shape[0] == 0:
        raise ValueError("delta_p must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("delta_p contains non-finite values")
    if not np.allclose(
        matrix,
        matrix.conj().T,
        rtol=0.0,
        atol=hermitian_limit,
    ):
        raise ValueError("delta_p must be Hermitian")

    trace = complex(np.trace(matrix))
    if abs(trace) > trace_limit:
        raise ValueError(f"delta_p must be traceless; found trace {trace.real:.12g}")

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise ValueError("NOCV diagonalization produced non-finite values")

    channels, unpaired = _match_nocv_channels(
        eigenvalues,
        eigenvectors,
        zero_tolerance=zero,
        pair_tolerance=pair,
    )
    return NOCVDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        channels=channels,
        unpaired_indices=unpaired,
        trace=float(trace.real),
        zero_tolerance=zero,
        pair_tolerance=pair,
    )


def nocv_channel_energies(
    decomposition: NOCVDecomposition,
    f_ts: np.ndarray,
    *,
    hermitian_tolerance: float = 1e-8,
) -> NOCVEnergyDecomposition:
    """Assign an ``F_TS`` orbital-energy contribution to every NOCV channel.

    ``f_ts`` must be the transition-state Fock matrix in the same orthonormal
    basis as the NOCV eigenvectors, and its values must be in Ry.  For a paired
    channel the contribution is::

        delta_E_i = v_i [<psi_i|F_TS|psi_i> - <psi_-i|F_TS|psi_-i>]

    The function also returns the full trace contribution so paired and
    unpaired eigenvectors can be checked against ``Tr(delta_p @ F_TS)``.
    A spin-polarized calculation calls this function once per spin channel and
    sums the resulting energies.

    Args:
        decomposition: NOCV eigensystem from :func:`diagonalize_nocv`.
        f_ts: Transition-state Fock matrix in the orthonormal NOCV basis, in Ry.
        hermitian_tolerance: Absolute tolerance for the Hermiticity check.

    Returns:
        Per-channel energies, the paired sum, the unpaired remainder and the
        full orbital-energy contribution, all converted to eV.

    Raises:
        ValueError: If the eigensystem or Fock matrix is inconsistent,
            non-Hermitian or non-finite.
    """

    limit = _validate_tolerance(hermitian_tolerance, "hermitian_tolerance")
    eigenvalues = np.asarray(decomposition.eigenvalues, dtype=float)
    eigenvectors = np.asarray(decomposition.eigenvectors, dtype=np.complex128)
    if eigenvalues.ndim != 1 or eigenvectors.ndim != 2:
        raise ValueError("NOCV eigenvalues and eigenvectors have invalid dimensions")
    if eigenvectors.shape[0] != eigenvectors.shape[1]:
        raise ValueError("NOCV eigenvectors must form a square matrix")
    if eigenvectors.shape[1] != eigenvalues.size:
        raise ValueError("NOCV eigenvector count does not match the eigenvalues")

    fock = np.asarray(f_ts, dtype=np.complex128)
    if fock.shape != eigenvectors.shape:
        raise ValueError("f_ts must match the NOCV eigenvector dimension")
    if not np.all(np.isfinite(fock)):
        raise ValueError("f_ts contains non-finite values")
    if not np.allclose(fock, fock.conj().T, rtol=0.0, atol=limit):
        raise ValueError("f_ts must be Hermitian")

    expectations = np.real(
        np.einsum(
            "ij,ij->j",
            eigenvectors.conj(),
            fock @ eigenvectors,
        )
    )
    if not np.all(np.isfinite(expectations)):
        raise ValueError("F_TS expectations are not finite")

    channels = tuple(
        NOCVChannelEnergy(
            index=channel.index,
            eigenvalue=channel.eigenvalue,
            positive_expectation_ry=float(expectations[channel.positive_index]),
            negative_expectation_ry=float(expectations[channel.negative_index]),
            energy_ev=float(
                channel.eigenvalue
                * (expectations[channel.positive_index] - expectations[channel.negative_index])
                * RY_TO_EV
            ),
        )
        for channel in decomposition.channels
    )
    paired_total = float(sum(channel.energy_ev for channel in channels))
    total = float(np.sum(eigenvalues * expectations) * RY_TO_EV)
    if not np.isfinite(paired_total) or not np.isfinite(total):
        raise ValueError("NOCV channel energies are not finite")
    return NOCVEnergyDecomposition(
        channels=channels,
        paired_total_ev=paired_total,
        unpaired_total_ev=float(total - paired_total),
        total_ev=total,
    )
