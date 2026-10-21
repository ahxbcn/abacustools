"""Tests for pEDA natural orbitals for chemical valence."""

from __future__ import annotations

import numpy as np
import pytest

from abacustools.core.constant import RY_TO_EV
from abacustools.data.nocv import diagonalize_nocv, nocv_channel_energies


def _reconstruct(result):
    matrix = np.zeros_like(result.eigenvectors)
    basis = result.eigenvectors
    for index, value in enumerate(result.eigenvalues):
        vector = basis[:, index]
        matrix += value * np.outer(vector, vector.conj())
    return matrix


def test_diagonalize_nocv_pairs_opposite_eigenvalues():
    delta_p = np.diag([-2.0, -0.5, 0.0, 0.5, 2.0])

    result = diagonalize_nocv(delta_p)

    assert result.trace == pytest.approx(0.0)
    assert [channel.eigenvalue for channel in result.channels] == pytest.approx([2.0, 0.5])
    assert [channel.mismatch for channel in result.channels] == pytest.approx([0.0, 0.0])
    assert result.unpaired_indices == (2,)
    np.testing.assert_allclose(_reconstruct(result), delta_p, atol=1e-12)


def test_diagonalize_nocv_reconstructs_complex_hermitian_matrix():
    rng = np.random.default_rng(4)
    random = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    unitary, _ = np.linalg.qr(random)
    delta_p = unitary @ np.diag([-1.2, -0.3, 0.3, 1.2]) @ unitary.conj().T

    result = diagonalize_nocv(delta_p)

    assert len(result.channels) == 2
    assert [channel.eigenvalue for channel in result.channels] == pytest.approx([1.2, 0.3])
    np.testing.assert_allclose(_reconstruct(result), delta_p, atol=1e-12)
    for channel in result.channels:
        np.testing.assert_allclose(
            channel.positive_vector.conj() @ channel.negative_vector,
            0.0,
            atol=1e-12,
        )


def test_diagonalize_nocv_keeps_unmatched_eigenvalues_unpaired():
    delta_p = np.diag([-1.0, 0.5, 0.5])

    result = diagonalize_nocv(delta_p)

    assert result.channels == ()
    assert result.unpaired_indices == (0, 1, 2)
    np.testing.assert_allclose(_reconstruct(result), delta_p, atol=1e-12)


def test_diagonalize_nocv_rejects_nonzero_trace():
    with pytest.raises(ValueError, match="traceless"):
        diagonalize_nocv(np.diag([1.0, 1.0]))


def test_diagonalize_nocv_rejects_non_hermitian_matrix():
    with pytest.raises(ValueError, match="Hermitian"):
        diagonalize_nocv(np.array([[0.0, 1.0], [0.0, 0.0]]))


def test_diagonalize_nocv_rejects_invalid_tolerance():
    with pytest.raises(ValueError, match="pair_tolerance"):
        diagonalize_nocv(np.diag([-1.0, 1.0]), pair_tolerance=-1.0)


def test_nocv_channel_energies_assign_paired_fock_contributions():
    delta_p = np.diag([-2.0, -0.5, 0.0, 0.5, 2.0])
    f_ts = np.diag([-0.1, -0.2, 0.05, -0.3, -0.4])
    decomposition = diagonalize_nocv(delta_p)

    energies = nocv_channel_energies(decomposition, f_ts)

    assert len(energies.channels) == 2
    assert energies.channels[0].index == 1
    assert energies.channels[0].eigenvalue == pytest.approx(2.0)
    assert energies.channels[0].positive_expectation_ry == pytest.approx(-0.4)
    assert energies.channels[0].negative_expectation_ry == pytest.approx(-0.1)
    assert energies.channels[0].energy_ev == pytest.approx(-0.6 * RY_TO_EV)
    assert energies.channels[1].energy_ev == pytest.approx(-0.05 * RY_TO_EV)
    assert energies.paired_total_ev == pytest.approx(-0.65 * RY_TO_EV)
    assert energies.unpaired_total_ev == pytest.approx(0.0)
    assert energies.total_ev == pytest.approx(float(np.real(np.trace(delta_p @ f_ts))) * RY_TO_EV)


def test_nocv_channel_energies_keep_unpaired_fock_contribution():
    delta_p = np.diag([-1.0, -0.5, 0.25, 0.25, 1.0])
    f_ts = np.diag([0.0, 0.2, 0.0, 0.0, 0.0])
    decomposition = diagonalize_nocv(delta_p)

    energies = nocv_channel_energies(decomposition, f_ts)

    assert len(energies.channels) == 1
    assert energies.paired_total_ev == pytest.approx(0.0)
    assert energies.unpaired_total_ev == pytest.approx(-0.1 * RY_TO_EV)
    assert energies.total_ev == pytest.approx(-0.1 * RY_TO_EV)


def test_nocv_channel_energies_reject_non_hermitian_fock():
    decomposition = diagonalize_nocv(np.diag([-1.0, 1.0]))

    with pytest.raises(ValueError, match="Hermitian"):
        nocv_channel_energies(
            decomposition,
            np.array([[0.0, 1.0], [0.0, 0.0]]),
        )


def test_nocv_channel_energies_handle_complex_fock_matrices():
    rng = np.random.default_rng(11)
    random = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    unitary, _ = np.linalg.qr(random)
    delta_p = unitary @ np.diag([-1.0, 1.0]) @ unitary.conj().T
    f_ts = unitary @ np.diag([-0.2, 0.3]) @ unitary.conj().T

    energies = nocv_channel_energies(diagonalize_nocv(delta_p), f_ts)

    assert energies.channels[0].energy_ev == pytest.approx(0.5 * RY_TO_EV)
    assert energies.total_ev == pytest.approx(0.5 * RY_TO_EV)
    assert energies.unpaired_total_ev == pytest.approx(0.0)
