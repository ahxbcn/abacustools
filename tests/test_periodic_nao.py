"""Tests for the strict periodic pre-NAO implementation."""

from __future__ import annotations

import numpy as np

from abacustools.data.periodic_nao import (
    _generalized_eigh,
    _partition_basis,
    _schmidt_coefficient_matrix,
    _symmetric_orthog_coefficients,
)


def test_generalized_eigh_returns_overlap_normalized_vectors():
    density = np.diag([1.5, 0.3])
    overlap = np.array([[1.0, 0.2], [0.2, 1.0]])
    values, vectors = _generalized_eigh(density, overlap)
    assert np.all(np.diff(values) <= 0.0)
    assert np.allclose(vectors.T @ overlap @ vectors, np.eye(2), atol=1e-12)
    assert np.allclose(vectors.T @ density @ vectors, np.diag(values), atol=1e-12)


def test_schmidt_coefficients_orthogonalize_rydberg_to_nmb():
    overlap = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])
    coefficients = _schmidt_coefficient_matrix(overlap, 1)
    transformed = coefficients.T @ overlap @ coefficients
    assert abs(transformed[0, 1]) < 1e-12
    assert abs(transformed[0, 2]) < 1e-12


def test_weighted_coefficients_make_metric_identity():
    overlap = np.array([[1.0, 0.2], [0.2, 1.0]])
    weights = np.array([1.8, 0.4])
    coefficients = _symmetric_orthog_coefficients(overlap, weights)
    assert np.allclose(
        coefficients.T @ overlap @ coefficients,
        np.eye(2),
        atol=1e-12,
    )


def test_partition_mapping_moves_true_entries_first():
    mask = np.array([False, True, False, True])
    mapping, inverse = _partition_basis(mask)
    assert mapping.tolist() == [1, 3, 0, 2]
    assert inverse.tolist() == [2, 0, 3, 1]
