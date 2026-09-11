"""Tests for the DFT+U linear-response workflow."""

from __future__ import annotations

import numpy as np
import pytest

from abacustools.data.dftu import (
    OccupationData,
    ResponseData,
    compute_occupation_eigenvalues,
    compute_response_function,
    parse_onsite_dm,
    task_names_for_u_values,
    total_occupation,
)


class TestTaskNames:
    """Test task name generation."""

    def test_task_names_basic(self):
        names = task_names_for_u_values([0.0, 2.0, 4.0])
        assert names == ["dftu_U000p00", "dftu_U002p00", "dftu_U004p00"]

    def test_task_names_with_decimals(self):
        names = task_names_for_u_values([0.5, 1.5, 2.5])
        assert names == ["dftu_U000p50", "dftu_U001p50", "dftu_U002p50"]


class TestOccupationEigenvalues:
    """Test occupation matrix diagonalization."""

    def test_diagonal_matrix(self):
        matrix = np.diag([0.5, 0.8, 1.0, 0.3, 0.6])
        eigenvalues = compute_occupation_eigenvalues(matrix)
        np.testing.assert_allclose(eigenvalues, [0.3, 0.5, 0.6, 0.8, 1.0])

    def test_symmetric_matrix(self):
        matrix = np.array([
            [1.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9],
        ])
        eigenvalues = compute_occupation_eigenvalues(matrix)
        assert len(eigenvalues) == 5
        assert all(eigenvalues >= 0)

    def test_total_occupation(self):
        eigenvalues = np.array([0.3, 0.5, 0.6, 0.8, 1.0])
        total = total_occupation(eigenvalues)
        assert abs(total - 3.2) < 1e-10


class TestResponseFunction:
    """Test response function computation."""

    def test_linear_response_basic(self):
        u_values = [0.0, 2.0, 4.0, 6.0, 8.0]
        occupations = [5.0, 4.8, 4.6, 4.4, 4.2]
        chi_0, chi, hubbard_u = compute_response_function(u_values, occupations)
        # Linear decrease: dn/dU = -0.1
        assert chi_0 > 0
        assert chi > 0

    def test_insufficient_data(self):
        with pytest.raises(ValueError, match="At least 2"):
            compute_response_function([0.0], [5.0])


class TestParseOnsiteDm:
    """Test onsite.dm file parsing."""

    def test_parse_simple(self, tmp_path):
        content = """atoms  0
L  2
zeta  0
spin  0
  0.5  0.1  0.0  0.0  0.0
  0.1  0.6  0.0  0.0  0.0
  0.0  0.0  0.8  0.0  0.0
  0.0  0.0  0.0  0.7  0.0
  0.0  0.0  0.0  0.0  0.9
spin  1
  0.9  0.0  0.0  0.0  0.0
  0.0  0.8  0.0  0.0  0.0
  0.0  0.0  0.7  0.0  0.0
  0.0  0.0  0.0  0.6  0.0
  0.0  0.0  0.0  0.0  0.5
"""
        dm_file = tmp_path / "onsite.dm"
        dm_file.write_text(content)

        results = parse_onsite_dm(dm_file)
        assert len(results) == 1
        assert results[0].atom_index == 0
        assert results[0].angular_momentum == 2
        assert len(results[0].occupation_matrix) == 2  # spin 0 and 1
        assert results[0].occupation_matrix[0].shape == (5, 5)
        assert results[0].occupation_matrix[1].shape == (5, 5)

    def test_parse_multiple_atoms(self, tmp_path):
        content = """atoms  0
L  2
zeta  0
spin  0
  0.5  0.0  0.0  0.0  0.0
  0.0  0.6  0.0  0.0  0.0
  0.0  0.0  0.8  0.0  0.0
  0.0  0.0  0.0  0.7  0.0
  0.0  0.0  0.0  0.0  0.9
atoms  1
L  2
zeta  0
spin  0
  0.9  0.0  0.0  0.0  0.0
  0.0  0.8  0.0  0.0  0.0
  0.0  0.0  0.7  0.0  0.0
  0.0  0.0  0.0  0.6  0.0
  0.0  0.0  0.0  0.0  0.5
"""
        dm_file = tmp_path / "onsite.dm"
        dm_file.write_text(content)

        results = parse_onsite_dm(dm_file)
        assert len(results) == 2
        assert results[0].atom_index == 0
        assert results[1].atom_index == 1

    def test_parse_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_onsite_dm(tmp_path / "nonexistent.dm")


class TestDataClasses:
    """Test data class structures."""

    def test_occupation_data(self):
        matrix = np.eye(5) * 0.5
        occ = OccupationData(
            atom_index=0,
            angular_momentum=2,
            zeta=0,
            occupation_matrix={0: matrix, 1: matrix},
            eigenvalues={},
        )
        assert occ.atom_index == 0
        assert occ.angular_momentum == 2
        assert len(occ.occupation_matrix) == 2

    def test_response_data(self):
        resp = ResponseData(
            atom_index=0,
            angular_momentum=2,
            u_values=[0.0, 2.0, 4.0],
            occupations=[5.0, 4.8, 4.6],
            chi_0=0.1,
            chi=0.05,
            hubbard_u=10.0,
        )
        assert resp.atom_index == 0
        assert len(resp.u_values) == 3
        assert resp.hubbard_u == 10.0
