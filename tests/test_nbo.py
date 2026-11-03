"""Tests for Periodic NBO helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from abacustools.data.nbo import (
    NBOComponent,
    NBOWavefunction,
    PeriodicNBOAnalysis,
    _generalized_pair_occupancies,
    _natural_minimum_basis_count,
    _pair_budget_occupancies,
    _periodic_images_within_cutoff,
    _valence_lone_pair_count,
    kpoint_matrices_to_real_space,
)


def test_kpoint_matrices_to_real_space_round_trip():
    kpoints = [
        ((0.0, 0.0, 0.0), 0.5, np.eye(2)),
        ((0.5, 0.0, 0.0), 0.5, np.diag([1.0, -1.0])),
    ]
    blocks = kpoint_matrices_to_real_space(kpoints, [(0, 0, 0), (1, 0, 0)])
    assert np.allclose(blocks[(0, 0, 0)], np.diag([1.0, 0.0]))
    assert np.allclose(blocks[(1, 0, 0)], np.diag([0.0, 1.0]))


def test_kpoint_matrices_can_skip_a_second_weight():
    kpoints = [
        ((0.0, 0.0, 0.0), 0.5, 2.0 * np.eye(2)),
        ((0.5, 0.0, 0.0), 0.5, 2.0 * np.diag([1.0, -1.0])),
    ]
    blocks = kpoint_matrices_to_real_space(
        kpoints,
        [(0, 0, 0)],
        apply_weight=False,
    )
    assert np.allclose(blocks[(0, 0, 0)], np.diag([4.0, 0.0]))


def test_generalized_pair_projection_returns_density_eigenvalues():
    density = np.diag([1.6, 0.2])
    overlap = np.eye(2)
    high, low = _generalized_pair_occupancies(density, overlap)
    assert high == pytest.approx(1.6)
    assert low == pytest.approx(0.2)


def test_pair_budget_occupancies_preserve_projected_split():
    bond, anti = _pair_budget_occupancies(1.0, 1.6, 0.2)
    assert bond + anti == pytest.approx(2.0)
    assert bond / anti == pytest.approx(8.0)


def test_h2o_valence_lone_pair_count():
    assert _valence_lone_pair_count(6.0, 2.9143) == 2
    assert _valence_lone_pair_count(1.0, 1.4572) == 0
    assert _valence_lone_pair_count(4.0, 4.0) == 0


def test_periodic_images_are_expanded_inside_cutoff():
    structure = SimpleNamespace(
        cell=np.diag([4.211, 4.211, 4.211]),
        coords_direct=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    images = _periodic_images_within_cutoff(structure, 0, 1, 2.8, 1)
    assert len(images) == 2
    assert {image for image, _distance in images} == {(0, 0, 0), (-1, 0, 0)}
    assert all(abs(distance - 2.1055) < 1.0e-6 for _image, distance in images)


def test_natural_minimum_basis_counts_for_common_elements():
    assert _natural_minimum_basis_count(1, 1) == 1
    assert _natural_minimum_basis_count(8, 6) == 4
    assert _natural_minimum_basis_count(14, 4) == 4
    assert _natural_minimum_basis_count(26, 8) == 9


def test_periodic_nbo_report_serialises_complex_coefficients():
    orbital = NBOWavefunction(
        index=1,
        kind="lone_pair",
        occupancy=1.9,
        centers=((0, (0, 0, 0)),),
        components=(
            NBOComponent(
                atom=0,
                image=(0, 0, 0),
                coefficients=(1.0 + 0.0j, 0.5j),
            ),
        ),
    )
    report = PeriodicNBOAnalysis(
        job="job",
        nspin=1,
        gamma_only=False,
        kpoint_count=1,
        basis_functions=2,
        total_electrons=2.0,
        occupancy_convention="test",
        atoms=(),
        natural_orbitals=(),
        wavefunctions=(orbital,),
        donor_acceptor=(),
        data_files=(),
    ).to_dict()
    coefficients = report["wavefunctions"][0]["components"][0]["coefficients"]
    assert coefficients == [1.0, [0.0, 0.5]]
    assert report["population_method"] == "mulliken"
    assert report["occupancy_method"] == "NHO density-matrix projection"
