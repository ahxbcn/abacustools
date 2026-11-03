"""Tests for Periodic NBO helpers."""

from __future__ import annotations

import numpy as np

from abacustools.data.nbo import (
    NBOComponent,
    NBOWavefunction,
    PeriodicNBOAnalysis,
    _natural_minimum_basis_count,
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
