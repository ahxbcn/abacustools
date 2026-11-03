"""Integration tests for the reference periodic NHO search."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from abacustools.data.nbo import NBOShell
from abacustools.data.nbo_search import search_periodic_nbos
from abacustools.data.periodic_nao import PeriodicNAOResult


def _two_atom_density() -> PeriodicNAOResult:
    vector = np.ones(2) / np.sqrt(2.0)
    density = 2.0 * np.outer(vector, vector)
    structure = SimpleNamespace(
        atoms=[
            SimpleNamespace(element="H", label="H"),
            SimpleNamespace(element="H", label="H"),
        ],
        cell=np.diag([10.0, 10.0, 10.0]),
        coords_direct=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
    )
    return PeriodicNAOResult(
        transformations=(np.eye(2, dtype=complex),),
        kpoint_overlaps=(np.eye(2, dtype=complex),),
        occupancies=np.diag(density).copy(),
        populations=np.asarray([1.0, 1.0]),
        charges=np.asarray([0.0, 0.0]),
        density_blocks={
            (0, 0, 0): density.astype(complex),
            (1, 0, 0): np.zeros((2, 2), dtype=complex),
            (-1, 0, 0): np.zeros((2, 2), dtype=complex),
            (0, 1, 0): np.zeros((2, 2), dtype=complex),
            (0, -1, 0): np.zeros((2, 2), dtype=complex),
            (0, 0, 1): np.zeros((2, 2), dtype=complex),
            (0, 0, -1): np.zeros((2, 2), dtype=complex),
        },
        fock_blocks=None,
        spin_density_blocks=None,
        overlap_at_origin=np.eye(2),
        basis_l=np.zeros(2, dtype=int),
        basis_shell=np.asarray([1, 2]),
        nmb_mask=np.asarray([True, True]),
        weights=np.asarray([1.0, 1.0]),
    ), structure


def test_two_center_bond_and_antibond_are_found():
    nao, structure = _two_atom_density()
    shells = (
        NBOShell(0, "H", 0, 0, (0,)),
        NBOShell(1, "H", 0, 0, (1,)),
    )
    result = search_periodic_nbos(
        structure,
        [(0, 1), (1, 2)],
        shells,
        nao,
    )
    bonds = [item for item in result.wavefunctions if item.kind == "bond"]
    antibonds = [item for item in result.wavefunctions if item.kind == "antibond"]
    assert len(bonds) == 1
    assert len(antibonds) == 1
    assert np.isclose(bonds[0].occupancy, 2.0)
    assert np.isclose(antibonds[0].occupancy, 0.0, atol=1e-10)
    assert np.isclose(result.assigned_electrons, 2.0)
