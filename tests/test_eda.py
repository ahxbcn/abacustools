from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import abacustools.data.lcao as lcao
from abacustools.data.mayer import read_csr_matrix_blocks
from abacustools.io.stru import AbacusATOM, AbacusSTRU
from abacustools.data.lcao import (
    KPointOrbitals,
    assemble_peda_kpoint_state,
    build_peda_kpoint_states,
    write_peda_state_dmr,
)


ROOT2 = np.sqrt(0.5)


def _full_coefficients() -> np.ndarray:
    return np.array(
        [
            [ROOT2, 0.0, ROOT2, 0.0],
            [ROOT2, 0.0, -ROOT2, 0.0],
            [0.0, ROOT2, 0.0, ROOT2],
            [0.0, ROOT2, 0.0, -ROOT2],
        ],
        dtype=complex,
    )


def _fragment_coefficients() -> np.ndarray:
    return np.eye(2, dtype=complex)


def _state(
    *,
    coefficients: dict[int, np.ndarray],
    occupations: dict[int, np.ndarray],
    overlap: np.ndarray,
    direct: tuple[float, float, float] = (0.0, 0.0, 0.0),
    index: int = 1,
    weight: float = 1.0,
) -> KPointOrbitals:
    return KPointOrbitals(
        index=index,
        direct=direct,
        weight=weight,
        overlap=overlap,
        coefficients=coefficients,
        energies={spin: np.zeros(values.shape[1]) for spin, values in coefficients.items()},
        occupations=occupations,
        data_files=(),
    )


def test_assemble_peda_kpoint_state_closed_shell():
    overlap = np.eye(4)
    full_coefficients = _full_coefficients()
    fragment_coefficients = _fragment_coefficients()
    full = _state(
        coefficients={1: full_coefficients},
        occupations={1: np.array([2.0, 2.0, 0.0, 0.0])},
        overlap=overlap,
    )
    part_a = _state(
        coefficients={1: fragment_coefficients},
        occupations={1: np.array([2.0, 0.0])},
        overlap=np.eye(2),
    )
    part_b = _state(
        coefficients={1: fragment_coefficients},
        occupations={1: np.array([2.0, 0.0])},
        overlap=np.eye(2),
    )

    state = assemble_peda_kpoint_state(
        full,
        part_a,
        part_b,
        [(0, 2), (2, 4)],
        [0],
        [1],
        nspin=1,
    )

    channel = state.channels[1]
    np.testing.assert_allclose(channel.p0_lambda, np.diag([2.0, 2.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.trace(channel.p0_nao @ overlap), 4.0)
    np.testing.assert_allclose(np.trace(channel.p_ab_nao @ overlap), 4.0)
    np.testing.assert_allclose(np.trace(channel.p_ts_nao @ overlap), 4.0)
    np.testing.assert_allclose(np.trace(channel.delta_p_lambda), 0.0, atol=1e-12)
    np.testing.assert_allclose(channel.p0_nao @ overlap @ channel.p0_nao, 2.0 * channel.p0_nao)
    assert np.linalg.norm(channel.delta_p_lambda) > 0.1


def test_assemble_peda_kpoint_state_two_spins():
    overlap = np.eye(4)
    full_coefficients = _full_coefficients()
    fragment_coefficients = _fragment_coefficients()
    full = _state(
        coefficients={1: full_coefficients, 2: full_coefficients},
        occupations={
            1: np.array([1.0, 1.0, 0.0, 0.0]),
            2: np.array([1.0, 1.0, 0.0, 0.0]),
        },
        overlap=overlap,
    )
    part_a = _state(
        coefficients={1: fragment_coefficients, 2: fragment_coefficients},
        occupations={1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])},
        overlap=np.eye(2),
    )
    part_b = _state(
        coefficients={1: fragment_coefficients, 2: fragment_coefficients},
        occupations={1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])},
        overlap=np.eye(2),
    )

    state = assemble_peda_kpoint_state(
        full,
        part_a,
        part_b,
        [(0, 2), (2, 4)],
        [0],
        [1],
        nspin=2,
    )

    assert set(state.channels) == {1, 2}
    for channel in state.channels.values():
        np.testing.assert_allclose(np.trace(channel.p0_nao @ overlap), 2.0)
        np.testing.assert_allclose(channel.p0_nao @ overlap @ channel.p0_nao, channel.p0_nao)
        np.testing.assert_allclose(np.trace(channel.delta_p_lambda), 0.0, atol=1e-12)


def test_assemble_peda_kpoint_state_rejects_fractional_fragment_occupation():
    full = _state(
        coefficients={1: _full_coefficients()},
        occupations={1: np.array([2.0, 2.0, 0.0, 0.0])},
        overlap=np.eye(4),
    )
    part_a = _state(
        coefficients={1: _fragment_coefficients()},
        occupations={1: np.array([1.5, 0.5])},
        overlap=np.eye(2),
    )
    part_b = _state(
        coefficients={1: _fragment_coefficients()},
        occupations={1: np.array([2.0, 0.0])},
        overlap=np.eye(2),
    )

    with pytest.raises(lcao.LCAODataError, match="integer fragment occupations"):
        assemble_peda_kpoint_state(
            full,
            part_a,
            part_b,
            [(0, 2), (2, 4)],
            [0],
            [1],
            nspin=1,
        )


def test_build_peda_kpoint_states_aligns_fragment_kpoints(monkeypatch):
    overlap = np.eye(4)
    full_coefficients = _full_coefficients()
    fragment_coefficients = _fragment_coefficients()

    def make_state(direct, index):
        return _state(
            coefficients={1: full_coefficients},
            occupations={1: np.array([2.0, 2.0, 0.0, 0.0])},
            overlap=overlap,
            direct=direct,
            index=index,
            weight=0.5,
        )

    def make_fragment(direct, index):
        return _state(
            coefficients={1: fragment_coefficients},
            occupations={1: np.array([2.0, 0.0])},
            overlap=np.eye(2),
            direct=direct,
            index=index,
            weight=0.5,
        )

    points = [((0.0, 0.0, 0.0), 1), ((0.5, 0.0, 0.0), 2)]
    full_states = [make_state(direct, index) for direct, index in points]
    fragment_a_states = [make_fragment(direct, index) for direct, index in reversed(points)]
    fragment_b_states = [make_fragment(direct, index) for direct, index in points]

    def fake_loader(job, **_kwargs):
        return {
            "full": full_states,
            "part_a": fragment_a_states,
            "part_b": fragment_b_states,
        }[str(job)]

    monkeypatch.setattr(lcao, "load_lcao_kpoint_orbitals", fake_loader)
    monkeypatch.setattr(
        lcao,
        "read_nao_basis_layout",
        lambda _job: (SimpleNamespace(natoms=2), [(0, 2), (2, 4)], 4),
    )

    states = build_peda_kpoint_states(
        "full",
        "part_a",
        "part_b",
        [0],
        [1],
        nspin=1,
    )

    assert [state.direct for state in states] == [point for point, _index in points]


def _writer_structure() -> AbacusSTRU:
    return AbacusSTRU(
        cell=np.diag([4.0, 4.0, 4.0]),
        atoms=[
            AbacusATOM(label="H", coord=(0.0, 0.0, 0.0)),
            AbacusATOM(label="H", coord=(1.0, 0.0, 0.0)),
        ],
        metadata={"lattice_constant": 1.889726, "lattice_name": "test"},
    )


def _state_with_densities(p0: np.ndarray, p_ts: np.ndarray, spins: tuple[int, ...]):
    channels = {
        spin: SimpleNamespace(
            p0_nao=p0,
            p_ab_nao=p_ts,
            p_ts_nao=p_ts,
        )
        for spin in spins
    }
    return SimpleNamespace(
        index=1,
        direct=(0.0, 0.0, 0.0),
        channels=channels,
    )


def test_write_peda_state_dmr_gamma_single_spin(tmp_path):
    matrix = np.array([[1.0, 0.25], [0.25, 2.0]])
    state = _state_with_densities(matrix, 0.5 * matrix, (1,))

    paths = write_peda_state_dmr([state], _writer_structure(), tmp_path, field="p0")

    assert paths[1].name == "dmrs1_nao.csr"
    np.testing.assert_allclose(read_csr_matrix_blocks(paths[1])[(0, 0, 0)], matrix)


def test_write_peda_state_dmr_two_spins_uses_separate_files(tmp_path):
    p0 = np.eye(2)
    p_ts = 0.5 * np.eye(2)
    state = _state_with_densities(p0, p_ts, (1, 2))

    paths = write_peda_state_dmr([state], _writer_structure(), tmp_path, field="p_ts")

    assert set(paths) == {1, 2}
    np.testing.assert_allclose(read_csr_matrix_blocks(paths[1])[(0, 0, 0)], p_ts)
    np.testing.assert_allclose(read_csr_matrix_blocks(paths[2])[(0, 0, 0)], p_ts)
    assert "2 # number of spin directions" in paths[1].read_text()


def test_write_peda_state_dmr_rejects_multik(tmp_path):
    state = _state_with_densities(np.eye(2), np.eye(2), (1,))
    state.direct = (0.5, 0.0, 0.0)

    with pytest.raises(ValueError, match="only Gamma"):
        write_peda_state_dmr([state], _writer_structure(), tmp_path)


def test_write_peda_state_dmr_rejects_unknown_field(tmp_path):
    state = _state_with_densities(np.eye(2), np.eye(2), (1,))

    with pytest.raises(ValueError, match="unknown pEDA state field"):
        write_peda_state_dmr([state], _writer_structure(), tmp_path, field="bad")
