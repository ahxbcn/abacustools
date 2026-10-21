from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import abacustools.data.lcao as lcao
from abacustools.commands.workflow.eda import prepare_gamma_state_job
from abacustools.data.eda import read_gamma_fixed_density_result
from abacustools.data.mayer import read_csr_matrix_blocks
from abacustools.io.abacus import ReadInput
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
            AbacusATOM(label="H", coord=(0.0, 0.0, 0.0), pp="H.upf", orb="H.orb"),
            AbacusATOM(label="H", coord=(1.0, 0.0, 0.0), pp="H.upf", orb="H.orb"),
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


def _reference_gamma_job(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    structure = _writer_structure()
    assert structure.write(job / "STRU")
    (job / "H.upf").write_text("", encoding="utf-8")
    (job / "H.orb").write_text("", encoding="utf-8")
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        "gamma_only 1\n"
        "nspin 1\n"
        "stru_file STRU\n"
        "pseudo_dir .\n"
        "orbital_dir .\n",
        encoding="utf-8",
    )
    return job


def test_prepare_gamma_state_job_writes_fixed_density_input(tmp_path):
    job = _reference_gamma_job(tmp_path)
    matrix = np.array([[1.0, 0.25], [0.25, 2.0]])
    state = _state_with_densities(matrix, 0.5 * matrix, (1,))

    output = prepare_gamma_state_job(
        job,
        [state],
        "eda/states/rho0",
        field="p0",
    )

    inputs = ReadInput(str(output / "INPUT"))
    assert inputs.get("calculation") == "scf"
    assert inputs.get("init_chg") == "dm"
    assert inputs.get("read_file_dir") == "."
    assert float(inputs.get("mixing_beta")) == 0.0
    assert int(inputs.get("scf_nmax")) == 1
    assert int(inputs.get("out_hsk")) == 1
    assert (output / "STRU").is_file()
    assert (output / "H.upf").is_file()
    assert (output / "H.orb").is_file()
    np.testing.assert_allclose(
        read_csr_matrix_blocks(output / "dmrs1_nao.csr")[(0, 0, 0)],
        matrix,
    )


def test_prepare_gamma_state_job_requires_override(tmp_path):
    job = _reference_gamma_job(tmp_path)
    state = _state_with_densities(np.eye(2), 0.5 * np.eye(2), (1,))
    destination = "eda/states/rho0"

    prepare_gamma_state_job(job, [state], destination)
    with pytest.raises(RuntimeError, match="already exists"):
        prepare_gamma_state_job(job, [state], destination)
    output = prepare_gamma_state_job(job, [state], destination, override=True)
    assert output.is_dir()


def test_prepare_gamma_state_job_rejects_non_gamma_state(tmp_path):
    job = _reference_gamma_job(tmp_path)
    state = _state_with_densities(np.eye(2), 0.5 * np.eye(2), (1,))
    state.direct = (0.5, 0.0, 0.0)

    with pytest.raises(ValueError, match="Gamma"):
        prepare_gamma_state_job(job, [state], tmp_path / "state")


def _write_develop_matrix(path: Path, matrix: np.ndarray) -> None:
    lines = [f"# rows {matrix.shape[0]}", f"# columns {matrix.shape[1]}"]
    for row in range(matrix.shape[0]):
        values = " ".join(
            f"({matrix[row, column].real:.12f},{matrix[row, column].imag:.12f})"
            for column in range(row, matrix.shape[1])
        )
        lines.append(f"Row {row + 1}")
        lines.append(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixed_density_job(
    tmp_path: Path,
    *,
    nspin: int = 1,
    energy: float = -10.0,
    vdw: float = -0.5,
) -> tuple[Path, dict[int, np.ndarray]]:
    job = tmp_path / "state"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        "gamma_only 1\n"
        f"nspin {nspin}\n"
        "suffix ABACUS\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(
        f"E_KohnSham = {energy:.8f} eV\n"
        f"E_vdwD3 = {vdw:.8f} eV\n"
        "charge density convergence is achieved\n"
        f"final etot is {energy:.8f} eV\n",
        encoding="utf-8",
    )
    matrices = {}
    for spin in range(1, nspin + 1):
        matrix = np.array([[1.0 + spin, 0.25j], [-0.25j, 2.0 + spin]])
        matrices[spin] = matrix
        name = "hk_nao.txt" if nspin == 1 else f"hks{spin}_nao.txt"
        _write_develop_matrix(output / name, matrix)
    return job, matrices


def test_read_gamma_fixed_density_result_single_spin(tmp_path):
    job, matrices = _fixed_density_job(tmp_path)

    result = read_gamma_fixed_density_result(job)

    assert result.energy_ev == pytest.approx(-10.0)
    assert result.vdw_energy_ev == pytest.approx(-0.5)
    assert result.electronic_energy_ev == pytest.approx(-9.5)
    assert result.nspin == 1
    np.testing.assert_allclose(result.hamiltonians[1], matrices[1])


def test_read_gamma_fixed_density_result_two_spins(tmp_path):
    job, matrices = _fixed_density_job(tmp_path, nspin=2)

    result = read_gamma_fixed_density_result(job)

    assert set(result.hamiltonians) == {1, 2}
    np.testing.assert_allclose(result.hamiltonians[1], matrices[1])
    np.testing.assert_allclose(result.hamiltonians[2], matrices[2])


def test_read_gamma_fixed_density_result_rejects_non_gamma(tmp_path):
    job, _matrices = _fixed_density_job(tmp_path)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        "gamma_only 0\n"
        "nspin 1\n"
        "suffix ABACUS\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gamma_only"):
        read_gamma_fixed_density_result(job)


def test_read_gamma_fixed_density_result_requires_hamiltonian(tmp_path):
    job, _matrices = _fixed_density_job(tmp_path)
    (job / "OUT.ABACUS" / "hk_nao.txt").unlink()

    with pytest.raises(FileNotFoundError, match="Hamiltonian"):
        read_gamma_fixed_density_result(job)
