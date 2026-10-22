from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import abacustools.data.lcao as lcao
from abacustools.commands.workflow.eda import (
    prepare_gamma_state_job,
    prepare_peda_state_job,
)
from abacustools.core.constant import COULOMB_EV_ANGSTROM, RY_TO_EV
from abacustools.data.charge import ChargeDensityError, read_job_total_density
from abacustools.data.eda import (
    analyze_gamma_peda_nocv,
    analyze_kpoint_peda,
    compute_peda_energy_terms,
    electron_electron_cross_term,
    electron_nucleus_cross_term,
    electrostatic_cross_term,
    electrostatic_cross_term_from_jobs,
    fragment_nuclei,
    ion_ion_cross_term,
    read_abacus_ewald_energy,
    read_abacus_harris_energy,
    read_gamma_fixed_density_result,
    read_kpoint_fixed_density_result,
)
from abacustools.data.grid import Charge
from abacustools.data.mayer import read_csr_matrix_blocks
from abacustools.data.nocv import diagonalize_nocv
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusATOM, AbacusSTRU
from abacustools.data.lcao import (
    KPointOrbitals,
    assemble_peda_kpoint_state,
    build_peda_kpoint_states,
    real_space_density_to_kpoints,
    suggest_peda_dmr_r_blocks,
    write_peda_state_dmr,
)


ROOT2 = np.sqrt(0.5)

H_ORBITAL = """Element H
Energy Cutoff(Ry) 100
Radius Cutoff(a.u.) 6
Lmax 0
Number of Sorbital--> 1
SUMMARY  END

Mesh 1
dr 0.01
Type L N
0 0 0
1.0
"""


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

    decomposition = diagonalize_nocv(channel.delta_p_lambda)
    assert decomposition.trace == pytest.approx(0.0, abs=1e-10)
    assert float(np.sum(decomposition.eigenvalues)) == pytest.approx(0.0, abs=1e-10)


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


def _state_with_densities(
    p0: np.ndarray,
    p_ts: np.ndarray,
    spins: tuple[int, ...],
    *,
    direct: tuple[float, float, float] = (0.0, 0.0, 0.0),
    index: int = 1,
    weight: float = 1.0,
):
    channels = {
        spin: SimpleNamespace(
            p0_nao=p0,
            p_ab_nao=p_ts,
            p_ts_nao=p_ts,
        )
        for spin in spins
    }
    return SimpleNamespace(
        index=index,
        direct=direct,
        weight=weight,
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


def test_write_peda_state_dmr_rejects_multik_without_r_blocks(tmp_path):
    state = _state_with_densities(
        np.eye(2),
        np.eye(2),
        (1,),
        direct=(0.5, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="explicit r_blocks"):
        write_peda_state_dmr([state], _writer_structure(), tmp_path)


def test_write_peda_state_dmr_multik_two_spins_round_trips_kpoints(tmp_path):
    a = np.array([[1.0, 0.1], [0.1, 2.0]])
    b_up = np.array([[0.2, 0.0], [0.0, -0.2]])
    b_down = np.array([[0.3, 0.0], [0.0, -0.1]])
    states = [
        _state_with_densities(
            a + b_up,
            0.5 * (a + b_up),
            (1,),
            direct=(0.0, 0.0, 0.0),
            index=1,
            weight=0.5,
        ),
        _state_with_densities(
            a - b_up,
            0.5 * (a - b_up),
            (1,),
            direct=(0.5, 0.0, 0.0),
            index=2,
            weight=0.5,
        ),
    ]

    paths = write_peda_state_dmr(
        states,
        _writer_structure(),
        tmp_path,
        field="p0",
        r_blocks=((1, 0, 0), (0, 0, 0)),
    )

    blocks = read_csr_matrix_blocks(paths[1])
    np.testing.assert_allclose(blocks[(0, 0, 0)], a)
    np.testing.assert_allclose(blocks[(1, 0, 0)], b_up)
    reconstructed = real_space_density_to_kpoints(
        blocks,
        [((0.0, 0.0, 0.0), 0.5), ((0.5, 0.0, 0.0), 0.5)],
    )
    np.testing.assert_allclose(reconstructed[0][2], a + b_up, atol=1e-12)
    np.testing.assert_allclose(reconstructed[1][2], a - b_up, atol=1e-12)

    states_both = []
    for index, direct, up_density, down_density in (
        (1, (0.0, 0.0, 0.0), a + b_up, a + b_down),
        (2, (0.5, 0.0, 0.0), a - b_up, a - b_down),
    ):
        states_both.append(
            SimpleNamespace(
                index=index,
                direct=direct,
                weight=0.5,
                channels={
                    1: SimpleNamespace(
                        p0_nao=up_density,
                        p_ab_nao=0.5 * up_density,
                        p_ts_nao=0.5 * up_density,
                    ),
                    2: SimpleNamespace(
                        p0_nao=down_density,
                        p_ab_nao=0.5 * down_density,
                        p_ts_nao=0.5 * down_density,
                    ),
                },
            )
        )

    spin_paths = write_peda_state_dmr(
        states_both,
        _writer_structure(),
        tmp_path / "spin2",
        field="p0",
        r_blocks=((0, 0, 0), (1, 0, 0)),
    )
    assert set(spin_paths) == {1, 2}
    np.testing.assert_allclose(
        read_csr_matrix_blocks(spin_paths[1])[(1, 0, 0)],
        b_up,
    )
    np.testing.assert_allclose(
        read_csr_matrix_blocks(spin_paths[2])[(1, 0, 0)],
        b_down,
    )
    assert "2 # number of spin directions" in spin_paths[1].read_text()


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
    (job / "H.orb").write_text(H_ORBITAL, encoding="utf-8")
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


def _reference_multik_job(tmp_path: Path, *, symmetry: int = 0) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    structure = _writer_structure()
    assert structure.write(job / "STRU")
    (job / "H.upf").write_text("", encoding="utf-8")
    (job / "H.orb").write_text(H_ORBITAL, encoding="utf-8")
    (job / "KPT").write_text(
        "K_POINTS\n"
        "0\n"
        "Gamma\n"
        "2 2 2 0 0 0\n",
        encoding="utf-8",
    )
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        "gamma_only 0\n"
        "nspin 2\n"
        f"symmetry {symmetry}\n"
        "stru_file STRU\n"
        "pseudo_dir .\n"
        "orbital_dir .\n",
        encoding="utf-8",
    )
    return job


def _multik_writer_states() -> list[SimpleNamespace]:
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
    return [
        _state_with_densities(
            matrix,
            0.5 * matrix,
            (1, 2),
            direct=(0.0, 0.0, 0.0),
            index=1,
            weight=0.5,
        ),
        _state_with_densities(
            matrix,
            0.5 * matrix,
            (1, 2),
            direct=(0.5, 0.0, 0.0),
            index=2,
            weight=0.5,
        ),
    ]


def test_prepare_peda_state_job_writes_multik_input_and_dmr(tmp_path):
    job = _reference_multik_job(tmp_path, symmetry=0)

    output = prepare_peda_state_job(
        job,
        _multik_writer_states(),
        "eda/states/rho0",
        field="p0",
        r_blocks=((0, 0, 0), (1, 0, 0)),
    )

    inputs = ReadInput(str(output / "INPUT"))
    assert inputs.get("calculation") == "nscf"
    assert int(inputs.get("gamma_only")) == 0
    assert int(inputs.get("symmetry")) == 0
    assert int(inputs.get("out_hsk")) == 1
    assert (output / "KPT").is_file()
    for spin in (1, 2):
        blocks = read_csr_matrix_blocks(output / f"dmrs{spin}_nao.csr")
        assert set(blocks) == {(0, 0, 0), (1, 0, 0)}


def test_suggest_peda_dmr_r_blocks_from_nao_cutoff(tmp_path):
    job = _reference_multik_job(tmp_path, symmetry=0)

    blocks = suggest_peda_dmr_r_blocks(job)

    assert (0, 0, 0) in blocks
    assert any(block != (0, 0, 0) for block in blocks)


def test_prepare_peda_state_job_generates_r_blocks_for_multik(tmp_path):
    job = _reference_multik_job(tmp_path, symmetry=0)

    output = prepare_peda_state_job(
        job,
        _multik_writer_states(),
        "eda/states/rho0",
        field="p0",
    )

    blocks = read_csr_matrix_blocks(output / "dmrs1_nao.csr")
    assert (0, 0, 0) in blocks
    assert len(blocks) > 1


def test_prepare_peda_state_job_rejects_symmetry_one_multik(tmp_path):
    job = _reference_multik_job(tmp_path, symmetry=1)

    with pytest.raises(ValueError, match="symmetry=-1 or 0"):
        prepare_peda_state_job(
            job,
            _multik_writer_states(),
            "eda/states/rho0",
            r_blocks=((0, 0, 0),),
        )


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
    assert inputs.get("calculation") == "nscf"
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


def test_prepare_gamma_state_job_rewrites_resource_directories(tmp_path):
    job = _reference_gamma_job(tmp_path)
    resources = job / "pp"
    resources.mkdir()
    (job / "H.upf").rename(resources / "H.upf")
    (job / "H.orb").rename(resources / "H.orb")
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        "gamma_only 1\n"
        "nspin 1\n"
        "stru_file STRU\n"
        "pseudo_dir pp\n"
        "orbital_dir pp\n",
        encoding="utf-8",
    )
    state = _state_with_densities(np.eye(2), 0.5 * np.eye(2), (1,))

    output = prepare_gamma_state_job(job, [state], "eda/states/rho0")

    inputs = ReadInput(str(output / "INPUT"))
    assert inputs.get("pseudo_dir") == "."
    assert inputs.get("orbital_dir") == "."
    assert (output / "H.upf").is_file()
    assert (output / "H.orb").is_file()


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
    harris: float = -9.75,
    band: float = -8.0,
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
        f"E_Harris = {harris:.8f} eV\n"
        f"E_band = {band:.8f} eV\n"
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


def _multik_fixed_density_job(
    tmp_path: Path,
    *,
    nspin: int = 2,
    symmetry: int = -1,
    energy: float = -10.0,
    harris: float = -9.75,
    band: float = -8.0,
    vdw: float = -0.5,
) -> tuple[Path, dict[tuple[int, int], np.ndarray]]:
    job = tmp_path / "state"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation nscf\n"
        "basis_type lcao\n"
        "gamma_only 0\n"
        f"nspin {nspin}\n"
        f"symmetry {symmetry}\n"
        "suffix ABACUS\n",
        encoding="utf-8",
    )
    (output / "kpoints").write_text(
        " nkstot now = 2\n"
        "K-POINTS DIRECT COORDINATES\n"
        " KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT\n"
        " 1 0.0 0.0 0.0 0.5000\n"
        " 2 0.5 0.0 0.0 0.5000\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(
        f"E_KohnSham = {energy:.8f} eV\n"
        f"E_Harris = {harris:.8f} eV\n"
        f"E_band = {band:.8f} eV\n"
        f"E_vdwD3 = {vdw:.8f} eV\n"
        "charge density convergence is achieved\n"
        f"final etot is {energy:.8f} eV\n",
        encoding="utf-8",
    )
    matrices: dict[tuple[int, int], np.ndarray] = {}
    for ik in (1, 2):
        for spin in ((1,) if nspin == 1 else (1, 2)):
            matrix = np.array([[1.0 + ik + spin, 0.25j], [-0.25j, 2.0 + ik + spin]])
            matrices[(ik, spin)] = matrix
            name = f"hk{ik}_nao.txt" if nspin == 1 else f"hk{ik}s{spin}_nao.txt"
            _write_develop_matrix(output / name, matrix)
    return job, matrices


def test_read_kpoint_fixed_density_result_multik_two_spins(tmp_path):
    job, matrices = _multik_fixed_density_job(tmp_path)

    result = read_kpoint_fixed_density_result(job)

    assert result.energy_ev == pytest.approx(-10.0)
    assert result.band_energy_ev == pytest.approx(-8.0)
    assert result.electronic_energy_ev == pytest.approx(-9.5)
    assert result.symmetry == -1
    assert result.nspin == 2
    assert [kpoint.index for kpoint in result.kpoints] == [1, 2]
    assert [kpoint.weight for kpoint in result.kpoints] == pytest.approx([0.5, 0.5])
    for kpoint in result.kpoints:
        assert set(kpoint.hamiltonians) == {1, 2}
        for spin in (1, 2):
            np.testing.assert_allclose(
                kpoint.hamiltonians[spin],
                matrices[(kpoint.index, spin)],
            )


def test_read_kpoint_fixed_density_result_accepts_caller_kpoints(tmp_path):
    job, matrices = _multik_fixed_density_job(tmp_path)
    (job / "OUT.ABACUS" / "kpoints").unlink()

    result = read_kpoint_fixed_density_result(
        job,
        kpoints=[((0.0, 0.0, 0.0), 0.5), ((0.5, 0.0, 0.0), 0.5)],
    )

    assert [kpoint.weight for kpoint in result.kpoints] == pytest.approx([0.5, 0.5])
    np.testing.assert_allclose(result.kpoints[0].hamiltonians[1], matrices[(1, 1)])


def test_read_kpoint_fixed_density_result_accepts_time_reversal_symmetry(tmp_path):
    job, _matrices = _multik_fixed_density_job(tmp_path, symmetry=0)

    result = read_kpoint_fixed_density_result(job)

    assert result.symmetry == 0


def test_read_kpoint_fixed_density_result_rejects_symmetry_one(tmp_path):
    job, _matrices = _multik_fixed_density_job(tmp_path, symmetry=1)

    with pytest.raises(ValueError, match="symmetry=-1 or 0"):
        read_kpoint_fixed_density_result(job)


def test_read_gamma_fixed_density_result_single_spin(tmp_path):
    job, matrices = _fixed_density_job(tmp_path)

    result = read_gamma_fixed_density_result(job)

    assert result.energy_ev == pytest.approx(-10.0)
    assert result.harris_energy_ev == pytest.approx(-9.75)
    assert result.band_energy_ev == pytest.approx(-8.0)
    assert result.calculation == "scf"
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


def test_compute_peda_energy_terms_and_fractions():
    terms = compute_peda_energy_terms(
        e_ab=-100.0,
        e_a=-40.0,
        e_b=-30.0,
        e_a0=-45.0,
        e_b0=-35.0,
        e0=-80.0,
        delta_e_elstat=-15.0,
        e_vdw_ab=-1.0,
        e_vdw_a=-0.2,
        e_vdw_b=-0.3,
    )

    assert terms.preparation == pytest.approx(10.0)
    assert terms.electrostatic == pytest.approx(-15.0)
    assert terms.pauli == pytest.approx(5.0)
    assert terms.orbital == pytest.approx(-20.0)
    assert terms.dispersion == pytest.approx(-0.5)
    assert terms.interaction_electronic == pytest.approx(-30.0)
    assert terms.interaction == pytest.approx(-30.5)
    assert terms.bond == pytest.approx(-20.5)
    assert terms.electrostatic_fraction == pytest.approx(15.0 / 35.0)
    assert terms.orbital_fraction == pytest.approx(20.0 / 35.0)
    assert terms.dispersion_fraction == pytest.approx(0.5 / 30.5)


def test_compute_peda_energy_terms_without_dispersion():
    terms = compute_peda_energy_terms(
        e_ab=-100.0,
        e_a=-40.0,
        e_b=-30.0,
        e_a0=-45.0,
        e_b0=-35.0,
        e0=-80.0,
        delta_e_elstat=-15.0,
    )

    assert terms.dispersion == 0.0
    assert terms.dispersion_fraction == 0.0
    assert terms.interaction == pytest.approx(terms.interaction_electronic)


def test_compute_peda_energy_terms_rejects_partial_dispersion():
    with pytest.raises(ValueError, match="supplied together"):
        compute_peda_energy_terms(
            e_ab=-100.0,
            e_a=-40.0,
            e_b=-30.0,
            e_a0=-45.0,
            e_b0=-35.0,
            e0=-80.0,
            delta_e_elstat=-15.0,
            e_vdw_ab=-1.0,
        )


def test_compute_peda_energy_terms_rejects_non_finite_energy():
    with pytest.raises(ValueError, match="finite"):
        compute_peda_energy_terms(
            e_ab=float("nan"),
            e_a=-40.0,
            e_b=-30.0,
            e_a0=-45.0,
            e_b0=-35.0,
            e0=-80.0,
            delta_e_elstat=-15.0,
        )


def _cosine_density(amplitude: float, *, shape: int = 8) -> Charge:
    coordinates = np.arange(shape) / shape
    profile = amplitude * np.cos(2.0 * np.pi * coordinates)
    data = np.broadcast_to(profile[:, None, None], (shape, shape, shape)).copy()
    return Charge(
        data,
        np.eye(3),
        atom_positions=np.zeros((1, 3)),
        atom_types=[0],
        atom_charges=[0.0],
    )


def test_electron_electron_cross_term_against_analytic_cosine():
    density_a = _cosine_density(1.0)
    density_b = _cosine_density(2.0)

    energy = electron_electron_cross_term(density_a, density_b)

    expected = COULOMB_EV_ANGSTROM * (1.0 * 2.0) / (2.0 * np.pi)
    assert energy == pytest.approx(expected, rel=1e-6)


def test_electron_electron_cross_term_rejects_grid_mismatch():
    with pytest.raises(ChargeDensityError, match="grid shape"):
        electron_electron_cross_term(_cosine_density(1.0, shape=8), _cosine_density(1.0, shape=4))


def test_electron_electron_cross_term_rejects_charged_fragment():
    charged = Charge(
        np.ones((4, 4, 4)),
        np.eye(3),
        atom_positions=np.zeros((1, 3)),
        atom_types=[0],
        atom_charges=[0.0],
    )
    neutral = _cosine_density(1.0, shape=4)

    with pytest.raises(ValueError, match="neutral fragments"):
        electron_electron_cross_term(charged, neutral)


def test_electron_electron_cross_term_requires_atom_charges():
    density = Charge(np.zeros((4, 4, 4)), np.eye(3))

    with pytest.raises(ValueError, match="atom_charges"):
        electron_electron_cross_term(density, density)


def test_electron_nucleus_cross_term_against_analytic_cosine():
    density = _cosine_density(1.0)

    energy = electron_nucleus_cross_term(
        density,
        np.array([[0.0, 0.0, 0.0]]),
        [1.0],
    )

    assert energy == pytest.approx(-COULOMB_EV_ANGSTROM / np.pi, rel=1e-6)


def test_electron_nucleus_cross_term_allows_a_positive_cross_energy():
    density = _cosine_density(1.0)

    energy = electron_nucleus_cross_term(
        density,
        np.array([[0.5, 0.0, 0.0]]),
        [1.0],
    )

    assert energy == pytest.approx(COULOMB_EV_ANGSTROM / np.pi, rel=1e-6)


def test_electron_nucleus_cross_term_scales_with_cell_volume():
    length = 2.0
    coordinates = np.arange(8) / 8
    profile = 1.0 + np.cos(2.0 * np.pi * coordinates)
    data = np.broadcast_to(profile[:, None, None], (8, 8, 8)).copy()
    density = Charge(
        data,
        length * np.eye(3),
        atom_positions=np.zeros((1, 3)),
        atom_types=[1],
        atom_charges=[length**3],
    )

    energy = electron_nucleus_cross_term(
        density,
        np.array([[0.0, 0.0, 0.0]]),
        [length**3],
    )

    assert energy == pytest.approx(-COULOMB_EV_ANGSTROM * length**5 / np.pi, rel=1e-6)


def test_electron_nucleus_cross_term_is_periodic():
    density = _cosine_density(1.0)
    origin = electron_nucleus_cross_term(
        density,
        np.array([[0.0, 0.0, 0.0]]),
        [1.0],
    )
    shifted = electron_nucleus_cross_term(
        density,
        np.array([[1.0, 2.0, -1.0]]),
        [1.0],
    )

    assert shifted == pytest.approx(origin, rel=1e-10)


def test_electron_nucleus_cross_term_rejects_charge_mismatch():
    density = _cosine_density(1.0)

    with pytest.raises(ValueError, match="nuclear_charges"):
        electron_nucleus_cross_term(
            density,
            np.zeros((2, 3)),
            [1.0],
        )


def test_electron_nucleus_cross_term_rejects_empty_nuclei():
    density = _cosine_density(1.0)

    with pytest.raises(ValueError, match="at least one nucleus"):
        electron_nucleus_cross_term(
            density,
            np.empty((0, 3)),
            [],
        )


def test_ion_ion_cross_term_from_ewald_energies():
    value = ion_ion_cross_term(-10.0, -6.0, -3.0)

    assert value == pytest.approx(-1.0)


def test_electrostatic_cross_term_sums_all_components():
    value = electrostatic_cross_term(
        electron_electron=-5.0,
        electron_nucleus_ab=-3.0,
        electron_nucleus_ba=-2.0,
        ion_ion=1.0,
    )

    assert value == pytest.approx(-9.0)


def test_electrostatic_cross_term_rejects_non_finite_component():
    with pytest.raises(ValueError, match="finite"):
        electrostatic_cross_term(
            electron_electron=float("inf"),
            electron_nucleus_ab=-3.0,
            electron_nucleus_ba=-2.0,
            ion_ion=1.0,
        )


def test_read_abacus_ewald_energy(tmp_path):
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\ngamma_only 1\nsuffix ABACUS\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(
        "E_Ewald 1.000000 13.605693\n",
        encoding="utf-8",
    )

    assert read_abacus_ewald_energy(job) == pytest.approx(13.605693)


def _minimal_upf(element: str, valence: float) -> str:
    return f"""<UPF version="2.0.1">
  <PP_HEADER element="{element}" z_valence="{valence}" l_max="0" mesh_size="2"/>
  <PP_MESH>
    <PP_R size="2">0.0 1.0</PP_R>
    <PP_RAB size="2">1.0 1.0</PP_RAB>
  </PP_MESH>
  <PP_LOCAL size="2">0.0 0.0</PP_LOCAL>
  <PP_RHOATOM size="2">0.0 0.0</PP_RHOATOM>
</UPF>
"""


def _write_fragment_job(tmp_path, *, hydrogen_label="H"):
    job = tmp_path / "fragment"
    pseudopotentials = job / "pp"
    pseudopotentials.mkdir(parents=True)
    (pseudopotentials / "H.upf").write_text(_minimal_upf("H", 1.0), encoding="utf-8")
    (pseudopotentials / "O.upf").write_text(_minimal_upf("O", 6.0), encoding="utf-8")
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nstru_file STRU\npseudo_dir pp\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(
        f"""ATOMIC_SPECIES
{hydrogen_label} 1.0 H.upf
O 16.0 O.upf

LATTICE_CONSTANT
1.889726125457828

LATTICE_VECTORS
5 0 0
0 5 0
0 0 5

ATOMIC_POSITIONS
Direct

{hydrogen_label}
0.0
1
0 0 0

O
0.0
1
0.5 0.5 0.5
""",
        encoding="utf-8",
    )
    return job


def test_fragment_nuclei_reads_positions_cell_and_valence_charges(tmp_path):
    job = _write_fragment_job(tmp_path)

    fragment = fragment_nuclei(job)

    assert fragment.labels == ("H", "O")
    assert fragment.pseudopotentials == ("H.upf", "O.upf")
    assert fragment.unit == "angstrom"
    np.testing.assert_allclose(
        fragment.positions,
        [[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
        atol=1e-6,
    )
    np.testing.assert_allclose(fragment.cell, 5.0 * np.eye(3), atol=1e-6)
    np.testing.assert_allclose(fragment.charges, [1.0, 6.0])


def test_fragment_nuclei_rejects_missing_pseudopotential(tmp_path):
    job = _write_fragment_job(tmp_path)
    (job / "pp" / "O.upf").unlink()

    with pytest.raises(ValueError, match="pseudopotential file not found"):
        fragment_nuclei(job)


def test_fragment_nuclei_rejects_ghost_atoms(tmp_path):
    job = _write_fragment_job(tmp_path, hydrogen_label="H_empty")

    with pytest.raises(ValueError, match="ghost atoms"):
        fragment_nuclei(job)


def _write_ewald_job(job: Path, energy: float) -> Path:
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True, exist_ok=True)
    if not (job / "INPUT").is_file():
        (job / "INPUT").write_text(
            "INPUT_PARAMETERS\ncalculation scf\nsuffix ABACUS\n",
            encoding="utf-8",
        )
    (output / "running_scf.log").write_text(f"E_Ewald 0.0 {energy}\n", encoding="utf-8")
    return job


def _write_density_fragment_job(
    job: Path,
    *,
    direct_position: tuple[float, float, float],
    amplitude: float,
    density_cell: np.ndarray | None = None,
) -> Path:
    pseudopotentials = job / "pp"
    output = job / "OUT.ABACUS"
    pseudopotentials.mkdir(parents=True)
    output.mkdir(parents=True)
    (pseudopotentials / "H.upf").write_text(_minimal_upf("H", 1.0), encoding="utf-8")
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nsuffix ABACUS\nstru_file STRU\npseudo_dir pp\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(
        f"""ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.889726125457828

LATTICE_VECTORS
1 0 0
0 1 0
0 0 1

ATOMIC_POSITIONS
Direct

H
0.0
1
{direct_position[0]} {direct_position[1]} {direct_position[2]}
""",
        encoding="utf-8",
    )

    coordinates = np.arange(8) / 8
    profile = 1.0 + amplitude * np.cos(2.0 * np.pi * coordinates)
    data = np.broadcast_to(profile[:, None, None], (8, 8, 8)).copy()
    cell = np.eye(3) if density_cell is None else np.asarray(density_cell, dtype=float)
    charge = Charge(
        data,
        cell,
        atom_positions=np.asarray([direct_position], dtype=float) @ cell,
        atom_types=[1],
        atom_charges=[1.0],
    )
    charge.save_cube(str(output / "SPIN1_CHG.cube"))
    return job


def test_electrostatic_cross_term_from_three_jobs(tmp_path):
    full_job = _write_ewald_job(tmp_path / "full", -10.0)
    fragment_a_job = _write_density_fragment_job(
        tmp_path / "fragment_a",
        direct_position=(0.0, 0.0, 0.0),
        amplitude=0.1,
    )
    fragment_b_job = _write_density_fragment_job(
        tmp_path / "fragment_b",
        direct_position=(0.5, 0.5, 0.5),
        amplitude=0.2,
    )
    _write_ewald_job(fragment_a_job, -6.0)
    _write_ewald_job(fragment_b_job, -3.0)

    terms = electrostatic_cross_term_from_jobs(
        full_job,
        fragment_a_job,
        fragment_b_job,
    )

    density_a = read_job_total_density(fragment_a_job)
    density_b = read_job_total_density(fragment_b_job)
    nuclei_a = fragment_nuclei(fragment_a_job)
    nuclei_b = fragment_nuclei(fragment_b_job)
    expected_ee = electron_electron_cross_term(density_a, density_b)
    expected_ab = electron_nucleus_cross_term(density_a, nuclei_b.positions, nuclei_b.charges)
    expected_ba = electron_nucleus_cross_term(density_b, nuclei_a.positions, nuclei_a.charges)

    assert terms.electron_electron == pytest.approx(expected_ee)
    assert terms.electron_nucleus_ab == pytest.approx(expected_ab)
    assert terms.electron_nucleus_ba == pytest.approx(expected_ba)
    assert terms.ion_ion == pytest.approx(-1.0)
    assert terms.total == pytest.approx(expected_ee + expected_ab + expected_ba - 1.0)
    assert terms.unit == "eV"


def test_electrostatic_cross_term_from_jobs_accepts_rounding_in_cell(tmp_path):
    full_job = _write_ewald_job(tmp_path / "full", -10.0)
    fragment_a_job = _write_density_fragment_job(
        tmp_path / "fragment_a",
        direct_position=(0.0, 0.0, 0.0),
        amplitude=0.1,
        density_cell=(1.0 + 6e-7) * np.eye(3),
    )
    fragment_b_job = _write_density_fragment_job(
        tmp_path / "fragment_b",
        direct_position=(0.5, 0.5, 0.5),
        amplitude=0.2,
    )
    _write_ewald_job(fragment_a_job, -6.0)
    _write_ewald_job(fragment_b_job, -3.0)

    terms = electrostatic_cross_term_from_jobs(full_job, fragment_a_job, fragment_b_job)

    assert np.isfinite(terms.total)


def test_electrostatic_cross_term_from_jobs_rejects_density_cell_mismatch(tmp_path):
    full_job = _write_ewald_job(tmp_path / "full", -10.0)
    fragment_a_job = _write_density_fragment_job(
        tmp_path / "fragment_a",
        direct_position=(0.0, 0.0, 0.0),
        amplitude=0.1,
        density_cell=2.0 * np.eye(3),
    )
    fragment_b_job = _write_density_fragment_job(
        tmp_path / "fragment_b",
        direct_position=(0.5, 0.5, 0.5),
        amplitude=0.2,
    )
    _write_ewald_job(fragment_a_job, -6.0)
    _write_ewald_job(fragment_b_job, -3.0)

    with pytest.raises(ValueError, match="nuclear cell"):
        electrostatic_cross_term_from_jobs(full_job, fragment_a_job, fragment_b_job)


def test_read_abacus_harris_energy(tmp_path):
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\ngamma_only 1\nsuffix ABACUS\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(
        "E_KohnSham = -10.000000 eV\nE_Harris = -9.750000 eV\n",
        encoding="utf-8",
    )

    assert read_abacus_harris_energy(job) == pytest.approx(-9.75)


def test_analyze_kpoint_peda_assembles_non_nocv_terms(monkeypatch):
    import abacustools.data.eda as eda_module
    import abacustools.data.lcao as lcao_module

    states = [
        SimpleNamespace(
            index=index,
            direct=direct,
            weight=0.5,
            channels={1: SimpleNamespace(), 2: SimpleNamespace()},
            data_files=(f"wf{kpoint}",),
        )
        for index, direct, kpoint in (
            (1, (0.0, 0.0, 0.0), 1),
            (2, (0.5, 0.0, 0.0), 2),
        )
    ]
    monkeypatch.setattr(eda_module, "_require_multik_peda_job", lambda *_args: None)
    monkeypatch.setattr(lcao_module, "build_peda_kpoint_states", lambda *_args, **_kwargs: states)
    monkeypatch.setattr(
        eda_module,
        "read_kpoint_fixed_density_result",
        lambda *_args, **_kwargs: SimpleNamespace(data_files=("hk1", "hk2")),
    )
    monkeypatch.setattr(
        eda_module,
        "_p0_energy_from_kpoint_fixed_density",
        lambda _states, _result: -18.5,
    )

    def fake_result(job, _keys, version=None):
        return {"energy": {
            "full": -20.0,
            "fa": -8.0,
            "fb": -5.0,
            "ra": -7.5,
            "rb": -5.5,
        }[job]}

    monkeypatch.setattr(eda_module, "get_result_from_job", fake_result)
    monkeypatch.setattr(
        eda_module,
        "electrostatic_cross_term_from_jobs",
        lambda *_args, **_kwargs: SimpleNamespace(total=-2.0),
    )

    result = analyze_kpoint_peda(
        "full",
        "fa",
        "fb",
        "p0",
        [0],
        [1],
        reference_a_job="ra",
        reference_b_job="rb",
    )

    assert result.kpoint_count == 2
    assert result.nspin == 2
    assert result.e0 == pytest.approx(-18.5)
    assert result.terms.preparation == pytest.approx(0.0)
    assert result.terms.electrostatic == pytest.approx(-2.0)
    assert result.terms.pauli == pytest.approx(-3.5)
    assert result.terms.orbital == pytest.approx(-1.5)
    assert result.terms.interaction == pytest.approx(-7.0)
    assert result.terms.bond == pytest.approx(-7.0)
    assert set(result.data_files) == {"wf1", "wf2", "hk1", "hk2"}


def test_analyze_kpoint_peda_rejects_symmetry_one(tmp_path):
    job = tmp_path / "full"
    job.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        "gamma_only 0\n"
        "symmetry 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="symmetry=-1 or 0"):
        analyze_kpoint_peda(job, "fa", "fb", "p0", [0], [1])


def _mock_gamma_peda_jobs(monkeypatch, *, fock_expectation_difference):
    import abacustools.data.eda as eda_module
    import abacustools.data.lcao as lcao_module

    fake_state = SimpleNamespace(
        direct=(0.0, 0.0, 0.0),
        channels={
            1: SimpleNamespace(
                delta_p_lambda=np.diag([-1.0, 1.0]),
                basis=SimpleNamespace(combined=np.eye(2)),
            )
        },
    )
    monkeypatch.setattr(
        lcao_module,
        "build_peda_kpoint_states",
        lambda *args, **kwargs: [fake_state],
    )
    monkeypatch.setattr(
        eda_module,
        "read_gamma_fixed_density_result",
        lambda job, **kwargs: (
            SimpleNamespace(
                harris_energy_ev=-9.0,
                band_energy_ev=None,
                electronic_energy_ev=-9.0,
                calculation="scf",
                hamiltonians={},
            )
            if str(job).endswith("p0")
            else SimpleNamespace(
                hamiltonians={
                    1: np.diag(
                        [-0.5 * fock_expectation_difference, 0.5 * fock_expectation_difference]
                    )
                }
            )
        ),
    )
    energy_map = {"full": -10.0, "fragment_a": -4.0, "fragment_b": -4.0}

    def fake_result(job, *args, **kwargs):
        name = Path(job).name
        return {"energy": energy_map[name]}

    monkeypatch.setattr(eda_module, "get_result_from_job", fake_result)
    monkeypatch.setattr(
        eda_module,
        "electrostatic_cross_term_from_jobs",
        lambda *args, **kwargs: SimpleNamespace(total=-0.5),
    )


def test_analyze_gamma_peda_nocv_validates_orbital_sum(monkeypatch):
    _mock_gamma_peda_jobs(
        monkeypatch,
        fock_expectation_difference=-1.0 / RY_TO_EV,
    )

    result = analyze_gamma_peda_nocv(
        "/tmp/full",
        "/tmp/fragment_a",
        "/tmp/fragment_b",
        "/tmp/p0",
        "/tmp/p_ts",
        [0],
        [1],
    )

    assert result.terms.orbital == pytest.approx(-1.0)
    assert result.nocv_total_ev == pytest.approx(-1.0)
    assert result.orbital_error_ev == pytest.approx(0.0, abs=1e-12)


def test_analyze_gamma_peda_nocv_rejects_orbital_mismatch(monkeypatch):
    _mock_gamma_peda_jobs(
        monkeypatch,
        fock_expectation_difference=-0.2,
    )

    with pytest.raises(ValueError, match="NOCV channel energies"):
        analyze_gamma_peda_nocv(
            "/tmp/full",
            "/tmp/fragment_a",
            "/tmp/fragment_b",
            "/tmp/p0",
            "/tmp/p_ts",
            [0],
            [1],
        )


def test_multik_fixed_density_nscf_energy_uses_k_and_spin_weights():
    import abacustools.data.eda as eda_module

    states = [
        SimpleNamespace(
            index=index,
            direct=direct,
            weight=0.5,
            channels={
                spin: SimpleNamespace(
                    p0_lambda=np.diag([1.0, 0.0]),
                    basis=SimpleNamespace(combined=np.eye(2)),
                )
                for spin in (1, 2)
            },
        )
        for index, direct in ((1, (0.0, 0.0, 0.0)), (2, (0.5, 0.0, 0.0)))
    ]
    p0_result = SimpleNamespace(
        calculation="nscf",
        electronic_energy_ev=-10.0,
        band_energy_ev=-8.0,
        harris_energy_ev=-9.0,
        kpoints=tuple(
            SimpleNamespace(
                index=index,
                direct=direct,
                weight=0.5,
                hamiltonians={1: np.diag([1.0, 2.0]), 2: np.diag([1.0, 2.0])},
            )
            for index, direct in ((1, (0.0, 0.0, 0.0)), (2, (0.5, 0.0, 0.0)))
        ),
    )

    energy = eda_module._p0_energy_from_kpoint_fixed_density(states, p0_result)

    assert energy == pytest.approx(-2.0 + 2.0 * RY_TO_EV)


def test_fixed_density_nscf_energy_replaces_diagonalized_band_term():
    import abacustools.data.eda as eda_module

    state = SimpleNamespace(
        channels={
            1: SimpleNamespace(
                p0_lambda=np.eye(2),
                basis=SimpleNamespace(combined=np.eye(2)),
            )
        }
    )
    p0_result = SimpleNamespace(
        calculation="nscf",
        electronic_energy_ev=-10.0,
        band_energy_ev=-8.0,
        harris_energy_ev=-9.0,
        hamiltonians={1: np.eye(2)},
    )

    energy = eda_module._p0_energy_from_fixed_density(state, p0_result)

    assert energy == pytest.approx(-2.0 + 2.0 * RY_TO_EV)
