"""Tests for the electronic dielectric tensor data layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacustools.data.dielectric import (
    FREQUENCY_TENSOR_FILE,
    STATIC_TENSOR_FILE,
    copy_matrices,
    dielectric_job_parameters,
    dielectric_summary,
    fermi_energy,
    lattice_block,
    matrix_files,
    omega_window_warning,
    occupied_band_count,
    prepare_matrix_output,
    pyatb_input_text,
    read_static_dielectric,
    read_zero_frequency_dielectric,
)
from abacustools.integrations.pyatb import pyatb_available
from abacustools.io.stru import AbacusATOM, AbacusSTRU


def _structure(cell: float = 5.64) -> AbacusSTRU:
    """Return a two-atom cubic cell."""
    return AbacusSTRU(
        cell=[
            [0.0, cell / 2, cell / 2],
            [cell / 2, 0.0, cell / 2],
            [cell / 2, cell / 2, 0.0],
        ],
        atoms=[
            AbacusATOM(label="Na", element="Na", coord=(0.0, 0.0, 0.0)),
            AbacusATOM(label="Cl", element="Cl", coord=(cell / 2, cell / 2, cell / 2)),
        ],
        metadata={"atom_type": "cartesian"},
    )


def _job_with_matrices(job: Path) -> Path:
    """Write a job holding the three matrices pyatb reads."""
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "data-HR-sparse_SPIN0.csr",
        "data-SR-sparse_SPIN0.csr",
        "data-rR-sparse.csr",
    ):
        (output / name).write_text("Matrix Dimension of H(R): 1\n", encoding="utf-8")
    return job


def test_prepare_matrix_output_switches_the_matrices_on() -> None:
    prepared = prepare_matrix_output(
        {"basis_type": "lcao", "gamma_only": 0, "calculation": "scf"}
    )

    assert prepared["out_mat_hs2"] == 1
    assert prepared["out_mat_r"] == 1
    # The sum runs over the full zone, so the k points must not be reduced.
    assert prepared["symmetry"] == 0
    assert prepared["calculation"] == "scf"


def test_prepare_matrix_output_rejects_a_plane_wave_basis() -> None:
    with pytest.raises(ValueError, match="basis_type"):
        prepare_matrix_output({"basis_type": "pw"})


def test_prepare_matrix_output_rejects_gamma_only() -> None:
    with pytest.raises(ValueError, match="gamma point only"):
        prepare_matrix_output({"basis_type": "lcao", "gamma_only": 1})
    with pytest.raises(ValueError, match="gamma point only"):
        prepare_matrix_output({"basis_type": "lcao", "gamma_only": True})


def test_matrix_files_are_located(tmp_path: Path) -> None:
    job = _job_with_matrices(tmp_path / "job")

    found = matrix_files(job)

    assert set(found) == {"HR", "SR", "rR"}
    assert found["rR"].name == "data-rR-sparse.csr"


def test_matrix_files_report_a_missing_matrix(tmp_path: Path) -> None:
    job = _job_with_matrices(tmp_path / "job")
    (job / "OUT.ABACUS" / "data-rR-sparse.csr").unlink()

    with pytest.raises(FileNotFoundError, match="rR matrix"):
        matrix_files(job)


def test_matrix_files_report_a_missing_output_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OUT"):
        matrix_files(tmp_path / "job")


def test_copy_matrices_moves_them_into_the_working_directory(tmp_path: Path) -> None:
    job = _job_with_matrices(tmp_path / "job")
    work = tmp_path / "work"

    names = copy_matrices(job, work)

    assert names["HR"] == "data-HR-sparse_SPIN0.csr"
    assert (work / names["rR"]).is_file()


def test_lattice_block_is_written_in_bohr() -> None:
    text = lattice_block(_structure())

    assert "lattice_constant_unit   Bohr" in text
    assert "lattice_vector" in text
    assert text.count("\n") > 6


def test_pyatb_input_holds_the_expected_blocks() -> None:
    text = pyatb_input_text(
        _structure(),
        nspin=1,
        fermi_energy=6.389728305291531,
        grid=(50, 50, 50),
        occ_band=8,
    )

    assert "INPUT_PARAMETERS" in text
    assert "OPTICAL_CONDUCTIVITY" in text
    assert "LATTICE" in text
    assert "package                 ABACUS" in text
    assert "HR_route                data-HR-sparse_SPIN0.csr" in text
    assert "rR_route                data-rR-sparse.csr" in text
    assert "HR_unit                 Ry" in text
    assert "rR_unit                 Bohr" in text
    assert "occ_band       8" in text
    assert "grid           50 50 50" in text
    assert f"fermi_energy            {6.389728305291531:.10f}" in text
    # The static only switch belongs to the newer builds and is off by default.
    assert "static_dielectric_only" not in text


def test_pyatb_input_can_ask_for_the_static_limit() -> None:
    text = pyatb_input_text(
        _structure(), nspin=1, fermi_energy=1.0, occ_band=4, static_only=True
    )

    assert "static_dielectric_only     1" in text


def test_pyatb_input_window_reaches_across_the_band_gap() -> None:
    """The window has to reach past the gap, and start at zero.

    A window that sits inside the gap holds no transition at all, which is
    what leaves an empty spectrum, and the zero frequency row it starts with
    is the electronic dielectric tensor itself.
    """
    text = pyatb_input_text(_structure(), nspin=1, fermi_energy=1.0, occ_band=4)

    line = next(line for line in text.splitlines() if line.strip().startswith("omega"))
    start, end = (float(value) for value in line.split()[1:3])
    assert start == pytest.approx(0.0)
    assert end >= 10.0


def test_pyatb_input_rejects_a_collinear_spin_polarised_job() -> None:
    """pyatb has no optical conductivity for two spin channels."""
    with pytest.raises(ValueError, match="nspin 1 or 4"):
        pyatb_input_text(_structure(), nspin=2, fermi_energy=1.0, occ_band=4)


def test_occupied_band_count_is_read_from_the_log(tmp_path: Path) -> None:
    log = tmp_path / "running_scf.log"
    log.write_text(
        "AUTOSET number of electrons:  = 16\n"
        "                     occupied bands = 8\n"
        "                             NBANDS = 18\n",
        encoding="utf-8",
    )

    assert occupied_band_count(log) == 8


def test_occupied_band_count_is_missing_without_the_line(tmp_path: Path) -> None:
    log = tmp_path / "running_scf.log"
    log.write_text("Final Etot = -1.0 eV\n", encoding="utf-8")

    assert occupied_band_count(log) is None


def test_fermi_energy_is_read_from_the_log(tmp_path: Path) -> None:
    log = tmp_path / "running_scf.log"
    log.write_text(
        " E_Fermi        0.2101224127         2.8588620907\n"
        " EFERMI = 2.8588620907 eV\n",
        encoding="utf-8",
    )

    assert fermi_energy(log) == pytest.approx(2.8588620907)


def test_fermi_energy_is_missing_without_the_line(tmp_path: Path) -> None:
    log = tmp_path / "running_scf.log"
    log.write_text("Final Etot = -1.0 eV\n", encoding="utf-8")

    assert fermi_energy(log) is None


def _job_with_log(job: Path, *, nspin: int = 1, log: str | None = None) -> Path:
    """Write a job whose INPUT and running log describe a finished SCF."""
    (job / "OUT.ABACUS").mkdir(parents=True, exist_ok=True)
    (job / "INPUT").write_text(
        f"INPUT_PARAMETERS\ncalculation scf\nbasis_type lcao\nnspin {nspin}\n",
        encoding="utf-8",
    )
    (job / "OUT.ABACUS" / "running_scf.log").write_text(
        log
        if log is not None
        else "AUTOSET number of electrons:  = 16\n"
        "        occupied bands = 8\n"
        " EFERMI = 2.8588620907 eV\n",
        encoding="utf-8",
    )
    return job


def test_dielectric_job_parameters_read_the_input_and_the_log(tmp_path: Path) -> None:
    job = _job_with_log(tmp_path / "job")

    parameters = dielectric_job_parameters(job)

    assert parameters["nspin"] == 1
    assert parameters["occ_band"] == 8
    assert parameters["fermi_energy"] == pytest.approx(2.8588620907)
    assert parameters["log"].endswith("running_scf.log")


def test_dielectric_job_parameters_reject_a_collinear_spin_job(tmp_path: Path) -> None:
    job = _job_with_log(tmp_path / "job", nspin=2)

    with pytest.raises(ValueError, match="nspin 1 or 4"):
        dielectric_job_parameters(job)


def test_dielectric_job_parameters_need_the_occupied_band_count(tmp_path: Path) -> None:
    job = _job_with_log(tmp_path / "job", log=" EFERMI = 2.8588620907 eV\n")

    with pytest.raises(ValueError, match="occupied band count"):
        dielectric_job_parameters(job)


def test_dielectric_job_parameters_need_the_fermi_energy(tmp_path: Path) -> None:
    job = _job_with_log(tmp_path / "job", log="occupied bands = 8\n")

    with pytest.raises(ValueError, match="EFERMI"):
        dielectric_job_parameters(job)


def test_dielectric_job_parameters_need_a_running_log(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "OUT.ABACUS").mkdir(parents=True)
    (job / "INPUT").write_text("INPUT_PARAMETERS\nbasis_type lcao\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="running log"):
        dielectric_job_parameters(job)


def test_omega_window_warning_flags_a_short_or_shifted_window() -> None:
    assert omega_window_warning((0.0, 80.0)) is None
    assert "below the 40 eV" in omega_window_warning((0.0, 20.0))
    assert "starts at 0.5 eV" in omega_window_warning((0.5, 80.0))


def test_read_static_dielectric(tmp_path: Path) -> None:
    path = tmp_path / STATIC_TENSOR_FILE
    path.write_text(
        "#             xx             xy  ...\n"
        "   2.340000e+00   0.000000e+00   0.000000e+00"
        "   0.000000e+00   2.340000e+00   0.000000e+00"
        "   0.000000e+00   0.000000e+00   2.340000e+00\n",
        encoding="utf-8",
    )

    tensor = read_static_dielectric(path)

    np.testing.assert_allclose(tensor, np.eye(3) * 2.34)


def test_read_static_dielectric_rejects_a_short_row(tmp_path: Path) -> None:
    path = tmp_path / STATIC_TENSOR_FILE
    path.write_text("1.0 2.0 3.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nine components"):
        read_static_dielectric(path)


def test_read_zero_frequency_dielectric_takes_the_first_row(tmp_path: Path) -> None:
    """The spectrum starts at zero, so its first row is the static limit."""
    path = tmp_path / FREQUENCY_TENSOR_FILE
    path.write_text(
        "#  omega(eV)   xx   xy\n"
        "   0.00000   2.340000e+00   0.000000e+00   0.000000e+00"
        "   0.000000e+00   2.340000e+00   0.000000e+00"
        "   0.000000e+00   0.000000e+00   2.340000e+00\n"
        "   0.10000   2.400000e+00   0.000000e+00   0.000000e+00"
        "   0.000000e+00   2.400000e+00   0.000000e+00"
        "   0.000000e+00   0.000000e+00   2.400000e+00\n",
        encoding="utf-8",
    )

    tensor = read_zero_frequency_dielectric(path)

    np.testing.assert_allclose(tensor, np.eye(3) * 2.34)


def test_read_zero_frequency_dielectric_needs_a_row(tmp_path: Path) -> None:
    path = tmp_path / FREQUENCY_TENSOR_FILE
    path.write_text("# only a header\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no dielectric row"):
        read_zero_frequency_dielectric(path)


def test_dielectric_summary_detects_an_isotropic_tensor() -> None:
    summary = dielectric_summary(np.eye(3) * 2.34)

    assert summary["isotropic"] is True
    assert summary["diagonal_mean"] == pytest.approx(2.34)
    assert summary["diagonal_spread"] == pytest.approx(0.0)
    assert summary["max_off_diagonal"] == pytest.approx(0.0)


def test_dielectric_summary_flags_an_anisotropic_tensor() -> None:
    summary = dielectric_summary(np.diag([2.0, 2.5, 3.0]))

    assert summary["isotropic"] is False
    assert summary["diagonal_mean"] == pytest.approx(2.5)
    assert summary["diagonal_spread"] == pytest.approx(1.0)


def test_pyatb_availability_is_reported_without_importing() -> None:
    assert isinstance(pyatb_available(), bool)
