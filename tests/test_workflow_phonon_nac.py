"""Tests for the non-analytical correction of the phonon workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.workflow.phonon import (
    _born_charges,
    _dielectric_tensor,
    postprocess,
)

from test_workflow_phonon_postprocess import (
    _build_synthetic_phonon_job,
    _postprocess_args,
    _read_report,
)


#: A zincblende-like cell whose two sublattices carry opposite effective
#: charges, which is what makes a longitudinal optical mode split off.
_DIELECTRIC = "[2.34, 0, 0, 0, 2.34, 0, 0, 0, 2.34]"
_BORN = "[[[1.1,0,0],[0,1.1,0],[0,0,1.1]],[[-1.1,0,0],[0,-1.1,0],[0,0,-1.1]]]"


def _ionic_job(tmp_path: Path) -> Path:
    """Write a two-sublattice job whose sublattices have opposite charges."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(
        job,
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=5.64,
    )
    return job


def _primitive_volume(job: Path) -> float:
    """Return the volume of the reference cell as the calculation sees it."""
    from abacustools.io.stru import AbacusSTRU

    cell = np.asarray(AbacusSTRU.read(job / "STRU").cell, dtype=float)
    return float(abs(np.linalg.det(cell)))


def test_dielectric_tensor_accepts_scalar_vector_and_matrix() -> None:
    np.testing.assert_allclose(_dielectric_tensor(2.34), np.eye(3) * 2.34)
    np.testing.assert_allclose(_dielectric_tensor([2.34, 2.34, 2.34]), np.eye(3) * 2.34)
    matrix = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    np.testing.assert_allclose(_dielectric_tensor(matrix), matrix)


def test_dielectric_tensor_rejects_a_bad_shape() -> None:
    with pytest.raises(ValueError, match="dielectric tensor"):
        _dielectric_tensor([1.0, 2.0])


def test_born_charges_need_one_tensor_per_atom() -> None:
    charges = _born_charges(json.loads(_BORN), 2)

    assert charges.shape == (2, 3, 3)
    assert charges[0][0][0] == pytest.approx(1.1)
    assert charges[1][0][0] == pytest.approx(-1.1)
    with pytest.raises(ValueError, match="one 3x3 tensor per atom"):
        _born_charges(json.loads(_BORN), 3)


def test_born_charges_accept_a_lone_tensor_for_one_atom() -> None:
    charges = _born_charges(json.loads("[[1.0,0,0],[0,1.0,0],[0,0,1.0]]"), 1)

    assert charges.shape == (1, 3, 3)


def test_nac_needs_both_inputs(tmp_path: Path) -> None:
    job = _ionic_job(tmp_path)

    # Each half of the correction is reported as missing when it is the one
    # that was left out.
    with pytest.raises(ValueError, match="needs --dielectric"):
        postprocess(_postprocess_args(job, born=json.loads(_BORN)))
    with pytest.raises(ValueError, match="Born effective charges"):
        postprocess(_postprocess_args(job, dielectric=json.loads(_DIELECTRIC)))


def test_postprocess_without_nac_keeps_the_optical_triplet_degenerate(
    tmp_path: Path,
) -> None:
    """Without the correction a zincblende optical mode stays a triplet."""
    job = _ionic_job(tmp_path)

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    modes = _read_report(job)["gamma_modes"]
    assert [mode["degeneracy"] for mode in modes] == [3] * 6
    assert len({round(mode["frequency_thz"], 6) for mode in modes[3:]}) == 1
    assert "non_analytical_correction" in _read_report(job)
    assert _read_report(job)["non_analytical_correction"] is None


def test_postprocess_splits_the_longitudinal_mode(tmp_path: Path) -> None:
    """The correction must raise one optical mode and leave two in place."""
    job = _ionic_job(tmp_path)

    assert (
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                born=json.loads(_BORN),
            )
        )
        == 0
    )

    report = _read_report(job)
    modes = report["gamma_modes"]
    characters = [mode["character"] for mode in modes]
    assert characters[:3] == ["acoustic"] * 3
    # Two transverse modes keep their frequency, one longitudinal mode is
    # raised above them.
    assert characters[3:] == ["TO", "TO", "LO"]
    transverse = report["gamma_modes"][3]["frequency_thz"]
    longitudinal = report["gamma_modes"][5]["frequency_thz"]
    assert longitudinal > transverse
    # The longitudinal mode is what sets the highest frequency of the spectrum.
    assert report["max_frequency_thz"] == pytest.approx(longitudinal, rel=1e-9)


def test_the_longitudinal_shift_grows_with_the_effective_charge(
    tmp_path: Path,
) -> None:
    """A larger effective charge must raise the longitudinal mode further."""
    shifts = []
    for index, charge in enumerate((0.5, 1.5)):
        job = tmp_path / f"job{index}"
        _build_synthetic_phonon_job(
            job,
            symbols=["Na", "Cl"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=5.64,
        )
        born = json.dumps(
            [[[charge, 0, 0], [0, charge, 0], [0, 0, charge]],
             [[-charge, 0, 0], [0, -charge, 0], [0, 0, -charge]]]
        )
        assert (
            postprocess(
                _postprocess_args(
                    job,
                    mesh=[2, 2, 2],
                    dielectric=json.loads(_DIELECTRIC),
                    born=json.loads(born),
                )
            )
            == 0
        )
        report = _read_report(job)
        transverse = report["gamma_modes"][3]["frequency_thz"]
        longitudinal = report["gamma_modes"][5]["frequency_thz"]
        shifts.append(longitudinal**2 - transverse**2)

    # The splitting is quadratic in the effective charge, so the ratio of the
    # two shifts is the ratio of the squared charges, 9.
    assert shifts[1] / shifts[0] == pytest.approx(9.0, rel=1e-6)


def test_the_longitudinal_shift_scales_with_the_inverse_dielectric(
    tmp_path: Path,
) -> None:
    """Doubling the screening must halve the squared frequency shift."""
    shifts = []
    for index, epsilon in enumerate((2.0, 4.0)):
        job = tmp_path / f"job{index}"
        _build_synthetic_phonon_job(
            job,
            symbols=["Na", "Cl"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=5.64,
        )
        assert (
            postprocess(
                _postprocess_args(
                    job,
                    mesh=[2, 2, 2],
                    dielectric=json.loads(f"[{epsilon},0,0,0,{epsilon},0,0,0,{epsilon}]"),
                    born=json.loads(_BORN),
                )
            )
            == 0
        )
        report = _read_report(job)
        transverse = report["gamma_modes"][3]["frequency_thz"]
        longitudinal = report["gamma_modes"][5]["frequency_thz"]
        shifts.append(longitudinal**2 - transverse**2)

    assert shifts[0] / shifts[1] == pytest.approx(2.0, rel=1e-6)


def test_the_reported_shift_matches_the_lattice_dynamics_expression(
    tmp_path: Path,
) -> None:
    """The splitting must obey the standard long wavelength expression.

    The volume of the reference cell is read back from the calculation rather
    than assumed, because the cell that ABACUS sees follows the convention of
    the test fixture and need not be the number written into it.
    """
    job = _ionic_job(tmp_path)
    assert (
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                born=json.loads(_BORN),
            )
        )
        == 0
    )
    report = _read_report(job)
    transverse = report["gamma_modes"][3]["frequency_thz"]
    longitudinal = report["gamma_modes"][5]["frequency_thz"]

    # omega_LO^2 - omega_TO^2 = 4 pi e^2 Z*^2 / (Omega eps mu) in atomic units
    # of eV, Angstrom and amu, converted to THz by phonopy's constant.
    unit = 15.633302300230191
    mass_na, mass_cl = 22.98976928, 35.453
    reduced = mass_na * mass_cl / (mass_na + mass_cl)
    volume = _primitive_volume(job)
    expected = (
        4.0 * math.pi * 14.399645 * 1.1**2 / (volume * 2.34 * reduced) * unit**2
    )

    assert longitudinal**2 - transverse**2 == pytest.approx(expected, rel=1e-6)


def test_nac_direction_is_reported_in_cartesian_coordinates(tmp_path: Path) -> None:
    """The direction the limit was taken along must appear in the report."""
    job = _ionic_job(tmp_path)

    assert (
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                born=json.loads(_BORN),
            )
        )
        == 0
    )

    correction = _read_report(job)["non_analytical_correction"]
    assert correction["born"][0][0][0] == pytest.approx(1.1)
    assert correction["dielectric"][0][0] == pytest.approx(2.34)
    assert len(correction["direction_cartesian"]) == 3
    # The default direction is the first lattice vector.
    assert np.linalg.norm(correction["direction_cartesian"]) > 0.0


def test_nac_direction_can_be_chosen(tmp_path: Path) -> None:
    """A cubic crystal gives the same longitudinal mode along any axis."""
    job = _ionic_job(tmp_path)

    assert (
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                born=json.loads(_BORN),
                nac_direction=[1.0, 0.0, 0.0],
            )
        )
        == 0
    )

    longitudinal = _read_report(job)["gamma_modes"][5]["frequency_thz"]
    assert longitudinal > 0.0


def test_nac_rejects_a_zero_direction(tmp_path: Path) -> None:
    job = _ionic_job(tmp_path)

    with pytest.raises(ValueError, match="must not be zero"):
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                born=json.loads(_BORN),
                nac_direction=[0.0, 0.0, 0.0],
            )
        )


def test_nac_changes_the_thermal_properties(tmp_path: Path) -> None:
    """The mesh applies the correction, so the summed quantities move."""
    plain = tmp_path / "plain"
    corrected = tmp_path / "corrected"
    _ionic_job(tmp_path)
    for job in (plain, corrected):
        _build_synthetic_phonon_job(
            job,
            symbols=["Na", "Cl"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=5.64,
        )
    assert postprocess(_postprocess_args(plain, mesh=[4, 4, 4])) == 0
    assert (
        postprocess(
            _postprocess_args(
                corrected,
                mesh=[4, 4, 4],
                dielectric=json.loads(_DIELECTRIC),
                born=json.loads(_BORN),
            )
        )
        == 0
    )

    plain_report = _read_report(plain)
    corrected_report = _read_report(corrected)

    assert corrected_report["heat_capacity"] != pytest.approx(
        plain_report["heat_capacity"], rel=1e-9
    )
    assert corrected_report["max_frequency_thz"] > plain_report["max_frequency_thz"]


def _bec_results(path: Path, *, charge: float = 1.1, drop_row: bool = False) -> Path:
    """Write a minimal ``bec_results.json`` of the BEC workflow."""
    tensor = [[charge, 0.0, 0.0], [0.0, charge, 0.0], [0.0, 0.0, charge]]
    if drop_row:
        tensor = [[charge, 0.0, 0.0], None, [0.0, 0.0, charge]]
    payload = {
        "workflow": "bec",
        "atoms": [
            {"index": 1, "label": "Na", "bec_tensor": tensor},
            {
                "index": 2,
                "label": "Cl",
                "bec_tensor": [[-v for v in row] for row in tensor if row is not None]
                if not drop_row
                else [[-charge, 0.0, 0.0], None, [0.0, 0.0, -charge]],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_born_charges_can_be_read_from_the_bec_workflow(tmp_path: Path) -> None:
    """The BEC workflow output must be usable without hand transcription."""
    job = _ionic_job(tmp_path)
    results = _bec_results(tmp_path / "bec_results.json")

    assert (
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                bec_results=results,
            )
        )
        == 0
    )

    report = _read_report(job)
    assert report["non_analytical_correction"]["born"][0][0][0] == pytest.approx(1.1)
    assert [mode["character"] for mode in report["gamma_modes"][3:]] == [
        "TO",
        "TO",
        "LO",
    ]


def test_bec_results_give_the_same_spectrum_as_an_explicit_tensor(
    tmp_path: Path,
) -> None:
    """Reading the file must equal passing the same numbers by hand."""
    explicit = tmp_path / "explicit"
    from_file = tmp_path / "fromfile"
    for job in (explicit, from_file):
        _build_synthetic_phonon_job(
            job,
            symbols=["Na", "Cl"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=5.64,
        )
    results = _bec_results(tmp_path / "bec.json")

    assert (
        postprocess(
            _postprocess_args(
                explicit,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                born=json.loads(_BORN),
            )
        )
        == 0
    )
    assert (
        postprocess(
            _postprocess_args(
                from_file,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                bec_results=results,
            )
        )
        == 0
    )

    assert _read_report(from_file)["gamma_modes"] == _read_report(explicit)["gamma_modes"]


def test_bec_results_must_hold_every_direction(tmp_path: Path) -> None:
    job = _ionic_job(tmp_path)
    results = _bec_results(tmp_path / "bec.json", drop_row=True)

    with pytest.raises(ValueError, match="missing the displacement"):
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                bec_results=results,
            )
        )


def test_bec_results_must_hold_every_atom(tmp_path: Path) -> None:
    job = _ionic_job(tmp_path)
    results = tmp_path / "bec.json"
    results.write_text(
        json.dumps(
            {
                "workflow": "bec",
                "atoms": [
                    {
                        "index": 1,
                        "label": "Na",
                        "bec_tensor": [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="every atom"):
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                bec_results=results,
            )
        )


def test_bec_results_file_must_exist(tmp_path: Path) -> None:
    job = _ionic_job(tmp_path)

    with pytest.raises(FileNotFoundError, match="could not find the BEC results"):
        postprocess(
            _postprocess_args(
                job,
                mesh=[2, 2, 2],
                dielectric=json.loads(_DIELECTRIC),
                bec_results=tmp_path / "absent.json",
            )
        )
