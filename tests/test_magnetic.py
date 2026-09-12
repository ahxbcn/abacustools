"""Tests for collecting magnetic moments from ABACUS output."""

from __future__ import annotations

import json

from abacustools.data.abacus_result import (
    get_result_from_job,
    read_mulliken_magnetization,
    read_orbital_magnetization,
)
from abacustools.main import main


COLLINEAR_LOG = """\
                              ABACUS v3.10.1

          total magnetism (Bohr mag/cell) = 2.07604
       absolute magnetism (Bohr mag/cell) = 2.08357
          total magnetism (Bohr mag/cell) = 2.23663
       absolute magnetism (Bohr mag/cell) = 2.31331
 Total  Time  : 0 h 0 mins 3 secs
"""

NONCOLLINEAR_LOG = """\
                              ABACUS v3.10.1

total magnetism (Bohr mag/cell)\t1.52266e-16\t-2.9507e-16\t0.00466474
       absolute magnetism (Bohr mag/cell) = 0.0908443
"""

MULLIKEN_TEXT = """\
STEP: 0
CALCULATE THE MULLIkEN ANALYSIS FOR EACH ATOM
SUM OVER M+Zeta+L                                     9.1390              6.8610             16.0000              2.2780

Total Charge on atom:                 Fe             16.0000
Total Magnetism on atom:              Fe              2.2780

STEP: 1
CALCULATE THE MULLIkEN ANALYSIS FOR EACH ATOM
SUM OVER M+Zeta+L                                     9.1390              6.8610             16.0000              2.3000

Total Charge on atom:                 Fe             16.0000
Total Magnetism on atom:              Fe              2.3000
"""

NONCOLLINEAR_MULLIKEN_TEXT = """\
STEP: 0
CALCULATE THE MULLIkEN ANALYSIS FOR EACH ATOM

Total Magnetism on atom:              Fe   ( 0.1000, 0.2000, 3.0000 )
Total Magnetism on atom:              O    ( 0.0000, -0.1000, -0.5000 )
"""

ORBITAL_BLOCK_LOG = """\
                              ABACUS v3.10.1

-------------------------------------------------------------------------------------------
Orbital Charge Analysis      Charge         Mag(x)         Mag(y)         Mag(z)
-------------------------------------------------------------------------------------------
Fe1
                   s         1.0799        -0.0000         0.0000         0.0034
                   d         6.5446        -0.0039         0.0013         3.0690
                 Sum        13.6214        -0.0039         0.0013         3.0729
Fe2
                   s         1.0827         0.0000        -0.0000         0.0040
                   d         6.6497         0.0025        -0.0002         2.9733
                 Sum        13.7294         0.0025        -0.0002         2.9776
-------------------------------------------------------------------------------------------
                 Sum        13.6214        -0.0039         0.0013         3.0729
-------------------------------------------------------------------------------------------
Orbital Charge Analysis      Charge         Mag(x)         Mag(y)         Mag(z)
-------------------------------------------------------------------------------------------
Fe1
                 Sum        13.7000        -0.0040         0.0014         3.0800
Fe2
                 Sum        13.8000         0.0026        -0.0003         2.9800
-------------------------------------------------------------------------------------------
"""


def _job_with_log(tmp_path, log_text, *, mulliken=None):
    output = tmp_path / "OUT.ABACUS"
    output.mkdir()
    (tmp_path / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\ncalculation scf\n",
        encoding="utf-8",
    )
    (output / "running_scf.log").write_text(log_text, encoding="utf-8")
    if mulliken is not None:
        (output / "mulliken.txt").write_text(mulliken, encoding="utf-8")
    return output


def test_collects_collinear_moments_and_mulliken_atoms(tmp_path):
    _job_with_log(tmp_path, COLLINEAR_LOG, mulliken=MULLIKEN_TEXT)

    result = get_result_from_job(
        tmp_path,
        ["total_mag", "absolute_mag", "atom_mag_mulliken"],
        version="",
    )

    assert result["total_mag"] == 2.23663
    assert result["absolute_mag"] == 2.31331
    assert result["atom_mag_mulliken"] == [2.3]


def test_collects_noncollinear_moment_components(tmp_path):
    _job_with_log(tmp_path, NONCOLLINEAR_LOG)

    result = get_result_from_job(tmp_path, ["total_mag", "absolute_mag"], version="")

    assert result["total_mag"] == [1.52266e-16, -2.9507e-16, 0.00466474]
    assert result["absolute_mag"] == 0.0908443


def test_collects_last_orbital_projection_block(tmp_path):
    _job_with_log(tmp_path, ORBITAL_BLOCK_LOG)

    result = get_result_from_job(tmp_path, ["atom_orb_mag"], version="")

    assert result["atom_orb_mag"] == [
        [-0.0040, 0.0014, 3.0800],
        [0.0026, -0.0003, 2.9800],
    ]


def test_missing_mulliken_file_reports_none(tmp_path):
    _job_with_log(tmp_path, COLLINEAR_LOG)

    result = get_result_from_job(tmp_path, ["atom_mag_mulliken"], version="")

    assert result == {"atom_mag_mulliken": None}


def test_magnetic_moments_are_not_collected_by_default(tmp_path):
    _job_with_log(tmp_path, COLLINEAR_LOG, mulliken=MULLIKEN_TEXT)

    result = get_result_from_job(tmp_path, None, version="")

    assert "total_mag" not in result


def test_magnetic_moments_are_unavailable_without_a_log(tmp_path):
    (tmp_path / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\n",
        encoding="utf-8",
    )

    result = get_result_from_job(
        tmp_path,
        ["total_mag", "absolute_mag", "atom_orb_mag", "atom_mag_mulliken"],
        version="",
    )

    assert result == {
        "total_mag": None,
        "absolute_mag": None,
        "atom_orb_mag": None,
        "atom_mag_mulliken": None,
    }


def test_read_mulliken_magnetization_keeps_every_step(tmp_path):
    mulliken = tmp_path / "mulliken.txt"
    mulliken.write_text(MULLIKEN_TEXT, encoding="utf-8")

    steps = read_mulliken_magnetization(mulliken)

    assert steps == [[2.278], [2.3]]


def test_read_mulliken_magnetization_reads_noncollinear_components(tmp_path):
    mulliken = tmp_path / "mulliken.txt"
    mulliken.write_text(NONCOLLINEAR_MULLIKEN_TEXT, encoding="utf-8")

    steps = read_mulliken_magnetization(mulliken)

    assert steps == [
        [[0.1, 0.2, 3.0], [0.0, -0.1, -0.5]],
    ]


def test_read_orbital_magnetization_returns_none_without_a_block():
    assert read_orbital_magnetization(["nothing to see here"]) is None


def test_result_command_prints_magnetic_moments_as_json(tmp_path, capsys):
    _job_with_log(tmp_path, COLLINEAR_LOG, mulliken=MULLIKEN_TEXT)

    assert main(
        [
            "postprocess",
            "result",
            "-j",
            str(tmp_path),
            "-p",
            "total_mag",
            "atom_mag_mulliken",
            "--json",
        ]
    ) == 0

    # The entry point prints its banner before the JSON payload.
    output_text = capsys.readouterr().out
    output = json.loads(output_text[output_text.index("{") :])
    job = output[str(tmp_path)]
    assert job["total_mag"] == 2.23663
    assert job["atom_mag_mulliken"] == [2.3]
