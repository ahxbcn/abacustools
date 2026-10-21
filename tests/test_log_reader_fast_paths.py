"""Tests for the single-pass and metric-gated log readers."""

from __future__ import annotations

from pathlib import Path

import pytest

from abacustools.data import abacus_result
from abacustools.data.abacus_result import (
    _gate,
    _normalize_pattern,
    collect_relax_results,
    collect_scf_results,
    grouped_params,
    read_normal_end,
    read_relaxation_history,
)


SCF_LOG = """\
 LCAO ALGORITHM --------------- ION=   1  ELEC=   1--------------------------------
 Density error is 0.0491537746297
 E_KohnSham     -236.6840809437      -3220.2521267278

 LCAO ALGORITHM --------------- ION=   1  ELEC=   2--------------------------------
 Density error is 0.0187015006713
 E_KohnSham     -236.6965515963      -3220.4217986602
 E_Fermi        0.7913484839         10.7668484843
 charge density convergence is achieved
 final etot is -3220.4217986602 eV
 TOTAL-FORCE (eV/Angstrom)
------------------------------------------------------------------------------------------
                        Ga1         0.1000000000         0.2000000000         0.3000000000
------------------------------------------------------------------------------------------
 TOTAL-STRESS (KBAR)
------------------------------------------------------------------------------------------
         1.0000000000         2.0000000000         3.0000000000
         2.0000000000         4.0000000000         5.0000000000
         3.0000000000         5.0000000000         6.0000000000
------------------------------------------------------------------------------------------
 Total  Time  : 0 h 0 mins 3 secs
"""


RELAX_LOG = """\
 STEP OF RELAXATION : 1
 TOTAL-FORCE (eV/Angstrom)
------------------------------------------------------------------------------------------
                        Ga1         0.1000000000         0.2000000000         0.3000000000
------------------------------------------------------------------------------------------
 TOTAL-STRESS (KBAR)
------------------------------------------------------------------------------------------
         1.0000000000         2.0000000000         3.0000000000
         2.0000000000         4.0000000000         5.0000000000
         3.0000000000         5.0000000000         6.0000000000
------------------------------------------------------------------------------------------
 Largest gradient in force is 0.300000 eV/A.
 Largest gradient in stress is 6.000000 kBar.
 Relaxation is converged!
"""


def _make_job(tmp_path: Path, log_text: str, calculation: str = "scf") -> Path:
    """Create a minimal ABACUS job directory with one running log."""
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(
        f"INPUT_PARAMETERS\ncalculation {calculation}\nsuffix ABACUS\n",
        encoding="utf-8",
    )
    output = job / "OUT.ABACUS"
    output.mkdir()
    (output / f"running_{calculation}.log").write_text(log_text, encoding="utf-8")
    return job


def test_scf_subset_matches_full_collection(tmp_path: Path) -> None:
    """A status update must get the same values as a full collection."""
    job = _make_job(tmp_path, SCF_LOG)
    full = collect_scf_results(str(job), list(grouped_params["scf_results"]), version="")
    subset = collect_scf_results(
        str(job),
        ["energy", "drho", "denergy", "scf_steps", "converged"],
        version="",
    )

    assert set(subset) == {"energy", "drho", "denergy", "scf_steps", "converged"}
    for name, value in subset.items():
        if isinstance(value, float):
            assert value == pytest.approx(full[name])
        else:
            assert value == full[name]
    # The unrequested tables are not parsed, but they still read correctly
    # when they are asked for.
    assert full["force"] == [[0.1, 0.2, 0.3]]
    assert full["stress"] == [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]
    assert full["efermi"] == pytest.approx(10.7668484843)


def test_relax_subset_matches_full_collection(tmp_path: Path) -> None:
    job = _make_job(tmp_path, RELAX_LOG, calculation="relax")
    full = collect_relax_results(str(job), list(grouped_params["relax_results"]), version="")
    subset = collect_relax_results(str(job), ["largest_force"], version="")

    assert subset == {"largest_force": pytest.approx(full["largest_force"])}
    assert full["largest_stress"] == pytest.approx(6.0)
    assert full["relax_steps"] == 1
    assert full["relax_converged"] is True


def test_normal_end_reads_only_the_tail(tmp_path: Path) -> None:
    """The footer check must not scan a large log from the front."""
    log = tmp_path / "running_scf.log"
    log.write_text("padding line\n" * 20000 + " Total  Time  : 1 s\n", encoding="utf-8")
    assert read_normal_end(log)

    early = tmp_path / "early.log"
    early.write_text(" Total  Time  : 1 s\n" + "padding line\n" * 20000, encoding="utf-8")
    assert not read_normal_end(early)


def test_normalize_pattern_keeps_case_sensitive_constructs() -> None:
    assert _normalize_pattern("eV/Angstrom") == ("ev/angstrom", False)
    # Every uppercase letter of the class also occurs lowercased.
    assert _normalize_pattern("[EeDd]") == ("[EeDd]", False)
    # An uppercase-only class still needs case-insensitive matching.
    assert _normalize_pattern("[A-Z]") == ("[A-Z]", True)
    # Escapes keep their meaning.
    assert _normalize_pattern(r"\S+eV") == (r"\S+ev", False)
    # Named groups are left alone.
    assert _normalize_pattern("(?P<Name>eV)") == ("(?P<Name>eV)", True)


def test_gate_matches_mixed_case_patterns() -> None:
    gate = _gate(
        ("total-force",),
        (r"largest\s+force\s+is\s+(" + abacus_result._FLOAT + r")\s*eV\s*/\s*Angstrom",),
    )
    assert gate.search(" total-force (ev/angstrom)")
    assert gate.search("largest force is 0.5 ev/angstrom")
    assert not gate.search("largest force is 0.5 ev/kbar")

    uppercase_class = _gate((), ("[A-Z]+",))
    assert uppercase_class.search("abc")


def test_gate_accepts_global_inline_flags() -> None:
    gate = _gate((), (r"(?i)total\s+force",))
    assert gate.search("total force")
    assert gate.search("TOTAL FORCE".lower())


def test_relax_markers_take_precedence_over_ion_markers(tmp_path: Path) -> None:
    log = tmp_path / "running_relax.log"
    log.write_text(
        " ION= 1 elec= 1\n"
        " final etot is -1.0 eV\n"
        " STEP OF RELAXATION : 5\n"
        " final etot is -2.0 eV\n",
        encoding="utf-8",
    )

    history = read_relaxation_history(log, version="")

    assert [item["step"] for item in history] == [5]
    assert history[0]["energy"] == pytest.approx(-2.0)


def test_block_parsers_stop_at_the_block_boundary(tmp_path: Path) -> None:
    log = tmp_path / "running_relax.log"
    log.write_text(RELAX_LOG, encoding="utf-8")

    history = read_relaxation_history(log, version="")

    assert history[0]["forces"] == [[0.1, 0.2, 0.3]]
    assert history[0]["stress"] == [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]
    assert history[0]["stress_component"] == "zz"
