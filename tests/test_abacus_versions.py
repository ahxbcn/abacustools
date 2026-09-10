"""Tests for reading results from different ABACUS output dialects."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from abacustools.core.config import CONFIG
from abacustools.data import versions
from abacustools.data.abacus_result import get_result_from_job, read_relaxation_history


DEVELOP_SCF_LOG = """\
                              ABACUS v3.11.0-beta9

 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 --> #ION MOVE#         1  #ELEC ITER#         1
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Electron density deviation 0.0038434
 ----------------------------------------------------------
      Energy           Rydberg                 eV
 ----------------------------------------------------------
  E_KohnSham     -7.8461437727        -106.7522626359
  E_Fermi        0.7913484839         10.7668484843
 ----------------------------------------------------------

 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 --> #ION MOVE#         1  #ELEC ITER#         2
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Electron density deviation 6.49158e-10
  E_KohnSham     -7.8461741776        -106.7526763159
  E_Fermi        0.7884168484         10.7269615368
 #TOTAL-FORCE (eV/Angstrom)#
 -------------------------------------------------------------------------
     Atoms              Force_x              Force_y              Force_z
 -------------------------------------------------------------------------
       Si1         0.1000000000         0.2000000000         0.3000000000
 -------------------------------------------------------------------------
 #TOTAL-STRESS (kbar)#
 ----------------------------------------------------------------
              Stress_x             Stress_y             Stress_z
 ----------------------------------------------------------------
          1.0000000000        2.0000000000         3.0000000000
          2.0000000000        4.0000000000         5.0000000000
          3.0000000000        5.0000000000         6.0000000000
 ----------------------------------------------------------------
 #SCF IS CONVERGED#
 #TOTAL ENERGY# -106.75267632 eV
 !FINAL_ETOT_IS -106.7526763159347 eV
 Total  Time  : 0 h 0 mins 3 secs
"""


DEVELOP_RELAX_LOG = """\
                              ABACUS v3.11.0-beta9

 ================================================================
 RELAX STEP: 1
 ================================================================
 #SCF IS CONVERGED#
 #TOTAL ENERGY# -106.62330387 eV
 Largest force is 0.100000 eV/Angstrom while threshold is 0.020000 eV/Angstrom
 Largest stress is 0.000000 kbar while threshold is 0.500000 kbar
 Relaxation is not converged yet!

 ================================================================
 RELAX STEP: 2
 ================================================================
 #SCF IS CONVERGED#
 #TOTAL ENERGY# -106.62520584347 eV
 Largest force is 0.005000 eV/Angstrom while threshold is 0.020000 eV/Angstrom
 Largest stress is 0.100000 kbar while threshold is 0.500000 kbar
 Relaxation is converged!
 !FINAL_ETOT_IS -106.6252316276676737 eV
 Total  Time  : 0 h 0 mins 9 secs
"""


LTS_RELAX_LOG = """\
STEP OF RELAXATION : 1
final etot is -10.000000 eV
 Largest gradient in force is 0.50 eV/A.
 Largest gradient in stress is 12.0 kBar.
 Relaxation is not converged yet!

STEP OF RELAXATION : 2
final etot is -10.250000 eV
 Largest gradient in force is 0.05 eV/A.
 Largest gradient in stress is 1.5 kBar.
 Relaxation is converged!
"""


def _make_job(
    tmp_path: Path,
    log_text: str,
    calculation: str = "scf",
    suffix: str = "ABACUS",
) -> Path:
    """Create a minimal ABACUS job directory holding one running log."""
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n\n"
        f"calculation {calculation}\n"
        f"suffix {suffix}\n",
        encoding="utf-8",
    )
    output = job / f"OUT.{suffix}"
    output.mkdir()
    (output / f"running_{calculation}.log").write_text(log_text, encoding="utf-8")
    return job


def test_resolve_version_aliases() -> None:
    assert versions.resolve_version("develop").key == "develop"
    assert versions.resolve_version("dev").key == "develop"
    assert versions.resolve_version("3.11").key == "develop"
    assert versions.resolve_version("3.11.0-beta9").key == "develop"
    assert versions.resolve_version("LTS3.10.1").key == "3.10.1LTS"
    assert versions.resolve_version("3.10.1LTS").key == "3.10.1LTS"
    assert versions.resolve_version("auto").key == "auto"
    assert versions.resolve_version(None).key == "auto"


def test_detect_version_from_log_banner() -> None:
    profile = versions.detect_version_from_text(DEVELOP_SCF_LOG)
    assert profile is not None
    assert profile.key == "develop"
    assert versions.detect_version_from_text("no banner here") is None


def test_develop_scf_results(tmp_path: Path) -> None:
    job = _make_job(tmp_path, DEVELOP_SCF_LOG)
    parameters = [
        "energy",
        "drho",
        "denergy",
        "scf_steps",
        "converged",
        "normal_end",
        "efermi",
        "force",
        "stress",
    ]

    results = get_result_from_job(str(job), parameters, "develop")

    assert results["energy"] == pytest.approx(-106.7526763159347)
    assert results["drho"] == pytest.approx(6.49158e-10)
    assert results["denergy"] == pytest.approx(-106.7526763159 + 106.7522626359)
    assert results["scf_steps"] == 2
    assert results["converged"] is True
    assert results["normal_end"] is True
    assert results["efermi"] == pytest.approx(10.7269615368)
    assert results["force"] == [[0.1, 0.2, 0.3]]
    assert results["stress"] == [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]


def test_develop_scf_results_are_detected_automatically(tmp_path: Path) -> None:
    job = _make_job(tmp_path, DEVELOP_SCF_LOG)
    results = get_result_from_job(str(job), ["converged", "drho", "efermi"])

    assert results == {
        "converged": True,
        "drho": pytest.approx(6.49158e-10),
        "efermi": pytest.approx(10.7269615368),
    }


def test_develop_relax_history_and_metrics(tmp_path: Path) -> None:
    job = _make_job(tmp_path, DEVELOP_RELAX_LOG, calculation="relax")
    log = job / "OUT.ABACUS" / "running_relax.log"

    history = read_relaxation_history(log)

    assert [item["step"] for item in history] == [1, 2]
    assert history[0]["energy"] == pytest.approx(-106.62330387)
    assert history[0]["max_force"] == pytest.approx(0.1)
    assert history[0]["max_stress"] == pytest.approx(0.0)
    assert history[0]["converged"] is False
    assert history[1]["energy"] == pytest.approx(-106.6252316276676737)
    assert history[1]["energy_change"] == pytest.approx(-106.6252316276676737 + 106.62330387)
    assert history[1]["converged"] is True

    results = get_result_from_job(
        str(job),
        ["largest_force", "largest_stress", "relax_steps", "relax_converged"],
    )
    assert results["largest_force"] == pytest.approx(0.005)
    assert results["largest_stress"] == pytest.approx(0.1)
    assert results["relax_steps"] == 2
    assert results["relax_converged"] is True


def test_lts_log_is_still_read_without_banner(tmp_path: Path) -> None:
    job = _make_job(tmp_path, LTS_RELAX_LOG, calculation="relax")

    history = read_relaxation_history(job / "OUT.ABACUS" / "running_relax.log")

    assert [item["step"] for item in history] == [1, 2]
    assert history[1]["energy"] == pytest.approx(-10.25)
    assert history[1]["max_force"] == pytest.approx(0.05)
    assert history[1]["max_stress"] == pytest.approx(1.5)
    assert history[1]["converged"] is True


def test_log_version_overrides_requested_version(tmp_path: Path) -> None:
    job = _make_job(tmp_path, DEVELOP_SCF_LOG)
    versions._WARNED_MISMATCHES.clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        profile = versions.resolve_version("3.10.1LTS", job_dir=job)

    assert profile.key == "develop"
    assert any("reports version" in str(item.message) for item in caught)


def test_configured_profile_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        CONFIG,
        "abacus",
        {
            "version": "custom",
            "versions": {
                "custom": {
                    "aliases": ["myver"],
                    "version_prefixes": ["9.9"],
                    "density_error_keywords": ["custom density marker"],
                }
            },
        },
    )

    profile = versions.resolve_version("myver")

    assert profile.key == "custom"
    assert profile.density_error_keywords == ("custom density marker",)
    assert profile.scf_converged_keywords == (
        "charge density convergence is achieved",
    )


def test_unknown_profile_field_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        CONFIG,
        "abacus",
        {"version": "custom", "versions": {"custom": {"unknown_field": ["x"]}}},
    )

    with pytest.raises(ValueError, match="unknown version profile field"):
        versions.resolve_version("custom")
