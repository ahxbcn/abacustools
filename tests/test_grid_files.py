"""Tests for the volumetric file discovery of both ABACUS branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from abacustools.data.grid_files import (
    DEVELOP,
    LTS,
    GridFileError,
    grid_files,
    scan_grid_files,
    spin_channels,
)


def _outdir(tmp_path: Path, *names: str) -> Path:
    outdir = tmp_path / "OUT.ABACUS"
    outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (outdir / name).write_text("cube", encoding="utf-8")
    return outdir


def test_spin_channels_of_a_calculation() -> None:
    assert spin_channels(1) == (1,)
    assert spin_channels(2) == (1, 2)
    assert spin_channels(4) == (1, 2, 3, 4)


def test_lts_names_are_classified(tmp_path: Path) -> None:
    outdir = _outdir(
        tmp_path,
        "SPIN1_CHG.cube",
        "SPIN1_CHG_INI.cube",
        "SPIN1_POT.cube",
        "SPIN2_POT_INI.cube",
        "SPIN1_TAU.cube",
        "ELF.cube",
        "ELF_SPIN2.cube",
        "ElecStaticPot.cube",
        "BAND3_GAMMA_SPIN1_CHG.cube",
        "BAND4_K2_SPIN2_CHG.cube",
    )

    found = {(item.quantity, item.path.name): item for item in scan_grid_files(outdir)}

    assert found[("charge", "SPIN1_CHG.cube")].naming == LTS
    assert found[("charge", "SPIN1_CHG.cube")].spin == 1
    assert found[("charge", "SPIN1_CHG_INI.cube")].initial is True
    assert found[("potential", "SPIN1_POT.cube")].naming == LTS
    assert found[("potential", "SPIN2_POT_INI.cube")].initial is True
    assert found[("tau", "SPIN1_TAU.cube")].spin == 1
    assert found[("elf_total", "ELF.cube")].naming == LTS
    assert found[("elf", "ELF_SPIN2.cube")].spin == 2
    assert found[("potential_es", "ElecStaticPot.cube")].naming == LTS
    gamma = found[("partial_charge", "BAND3_GAMMA_SPIN1_CHG.cube")]
    assert (gamma.band, gamma.spin, gamma.kpoint) == (3, 1, 0)
    kpoint = found[("partial_charge", "BAND4_K2_SPIN2_CHG.cube")]
    assert (kpoint.band, kpoint.spin, kpoint.kpoint) == (4, 2, 2)


def test_develop_names_are_classified(tmp_path: Path) -> None:
    outdir = _outdir(
        tmp_path,
        "chg.cube",
        "chgg3_ini.cube",
        "chgs2.cube",
        "pots1.cube",
        "pots2_ini.cube",
        "potes.cube",
        "tau.cube",
        "taus2.cube",
        "elftot.cube",
        "elfs2g5.cube",
        "LDOS_-7.5eV.cube",
        "pchgi1s1.cube",
        "pchgi2s2k3.cube",
    )

    found = {(item.quantity, item.path.name): item for item in scan_grid_files(outdir)}

    charge = found[("charge", "chg.cube")]
    assert (charge.naming, charge.spin, charge.step) == (DEVELOP, 1, None)
    assert found[("charge", "chgg3_ini.cube")].step == 3
    assert found[("charge", "chgg3_ini.cube")].initial is True
    assert found[("charge", "chgs2.cube")].spin == 2
    assert found[("potential", "pots1.cube")].naming == DEVELOP
    assert found[("potential", "pots2_ini.cube")].initial is True
    assert found[("potential_es", "potes.cube")].naming == DEVELOP
    assert found[("tau", "tau.cube")].spin == 1
    assert found[("tau", "taus2.cube")].spin == 2
    assert found[("elf_total", "elftot.cube")].naming == DEVELOP
    assert found[("elf", "elfs2g5.cube")].step == 5
    assert found[("ldos", "LDOS_-7.5eV.cube")].energy == pytest.approx(-7.5)
    partial = found[("partial_charge", "pchgi2s2k3.cube")]
    assert (partial.band, partial.spin, partial.kpoint) == (2, 2, 3)
    assert found[("partial_charge", "pchgi1s1.cube")].kpoint == 0


def test_documented_pot_es_name_is_recognised(tmp_path: Path) -> None:
    outdir = _outdir(tmp_path, "pot_es.cube")

    found = grid_files(outdir, "potential_es")

    assert found[0].path.name == "pot_es.cube"
    assert found[0].naming == DEVELOP


def test_charge_selection_reads_both_conventions(tmp_path: Path) -> None:
    lts = _outdir(tmp_path / "lts", "SPIN1_CHG.cube", "SPIN2_CHG.cube")
    develop = _outdir(tmp_path / "develop", "chgs1.cube", "chgs2.cube")

    lts_files = grid_files(lts, "charge", spins=spin_channels(2))
    develop_files = grid_files(develop, "charge", spins=spin_channels(2))

    assert [item.path.name for item in lts_files] == ["SPIN1_CHG.cube", "SPIN2_CHG.cube"]
    assert [item.path.name for item in develop_files] == ["chgs1.cube", "chgs2.cube"]
    assert {item.naming for item in develop_files} == {DEVELOP}


def test_geometry_steps_prefer_the_plain_file_then_the_last_step(tmp_path: Path) -> None:
    stepped = _outdir(tmp_path / "stepped", "chgs1g1.cube", "chgs1g2.cube", "chgs1g5.cube")
    plain = _outdir(tmp_path / "plain", "chgs1.cube", "chgs1g5.cube")

    assert grid_files(stepped, "charge", spins=(1,))[0].step == 5
    assert grid_files(stepped, "charge", spins=(1,), step=2)[0].step == 2
    assert grid_files(plain, "charge", spins=(1,))[0].step is None


def test_missing_channel_and_unknown_quantity_are_reported(tmp_path: Path) -> None:
    outdir = _outdir(tmp_path, "chgs1.cube")

    with pytest.raises(GridFileError, match="no final charge file for spin 2"):
        grid_files(outdir, "charge", spins=(1, 2))
    with pytest.raises(GridFileError, match="unknown volumetric quantity"):
        grid_files(outdir, "spin_texture")
    with pytest.raises(GridFileError, match="output directory not found"):
        scan_grid_files(tmp_path / "missing")


def test_mixed_naming_conventions_are_rejected(tmp_path: Path) -> None:
    outdir = _outdir(tmp_path, "SPIN1_CHG.cube", "chgs2.cube")

    with pytest.raises(GridFileError, match="mix the LTS and develop"):
        grid_files(outdir, "charge", spins=(1, 2))


def test_describe_names_the_convention(tmp_path: Path) -> None:
    outdir = _outdir(tmp_path, "chgs1g2.cube", "SPIN1_CHG.cube", "LDOS_3.5eV.cube")

    develop = grid_files(outdir, "charge", spins=(1,), step=2)[0]
    lts = grid_files(outdir, "charge", spins=(1,), step=None)[0]
    ldos = grid_files(outdir, "ldos")[0]

    assert develop.describe() == "chgs1g2.cube (develop, step 2)"
    assert lts.describe() == "SPIN1_CHG.cube (lts)"
    assert ldos.describe() == "LDOS_3.5eV.cube (develop, 3.5 eV)"
