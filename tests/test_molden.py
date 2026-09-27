"""Tests for the LCAO-to-Molden export."""

from __future__ import annotations

import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.postprocess.molden import run
from abacustools.data.mayer import NAOData, NAOOrbital
from abacustools.data.molden import (
    _wfc_channels,
    convert_wfc_to_molden,
    fit_nao_to_cgto,
)
from abacustools.io.molden import MoldenAtom, MoldenOrbital, MoldenShell, format_molden


UPF = """\
<UPF version="2.0.1">
  <PP_HEADER element="H" z_valence="1.0" l_max="0" mesh_size="3"/>
  <PP_MESH>
    <PP_R type="real" size="3">0.0 1.0 2.0</PP_R>
    <PP_RAB type="real" size="3">1.0 1.0 1.0</PP_RAB>
  </PP_MESH>
  <PP_LOCAL size="3">-4.0 -2.0 -1.0</PP_LOCAL>
  <PP_RHOATOM size="3">0.1 0.2 0.3</PP_RHOATOM>
</UPF>
"""


def _write_orb(path: Path, element: str = "H") -> None:
    """Write a small hydrogen 1s numerical orbital."""

    mesh = 401
    dr = 0.02
    r = np.arange(mesh) * dr
    values = 2.0 * np.exp(-1.7 * r**2)
    body = " ".join(f"{value:.12e}" for value in values)
    path.write_text(
        f"""Element {element}
Energy Cutoff(Ry) 100
Radius Cutoff(a.u.) 8
Lmax 0
Number of Sorbital--> 1
SUMMARY  END

Mesh {mesh}
dr {dr}
Type L N
0 0 0
{body}
""",
        encoding="utf-8",
    )


def _write_job(job: Path, *, nspin: int = 1, gamma_only: bool = True, cell: bool = True) -> None:
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "H.upf").write_text(UPF, encoding="utf-8")
    _write_orb(job / "H.orb")
    lattice = (
        "LATTICE_VECTORS\n5 0 0\n0 5 0\n0 0 5\n\n" if cell else "LATTICE_CONSTANT\n1.889726\n\n"
    )
    (job / "STRU").write_text(
        "ATOMIC_SPECIES\n"
        "H 1.008 H.upf\n\n"
        "NUMERICAL_ORBITAL\n"
        "H.orb\n\n" + lattice + "ATOMIC_POSITIONS\n"
        "Cartesian\n\n"
        "H\n"
        "0.0\n"
        "2\n"
        "0.0 0.0 -0.35 1 1 1\n"
        "0.0 0.0 0.35 1 1 1\n",
        encoding="utf-8",
    )
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "basis_type lcao\n"
        f"nspin {nspin}\n"
        f"gamma_only {1 if gamma_only else 0}\n"
        "suffix ABACUS\n",
        encoding="utf-8",
    )
    for ispin in range(1, nspin + 1):
        (output / f"WFC_NAO_GAMMA{ispin}.txt").write_text(
            "2 (number of bands)\n"
            "2 (number of orbitals)\n"
            "1 (band)\n"
            "-0.75 (Ry)\n"
            "1.0 (Occupations)\n"
            "0.7071067811865476 0.7071067811865476\n"
            "2 (band)\n"
            "0.25 (Ry)\n"
            "0.0 (Occupations)\n"
            "0.7071067811865476 -0.7071067811865476\n",
            encoding="utf-8",
        )


def test_fit_reproduces_gaussian_and_nodal_orbitals():
    r = np.arange(600, dtype=float) * 0.02
    # A nodal function: the second coefficient is negative, so the fit has to
    # resolve a radial node rather than only a positive Gaussian blob.
    values = 3.0 * np.exp(-2.0 * r**2) - 0.7 * np.exp(-0.6 * r**2)
    orbitals = (NAOOrbital(l=0, n=0, values=values),)
    nao = NAOData("H", 100.0, 12.0, 0, (1,), len(r), 0.02, orbitals)

    shells, error = fit_nao_to_cgto(nao, nprim=6)

    assert error < 5e-3
    assert len(shells) == 1
    exponents = shells[0].exponents
    coefficients = shells[0].coefficients
    matrix = np.array(
        [
            [_normalization(0, exponent) * np.exp(-exponent * radius**2) for exponent in exponents]
            for radius in r
        ]
    )
    reconstructed = matrix @ coefficients
    np.testing.assert_allclose(reconstructed, values, atol=5e-3 * np.abs(values).max())
    assert np.max(np.abs(coefficients)) < 50.0


def _normalization(l: int, alpha: float) -> float:
    return math.sqrt(2.0 * (2.0 * alpha) ** ((2 * l + 3) / 2.0) / math.gamma(l + 1.5))


def test_format_molden_has_all_sections():
    atoms = [
        MoldenAtom("H", 1, (0.0, 0.0, -0.35), (MoldenShell(0, [1.0, 0.3], [0.5, 0.6]),)),
        MoldenAtom("H", 1, (0.0, 0.0, 0.35), (MoldenShell(0, [1.0, 0.3], [0.5, 0.6]),)),
    ]
    orbitals = [MoldenOrbital(-0.5, "Alpha", 2.0, [0.7, 0.7])]
    text = format_molden(np.eye(3) * 5.0, atoms, {"H": 1.0}, orbitals)

    assert "[Molden Format]" in text
    assert "[Cell]" in text
    assert "[Nval]" in text
    assert " H 1" in text
    assert "[Atoms] AU" in text
    assert "[GTO]" in text
    assert "1 0" in text and "2 0" in text
    assert "[MO]" in text
    assert "Ene= -0.500000000000" in text
    assert "Spin= Alpha" in text


def test_format_molden_uses_exponent_first():
    atoms = [MoldenAtom("He", 2, (0.0, 0.0, 0.0), (MoldenShell(0, [3.0], [0.4]),))]
    text = format_molden([], atoms, {"He": 2.0}, [MoldenOrbital(0.0, "Alpha", 1.0, [1.0])])
    lines = text.splitlines()
    index = lines.index(" s    1 1.000000")
    fields = lines[index + 1].split()
    assert float(fields[0]) == pytest.approx(3.0)
    assert float(fields[1]) == pytest.approx(0.4)


def test_wfc_channels_gamma_and_develop(tmp_path: Path):
    output = tmp_path / "OUT.ABACUS"
    output.mkdir()
    (output / "WFC_NAO_GAMMA1.txt").write_text("", encoding="utf-8")
    (output / "WFC_NAO_GAMMA2.txt").write_text("", encoding="utf-8")
    channels, gamma, nk = _wfc_channels(output, 2)
    assert gamma and nk == 1
    assert channels[(1, 1)].name == "WFC_NAO_GAMMA1.txt"
    assert channels[(2, 1)].name == "WFC_NAO_GAMMA2.txt"

    slow = tmp_path / "OUT.SLOW"
    slow.mkdir()
    for name in ("wfk1s1_nao.txt", "wfk2s1_nao.txt", "wfk1s2_nao.txt", "wfk2s2_nao.txt"):
        (slow / name).write_text("", encoding="utf-8")
    channels, gamma, nk = _wfc_channels(slow, 2)
    assert not gamma and nk == 2
    assert channels[(2, 2)].name == "wfk2s2_nao.txt"


def test_molden_command_writes_file(tmp_path: Path):
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)

    assert (
        run(
            Namespace(
                job=job,
                output="wfc.molden",
                kpoint=None,
                gto_primitives=6,
                atoms_unit="bohr",
                json=False,
            )
        )
        == 0
    )
    text = (job / "wfc.molden").read_text(encoding="utf-8")
    assert text.startswith("[Molden Format]")
    assert "[Cell]" in text
    assert "[Atoms] AU" in text
    assert "Occup= 1.000000000000" in text
    assert "Ene= -0.375000000000" in text
    # The fitted hydrogen shell is listed for both atoms.
    assert text.count(" s    6 1.000000") == 2


def test_molden_command_reports_json(tmp_path: Path, capsys):
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)
    assert (
        run(
            Namespace(
                job=job,
                output=None,
                kpoint=None,
                gto_primitives=6,
                atoms_unit="bohr",
                json=True,
            )
        )
        == 0
    )
    import json

    report = json.loads(capsys.readouterr().out)
    assert report["basis_functions"] == 2
    assert report["nbands"] == 2
    assert report["gamma_only"] is True


def test_molden_command_handles_spin(tmp_path: Path):
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job, nspin=2)
    result = convert_wfc_to_molden(job)
    text = Path(result.output).read_text(encoding="utf-8")
    assert "Spin= Alpha" in text
    assert "Spin= Beta" in text
    assert result.nspin == 2


def test_molden_rejects_complex_wavefunction(tmp_path: Path):
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job, gamma_only=False)
    output = job / "OUT.ABACUS"
    (output / "WFC_NAO_GAMMA1.txt").unlink()
    (output / "WFC_NAO_K1.txt").write_text(
        "2 (number of bands)\n"
        "2 (number of orbitals)\n"
        "1 (band)\n"
        "-0.75 (Ry)\n"
        "1.0 (Occupations)\n"
        "0.7071067811865476 0.0000000000000000 0.7071067811865476 0.1000000000000000\n"
        "2 (band)\n"
        "0.25 (Ry)\n"
        "0.0 (Occupations)\n"
        "0.7071067811865476 0.0000000000000000 -0.7071067811865476 0.0000000000000000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="real"):
        convert_wfc_to_molden(job)


def test_molden_requires_lcao(tmp_path: Path):
    job = tmp_path / "job"
    job.mkdir()
    _write_job(job)
    (job / "INPUT").write_text("INPUT_PARAMETERS\nbasis_type pw\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lcao"):
        convert_wfc_to_molden(job)


def test_gto_atom_indices_are_one_based_and_match_atoms():
    """The [GTO] atom index must be 1-based, as CP2K and Multiwfn write it."""

    atoms = [
        MoldenAtom("H", 1, (0.0, 0.0, -0.35), (MoldenShell(0, [1.0], [0.5]),)),
        MoldenAtom("H", 1, (0.0, 0.0, 0.35), (MoldenShell(0, [1.0], [0.5]),)),
    ]
    text = format_molden([], atoms, {"H": 1.0}, [MoldenOrbital(0.0, "Alpha", 2.0, [0.7, 0.7])])
    lines = text.splitlines()
    gto = lines.index("[GTO]")
    assert lines[gto + 1] == "1 0"
    assert lines[gto + 4] == ""  # blank line separating the atom blocks
    assert lines[gto + 5] == "2 0"
