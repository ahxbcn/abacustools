"""Tests for the k-point-spacing convergence workflow."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from abacustools.commands.workflow.kspacing import prepare
from abacustools.io.abacus import ReadInput


STRU = """\
ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
3 0 0
0 3 0
0 0 3

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0
"""


def test_prepare_overrides_gamma_and_does_not_copy_kpt(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "calculation scf\n"
        "suffix ABACUS\n"
        "gamma_only 1\n"
        "kpoint_file KPT\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(STRU, encoding="utf-8")
    (job / "H.upf").write_text("pseudo", encoding="utf-8")
    (job / "KPT").write_text(
        "K_POINTS\n0\nGamma\n2 2 2 0 0 0\n", encoding="utf-8"
    )

    assert prepare(Namespace(job=job, values=[0.4, 0.2], override=False)) == 0

    generated = ReadInput(job / "kspacing_01" / "INPUT")
    assert generated["kspacing"] == 0.4
    assert generated["gamma_only"] == 0
    assert not (job / "kspacing_01" / "KPT").exists()
