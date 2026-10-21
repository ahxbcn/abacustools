"""Tests for shared workflow job-construction helpers."""

from __future__ import annotations

import numpy as np

from abacustools.commands.workflow.common import write_abacus_job
from abacustools.io.stru import AbacusATOM, AbacusSTRU


def test_write_abacus_job_links_resources_from_input_directories(tmp_path):
    job = tmp_path / "job"
    pp_dir = job / "assets" / "pp"
    orb_dir = job / "assets" / "orb"
    pp_dir.mkdir(parents=True)
    orb_dir.mkdir(parents=True)
    (pp_dir / "H.upf").write_text("pseudo", encoding="utf-8")
    (orb_dir / "H.orb").write_text("orbital", encoding="utf-8")
    structure = AbacusSTRU(
        cell=np.diag([10.0, 10.0, 10.0]),
        atoms=[AbacusATOM(label="H", coord=(0.0, 0.0, 0.0), pp="H.upf", orb="H.orb")],
        metadata={"lattice_constant": 1.889726, "lattice_name": "test"},
    )
    inputs = {
        "pseudo_dir": "assets/pp",
        "orbital_dir": "assets/orb",
        "basis_type": "lcao",
    }
    destination = job / "generated"

    write_abacus_job(
        inputs,
        structure,
        job,
        destination,
        stru_filename="STRU",
    )

    assert (destination / "H.upf").is_symlink()
    assert (destination / "H.orb").is_symlink()
    assert (destination / "H.upf").resolve() == (pp_dir / "H.upf").resolve()
    assert (destination / "H.orb").resolve() == (orb_dir / "H.orb").resolve()
