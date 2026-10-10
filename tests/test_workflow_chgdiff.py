"""Tests for the charge-density difference workflow."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.workflow.chgdiff import postprocess
from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.charge import Charge
from abacustools.data.grid import Grid


def _subsystem(job: Path, value: float) -> Path:
    """Write one converged subsystem job with a constant charge density."""
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 1\n", encoding="utf-8"
    )
    Charge(
        np.full((2, 2, 2), value),
        np.diag([4.0, 4.0, 4.0]),
        np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        [14, 14],
        [4.0, 4.0],
    ).save_cube(str(output / "SPIN1_CHG.cube"))
    return job


def test_postprocess_writes_the_difference_in_abacus_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job"
    _subsystem(job / "full_system", 1.0)
    _subsystem(job / "subsys1", 0.25)
    _subsystem(job / "subsys2", 0.05)
    (job / "workflow_chgdiff.json").write_text(
        json.dumps(
            {
                "format": 1,
                "workflow": "chgdiff",
                "tasks": ["full_system", "subsys1", "subsys2"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "abacustools.data.charge.get_result_from_job",
        lambda *args, **kwargs: {"converged": True},
    )
    expected = 1.0 - 0.25 - 0.05

    assert postprocess(Namespace(job=job, version="LTS3.10.1", output="difference.cube")) == 0

    written = job / "difference.cube"
    np.testing.assert_allclose(
        Charge.from_cube(str(written), format="abacus").data, expected, rtol=1e-9
    )
    np.testing.assert_allclose(
        Grid.from_cube(str(written)).data, expected * BOHR_TO_ANG**3, rtol=1e-9
    )
