"""Tests for the top-level package exports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import abacustools


def test_every_public_name_resolves() -> None:
    failures = []
    for name in abacustools._LAZY_EXPORTS:
        try:
            getattr(abacustools, name)
        except Exception as error:  # noqa: BLE001 - collect every failure
            failures.append((name, repr(error)))
    assert failures == []


def test_from_import_pattern() -> None:
    from abacustools import (
        AbacusATOM,
        AbacusAtomType,
        AbacusNAO,
        AbacusSTRU,
        ReadInput,
        Unitcell,
        UPF,
    )

    assert AbacusSTRU.__name__ == "AbacusSTRU"
    assert AbacusATOM.__name__ == "AbacusATOM"
    assert AbacusAtomType.__name__ == "AbacusAtomType"
    assert AbacusNAO.__name__ == "AbacusNAO"
    assert Unitcell.__name__ == "Unitcell"
    assert UPF.__name__ == "UPF"
    assert callable(ReadInput)


def test_all_matches_the_lazy_table() -> None:
    assert set(abacustools.__all__) == {"__version__", *abacustools._LAZY_EXPORTS}
    assert "__version__" in dir(abacustools)


def test_importing_abacustools_is_lazy() -> None:
    """Importing the package must not import the heavier submodules."""
    source_root = str(Path(abacustools.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=source_root)
    code = (
        "import sys, abacustools; "
        "print('abacustools.data.band' in sys.modules, "
        "'abacustools.io.stru' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"
