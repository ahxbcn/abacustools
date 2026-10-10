"""The ``file struinfo`` command name and its removed aliases."""

from __future__ import annotations

from pathlib import Path

import pytest

from abacustools.main import main


STRU = """ATOMIC_SPECIES
Si 28.0855 Si.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
5.43 0 0
0 5.43 0
0 0 5.43

ATOMIC_POSITIONS
Direct

Si
0.0
2
0 0 0
0.25 0.25 0.25
"""


def test_struinfo_runs_and_the_old_aliases_are_gone(tmp_path: Path) -> None:
    path = tmp_path / "STRU"
    path.write_text(STRU, encoding="utf-8")

    assert main(["file", "struinfo", str(path), "--summary"]) == 0

    for name in ("info", "stru-info", "structure-info"):
        with pytest.raises(SystemExit) as error:
            main(["file", name, str(path)])
        assert error.value.code == 2
