"""Tests for the batch submission configuration of ``job prepare``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abacustools.core.config import CONFIG
from abacustools.core.submission import resolve_batch_config, write_batch_config
from abacustools.main import main


STRU = """ATOMIC_SPECIES
H 1.0 H.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Cartesian

H
0.0
1
1 2 3
"""

TEMPLATE = """{
  "example": {examples},
  "count": "{count}",
  "type": "{job_type}",
  "command": "{abacus_command}",
  "config": {"project_id": 0}
}
"""


def _source_and_library(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "water.stru"
    source.write_text(STRU, encoding="utf-8")
    library = tmp_path / "library"
    library.mkdir()
    (library / "H.upf").write_text("pseudo", encoding="utf-8")
    return source, library


def _config(**batch) -> dict:
    return {
        "submission": {
            "abacus_command": "mpirun -np 4 abacus",
            "batch": batch,
        }
    }


def test_batch_config_is_disabled_by_default() -> None:
    assert resolve_batch_config(config={"submission": {}}) is None
    assert write_batch_config(Path("."), [], config={"submission": {}}) is None


def test_batch_config_renders_the_generated_directories(tmp_path: Path) -> None:
    result = write_batch_config(
        tmp_path,
        [tmp_path / "000_a", tmp_path / "001_b"],
        job_type="relax",
        config=_config(generate=True, filename="job.json", template=TEMPLATE),
    )

    document = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    assert result == {"config_file": "job.json", "job_count": 2, "template_file": None}
    assert document["example"] == ["000_a", "001_b"]
    assert document["count"] == "2"
    assert document["type"] == "relax"
    assert document["command"] == "mpirun -np 4 abacus"
    # braces that are not a placeholder, such as the ones of a JSON object,
    # survive the substitution
    assert document["config"] == {"project_id": 0}


def test_batch_config_reads_the_configured_template_file(tmp_path: Path) -> None:
    template = tmp_path / "abacustest.json"
    template.write_text(TEMPLATE, encoding="utf-8")

    result = write_batch_config(
        tmp_path / "runs",
        [tmp_path / "runs" / "only"],
        config=_config(generate=True, template_file=str(template)),
    )

    assert result["template_file"] == str(template)
    assert result["config_file"] == "job.json"
    document = json.loads((tmp_path / "runs" / "job.json").read_text(encoding="utf-8"))
    assert document["example"] == ["only"]


def test_batch_config_rejects_a_missing_template_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="template_file not found"):
        resolve_batch_config(
            config=_config(generate=True, template_file=str(tmp_path / "none.json"))
        )


def test_batch_config_rejects_an_unknown_placeholder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown submission-template placeholder"):
        write_batch_config(
            tmp_path,
            [tmp_path / "a"],
            config=_config(generate=True, template='{"image": {image}}'),
        )


def test_prepare_writes_the_configured_batch_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``job prepare --submit-config`` lists the generated directories."""
    source, library = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": str(library), "orb": str(library)}}},
    )

    runs = tmp_path / "runs"
    assert main(
        [
            "job", "prepare", "-f", str(source), "--ftype", "stru", "--basis", "pw",
            "-o", str(runs), "--folder-syntax", "job", "--submit-config",
        ]
    ) == 0

    document = json.loads((runs / "job.json").read_text(encoding="utf-8"))
    assert document["run_dft"][0]["example"] == ["job"]
    assert document["post_dft"]["metrics"]["path"] == ["job"]


def test_prepare_honours_the_configured_submit_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --submit-config the configured submission.batch.generate applies."""
    source, library = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": str(library), "orb": str(library)}}},
    )
    monkeypatch.setitem(
        CONFIG,
        "submission",
        {"batch": {"generate": True, "filename": "job.json", "template": TEMPLATE}},
    )

    runs = tmp_path / "runs"
    assert main(
        [
            "job", "prepare", "-f", str(source), "--ftype", "stru", "--basis", "pw",
            "-o", str(runs), "--folder-syntax", "job",
        ]
    ) == 0

    document = json.loads((runs / "job.json").read_text(encoding="utf-8"))
    assert document["example"] == ["job"]
