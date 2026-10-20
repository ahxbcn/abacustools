"""Read the job-root ``abacus.json`` summary that ABACUS writes on develop.

ABACUS keeps one JSON summary per job directory.  Its ``comment`` field
states the units ("the unit of energy is eV and the unit of length is
Angstrom"), the ``init`` section records the input the calculation started
from and the ``output`` section appends one entry per ionic step.  Readers
consume this file before reconstructing the same facts from log keywords.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

ABACUS_JSON_NAME = "abacus.json"


def read_abacus_json(job_dir) -> Optional[Dict[str, Any]]:
    """Return the parsed ``abacus.json`` of a job, or ``None`` when absent.

    Args:
        job_dir: ABACUS job directory; the file sits in the job root.

    Returns:
        The parsed summary as a plain dict.  ``None`` when ABACUS did not
        write the file, which is normal for the LTS release.

    Raises:
        ValueError: when the file exists but is not a JSON object.
    """
    path = Path(job_dir) / ABACUS_JSON_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{path} must contain a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def latest_ionic_output(summary: Any) -> Optional[Dict[str, Any]]:
    """Return the output entry of the latest ionic step, or ``None``.

    Args:
        summary: A summary as returned by :func:`read_abacus_json`.

    Returns:
        The last ``output`` entry, which collects ``e_fermi``, ``energy``,
        ``force``, ``stress`` and related quantities of one ionic step.
        ``None`` when the summary is absent-shaped or has no output yet.
    """
    if not isinstance(summary, dict):
        return None
    output = summary.get("output")
    if not isinstance(output, list) or not output:
        return None
    entry = output[-1]
    return entry if isinstance(entry, dict) else None
