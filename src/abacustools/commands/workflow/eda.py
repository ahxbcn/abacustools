"""Helpers for the pEDA/NOCV workflow."""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Sequence

from abacustools.core.job import read_job_structure
from abacustools.io.abacus import WriteInput
from abacustools.data.lcao import PEDAOrbitalState, write_peda_state_dmr

from .common import kpoint_filename, write_abacus_job


def _is_gamma_point(direct: Sequence[float]) -> bool:
    return all(abs((float(value) + 0.5) % 1.0 - 0.5) <= 1e-6 for value in direct)


def prepare_peda_state_job(
    job: str | Path,
    states: Sequence[PEDAOrbitalState],
    destination: str | Path,
    *,
    field: str = "p0",
    r_blocks: Sequence[tuple[int, int, int]] | None = None,
    override: bool = False,
) -> Path:
    """Prepare a fixed-density pEDA state job.

    Gamma-only reference jobs keep the current single-``R=(0,0,0)`` behavior.
    Multi-k reference jobs preserve the input KPT and symmetry mode
    (``symmetry=-1`` or ``0``) and require explicit R blocks for the DMR
    Fourier transform.

    Args:
        job: Reference full-system ABACUS job directory.
        states: pEDA states for one or more k-points.
        destination: Output directory, relative to ``job`` when not absolute.
        field: ``p0``, ``p_ts`` or ``p_ab``.
        r_blocks: Explicit R blocks for multi-k DMR export.
        override: Replace an existing destination directory.

    Returns:
        The generated job directory.

    Raises:
        ValueError: If the reference job, k-points, symmetry mode or R blocks
            are inconsistent with the requested state job.
        RuntimeError: If the destination exists and ``override`` is false.
    """

    if not states:
        raise ValueError("state preparation requires at least one k-point")
    if field not in {"p0", "p_ts", "p_ab"}:
        raise ValueError(f"unknown pEDA state field: {field}")

    job_path = Path(job).resolve()
    if not job_path.is_dir():
        raise RuntimeError(f"job directory does not exist: {job_path}")
    inputs, stru_filename, structure = read_job_structure(job_path)
    if str(inputs.get("basis_type", "pw")).lower() != "lcao":
        raise ValueError("pEDA state jobs require basis_type=lcao")
    gamma_only = bool(int(inputs.get("gamma_only", 0) or 0))
    if gamma_only:
        if len(states) != 1 or not _is_gamma_point(states[0].direct):
            raise ValueError("Gamma state preparation only accepts one Gamma k-point")
    else:
        symmetry_value = inputs.get("symmetry")
        if symmetry_value is None:
            raise ValueError(
                "multi-k pEDA state jobs require an explicit symmetry=-1 or 0"
            )
        symmetry = int(symmetry_value[0] if isinstance(symmetry_value, (list, tuple)) else symmetry_value)
        if symmetry not in (-1, 0):
            raise ValueError(
                "multi-k pEDA state jobs currently support symmetry=-1 or 0; "
                f"got symmetry={symmetry}"
            )
        if r_blocks is None:
            raise ValueError("multi-k pEDA state jobs require explicit r_blocks")

    output = Path(destination)
    if not output.is_absolute():
        output = job_path / output
    if output.exists() or output.is_symlink():
        if not override:
            raise RuntimeError(
                f"state job directory already exists: {output}; use override to replace it"
            )
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        else:
            output.unlink()

    state_inputs = deepcopy(inputs)
    state_inputs.update(
        {
            "calculation": "nscf",
            "init_chg": "dm",
            "read_file_dir": ".",
            "mixing_beta": 0.0,
            "scf_nmax": 1,
            "out_hsk": 1,
            "out_chg": 1,
            "out_dmk": 0,
            "out_dmr": 0,
            "out_wfc_lcao": 0,
        }
    )
    if gamma_only:
        state_inputs["symmetry"] = -1
    kpoint_file = kpoint_filename(job_path, inputs)
    write_abacus_job(
        state_inputs,
        structure,
        job_path,
        output,
        stru_filename=stru_filename,
        kpoint=kpoint_file,
    )
    # The generated job contains symlinks to every resource, but retaining the
    # source job's relative pseudo/orbital directories would make ABACUS look
    # for them below the new state directory.
    state_inputs["pseudo_dir"] = "."
    state_inputs["orbital_dir"] = "."
    WriteInput(state_inputs, output / "INPUT")
    write_peda_state_dmr(
        states,
        structure,
        output,
        field=field,
        r_blocks=r_blocks,
    )
    return output


def prepare_gamma_state_job(
    job: str | Path,
    states: Sequence[PEDAOrbitalState],
    destination: str | Path,
    *,
    field: str = "p0",
    override: bool = False,
) -> Path:
    """Prepare one Gamma-point fixed-density ABACUS job."""

    if len(states) != 1:
        raise ValueError("Gamma state preparation requires exactly one k-point")
    if not _is_gamma_point(states[0].direct):
        raise ValueError("Gamma state preparation only accepts a Gamma k-point")
    return prepare_peda_state_job(
        job,
        states,
        destination,
        field=field,
        override=override,
    )
