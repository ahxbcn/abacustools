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


def prepare_gamma_state_job(
    job: str | Path,
    states: Sequence[PEDAOrbitalState],
    destination: str | Path,
    *,
    field: str = "p0",
    override: bool = False,
) -> Path:
    """Prepare one Gamma-point fixed-density ABACUS job.

    The generated directory contains the original structure and referenced
    pseudopotential/orbital files, an INPUT configured to read a DMR and hold
    the density fixed, and the ``dmrs{spin}_nao.csr`` files for the requested
    pEDA state.

    Args:
        job: Reference full-system ABACUS job directory.
        states: pEDA states; exactly one Gamma-point state is required.
        destination: Output directory, relative to ``job`` when not absolute.
        field: ``p0`` or ``p_ts``.
        override: Replace an existing destination directory.

    Returns:
        The generated job directory.

    Raises:
        ValueError: If the state is not a single Gamma point or the reference
            job is not an LCAO Gamma calculation.
        RuntimeError: If the destination exists and ``override`` is false.
    """

    if len(states) != 1:
        raise ValueError("Gamma state preparation requires exactly one k-point")
    if not _is_gamma_point(states[0].direct):
        raise ValueError("Gamma state preparation only accepts a Gamma k-point")
    if field not in {"p0", "p_ts", "p_ab"}:
        raise ValueError(f"unknown pEDA state field: {field}")

    job_path = Path(job).resolve()
    if not job_path.is_dir():
        raise RuntimeError(f"job directory does not exist: {job_path}")
    inputs, stru_filename, structure = read_job_structure(job_path)
    if str(inputs.get("basis_type", "pw")).lower() != "lcao":
        raise ValueError("pEDA state jobs require basis_type=lcao")
    if not bool(int(inputs.get("gamma_only", 0) or 0)):
        raise ValueError("pEDA Gamma state jobs require gamma_only=1 in the reference INPUT")

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
            "symmetry": -1,
            "out_hsk": 1,
            "out_chg": 1,
            "out_dmk": 0,
            "out_dmr": 0,
            "out_wfc_lcao": 0,
        }
    )
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
    )
    return output
