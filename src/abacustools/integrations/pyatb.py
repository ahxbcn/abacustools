"""Adapter that drives the ``pyatb`` package for a dielectric calculation.

``pyatb`` evaluates the Kubo-Greenwood sum over the tight-binding matrices
ABACUS writes for a LCAO calculation, which is the electronic dielectric
tensor a phonon non-analytical correction needs.  The package is an optional
dependency: it is imported inside the function that runs it, and importing this
module never imports it.

``pyatb`` computes in C++ behind a compiled extension and parallelises through
``mpi4py``, so it needs an MPI runtime on the machine that runs it.  That is
why the calculation is a postprocessing step: the ABACUS job stays on the
cluster and only its matrices travel back.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from abacustools.data.dielectric import (
    DEFAULT_DOMEGA,
    DEFAULT_ETA,
    DEFAULT_GRID,
    DEFAULT_OMEGA,
    FREQUENCY_TENSOR_FILE,
    PYATB_INPUT_FILE,
    PYATB_OUTPUT_DIRECTORY,
    STATIC_TENSOR_FILE,
    dielectric_summary,
    pyatb_input_text,
    read_static_dielectric,
    read_zero_frequency_dielectric,
    write_pyatb_input,
)


def pyatb_available() -> bool:
    """Return whether the optional ``pyatb`` package can be imported."""
    import importlib.util

    return importlib.util.find_spec("pyatb") is not None


def pyatb_version() -> Optional[str]:
    """Return the version of the installed ``pyatb`` package.

    Returns:
        The version, or ``None`` when the package is not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pyatb")
    except PackageNotFoundError:
        return None


def load_pyatb():
    """Import and return the ``pyatb`` package.

    Returns:
        The imported ``pyatb`` module.

    Raises:
        ImportError: When ``pyatb`` or its MPI runtime is missing.
    """
    try:
        import pyatb
    except ImportError as error:
        raise ImportError(
            "the electronic dielectric tensor needs the optional `pyatb` package; "
            "install it with `pip install pyatb` and make sure an MPI runtime is "
            "available, for example with `conda install -c conda-forge mpich`"
        ) from error
    except RuntimeError as error:  # raised by mpi4py when no MPI library loads
        raise ImportError(
            "`pyatb` needs an MPI runtime through `mpi4py`, which could not be "
            f"loaded: {error}. Install one, for example with "
            "`conda install -c conda-forge mpich`."
        ) from error
    return pyatb


def dielectric_tensor(
    workdir: Path,
    structure,
    *,
    nspin: int,
    fermi_energy: float,
    occ_band: int,
    grid: Sequence[int] = DEFAULT_GRID,
    matrices: Optional[Dict[str, str]] = None,
    max_kpoint_num: int = 8000,
    omega: Sequence[float] = DEFAULT_OMEGA,
    domega: float = DEFAULT_DOMEGA,
    eta: float = DEFAULT_ETA,
    static_only: bool = False,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """Run pyatb and return the electronic dielectric tensor.

    Args:
        workdir: Directory holding the ABACUS matrices and the ``Input`` file;
            pyatb writes its results below it.
        structure: Reference structure of the calculation.
        nspin: Number of spin channels of the ABACUS calculation.
        fermi_energy: Fermi level in eV.
        occ_band: Number of occupied bands.
        grid: Dense Brillouin zone grid of the Kubo-Greenwood sum.
        matrices: File names of the ``HR``, ``SR`` and ``rR`` matrices.
        max_kpoint_num: Largest number of k points held in memory at once.
        omega: Photon energy window in eV.
        domega: Photon energy step in eV.
        eta: Gaussian broadening in eV.
        static_only: Whether to ask pyatb for the static limit alone; the
            released builds lack that switch and the zero frequency row of the
            spectrum is read instead.
        command: Unused placeholder for a callable override; pyatb is called
            through its own command line when given, and in process otherwise.

    Returns:
        Mapping with the tensor, a summary of it, the grid, the occupied band
        count and the output directory.

    Raises:
        ImportError: When ``pyatb`` or its MPI runtime is missing.
        FileNotFoundError: When pyatb writes no dielectric output.
        ValueError: When the dielectric output cannot be read.
    """
    load_pyatb()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    text = pyatb_input_text(
        structure,
        nspin=nspin,
        fermi_energy=fermi_energy,
        grid=grid,
        occ_band=occ_band,
        matrices=matrices,
        max_kpoint_num=max_kpoint_num,
        omega=(float(omega[0]), float(omega[1])),
        domega=domega,
        eta=eta,
        static_only=static_only,
    )
    write_pyatb_input(workdir / PYATB_INPUT_FILE, text)

    if command:
        _run_command(command, workdir, static_only=static_only)
    else:
        _run_in_process(workdir)

    output = workdir / PYATB_OUTPUT_DIRECTORY
    if static_only:
        tensor = read_static_dielectric(output / STATIC_TENSOR_FILE)
    else:
        tensor = read_zero_frequency_dielectric(output / FREQUENCY_TENSOR_FILE)

    summary = dielectric_summary(tensor)
    summary.update(
        {
            "grid": [int(value) for value in grid],
            "occ_band": int(occ_band),
            "nspin": int(nspin),
            "fermi_energy_ev": float(fermi_energy),
            "static_only": bool(static_only),
            "omega_ev": [float(omega[0]), float(omega[1])],
            "domega_ev": float(domega),
            "eta_ev": float(eta),
            "pyatb_version": pyatb_version(),
            "output": str(output),
        }
    )
    return summary


def _run_in_process(workdir: Path) -> None:
    """Run pyatb inside this process, in the working directory it expects.

    ``pyatb`` records ``os.getcwd()`` as its input and output directory while
    its package is being imported, so the working directory has to be entered
    before the import happens rather than around the call.  Running the module
    by name does that, because the import then takes place during the run, and
    any copy that a previous run already cached has to be dropped first or the
    directory of that earlier run would be reused.
    """
    import runpy
    import sys

    previous = Path.cwd()
    previous_argv = sys.argv
    cached = [name for name in sys.modules if name == "pyatb" or name.startswith("pyatb.")]
    try:
        for name in cached:
            del sys.modules[name]
        os.chdir(workdir)
        sys.argv = ["pyatb"]
        runpy.run_module("pyatb.main", run_name="__main__", alter_sys=True)
    except SystemExit as exit_status:
        if exit_status.code not in (None, 0):
            raise RuntimeError(f"pyatb exited with status {exit_status.code}") from None
    finally:
        sys.argv = previous_argv
        os.chdir(previous)


def _run_command(command: str, workdir: Path, *, static_only: bool) -> None:
    """Run a user supplied pyatb command inside the working directory."""
    import shlex
    import subprocess

    if static_only:
        command = f"{command} --static_dielectric_only 1"
    completed = subprocess.run(
        shlex.split(command),
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"pyatb failed with status {completed.returncode}:\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
