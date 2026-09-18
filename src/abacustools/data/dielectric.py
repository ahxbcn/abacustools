"""Electronic dielectric tensor from a tight-binding Kubo-Greenwood sum.

The clamped-ion (electronic) dielectric tensor ``epsilon_inf`` is the response
of the electrons alone, which is what a phonon non-analytical correction needs:
the long range field of a longitudinal optical vibration is screened by the
electrons but not by the ions, which are the ones moving.

The tensor is obtained from the ``pyatb`` package, which reads the Hamiltonian,
overlap and position matrices that ABACUS writes for a LCAO calculation and
evaluates the Kubo-Greenwood sum on a dense Brillouin zone grid.  This module
holds the pure parts of that exchange: the text of the ``pyatb`` input, the
parsing of its output, and the reading of the ABACUS matrices it needs.  The
call into ``pyatb`` itself lives in
:mod:`abacustools.integrations.pyatb`.
"""

from __future__ import annotations

import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from abacustools.core.constant import ANG_TO_BOHR
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


#: Input keywords that make ABACUS write the matrices ``pyatb`` reads.  The
#: Hamiltonian and overlap go to ``data-HR-sparse_SPIN0.csr`` and
#: ``data-SR-sparse_SPIN0.csr``, the position matrix to
#: ``data-rR-sparse.csr``, all below the ``OUT.<suffix>`` directory.
MATRIX_KEYWORDS = ("out_mat_hs2", "out_mat_r")

#: Name of the dielectric tensor ``pyatb`` writes for a static only run.
STATIC_TENSOR_FILE = "static_dielectric_function.dat"

#: Name of the frequency dependent dielectric tensor ``pyatb`` writes when the
#: static only switch is unavailable; its first row is the zero frequency limit.
FREQUENCY_TENSOR_FILE = "dielectric_function_real_part.dat"

#: Directory below the pyatb working directory that holds its results.
PYATB_OUTPUT_DIRECTORY = "Out/Optical_Conductivity"

#: Input file pyatb reads from its working directory.
PYATB_INPUT_FILE = "Input"

#: Default dense Brillouin zone grid of the Kubo-Greenwood sum.
DEFAULT_GRID = (50, 50, 50)

#: Default photon energy window in eV.  It has to start at zero, because the
#: zero frequency row is the electronic dielectric tensor, and it has to reach
#: far above the band gap, because the sum runs over every transition and the
#: high energy ones keep contributing: for NaCl the tensor moves by about a
#: percent between a 20 eV window and an 80 eV one.  A window that sits inside
#: the band gap captures no transition at all and leaves an empty spectrum.
DEFAULT_OMEGA = (0.0, 80.0)

#: Default photon energy step in eV.  Only the zero frequency row is used, so
#: the step sets the resolution of a spectrum that is not read back.
DEFAULT_DOMEGA = 0.5

#: Default Gaussian broadening in eV.
DEFAULT_ETA = 0.2

#: Photon energy window below which a spectrum is likely to have been truncated
#: before the transitions stopped contributing.
MIN_WINDOW_EV = 40.0

#: Spin channels pyatb's Kubo-Greenwood sum accepts.  A collinear spin
#: polarised job is reported in two channels, which the module refuses.
SUPPORTED_NSPIN = (1, 4)

#: Number pattern of the ABACUS log, which prints exponents in either the
#: C or the Fortran style.
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def prepare_matrix_output(
    inputs: Union[str, Path, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return an ``INPUT`` that makes ABACUS write the tight-binding matrices.

    Args:
        inputs: Parsed ``INPUT`` of the source job, or the path to it.

    Returns:
        A copy of the parameters with the matrix output switched on.

    Raises:
        ValueError: When the job cannot produce the matrices, because the basis
            is not LCAO or because the calculation is gamma point only.
    """
    parameters = _as_parameters(inputs)
    basis = str(parameters.get("basis_type", "pw")).strip().lower()
    if basis != "lcao":
        raise ValueError(
            "pyatb reads the matrices of an LCAO calculation, but the job uses "
            f"basis_type {basis!r}; the position and Hamiltonian matrices are not "
            "available for a plane wave basis"
        )
    if _is_enabled(parameters.get("gamma_only", 0)):
        raise ValueError(
            "out_mat_hs2 and out_mat_r are not available for a gamma point only "
            "calculation; set gamma_only to 0 and give the job a k point mesh"
        )

    prepared = deepcopy(parameters)
    for keyword in MATRIX_KEYWORDS:
        prepared[keyword] = 1
    # The sum runs over the full Brillouin zone, so the k point set must not be
    # reduced by symmetry; a reduced set leaves the lattice sum incomplete.
    prepared["symmetry"] = 0
    return prepared


def _as_parameters(inputs: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a parsed ``INPUT`` from either a mapping or a file path."""
    if isinstance(inputs, dict):
        return dict(inputs)
    return ReadInput(str(inputs))


def _is_enabled(value: Any) -> bool:
    """Return whether an ABACUS boolean style value means true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return False


def matrix_files(job: Union[str, Path]) -> Dict[str, Path]:
    """Return the matrices of a finished ABACUS job.

    Args:
        job: ABACUS job directory holding ``the OUT.<suffix>`` folder.

    Returns:
        Mapping from ``"HR"``, ``"SR"`` and ``"rR"`` to their file.

    Raises:
        FileNotFoundError: When the job holds no ``OUT.*`` directory, or when
            one of the matrices is missing.
    """
    job_path = Path(job)
    outputs = sorted(path for path in job_path.glob("OUT.*") if path.is_dir())
    if not outputs:
        raise FileNotFoundError(f"could not find an OUT.* directory in {job_path}")
    output = outputs[0]

    wanted = {
        "HR": "data-HR-sparse_SPIN0.csr",
        "SR": "data-SR-sparse_SPIN0.csr",
        "rR": "data-rR-sparse.csr",
    }
    found = {}
    for name, filename in wanted.items():
        path = output / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"the {name} matrix of {job_path} is missing: {path}. Run the "
                "preparation stage so that ABACUS writes out_mat_hs2 and out_mat_r."
            )
        found[name] = path
    return found


def lattice_block(structure: AbacusSTRU) -> str:
    """Return the ``LATTICE`` block of a pyatb input.

    Args:
        structure: Reference structure of the calculation.

    Returns:
        The text of the block, ``lattice_constant`` included.  pyatb reads a
        lattice vector as ``lattice_constant * lattice_vector``, so the vectors
        are written in Bohr with a unit constant, which reproduces the cell of
        the ABACUS job without going through its lattice constant.
    """
    cell = np.asarray(structure.cell, dtype=float)
    vectors = cell * ANG_TO_BOHR
    rows = "\n".join(" ".join(f"{value:.10f}" for value in row) for row in vectors)
    return (
        "LATTICE\n{\n"
        "    lattice_constant        1.0000000000\n"
        "    lattice_constant_unit   Bohr\n"
        "    lattice_vector\n"
        f"{rows}\n"
        "}\n"
    )


def pyatb_input_text(
    structure: AbacusSTRU,
    *,
    nspin: int,
    fermi_energy: float,
    grid: Sequence[int] = DEFAULT_GRID,
    occ_band: int,
    matrices: Optional[Dict[str, str]] = None,
    max_kpoint_num: int = 8000,
    omega: Tuple[float, float] = DEFAULT_OMEGA,
    domega: float = DEFAULT_DOMEGA,
    eta: float = DEFAULT_ETA,
    static_only: bool = False,
) -> str:
    """Return the text of the pyatb input of a dielectric calculation.

    Args:
        structure: Reference structure of the calculation.
        nspin: Number of spin channels of the ABACUS calculation.
        fermi_energy: Fermi level of the calculation, in eV.
        grid: Dense Brillouin zone grid of the Kubo-Greenwood sum.
        occ_band: Number of occupied bands; pyatb sums transitions from these.
        matrices: File names of the ``HR``, ``SR`` and ``rR`` matrices; the
            names ABACUS writes by default when omitted.
        max_kpoint_num: Largest number of k points pyatb keeps in memory at
            once; the grid is split into batches beyond it.
        omega: Photon energy window in eV. It has to start at zero, since the
            zero frequency row is the tensor, and to reach far above the band
            gap, since every transition contributes to it; a window that sits
            inside the gap leaves an empty spectrum.
        domega: Photon energy step in eV.
        eta: Gaussian broadening in eV.
        static_only: Whether to ask for the static limit alone, which needs a
            pyatb that offers the switch; the released builds compute the
            window instead and the zero frequency row is read back.

    Returns:
        The text written to the ``Input`` file pyatb reads.

    Raises:
        ValueError: When the spin channel count is one pyatb's optical
            conductivity does not accept.
    """
    if int(nspin) not in SUPPORTED_NSPIN:
        raise ValueError(
            f"the Kubo-Greenwood sum needs nspin 1 or 4, got {nspin}; pyatb has "
            "no optical conductivity for a collinear spin polarised job"
        )
    names = dict(
        {
            "HR": "data-HR-sparse_SPIN0.csr",
            "SR": "data-SR-sparse_SPIN0.csr",
            "rR": "data-rR-sparse.csr",
        }
    )
    if matrices is not None:
        names.update(matrices)

    header = [
        "INPUT_PARAMETERS",
        "{",
        f"    nspin                   {nspin}",
        "    package                 ABACUS",
        f"    fermi_energy            {fermi_energy:.10f}",
        "    fermi_energy_unit       eV",
        f"    HR_route                {names['HR']}",
        f"    SR_route                {names['SR']}",
        f"    rR_route                {names['rR']}",
        "    HR_unit                 Ry",
        "    rR_unit                 Bohr",
        f"    max_kpoint_num          {max_kpoint_num}",
        "}",
        "",
    ]
    optical = [
        "OPTICAL_CONDUCTIVITY",
        "{",
        f"    occ_band       {occ_band}",
        f"    omega          {omega[0]} {omega[1]}",
        f"    domega         {domega}",
        f"    eta            {eta}",
        f"    grid           {' '.join(str(value) for value in grid)}",
    ]
    if static_only:
        optical.append("    static_dielectric_only     1")
    optical += ["}", ""]
    return "\n".join(header + optical) + "\n" + lattice_block(structure)


def write_pyatb_input(
    path: Path,
    text: str,
) -> Path:
    """Write the pyatb input file.

    Args:
        path: File to write.
        text: Text returned by :func:`pyatb_input_text`.

    Returns:
        The file that was written.
    """
    path.write_text(text, encoding="utf-8")
    return path


def occupied_band_count(log: Union[str, Path]) -> Optional[int]:
    """Return the number of occupied bands written in an ABACUS log.

    ABACUS prints the count next to the electron count it autoset, and pyatb
    looks for the same line, so the value is read rather than derived.

    Args:
        log: ABACUS running log.

    Returns:
        The number of occupied bands, or ``None`` when the log does not
        declare one.
    """
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"(?:occupied\s+bands|occupied\s+electronic\s+states)\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    return None if match is None else int(match.group(1))


def fermi_energy(log: Union[str, Path]) -> Optional[float]:
    """Return the Fermi energy in eV written in an ABACUS log.

    pyatb asks for a Fermi energy in its ``INPUT_PARAMETERS`` and the ABACUS
    examples pass the ``EFERMI`` of the source job, but the value is only used
    when the occupation is left to the Fermi level; the Kubo-Greenwood sum is
    given the occupied band count instead, so this is a recorded reference
    rather than something the tensor depends on.

    Args:
        log: ABACUS running log.

    Returns:
        The Fermi energy, or ``None`` when the log does not declare one.
    """
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"EFERMI\s*=\s*({_NUMBER})\s*eV", text, re.IGNORECASE)
    return None if match is None else float(match.group(1).replace("D", "E").replace("d", "e"))


def dielectric_job_parameters(job: Union[str, Path]) -> Dict[str, Any]:
    """Return the parameters a pyatb dielectric run needs from an ABACUS job.

    The spin channel count comes from the ``INPUT`` of the job and the occupied
    band count and Fermi energy from its running log, which is where ABACUS
    states the electron count it autoset.  Both are read rather than derived,
    because the number of occupied bands of a LCAO calculation follows from the
    electron count and not from the number of orbitals.

    Args:
        job: ABACUS job directory holding ``INPUT`` and an ``OUT.*`` folder.

    Returns:
        Mapping with ``nspin``, ``occ_band``, ``fermi_energy`` and ``log``.

    Raises:
        FileNotFoundError: When the job has no ``INPUT`` or no running log.
        ValueError: When the log does not declare the occupied band count or
            the Fermi energy, or when the spin channel count is one pyatb's
            optical conductivity does not accept.
    """
    from abacustools.data.abacus_result import find_job_log

    job_path = Path(job)
    inputs = ReadInput(str(job_path / "INPUT"))
    try:
        nspin = int(float(inputs.get("nspin", 1)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid nspin in {job_path / 'INPUT'}") from error
    if nspin not in SUPPORTED_NSPIN:
        raise ValueError(
            f"the dielectric tensor needs nspin 1 or 4, but the job uses {nspin}; "
            "pyatb has no Kubo-Greenwood sum for a collinear spin polarised job"
        )

    log_path = find_job_log(str(job_path))
    if log_path is None:
        raise FileNotFoundError(
            f"could not find the running log of {job_path}; the job has to finish "
            "before its dielectric tensor can be calculated"
        )
    occ_band = occupied_band_count(log_path)
    if occ_band is None:
        raise ValueError(
            f"the running log of {job_path} does not declare the occupied band "
            "count, which the Kubo-Greenwood sum is given to separate the "
            f"occupied from the empty states: {log_path}"
        )
    efermi = fermi_energy(log_path)
    if efermi is None:
        raise ValueError(
            f"the running log of {job_path} does not declare EFERMI, which pyatb "
            f"asks for although the occupied band count decides the occupation: "
            f"{log_path}"
        )
    return {
        "nspin": nspin,
        "occ_band": occ_band,
        "fermi_energy": efermi,
        "log": str(log_path),
    }


def omega_window_warning(omega: Sequence[float]) -> Optional[str]:
    """Return a warning for a photon energy window that is too short.

    The Kubo-Greenwood sum covers every transition, so the window has to reach
    well above the band gap before the zero frequency row stops moving.  For
    NaCl the tensor is about a percent low in a 20 eV window and a tenth of a
    percent low in a 40 eV one.

    Args:
        omega: Photon energy window in eV.

    Returns:
        The warning, or ``None`` when the window is long enough.
    """
    if float(omega[0]) != 0.0:
        return (
            f"the photon energy window starts at {float(omega[0])} eV rather than "
            "at zero, so its first row is not the electronic dielectric tensor"
        )
    if float(omega[1]) < MIN_WINDOW_EV:
        return (
            f"the photon energy window ends at {float(omega[1])} eV, below the "
            f"{MIN_WINDOW_EV:.0f} eV the zero frequency tensor needs before every "
            "transition is covered; the dielectric tensor is likely too small"
        )
    return None


def _parse_dielectric_values(text: str) -> List[float]:
    """Return the nine tensor components of one pyatb row."""
    values = []
    for token in text.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def read_static_dielectric(
    path: Union[str, Path],
) -> np.ndarray:
    """Read the static dielectric tensor pyatb writes on its own.

    Args:
        path: ``static_dielectric_function.dat`` of a static only run.

    Returns:
        The 3x3 tensor.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When it does not hold nine components.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"could not find the pyatb dielectric output: {file}")
    lines = [
        line
        for line in file.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"the pyatb dielectric output is empty: {file}")
    values = _parse_dielectric_values(lines[0])
    if len(values) != 9:
        raise ValueError(
            f"expected nine components in {file}, found {len(values)}"
        )
    return np.asarray(values, dtype=float).reshape(3, 3)


def read_zero_frequency_dielectric(
    path: Union[str, Path],
) -> np.ndarray:
    """Read the zero frequency limit of a pyatb dielectric spectrum.

    The released pyatb builds have no static only switch, so the electronic
    dielectric tensor is the first row of the spectrum they write for the
    requested frequency window.  That row is the zero frequency value because
    the spectrum starts at zero by default.

    Args:
        path: ``dielectric_function_real_part.dat`` of a pyatb run.

    Returns:
        The 3x3 tensor.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When no row holds nine components.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"could not find the pyatb dielectric output: {file}")
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # The row holds the photon energy first, then the nine components.
        values = _parse_dielectric_values(stripped)
        if len(values) < 10:
            continue
        return np.asarray(values[1:10], dtype=float).reshape(3, 3)
    raise ValueError(f"no dielectric row was found in {file}")


def dielectric_summary(tensor: np.ndarray) -> Dict[str, Any]:
    """Return a summary of a dielectric tensor.

    Args:
        tensor: The 3x3 tensor.

    Returns:
        Mapping with the tensor, whether it is isotropic, and its diagonal
        mean, diagonal spread and largest off diagonal element.
    """
    array = np.asarray(tensor, dtype=float)
    off_diagonal = array - np.diag(np.diag(array))
    diagonal = np.diag(array)
    return {
        "tensor": array.tolist(),
        "isotropic": bool(np.ptp(diagonal) < 1e-6 and np.all(np.abs(off_diagonal) < 1e-6)),
        "diagonal_mean": float(np.mean(diagonal)),
        "diagonal_spread": float(np.ptp(diagonal)),
        "max_off_diagonal": float(np.max(np.abs(off_diagonal))),
    }


def copy_matrices(job: Path, destination: Path) -> Dict[str, str]:
    """Copy the ABACUS matrices into a pyatb working directory.

    Args:
        job: ABACUS job directory holding the matrices.
        destination: Directory pyatb is run in.

    Returns:
        Mapping from ``"HR"``, ``"SR"`` and ``"rR"`` to the copied file name.

    Raises:
        FileNotFoundError: When a matrix is missing.
    """
    destination.mkdir(parents=True, exist_ok=True)
    names = {}
    for key, path in matrix_files(job).items():
        target = destination / path.name
        shutil.copyfile(path, target)
        names[key] = target.name
    return names
