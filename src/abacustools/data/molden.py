"""Convert ABACUS LCAO wavefunctions into Molden-format files.

The Molden format only carries Gaussian-type orbitals, so the ABACUS numerical
atomic orbitals are first expanded in contracted Gaussian functions and the
LCAO wavefunction coefficients are then written on that same basis.  Each
radial numerical orbital ``R(r)`` is fit to

``R(r) = sum_i c_i * N_l(alpha_i) * r^l * exp(-alpha_i r^2)``

with ``N_l`` the normalization of the primitive with a unit real spherical
harmonic.  The exponents are scanned over even-tempered families and the
linear coefficients follow from a column-scaled, lightly regularized
least-squares solve, which keeps the contraction well conditioned while still
resolving the radial nodes of the inner orbitals.

Molden represents a single real-valued set of molecular orbitals, so the
command works on a Gamma-point-only run (``gamma_only 1``) or on a single real
k-point of a non-gamma run.  Spin-polarized jobs write both spin channels into
one file.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ase.data import atomic_numbers

from abacustools.core.constant import HARTREE_TO_EV, RY_TO_EV
from abacustools.data.charge import valence_electrons
from abacustools.data.mayer import NAOData, read_nao_file, read_wfc_nao_k_data
from abacustools.io.abacus import ReadInput
from abacustools.io.molden import MoldenAtom, MoldenOrbital, MoldenShell, write_molden
from abacustools.io.stru import AbacusSTRU


#: Files that carry the LCAO coefficients, newest ABACUS branch first.
_PLAIN_GAMMA = re.compile(r"^WFC_NAO_GAMMA(\d+)\.TXT$", re.IGNORECASE)
_PLAIN_K = re.compile(r"^WFC_NAO_K(\d+)\.TXT$", re.IGNORECASE)
_DEVELOP_GAMMA = re.compile(r"^wf_nao\.txt$", re.IGNORECASE)
_DEVELOP_GAMMA_SPIN = re.compile(r"^wfs(\d+)_nao\.txt$", re.IGNORECASE)
_DEVELOP_K = re.compile(r"^wfk(\d+)_nao\.txt$", re.IGNORECASE)
_DEVELOP_K_SPIN = re.compile(r"^wfk(\d+)s(\d+)_nao\.txt$", re.IGNORECASE)

#: Smallest and largest ratio between neighbouring exponents of a fit.
_MIN_EXPONENT_RATIO = 1.6
_MAX_EXPONENT_RATIO = 3.0

#: Ridge penalty added to the column-scaled normal equations.  It only keeps
#: the solve away from the singular limit; the fit stays close to least squares.
_RIDGE = 1.0e-8


@dataclass(frozen=True)
class MoldenResult:
    """Summary of one LCAO-to-Molden conversion."""

    job: str
    output: str
    gamma_only: bool
    kpoint: int
    nkpoints: int
    nspin: int
    nbands: int
    basis_functions: int
    gto_primitives: int
    max_relative_error: float


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _output_directory(job: Path, inputs: dict) -> Path:
    """Return the ``OUT.*`` directory that holds the job's wavefunctions."""

    suffix = str(_scalar(inputs.get("suffix", "ABACUS")))
    expected = job / f"OUT.{suffix}"
    if expected.is_dir():
        return expected
    candidates = sorted(path for path in job.glob("OUT.*") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"could not find an OUT.* directory in {job}")
    raise FileNotFoundError(f"could not determine the output directory of {job}")


def _resolve_orbital(job: Path, orbital_dir: Any, filename: str) -> Path:
    base = Path(str(orbital_dir)) if orbital_dir else job
    return (base if base.is_absolute() else job / base) / filename


def _primitive_normalization(l: int, alpha: float) -> float:
    """Normalization of ``r^l exp(-alpha r^2) Y_lm`` with a unit ``Y_lm``."""

    gamma = math.gamma(l + 1.5)
    return math.sqrt(2.0 * (2.0 * alpha) ** ((2 * l + 3) / 2.0) / gamma)


def _primitive_matrix(l: int, exponents: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return normalized Gaussian primitives sampled on ``r``, one per column."""

    normalization = np.array([_primitive_normalization(l, alpha) for alpha in exponents])
    return (r[:, np.newaxis] ** l) * np.exp(-np.outer(r**2, exponents)) * normalization


def _fit_exponents(
    values: np.ndarray, l: int, r: np.ndarray, nprim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one radial function and return ``(exponents, coefficients)``.

    The exponents are a scanned even-tempered family and the coefficients
    multiply normalized primitives.  Column scaling keeps the normal equations
    well conditioned whatever the spread of the exponents.
    """

    best_residual: Optional[float] = None
    best: Optional[tuple[np.ndarray, np.ndarray]] = None
    for alpha_min in np.geomspace(1.0e-3, 1.0, 30):
        for ratio in np.linspace(_MIN_EXPONENT_RATIO, _MAX_EXPONENT_RATIO, 15):
            exponents = alpha_min * ratio ** np.arange(nprim - 1, -1, -1, dtype=float)
            matrix = _primitive_matrix(l, exponents, r)
            scale = np.sqrt(np.sum(matrix * matrix, axis=0)) + 1.0e-300
            scaled = matrix / scale
            normal = scaled.T @ scaled + _RIDGE * np.eye(nprim)
            coefficients = np.linalg.solve(normal, scaled.T @ values) / scale
            residual = float(np.sqrt(np.mean((values - matrix @ coefficients) ** 2)))
            if best_residual is None or residual < best_residual:
                best_residual = residual
                best = (exponents, coefficients)
    if best is None:  # pragma: no cover - the scan always yields candidates.
        raise ValueError("could not build a Gaussian exponent set")
    return best


def fit_nao_to_cgto(nao: NAOData, nprim: int = 6) -> tuple[list[MoldenShell], float]:
    """Expand every orbital of a numerical-orbital file into Gaussian shells.

    Args:
        nao: Parsed numerical-orbital file.
        nprim: Number of Gaussian primitives per shell.

    Returns:
        The shells in basis order and the largest relative radial fit error.
    """

    if nprim < 1:
        raise ValueError("nprim must be positive")
    r = np.arange(nao.mesh, dtype=float) * nao.dr
    shells: list[MoldenShell] = []
    max_error = 0.0
    for orbital in nao.orbitals:
        exponents, coefficients = _fit_exponents(orbital.values, orbital.l, r, nprim)
        shells.append(MoldenShell(orbital.l, exponents, coefficients))
        scale = float(np.max(np.abs(orbital.values)))
        if scale > 0.0:
            matrix = _primitive_matrix(orbital.l, exponents, r)
            residual = float(np.sqrt(np.mean((orbital.values - matrix @ coefficients) ** 2)))
            max_error = max(max_error, residual / scale)
    return shells, max_error


def _wfc_channels(output: Path, nspin: int) -> tuple[dict[tuple[int, int], Path], bool, int]:
    """Locate wavefunction files and return ``(channels, gamma_only, nkpoints)``.

    Keys of ``channels`` are ``(ispin, ik)`` with one-based indices.
    """

    gamma: dict[int, Path] = {}
    k_lts: dict[int, Path] = {}
    k_dev: dict[tuple[int, int], Path] = {}
    for path in sorted(output.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if (match := _PLAIN_GAMMA.match(name)) is not None:
            gamma[int(match.group(1))] = path
        elif (match := _PLAIN_K.match(name)) is not None:
            k_lts[int(match.group(1))] = path
        elif _DEVELOP_GAMMA.match(name) is not None:
            gamma[1] = path
        elif (match := _DEVELOP_GAMMA_SPIN.match(name)) is not None:
            gamma[int(match.group(1))] = path
        elif (match := _DEVELOP_K.match(name)) is not None:
            k_dev[(int(match.group(1)), 1)] = path
        elif (match := _DEVELOP_K_SPIN.match(name)) is not None:
            k_dev[(int(match.group(1)), int(match.group(2)))] = path

    if gamma:
        return {(ispin, 1): path for ispin, path in gamma.items()}, True, 1
    if k_dev:
        nkpoints = max(ik for ik, _ in k_dev)
        return {(ispin, ik): path for (ik, ispin), path in k_dev.items()}, False, nkpoints
    if k_lts:
        if nspin == 1:
            return {(1, index): path for index, path in k_lts.items()}, False, max(k_lts)
        nkpoints = len(k_lts) // 2
        channels: dict[tuple[int, int], Path] = {}
        for index, path in k_lts.items():
            if index <= nkpoints:
                channels[(1, index)] = path
            else:
                channels[(2, index - nkpoints)] = path
        return channels, False, nkpoints
    raise FileNotFoundError(f"no LCAO wavefunction files found in {output}")


def _element_valences(structure: AbacusSTRU, inputs: dict, job: Path) -> dict[str, float]:
    """Return ``{element: valence electrons}`` from the pseudopotentials."""

    valences: dict[str, float] = {}
    per_atom = valence_electrons(structure, inputs.get("pseudo_dir"), job)
    for atom, valence in zip(structure.atoms, per_atom):
        valences.setdefault(atom.element or atom.label, float(valence))
    return valences


def convert_wfc_to_molden(
    job: str | Path,
    output: Optional[str | Path] = None,
    *,
    kpoint: Optional[int] = None,
    gto_primitives: int = 6,
    atoms_unit: str = "bohr",
) -> MoldenResult:
    """Convert the LCAO wavefunction of a job into a Molden file.

    Args:
        job: ABACUS job directory with ``INPUT``, ``STRU``, the orbital files
            and an ``OUT.*`` directory carrying ``WFC_NAO_*``.
        output: Output filename, relative to ``job``; defaults to
            ``wfc.molden``.
        kpoint: One-based k-point index to write; defaults to Gamma when the
            run stored Gamma wavefunctions and to the first k-point otherwise.
        gto_primitives: Gaussian primitives per shell.
        atoms_unit: Unit of the ``[Atoms]`` block.

    Returns:
        A summary of the conversion.
    """

    job_path = Path(job).resolve()
    input_file = job_path / "INPUT"
    if not input_file.is_file():
        raise FileNotFoundError(f"could not find INPUT in {job_path}")
    inputs = ReadInput(str(input_file))
    if not inputs:
        raise ValueError(f"could not read any parameter from {input_file}")
    if str(_scalar(inputs.get("basis_type", "pw"))).lower() != "lcao":
        raise ValueError("the Molden export requires basis_type=lcao")
    nspin = int(_scalar(inputs.get("nspin", 1)))
    if nspin not in (1, 2):
        raise ValueError(f"the Molden export supports nspin=1 or 2, got {nspin}")

    output_dir = _output_directory(job_path, inputs)
    channels, stored_gamma, nkpoints = _wfc_channels(output_dir, nspin)
    if kpoint is None:
        kpoint = 1
    if kpoint < 1 or kpoint > nkpoints:
        raise ValueError(f"kpoint index {kpoint} is outside 1..{nkpoints}")
    if stored_gamma and kpoint != 1:
        raise ValueError("the job only stored the Gamma point; kpoint must be 1")

    structure_path = job_path / str(_scalar(inputs.get("stru_file", "STRU")))
    structure = AbacusSTRU.read(str(structure_path))
    if structure is None:
        raise ValueError(f"could not read structure: {structure_path}")

    orbital_dir = inputs.get("orbital_dir")
    shells_cache: dict[str, tuple[list[MoldenShell], float]] = {}
    atoms: list[MoldenAtom] = []
    basis_functions = 0
    max_error = 0.0
    for atom in structure.atoms:
        if not atom.orb:
            raise ValueError("every atom needs an orbital file to build a Molden file")
        filename = str(atom.orb)
        if filename not in shells_cache:
            parsed = read_nao_file(_resolve_orbital(job_path, orbital_dir, filename))
            shells_cache[filename] = fit_nao_to_cgto(parsed, nprim=gto_primitives)
        shells, error = shells_cache[filename]
        max_error = max(max_error, error)
        basis_functions += sum(2 * shell.l + 1 for shell in shells)
        number = atomic_numbers.get(atom.element or "")
        if number is None:
            raise ValueError(f"unknown element for atom {atom.label!r}")
        atoms.append(MoldenAtom(atom.element or atom.label, int(number), atom.coord, tuple(shells)))

    valences = _element_valences(structure, inputs, job_path)

    orbitals: list[MoldenOrbital] = []
    nbands = 0
    for ispin in range(1, nspin + 1):
        path = channels.get((ispin, kpoint))
        if path is None:
            raise FileNotFoundError(
                f"missing wavefunction for spin {ispin} at k-point {kpoint} in {output_dir}"
            )
        wfc, energies, occupations = read_wfc_nao_k_data(path)
        if wfc.shape[0] != basis_functions:
            raise ValueError(
                f"{path.name} has {wfc.shape[0]} orbitals but the structure has "
                f"{basis_functions} basis functions"
            )
        imaginary = float(np.max(np.abs(wfc.imag))) if wfc.size else 0.0
        magnitude = float(np.max(np.abs(wfc))) if wfc.size else 0.0
        if imaginary > 1e-6 * max(magnitude, 1.0):
            raise ValueError(
                f"{path.name} has complex coefficients; the Molden format needs a real "
                "Gamma-point-only run (gamma_only 1)"
            )
        coefficients = wfc.real
        spin = "Alpha" if ispin == 1 else "Beta"
        for band in range(coefficients.shape[1]):
            orbitals.append(
                MoldenOrbital(
                    energy=float(energies[band]) * RY_TO_EV / HARTREE_TO_EV,
                    spin=spin,
                    occupation=float(occupations[band]),
                    coefficients=coefficients[:, band],
                )
            )
        nbands = max(nbands, coefficients.shape[1])

    target = Path(output) if output is not None else Path("wfc.molden")
    if not target.is_absolute():
        target = job_path / target
    write_molden(target, structure.cell, atoms, valences, orbitals, atoms_unit=atoms_unit)

    return MoldenResult(
        job=str(job_path),
        output=str(target),
        gamma_only=stored_gamma,
        kpoint=kpoint,
        nkpoints=nkpoints,
        nspin=nspin,
        nbands=nbands,
        basis_functions=basis_functions,
        gto_primitives=gto_primitives,
        max_relative_error=max_error,
    )
