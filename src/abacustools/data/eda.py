"""Readers for pEDA-related ABACUS results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from abacustools.data.abacus_result import get_result_from_job
from abacustools.data.mayer import read_overlap_matrix, read_overlap_matrix_develop
from abacustools.io.abacus import ReadInput


_DEVELOP_H_FILE = re.compile(
    r"^hk(?:s(?P<is>\d+))?(?:g(?P<step>\d+))?_nao\.txt$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GammaFixedDensityResult:
    """Energy and Hamiltonian of one Gamma-point fixed-density calculation.

    Attributes:
        energy_ev: Total energy printed by ABACUS, in eV.
        vdw_energy_ev: DFT-D energy when present, in eV.
        electronic_energy_ev: Total energy without the separate DFT-D term.
        converged: ABACUS SCF convergence flag.  A one-step fixed-density job
            may legitimately report ``False`` here.
        normal_end: Whether ABACUS reached its normal job end.
        nspin: Number of spin channels.
        hamiltonians: One H(k) matrix per spin channel, in Ry.
        data_files: Hamiltonian files used for the result.
    """

    job: str
    energy_ev: float
    vdw_energy_ev: Optional[float]
    electronic_energy_ev: float
    converged: Optional[bool]
    normal_end: Optional[bool]
    nspin: int
    hamiltonians: dict[int, np.ndarray]
    data_files: tuple[str, ...]


def _output_directory(job: Path, inputs: dict) -> Path:
    suffix = inputs.get("suffix", "ABACUS")
    if isinstance(suffix, (list, tuple)):
        suffix = suffix[0]
    path = job / f"OUT.{suffix}"
    if not path.is_dir():
        raise FileNotFoundError(f"ABACUS output directory does not exist: {path}")
    return path


def _find_develop_hamiltonians(output: Path, nspin: int) -> dict[int, Path]:
    candidates: dict[int, tuple[int, Path]] = {}
    for path in output.iterdir():
        if not path.is_file():
            continue
        match = _DEVELOP_H_FILE.match(path.name)
        if match is None:
            continue
        spin = int(match.group("is")) if match.group("is") else 1
        step = int(match.group("step")) if match.group("step") else 0
        previous = candidates.get(spin)
        if previous is None or step > previous[0]:
            candidates[spin] = (step, path)
    expected = {1} if nspin == 1 else {1, 2}
    if set(candidates) == expected:
        return {spin: candidates[spin][1] for spin in expected}
    if not candidates:
        return {}
    raise ValueError(
        f"Hamiltonian files in {output} do not match nspin={nspin}; "
        f"found spins {sorted(candidates)}"
    )


def read_gamma_fixed_density_result(
    job: str | Path,
    *,
    version: Optional[str] = None,
) -> GammaFixedDensityResult:
    """Read energy and H(k) from a Gamma-point fixed-density ABACUS run.

    Args:
        job: ABACUS job directory produced by ``prepare_gamma_state_job``.
        version: Optional ABACUS log version; ``None`` uses automatic detection.

    Returns:
        The parsed energy, convergence metadata and spin-resolved H(k).

    Raises:
        FileNotFoundError: If INPUT, OUT.* or a Hamiltonian file is missing.
        ValueError: If the job is not a Gamma LCAO run or the energy is absent.
    """

    job_path = Path(job).resolve()
    input_path = job_path / "INPUT"
    if not input_path.is_file():
        raise FileNotFoundError(f"INPUT file does not exist: {input_path}")
    inputs = ReadInput(str(input_path))
    if str(inputs.get("basis_type", "pw")).lower() != "lcao":
        raise ValueError("fixed-density pEDA results require basis_type=lcao")
    gamma_only = inputs.get("gamma_only", 0)
    if isinstance(gamma_only, (list, tuple)):
        gamma_only = gamma_only[0]
    if not bool(int(gamma_only or 0)):
        raise ValueError("fixed-density pEDA result reader only supports gamma_only=1")
    nspin = int(inputs.get("nspin", 1))
    if nspin not in (1, 2):
        raise ValueError(f"fixed-density pEDA supports nspin=1 or 2, got {nspin}")

    output = _output_directory(job_path, inputs)
    files = _find_develop_hamiltonians(output, nspin)
    reader = read_overlap_matrix_develop
    if not files:
        lts = output / "data-0-H"
        if nspin == 1 and lts.is_file():
            files = {1: lts}
            reader = read_overlap_matrix
        else:
            raise FileNotFoundError(
                f"no Gamma Hamiltonian output found in {output}: expected "
                "hk_nao.txt/hks*_nao.txt or data-0-H"
            )

    result = get_result_from_job(
        str(job_path),
        ["energy", "vdw_energy", "converged", "normal_end"],
        version=version,
    )
    energy = result.get("energy")
    if energy is None or not np.isfinite(float(energy)):
        raise ValueError(f"no total energy found for fixed-density job: {job_path}")
    vdw = result.get("vdw_energy")
    vdw_value = None if vdw is None else float(vdw)
    electronic = float(energy) - (0.0 if vdw_value is None else vdw_value)

    hamiltonians = {
        spin: np.asarray(reader(path), dtype=np.complex128) for spin, path in sorted(files.items())
    }
    return GammaFixedDensityResult(
        job=str(job_path),
        energy_ev=float(energy),
        vdw_energy_ev=vdw_value,
        electronic_energy_ev=electronic,
        converged=result.get("converged"),
        normal_end=result.get("normal_end"),
        nspin=nspin,
        hamiltonians=hamiltonians,
        data_files=tuple(str(path) for _spin, path in sorted(files.items())),
    )


@dataclass(frozen=True)
class PEDAEnergyTerms:
    """Energy terms of one pEDA analysis, all in eV."""

    preparation: float
    electrostatic: float
    pauli: float
    orbital: float
    dispersion: float
    interaction_electronic: float
    interaction: float
    bond: float
    electrostatic_fraction: Optional[float]
    orbital_fraction: Optional[float]
    dispersion_fraction: Optional[float]
    unit: str = "eV"


def _finite_energy(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite energy")
    return result


def compute_peda_energy_terms(
    *,
    e_ab: float,
    e_a: float,
    e_b: float,
    e_a0: float,
    e_b0: float,
    e0: float,
    delta_e_elstat: float,
    e_vdw_ab: Optional[float] = None,
    e_vdw_a: Optional[float] = None,
    e_vdw_b: Optional[float] = None,
) -> PEDAEnergyTerms:
    """Assemble the pEDA energy terms from electronic and dispersion energies.

    ``e_ab``, ``e_a``, ``e_b``, ``e_a0``, ``e_b0`` and ``e0`` must be
    electronic energies without the separate DFT-D correction.  The prepared
    fragment energies are ``e_a``/``e_b`` at the complex geometry, while
    ``e_a0``/``e_b0`` are the isolated relaxed fragments.  This gives the
    positive preparation cost used by pEDA:

    ``ΔE_prep = E_A + E_B - E_A0 - E_B0``

    Args:
        e_ab: Full-system electronic energy.
        e_a: Prepared fragment A energy.
        e_b: Prepared fragment B energy.
        e_a0: Isolated relaxed fragment A energy.
        e_b0: Isolated relaxed fragment B energy.
        e0: Energy of the orthogonalized fragment intermediate.
        delta_e_elstat: Fragment electrostatic cross term.
        e_vdw_ab: Dispersion energy of the full system.
        e_vdw_a: Dispersion energy of fragment A.
        e_vdw_b: Dispersion energy of fragment B.

    Returns:
        The pEDA energy terms and attractive-term fractions.

    Raises:
        ValueError: If an energy is non-finite or the dispersion inputs are
            only partially supplied.
    """

    energies = {
        "e_ab": _finite_energy("e_ab", e_ab),
        "e_a": _finite_energy("e_a", e_a),
        "e_b": _finite_energy("e_b", e_b),
        "e_a0": _finite_energy("e_a0", e_a0),
        "e_b0": _finite_energy("e_b0", e_b0),
        "e0": _finite_energy("e0", e0),
        "delta_e_elstat": _finite_energy("delta_e_elstat", delta_e_elstat),
    }
    dispersion_values = (e_vdw_ab, e_vdw_a, e_vdw_b)
    if all(value is None for value in dispersion_values):
        dispersion = 0.0
    elif any(value is None for value in dispersion_values):
        raise ValueError("e_vdw_ab, e_vdw_a and e_vdw_b must be supplied together")
    else:
        dispersion = (
            _finite_energy("e_vdw_ab", e_vdw_ab)
            - _finite_energy("e_vdw_a", e_vdw_a)
            - _finite_energy("e_vdw_b", e_vdw_b)
        )

    preparation = energies["e_a"] + energies["e_b"] - energies["e_a0"] - energies["e_b0"]
    electrostatic = energies["delta_e_elstat"]
    interaction_electronic = energies["e0"] - energies["e_a"] - energies["e_b"]
    pauli = interaction_electronic - electrostatic
    orbital = energies["e_ab"] - energies["e0"]
    interaction = interaction_electronic + dispersion
    bond = preparation + interaction

    attractive = electrostatic + orbital
    if abs(attractive) > 1e-12:
        electrostatic_fraction = electrostatic / attractive
        orbital_fraction = orbital / attractive
    else:
        electrostatic_fraction = None
        orbital_fraction = None
    dispersion_fraction = dispersion / interaction if abs(interaction) > 1e-12 else None

    return PEDAEnergyTerms(
        preparation=float(preparation),
        electrostatic=float(electrostatic),
        pauli=float(pauli),
        orbital=float(orbital),
        dispersion=float(dispersion),
        interaction_electronic=float(interaction_electronic),
        interaction=float(interaction),
        bond=float(bond),
        electrostatic_fraction=(
            None if electrostatic_fraction is None else float(electrostatic_fraction)
        ),
        orbital_fraction=None if orbital_fraction is None else float(orbital_fraction),
        dispersion_fraction=(None if dispersion_fraction is None else float(dispersion_fraction)),
    )
