"""Mulliken, Loewdin and Wiberg population analysis for ABACUS LCAO jobs.

The analysis works from k-resolved density matrices and overlap matrices.  It
supports the two common ABACUS output layouts:

* develop output with ``dmk*_nao.txt`` and ``sk*_nao.txt``;
* LTS output with ``WFC_NAO_K*.txt`` and ``data-*-S``.

Mulliken populations and overlap populations are linear in the density matrix.
Loewdin populations and Wiberg bond orders use the symmetric orthogonalisation
``S(k)**0.5`` and are evaluated per k-point before the Brillouin-zone sum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from abacustools.data.charge import valence_electrons
from abacustools.data.mayer import (
    KPointTable,
    atom_ranges,
    calculate_density_matrix_k,
    develop_density_files,
    develop_overlap_file,
    exact_kpoint_weights,
    mesh_permutations,
    minimum_distance,
    read_develop_kpoints,
    read_develop_overlap,
    read_density_matrix,
    read_density_matrix_develop,
    read_kpoint_table,
    read_nao_file,
    read_overlap_matrix,
    read_wfc_nao_k_data,
    select_atom_pairs,
)
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU, periodic_lattice


class PopulationAnalysisError(RuntimeError):
    """Raised when a population analysis cannot be assembled."""


@dataclass(frozen=True)
class KPointDensity:
    """Density and overlap matrices of one k-point and its BZ weight."""

    index: int
    direct: tuple[float, float, float]
    weight: float
    overlap: np.ndarray
    densities: dict[int, np.ndarray]
    data_files: tuple[str, ...]


@dataclass(frozen=True)
class AtomPopulation:
    """Population of one atom in the selected spin channel."""

    index: int
    element: str
    population: float
    charge: Optional[float]
    magnetization: Optional[float]


@dataclass(frozen=True)
class BondOrder:
    """One atom-pair population or bond order."""

    atom1: int
    atom2: int
    element1: str
    element2: str
    distance: float
    value: float


@dataclass(frozen=True)
class PopulationAnalysis:
    """Complete result of a Mulliken, Loewdin or Wiberg analysis."""

    job: str
    method: str
    bond_quantity: str
    spin: str
    calculation: str
    nspin: int
    gamma_only: bool
    basis_functions: int
    kpoint_count: int
    occupancy_convention: str
    total_electrons: float
    expected_electrons: Optional[float]
    total_magnetization: Optional[float]
    atoms: tuple[AtomPopulation, ...]
    bonds: tuple[BondOrder, ...]
    data_files: tuple[str, ...]


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def _job_directory(job: str | Path) -> Path:
    path = Path(job).resolve()
    if not path.is_dir():
        raise PopulationAnalysisError(f"job directory does not exist: {path}")
    return path


def _output_directory(job: Path, inputs: dict[str, Any]) -> Path:
    suffix = str(_scalar(inputs.get("suffix", "ABACUS")))
    path = job / f"OUT.{suffix}"
    if not path.is_dir():
        raise PopulationAnalysisError(f"ABACUS output directory does not exist: {path}")
    return path


def _expected_electrons(
    inputs: dict[str, Any],
    structure: AbacusSTRU,
    job: Path,
) -> Optional[float]:
    """Return the electron count implied by INPUT or the pseudopotentials."""

    nelec = _as_float(_scalar(inputs.get("nelec", 0.0)))
    if nelec > 0:
        return nelec + _as_float(_scalar(inputs.get("nelec_delta", 0.0)))
    try:
        values = valence_electrons(structure, inputs.get("pseudo_dir"), job)
    except (OSError, ValueError, KeyError):
        return None
    return float(sum(values))


def _orbital_ranges(
    structure: AbacusSTRU,
    job: Path,
    inputs: dict[str, Any],
) -> tuple[list[tuple[int, int]], int]:
    nao_by_file = {}
    for index, atom in enumerate(structure.atoms, 1):
        if not atom.orb:
            raise PopulationAnalysisError(f"atom {index} has no numerical orbital file")
        if atom.orb not in nao_by_file:
            orbital_dir = inputs.get("orbital_dir")
            path = Path(str(atom.orb))
            if not path.is_absolute():
                base = Path(str(orbital_dir)) if orbital_dir else job
                path = base / path
            if not path.is_file():
                raise PopulationAnalysisError(f"numerical orbital file does not exist: {path}")
            nao_by_file[atom.orb] = read_nao_file(path)
    return atom_ranges(structure, nao_by_file)


def _weights_for_table(table: KPointTable, nspin: int) -> tuple[float, ...]:
    """Return exact BZ weights for the k-points of a run."""

    attempts = tuple(dict.fromkeys((2.0 / nspin, 1.0, 2.0)))
    last_error: Optional[Exception] = None
    for scale in attempts:
        try:
            return exact_kpoint_weights(table, printed_scale=scale)
        except ValueError as error:
            last_error = error
    raise PopulationAnalysisError(
        "the k-point weights cannot be reconstructed from the output; "
        "rerun with symmetry=0 or symmetry=-1"
    ) from last_error


def _load_develop_density_data(output: Path, nspin: int) -> list[KPointDensity]:
    entries = develop_density_files(output)
    if not entries:
        return []
    kpoint_table = read_develop_kpoints(entries)
    kpoint_file = output / "kpoints"
    if kpoint_file.is_file():
        table_with_mesh = read_kpoint_table(kpoint_file)
        if len(table_with_mesh.ibz) == len(kpoint_table.ibz):
            kpoint_table = table_with_mesh
    weights = _weights_for_table(kpoint_table, nspin)
    indices = sorted({ik for ik, _spin, _path in entries})
    if len(indices) != len(kpoint_table.ibz):
        raise PopulationAnalysisError("the density-matrix files and the recorded k-points disagree")

    result: list[KPointDensity] = []
    expected_spins = {1} if nspin == 1 else {1, 2}
    for position, ik in enumerate(indices):
        kpoint = kpoint_table.ibz[position]
        overlap_path = develop_overlap_file(output, ik)
        if not overlap_path.is_file():
            raise PopulationAnalysisError(f"overlap matrix does not exist: {overlap_path}")
        overlap = read_develop_overlap(overlap_path, kpoint=kpoint.direct)
        densities: dict[int, np.ndarray] = {}
        files: list[str] = [str(overlap_path)]
        for entry_ik, spin, path in entries:
            if entry_ik != ik:
                continue
            density = read_density_matrix_develop(path)
            if density.shape != overlap.shape:
                raise PopulationAnalysisError(
                    f"{path}: density-matrix and overlap dimensions do not match"
                )
            densities[spin] = density
            files.append(str(path))
        if set(densities) != expected_spins:
            raise PopulationAnalysisError(
                f"k-point {ik}: found spin channels {sorted(densities)}; "
                f"expected {sorted(expected_spins)} for nspin={nspin}"
            )
        result.append(
            KPointDensity(
                index=ik,
                direct=tuple(float(value) for value in kpoint.direct),
                weight=float(weights[position]),
                overlap=overlap,
                densities=densities,
                data_files=tuple(files),
            )
        )
    return result


def _lts_wfc_paths(output: Path) -> list[Path]:
    paths = []
    for path in output.glob("WFC_NAO_K*.txt"):
        match = re.search(r"K(\d+)", path.name)
        if match is not None:
            paths.append((int(match.group(1)), path.name, path))
    return [path for _index, _name, path in sorted(paths)]


def _load_lts_wfc_data(output: Path, nspin: int) -> list[KPointDensity]:
    paths = _lts_wfc_paths(output)
    if not paths:
        return []
    table = read_kpoint_table(output / "kpoints")
    weights = _weights_for_table(table, nspin)
    count = len(weights)
    if nspin == 1:
        if len(paths) != count:
            raise PopulationAnalysisError(
                f"found {len(paths)} wavefunction files for {count} k-points"
            )
        channels = {1: paths}
    else:
        if len(paths) != 2 * count:
            raise PopulationAnalysisError(
                f"found {len(paths)} wavefunction files for {count} k-points "
                "and nspin=2; expected twice as many"
            )
        channels = {1: paths[:count], 2: paths[count:]}

    result: list[KPointDensity] = []
    for position, kpoint in enumerate(table.ibz):
        overlap_path = output / f"data-{position}-S"
        if not overlap_path.is_file():
            raise PopulationAnalysisError(f"overlap matrix does not exist: {overlap_path}")
        overlap = read_overlap_matrix(overlap_path)
        densities: dict[int, np.ndarray] = {}
        files: list[str] = [str(overlap_path)]
        for spin, spin_paths in channels.items():
            coefficients, _energies, occupations = read_wfc_nao_k_data(spin_paths[position])
            density = calculate_density_matrix_k(coefficients, occupations)
            if density.shape != overlap.shape:
                raise PopulationAnalysisError(
                    f"wavefunction and overlap dimensions do not match for k-point {position + 1}"
                )
            densities[spin] = density
            files.append(str(spin_paths[position]))
        result.append(
            KPointDensity(
                index=position + 1,
                direct=tuple(float(value) for value in kpoint.direct),
                weight=float(weights[position]),
                overlap=overlap,
                densities=densities,
                data_files=tuple(files),
            )
        )
    return result


def _load_lts_gamma_data(output: Path, nspin: int) -> list[KPointDensity]:
    overlap_path = output / "data-0-S"
    density_path = output / "SPIN1_DM"
    if not overlap_path.is_file() or not density_path.is_file():
        return []
    overlap = read_overlap_matrix(overlap_path)
    densities = {1: read_density_matrix(density_path)}
    files = [str(overlap_path), str(density_path)]
    if nspin == 2:
        down_path = output / "SPIN2_DM"
        if not down_path.is_file():
            raise PopulationAnalysisError(f"spin-down density matrix does not exist: {down_path}")
        densities[2] = read_density_matrix(down_path)
        files.append(str(down_path))
    for spin, density in densities.items():
        if density.shape != overlap.shape:
            raise PopulationAnalysisError(
                f"spin {spin}: density-matrix and overlap dimensions do not match"
            )
    return [
        KPointDensity(
            index=1,
            direct=(0.0, 0.0, 0.0),
            weight=1.0,
            overlap=overlap,
            densities=densities,
            data_files=tuple(files),
        )
    ]


def _load_density_data(output: Path, nspin: int, gamma_only: bool) -> list[KPointDensity]:
    kpoints = _load_develop_density_data(output, nspin)
    if kpoints:
        return kpoints
    if gamma_only:
        kpoints = _load_lts_gamma_data(output, nspin)
        if kpoints:
            return kpoints
    kpoints = _load_lts_wfc_data(output, nspin)
    if kpoints:
        return kpoints
    raise PopulationAnalysisError(
        "no k-resolved density matrix or wavefunction output was found; "
        "write out_dmk=1 for the develop layout, or out_wfc_lcao=1 for the LTS "
        "layout, before running the population analysis"
    )


def _trace_electrons(
    kpoints: Sequence[KPointDensity],
    *,
    divide_weight: bool,
    scale: float,
) -> float:
    total = 0.0
    for kpoint in kpoints:
        factor = scale / kpoint.weight if divide_weight else scale
        for density in kpoint.densities.values():
            total += kpoint.weight * float(np.real(np.trace((factor * density) @ kpoint.overlap)))
    return total


def _normalise_density_data(
    kpoints: Sequence[KPointDensity],
    expected: Optional[float],
) -> tuple[list[KPointDensity], str, float]:
    """Normalise the file occupation convention and return the electron count."""

    if expected is None:
        return (
            list(kpoints),
            "file values (no electron-count check)",
            _trace_electrons(kpoints, divide_weight=False, scale=1.0),
        )

    candidates = []
    for divide_weight in (False, True):
        for scale in (0.5, 1.0, 2.0):
            electrons = _trace_electrons(
                kpoints,
                divide_weight=divide_weight,
                scale=scale,
            )
            candidates.append((abs(electrons - expected), divide_weight, scale, electrons))
    error, divide_weight, scale, electrons = min(candidates, key=lambda item: item[0])
    tolerance = 1e-4 * max(1.0, abs(expected)) + 1e-6
    if error > tolerance:
        details = ", ".join(
            f"{'D/w' if divide else 'D'}*{factor:g}={value:.6f}"
            for _err, divide, factor, value in sorted(candidates)[:6]
        )
        raise PopulationAnalysisError(
            f"cannot match the expected {expected:g} electrons; tried {details}. "
            "Check nelec/nelec_delta and the occupation convention of the output."
        )

    normalised = []
    for kpoint in kpoints:
        factor = scale / kpoint.weight if divide_weight else scale
        normalised.append(
            KPointDensity(
                index=kpoint.index,
                direct=kpoint.direct,
                weight=kpoint.weight,
                overlap=kpoint.overlap,
                densities={spin: factor * density for spin, density in kpoint.densities.items()},
                data_files=kpoint.data_files,
            )
        )
    convention = f"{'divide' if divide_weight else 'keep'} k weight, spin scale {scale:g}"
    return normalised, convention, electrons


def _sqrtm_hermitian(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian square root of a positive-semidefinite matrix."""

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise PopulationAnalysisError("the overlap matrix is not positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T


def _mulliken_atom_values(
    density: np.ndarray,
    overlap: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    *,
    product: Optional[np.ndarray] = None,
) -> np.ndarray:
    if product is None:
        product = density @ overlap
    return np.asarray(
        [float(np.real(np.trace(product[start:stop, start:stop]))) for start, stop in ranges]
    )


def _mulliken_overlap_value(
    density: np.ndarray,
    overlap: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    first: int,
    second: int,
) -> float:
    a0, a1 = ranges[first]
    b0, b1 = ranges[second]
    density_block = density[a0:a1, b0:b1]
    overlap_block = overlap[b0:b1, a0:a1]
    return float(2.0 * np.real(np.sum(density_block * overlap_block.T)))


def _mulliken_bond_order_value(
    density: np.ndarray,
    overlap: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    first: int,
    second: int,
    *,
    product: Optional[np.ndarray] = None,
) -> float:
    """Return the quadratic Mulliken/Mayer bond order of one atom pair."""

    a0, a1 = ranges[first]
    b0, b1 = ranges[second]
    if product is None:
        product = density @ overlap
    forward = product[a0:a1, b0:b1]
    backward = product[b0:b1, a0:a1]
    return float(np.real(np.sum(forward * backward.T)))


def _lowdin_values(
    density: np.ndarray,
    overlap: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    root = _sqrtm_hermitian(overlap)
    product = root @ density @ root
    atoms = np.asarray(
        [float(np.real(np.trace(product[start:stop, start:stop]))) for start, stop in ranges]
    )
    values = {}
    for first, second in pairs:
        a0, a1 = ranges[first]
        b0, b1 = ranges[second]
        block = product[a0:a1, b0:b1]
        values[(first, second)] = float(np.sum(np.abs(block) ** 2))
    return atoms, values


def _stars_for_table(
    table: KPointTable,
    structure: AbacusSTRU,
) -> dict[int, list[tuple[int, ...]]]:
    if not table.mesh:
        return {}
    permutations = mesh_permutations(table, structure)
    stars: dict[int, list[tuple[int, ...]]] = {}
    for (_point, index), permutation in zip(table.mesh, permutations):
        stars.setdefault(index, []).append(permutation)
    return stars


def _expand_atom_values(
    kpoints: Sequence[KPointDensity],
    stars: dict[int, list[tuple[int, ...]]],
    values: dict[int, np.ndarray],
    natoms: int,
) -> np.ndarray:
    result = np.zeros(natoms, dtype=float)
    identity = tuple(range(natoms))
    for kpoint in kpoints:
        current = values[kpoint.index]
        members = stars.get(kpoint.index, [identity])
        for atom in range(natoms):
            result[atom] += kpoint.weight * float(
                np.mean([current[permutation[atom]] for permutation in members])
            )
    return result


def _expand_pair_values(
    kpoints: Sequence[KPointDensity],
    stars: dict[int, list[tuple[int, ...]]],
    values: dict[int, dict[tuple[int, int], float]],
    pairs: Sequence[tuple[int, int]],
    natoms: int,
) -> dict[tuple[int, int], float]:
    result = {pair: 0.0 for pair in pairs}
    identity = tuple(range(natoms))
    for kpoint in kpoints:
        current = values[kpoint.index]
        members = stars.get(kpoint.index, [identity])
        for pair in pairs:
            contributions = []
            for permutation in members:
                mapped = tuple(sorted((permutation[pair[0]], permutation[pair[1]])))
                contributions.append(current[mapped])
            result[pair] += kpoint.weight * float(np.mean(contributions))
    return result


def _needed_pairs(
    kpoints: Sequence[KPointDensity],
    stars: dict[int, list[tuple[int, ...]]],
    pairs: Sequence[tuple[int, int]],
    natoms: int,
) -> dict[int, list[tuple[int, int]]]:
    identity = tuple(range(natoms))
    needed: dict[int, set[tuple[int, int]]] = {}
    for kpoint in kpoints:
        members = stars.get(kpoint.index, [identity])
        values = needed.setdefault(kpoint.index, set())
        for first, second in pairs:
            for permutation in members:
                values.add(tuple(sorted((permutation[first], permutation[second]))))
    return {index: sorted(value) for index, value in needed.items()}


def _combine_spin(
    up: np.ndarray,
    down: Optional[np.ndarray],
    spin: str,
) -> np.ndarray:
    if spin == "up":
        if down is None:
            raise PopulationAnalysisError("--spin up requires nspin=2")
        return up
    if spin == "down":
        if down is None:
            raise PopulationAnalysisError("--spin down requires nspin=2")
        return down
    if spin == "difference":
        if down is None:
            raise PopulationAnalysisError("--spin difference requires nspin=2")
        return up - down
    return up if down is None else up + down


def analyze_population(
    job: str | Path,
    *,
    method: str = "mulliken",
    spin: str = "total",
    cutoff: Optional[float] = None,
    pairs: Optional[str] = None,
    pairs_file: Optional[str | Path] = None,
) -> PopulationAnalysis:
    """Analyze Mulliken, Loewdin or Wiberg populations and bond orders.

    Args:
        job: ABACUS LCAO job directory.
        method: ``mulliken`` for Mulliken populations and the quadratic
            Mulliken/Mayer bond order; ``mulliken-overlap`` for the linear
            Mulliken overlap population; ``lowdin``/``wiberg`` for Loewdin
            populations and Wiberg bond orders.
        spin: ``total``, ``up``, ``down`` or ``difference``.  The latter three
            require an nspin=2 calculation.
        cutoff: Only analyze atom pairs within this distance in Angstrom.
        pairs: Comma-separated one-based atom pairs such as ``1-2,1-3``.
        pairs_file: File holding one one-based atom pair per line.

    Returns:
        The population and bond-order report.

    Raises:
        PopulationAnalysisError: If the output layout or occupation convention
            cannot be interpreted.
    """

    method = method.lower()
    spin = spin.lower()
    if method not in {"mulliken", "mulliken-overlap", "lowdin", "wiberg"}:
        raise PopulationAnalysisError(f"unsupported population method: {method}")
    if spin not in {"total", "up", "down", "difference"}:
        raise PopulationAnalysisError(f"unsupported spin selection: {spin}")

    job_path = _job_directory(job)
    inputs = ReadInput(str(job_path / "INPUT"))
    if not inputs:
        raise PopulationAnalysisError(f"could not read INPUT in {job_path}")
    calculation = str(inputs.get("calculation", "scf")).lower()
    nspin = int(_scalar(inputs.get("nspin", 1)))
    if nspin not in (1, 2):
        raise PopulationAnalysisError(f"population analysis supports nspin=1 or 2, got {nspin}")
    if str(_scalar(inputs.get("basis_type", "pw"))).lower() != "lcao":
        raise PopulationAnalysisError("population analysis requires basis_type=lcao")
    gamma_only = bool(int(_scalar(inputs.get("gamma_only", 0)) or 0))
    if spin in {"up", "down", "difference"} and nspin == 1:
        raise PopulationAnalysisError(f"--spin {spin} requires nspin=2")

    output = _output_directory(job_path, inputs)
    structure_path = job_path / str(_scalar(inputs.get("stru_file", "STRU")))
    if not structure_path.is_file():
        raise PopulationAnalysisError(f"structure file does not exist: {structure_path}")
    structure = AbacusSTRU.read(str(structure_path))
    if structure is None:
        raise PopulationAnalysisError(f"cannot read structure: {structure_path}")

    ranges, basis_functions = _orbital_ranges(structure, job_path, inputs)
    selected_pairs = select_atom_pairs(
        structure,
        cutoff=cutoff,
        pairs_str=pairs,
        pairs_file=pairs_file,
    )
    kpoints = _load_density_data(output, nspin, gamma_only)
    for kpoint in kpoints:
        if kpoint.overlap.shape != (basis_functions, basis_functions):
            raise PopulationAnalysisError(
                f"overlap dimension at k-point {kpoint.index} does not match "
                f"the NAO basis size {basis_functions}"
            )

    expected = _expected_electrons(inputs, structure, job_path)
    kpoints, convention, electrons = _normalise_density_data(kpoints, expected)

    table = None
    if not gamma_only:
        if (output / "kpoints").is_file():
            table = read_kpoint_table(output / "kpoints")
        else:
            entries = develop_density_files(output)
            if entries:
                table = read_develop_kpoints(entries)
    stars = _stars_for_table(table, structure) if table is not None else {}

    atom_values_up = {kpoint.index: [] for kpoint in kpoints}
    pair_values_up = {kpoint.index: {} for kpoint in kpoints}
    atom_values_down = {kpoint.index: [] for kpoint in kpoints}
    pair_values_down = {kpoint.index: {} for kpoint in kpoints}
    needed = _needed_pairs(kpoints, stars, selected_pairs, structure.natoms)

    for kpoint in kpoints:
        density_up = kpoint.densities[1]
        if method in {"mulliken", "mulliken-overlap"}:
            product_up = density_up @ kpoint.overlap
            atom_values_up[kpoint.index] = _mulliken_atom_values(
                density_up, kpoint.overlap, ranges, product=product_up
            )
            if method == "mulliken":
                pair_values_up[kpoint.index] = {
                    pair: _mulliken_bond_order_value(
                        density_up,
                        kpoint.overlap,
                        ranges,
                        pair[0],
                        pair[1],
                        product=product_up,
                    )
                    for pair in needed[kpoint.index]
                }
            else:
                pair_values_up[kpoint.index] = {
                    pair: _mulliken_overlap_value(
                        density_up, kpoint.overlap, ranges, pair[0], pair[1]
                    )
                    for pair in needed[kpoint.index]
                }
        else:
            atom_values_up[kpoint.index], values = _lowdin_values(
                density_up, kpoint.overlap, ranges, needed[kpoint.index]
            )
            pair_values_up[kpoint.index] = values

        if nspin == 2:
            density_down = kpoint.densities[2]
            if method in {"mulliken", "mulliken-overlap"}:
                product_down = density_down @ kpoint.overlap
                atom_values_down[kpoint.index] = _mulliken_atom_values(
                    density_down, kpoint.overlap, ranges, product=product_down
                )
                if method == "mulliken":
                    pair_values_down[kpoint.index] = {
                        pair: _mulliken_bond_order_value(
                            density_down,
                            kpoint.overlap,
                            ranges,
                            pair[0],
                            pair[1],
                            product=product_down,
                        )
                        for pair in needed[kpoint.index]
                    }
                else:
                    pair_values_down[kpoint.index] = {
                        pair: _mulliken_overlap_value(
                            density_down, kpoint.overlap, ranges, pair[0], pair[1]
                        )
                        for pair in needed[kpoint.index]
                    }
            else:
                atom_values_down[kpoint.index], values = _lowdin_values(
                    density_down, kpoint.overlap, ranges, needed[kpoint.index]
                )
                pair_values_down[kpoint.index] = values

    atoms_up = _expand_atom_values(kpoints, stars, atom_values_up, structure.natoms)
    atoms_down = (
        _expand_atom_values(kpoints, stars, atom_values_down, structure.natoms)
        if nspin == 2
        else None
    )
    pairs_up = _expand_pair_values(kpoints, stars, pair_values_up, selected_pairs, structure.natoms)
    pairs_down = (
        _expand_pair_values(kpoints, stars, pair_values_down, selected_pairs, structure.natoms)
        if nspin == 2
        else None
    )

    atom_population = _combine_spin(atoms_up, atoms_down, spin)
    pair_population = {
        pair: float(
            _combine_spin(
                np.asarray([pairs_up[pair]]),
                None if pairs_down is None else np.asarray([pairs_down[pair]]),
                spin,
            )[0]
        )
        for pair in selected_pairs
    }

    valences = None
    try:
        valences = valence_electrons(structure, inputs.get("pseudo_dir"), job_path)
    except (OSError, ValueError, KeyError):
        valences = None

    lattice = periodic_lattice(structure)
    fractions = structure.coords_direct
    atom_records = []
    for index, atom in enumerate(structure.atoms):
        charge = None
        if spin == "total" and valences is not None:
            charge = float(valences[index] - atom_population[index])
        magnetization = None
        if nspin == 2:
            if spin in {"total", "difference"}:
                magnetization = float(atoms_up[index] - atoms_down[index])
        atom_records.append(
            AtomPopulation(
                index=index + 1,
                element=str(atom.element or atom.label),
                population=float(atom_population[index]),
                charge=charge,
                magnetization=magnetization,
            )
        )

    bond_records = []
    for first, second in selected_pairs:
        bond_records.append(
            BondOrder(
                atom1=first + 1,
                atom2=second + 1,
                element1=str(structure.atoms[first].element or structure.atoms[first].label),
                element2=str(structure.atoms[second].element or structure.atoms[second].label),
                distance=float(minimum_distance(lattice, fractions[first], fractions[second])),
                value=float(pair_population[(first, second)]),
            )
        )

    total_magnetization = None
    if atoms_down is not None:
        total_magnetization = float(np.sum(atoms_up - atoms_down))
    data_files = tuple(dict.fromkeys(path for kpoint in kpoints for path in kpoint.data_files))
    return PopulationAnalysis(
        job=str(job_path),
        method=method,
        bond_quantity=(
            "mulliken_bond_order"
            if method == "mulliken"
            else (
                "mulliken_overlap_population"
                if method == "mulliken-overlap"
                else "wiberg_bond_order"
            )
        ),
        spin=spin,
        calculation=calculation,
        nspin=nspin,
        gamma_only=gamma_only,
        basis_functions=basis_functions,
        kpoint_count=len(kpoints),
        occupancy_convention=convention,
        total_electrons=float(electrons),
        expected_electrons=None if expected is None else float(expected),
        total_magnetization=total_magnetization,
        atoms=tuple(atom_records),
        bonds=tuple(bond_records),
        data_files=data_files,
    )
