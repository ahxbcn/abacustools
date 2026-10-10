"""Mayer bond-order analysis for ABACUS LCAO calculations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from abacustools.core.constant import RY_TO_EV
from abacustools.data.symmetry import space_group_operations
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU, periodic_lattice

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_COMPLEX_TOKEN = re.compile(rf"^\(({_FLOAT}),({_FLOAT})\)$")
_ORBITAL_HEADER = re.compile(r"^\s*Type\s+L\s+N\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class NAOOrbital:
    l: int
    n: int
    values: np.ndarray


@dataclass(frozen=True)
class NAOData:
    element: str
    energy_cutoff: float
    radius: float
    lmax: int
    orbitals_per_l: tuple[int, ...]
    mesh: int
    dr: float
    orbitals: tuple[NAOOrbital, ...]


@dataclass(frozen=True)
class MayerPair:
    atom1: int
    atom2: int
    element1: str
    element2: str
    distance: float
    bond_order: float


@dataclass(frozen=True)
class MayerAnalysis:
    job: str
    calculation: str
    nspin: int
    gamma_only: bool
    basis_functions: int
    output_directory: str
    data_files: tuple[str, ...]
    pairs: tuple[MayerPair, ...]


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _required_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {description}: {path}")
    return path


def _required_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Cannot find {description}: {path}")
    return path


def _parse_summary_value(lines: list[str], label: str, path: Path) -> str:
    pattern = re.compile(rf"^\s*{re.escape(label)}\s+(.+?)\s*$", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    raise ValueError(f"{path}: missing '{label}' in NAO summary")


def read_nao_file(nao_file: str | Path) -> NAOData:
    """Read an ABACUS numerical-orbital file with format validation."""

    path = Path(nao_file)
    _required_file(path, "NAO file")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        summary_end = next(i for i, line in enumerate(lines) if "SUMMARY  END" in line)
    except StopIteration as exc:
        raise ValueError(f"{path}: missing 'SUMMARY  END'") from exc
    element = _parse_summary_value(lines[:summary_end], "Element", path)
    energy_cutoff = _number(_parse_summary_value(lines[:summary_end], "Energy Cutoff(Ry)", path))
    radius = _number(_parse_summary_value(lines[:summary_end], "Radius Cutoff(a.u.)", path))
    lmax = int(_parse_summary_value(lines[:summary_end], "Lmax", path))

    orbitals_per_l: list[int] = []
    orbital_pattern = re.compile(r"^\s*Number of ([SPDF])orbital--?>\s+(\d+)\s*$", re.IGNORECASE)
    for line in lines[:summary_end]:
        match = orbital_pattern.match(line)
        if match:
            l = "spdf".index(match.group(1).lower())
            while len(orbitals_per_l) <= l:
                orbitals_per_l.append(0)
            orbitals_per_l[l] = int(match.group(2))
    if len(orbitals_per_l) != lmax + 1:
        raise ValueError(f"{path}: expected orbital counts for l=0..{lmax}, got {orbitals_per_l}")

    mesh = int(_parse_summary_value(lines[summary_end + 1 :], "Mesh", path))
    dr = _number(_parse_summary_value(lines[summary_end + 1 :], "dr", path))
    if mesh <= 0 or dr <= 0:
        raise ValueError(f"{path}: mesh and dr must be positive")

    headers = [i for i, line in enumerate(lines[summary_end + 1 :], summary_end + 1) if _ORBITAL_HEADER.match(line)]
    expected_orbitals = sum(orbitals_per_l)
    if len(headers) != expected_orbitals:
        raise ValueError(f"{path}: summary declares {expected_orbitals} orbitals, found {len(headers)}")

    orbitals: list[NAOOrbital] = []
    for index, header in enumerate(headers):
        if header + 1 >= len(lines):
            raise ValueError(f"{path}: missing orbital descriptor after line {header + 1}")
        descriptor = lines[header + 1].split()
        if len(descriptor) < 3:
            raise ValueError(f"{path}: invalid orbital descriptor on line {header + 2}")
        try:
            l, n = int(descriptor[1]), int(descriptor[2])
        except ValueError as exc:
            raise ValueError(f"{path}: invalid orbital descriptor on line {header + 2}") from exc
        end = headers[index + 1] if index + 1 < len(headers) else len(lines)
        try:
            values = [_number(token) for line in lines[header + 2 : end] for token in line.split()]
        except ValueError as exc:
            raise ValueError(f"{path}: invalid radial value after line {header + 2}") from exc
        if len(values) != mesh:
            raise ValueError(f"{path}: orbital l={l}, n={n} has {len(values)} radial values; expected {mesh}")
        orbitals.append(NAOOrbital(l, n, np.asarray(values, dtype=float)))
    return NAOData(element, energy_cutoff, radius, lmax, tuple(orbitals_per_l), mesh, dr, tuple(orbitals))


def get_nao_basis_num(nao: NAOData) -> int:
    """Return the number of Cartesian basis functions represented by an NAO."""

    return sum((2 * l + 1) * count for l, count in enumerate(nao.orbitals_per_l))


def _parse_matrix_token(token: str) -> complex:
    match = _COMPLEX_TOKEN.match(token)
    if match:
        return complex(_number(match.group(1)), _number(match.group(2)))
    try:
        return complex(_number(token), 0.0)
    except ValueError as exc:
        raise ValueError(f"invalid matrix value {token!r}") from exc


def _maybe_matrix_value(token: str) -> Optional[complex]:
    """Return a matrix value, or ``None`` when the token is not a number.

    The develop version writes complex pairs ``(real,imag)`` for multi-k runs
    but plain real numbers for gamma-only runs, so both spellings are accepted.
    """
    try:
        return _parse_matrix_token(token)
    except ValueError:
        return None


def read_overlap_matrix(ovlp_mat_file: str | Path) -> np.ndarray:
    """Read ABACUS ``data-*-S`` upper-triangular matrix output."""

    path = Path(ovlp_mat_file)
    _required_file(path, "overlap matrix")
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty overlap matrix")
    try:
        ndim = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: first token must be the matrix dimension") from exc
    if ndim <= 0:
        raise ValueError(f"{path}: matrix dimension must be positive")
    rows: list[list[complex]] = []
    for row_index, line in enumerate(lines):
        tokens = line.split()
        if row_index == 0:
            tokens = tokens[1:]
        expected = ndim - row_index
        if row_index >= ndim or len(tokens) != expected:
            raise ValueError(f"{path}: row {row_index + 1} contains {len(tokens)} values; expected {expected}")
        rows.append([_parse_matrix_token(token) for token in tokens])
    if len(rows) != ndim:
        raise ValueError(f"{path}: expected {ndim} matrix rows, found {len(rows)}")
    matrix = np.zeros((ndim, ndim), dtype=np.complex128)
    for i, row in enumerate(rows):
        for offset, value in enumerate(row):
            j = i + offset
            matrix[i, j] = value
            matrix[j, i] = value.conjugate()
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path}: overlap matrix contains non-finite values")
    return matrix


def read_overlap_matrix_develop(ovlp_mat_file: str | Path) -> np.ndarray:
    """Read ABACUS develop version ``sk*_nao.txt`` upper-triangular matrix output.

    Format:
    #------------------------------------------------------------------------
    # ionic step 1
    # filename OUT.ABACUS/sk1_nao.txt
    # gamma only 0
    # rows 104
    # columns 104
    #------------------------------------------------------------------------
    Row 1
     (1.08393052e+00,0.00000000e+00) (-3.05029322e-01,0.00000000e+00) ...
    Row 2
     (8.80655982e-01,0.00000000e+00) (0.00000000e+00,0.00000000e+00) ...

    Only the upper triangle is stored: row ``i`` (one-based) holds the
    ``nrows - i + 1`` elements from the diagonal to the last column, wrapped
    over several physical lines.  The lower triangle is rebuilt by conjugation.
    """
    path = Path(ovlp_mat_file)
    _required_file(path, "overlap matrix (develop format)")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # A run with out_app_flag writes every ionic step into the same file;
    # only the last block describes the final state.
    blocks = [i for i, line in enumerate(lines) if line.strip().startswith("# rows")]
    if not blocks:
        raise ValueError(f"{path}: cannot find matrix dimensions in header")
    start = blocks[-1]
    nrows = int(lines[start].split()[-1])
    ncols = None
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("# columns"):
            ncols = int(stripped.split()[-1])
            break
    if nrows is None or ncols is None:
        raise ValueError(f"{path}: cannot find matrix dimensions in header")
    if nrows != ncols:
        raise ValueError(f"{path}: matrix must be square, got {nrows}x{ncols}")

    matrix = np.zeros((nrows, ncols), dtype=np.complex128)
    current_row: int | None = None
    column = 0
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("Row "):
            current_row = int(stripped.split()[1]) - 1
            if current_row >= nrows:
                raise ValueError(f"{path}: row index {current_row + 1} out of range")
            # The first stored element of a row sits on the diagonal.
            column = current_row
            continue
        if current_row is None or not stripped or stripped.startswith("#"):
            continue
        for token in stripped.split():
            value = _maybe_matrix_value(token)
            if value is None:
                continue
            if column >= ncols:
                raise ValueError(f"{path}: row {current_row + 1} contains too many values")
            matrix[current_row, column] = value
            matrix[column, current_row] = value.conjugate()
            column += 1

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path}: overlap matrix contains non-finite values")
    return matrix


def read_density_matrix(rho_mat_file: str | Path) -> np.ndarray:
    """Read a text density matrix by locating its final dimension block."""

    path = Path(rho_mat_file)
    _required_file(path, "density matrix")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    candidates: list[tuple[int, tuple[int, int], list[str]]] = []
    for index, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) != 2:
            continue
        try:
            candidate = tuple(int(token) for token in tokens)
        except ValueError:
            continue
        if candidate[0] > 0 and candidate[1] > 0:
            candidates.append((index, candidate, [token for rest in lines[index + 1 :] for token in rest.split()]))
    valid_candidates = [item for item in candidates if item[1][0] == item[1][1] and len(item[2]) == item[1][0] ** 2]
    if not valid_candidates:
        raise ValueError(f"{path}: cannot find density-matrix dimensions")
    dimension_index, dimension, value_tokens = valid_candidates[-1]
    if dimension[0] != dimension[1]:
        raise ValueError(f"{path}: density matrix must be square, got {dimension[0]}x{dimension[1]}")
    try:
        values = [_number(token) for token in value_tokens]
    except ValueError as exc:
        raise ValueError(f"{path}: invalid density-matrix value") from exc
    expected = dimension[0] * dimension[1]
    if len(values) != expected:
        raise ValueError(f"{path}: found {len(values)} density-matrix values; expected {expected}")
    matrix = np.asarray(values, dtype=float).reshape(dimension)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path}: density matrix contains non-finite values")
    return matrix



def read_csr_matrix_blocks(
    matrix_file: str | Path,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Read all CSR blocks of the final ionic step.

    ABACUS H(R), S(R) and DM(R) text files store one CSR block per Bravais
    lattice vector.  The returned mapping is keyed by ``(Rx, Ry, Rz)``.

    Args:
        matrix_file: Path to the CSR matrix.

    Returns:
        The dense matrix of every stored R block.

    Raises:
        ValueError: If the CSR blocks are missing or inconsistent.
    """

    path = _required_file(Path(matrix_file), "CSR matrix")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    steps = [
        index for index, line in enumerate(lines)
        if "Ionic Step" in line
    ]
    start = steps[-1] if steps else 0
    value_markers = [
        index for index in range(start, len(lines))
        if lines[index].strip().startswith("# CSR values")
    ]
    if not value_markers:
        raise ValueError(f"{path}: cannot find CSR values block")

    dimension: int | None = None
    for line in lines[start:]:
        if "# number of localized basis" in line:
            dimension = int(line.split()[0])
    if dimension is None or dimension <= 0:
        raise ValueError(f"{path}: cannot find the localized basis dimension")

    def marker(start_index: int, stop: int, name: str) -> int:
        for index in range(start_index, stop):
            if lines[index].strip().startswith(name):
                return index
        raise ValueError(f"{path}: cannot find {name.removeprefix('# ')} block")

    def header(start_index: int, stop: int) -> tuple[int, int, int, int]:
        for line in reversed(lines[start_index:stop]):
            tokens = line.split()
            if len(tokens) != 4:
                continue
            try:
                candidate = tuple(int(token) for token in tokens)
            except ValueError:
                continue
            if candidate[3] >= 0:
                return candidate
        raise ValueError(f"{path}: cannot find CSR matrix header")

    def tokens(start_index: int, stop: int, expected: int) -> list[str]:
        if expected == 0:
            return []
        values: list[str] = []
        for line in lines[start_index:stop]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in stripped.split():
                values.append(token)
                if len(values) == expected:
                    return values
        return values

    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    for position, values_start in enumerate(value_markers):
        block_start = value_markers[position - 1] + 1 if position else start
        block_stop = value_markers[position + 1] if position + 1 < len(value_markers) else len(lines)
        rx, ry, rz, nnz = header(block_start, values_start)
        columns_start = marker(values_start + 1, block_stop, "# CSR column indices")
        rows_start = marker(columns_start + 1, block_stop, "# CSR row pointers")

        value_tokens = tokens(values_start + 1, columns_start, nnz)
        column_tokens = tokens(columns_start + 1, rows_start, nnz)
        row_tokens = tokens(rows_start + 1, block_stop, dimension + 1)
        if len(value_tokens) != nnz:
            raise ValueError(
                f"{path}: found {len(value_tokens)} CSR values; expected {nnz}"
            )
        if len(column_tokens) != nnz:
            raise ValueError(
                f"{path}: found {len(column_tokens)} CSR column indices; expected {nnz}"
            )
        if len(row_tokens) != dimension + 1:
            raise ValueError(
                f"{path}: found {len(row_tokens)} CSR row pointers; expected {dimension + 1}"
            )

        try:
            values = np.asarray(
                [_parse_matrix_token(token) for token in value_tokens],
                dtype=np.complex128,
            )
            columns = np.asarray([int(token) for token in column_tokens], dtype=np.int64)
            row_pointers = np.asarray([int(token) for token in row_tokens], dtype=np.int64)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid CSR numeric value") from exc

        if row_pointers[0] != 0 or row_pointers[-1] != nnz:
            raise ValueError(f"{path}: CSR row pointers must start at 0 and end at {nnz}")
        if np.any(np.diff(row_pointers) < 0):
            raise ValueError(f"{path}: CSR row pointers must be non-decreasing")
        if np.any((columns < 0) | (columns >= dimension)):
            raise ValueError(f"{path}: CSR column index is outside the matrix")

        matrix = np.zeros((dimension, dimension), dtype=np.complex128)
        for row in range(dimension):
            first = int(row_pointers[row])
            last = int(row_pointers[row + 1])
            matrix[row, columns[first:last]] = values[first:last]
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{path}: CSR matrix contains non-finite values")
        blocks[(rx, ry, rz)] = matrix

    return blocks


def read_csr_matrix(matrix_file: str | Path) -> np.ndarray:
    """Read an ABACUS gamma-only ``sr_nao.csr`` or ``hrs*_nao.csr`` matrix.

    For a multi-R file the ``R = (0, 0, 0)`` block is returned, which is the
    in-cell part of both the gamma-folded H/S matrices and the real-space
    density matrix.

    Args:
        matrix_file: Path to the CSR matrix.

    Returns:
        The dense square matrix for ``R = (0, 0, 0)``.

    Raises:
        ValueError: If the CSR blocks are missing or inconsistent.
    """

    blocks = read_csr_matrix_blocks(matrix_file)
    if (0, 0, 0) in blocks:
        return blocks[(0, 0, 0)]
    if len(blocks) == 1:
        return next(iter(blocks.values()))
    return next(reversed(blocks.values()))


def read_eig_occ(
    eig_occ_file: str | Path,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Read ABACUS ``eig_occ.txt`` energies and occupations.

    The file groups the bands by spin and k-point.  Only the final ionic step
    is returned, because earlier steps belong to the SCF history rather than
    the converged density matrix.

    Args:
        eig_occ_file: Path to ``eig_occ.txt``.

    Returns:
        Mapping ``(spin, kpoint)`` to ``(energies_eV, occupations)``.

    Raises:
        ValueError: If no complete spin/k-point blocks can be read.
    """

    path = _required_file(Path(eig_occ_file), "eigenvalue and occupation table")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [
        index for index, line in enumerate(lines)
        if re.match(r"^\s*\d+\s*#\s*ionic step", line, re.IGNORECASE)
    ]
    start = starts[-1] if starts else 0
    records: dict[tuple[int, int], list[tuple[float, float]]] = {}
    current: tuple[int, int] | None = None
    marker = re.compile(r"^\s*spin=(\d+)\s+k-point=(\d+)/(\d+)", re.IGNORECASE)
    for line in lines[start:]:
        match = marker.match(line)
        if match:
            current = (int(match.group(1)), int(match.group(2)))
            records.setdefault(current, [])
            continue
        if current is None:
            continue
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            _band, energy, occupation = fields
            int(_band)
            records[current].append((_number(energy), _number(occupation)))
        except ValueError:
            continue

    result: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, values in records.items():
        if not values:
            continue
        energies, occupations = zip(*values)
        energy_array = np.asarray(energies, dtype=float)
        occupation_array = np.asarray(occupations, dtype=float)
        if not np.all(np.isfinite(energy_array)):
            raise ValueError(f"{path}: non-finite band energy for spin {key[0]}")
        if not np.all(np.isfinite(occupation_array)) or np.any(occupation_array < 0):
            raise ValueError(f"{path}: invalid occupations for spin {key[0]}")
        result[key] = (energy_array, occupation_array)
    if not result:
        raise ValueError(f"{path}: cannot find any spin/k-point occupation blocks")
    return result


def calculate_density_matrix_from_matrices(
    hamiltonian: np.ndarray,
    overlap: np.ndarray,
    occupations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a density matrix from Hamiltonian, overlap and occupations.

    The Hamiltonian and overlap are the gamma-point matrices folded by ABACUS.
    The generalized eigenproblem ``H C = S C E`` is solved through the
    Cholesky factorization of ``S``.

    Args:
        hamiltonian: Hamiltonian matrix in Ry.
        overlap: Overlap matrix.
        occupations: Band occupations, one value per retained band.

    Returns:
        The band energies in Ry and the density matrix in the AO basis.

    Raises:
        ValueError: If the matrices have incompatible shapes or the overlap is
            not positive definite.
    """

    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    overlap = np.asarray(overlap, dtype=np.complex128)
    occupations = np.asarray(occupations, dtype=float)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian matrix must be square")
    if overlap.shape != hamiltonian.shape:
        raise ValueError("Hamiltonian and overlap matrices must have the same shape")
    if occupations.ndim != 1 or len(occupations) == 0:
        raise ValueError("occupations must be a non-empty one-dimensional array")
    if len(occupations) > hamiltonian.shape[0]:
        raise ValueError("more occupations than basis functions")
    if np.any(~np.isfinite(occupations)) or np.any(occupations < 0):
        raise ValueError("occupations must be finite and non-negative")
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=1e-8, atol=1e-10):
        raise ValueError("Hamiltonian matrix is not Hermitian")
    if not np.allclose(overlap, overlap.conj().T, rtol=1e-8, atol=1e-10):
        raise ValueError("overlap matrix is not Hermitian")

    try:
        cholesky = np.linalg.cholesky(overlap)
    except np.linalg.LinAlgError as exc:
        raise ValueError("overlap matrix is not positive definite") from exc

    transformed = np.linalg.solve(cholesky, hamiltonian)
    transformed = np.linalg.solve(cholesky, transformed.conj().T).conj().T
    eigenvalues, eigenvector_columns = np.linalg.eigh(transformed)
    coefficients = np.linalg.solve(cholesky.conj().T, eigenvector_columns)
    retained = coefficients[:, : len(occupations)]
    weighted = retained * np.sqrt(occupations)[np.newaxis, :]
    density = weighted @ weighted.conj().T
    return eigenvalues, density


def read_wfc_nao_k_data(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one WFC file, returning coefficients, band energies, and occupations."""

    path = Path(file_path)
    _required_file(path, "NAO wavefunction")
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    nbands = nlocal = None
    for line in lines:
        if line.endswith("(number of bands)"):
            nbands = int(line.split()[0])
        elif line.endswith("(number of orbitals)"):
            nlocal = int(line.split()[0])
    if nbands is None or nlocal is None or nbands <= 0 or nlocal <= 0:
        raise ValueError(f"{path}: missing or invalid number of bands/orbitals")
    wfc = np.zeros((nlocal, nbands), dtype=np.complex128)
    energies = np.full(nbands, np.nan, dtype=float)
    occupations = np.full(nbands, np.nan, dtype=float)
    current_band: Optional[int] = None
    reading_coefficients = False
    coefficients: dict[int, list[float]] = {}
    for line in lines:
        if line.endswith("(band)"):
            current_band = int(line.split()[0]) - 1
            if not 0 <= current_band < nbands:
                raise ValueError(f"{path}: band index out of range: {current_band + 1}")
            coefficients[current_band] = []
            reading_coefficients = False
        elif line.endswith("(Ry)"):
            if current_band is None:
                raise ValueError(f"{path}: energy without a band header")
            energies[current_band] = _number(line.split()[0])
        elif line.endswith("(Occupations)"):
            if current_band is None:
                raise ValueError(f"{path}: occupation without a band header")
            occupations[current_band] = _number(line.split()[0])
            if occupations[current_band] < 0:
                raise ValueError(f"{path}: negative occupation for band {current_band + 1}")
            reading_coefficients = True
        elif reading_coefficients and line and not line.endswith(")"):
            tokens = line.split()
            coefficients[current_band].extend(_number(token) for token in tokens)
    if len(coefficients) != nbands or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(occupations)):
        raise ValueError(f"{path}: incomplete band or occupation data")
    for band in range(nbands):
        raw = coefficients.get(band, [])
        if len(raw) == nlocal:
            wfc[:, band] = np.asarray(raw, dtype=float)
        elif len(raw) == 2 * nlocal:
            wfc[:, band] = [complex(raw[index], raw[index + 1]) for index in range(0, len(raw), 2)]
        else:
            raise ValueError(f"{path}: band {band + 1} has {len(raw)} wavefunction values; expected {nlocal} or {2 * nlocal}")
    return wfc, energies, occupations


def read_wfc_nao_k(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one ``WFC_NAO_K*.txt`` file and its band occupations."""

    wfc, _energies, occupations = read_wfc_nao_k_data(file_path)
    return wfc, occupations


def calculate_density_matrix_k(wfc: np.ndarray, occupations: np.ndarray) -> np.ndarray:
    """Build a k-point density matrix from coefficients and occupations."""

    if wfc.ndim != 2 or occupations.shape != (wfc.shape[1],):
        raise ValueError("wavefunction and occupation dimensions do not match")
    if np.any(occupations < 0):
        raise ValueError("occupations must be non-negative")
    weighted = wfc * np.sqrt(occupations)[np.newaxis, :]
    return weighted @ weighted.conj().T


def read_kpoint_weights(kpoints_file: str | Path) -> np.ndarray:
    """Read the weights printed in the ``KPOINTS DIRECT_X ... WEIGHT`` table."""

    table = read_kpoint_table(kpoints_file)
    weights = np.asarray([kpoint.weight for kpoint in table.ibz], dtype=float)
    if not len(weights) or not np.all(np.isfinite(weights)) or sum(weights) <= 0:
        raise ValueError(f"{kpoints_file}: no valid k-point weights found")
    return weights




#: Distance tolerances in Angstrom tried when the space-group operations of a
#: symmetry-reduced run are rebuilt from the structure; the tightest one that
#: reproduces the k-point reduction of the run is used.
SYMMETRY_TOLERANCES = (1e-5, 1e-4, 1e-3, 1e-2)


@dataclass(frozen=True)
class KPoint:
    """One k-point in direct coordinates with the weight its file prints."""

    direct: tuple[float, float, float]
    weight: float


@dataclass(frozen=True)
class KPointTable:
    """The k-points of an ABACUS run.

    Attributes:
        ibz: The computed k-points, in the order of the WFC and density-matrix
            files.
        mesh: The full k-mesh of a symmetry-reduced run, every point with the
            one-based index of the computed k-point it belongs to. Empty when
            ABACUS kept the whole mesh or reduced it by time reversal only.
        kpoint_count: Number of k-points of the unreduced mesh, when the
            output records it (the develop density matrices do).
    """

    ibz: tuple[KPoint, ...]
    mesh: tuple[tuple[tuple[float, float, float], int], ...] = ()
    kpoint_count: Optional[int] = None


def _kpoint_rows(lines: list[str], marker: str, columns: int) -> list[list[str]]:
    """Return the numeric rows of the k-point table that follows a marker."""
    start = next((index for index, line in enumerate(lines) if marker in line), None)
    if start is None:
        return []
    while start < len(lines) and "DIRECT_X" not in lines[start]:
        start += 1
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        fields = line.split()
        if not fields:
            if rows:
                break
            continue
        try:
            [float(token) for token in fields[1:4]]
        except ValueError:
            continue
        if len(fields) == columns:
            rows.append(fields)
    return rows


def read_kpoint_table(kpoints_file: str | Path) -> KPointTable:
    """Read the ``kpoints`` file of an ABACUS output directory.

    The file lists the computed k-points and, when ABACUS reduced the mesh by
    symmetry, the full mesh with the index of the computed k-point every mesh
    point belongs to.
    """
    path = _required_file(Path(kpoints_file), "k-point table")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ibz_rows = _kpoint_rows(lines, "KPOINTS", 5)
    if not ibz_rows:
        raise ValueError(f"{path}: cannot find KPOINTS DIRECT_X table")
    ibz = tuple(
        KPoint(tuple(_number(token) for token in row[1:4]), _number(row[4]))
        for row in ibz_rows
    )
    mesh = tuple(
        (tuple(_number(token) for token in row[1:4]), int(row[4]))
        for row in _kpoint_rows(lines, "K-POINTS REDUCTION", 8)
    )
    return KPointTable(ibz, mesh)


def read_develop_kpoints(files: Iterable[tuple[int, int, Path]]) -> KPointTable:
    """Read the k-points recorded in the develop ``dm*_nao.txt`` headers."""
    per_k: dict[int, KPoint] = {}
    total: Optional[int] = None
    for ik, _ispin, path in files:
        if ik in per_k:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        counts = [int(line.split()[0]) for line in lines if "# total k points" in line]
        direct = [line for line in lines if "# k point coordinate (direct)" in line]
        weight = [line for line in lines if "# weight of this k point" in line]
        if len(counts) < 2 or not direct or not weight:
            raise ValueError(f"{path}: cannot read the k-point header")
        if total is None:
            total = counts[0]
        per_k[ik] = KPoint(
            tuple(_number(token) for token in direct[-1].split()[:3]),
            _number(weight[-1].split()[0]),
        )
    return KPointTable(tuple(per_k[ik] for ik in sorted(per_k)), (), total)


def _time_reversal_invariant(direct: Sequence[float], *, tolerance: float = 1e-8) -> bool:
    """Return whether a k-point is its own time-reversed partner."""
    return all(abs(2.0 * value - round(2.0 * value)) < tolerance for value in direct)


def exact_kpoint_weights(table: KPointTable, *, printed_scale: float = 1.0) -> tuple[float, ...]:
    """Return the weight of every computed k-point, free of file rounding.

    The printed weights carry four decimals, so a run whose weights are 1/64,
    6/64, ... would be summed with a small bias. The exact weights follow from
    the symmetry of the reduction: the star sizes of a symmetry-reduced run,
    one per k-point of a full mesh, and the time-reversal multiplicity of the
    k-point in between.

    Args:
        table: k-points of the run.
        printed_scale: Factor between the printed weight and the k-point
            weight, which is two for the LTS wavefunctions of a spin-unpaired
            run and one for its spin-paired ones.

    Returns:
        The weight of every computed k-point, in the file order.

    Raises:
        ValueError: If the weights do not reproduce the printed ones, which
            happens when the output does not record the reduction.
    """
    if table.mesh:
        counts = [0] * len(table.ibz)
        for _point, index in table.mesh:
            if not 1 <= index <= len(counts):
                raise ValueError(f"k-point table refers to k-point {index}, which was not computed")
            counts[index - 1] += 1
    else:
        uniform = True
        printed = [kpoint.weight for kpoint in table.ibz]
        if printed:
            uniform = max(printed) - min(printed) <= 1e-4 * max(printed)
        if uniform:
            counts = [1] * len(table.ibz)
        else:
            counts = [1 if _time_reversal_invariant(kpoint.direct) else 2 for kpoint in table.ibz]
    total = sum(counts)
    if table.kpoint_count is not None and total != table.kpoint_count:
        raise ValueError(
            f"the k-points follow a mesh of {total} points but the output records "
            f"{table.kpoint_count}; the reduction cannot be rebuilt from this output"
        )
    weights = []
    for kpoint, count in zip(table.ibz, counts):
        weight = count / total
        if abs(weight * printed_scale - kpoint.weight) > 1e-4:
            raise ValueError(
                f"k-point {kpoint.direct} has weight {kpoint.weight} in the output "
                f"but {weight:g} follows from the symmetry; rerun the calculation "
                "with symmetry=0 or -1 to analyse it"
            )
        weights.append(weight)
    return tuple(weights)


def _mesh_permutations(table: KPointTable, structure: AbacusSTRU) -> list[tuple[int, ...]]:
    """Return the atom permutation that reaches every mesh point from its k-point."""
    coordinates = [np.asarray(kpoint.direct, dtype=float) for kpoint in table.ibz]
    for symprec in SYMMETRY_TOLERANCES:
        try:
            operations = space_group_operations(structure, symprec=symprec)
        except ValueError:
            continue
        permutations: list[tuple[int, ...]] = []
        for point, index in table.mesh:
            if not 1 <= index <= len(coordinates):
                raise ValueError(f"k-point table refers to k-point {index}, which was not computed")
            target = coordinates[index - 1]
            found = None
            for operation in operations:
                rotated = operation.rotate_kpoint(target)
                for candidate in (rotated, -rotated):
                    delta = candidate - np.asarray(point, dtype=float)
                    delta -= np.round(delta)
                    if np.all(np.abs(delta) < 1e-6):
                        found = operation.permutation
                        break
                if found is not None:
                    break
            if found is None:
                permutations = []
                break
            permutations.append(found)
        if permutations:
            return permutations
    raise ValueError(
        "the space-group operations that reduced the k-points of this run cannot "
        "be rebuilt from the structure; rerun the calculation with symmetry=0 or -1"
    )


def expanded_orders(
    stars: dict[int, Sequence[tuple[int, ...]]],
    weights: Sequence[float],
    values_of: Any,
    pairs: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Average the star members of every reduced k-point.

    A quantity like the Mayer bond order is not linear in the k-resolved
    density matrix, so the star of a computed k-point cannot be folded into
    its weight: the value of every star member has to be evaluated separately.
    Applying a symmetry operation ``{R|w}`` to the wavefunction of k moves the
    orbital labels onto the atoms the operation maps them to, which means that
    a pair value at the star member ``R k`` equals the value at ``k`` for the
    pair of the atoms that the operation moves onto the original pair.

    Args:
        stars: One-based computed k-point index -> atom permutation of every
            mesh point of its star.
        weights: Weight of every computed k-point.
        values_of: Callable ``(k-point index, pairs) -> {pair: value}`` that
            evaluates the pairs at one computed k-point.
        pairs: Zero-based atom pairs of interest.

    Returns:
        The bond order of every pair, summed over the full mesh.
    """
    orders = {pair: 0.0 for pair in pairs}
    for index, members in stars.items():
        weight = weights[index - 1]
        needed = {
            tuple(sorted((permutation[pair[0]], permutation[pair[1]])))
            for permutation in members
            for pair in pairs
        }
        values = values_of(index - 1, sorted(needed))
        for pair in pairs:
            total = 0.0
            for permutation in members:
                total += values[tuple(sorted((permutation[pair[0]], permutation[pair[1]])))]
            orders[pair] += total / len(members) / weight
    return orders


def _symmetry_expanded_orders(
    table: KPointTable,
    weights: Sequence[float],
    values_of: Any,
    pairs: Sequence[tuple[int, int]],
    structure: AbacusSTRU,
) -> dict[tuple[int, int], float]:
    """Rebuild the k-points that the space group of a run reduced away."""
    permutations = _mesh_permutations(table, structure)
    stars: dict[int, list[tuple[int, ...]]] = {}
    for (_point, index), permutation in zip(table.mesh, permutations):
        stars.setdefault(index, []).append(permutation)
    return expanded_orders(stars, weights, values_of, pairs)


def _highest_step_file(paths: list[Path], pattern: str) -> Path | None:
    """Return the file with the highest one-based ionic step in its name."""

    ranked: list[tuple[int, Path]] = []
    for path in paths:
        match = re.fullmatch(pattern, path.name)
        if match:
            ranked.append((int(match.group(1)), path))
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[0])[1]


def _csr_overlap_file(output: Path) -> Path:
    """Return the current-ABACUS gamma-point overlap matrix file."""

    for name in ("sr_nao.csr", "data-SR-sparse_SPIN0.csr"):
        candidate = output / name
        if candidate.is_file():
            return candidate
    stepped = _highest_step_file(
        list(output.glob("srg*_nao.csr")), r"srg(\d+)_nao\.csr"
    )
    if stepped is not None:
        return stepped
    return output / "sr_nao.csr"


def _csr_hamiltonian_file(output: Path, spin: int) -> Path:
    """Return the Hamiltonian matrix file for one collinear spin channel."""

    candidates = [output / f"hrs{spin}_nao.csr"]
    if spin == 1:
        candidates.append(output / "hrs_nao.csr")
    candidates.append(output / f"data-HR-sparse_SPIN{spin - 1}.csr")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for candidate in sorted(output.glob("hrs*_nao.csr")):
        match = re.fullmatch(r"hrs(\d+)_nao\.csr", candidate.name)
        if match and int(match.group(1)) == spin:
            return candidate
    stepped = _highest_step_file(
        list(output.glob(f"hrs{spin}g*_nao.csr")),
        rf"hrs{spin}g(\d+)_nao\.csr",
    )
    if stepped is not None:
        return stepped
    return candidates[0]


def _dmr_file(output: Path, spin: int) -> Path:
    """Return the develop-version density-matrix file for one spin channel."""

    candidates = [
        output / f"dmrs{spin}_nao.csr",
        output / f"data-DMR-sparse_SPIN{spin - 1}.csr",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    stepped = _highest_step_file(
        list(output.glob(f"dmrs{spin}g*_nao.csr")),
        rf"dmrs{spin}g(\d+)_nao\.csr",
    )
    if stepped is not None:
        return stepped
    return candidates[0]


def _has_dmr(output: Path) -> bool:
    """Return whether a develop- or LTS-style DM(R) output is present."""

    return bool(list(output.glob("dmrs*_nao.csr"))) or bool(
        list(output.glob("data-DMR-sparse_SPIN*.csr"))
    )


def _gamma_overlap_matrix(output: Path) -> tuple[Path, np.ndarray]:
    """Read the gamma-point overlap matrix from whichever output is present."""

    for name in ("sr_nao.csr", "data-SR-sparse_SPIN0.csr"):
        path = output / name
        if path.is_file():
            return path, read_csr_matrix(path)
    for name in ("sk_nao.txt", "data-0-S"):
        path = output / name
        if path.is_file():
            return path, (
                read_overlap_matrix_develop(path)
                if name.endswith(".txt")
                else read_overlap_matrix(path)
            )
    raise FileNotFoundError(
        f"Cannot find gamma overlap matrix in {output}; expected sr_nao.csr, "
        "sk_nao.txt, data-SR-sparse_SPIN0.csr or data-0-S"
    )


def detect_matrix_format(
    output_dir: Path,
    out_dmk: int,
    *,
    out_dmr: int = 0,
    out_mat_hs: int = 0,
    out_mat_hs2: int = 0,
) -> str:
    """Return the density-matrix or matrix layout to read.

    Density-matrix outputs are preferred for the develop version: ``out_dmk``
    gives k-resolved DM(k), and ``out_dmr`` gives DM(R).  H(R)/S(R) and the
    LTS layout are fallbacks for jobs that did not request a density matrix.

    Args:
        output_dir: ABACUS ``OUT.*`` directory.
        out_dmk: Value of the ``out_dmk`` input parameter.
        out_dmr: Value of the ``out_dmr`` input parameter.
        out_mat_hs: Value of the legacy ``out_mat_hs`` input parameter.
        out_mat_hs2: Value of the legacy ``out_mat_hs2`` input parameter.

    Returns:
        str: ``"develop"`` for DM(k), ``"dmr"`` for DM(R), ``"csr"`` for
        current-ABACUS H(R)/S(R), or ``"lts"`` for the legacy layout.
    """

    has_dmk = bool(list(output_dir.glob("dm*_nao.txt")))
    has_dmr = _has_dmr(output_dir)
    has_lts = bool(list(output_dir.glob("data-*-S"))) or (output_dir / "SPIN1_DM").is_file()
    has_csr = _csr_overlap_file(output_dir).is_file() and (
        bool(list(output_dir.glob("hrs*_nao.csr")))
        or (output_dir / "data-HR-sparse_SPIN0.csr").is_file()
    )
    if out_dmk == 1 and has_dmk:
        return "develop"
    if out_dmr == 1 and has_dmr:
        return "dmr"
    if out_mat_hs2 == 1 and has_csr:
        return "csr"
    if out_mat_hs == 1 and has_lts:
        return "lts"
    if has_dmk:
        return "develop"
    if has_dmr:
        return "dmr"
    if has_lts:
        return "lts"
    if has_csr:
        return "csr"
    return "lts"


def read_density_matrix_develop(dmk_file: str | Path) -> np.ndarray:
    """Read ABACUS develop version ``dmk*_nao.txt`` density matrix.

    Format:
     --- Ionic Step 1 ---
     1 # number of spin directions
     1 # spin index
     64 # total k points
     64 # total k points after symmetrized (if open)
     1 # k-point index
     0 0 0 # k point coordinate (Cartesian)
     0 0 0 # k point coordinate (direct)
     0.03125 # weight of this k point
     1.19007 # Fermi energy in Ry
     104 # number of localized basis
     104 104 # size of this matrix

     [structure info]

     (real,imag) (real,imag) ...
     (real,imag) (real,imag) ...

    Unlike ``sk*_nao.txt`` this file stores the complete square matrix, four
    values per line and without ``Row`` markers.
    """
    path = Path(dmk_file)
    _required_file(path, "density matrix (develop format)")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Appended ionic steps share one file, so keep the last size header.
    sizes = [i for i, line in enumerate(lines) if "# size of this matrix" in line]
    if not sizes:
        raise ValueError(f"{path}: cannot find matrix dimensions in header")
    block = sizes[-1]
    nrows, ncols = (int(part) for part in lines[block].split()[:2])

    # The cell and atomic positions of the structure block precede the matrix,
    # so the matrix is taken as the last nrows * ncols numeric values.
    values: list[complex] = []
    for line in lines[block + 1 :]:
        stripped = line.strip()
        if stripped.startswith("---") or "Ionic Step" in stripped:
            break
        for token in stripped.split():
            value = _maybe_matrix_value(token)
            if value is not None:
                values.append(value)
    expected = nrows * ncols
    if len(values) < expected:
        raise ValueError(
            f"{path}: found {len(values)} matrix values; expected {expected}"
        )
    values = values[-expected:]
    matrix = np.asarray(values, dtype=np.complex128).reshape(nrows, ncols)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path}: density matrix contains non-finite values")
    return matrix


def _minimum_distance(lattice, frac1: Iterable[float], frac2: Iterable[float]) -> float:
    """Return the shortest periodic distance between two fractional coordinates.

    The minimum-image search is delegated to pymatgen, which LLL-reduces the
    lattice before scanning the neighbouring cells. Searching the images of the
    raw cell instead overestimates the distance as soon as the cell is strongly
    skewed, because the nearest image may then sit more than one cell away.
    """
    distance, _image = lattice.get_distance_and_image(frac1, frac2)
    return float(distance)


def _validate_pair(first: int, second: int, natoms: int, context: str) -> tuple[int, int]:
    if first == second or not (0 <= first < natoms and 0 <= second < natoms):
        raise ValueError(f"{context}: invalid atom pair {first + 1}-{second + 1} for {natoms} atoms")
    return min(first, second), max(first, second)


def _unique_pairs(pairs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(dict.fromkeys(pairs))


def read_pairs_from_file(pairs_file: str | Path, natoms: int) -> list[tuple[int, int]]:
    """Read one-based atom pairs from a text file and return zero-based pairs."""

    path = _required_file(Path(pairs_file), "pairs file")
    pairs: list[tuple[int, int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected two atom indices")
        try:
            pair = _validate_pair(int(fields[0]) - 1, int(fields[1]) - 1, natoms, f"{path}:{line_number}")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        pairs.append(pair)
    return _unique_pairs(pairs)


def select_atom_pairs(structure: AbacusSTRU, cutoff: Optional[float] = None, pairs_str: Optional[str] = None, pairs_file: Optional[str | Path] = None) -> list[tuple[int, int]]:
    """Select atom pairs by explicit indices, file, or periodic distance."""

    if sum(value is not None for value in (cutoff, pairs_str, pairs_file)) > 1:
        raise ValueError("cutoff, pairs_str, and pairs_file are mutually exclusive")
    if cutoff is not None:
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        lattice = periodic_lattice(structure)
        fractions = structure.coords_direct
        return [(i, j) for i in range(structure.natoms) for j in range(i + 1, structure.natoms) if _minimum_distance(lattice, fractions[i], fractions[j]) <= cutoff]
    if pairs_file is not None:
        return read_pairs_from_file(pairs_file, structure.natoms)
    if pairs_str is not None:
        pairs = []
        for token in pairs_str.split(","):
            fields = token.strip().split("-")
            if len(fields) != 2:
                raise ValueError(f"invalid pair {token!r}; expected i-j")
            pairs.append(_validate_pair(int(fields[0]) - 1, int(fields[1]) - 1, structure.natoms, "--pairs"))
        if not pairs:
            raise ValueError("--pairs did not contain any atom pairs")
        return _unique_pairs(pairs)
    return [(i, j) for i in range(structure.natoms) for j in range(i + 1, structure.natoms)]


def _mayer_order(
    dm: np.ndarray,
    overlap: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    *,
    product: Optional[np.ndarray] = None,
) -> float:
    """Return the Mayer order of one atom pair.

    ``product`` may be supplied when the caller evaluates several pairs with
    the same density and overlap matrices, avoiding a repeated matrix
    multiplication for every pair.
    """

    left_start, left_end = first
    right_start, right_end = second
    ps = dm @ overlap if product is None else product
    return float(
        np.sum(
            ps[left_start:left_end, right_start:right_end]
            * ps[right_start:right_end, left_start:left_end].T
        ).real
    )


def cal_mayer_bond_order_between_atom_pair(
    i: int,
    j: int,
    atom_orb_ranges: list[tuple[int, int]],
    ovlp_mat: np.ndarray,
    dm: np.ndarray,
    dm_dn: Optional[np.ndarray] = None,
) -> float:
    """Calculate one gamma-point Mayer order using zero-based atom indices."""

    value = _mayer_order(
        dm,
        ovlp_mat,
        atom_orb_ranges[i],
        atom_orb_ranges[j],
        product=dm @ ovlp_mat,
    )
    if dm_dn is not None:
        value = 2.0 * (
            value
            + _mayer_order(
                dm_dn,
                ovlp_mat,
                atom_orb_ranges[i],
                atom_orb_ranges[j],
                product=dm_dn @ ovlp_mat,
            )
        )
    return value


def cal_mayer_bond_order_between_atom_pair_k(
    i: int,
    j: int,
    atom_orb_ranges: list[tuple[int, int]],
    ovlp_mat: np.ndarray,
    dm: np.ndarray,
) -> float:
    """Calculate one k-point Mayer order using zero-based atom indices."""

    return _mayer_order(dm, ovlp_mat, atom_orb_ranges[i], atom_orb_ranges[j])


def _atom_ranges(structure: AbacusSTRU, nao_by_file: dict[str, NAOData]) -> tuple[list[tuple[int, int]], int]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for index, atom in enumerate(structure.atoms, 1):
        if not atom.orb or atom.orb not in nao_by_file:
            raise ValueError(f"atom {index} has no parsed numerical orbital file")
        start_next = start + get_nao_basis_num(nao_by_file[atom.orb])
        ranges.append((start, start_next))
        start = start_next
    return ranges, start


def _resolve_orbital(job: Path, orbital_dir: Any, filename: str) -> Path:
    base = Path(str(orbital_dir)) if orbital_dir else job
    return (base if base.is_absolute() else job / base) / filename


_DEVELOP_DM_NAME = re.compile(
    r"^dm(?:k(?P<ik>\d+))?(?:s(?P<is>\d+))?(?:g(?P<step>\d+))?_nao\.txt$"
)


def _develop_density_files(output: Path) -> list[tuple[int, int, Path]]:
    """Return ``(ik, ispin, path)`` for every develop k-resolved density matrix.

    The develop version names the files ``dmg1_nao.txt`` (gamma-only,
    ``nspin=1``), ``dms{is}g1_nao.txt`` (gamma-only, ``nspin=2``),
    ``dmk{ik}g1_nao.txt`` (multi-k, ``nspin=1``) and
    ``dmk{ik}s{is}g1_nao.txt`` (multi-k, ``nspin=2``).  The k- and spin-index
    are therefore read from the file name instead of assumed.  The ``g``
    ionic-step marker only appears when ``out_app_flag`` is disabled, so it is
    optional.
    """
    entries: list[tuple[int, int, Path]] = []
    for path in sorted(output.glob("dm*_nao.txt")):
        match = _DEVELOP_DM_NAME.match(path.name)
        if match is None:
            continue
        ik = int(match.group("ik")) if match.group("ik") else 1
        ispin = int(match.group("is")) if match.group("is") else 1
        entries.append((ik, ispin, path))
    entries.sort(key=lambda item: (item[0], item[1]))
    return entries


def _develop_overlap_file(output: Path, ik: int) -> Path:
    """Return the overlap matrix that belongs to one k-point.

    Prefer the k-resolved ``sk*_nao.txt`` output.  If the job only wrote the
    CSR overlap, use ``sr_nao.csr`` and reconstruct ``S(k)`` from its R blocks.
    """

    gamma = output / "sk_nao.txt"
    if gamma.is_file():
        return gamma
    kpoint = output / f"sk{ik}_nao.txt"
    if kpoint.is_file():
        return kpoint
    for name in ("sr_nao.csr", "data-SR-sparse_SPIN0.csr"):
        candidate = output / name
        if candidate.is_file():
            return candidate
    return kpoint


def _read_develop_overlap(
    path: Path,
    *,
    kpoint: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Read a develop overlap matrix from text or CSR output.

    A multi-R CSR overlap is Fourier transformed to ``S(k)``.  A single
    ``R = (0, 0, 0)`` block is the gamma-folded representation and is only
    valid at Gamma.
    """

    if path.suffix != ".csr":
        return read_overlap_matrix_develop(path)
    blocks = read_csr_matrix_blocks(path)
    if len(blocks) == 1:
        if kpoint is not None and any(abs(value) > 1e-10 for value in kpoint):
            raise ValueError(
                f"{path}: a single R block is gamma-folded and cannot be used "
                f"at non-Gamma k-point {kpoint}"
            )
        return next(iter(blocks.values()))
    if kpoint is None:
        raise ValueError(
            f"{path}: a k-point is required to Fourier transform a multi-R overlap"
        )
    matrix = np.zeros_like(next(iter(blocks.values())))
    for (rx, ry, rz), block in blocks.items():
        phase = np.exp(-2j * np.pi * (kpoint[0] * rx + kpoint[1] * ry + kpoint[2] * rz))
        matrix += phase * block
    return matrix


def analyze_mayer_bond_order(job: str | Path, *, cutoff: Optional[float] = None, pairs: Optional[str] = None, pairs_file: Optional[str | Path] = None) -> MayerAnalysis:
    """Analyze Mayer bond orders from an ABACUS LCAO job directory.

    Density-matrix outputs are preferred: ``out_dmk`` (DM(k)) and ``out_dmr``
    (DM(R)) for the develop layout, then the LTS density matrices or
    wavefunctions, with H/S matrix output used only when necessary.
    """

    job_path = Path(job).resolve()
    inputs = ReadInput(str(_required_file(job_path / "INPUT", "INPUT file")))
    calculation = str(inputs.get("calculation", "scf")).lower()
    nspin = int(_scalar(inputs.get("nspin", 1)))
    if nspin not in (1, 2):
        raise ValueError(f"Mayer analysis supports nspin=1 or 2, got {nspin}")
    if str(_scalar(inputs.get("basis_type", "pw"))).lower() != "lcao":
        raise ValueError("Mayer analysis requires basis_type=lcao")
    gamma_only = bool(int(_scalar(inputs.get("gamma_only", 0)) or 0))
    suffix = str(_scalar(inputs.get("suffix", "ABACUS")))
    output = _required_directory(job_path / f"OUT.{suffix}", "ABACUS output directory")
    structure_path = job_path / str(_scalar(inputs.get("stru_file", "STRU")))
    structure = AbacusSTRU.read(str(_required_file(structure_path, "STRU file")))
    if structure is None:
        raise ValueError(f"cannot read structure: {structure_path}")

    # Choose the layout that matches the requested outputs and the files present
    out_mat_hs2 = max(
        int(_scalar(inputs.get("out_mat_hs2", 0)) or 0),
        int(_scalar(inputs.get("out_hsr", 0)) or 0),
    )
    out_dmr = max(
        int(_scalar(inputs.get("out_dmr", 0)) or 0),
        int(_scalar(inputs.get("out_dm1", 0)) or 0),
    )
    version = detect_matrix_format(
        output,
        int(_scalar(inputs.get("out_dmk", 0)) or 0),
        out_dmr=out_dmr,
        out_mat_hs=int(_scalar(inputs.get("out_mat_hs", 0)) or 0),
        out_mat_hs2=out_mat_hs2,
    )
    
    nao_by_file: dict[str, NAOData] = {}
    for index, atom in enumerate(structure.atoms, 1):
        if not atom.orb:
            raise ValueError(f"atom {index} has no numerical orbital file")
        if atom.orb not in nao_by_file:
            nao_by_file[atom.orb] = read_nao_file(_resolve_orbital(job_path, inputs.get("orbital_dir"), atom.orb))
    atom_ranges, basis_functions = _atom_ranges(structure, nao_by_file)
    selected_pairs = select_atom_pairs(structure, cutoff, pairs, pairs_file)
    orders = {pair: 0.0 for pair in selected_pairs}
    data_files: list[str] = []

    if version == "csr":
        if not gamma_only:
            raise ValueError(
                "Mayer analysis of the sr_nao.csr/hrs*_nao.csr output is "
                "currently supported for gamma_only=1 runs"
            )
        overlap_path = _required_file(
            _csr_overlap_file(output),
            "current-ABACUS gamma overlap matrix",
        )
        overlap = read_csr_matrix(overlap_path)
        if overlap.shape != (basis_functions, basis_functions):
            raise ValueError(
                "overlap dimension does not match NAO basis size "
                f"{basis_functions}"
            )
        eig_occ_path = _required_file(
            output / "eig_occ.txt",
            "eigenvalue and occupation table",
        )
        eigenstates = read_eig_occ(eig_occ_path)
        data_files.extend([str(overlap_path), str(eig_occ_path)])
        tolerance = 1e-3
        spin_factor = 2.0 if nspin == 2 else 1.0
        for spin in range(1, nspin + 1):
            hamiltonian_path = _required_file(
                _csr_hamiltonian_file(output, spin),
                f"Hamiltonian matrix for spin {spin}",
            )
            hamiltonian = read_csr_matrix(hamiltonian_path)
            if hamiltonian.shape != overlap.shape:
                raise ValueError(
                    f"spin {spin}: Hamiltonian and overlap dimensions do not match"
                )
            try:
                printed_energies, occupations = eigenstates[(spin, 1)]
            except KeyError as exc:
                raise ValueError(
                    f"eig_occ.txt has no spin={spin} k-point=1 block"
                ) from exc
            eigenvalues, density = calculate_density_matrix_from_matrices(
                hamiltonian, overlap, occupations
            )
            computed_energies = eigenvalues[: len(printed_energies)] * RY_TO_EV
            if not np.allclose(
                computed_energies,
                printed_energies,
                rtol=1e-8,
                atol=tolerance,
            ):
                raise ValueError(
                    f"spin {spin}: Hamiltonian eigenvalues do not match eig_occ.txt"
                )
            data_files.append(str(hamiltonian_path))
            population = density @ overlap
            for pair in selected_pairs:
                orders[pair] += spin_factor * _mayer_order(
                    density,
                    overlap,
                    atom_ranges[pair[0]],
                    atom_ranges[pair[1]],
                    product=population,
                )
    elif version == "dmr":
        overlap_path, overlap = _gamma_overlap_matrix(output)
        data_files.append(str(overlap_path))
        if overlap.shape != (basis_functions, basis_functions):
            raise ValueError(
                "overlap dimension does not match NAO basis size "
                f"{basis_functions}"
            )
        spin_factor = 2.0 if nspin == 2 else 1.0
        for spin in range(1, nspin + 1):
            dmr_path = _required_file(
                _dmr_file(output, spin),
                f"density matrix DM(R) for spin {spin}",
            )
            blocks = read_csr_matrix_blocks(dmr_path)
            if (0, 0, 0) in blocks and len(blocks) == 1:
                density = blocks[(0, 0, 0)]
            elif len(blocks) == 1:
                density = next(iter(blocks.values()))
            else:
                raise ValueError(
                    f"{dmr_path}: multi-R DM(R) alone is not sufficient for "
                    "k-resolved Mayer analysis; use out_dmk=1 or keep "
                    "out_hsr/out_hsk output"
                )
            if density.shape != overlap.shape:
                raise ValueError(
                    f"spin {spin}: density matrix and overlap dimensions do not match"
                )
            data_files.append(str(dmr_path))
            population = density @ overlap
            for pair in selected_pairs:
                orders[pair] += spin_factor * _mayer_order(
                    density,
                    overlap,
                    atom_ranges[pair[0]],
                    atom_ranges[pair[1]],
                    product=population,
                )
    elif version == "develop":
        # Develop version: the k-resolved density matrices carry the k index,
        # the spin index and the k-point weight in their file name and header,
        # so they are read directly instead of being rebuilt from wavefunctions.
        density_files = _develop_density_files(output)
        if not density_files:
            raise FileNotFoundError(f"No dm*_nao.txt files found in {output}")

        kpoints = sorted({ik for ik, _ispin, _path in density_files})
        kpoint_table = read_develop_kpoints(density_files)
        weights = exact_kpoint_weights(kpoint_table, printed_scale=2.0 / nspin)
        overlaps: dict[int, np.ndarray] = {}
        for position, ik in enumerate(kpoints):
            overlap_path = _required_file(
                _develop_overlap_file(output, ik),
                f"overlap matrix for k-point {ik} (develop)",
            )
            matrix = overlaps.setdefault(
                ik,
                _read_develop_overlap(
                    overlap_path,
                    kpoint=kpoint_table.ibz[position].direct,
                ),
            )
            data_files.append(str(overlap_path))
            if matrix.shape != (basis_functions, basis_functions):
                raise ValueError(
                    f"k-point {ik}: overlap dimension does not match NAO basis size "
                    f"{basis_functions}"
                )
            spin_orders = {pair: 0.0 for pair in selected_pairs}
            channels = {
                ispin
                for entry_ik, ispin, _path in density_files
                if entry_ik == ik
            }
            expected_channels = {1} if nspin == 1 else {1, 2}
            if channels != expected_channels:
                raise ValueError(
                    f"k-point {ik}: found spin channels {sorted(channels)}; "
                    f"expected {sorted(expected_channels)} for nspin={nspin}"
                )
            for entry_ik, _ispin, path in density_files:
                if entry_ik != ik:
                    continue
                density = read_density_matrix_develop(path)
                if density.shape != matrix.shape:
                    raise ValueError(
                        f"k-point {ik}: density matrix dimension does not match overlap matrix"
                    )
                data_files.append(str(path))
                population = density @ matrix
                for pair in selected_pairs:
                    spin_orders[pair] += _mayer_order(
                        density,
                        matrix,
                        atom_ranges[pair[0]],
                        atom_ranges[pair[1]],
                        product=population,
                    )
            # DM(k) files store spin-resolved density matrices.  The
            # standard spin-unrestricted Mayer formula includes a factor two
            # for each spin channel.
            spin_factor = 2.0 if nspin == 2 else 1.0
            for pair in selected_pairs:
                orders[pair] += spin_factor * spin_orders[pair] / weights[position]
    else:
        # LTS version format (original code)
        if gamma_only:
            if int(_scalar(inputs.get("out_dm", 1))) != 1:
                raise ValueError("gamma-only Mayer analysis requires out_dm=1")
            overlap_path = _required_file(output / "data-0-S", "gamma overlap matrix")
            dm_up_path = _required_file(output / "SPIN1_DM", "spin-up density matrix")
            overlap = read_overlap_matrix(overlap_path)
            dm_up = read_density_matrix(dm_up_path)
            data_files.extend([str(overlap_path), str(dm_up_path)])
            if overlap.shape != (basis_functions, basis_functions) or dm_up.shape != overlap.shape:
                raise ValueError(f"matrix dimensions do not match NAO basis size {basis_functions}")
            dm_down = None
            if nspin == 2:
                dm_down_path = _required_file(output / "SPIN2_DM", "spin-down density matrix")
                dm_down = read_density_matrix(dm_down_path)
                data_files.append(str(dm_down_path))
                if dm_down.shape != overlap.shape:
                    raise ValueError("spin-down density matrix dimension does not match overlap matrix")
            population_up = dm_up @ overlap
            population_down = dm_down @ overlap if dm_down is not None else None
            for pair in selected_pairs:
                value = _mayer_order(
                    dm_up,
                    overlap,
                    atom_ranges[pair[0]],
                    atom_ranges[pair[1]],
                    product=population_up,
                )
                if population_down is not None:
                    value = 2.0 * (
                        value
                        + _mayer_order(
                            dm_down,
                            overlap,
                            atom_ranges[pair[0]],
                            atom_ranges[pair[1]],
                            product=population_down,
                        )
                    )
                orders[pair] = value
        else:
            wfc_paths = sorted(output.glob("WFC_NAO_K*.txt"), key=lambda path: int(re.search(r"K(\d+)", path.name).group(1)))
            if not wfc_paths:
                raise FileNotFoundError(f"No WFC_NAO_K*.txt files found in {output}")
            table = read_kpoint_table(_required_file(output / "kpoints", "k-point table"))
            weights = exact_kpoint_weights(table)
            if len(wfc_paths) not in (len(weights), 2 * len(weights)) or (nspin == 1 and len(wfc_paths) != len(weights)):
                raise ValueError(f"number of k-point weights ({len(weights)}) does not match WFC files ({len(wfc_paths)})")
            spin_paired = nspin == 2 and len(wfc_paths) == 2 * len(weights)
            matrices: dict[int, tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}

            def kpoint_values(ik: int, pairs: Sequence[tuple[int, int]]) -> dict[tuple[int, int], float]:
                """Return the spin-summed value of the pairs at one computed k-point."""
                if ik not in matrices:
                    overlap_path = _required_file(output / f"data-{ik}-S", f"overlap matrix for k-point {ik + 1}")
                    overlap = read_overlap_matrix(overlap_path)
                    up_wfc, up_occ = read_wfc_nao_k(wfc_paths[ik])
                    up_dm = calculate_density_matrix_k(up_wfc, up_occ)
                    if overlap.shape != (basis_functions, basis_functions) or up_dm.shape != overlap.shape:
                        raise ValueError(f"k-point {ik + 1}: matrix dimensions do not match NAO basis size {basis_functions}")
                    down_dm = None
                    if spin_paired:
                        down_wfc, down_occ = read_wfc_nao_k(wfc_paths[ik + len(weights)])
                        down_dm = calculate_density_matrix_k(down_wfc, down_occ)
                        if down_dm.shape != overlap.shape:
                            raise ValueError(f"k-point {ik + 1}: spin-down matrix dimension mismatch")
                        data_files.append(str(wfc_paths[ik + len(weights)]))
                    matrices[ik] = (overlap, up_dm, down_dm)
                    data_files.extend([str(overlap_path), str(wfc_paths[ik])])
                overlap, up_dm, down_dm = matrices[ik]
                population_up = up_dm @ overlap
                population_down = down_dm @ overlap if down_dm is not None else None
                values = {}
                for pair in pairs:
                    value = _mayer_order(
                        up_dm,
                        overlap,
                        atom_ranges[pair[0]],
                        atom_ranges[pair[1]],
                        product=population_up,
                    )
                    if population_down is not None:
                        value = 2.0 * (
                            value
                            + _mayer_order(
                                down_dm,
                                overlap,
                                atom_ranges[pair[0]],
                                atom_ranges[pair[1]],
                                product=population_down,
                            )
                        )
                    values[pair] = value
                return values

            if table.mesh:
                # symmetry=1: rebuild the stars that ABACUS reduced away.
                orders.update(
                    _symmetry_expanded_orders(table, weights, kpoint_values, selected_pairs, structure)
                )
            else:
                for ik, weight in enumerate(weights):
                    for pair, value in kpoint_values(ik, selected_pairs).items():
                        orders[pair] += value / weight

    fractions = structure.coords_direct
    lattice = periodic_lattice(structure)
    result_pairs = tuple(MayerPair(first + 1, second + 1, str(structure.atoms[first].element or structure.atoms[first].label), str(structure.atoms[second].element or structure.atoms[second].label), float(_minimum_distance(lattice, fractions[first], fractions[second])), float(orders[(first, second)])) for first, second in selected_pairs)
    return MayerAnalysis(str(job_path), calculation, nspin, gamma_only, basis_functions, str(output), tuple(dict.fromkeys(data_files)), result_pairs)


cal_mayer_bond_order = analyze_mayer_bond_order
