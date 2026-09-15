"""Mayer bond-order analysis for ABACUS LCAO calculations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

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
    if isinstance(value, (list, tuple)) and len(value) == 1:
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
    """Read weights from the ``KPOINTS DIRECT_X ... WEIGHT`` table."""

    path = Path(kpoints_file)
    _required_file(path, "k-point table")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    table = next((i for i, line in enumerate(lines) if "KPOINTS" in line.upper() and "DIRECT_X" in line.upper()), None)
    if table is None:
        raise ValueError(f"{path}: cannot find KPOINTS DIRECT_X table")
    weights: list[float] = []
    for line in lines[table + 1 :]:
        if not line.strip():
            if weights:
                break
            continue
        tokens = line.split()
        if len(tokens) < 5:
            continue
        try:
            index, weight = int(tokens[0]), _number(tokens[4])
        except ValueError:
            continue
        if index != len(weights) + 1 or weight < 0:
            raise ValueError(f"{path}: invalid k-point row {line!r}")
        weights.append(weight)
    if not weights or not np.all(np.isfinite(weights)) or sum(weights) <= 0:
        raise ValueError(f"{path}: no valid k-point weights found")
    return np.asarray(weights, dtype=float)




def detect_matrix_format(output_dir: Path, out_dmk: int) -> str:
    """Return ``"develop"`` or ``"lts"`` for an LCAO output directory.

    The develop layout is written by ``out_dmk=1`` and stores the k-resolved
    density matrices as ``dm*_nao.txt``; the LTS layout stores ``data-*-S``
    overlap matrices (or ``SPIN1_DM`` for gamma-only runs).  Some directories
    contain files of both layouts, so the requested inputs decide, with the
    available data as a fallback.

    Args:
        output_dir: ABACUS ``OUT.*`` directory.
        out_dmk: Value of the ``out_dmk`` input parameter.

    Returns:
        str: ``"develop"`` when the k-resolved density matrices are used.
    """
    has_develop = bool(list(output_dir.glob("dm*_nao.txt")))
    has_lts = bool(list(output_dir.glob("data-*-S"))) or (output_dir / "SPIN1_DM").is_file()
    if out_dmk == 1 and has_develop:
        return "develop"
    if not has_lts and has_develop:
        return "develop"
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


def _mayer_order(dm: np.ndarray, overlap: np.ndarray, first: tuple[int, int], second: tuple[int, int]) -> float:
    left_start, left_end = first
    right_start, right_end = second
    ps = dm @ overlap
    return float(np.sum(ps[left_start:left_end, right_start:right_end] * ps[right_start:right_end, left_start:left_end].T).real)


def cal_mayer_bond_order_between_atom_pair(
    i: int,
    j: int,
    atom_orb_ranges: list[tuple[int, int]],
    ovlp_mat: np.ndarray,
    dm: np.ndarray,
    dm_dn: Optional[np.ndarray] = None,
) -> float:
    """Calculate one gamma-point Mayer order using zero-based atom indices."""

    value = _mayer_order(dm, ovlp_mat, atom_orb_ranges[i], atom_orb_ranges[j])
    if dm_dn is not None:
        value = 2.0 * (value + _mayer_order(dm_dn, ovlp_mat, atom_orb_ranges[i], atom_orb_ranges[j]))
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

    A gamma-only run writes a single ``sk_nao.txt``; a multi-k run writes one
    ``sk{ik}_nao.txt`` per k-point.
    """
    gamma = output / "sk_nao.txt"
    if gamma.is_file():
        return gamma
    return output / f"sk{ik}_nao.txt"


def analyze_mayer_bond_order(job: str | Path, *, cutoff: Optional[float] = None, pairs: Optional[str] = None, pairs_file: Optional[str | Path] = None) -> MayerAnalysis:
    """Analyze Mayer bond orders from an ABACUS LCAO job directory.
    
    Supports both LTS 3.10.1 format (data-*-S, WFC_NAO_K*.txt) and 
    develop version format (sk*_nao.txt, wfk*_nao.txt).
    """

    job_path = Path(job).resolve()
    inputs = ReadInput(str(_required_file(job_path / "INPUT", "INPUT file")))
    calculation = str(inputs.get("calculation", "scf")).lower()
    nspin = int(_scalar(inputs.get("nspin", 1)))
    if nspin not in (1, 2):
        raise ValueError(f"Mayer analysis supports nspin=1 or 2, got {nspin}")
    if str(_scalar(inputs.get("basis_type", "pw"))).lower() != "lcao":
        raise ValueError("Mayer analysis requires basis_type=lcao")
    if int(_scalar(inputs.get("out_mat_hs", 1))) != 1:
        raise ValueError("Mayer analysis requires out_mat_hs=1")
    gamma_only = bool(int(_scalar(inputs.get("gamma_only", 0)) or 0))
    suffix = str(_scalar(inputs.get("suffix", "ABACUS")))
    output = _required_directory(job_path / f"OUT.{suffix}", "ABACUS output directory")
    structure_path = job_path / str(_scalar(inputs.get("stru_file", "STRU")))
    structure = AbacusSTRU.read(str(_required_file(structure_path, "STRU file")))
    if structure is None:
        raise ValueError(f"cannot read structure: {structure_path}")

    # Choose the layout that matches the requested outputs and the files present
    version = detect_matrix_format(
        output, int(_scalar(inputs.get("out_dmk", 0)) or 0)
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

    if version == "develop":
        # Develop version: the k-resolved density matrices carry the k index,
        # the spin index and the k-point weight in their file name and header,
        # so they are read directly instead of being rebuilt from wavefunctions.
        density_files = _develop_density_files(output)
        if not density_files:
            raise FileNotFoundError(f"No dm*_nao.txt files found in {output}")

        kpoints = sorted({ik for ik, _ispin, _path in density_files})
        weight = 1.0 / len(kpoints)
        overlaps: dict[int, np.ndarray] = {}
        for ik in kpoints:
            overlap_path = _required_file(
                _develop_overlap_file(output, ik),
                f"overlap matrix for k-point {ik} (develop)",
            )
            matrix = overlaps.setdefault(ik, read_overlap_matrix_develop(overlap_path))
            data_files.append(str(overlap_path))
            if matrix.shape != (basis_functions, basis_functions):
                raise ValueError(
                    f"k-point {ik}: overlap dimension does not match NAO basis size "
                    f"{basis_functions}"
                )
            spin_orders = {pair: 0.0 for pair in selected_pairs}
            channels = 0
            for entry_ik, _ispin, path in density_files:
                if entry_ik != ik:
                    continue
                density = read_density_matrix_develop(path)
                if density.shape != matrix.shape:
                    raise ValueError(
                        f"k-point {ik}: density matrix dimension does not match overlap matrix"
                    )
                data_files.append(str(path))
                channels += 1
                for pair in selected_pairs:
                    spin_orders[pair] += _mayer_order(
                        density, matrix, atom_ranges[pair[0]], atom_ranges[pair[1]]
                    )
            # The two spin channels share one occupation weight, as in the
            # LTS density matrices, so they are summed and weighted like a
            # single doubly-occupied manifold.
            factor = 2.0 if (nspin == 2 and channels == 2) else 1.0
            for pair in selected_pairs:
                orders[pair] += factor * spin_orders[pair] / weight
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
            for pair in selected_pairs:
                value = _mayer_order(dm_up, overlap, atom_ranges[pair[0]], atom_ranges[pair[1]])
                if dm_down is not None:
                    value = 2.0 * (value + _mayer_order(dm_down, overlap, atom_ranges[pair[0]], atom_ranges[pair[1]]))
                orders[pair] = value
        else:
            wfc_paths = sorted(output.glob("WFC_NAO_K*.txt"), key=lambda path: int(re.search(r"K(\d+)", path.name).group(1)))
            if not wfc_paths:
                raise FileNotFoundError(f"No WFC_NAO_K*.txt files found in {output}")
            weights = read_kpoint_weights(_required_file(output / "kpoints", "k-point table"))
            if len(wfc_paths) not in (len(weights), 2 * len(weights)) or (nspin == 1 and len(wfc_paths) != len(weights)):
                raise ValueError(f"number of k-point weights ({len(weights)}) does not match WFC files ({len(wfc_paths)})")
            spin_paired = nspin == 2 and len(wfc_paths) == 2 * len(weights)
            for ik, weight in enumerate(weights):
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
                for pair in selected_pairs:
                    value = _mayer_order(up_dm, overlap, atom_ranges[pair[0]], atom_ranges[pair[1]])
                    if down_dm is not None:
                        value = 2.0 * (value + _mayer_order(down_dm, overlap, atom_ranges[pair[0]], atom_ranges[pair[1]]))
                    orders[pair] += value / weight
                data_files.extend([str(overlap_path), str(wfc_paths[ik])])
                if spin_paired:
                    data_files.append(str(wfc_paths[ik + len(weights)]))

    fractions = structure.coords_direct
    lattice = periodic_lattice(structure)
    result_pairs = tuple(MayerPair(first + 1, second + 1, str(structure.atoms[first].element or structure.atoms[first].label), str(structure.atoms[second].element or structure.atoms[second].label), float(_minimum_distance(lattice, fractions[first], fractions[second])), float(orders[(first, second)])) for first, second in selected_pairs)
    return MayerAnalysis(str(job_path), calculation, nspin, gamma_only, basis_functions, str(output), tuple(dict.fromkeys(data_files)), result_pairs)


cal_mayer_bond_order = analyze_mayer_bond_order
