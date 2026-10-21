"""Shared LCAO matrix helpers.

The EDA workflow needs to hand a synthetic density matrix back to ABACUS.
ABACUS reads that matrix from a real-space CSR file such as
``dmrs1_nao.csr``.  This module provides the write side of that format so the
pEDA code can stay in Python while ABACUS remains responsible for the
density-to-energy conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU

if TYPE_CHECKING:
    from abacustools.data.population import KPointDensity


def _matrix_token(value: complex, precision: int) -> str:
    """Format one CSR value using the ABACUS text convention."""

    real = float(np.real(value))
    imag = float(np.imag(value))
    if abs(imag) <= 10.0 ** (-(precision + 2)):
        return f"{real:.{precision}e}"
    return f"({real:.{precision}e},{imag:.{precision}e})"


def _cell_header_lines(structure: AbacusSTRU) -> list[str]:
    """Return the unit-cell block written by ABACUS CSR matrix files."""

    lattice_constant = structure.metadata.get("lattice_constant", 1.0)
    try:
        lat0_bohr = float(lattice_constant)
    except (TypeError, ValueError) as error:
        raise ValueError("structure lattice_constant must be a number") from error
    if not np.isfinite(lat0_bohr) or lat0_bohr <= 0.0:
        raise ValueError("structure lattice_constant must be positive")

    lat0_angstrom = lat0_bohr * BOHR_TO_ANG
    cell = np.asarray(structure.cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("structure must have a finite 3x3 cell")
    if lat0_angstrom <= 0.0:
        raise ValueError("structure lattice constant must be positive")

    # ABACUS stores latvec in units of lat0 and writes lat0 in Angstrom.
    latvec = cell / lat0_angstrom
    lattice_name = str(structure.metadata.get("lattice_name", "test"))

    labels: list[str] = []
    for atom in structure.atoms:
        if atom.label not in labels:
            labels.append(atom.label)

    lines = [f" {lattice_name}", f" {lat0_angstrom:.15g}"]
    lines.extend(" " + " ".join(f"{value:.15g}" for value in row) for row in latvec)
    lines.append(" " + " ".join(labels))
    lines.append(
        " "
        + " ".join(str(sum(atom.label == label for atom in structure.atoms)) for label in labels)
    )
    lines.append(" Direct")
    direct_coords = structure.coords_direct
    for label in labels:
        for index, atom in enumerate(structure.atoms):
            if atom.label != label:
                continue
            direct = direct_coords[index]
            lines.append(" " + " ".join(f"{float(value):.15g}" for value in direct))
    return lines


def _csr_rows(
    matrix: np.ndarray,
    sparse_threshold: float,
) -> tuple[list[complex], list[int], list[int]]:
    """Return CSR values, column indices and row pointers for one R block."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("every CSR block must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("CSR matrix contains non-finite values")
    if sparse_threshold < 0.0 or not np.isfinite(sparse_threshold):
        raise ValueError("sparse_threshold must be a finite non-negative number")

    values: list[complex] = []
    columns: list[int] = []
    row_pointers = [0]
    for row in range(matrix.shape[0]):
        selected = np.flatnonzero(np.abs(matrix[row, :]) > sparse_threshold)
        values.extend(complex(matrix[row, col]) for col in selected)
        columns.extend(int(col) for col in selected)
        row_pointers.append(len(values))
    return values, columns, row_pointers


def write_csr_matrix_blocks(
    path: str | Path,
    blocks: Mapping[tuple[int, int, int], np.ndarray],
    *,
    structure: AbacusSTRU,
    nspin: int = 1,
    spin: int = 1,
    precision: int = 8,
    sparse_threshold: float = 1e-10,
    label: str = "density matrix",
) -> None:
    """Write an ABACUS real-space CSR matrix file.

    The output follows the text layout used by ``out_dmr`` and
    ``init_chg=dm``.  It is intentionally limited to the information ABACUS
    needs to reconstruct the matrix: the cell header, one spin channel, and
    one CSR block per Bravais lattice vector.

    Args:
        path: Destination file, for example ``dmrs1_nao.csr``.
        blocks: Mapping from ``(Rx, Ry, Rz)`` to a square dense matrix.
        structure: Structure whose cell header is written.
        nspin: Number of spin channels declared in the file header.
        spin: One-based spin index written in the file header.
        precision: Number of digits used for floating-point values.
        sparse_threshold: Entries with ``abs(value) <= threshold`` are omitted.
        label: Matrix label used in the human-readable header.

    Raises:
        ValueError: If the matrix blocks, spin indices or metadata are invalid.
    """

    if nspin not in (1, 2, 4):
        raise ValueError("nspin must be 1, 2 or 4")
    if not 1 <= spin <= nspin:
        raise ValueError("spin must be a one-based index within nspin")
    if precision <= 0:
        raise ValueError("precision must be positive")
    if not blocks:
        raise ValueError("at least one R block is required")

    ordered = sorted(blocks.items(), key=lambda item: item[0])
    dimension: int | None = None
    for key, matrix in ordered:
        if len(key) != 3 or any(not isinstance(value, (int, np.integer)) for value in key):
            raise ValueError(f"invalid R block key: {key!r}")
        array = np.asarray(matrix)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"R block {key} must be a square matrix")
        if dimension is None:
            dimension = int(array.shape[0])
        elif array.shape != (dimension, dimension):
            raise ValueError("all R blocks must have the same dimensions")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"R block {key} contains non-finite values")
    if dimension is None or dimension <= 0:
        raise ValueError("matrix dimension must be positive")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(" --- Ionic Step 1 ---\n")
        handle.write(f" # print {label} in real space {label}(R)\n")
        handle.write(f" {nspin} # number of spin directions\n")
        handle.write(f" {spin} # spin index\n")
        handle.write(f" {dimension} # number of localized basis\n")
        handle.write(f" {len(ordered)} # number of Bravais lattice vector R\n")
        handle.write("\n")
        for line in _cell_header_lines(structure):
            handle.write(line + "\n")
        handle.write("\n")
        handle.write(" #----------------------------------------------------------------------#\n")
        handle.write(" #                               CSR Format                             #\n")
        handle.write(" # The outer loop corresponds to the number of Bravais lattice vectors. #\n")
        handle.write(" # The first line contains the index of the Bravais lattice vector      #\n")
        handle.write(" # (Rx, Ry, Rz), followed by the number of non-zero elements.           #\n")
        handle.write(" # The subsequent lines consist of three blocks of data, which are      #\n")
        handle.write(" # values, column indices, row pointers.                                #\n")
        handle.write(" #----------------------------------------------------------------------#\n")
        handle.write("\n")

        for (rx, ry, rz), matrix in ordered:
            values, columns, row_pointers = _csr_rows(np.asarray(matrix), float(sparse_threshold))
            handle.write(f" {rx} {ry} {rz} {len(values)}\n")
            handle.write(" # CSR values\n")
            if values:
                handle.write(
                    " " + " ".join(_matrix_token(value, precision) for value in values) + "\n"
                )
            handle.write("\n # CSR column indices\n")
            if columns:
                handle.write(" " + " ".join(str(value) for value in columns) + "\n")
            handle.write("\n # CSR row pointers\n")
            handle.write(" " + " ".join(str(value) for value in row_pointers) + "\n")
            handle.write("\n")


def _validate_kpoint_weight_pairs(
    kpoints: Sequence[tuple[Sequence[float], float]],
) -> list[tuple[tuple[float, float, float], float]]:
    """Validate and normalize k-point coordinates and weights."""

    if not kpoints:
        raise ValueError("at least one k-point is required")
    normalized: list[tuple[tuple[float, float, float], float]] = []
    total = 0.0
    for point, weight in kpoints:
        if len(point) != 3:
            raise ValueError("each k-point must have three direct coordinates")
        direct = tuple(float(value) for value in point)
        if not np.all(np.isfinite(direct)):
            raise ValueError("k-point coordinates must be finite")
        value = float(weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("k-point weights must be finite and non-negative")
        normalized.append((direct, value))
        total += value
    if total <= 0.0:
        raise ValueError("k-point weights must sum to a positive value")
    if abs(total - 1.0) > 1e-8:
        raise ValueError("k-point weights must sum to one")
    return normalized


def kpoint_density_to_real_space(
    kpoints: Sequence[tuple[Sequence[float], float, np.ndarray]],
    r_blocks: Sequence[tuple[int, int, int]],
) -> dict[tuple[int, int, int], np.ndarray]:
    """Fourier transform k-resolved density matrices to R-space blocks.

    ABACUS uses ``H(k) = sum_R H(R) exp(-i k R)`` for reciprocal-space
    matrices.  The corresponding density-matrix convention used by
    ``cal_dmr`` is the conjugate transform::

        D(R) = sum_k w_k D(k) exp(+i k R)

    Args:
        kpoints: ``(direct_k, weight, density)`` entries.  Weights must sum
            to one.  Every density matrix must have the same shape.
        r_blocks: Bravais lattice vectors to keep.

    Returns:
        A mapping from ``(Rx, Ry, Rz)`` to dense density matrices.

    Raises:
        ValueError: If k-points, weights or matrix shapes are inconsistent.
    """

    entries = _validate_kpoint_weight_pairs([(point, weight) for point, weight, _ in kpoints])
    if not r_blocks:
        raise ValueError("at least one R block is required")
    if len(entries) != len(kpoints):
        raise AssertionError("k-point validation changed the number of entries")

    dimension: int | None = None
    for _point, _weight, density in kpoints:
        matrix = np.asarray(density)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("every density matrix must be square")
        if dimension is None:
            dimension = int(matrix.shape[0])
        elif matrix.shape != (dimension, dimension):
            raise ValueError("all density matrices must have the same shape")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("density matrices must be finite")
    assert dimension is not None

    result: dict[tuple[int, int, int], np.ndarray] = {}
    for r_block in r_blocks:
        key = tuple(int(value) for value in r_block)
        if len(key) != 3:
            raise ValueError("each R block must have three integer components")
        total = np.zeros((dimension, dimension), dtype=np.complex128)
        for (direct, weight), (_point, _unused, density) in zip(entries, kpoints):
            phase = np.exp(
                2j * np.pi * (direct[0] * key[0] + direct[1] * key[1] + direct[2] * key[2])
            )
            total += weight * phase * np.asarray(density, dtype=np.complex128)
        result[key] = total
    return result


def real_space_density_to_kpoints(
    blocks: Mapping[tuple[int, int, int], np.ndarray],
    kpoints: Sequence[tuple[Sequence[float], float]],
) -> list[tuple[tuple[float, float, float], float, np.ndarray]]:
    """Fourier transform R-space density matrices back to k-space.

    This is the inverse of :func:`kpoint_density_to_real_space`::

        D(k) = sum_R D(R) exp(-i k R)

    Args:
        blocks: Mapping from ``(Rx, Ry, Rz)`` to dense matrices.
        kpoints: ``(direct_k, weight)`` entries.  Weights must sum to one.

    Returns:
        A list of ``(direct_k, weight, density)`` entries.

    Raises:
        ValueError: If the R blocks or k-points are inconsistent.
    """

    entries = _validate_kpoint_weight_pairs(kpoints)
    if not blocks:
        raise ValueError("at least one R block is required")
    dimension: int | None = None
    for key, matrix in blocks.items():
        if len(key) != 3:
            raise ValueError("each R block must have three components")
        array = np.asarray(matrix)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"R block {key} must be square")
        if dimension is None:
            dimension = int(array.shape[0])
        elif array.shape != (dimension, dimension):
            raise ValueError("all R blocks must have the same shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"R block {key} must be finite")
    assert dimension is not None

    result: list[tuple[tuple[float, float, float], float, np.ndarray]] = []
    for direct, weight in entries:
        total = np.zeros((dimension, dimension), dtype=np.complex128)
        for (rx, ry, rz), matrix in blocks.items():
            phase = np.exp(-2j * np.pi * (direct[0] * rx + direct[1] * ry + direct[2] * rz))
            total += phase * np.asarray(matrix, dtype=np.complex128)
        result.append((direct, weight, total))
    return result


class LCAODataError(RuntimeError):
    """Raised when LCAO density-matrix data cannot be loaded."""


def _scalar(value: Any) -> Any:
    """Return the first element of an ABACUS scalar field."""

    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def load_lcao_kpoint_densities(
    job: str | Path,
    *,
    inputs: Any = None,
    nspin: Optional[int] = None,
    gamma_only: Optional[bool] = None,
) -> list[KPointDensity]:
    """Load the k-resolved density and overlap matrices of an LCAO job.

    This is the public entry point shared by population analysis, COHP/COOP and
    the future EDA workflow.  The implementation currently delegates to the
    existing population loader so the two analysis families cannot drift apart
    while the LCAO data layer is being consolidated.

    Args:
        job: ABACUS job directory.
        inputs: Parsed ``INPUT`` mapping, read from ``job`` when omitted.
        nspin: Override the spin channel count; otherwise read from ``INPUT``.
        gamma_only: Override the Gamma-only flag; otherwise read from
            ``INPUT``.

    Returns:
        One :class:`KPointDensity` per computed k-point, in file order.

    Raises:
        LCAODataError: If the job, ``INPUT``, output directory, or matrix
            outputs cannot be read consistently.
    """

    job_path = Path(job).resolve()
    if not job_path.is_dir():
        raise LCAODataError(f"job directory does not exist: {job_path}")

    if inputs is None:
        input_path = job_path / "INPUT"
        if not input_path.is_file():
            raise LCAODataError(f"INPUT file does not exist: {input_path}")
        inputs = ReadInput(str(input_path))
    if not inputs:
        raise LCAODataError(f"could not read INPUT in {job_path}")

    suffix = str(_scalar(inputs.get("suffix", "ABACUS")))
    output = job_path / f"OUT.{suffix}"
    if not output.is_dir():
        raise LCAODataError(f"ABACUS output directory does not exist: {output}")

    actual_nspin = int(_scalar(inputs.get("nspin", 1))) if nspin is None else int(nspin)
    if actual_nspin not in (1, 2):
        raise LCAODataError(f"LCAO density loading supports nspin=1 or 2, got {actual_nspin}")
    if gamma_only is None:
        actual_gamma = bool(int(_scalar(inputs.get("gamma_only", 0)) or 0))
    else:
        actual_gamma = bool(gamma_only)

    from abacustools.data.population import (
        PopulationAnalysisError,
        load_kpoint_densities_from_output,
    )

    try:
        return load_kpoint_densities_from_output(output, actual_nspin, actual_gamma)
    except (PopulationAnalysisError, OSError, ValueError) as error:
        raise LCAODataError(str(error)) from error


def _validated_full_atom_ranges(
    full_atom_ranges: Sequence[tuple[int, int]],
    full_dimension: Optional[int],
) -> tuple[list[tuple[int, int]], int]:
    """Validate atom-to-orbital ranges and return the full basis size."""

    if not full_atom_ranges:
        raise ValueError("full_atom_ranges must not be empty")
    ranges: list[tuple[int, int]] = []
    previous_stop = 0
    for index, item in enumerate(full_atom_ranges):
        if len(item) != 2:
            raise ValueError(f"atom range {index} must contain start and stop")
        start, stop = int(item[0]), int(item[1])
        if start < 0 or stop <= start:
            raise ValueError(f"atom range {index} is invalid: {(start, stop)}")
        if start < previous_stop:
            raise ValueError("full_atom_ranges must be ordered and non-overlapping")
        ranges.append((start, stop))
        previous_stop = stop
    inferred = previous_stop
    if full_dimension is None:
        return ranges, inferred
    dimension = int(full_dimension)
    if dimension < inferred:
        raise ValueError(
            f"full_dimension={dimension} is smaller than the last atom stop={inferred}"
        )
    return ranges, dimension


def _fragment_axis_slices(
    fragment_atom_indices: Sequence[int],
    full_atom_ranges: Sequence[tuple[int, int]],
    fragment_dimension: int,
    full_dimension: Optional[int],
) -> tuple[list[tuple[slice, slice]], int]:
    """Return corresponding fragment and full basis slices."""

    if not fragment_atom_indices:
        raise ValueError("at least one fragment atom is required")
    indices = [int(index) for index in fragment_atom_indices]
    if len(set(indices)) != len(indices):
        raise ValueError("fragment atom indices must be unique")
    ranges, full_size = _validated_full_atom_ranges(full_atom_ranges, full_dimension)
    pairs: list[tuple[slice, slice]] = []
    offset = 0
    for index in indices:
        if index < 0 or index >= len(ranges):
            raise ValueError(f"fragment atom index is outside the full structure: {index}")
        start, stop = ranges[index]
        length = stop - start
        pairs.append((slice(offset, offset + length), slice(start, stop)))
        offset += length
    if offset != int(fragment_dimension):
        raise ValueError(
            f"fragment matrix dimension is {fragment_dimension}, but the selected "
            f"atoms contain {offset} basis functions"
        )
    return pairs, full_size


def _fragment_index_map(
    fragment_atom_indices: Sequence[int],
    full_atom_ranges: Sequence[tuple[int, int]],
    fragment_dimension: int,
    full_dimension: Optional[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return fragment and full basis indices in matching order."""

    pairs, full_size = _fragment_axis_slices(
        fragment_atom_indices,
        full_atom_ranges,
        fragment_dimension,
        full_dimension,
    )
    fragment_indices = np.concatenate(
        [np.arange(fragment_slice.start, fragment_slice.stop) for fragment_slice, _ in pairs]
    )
    full_indices = np.concatenate(
        [np.arange(full_slice.start, full_slice.stop) for _, full_slice in pairs]
    )
    return fragment_indices, full_indices, full_size


def embed_fragment_matrix(
    matrix: np.ndarray,
    fragment_atom_indices: Sequence[int],
    full_atom_ranges: Sequence[tuple[int, int]],
    *,
    full_dimension: Optional[int] = None,
) -> np.ndarray:
    """Embed a fragment density-like matrix into the full NAO basis.

    The fragment matrix is assumed to use the atom order in
    ``fragment_atom_indices``.  That is the order produced by selecting a
    subset from the full structure with :meth:`AbacusSTRU.create_subset`.

    Args:
        matrix: Square fragment matrix in the fragment NAO basis.
        fragment_atom_indices: Zero-based atom indices in the full structure,
            in the same order as the fragment basis.
        full_atom_ranges: Half-open NAO basis range of every full atom.
        full_dimension: Optional full basis size; inferred from the ranges.

    Returns:
        A dense matrix in the full NAO basis with zeros outside the fragment
        blocks.

    Raises:
        ValueError: If the matrix, atom indices or ranges are inconsistent.
    """

    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("fragment matrix must be square")
    if not np.all(np.isfinite(array)):
        raise ValueError("fragment matrix contains non-finite values")
    fragment_indices, full_indices, full_size = _fragment_index_map(
        fragment_atom_indices,
        full_atom_ranges,
        int(array.shape[0]),
        full_dimension,
    )
    result = np.zeros((full_size, full_size), dtype=array.dtype)
    result[np.ix_(full_indices, full_indices)] = array[np.ix_(fragment_indices, fragment_indices)]
    return result


def embed_fragment_coefficients(
    coefficients: np.ndarray,
    fragment_atom_indices: Sequence[int],
    full_atom_ranges: Sequence[tuple[int, int]],
    *,
    full_dimension: Optional[int] = None,
) -> np.ndarray:
    """Embed fragment orbital coefficients into the full NAO basis.

    Rows follow the fragment NAO basis; columns are bands or other secondary
    dimensions and are preserved unchanged.

    Args:
        coefficients: Fragment coefficient matrix with shape
            ``(fragment_basis, n_columns)``.
        fragment_atom_indices: Zero-based atom indices in the full structure,
            in the same order as the fragment basis.
        full_atom_ranges: Half-open NAO basis range of every full atom.
        full_dimension: Optional full basis size; inferred from the ranges.

    Returns:
        A dense matrix with shape ``(full_basis, n_columns)``.

    Raises:
        ValueError: If the coefficients, atom indices or ranges are inconsistent.
    """

    array = np.asarray(coefficients)
    if array.ndim != 2:
        raise ValueError("fragment coefficients must be a two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("fragment coefficients contain non-finite values")
    fragment_indices, full_indices, full_size = _fragment_index_map(
        fragment_atom_indices,
        full_atom_ranges,
        int(array.shape[0]),
        full_dimension,
    )
    result = np.zeros((full_size, array.shape[1]), dtype=array.dtype)
    result[full_indices, :] = array[fragment_indices, :]
    return result
