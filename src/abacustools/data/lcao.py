"""Shared LCAO matrix helpers.

The EDA workflow needs to hand a synthetic density matrix back to ABACUS.
ABACUS reads that matrix from a real-space CSR file such as
``dmrs1_nao.csr``.  This module provides the write side of that format so the
pEDA code can stay in Python while ABACUS remains responsible for the
density-to-energy conversion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG
from abacustools.data.mayer import read_wfc_nao_k_data
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


@dataclass(frozen=True)
class OrthogonalizedFragmentBasis:
    """Orthonormalized occupied and virtual fragment orbitals.

    Attributes:
        occupied: Löwdin-orthonormalized occupied orbitals.
        virtual: Gram-Schmidt plus Löwdin orthonormalized virtual orbitals.
        combined: ``[occupied, virtual]`` in one coefficient matrix.
        discarded_virtuals: Number of virtual columns removed because they
            were linearly dependent on the occupied or other virtual orbitals.
    """

    occupied: np.ndarray
    virtual: np.ndarray
    combined: np.ndarray
    discarded_virtuals: int


def _hermitian_inverse_sqrt(
    matrix: np.ndarray,
    *,
    threshold: float,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the inverse square root and eigensystem of a Hermitian matrix."""

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{context} has non-finite eigenvalues")
    if np.any(eigenvalues <= threshold):
        raise ValueError(f"{context} is not positive definite above threshold {threshold:g}")
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    return inverse_sqrt, eigenvalues, eigenvectors


def orthogonalize_fragment_orbitals(
    occupied: np.ndarray,
    virtual: np.ndarray,
    overlap: np.ndarray,
    *,
    threshold: float = 1e-8,
) -> OrthogonalizedFragmentBasis:
    """Orthonormalize occupied and virtual fragment orbitals in a NAO basis.

    The procedure follows pEDA:

    1. Löwdin-orthogonalize the occupied fragment orbitals;
    2. project the virtual fragment orbitals onto the orthogonal complement of
       the occupied space (Gram-Schmidt step);
    3. Löwdin-orthogonalize the projected virtual orbitals and discard
       linearly dependent directions.

    Args:
        occupied: Occupied fragment coefficients with shape
            ``(n_basis, n_occupied)``.
        virtual: Virtual fragment coefficients with shape
            ``(n_basis, n_virtual)``.  The virtual set may contain redundant
            columns from multiple fragments; they are removed by the final
            Löwdin step.
        overlap: Full-basis overlap matrix ``S``.
        threshold: Positive eigenvalue threshold for the occupied and virtual
            metric matrices.

    Returns:
        An :class:`OrthogonalizedFragmentBasis` whose columns satisfy
        ``C.T @ S @ C = I``.

    Raises:
        ValueError: If the matrices are inconsistent, the overlap is not
            Hermitian, the occupied orbitals are linearly dependent, or no
            virtual direction survives.
    """

    occ = np.asarray(occupied)
    vir = np.asarray(virtual)
    s = np.asarray(overlap)
    if occ.ndim != 2 or vir.ndim != 2 or s.ndim != 2:
        raise ValueError("occupied, virtual and overlap must be two-dimensional")
    n_basis = int(s.shape[0])
    if s.shape != (n_basis, n_basis):
        raise ValueError("overlap must be square")
    if occ.shape[0] != n_basis or vir.shape[0] != n_basis:
        raise ValueError("occupied and virtual coefficients must match the overlap dimension")
    if occ.shape[1] <= 0:
        raise ValueError("at least one occupied fragment orbital is required")
    if not np.all(np.isfinite(occ)) or not np.all(np.isfinite(vir)) or not np.all(np.isfinite(s)):
        raise ValueError("fragment orbitals and overlap must be finite")
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError("threshold must be a finite non-negative number")
    if not np.allclose(s, s.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("overlap matrix must be Hermitian")

    occupied_metric = occ.conj().T @ s @ occ
    occupied_inverse_sqrt, _values, _vectors = _hermitian_inverse_sqrt(
        occupied_metric,
        threshold=threshold,
        context="occupied fragment overlap",
    )
    occupied_orth = occ @ occupied_inverse_sqrt

    virtual_projected = vir - occupied_orth @ (occupied_orth.conj().T @ s @ vir)
    virtual_metric = virtual_projected.conj().T @ s @ virtual_projected
    virtual_metric = 0.5 * (virtual_metric + virtual_metric.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(virtual_metric)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("virtual fragment overlap has non-finite eigenvalues")
    keep = eigenvalues > threshold
    discarded = int(np.count_nonzero(~keep))
    if not np.any(keep):
        raise ValueError("no linearly independent virtual fragment orbital remains")
    virtual_orth = (
        virtual_projected @ eigenvectors[:, keep] @ np.diag(1.0 / np.sqrt(eigenvalues[keep]))
    )

    combined = np.hstack((occupied_orth, virtual_orth))
    metric = combined.conj().T @ s @ combined
    identity = np.eye(metric.shape[0], dtype=metric.dtype)
    if not np.allclose(metric, identity, atol=1e-7, rtol=1e-7):
        raise ValueError("fragment orbital orthogonalization did not produce an orthonormal basis")

    return OrthogonalizedFragmentBasis(
        occupied=occupied_orth,
        virtual=virtual_orth,
        combined=combined,
        discarded_virtuals=discarded,
    )


def density_matrix_in_basis(
    density_nao: np.ndarray,
    overlap: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Transform a NAO density matrix into a coefficient basis.

    For NAO coefficients ``C`` whose columns are the target orbitals, the
    density matrix in that basis is::

        P_basis = C† S P_NAO S C

    Args:
        density_nao: Density matrix in the NAO basis, shape ``(n, n)``.
        overlap: NAO overlap matrix ``S``, shape ``(n, n)``.
        coefficients: Target orbital coefficients, shape ``(n, m)``.

    Returns:
        The transformed density matrix with shape ``(m, m)``.

    Raises:
        ValueError: If shapes, Hermiticity or finiteness are inconsistent.
    """

    density = np.asarray(density_nao)
    s = np.asarray(overlap)
    c = np.asarray(coefficients)
    if density.ndim != 2 or s.ndim != 2 or c.ndim != 2:
        raise ValueError("density, overlap and coefficients must be two-dimensional")
    n_basis = int(s.shape[0])
    if s.shape != (n_basis, n_basis):
        raise ValueError("overlap must be square")
    if density.shape != (n_basis, n_basis):
        raise ValueError("density must match the overlap dimension")
    if c.shape[0] != n_basis:
        raise ValueError("coefficients must have one row per NAO basis function")
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(c)):
        raise ValueError("density, overlap and coefficients must be finite")
    if not np.allclose(s, s.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("overlap matrix must be Hermitian")
    if not np.allclose(density, density.conj().T, atol=1e-8, rtol=1e-8):
        raise ValueError("density matrix must be Hermitian")

    transformed = c.conj().T @ s @ density @ s @ c
    transformed = 0.5 * (transformed + transformed.conj().T)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("transformed density matrix contains non-finite values")
    return transformed


def build_p0_lambda(
    occupations: Sequence[float],
    *,
    n_orbitals: Optional[int] = None,
) -> np.ndarray:
    """Build the diagonal P0 matrix in the orthonormal fragment basis.

    Args:
        occupations: Occupations of the occupied fragment orbitals.  Values
            are typically 1 for one spin channel or 2 for a closed-shell
            spin-summed density.
        n_orbitals: Total number of orbitals in the orthonormal fragment basis,
            including virtual orbitals.  Defaults to ``len(occupations)``.

    Returns:
        A diagonal matrix with the occupations followed by zeros for the
        virtual orbitals.

    Raises:
        ValueError: If occupations are invalid or exceed the target dimension.
    """

    values = np.asarray(occupations, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("occupations must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("occupations must be finite")
    if np.any(values < 0.0) or np.any(values > 2.0):
        raise ValueError("occupations must lie between 0 and 2")
    size = values.size if n_orbitals is None else int(n_orbitals)
    if size < values.size:
        raise ValueError("n_orbitals must not be smaller than the number of occupations")
    diagonal = np.zeros(size, dtype=float)
    diagonal[: values.size] = values
    return np.diag(diagonal)


def build_p0_nao(
    occupied_lambda: np.ndarray,
    occupations: Sequence[float],
) -> np.ndarray:
    """Build the P0 density matrix from orthonormal occupied orbitals.

    Args:
        occupied_lambda: Occupied orbitals in the orthonormal fragment basis,
            shape ``(n_basis, n_occupied)``.
        occupations: Occupations of those orbitals.

    Returns:
        The Hermitian NAO density matrix
        ``sum_i occupation_i * lambda_i lambda_i†``.

    Raises:
        ValueError: If dimensions or occupations are inconsistent.
    """

    orbitals = np.asarray(occupied_lambda)
    values = np.asarray(occupations, dtype=float)
    if orbitals.ndim != 2:
        raise ValueError("occupied_lambda must be a two-dimensional matrix")
    if values.ndim != 1 or values.size != orbitals.shape[1]:
        raise ValueError("occupations must match the number of occupied orbitals")
    if not np.all(np.isfinite(orbitals)) or not np.all(np.isfinite(values)):
        raise ValueError("occupied orbitals and occupations must be finite")
    if np.any(values < 0.0) or np.any(values > 2.0):
        raise ValueError("occupations must lie between 0 and 2")

    density = (orbitals * values[np.newaxis, :]) @ orbitals.conj().T
    density = 0.5 * (density + density.conj().T)
    if not np.all(np.isfinite(density)):
        raise ValueError("P0 density matrix contains non-finite values")
    return density


def build_transition_density(
    p_ab: np.ndarray,
    p0: np.ndarray,
    *,
    weight: float = 0.5,
) -> np.ndarray:
    """Build the transition density ``(1 - w) P0 + w P_AB``.

    The ETS-NOCV transition density uses ``w = 0.5``:

    ``P_TS = 0.5 * (P_AB + P0)``

    Args:
        p_ab: Fully relaxed density matrix.
        p0: Orthogonalized-fragment density matrix.
        weight: Mixing weight in the closed interval ``[0, 1]``.

    Returns:
        The Hermitian transition density matrix.

    Raises:
        ValueError: If shapes, Hermiticity, finiteness or weight are invalid.
    """

    first = np.asarray(p_ab)
    second = np.asarray(p0)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("P_AB and P0 must be square matrices of the same shape")
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie between 0 and 1")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("P_AB and P0 must be finite")
    if not np.allclose(first, first.conj().T, atol=1e-8, rtol=1e-8):
        raise ValueError("P_AB must be Hermitian")
    if not np.allclose(second, second.conj().T, atol=1e-8, rtol=1e-8):
        raise ValueError("P0 must be Hermitian")

    transition = (1.0 - weight) * second + weight * first
    transition = 0.5 * (transition + transition.conj().T)
    if not np.all(np.isfinite(transition)):
        raise ValueError("transition density contains non-finite values")
    return transition


@dataclass(frozen=True)
class KPointOrbitals:
    """Wavefunction coefficients and occupations of one computed k-point.

    Attributes:
        index: One-based k-point index used in ABACUS file names.
        direct: Direct k-point coordinate.
        weight: Brillouin-zone weight.
        overlap: Full-basis overlap matrix ``S(k)``.
        coefficients: Spin channel to ``(n_basis, n_bands)`` coefficients.
        energies: Spin channel to band energies in Ry.
        occupations: Spin channel to band occupations.
        data_files: Files used to assemble this k-point.
    """

    index: int
    direct: tuple[float, float, float]
    weight: float
    overlap: np.ndarray
    coefficients: dict[int, np.ndarray]
    energies: dict[int, np.ndarray]
    occupations: dict[int, np.ndarray]
    data_files: tuple[str, ...]


_DEVELOP_WFC_FILE = re.compile(
    r"^wf(?:k(?P<ik>\d+))?(?:s(?P<is>\d+))?(?:g(?P<step>\d+))?_nao\.txt$",
    re.IGNORECASE,
)
_LTS_GAMMA_WFC_FILE = re.compile(
    r"^WFC_NAO_GAMMA(?P<is>\d+)(?:_ION(?P<step>\d+))?\.TXT$",
    re.IGNORECASE,
)
_LTS_K_WFC_FILE = re.compile(
    r"^WFC_NAO_K(?P<ik>\d+)(?:_ION(?P<step>\d+))?\.TXT$",
    re.IGNORECASE,
)


def _discover_wfc_channels(
    output: Path,
    nspin: int,
    gamma_only: bool,
    n_kpoints: int,
) -> dict[tuple[int, int], Path]:
    """Return ``(k_index, spin) -> WFC file`` for the final output set."""

    candidates: dict[tuple[int, int], tuple[int, int, Path]] = {}

    def keep(key: tuple[int, int], priority: int, step: int, path: Path) -> None:
        previous = candidates.get(key)
        if previous is None or (priority, -step) < (previous[0], -previous[1]):
            candidates[key] = (priority, step, path)

    lts_k: dict[int, tuple[int, Path]] = {}
    for path in output.iterdir():
        if not path.is_file():
            continue
        name = path.name
        match = _DEVELOP_WFC_FILE.match(name)
        if match is not None:
            ik = int(match.group("ik")) if match.group("ik") else 1
            spin = int(match.group("is")) if match.group("is") else 1
            step = int(match.group("step")) if match.group("step") else 0
            keep((ik, spin), 0, step, path)
            continue

        match = _LTS_GAMMA_WFC_FILE.match(name)
        if match is not None and gamma_only:
            spin = int(match.group("is"))
            step = int(match.group("step")) if match.group("step") else 0
            keep((1, spin), 1, step, path)
            continue

        match = _LTS_K_WFC_FILE.match(name)
        if match is not None and not gamma_only:
            ik = int(match.group("ik"))
            step = int(match.group("step")) if match.group("step") else 0
            previous = lts_k.get(ik)
            if previous is None or step > previous[0]:
                lts_k[ik] = (step, path)

    ordered_lts = [lts_k[index][1] for index in sorted(lts_k)]
    if nspin == 1:
        for ik, path in enumerate(ordered_lts, start=1):
            keep((ik, 1), 1, 0, path)
    else:
        if len(ordered_lts) != 2 * n_kpoints:
            raise LCAODataError(
                f"found {len(ordered_lts)} LTS wavefunction files for {n_kpoints} "
                "k-points and nspin=2; expected twice as many"
            )
        for ik, path in enumerate(ordered_lts[:n_kpoints], start=1):
            keep((ik, 1), 1, 0, path)
        for ik, path in enumerate(ordered_lts[n_kpoints:], start=1):
            keep((ik, 2), 1, 0, path)

    return {key: value[2] for key, value in candidates.items()}


def load_lcao_kpoint_orbitals(
    job: str | Path,
    *,
    inputs: Any = None,
    nspin: Optional[int] = None,
    gamma_only: Optional[bool] = None,
) -> list[KPointOrbitals]:
    """Load LCAO wavefunction coefficients, energies and occupations.

    This complements :func:`load_lcao_kpoint_densities`; it reuses the same
    k-point table, weights and overlap matrices, then attaches the develop or
    LTS wavefunction files for every spin channel.

    Args:
        job: ABACUS job directory.
        inputs: Parsed ``INPUT`` mapping, read from ``job`` when omitted.
        nspin: Override the spin channel count; otherwise read from ``INPUT``.
        gamma_only: Override the Gamma-only flag; otherwise read from
            ``INPUT``.

    Returns:
        One :class:`KPointOrbitals` per computed k-point.

    Raises:
        LCAODataError: If the density/overlap data or a required WFC file cannot
            be read consistently.
    """

    job_path = Path(job).resolve()
    kpoints = load_lcao_kpoint_densities(
        job_path,
        inputs=inputs,
        nspin=nspin,
        gamma_only=gamma_only,
    )
    if inputs is None:
        input_path = job_path / "INPUT"
        if not input_path.is_file():
            raise LCAODataError(f"INPUT file does not exist: {input_path}")
        inputs = ReadInput(str(input_path))
    actual_nspin = int(_scalar(inputs.get("nspin", 1))) if nspin is None else int(nspin)
    if gamma_only is None:
        actual_gamma = bool(int(_scalar(inputs.get("gamma_only", 0)) or 0))
    else:
        actual_gamma = bool(gamma_only)
    suffix = str(_scalar(inputs.get("suffix", "ABACUS")))
    output = job_path / f"OUT.{suffix}"
    channels = _discover_wfc_channels(output, actual_nspin, actual_gamma, len(kpoints))

    expected_spins = (1,) if actual_nspin == 1 else (1, 2)
    result: list[KPointOrbitals] = []
    for kpoint in kpoints:
        coefficients: dict[int, np.ndarray] = {}
        energies: dict[int, np.ndarray] = {}
        occupations: dict[int, np.ndarray] = {}
        files = list(kpoint.data_files)
        for spin in expected_spins:
            path = channels.get((kpoint.index, spin))
            if path is None:
                raise LCAODataError(
                    f"wavefunction file is missing for k-point {kpoint.index}, spin {spin}"
                )
            coeff, band_energies, band_occupations = read_wfc_nao_k_data(path)
            if coeff.shape[0] != kpoint.overlap.shape[0]:
                raise LCAODataError(f"{path}: coefficient rows do not match the overlap dimension")
            if band_energies.shape != (coeff.shape[1],) or band_occupations.shape != (
                coeff.shape[1],
            ):
                raise LCAODataError(f"{path}: band energies or occupations are inconsistent")
            coefficients[spin] = coeff
            energies[spin] = band_energies
            occupations[spin] = band_occupations
            files.append(str(path))
        result.append(
            KPointOrbitals(
                index=kpoint.index,
                direct=kpoint.direct,
                weight=kpoint.weight,
                overlap=kpoint.overlap,
                coefficients=coefficients,
                energies=energies,
                occupations=occupations,
                data_files=tuple(dict.fromkeys(files)),
            )
        )
    return result
