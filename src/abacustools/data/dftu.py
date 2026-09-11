"""Core logic for linear-response DFT+U Hubbard parameter calculation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

def parse_running_scf_dftu(log_path: Path) -> dict[int, dict] | None:
    """Parse the last DFT+U block from running_scf.log.
    
    Returns:
        Dictionary mapping atom index to {'L': int, 'eigenvalues': {spin: [values]}}
        or None if no DFT+U block found.
    """
    if not log_path.is_file():
        return None
    
    content = log_path.read_text(encoding="utf-8")
    
    # Find all DFT+U blocks
    pattern = r'//=========================L\(S\)DA\+U===========================//\s*\n(.*?)(?=//=========================L\(S\)DA\+U===========================//|$)'
    blocks = re.findall(pattern, content, re.DOTALL)
    
    if not blocks:
        return None
    
    # Use the last block
    last_block = blocks[-1]
    
    # Parse atom data
    atoms = {}
    atom_pattern = r'atoms\s+(\d+)\s*\nL\s+(\d+)\s*\nzeta\s+(\d+)\s*\n(.*?)(?=atoms\s+\d+|$)'
    for match in re.finditer(atom_pattern, last_block, re.DOTALL):
        atom_idx = int(match.group(1))
        L = int(match.group(2))
        zeta = int(match.group(3))
        data = match.group(4)
        
        # Parse eigenvalues for each spin
        eigenvalues = {}
        eigen_pattern = r'eigenvalues\s+(\d+)\s*\n\s*([\d\s\.\-Ee\+]+?)(?=\n[^\d\s]|$)'
        for eigen_match in re.finditer(eigen_pattern, data):
            spin = int(eigen_match.group(1))
            # Parse values, handling scientific notation
            raw_values = eigen_match.group(2).strip().split()
            values = []
            for v in raw_values:
                try:
                    values.append(float(v))
                except ValueError:
                    # Skip non-numeric values
                    pass
            # The last value is the sum, exclude it
            if len(values) > 1:
                eigenvalues[spin] = values[:-1]
        
        atoms[atom_idx] = {
            'L': L,
            'zeta': zeta,
            'eigenvalues': eigenvalues
        }
    
    return atoms




@dataclass
class OccupationData:
    """Occupation matrix data for one atom and angular momentum."""

    atom_index: int  # 0-based atom index
    angular_momentum: int  # l quantum number (2 for d, 3 for f)
    zeta: int  # radial quantum number
    occupation_matrix: dict[int, np.ndarray]  # spin -> (2l+1, 2l+1) matrix
    eigenvalues: dict[int, np.ndarray]  # spin -> eigenvalues


def parse_onsite_dm(filepath: Path) -> list[OccupationData]:
    """Parse the onsite.dm file from ABACUS DFT+U calculations.

    Args:
        filepath: Path to the onsite.dm file.

    Returns:
        List of OccupationData for each atom and angular momentum.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"onsite.dm file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8")
    lines = content.strip().splitlines()

    results: list[OccupationData] = []
    current_atom: int | None = None
    current_l: int | None = None
    current_zeta: int | None = None
    current_spin: int | None = None
    current_matrix: np.ndarray | None = None
    current_eigenvalues: dict[int, np.ndarray] = {}
    current_matrices: dict[int, np.ndarray] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if line.startswith("atoms"):
            # Save previous data if exists
            if current_atom is not None and current_matrices:
                results.append(
                    OccupationData(
                        atom_index=current_atom,
                        angular_momentum=current_l if current_l is not None else 0,
                        zeta=current_zeta if current_zeta is not None else 0,
                        occupation_matrix=dict(current_matrices),
                        eigenvalues=dict(current_eigenvalues),
                    )
                )

            # Parse atom index
            parts = line.split()
            current_atom = int(parts[1])
            current_l = None
            current_zeta = None
            current_spin = None
            current_matrix = None
            current_eigenvalues = {}
            current_matrices = {}

        elif line.startswith("L"):
            parts = line.split()
            current_l = int(parts[1])

        elif line.startswith("zeta"):
            parts = line.split()
            current_zeta = int(parts[1])

        elif line.startswith("spin"):
            parts = line.split()
            current_spin = int(parts[1])

            # Initialize matrix for this spin
            if current_l is not None:
                size = 2 * current_l + 1
                current_matrix = np.zeros((size, size))

                # Read matrix rows
                for row in range(size):
                    i += 1
                    if i >= len(lines):
                        raise ValueError("Unexpected end of file while reading matrix")
                    row_line = lines[i].strip()
                    row_values = [float(x) for x in row_line.split()]
                    if len(row_values) != size:
                        raise ValueError(
                            f"Expected {size} values in row {row}, got {len(row_values)}"
                        )
                    current_matrix[row, :] = row_values

                current_matrices[current_spin] = current_matrix

        elif line.startswith("eigenvalues"):
            # Parse eigenvalues
            parts = line.split()
            spin = int(parts[1])

            i += 1
            if i >= len(lines):
                raise ValueError("Unexpected end of file while reading eigenvalues")

            eigen_line = lines[i].strip()
            eigen_values = [float(x) for x in eigen_line.split()]
            current_eigenvalues[spin] = np.array(eigen_values)

        i += 1

    # Save the last atom data
    if current_atom is not None and current_matrices:
        results.append(
            OccupationData(
                atom_index=current_atom,
                angular_momentum=current_l if current_l is not None else 0,
                zeta=current_zeta if current_zeta is not None else 0,
                occupation_matrix=dict(current_matrices),
                eigenvalues=dict(current_eigenvalues),
            )
        )

    return results


def compute_occupation_eigenvalues(occupation_matrix: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of an occupation matrix.

    Args:
        occupation_matrix: (2l+1, 2l+1) occupation matrix.

    Returns:
        Sorted eigenvalues.
    """
    eigenvalues = np.linalg.eigvalsh(occupation_matrix)
    return np.sort(eigenvalues)


def total_occupation(eigenvalues: np.ndarray) -> float:
    """Compute total occupation from eigenvalues.

    Args:
        eigenvalues: Occupation eigenvalues.

    Returns:
        Sum of eigenvalues (total occupation number).
    """
    return float(np.sum(eigenvalues))


@dataclass
class ResponseData:
    """Response function data for one atom."""

    atom_index: int
    angular_momentum: int
    u_values: list[float]  # U values in eV
    occupations: list[float]  # Total occupation for each U
    chi_0: float  # Bare response (eV^{-1})
    chi: float  # Screened response (eV^{-1})
    hubbard_u: float  # Computed Hubbard U in eV


def compute_response_function(
    u_values: list[float],
    occupations: list[float],
) -> tuple[float, float, float]:
    """Compute response functions and Hubbard U from occupation vs U data.

    The linear response method computes:
        U = chi_0^{-1} - chi^{-1}

    where chi_0 is the bare response (no Hubbard correction) and chi is the
    screened response (with Hubbard correction).

    Args:
        u_values: List of U values in eV.
        occupations: Total occupation for each U value.

    Returns:
        Tuple of (chi_0, chi, hubbard_u) in eV.

    Raises:
        ValueError: If insufficient data points or fitting fails.
    """
    if len(u_values) < 2:
        raise ValueError("At least 2 U values are required for linear response")

    u_array = np.array(u_values)
    occ_array = np.array(occupations)

    # Compute bare response chi_0: dn/dU at U=0 (no Hubbard correction)
    # Fit a line to the first few points to get the slope
    if u_values[0] == 0.0 and len(u_values) >= 3:
        fit_u = u_array[:3]
        fit_occ = occ_array[:3]
    else:
        fit_u = u_array
        fit_occ = occ_array

    # Linear fit: occ = a + b * U
    # chi_0 = -b (negative because occupation decreases with U)
    coeffs = np.polyfit(fit_u, fit_occ, 1)
    chi_0 = -coeffs[0]

    # Compute screened response chi: use the last few points
    if len(u_values) >= 3:
        fit_u_screen = u_array[-3:]
        fit_occ_screen = occ_array[-3:]
    else:
        fit_u_screen = u_array
        fit_occ_screen = occ_array

    coeffs_screen = np.polyfit(fit_u_screen, fit_occ_screen, 1)
    chi = -coeffs_screen[0]

    # Compute Hubbard U: U = chi_0^{-1} - chi^{-1}
    if abs(chi_0) < 1e-10 or abs(chi) < 1e-10:
        raise ValueError(
            f"Response functions too small: chi_0={chi_0:.6f}, chi={chi:.6f}. "
            "Cannot compute Hubbard U reliably."
        )

    hubbard_u = 1.0 / chi_0 - 1.0 / chi

    return chi_0, chi, hubbard_u


def _find_output_dir(task_dir: Path) -> Path | None:
    """Find the ABACUS output directory in a task directory."""
    out_dirs = sorted(task_dir.glob("OUT.*"))
    return out_dirs[0] if out_dirs else None


def linear_response_U(
    job_dir: Path,
    u_values: list[float],
    atom_indices: list[int] | None = None,
    angular_momentum: int | None = None,
) -> list[ResponseData]:
    """Compute Hubbard U values using linear response method.

    Args:
        job_dir: Directory containing the prepared DFT+U calculations.
        u_values: List of U values used in the calculations (eV).
        atom_indices: Optional list of 0-based atom indices to analyze.
            If None, analyze all atoms with occupation data.
        angular_momentum: Optional angular momentum to filter (2 for d, 3 for f).
            If None, analyze all angular momenta.

    Returns:
        List of ResponseData for each analyzed atom.

    Raises:
        FileNotFoundError: If required output files are missing.
        ValueError: If data is insufficient or inconsistent.
    """
    job_dir = Path(job_dir)

    # Collect occupation data for each U value
    occupations_by_atom: dict[tuple[int, int], list[float]] = {}

    for u_value in u_values:
        # Construct task directory name
        task_name = _task_name_for_u(u_value)
        task_dir = job_dir / task_name

        if not task_dir.is_dir():
            raise FileNotFoundError(
                f"Task directory not found: {task_dir}. "
                "Did you run 'abacustools workflow dftu prepare' first?"
            )

        # Look for running_scf.log in output directory
        out_dir = _find_output_dir(task_dir)
        if out_dir is None:
            raise FileNotFoundError(
                f"No output directory found in {task_dir}. "
                "Did the calculation complete successfully?"
            )

        log_path = out_dir / "running_scf.log"
        if not log_path.is_file():
            raise FileNotFoundError(
                f"running_scf.log not found in {out_dir}. "
                "The calculation may not have started."
            )

        # Parse occupation data from running_scf.log
        atoms_data = parse_running_scf_dftu(log_path)
        if atoms_data is None:
            raise ValueError(
                f"No DFT+U data found in {log_path}. "
                "Ensure dft_plus_u is set in INPUT."
            )

        for atom_idx, data in atoms_data.items():
            # Filter by atom index if specified
            if atom_indices is not None and atom_idx not in atom_indices:
                continue

            # Filter by angular momentum if specified
            if angular_momentum is not None and data['L'] != angular_momentum:
                continue

            # Compute total occupation across all spins
            total_occ = 0.0
            for spin, eigenvalues in data['eigenvalues'].items():
                total_occ += sum(eigenvalues)

            key = (atom_idx, data['L'])
            if key not in occupations_by_atom:
                occupations_by_atom[key] = []

            occupations_by_atom[key].append(total_occ)

    # Compute response functions for each atom
    results: list[ResponseData] = []

    for (atom_idx, ang_mom), occupations in occupations_by_atom.items():
        if len(occupations) != len(u_values):
            raise ValueError(
                f"Inconsistent data for atom {atom_idx}: "
                f"expected {len(u_values)} occupation values, got {len(occupations)}"
            )

        # Check if occupations vary with U
        occ_array = np.array(occupations)
        occ_range = np.max(occ_array) - np.min(occ_array)
        
        if occ_range < 1e-6:
            raise ValueError(
                f"Occupation for atom {atom_idx} does not vary with U "
                f"(range = {occ_range:.2e}). "
                "Linear response method requires occupation to change with U. "
                "This may indicate that the correlated orbitals are fully occupied "
                "or empty, or that the system is insensitive to Hubbard U correction."
            )

        chi_0, chi, hubbard_u = compute_response_function(u_values, occupations)

        results.append(
            ResponseData(
                atom_index=atom_idx,
                angular_momentum=ang_mom,
                u_values=list(u_values),
                occupations=occupations,
                chi_0=chi_0,
                chi=chi,
                hubbard_u=hubbard_u,
            )
        )

    return results

def _task_name_for_u(u_value: float) -> str:
    """Generate a stable task directory name for a U value."""
    # Use enough decimal places to distinguish small steps
    # Format: dftu_U000p05 for U=0.05, dftu_U001p00 for U=1.0
    return f"dftu_U{u_value:06.2f}".replace(".", "p")


def task_names_for_u_values(u_values: list[float]) -> list[str]:
    """Generate task directory names for a list of U values."""
    return [_task_name_for_u(u) for u in u_values]
