"""COOP and COHP analysis for ABACUS LCAO calculations.

The implementation uses the Hamiltonian/overlap matrices and the NAO
wavefunctions written by ABACUS.  It therefore requires an LCAO calculation
with ``out_mat_hs=1`` and ``out_wfc_lcao=1`` (or compatible WFC files).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from abacustools.core.constant import RY_TO_EV
from abacustools.data.mayer import (
    read_kpoint_weights,
    read_nao_basis_layout,
    read_overlap_matrix,
    read_overlap_matrix_develop,
    read_wfc_nao_k_data,
    select_atom_pairs,
)
from abacustools.io.abacus import ReadInput


_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_DATA_FILE = re.compile(r"^data-(\d+)-(H|S)$", re.IGNORECASE)
_WFC_FILE = re.compile(r"(?:WFC_NAO_K|LOWF_K_|WFC_NAO_GAMMA)(\d+)", re.IGNORECASE)
# Develop-version names: hk1_nao.txt, hk1s2g1_nao.txt, sk1_nao.txt, wfk1_nao.txt,
# wfk1s2g1_nao.txt, and the gamma-only hk_nao.txt/sk_nao.txt/wf_nao.txt/wfs1_nao.txt.
_DEVELOP_H_FILE = re.compile(r"^hk(?P<ik>\d+)?(?:s(?P<is>\d+))?(?:g\d+)?_nao\.txt$", re.IGNORECASE)
_DEVELOP_S_FILE = re.compile(r"^sk(?P<ik>\d+)?(?:g\d+)?_nao\.txt$", re.IGNORECASE)
_DEVELOP_WFC_FILE = re.compile(
    r"^wf(?:k(?P<ik>\d+))?(?:s(?P<is>\d+))?(?:g\d+)?_nao\.txt$", re.IGNORECASE
)
_DEVELOP_WEIGHT = re.compile(
    r"^dm(?:k(?P<ik>\d+))?(?:s\d+)?(?:g\d+)?_nao\.txt$", re.IGNORECASE
)


@dataclass(frozen=True)
class COHPResult:
    """Energy-resolved COHP/COOP data.

    ``energy`` is an absolute energy axis in eV.  Use ``energy_relative`` for
    the axis shifted by ``efermi``.  ``values`` retain the computed sign;
    inverting the sign is deliberately a plotting-only operation. For a curve
    obtained from an atom pair, ``atom_pair`` is one-based and
    ``orbital_i``/``orbital_j`` are the automatically mapped zero-based NAO
    indices.
    """

    method: str
    spin: str
    energy: np.ndarray
    values: np.ndarray
    efermi: Optional[float]
    raw_energy: np.ndarray
    raw_values: np.ndarray
    atom_pair: Optional[tuple[int, int]] = None
    orbital_i: tuple[int, ...] = ()
    orbital_j: tuple[int, ...] = ()

    @property
    def energy_relative(self) -> np.ndarray:
        """Return energy relative to the Fermi level when it is available."""

        if self.efermi is None:
            return self.energy.copy()
        return self.energy - self.efermi

    @property
    def ico_value(self) -> float:
        """Return the integrated band contribution up to the Fermi level.

        The occupied-band sum is evaluated from the discrete band
        contributions before binning or Gaussian smoothing. This keeps ICOHP
        and ICOOP independent of the plotting grid and avoids losing weight
        from a narrow peak between two grid points.
        """

        values = self.raw_values
        if self.efermi is not None:
            keep = self.raw_energy <= self.efermi + 1.0e-12
            values = values[keep]
        return float(np.sum(values))

    @property
    def label(self) -> str:
        """Return the conventional integral label for this method."""

        return "ICOHP" if self.method == "COHP" else "ICOOP"

    def write_data(self, file_path: str | Path, *, shifted: bool = True) -> None:
        """Write the energy-resolved curve as a two-column text file."""

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        energy = self.energy_relative if shifted else self.energy
        title = "Energy-Ef(eV)" if shifted and self.efermi is not None else "Energy(eV)"
        unit = "(eV)" if self.method == "COHP" else "(dimensionless)"
        np.savetxt(
            path,
            np.column_stack((energy, self.values)),
            header=f"{title} {self.method}{unit}",
        )

    def plot(
        self,
        file_path: str | Path,
        *,
        emin: Optional[float] = -10.0,
        emax: Optional[float] = 10.0,
        width: Optional[float] = None,
        invert: bool = False,
    ) -> None:
        """Plot the curve with energy on the vertical axis."""

        if emin is not None and emax is not None and emin >= emax:
            raise ValueError("emin must be smaller than emax")
        if width is not None and width <= 0:
            raise ValueError("width must be positive")

        import matplotlib.pyplot as plt

        energy = self.energy_relative
        values = -self.values if invert else self.values
        if width is None:
            width = max(float(np.max(np.abs(values))), 1.0e-12)
        else:
            width = float(width)
        fig, axis = plt.subplots(figsize=(6, 9))
        axis.plot(values, energy, color="tab:blue", linewidth=1.2)
        axis.fill_betweenx(energy, values, 0.0, where=energy <= 0.0, alpha=0.22, color="tab:blue")
        axis.axvline(0.0, color="black", linewidth=0.6)
        if self.efermi is not None:
            axis.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
        axis.set_xlim(-width, width)
        if emin is not None or emax is not None:
            axis.set_ylim(emin, emax)
        label = self.method + (" (eV)" if self.method == "COHP" else "")
        axis.set_xlabel(("-" if invert else "") + label)
        axis.set_ylabel("Energy - $E_F$ (eV)" if self.efermi is not None else "Energy (eV)")
        axis.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _scalar(value):
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


def read_efermi_from_log(log_file: str | Path) -> Optional[float]:
    """Read the last Fermi energy reported in an ABACUS running log."""

    path = Path(log_file)
    if not path.is_file():
        return None
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "fermi" not in line.lower():
            continue
        matches = re.findall(_FLOAT, line)
        if matches:
            values.append(_number(matches[-1]))
    return values[-1] if values else None


def _file_index(path: Path) -> int:
    match = _DATA_FILE.match(path.name)
    if match is None:
        raise ValueError(f"invalid ABACUS matrix filename: {path.name}")
    return int(match.group(1))


def _wfc_index(path: Path) -> int:
    match = _WFC_FILE.search(path.stem)
    if match is None:
        return 0
    return int(match.group(1))


def _find_wfc_files(output: Path) -> list[Path]:
    candidates: dict[int, tuple[int, Path]] = {}
    for path in output.iterdir():
        if not path.is_file():
            continue
        name = path.name.upper()
        if re.fullmatch(r"WFC_NAO_K\d+\.TXT", name):
            priority = 0
        elif re.fullmatch(r"WFC_NAO_K\d+_ION\d+\.TXT", name):
            priority = 1
        elif re.fullmatch(r"LOWF_K_\d+\.(TXT|DAT)", name):
            priority = 2
        elif re.fullmatch(r"WFC_NAO_GAMMA\d+\.TXT", name):
            priority = 3
        else:
            continue
        index = _wfc_index(path)
        previous = candidates.get(index)
        if previous is None or priority < previous[0]:
            candidates[index] = (priority, path)
    return [candidates[index][1] for index in sorted(candidates)]


def _parse_orbitals(orbitals: Iterable[int], name: str, dimension: int) -> list[int]:
    parsed = [int(index) for index in orbitals]
    if not parsed:
        raise ValueError(f"{name} must contain at least one orbital index")
    if len(set(parsed)) != len(parsed) or any(index < 0 or index >= dimension for index in parsed):
        raise ValueError(f"{name} contains an invalid orbital index for dimension {dimension}")
    return parsed


def _pair_value(operator: np.ndarray, coefficients: np.ndarray, left: list[int], right: list[int]) -> np.ndarray:
    block = operator[np.ix_(left, right)]
    return np.einsum("in,ij,jn->n", coefficients[left].conj(), block, coefficients[right]).real


def _energy_grid(
    energies: np.ndarray,
    values: np.ndarray,
    de: float,
    *,
    padding: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if de <= 0:
        raise ValueError("de must be positive")
    if padding < 1:
        raise ValueError("padding must be at least one energy bin")
    if len(energies) == 0:
        raise ValueError("no band energies were found")
    lower = np.floor(np.min(energies) / de) * de - padding * de
    upper = np.ceil(np.max(energies) / de) * de + padding * de
    grid = np.arange(lower, upper + de * 0.5, de, dtype=float)
    result = np.zeros_like(grid)
    indices = np.rint((energies - lower) / de).astype(int)
    valid = (indices >= 0) & (indices < len(grid))
    # ``values`` are band contributions, not spectral densities. Dividing by
    # the bin width turns each discrete band term into a delta-like peak whose
    # integral is the original contribution.
    np.add.at(result, indices[valid], values[valid] / de)
    return grid, result


def gaussian_smooth(values: np.ndarray, de: float, nstddev: float) -> np.ndarray:
    """Smooth a uniformly sampled curve with a Gaussian kernel."""

    if nstddev <= 0:
        raise ValueError("smooth_nstddev must be positive")
    if len(values) < 2:
        return values.copy()
    sigma = de * nstddev
    radius = max(1, int(np.ceil(4.0 * sigma / de)))
    radius = min(radius, (len(values) - 1) // 2)
    offsets = np.arange(-radius, radius + 1, dtype=float) * de
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def _output_directory(job: Path, inputs: dict) -> Path:
    suffix = str(_scalar(inputs.get("suffix", "ABACUS")))
    return _required_directory(job / f"OUT.{suffix}", "ABACUS output directory")


def _load_channel_files(output: Path) -> tuple[list[Path], list[Path], list[Path]]:
    h_files = {}
    s_files = {}
    for path in output.iterdir():
        match = _DATA_FILE.match(path.name)
        if match is None:
            continue
        (h_files if match.group(2).upper() == "H" else s_files)[int(match.group(1))] = path
    if not h_files:
        raise FileNotFoundError(f"No data-*-H files found in {output}")
    if set(h_files) != set(s_files):
        raise ValueError("data-*-H and data-*-S files do not have matching indices")
    indices = sorted(h_files)
    wfc_files = _find_wfc_files(output)
    if len(wfc_files) != len(indices):
        raise ValueError(f"found {len(indices)} matrix pairs but {len(wfc_files)} WFC files in {output}")
    return [h_files[index] for index in indices], [s_files[index] for index in indices], wfc_files


def _develop_kpoint_weights(output: Path) -> Optional[np.ndarray]:
    """Return the k-point weights of a develop run, normalized to sum to one.

    The develop version records the weight of every k-point in the header of
    its ``dm*_nao.txt`` density matrices.  Gamma-only runs write a single file
    without a k index.
    """
    entries: dict[int, float] = {}
    for path in output.iterdir():
        match = _DEVELOP_WEIGHT.match(path.name)
        if match is None:
            continue
        ik = int(match.group("ik")) if match.group("ik") else 1
        if ik in entries:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "# weight of this k point" in line:
                entries[ik] = float(line.split()[0])
                break
    if not entries:
        return None
    total = float(sum(entries.values()))
    if total <= 0:
        return None
    return np.asarray([entries[ik] for ik in sorted(entries)], dtype=float) / total


def _load_develop_channel_files(
    output: Path, nspin: int
) -> tuple[list[Path], list[Path], list[Path], Optional[np.ndarray]]:
    """Collect the develop-version H, S and WFC files in channel order.

    Channels are ordered like the LTS files: every k-point of spin up first,
    then every k-point of spin down, so the spin selection logic above works
    unchanged.
    """
    h_files: dict[tuple[int, int], Path] = {}
    s_files: dict[int, Path] = {}
    wfc_files: dict[tuple[int, int], Path] = {}
    for path in output.iterdir():
        name = path.name
        match = _DEVELOP_H_FILE.match(name)
        if match is not None:
            ik = int(match.group("ik")) if match.group("ik") else 1
            ispin = int(match.group("is")) if match.group("is") else 1
            h_files[(ik, ispin)] = path
            continue
        match = _DEVELOP_S_FILE.match(name)
        if match is not None:
            ik = int(match.group("ik")) if match.group("ik") else 1
            s_files[ik] = path
            continue
        match = _DEVELOP_WFC_FILE.match(name)
        if match is not None:
            ik = int(match.group("ik")) if match.group("ik") else 1
            ispin = int(match.group("is")) if match.group("is") else 1
            wfc_files[(ik, ispin)] = path
    if not h_files:
        raise FileNotFoundError(f"No hk*_nao.txt files found in {output}")

    kpoints = sorted({ik for ik, _ispin in h_files})
    spins = sorted({ispin for _ik, ispin in h_files})
    if nspin == 2 and len(spins) > 1:
        channels = [(ik, ispin) for ispin in spins for ik in kpoints]
    else:
        channels = [(ik, 1) for ik in kpoints]

    ordered_h: list[Path] = []
    ordered_s: list[Path] = []
    ordered_wfc: list[Path] = []
    for ik, ispin in channels:
        if (ik, ispin) not in h_files:
            raise FileNotFoundError(f"missing hk matrix for k-point {ik}, spin {ispin}")
        if ik not in s_files:
            raise FileNotFoundError(f"missing sk_nao.txt overlap matrix for k-point {ik}")
        if (ik, ispin) not in wfc_files:
            raise FileNotFoundError(f"missing wf*_nao.txt wavefunction for k-point {ik}, spin {ispin}")
        ordered_h.append(h_files[(ik, ispin)])
        ordered_s.append(s_files[ik])
        ordered_wfc.append(wfc_files[(ik, ispin)])

    weights = _develop_kpoint_weights(output)
    if weights is not None and len(weights) != len(kpoints):
        weights = None
    return ordered_h, ordered_s, ordered_wfc, weights


def _load_channels(
    output: Path, nspin: int
) -> tuple[list[Path], list[Path], list[Path], Optional[np.ndarray], callable]:
    """Dispatch between the LTS ``data-*`` and the develop ``hk*_nao.txt`` layouts."""
    if any(_DATA_FILE.match(path.name) for path in output.iterdir()):
        h_files, s_files, wfc_files = _load_channel_files(output)
        return h_files, s_files, wfc_files, None, read_overlap_matrix
    h_files, s_files, wfc_files, weights = _load_develop_channel_files(output, nspin)
    return h_files, s_files, wfc_files, weights, read_overlap_matrix_develop


def _analyze_cohp_groups(
    job: str | Path,
    orbital_pairs: Iterable[tuple[Iterable[int], Iterable[int]]],
    *,
    method: str,
    spin: str,
    de: float,
    smooth: bool,
    smooth_nstddev: float,
    efermi: Optional[float],
    inputs: Optional[dict] = None,
    expected_dimension: Optional[int] = None,
) -> tuple[COHPResult, ...]:
    """Calculate one curve for every pair of orbital groups in one job."""
    method = method.upper()
    if method not in {"COHP", "COOP"}:
        raise ValueError("method must be COHP or COOP")
    spin = spin.lower()
    if spin not in {"sum", "up", "down"}:
        raise ValueError("spin must be sum, up, or down")
    groups = tuple((tuple(left), tuple(right)) for left, right in orbital_pairs)
    if not groups:
        raise ValueError("at least one orbital group pair is required")

    job_path = Path(job).resolve()
    if inputs is None:
        inputs = ReadInput(str(_required_file(job_path / "INPUT", "INPUT file")))
    configured_nspin = int(_scalar(inputs.get("nspin", 1)))
    if configured_nspin not in {1, 2}:
        raise ValueError(f"COHP/COOP supports nspin=1 or 2, got {configured_nspin}")
    if str(_scalar(inputs.get("basis_type", "lcao"))).lower() != "lcao":
        raise ValueError("COHP/COOP requires basis_type=lcao")
    output = _output_directory(job_path, inputs)
    h_files, s_files, wfc_files, develop_weights, matrix_reader = _load_channels(
        output, configured_nspin
    )
    weights_path = output / "kpoints"
    if develop_weights is not None:
        base_weights = develop_weights
    elif weights_path.is_file():
        base_weights = read_kpoint_weights(weights_path)
    else:
        if configured_nspin == 2:
            if len(h_files) % 2:
                raise ValueError("cannot infer k-point count for nspin=2 without a kpoints file")
            base_weights = np.full(len(h_files) // 2, 1.0 / (len(h_files) // 2))
        else:
            base_weights = np.full(len(h_files), 1.0 / len(h_files))

    if len(h_files) == len(base_weights):
        nspin = 1
    elif configured_nspin == 2 and len(h_files) == 2 * len(base_weights):
        nspin = 2
    else:
        raise ValueError(f"cannot map {len(h_files)} matrix pairs onto {len(base_weights)} k-point weights")
    if nspin == 1 and spin != "sum":
        raise ValueError(f"requested spin={spin}, but the output contains nspin=1")
    if nspin == 2 and spin == "down":
        channel_offset = len(base_weights)
        channels = [channel_offset + index for index in range(len(base_weights))]
    elif nspin == 2 and spin == "up":
        channels = list(range(len(base_weights)))
    elif nspin == 2:
        channels = list(range(len(base_weights))) + [len(base_weights) + index for index in range(len(base_weights))]
    else:
        channels = list(range(len(base_weights)))

    energies: list[float] = []
    values: list[list[float]] = [[] for _ in groups]
    dimension: Optional[int] = None
    parsed_groups: Optional[list[tuple[list[int], list[int]]]] = None
    for channel in channels:
        h_matrix = matrix_reader(h_files[channel])
        s_matrix = matrix_reader(s_files[channel])
        coefficients, band_energies, occupations = read_wfc_nao_k_data(wfc_files[channel])
        if h_matrix.shape != s_matrix.shape or h_matrix.shape != (coefficients.shape[0], coefficients.shape[0]):
            raise ValueError(f"channel {channel + 1}: H, S, and WFC dimensions do not match")
        if dimension is None:
            dimension = h_matrix.shape[0]
            if expected_dimension is not None and dimension != expected_dimension:
                raise ValueError(
                    "matrix dimension does not match the NAO basis size "
                    f"{expected_dimension}: {dimension}"
                )
            parsed_groups = [
                (
                    _parse_orbitals(left, f"group {index + 1} left orbitals", dimension),
                    _parse_orbitals(right, f"group {index + 1} right orbitals", dimension),
                )
                for index, (left, right) in enumerate(groups)
            ]
        elif dimension != h_matrix.shape[0]:
            raise ValueError("matrix dimensions differ between channels")
        assert parsed_groups is not None

        operator = h_matrix if method == "COHP" else s_matrix
        energies.extend((band_energies * RY_TO_EV).tolist())
        if method == "COHP":
            # ABACUS writes H in Ry, while the exported energy axis and ICOHP
            # are in eV. COOP uses the dimensionless overlap matrix.
            unit_factor = RY_TO_EV
        else:
            unit_factor = 1.0
        for index, (left, right) in enumerate(parsed_groups):
            channel_values = _pair_value(operator, coefficients, left, right)
            # The occupations written with each WFC already contain the
            # k-point weight. For nspin=1 they also carry the spin degeneracy
            # of two, so using the occupation is sufficient in both spin modes.
            weighted = occupations * channel_values * unit_factor
            values[index].extend(weighted.tolist())

    raw_energy = np.asarray(energies, dtype=float)
    order = np.argsort(raw_energy, kind="stable")
    raw_energy = raw_energy[order]
    if efermi is None:
        for log_name in ("running_scf.log", "running_nscf.log"):
            efermi = read_efermi_from_log(output / log_name)
            if efermi is not None:
                break

    results: list[COHPResult] = []
    # Keep enough empty states on both sides for the Gaussian tail to move
    # into the grid. Otherwise a narrow peak near the lowest band loses weight
    # at the array boundary and the plotted curve no longer integrates to ICOHP.
    padding = max(1, int(np.ceil(4.0 * smooth_nstddev)) + 1) if smooth else 1
    for group_values in values:
        raw_values = np.asarray(group_values, dtype=float)[order]
        grid, processed = _energy_grid(
            raw_energy,
            raw_values,
            de,
            padding=padding,
        )
        if smooth:
            processed = gaussian_smooth(processed, de, smooth_nstddev)
        results.append(
            COHPResult(
                method,
                spin,
                grid,
                processed,
                efermi,
                raw_energy,
                raw_values,
            )
        )
    return tuple(results)


def analyze_cohp(
    job: str | Path,
    atom_i_orbs: Iterable[int],
    atom_j_orbs: Iterable[int],
    *,
    method: str = "COHP",
    spin: str = "sum",
    de: float = 0.1,
    smooth: bool = True,
    smooth_nstddev: float = 3.0,
    efermi: Optional[float] = None,
) -> COHPResult:
    """Calculate an energy-resolved COHP or COOP curve from an ABACUS job."""
    return _analyze_cohp_groups(
        job,
        ((atom_i_orbs, atom_j_orbs),),
        method=method,
        spin=spin,
        de=de,
        smooth=smooth,
        smooth_nstddev=smooth_nstddev,
        efermi=efermi,
    )[0]


def analyze_cohp_pairs(
    job: str | Path,
    *,
    cutoff: Optional[float] = None,
    pairs: Optional[str] = None,
    pairs_file: Optional[str | Path] = None,
    method: str = "COHP",
    spin: str = "sum",
    de: float = 0.1,
    smooth: bool = True,
    smooth_nstddev: float = 3.0,
    efermi: Optional[float] = None,
) -> tuple[tuple[tuple[int, int], ...], tuple[COHPResult, ...]]:
    """Analyze COHP/COOP for atom pairs selected like Mayer bond orders.

    Atom pairs use one-based indices. Each pair is expanded to the complete
    NAO ranges of its two atoms from ``STRU`` and the numerical-orbital files,
    so callers no longer have to know global NAO indices.

    Args:
        job: ABACUS LCAO job directory.
        cutoff: Largest periodic atom-pair distance in Angstrom.
        pairs: Comma-separated one-based atom pairs, for example ``1-2,1-3``.
        pairs_file: File with one pair per line, using one-based indices.
        method: ``COHP`` or ``COOP``.
        spin: ``sum``, ``up`` or ``down``.
        de: Energy-grid spacing in eV.
        smooth: Apply Gaussian smoothing to the curve.
        smooth_nstddev: Gaussian width in grid spacings.
        efermi: Fermi energy in eV, read from the running log when omitted.

    Returns:
        The selected zero-based atom pairs and one annotated curve per pair.
        ``COHPResult.atom_pair`` is one-based and ``orbital_i``/``orbital_j``
        hold the automatically mapped zero-based NAO indices.
    """
    job_path = Path(job).resolve()
    inputs = ReadInput(str(_required_file(job_path / "INPUT", "INPUT file")))
    structure, atom_ranges, basis_functions = read_nao_basis_layout(
        job_path,
        inputs=inputs,
    )
    selected_pairs = tuple(
        select_atom_pairs(structure, cutoff, pairs, pairs_file)
    )
    if not selected_pairs:
        raise ValueError("no atom pairs were selected for COHP/COOP")

    groups = tuple(
        (
            range(*atom_ranges[first]),
            range(*atom_ranges[second]),
        )
        for first, second in selected_pairs
    )
    results = _analyze_cohp_groups(
        job_path,
        groups,
        method=method,
        spin=spin,
        de=de,
        smooth=smooth,
        smooth_nstddev=smooth_nstddev,
        efermi=efermi,
        inputs=inputs,
        expected_dimension=basis_functions,
    )
    annotated = tuple(
        replace(
            result,
            atom_pair=(first + 1, second + 1),
            orbital_i=tuple(range(*atom_ranges[first])),
            orbital_j=tuple(range(*atom_ranges[second])),
        )
        for result, (first, second) in zip(results, selected_pairs)
    )
    return selected_pairs, annotated


def read_cohp_output(*args, **kwargs) -> COHPResult:
    """Compatibility alias for :func:`analyze_cohp`."""

    return analyze_cohp(*args, **kwargs)


__all__ = [
    "COHPResult",
    "analyze_cohp",
    "analyze_cohp_pairs",
    "gaussian_smooth",
    "read_cohp_output",
    "read_efermi_from_log",
]
