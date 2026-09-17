"""DDEC population analysis driven by the external Chargemol program.

Chargemol (https://ddec.sourceforge.net) computes DDEC6/DDEC3 net atomic
charges, atomic spin moments and bond orders from a valence electron density in
Gaussian cube format. This module builds that cube from the charge density of
an ABACUS job, writes the ``job_control.txt`` the program reads, runs it and
parses its ``*.xyz`` output.

Two properties of the interface shape the code below:

* Chargemol reads every input from its working directory and accepts no
  command-line arguments, so the cubes and ``job_control.txt`` are written into
  a scratch directory and the program runs with that directory as its working
  directory.
* A valence-only cube makes Chargemol insert the missing core electrons from
  the ``atomic_densities`` tables of its distribution, which fixes the number
  of core electrons of every element to ``ncore = Z - z_valence`` of the
  pseudopotential. The tables are addressed as
  ``core_{Z}_{Z}_{ncore}_500_100.txt`` and only a limited set of core counts
  exists, so :func:`reference_issues` reports the combinations that are
  missing before the program is started.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase.data import chemical_symbols

from abacustools.core.config import CONFIG
from abacustools.core.constant import ANG_TO_BOHR
from abacustools.data.charge import (
    ChargeDensityError,
    check_channel_count,
    combine,
    fft_grid_from_log,
    find_density_source,
    read_cube_charges,
    read_restart_charges,
    total_charge,
    valence_electrons,
)
from abacustools.data.grid import Charge
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


#: Largest grid cell volume Chargemol accepts, in Bohr**3. This is
#: ``maxpixelvolume`` of ``module_global_parameters.f08``; a cubic grid may not
#: be coarser than 0.25 Bohr per direction.
MAX_PIXEL_VOLUME_BOHR3 = 0.0157

#: Grid spacing the shipped reference densities are built for, in Bohr.
PREFERRED_GRID_SPACING_BOHR = 0.14

#: Radial mesh of the shipped reference densities: 100 shells up to 500 pm.
REFERENCE_CUTOFF_PM = 500
REFERENCE_SHELLS = 100

#: Values accepted by the ``<charge type>`` tag.
CHARGE_TYPES = ("DDEC6", "DDEC3")

#: File names of the results of each charge type.
CHARGE_TYPE_FILES = {
    "DDEC6": {
        "charges": "DDEC6_even_tempered_net_atomic_charges.xyz",
        "spin": "DDEC6_even_tempered_atomic_spin_moments.xyz",
        "bond_orders": "DDEC6_even_tempered_bond_orders.xyz",
    },
    "DDEC3": {
        "charges": "DDEC3_net_atomic_charges.xyz",
        "spin": "DDEC3_atomic_spin_moments.xyz",
        "bond_orders": "DDEC3_bond_orders.xyz",
    },
}

#: Radial moment files, which both charge types write with the same names.
RADIAL_MOMENT_FILES = {
    "r_squared": "DDEC_atomic_Rsquared_moments.xyz",
    "r_cubed": "DDEC_atomic_Rcubed_moments.xyz",
    "r_fourth": "DDEC_atomic_Rfourth_moments.xyz",
}

#: Reference-ion charges that Chargemol 3.5 allows for every element, copied
#: from ``available_reference_ion_range`` in
#: ``module_update_atomic_densities.f08`` (index 0 is hydrogen). The DDEC
#: iteration never asks for a reference ion outside this window, and every
#: electron count inside it has to be present in the ``c2_*`` tables.
_REFERENCE_ION_LOWER = (
    -2, -2, -2, -2, -2, -5, -4, -3, -2, -2, -2, -2, -2, -5, -4, -3, -3, -2, -2, -2,
    -2, -2, -2, -3, -4, -3, -2, -2, -2, -2, -2, -5, -4, -3, -2, -2, -2, -2, -2, -2,
    -2, -3, -4, -3, -2, -2, -2, -2, -2, -5, -4, -3, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3, -4, -3, -4, -3, -2, -2,
    -2, -5, -4, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2,
)
_REFERENCE_ION_UPPER = (
    1, 2, 2, 3, 4, 5, 6, 3, 2, 2, 2, 3, 4, 5, 6, 7, 8, 2, 2, 3,
    4, 5, 6, 7, 8, 7, 6, 5, 5, 3, 4, 5, 6, 7, 8, 3, 2, 3, 4, 5,
    6, 7, 8, 9, 7, 7, 5, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 5, 5,
    4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 9, 7, 6, 5,
    4, 5, 6, 7, 8, 7, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 5, 5, 5, 4,
    4, 4, 4, 5, 6, 7, 8, 9, 9,
)

_INT_LINE = re.compile(r"^\s*(\d+)\s*$")
_JMOL_CELL_BLOCK = re.compile(r"\{([^{}]*)\}")
_ANALYSIS_VALUE = {
    "ncore": re.compile(r"ncore\s*=\s*([-\d.eEdD+]+)"),
    "nvalence": re.compile(r"nvalence\s*=\s*([-\d.eEdD+]+)"),
    "integrated_valence": re.compile(
        r"numerically integrated valence density\s*=\s*([-\d.eEdD+]+)"
    ),
    "occupancy_correction": re.compile(
        r"sum_valence_occupancy_correction\s*=\s*([-\d.eEdD+]+)"
    ),
    "checkme": re.compile(r"checkme\s*=\s*([-\d.eEdD+]+)"),
}
_TOTAL_SPIN_LINE = re.compile(
    r"total spin magnetic moment of the unit cell is\s*([-\d.eEdD+]+)"
)
_BOND_BLOCK = re.compile(r"Printing BOs for ATOM #\s*(\d+)\s*\(\s*([A-Za-z]+)\s*\)")
_BOND_PAIR = re.compile(
    r"Bonded to the \(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*translated image "
    r"of atom number\s*(\d+)\s*\(\s*([A-Za-z]+)\s*\)\s*with bond order\s*=\s*(\S+)\s*"
    r"The average spin polarization of this bonding\s*=\s*(\S+)"
)


class DdecError(RuntimeError):
    """Raised when a DDEC analysis cannot be completed."""


@dataclass(frozen=True)
class DdecIssue:
    """One problem found before Chargemol is started.

    Attributes:
        level: ``"error"`` when the run would fail, ``"warning"`` when it only
            degrades the result.
        code: Stable machine-readable identifier.
        message: Human-readable description.
    """

    level: str
    code: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        """Return the issue as a plain dictionary."""
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class CoreElectrons:
    """Core electron count implied by the pseudopotential of one element.

    Attributes:
        element: Chemical symbol.
        z: Atomic number.
        z_valence: Valence charge of the pseudopotential.
        ncore: ``z - z_valence``, the number reported to Chargemol.
    """

    element: str
    z: int
    z_valence: float
    ncore: int

    def to_dict(self) -> Dict[str, Any]:
        """Return the entry as a plain dictionary."""
        return {
            "element": self.element,
            "z": self.z,
            "z_valence": self.z_valence,
            "ncore": self.ncore,
        }


@dataclass
class DdecAtom:
    """DDEC properties of one atom of the reference unit cell."""

    index: int
    element: str
    position: Tuple[float, float, float]
    net_charge: float
    spin_moment: Optional[float] = None
    r_squared: Optional[float] = None
    r_cubed: Optional[float] = None
    r_fourth: Optional[float] = None
    bond_order_sum: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the atom as a plain dictionary."""
        return {
            "index": self.index,
            "element": self.element,
            "position": list(self.position),
            "net_charge": self.net_charge,
            "spin_moment": self.spin_moment,
            "r_squared": self.r_squared,
            "r_cubed": self.r_cubed,
            "r_fourth": self.r_fourth,
            "bond_order_sum": self.bond_order_sum,
        }


@dataclass(frozen=True)
class DdecPair:
    """One bond order between an atom and a periodic image of another atom.

    Chargemol prints every bond from both sides, so a bond between two atoms of
    the cell appears twice with opposite image vectors.

    Attributes:
        atom1: One-based index of the atom in the reference unit cell.
        atom2: One-based index of the bonded atom before its translation.
        element1: Element of ``atom1``.
        element2: Element of ``atom2``.
        image: Translation ``(i, j, k)`` in lattice vectors that is applied to
            ``atom2``.
        bond_order: DDEC bond order.
        spin_polarization: Average spin polarization of the bonding.
        distance: Distance between the two atoms in Angstrom, ``None`` when the
            cell of the output could not be read.
    """

    atom1: int
    atom2: int
    element1: str
    element2: str
    image: Tuple[int, int, int]
    bond_order: float
    spin_polarization: float
    distance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the pair as a plain dictionary."""
        return {
            "atom1": self.atom1,
            "atom2": self.atom2,
            "element1": self.element1,
            "element2": self.element2,
            "image": list(self.image),
            "bond_order": self.bond_order,
            "spin_polarization": self.spin_polarization,
            "distance": self.distance,
        }


@dataclass
class DdecAnalysis:
    """Complete result of one Chargemol run."""

    job: str
    charge_type: str
    nspin: int
    density_source: str
    net_charge: float
    core_electrons: Tuple[CoreElectrons, ...]
    atoms: Tuple[DdecAtom, ...]
    pairs: Tuple[DdecPair, ...]
    total_net_charge: float
    total_spin_moment: Optional[float]
    electron_accounting: Dict[str, Any]
    warnings: Tuple[str, ...]
    workdir: str
    outputs: Tuple[str, ...]
    log: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the analysis as a JSON-serialisable dictionary."""
        return {
            "job": self.job,
            "charge_type": self.charge_type,
            "nspin": self.nspin,
            "density_source": self.density_source,
            "net_charge": self.net_charge,
            "core_electrons": [entry.to_dict() for entry in self.core_electrons],
            "total_net_charge": self.total_net_charge,
            "total_spin_moment": self.total_spin_moment,
            "electron_accounting": self.electron_accounting,
            "warnings": list(self.warnings),
            "workdir": self.workdir,
            "outputs": list(self.outputs),
            "log": self.log,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


def _float(text: str) -> float:
    """Read a floating point number written by Chargemol's Fortran code."""
    return float(text.replace("D", "E").replace("d", "e"))


def element_symbol(z: int) -> str:
    """Return the chemical symbol of an atomic number."""
    if 0 < z < len(chemical_symbols):
        return str(chemical_symbols[z])
    return f"Z{z}"


def core_reference_path(directory: str | os.PathLike, z: int, ncore: int) -> Path:
    """Return the core reference density file Chargemol would read."""
    return Path(directory) / (
        f"core_{z:03d}_{z:03d}_{ncore:03d}_{REFERENCE_CUTOFF_PM:03d}_"
        f"{REFERENCE_SHELLS:03d}.txt"
    )


def reference_ion_path(directory: str | os.PathLike, z: int, electrons: int) -> Path:
    """Return one reference ion density file of the ``c2`` tables."""
    return Path(directory) / (
        f"c2_{z:03d}_{z:03d}_{electrons:03d}_{REFERENCE_CUTOFF_PM:03d}_"
        f"{REFERENCE_SHELLS:03d}.txt"
    )


def _reference_inventory(
    directory: str | os.PathLike,
) -> Tuple[set, Dict[int, set]]:
    """List the core and reference-ion tables that a directory provides.

    Returns:
        A tuple ``(core, ions)`` where ``core`` holds every ``(Z, ncore)`` pair
        and ``ions`` maps an atomic number to the electron counts of its
        ``c2`` files.
    """
    core: set = set()
    ions: Dict[int, set] = {}
    for name in os.listdir(directory):
        core_match = re.match(r"^core_(\d{3})_(\d{3})_(\d{3})_(\d+)_(\d+)\.txt$", name)
        if core_match:
            core.add((int(core_match.group(1)), int(core_match.group(3))))
            continue
        ion_match = re.match(r"^c2_(\d{3})_(\d{3})_(\d{3})_(\d+)_(\d+)\.txt$", name)
        if ion_match:
            ions.setdefault(int(ion_match.group(1)), set()).add(int(ion_match.group(3)))
    return core, ions


def reference_issues(
    directory: str | os.PathLike,
    core_electrons: Sequence[CoreElectrons],
) -> List[DdecIssue]:
    """Check the atomic density tables Chargemol needs for a calculation.

    Args:
        directory: ``atomic_densities`` directory of the Chargemol distribution.
        core_electrons: Core electron count of every element of the job.

    Returns:
        One issue per missing file, plus warnings for reference-ion tables that
        the DDEC iteration may ask for but the distribution does not ship.
    """
    core, ions = _reference_inventory(directory)
    issues: List[DdecIssue] = []
    for entry in core_electrons:
        # Elements with zero core electrons need no ``core_*`` table, but the
        # DDEC iteration still builds their reference ions from the ``c2_*``
        # tables, so the reference-ion checks always run.
        if entry.ncore > 0 and (entry.z, entry.ncore) not in core:
            path = core_reference_path(directory, entry.z, entry.ncore)
            issues.append(
                DdecIssue(
                    "error",
                    "missing-core-density",
                    f"no core reference density for {entry.element} with "
                    f"{entry.ncore} core electrons (Z - z_valence = "
                    f"{entry.z} - {entry.z_valence:g}); Chargemol would look for "
                    f"{path.name} and stop",
                )
            )
        neutral = entry.z
        if neutral not in ions.get(entry.z, set()):
            path = reference_ion_path(directory, entry.z, neutral)
            issues.append(
                DdecIssue(
                    "error",
                    "missing-reference-ion",
                    f"no reference ion density for neutral {entry.element}: "
                    f"{path.name} is missing",
                )
            )
        missing = _missing_reference_ions(ions, entry.z)
        if missing:
            issues.append(
                DdecIssue(
                    "warning",
                    "incomplete-reference-ion-window",
                    f"{entry.element} has no reference ion density for "
                    f"{sorted(missing)} electrons, which a strongly reduced or "
                    f"oxidized atom would need; Chargemol stops if the DDEC "
                    f"iteration reaches them",
                )
            )
    return issues


def _missing_reference_ions(ions: Dict[int, set], z: int) -> List[int]:
    """Return the electron counts of an element's reference-ion window that are absent."""
    if not 0 < z <= len(_REFERENCE_ION_LOWER):
        return []
    available = ions.get(z, set())
    lower = _REFERENCE_ION_LOWER[z - 1]
    upper = _REFERENCE_ION_UPPER[z - 1]
    wanted = set(range(z - upper, z - lower + 1))
    return sorted(wanted - available)


def core_electron_counts(density: Charge) -> List[CoreElectrons]:
    """Derive the core electron count of every element from a charge density.

    The atom columns of a cube that ABACUS writes hold the atomic number and
    the valence charge of the pseudopotential, and the density that
    :mod:`abacustools.data.charge` builds from a restart file carries the same
    two columns, so no pseudopotential file is needed here.

    Args:
        density: One spin channel of the job density.

    Returns:
        One entry per element, in the order of the atoms.

    Raises:
        DdecError: If the valence charge implies a fractional core electron
            count, which Chargemol could not name.
    """
    entries: List[CoreElectrons] = []
    seen: Dict[int, float] = {}
    for number, valence in zip(density.atom_types, density.atom_charges):
        z = int(number)
        charge = float(valence)
        if z <= 0:
            raise DdecError(
                "the charge density does not carry atomic numbers; pass an "
                "ABACUS charge-density cube or convert the restart file first"
            )
        ncore = z - charge
        if abs(ncore - round(ncore)) > 1.0e-6:
            raise DdecError(
                f"{element_symbol(z)} has a fractional core electron count "
                f"({ncore:g}); correct the cube or pass --core-electrons"
            )
        if z in seen and abs(seen[z] - charge) > 1.0e-6:
            raise DdecError(
                f"{element_symbol(z)} appears with two valence charges in the "
                f"density ({seen[z]:g} and {charge:g})"
            )
        if z in seen:
            continue
        seen[z] = charge
        entries.append(
            CoreElectrons(
                element=element_symbol(z),
                z=z,
                z_valence=charge,
                ncore=int(round(ncore)),
            )
        )
    if not entries:
        raise DdecError("the charge density does not contain any atom")
    return entries


def apply_core_electron_overrides(
    entries: Sequence[CoreElectrons],
    overrides: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[CoreElectrons]:
    """Replace the core electron count of the elements given by ``overrides``."""
    if not overrides:
        return list(entries)
    wanted = {int(z): int(ncore) for z, ncore in overrides}
    known = {entry.z for entry in entries}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise DdecError(
            "the job has no atom with atomic number "
            + ", ".join(str(z) for z in unknown)
        )
    return [
        CoreElectrons(
            entry.element, entry.z, entry.z_valence, wanted.get(entry.z, entry.ncore)
        )
        for entry in entries
    ]


def grid_issues(density: Charge) -> List[DdecIssue]:
    """Check the real-space grid of a density against Chargemol's limits.

    Chargemol computes the volume of one grid cell and stops when it exceeds
    ``maxpixelvolume``; the reference densities are built for a spacing near
    0.14 Bohr, so a coarser grid is reported as a warning as well.
    """
    volume_angstrom3 = abs(float(np.linalg.det(np.asarray(density.cell, dtype=float))))
    pixel_angstrom3 = volume_angstrom3 / float(density.data.size)
    pixel_bohr3 = pixel_angstrom3 * ANG_TO_BOHR**3
    spacing = pixel_bohr3 ** (1.0 / 3.0)
    issues: List[DdecIssue] = []
    if pixel_bohr3 > MAX_PIXEL_VOLUME_BOHR3:
        issues.append(
            DdecIssue(
                "error",
                "coarse-grid",
                f"the charge density grid is too coarse: one grid cell spans "
                f"{pixel_bohr3:.4f} Bohr^3 ({spacing:.3f} Bohr per direction) "
                f"but Chargemol accepts at most {MAX_PIXEL_VOLUME_BOHR3} Bohr^3; "
                f"raise ecutrho and rerun the SCF calculation",
            )
        )
    elif spacing > 2.0 * PREFERRED_GRID_SPACING_BOHR:
        issues.append(
            DdecIssue(
                "warning",
                "coarse-grid",
                f"the charge density grid is coarse for DDEC6: {spacing:.3f} Bohr "
                f"per direction against the recommended "
                f"{PREFERRED_GRID_SPACING_BOHR} Bohr",
            )
        )
    return issues


def chargemol_executable(explicit: Optional[str] = None) -> str:
    """Resolve the Chargemol executable from an argument, the environment or config."""
    candidate = (
        explicit
        or os.environ.get("CHARGEMOL_EXE")
        or CONFIG.get("chargemol", {}).get("exe")
        or "chargemol"
    )
    resolved = shutil.which(candidate)
    if resolved is None:
        raise DdecError(
            f"Chargemol executable not found: {candidate!r}. Install it, set "
            "CHARGEMOL_EXE or pass --chargemol-exe."
        )
    return resolved


def atomic_densities_directory(explicit: Optional[str] = None) -> Path:
    """Resolve the ``atomic_densities`` directory of the Chargemol distribution."""
    candidate = (
        explicit
        or os.environ.get("CHARGEMOL_ATOMIC_DENSITIES")
        or CONFIG.get("chargemol", {}).get("atomic_densities")
    )
    if not candidate:
        raise DdecError(
            "the Chargemol atomic_densities directory is unknown; set "
            "chargemol.atomic_densities in ~/.abacustools/config.yaml, export "
            "CHARGEMOL_ATOMIC_DENSITIES or pass --atomic-densities"
        )
    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise DdecError(f"atomic_densities directory not found: {path}")
    return path


def write_job_control(
    path: str | os.PathLike,
    *,
    net_charge: float,
    periodicity: Sequence[bool],
    core_electrons: Sequence[CoreElectrons],
    atomic_densities: str | os.PathLike,
    charge_type: str = "DDEC6",
    compute_bond_orders: bool = True,
) -> None:
    """Write the ``job_control.txt`` file that Chargemol reads.

    Args:
        path: File to write.
        net_charge: Net charge of the unit cell.
        periodicity: Three flags, one per lattice vector.
        core_electrons: Core electron count of every element of the job.
        atomic_densities: Directory written as the complete path of the tables;
            a trailing separator is added because Chargemol concatenates the
            directory and the file name.
        charge_type: ``"DDEC6"`` or ``"DDEC3"``.
        compute_bond_orders: Whether to compute bond orders and overlap
            populations.
    """
    if charge_type not in CHARGE_TYPES:
        raise DdecError(f"unknown charge type: {charge_type!r}")
    directory = str(atomic_densities)
    if not directory.endswith(os.sep):
        directory += os.sep
    lines: List[str] = [
        "<net charge>",
        f"{float(net_charge):.6f}",
        "</net charge>",
        "",
        "<periodicity along A, B, and C vectors>",
    ]
    lines += [".true." if flag else ".false." for flag in periodicity]
    lines += [
        "</periodicity along A, B, and C vectors>",
        "",
        "<atomic densities directory complete path>",
        directory,
        "</atomic densities directory complete path>",
        "",
    ]
    if core_electrons:
        lines.append("<number of core electrons>")
        lines += [f"{entry.z} {entry.ncore}" for entry in core_electrons]
        lines += ["</number of core electrons>", ""]
    lines += [
        "<charge type>",
        charge_type,
        "</charge type>",
        "",
        "<compute BOs>",
        ".true." if compute_bond_orders else ".false.",
        "</compute BOs>",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_chargemol(
    workdir: str | os.PathLike,
    *,
    exe: Optional[str] = None,
    threads: Optional[int] = None,
    spin: bool = False,
) -> str:
    """Run Chargemol in a prepared directory and return its standard output.

    Args:
        workdir: Directory that holds ``job_control.txt`` and the cubes.
        exe: Chargemol executable, resolved by :func:`chargemol_executable`
            when omitted.
        threads: Value of ``OMP_NUM_THREADS`` for the OpenMP binary.
        spin: Whether a spin density was written, which selects the hint of the
            error message when the program crashes.

    Returns:
        Everything the program printed to standard output.

    Raises:
        DdecError: If Chargemol exits with a non-zero status.
    """
    executable = chargemol_executable(exe)
    environment = dict(os.environ)
    if threads:
        environment["OMP_NUM_THREADS"] = str(int(threads))
    completed = subprocess.run(
        [executable],
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        crashed = completed.returncode in (-11, 139) or "SIGSEGV" in completed.stderr
        hint = ""
        if crashed and spin:
            hint = (
                "\nChargemol 3.5 crashes on a valence-only cube together with "
                "spin_density.cube: module_read_spin_density_cube_files uses "
                "atomic_number2/coords2/effective_nuclear_charge2, which "
                "module_format_valence_cube_density never allocates. Patch and "
                "rebuild Chargemol, or rerun with --no-spin."
            )
        detail = completed.stderr.strip() or completed.stdout.strip().splitlines()[-1:]
        raise DdecError(
            f"Chargemol exited with code {completed.returncode}: {detail}{hint}"
        )
    return completed.stdout


def _read_xyz_table(path: Path) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """Read the atom table of a Chargemol ``*.xyz`` result file.

    The files start with the number of atoms, a jmol script line that carries
    the cell, and one line per atom with the element, the position in Angstrom
    and the quantity of the file.

    Returns:
        The atom records and the cell in Angstrom, or ``None`` when the jmol
        line does not carry one.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2 or not _INT_LINE.match(lines[0]):
        raise DdecError(f"{path.name} is not a Chargemol result file")
    natoms = int(lines[0].split()[0])
    cell = _cell_from_jmol_line(lines[1])
    records: List[Dict[str, Any]] = []
    for line in lines[2 : 2 + natoms]:
        fields = line.split()
        if len(fields) < 5:
            raise DdecError(f"unexpected line in {path.name}: {line.strip()!r}")
        records.append(
            {
                "element": fields[0],
                "position": (_float(fields[1]), _float(fields[2]), _float(fields[3])),
                "value": _float(fields[4]),
            }
        )
    if len(records) != natoms:
        raise DdecError(f"{path.name} lists {natoms} atoms but holds {len(records)}")
    return records, cell


def _cell_from_jmol_line(line: str) -> Optional[np.ndarray]:
    """Read the unit cell that Chargemol prints in its jmol script line."""
    marker = line.lower().find("unitcell")
    if marker < 0:
        return None
    blocks = _JMOL_CELL_BLOCK.findall(line[marker:])
    if len(blocks) < 3:
        return None
    try:
        cell = np.array([[float(value) for value in block.split()[:3]] for block in blocks[:3]])
    except ValueError:
        return None
    if cell.shape != (3, 3):
        return None
    return cell


def read_bond_orders(path: Path) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Read a Chargemol bond-order file.

    Returns:
        The sum of bond orders of every atom, and one record per printed bond.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2 or not _INT_LINE.match(lines[0]):
        raise DdecError(f"{path.name} is not a Chargemol bond-order file")
    natoms = int(lines[0].split()[0])
    sums: List[float] = []
    for line in lines[2 : 2 + natoms]:
        fields = line.split()
        if len(fields) < 5:
            raise DdecError(f"unexpected line in {path.name}: {line.strip()!r}")
        sums.append(_float(fields[4]))
    records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in lines[2 + natoms :]:
        block = _BOND_BLOCK.search(line)
        if block:
            current = {"atom1": int(block.group(1)), "element1": block.group(2)}
            continue
        pair = _BOND_PAIR.search(line)
        if pair and current is not None:
            records.append(
                {
                    "atom1": current["atom1"],
                    "element1": current["element1"],
                    "atom2": int(pair.group(4)),
                    "element2": pair.group(5),
                    "image": (int(pair.group(1)), int(pair.group(2)), int(pair.group(3))),
                    "bond_order": _float(pair.group(6)),
                    "spin_polarization": _float(pair.group(7)),
                }
            )
    return sums, records


def read_analysis_log(path: Path) -> Dict[str, Any]:
    """Read the electron accounting and grid verdict of a Chargemol run."""
    text = path.read_text(encoding="utf-8", errors="replace")
    values: Dict[str, Any] = {}
    for key, pattern in _ANALYSIS_VALUE.items():
        match = pattern.search(text)
        values[key] = None if match is None else _float(match.group(1))
    values["grid_adequate"] = (
        "The grid spacing in your electron density input file is adequate." in text
    )
    values["electrons_accounted"] = "all electrons are properly accounted for" in text
    return values


def _total_spin_moment(path: Path) -> Optional[float]:
    """Read the total spin moment that the spin-moment file reports."""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = _TOTAL_SPIN_LINE.search(text)
    return None if match is None else _float(match.group(1))


def _pair_distance(
    atom1: int,
    atom2: int,
    image: Sequence[int],
    positions: Sequence[Sequence[float]],
    cell: Optional[np.ndarray],
) -> Optional[float]:
    """Return the distance between an atom and a translated atom.

    Chargemol prints the pair as an atom of the reference cell bonded to a
    translated image of another atom, so the image is shifted by
    ``image @ cell`` before the distance is taken.
    """
    if cell is None:
        return None
    try:
        first = np.asarray(positions[atom1 - 1], dtype=float)
        second = (
            np.asarray(positions[atom2 - 1], dtype=float)
            + np.asarray(image, dtype=float) @ cell
        )
    except IndexError:
        return None
    return float(np.linalg.norm(second - first))


def parse_pairs(text: Optional[str]) -> Optional[set]:
    """Parse a ``--pairs`` selector such as ``1-2,1-3`` into atom tuples."""
    if not text:
        return None
    wanted = set()
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = re.split(r"[-:]", chunk)
        if len(parts) != 2:
            raise DdecError(f"cannot read the atom pair {chunk!r}; use forms like 1-2,1-3")
        try:
            wanted.add((int(parts[0]), int(parts[1])))
        except ValueError as error:
            raise DdecError(f"cannot read the atom pair {chunk!r}: {error}") from error
    return wanted or None


def read_pairs_file(path: str | os.PathLike) -> Optional[set]:
    """Read a file with one atom pair per line."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_pairs(",".join(line.strip() for line in text.splitlines() if line.strip()))


def select_pairs(
    pairs: Sequence[DdecPair],
    *,
    selection: Optional[set] = None,
    cutoff: Optional[float] = None,
    threshold: Optional[float] = None,
) -> List[DdecPair]:
    """Filter the printed bond orders.

    Args:
        pairs: Every bond that Chargemol printed.
        selection: Pairs of one-based atom indices; a bond is kept when both
            atoms appear in one of them, in either order.
        cutoff: Largest distance in Angstrom.
        threshold: Smallest bond order.
    """
    selected = []
    for pair in pairs:
        if threshold is not None and abs(pair.bond_order) < threshold:
            continue
        if cutoff is not None and (pair.distance is None or pair.distance > cutoff):
            continue
        if selection is not None and (pair.atom1, pair.atom2) not in selection and (
            pair.atom2,
            pair.atom1,
        ) not in selection:
            continue
        selected.append(pair)
    return selected


def _net_charge_from_input(inputs: Any, valences: Sequence[float]) -> Optional[float]:
    """Return the cell charge implied by the ``nelec`` keyword of INPUT."""
    value = inputs.get("nelec")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return float(sum(valences)) - float(value)
    except (TypeError, ValueError):
        return None


def assemble_density(
    job: str | os.PathLike,
    inputs: Any,
    *,
    cube: Optional[str] = None,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    lat0: float = ANG_TO_BOHR,
) -> Tuple[Charge, Optional[Charge], str]:
    """Build the total and spin densities that Chargemol reads.

    Args:
        job: ABACUS job directory.
        inputs: Parsed INPUT of the job.
        cube: Explicit charge-density cube or directory, relative to ``job``.
        grid_shape: FFT grid used to convert a ``*-CHARGE-DENSITY.restart``
            file, read from the running log when omitted.
        lat0: ``LATTICE_CONSTANT`` of the job in Bohr.

    Returns:
        The total density, the magnetization density of an ``nspin 2`` job and
        a description of the files the density came from.

    Raises:
        DdecError: If the job has no usable charge density.
    """
    job_path = Path(job)
    try:
        source = find_density_source(job_path, inputs, cube=cube)
        if source.kind == "cube":
            channels = read_cube_charges(source)
        else:
            structure_file = job_path / str(inputs.get("stru_file", "STRU"))
            structure = AbacusSTRU.read(str(structure_file))
            if structure is None:
                raise DdecError(f"cannot read the structure of {job_path}")
            valences = valence_electrons(
                structure, inputs.get("pseudo_dir"), job_path
            )
            outdir = job_path / f"OUT.{inputs.get('suffix', 'ABACUS')}"
            shape = grid_shape or fft_grid_from_log(outdir / "running_scf.log")
            if shape is None:
                raise DdecError(
                    "the restart file does not report the FFT grid; keep the "
                    "running log or pass --grid NX NY NZ"
                )
            channels = read_restart_charges(
                source,
                structure=structure,
                valences=valences,
                grid_shape=shape,
                lat0=lat0,
            )
        check_channel_count(source, len(channels))
    except ChargeDensityError as error:
        raise DdecError(str(error)) from error
    total = total_charge(channels)
    spin = combine(channels[0], channels[1], -1.0) if len(channels) == 2 else None
    return total, spin, source.describe()


def analyze_ddec(
    job: str | os.PathLike,
    *,
    charge_type: str = "DDEC6",
    net_charge: Optional[float] = None,
    periodicity: Sequence[bool] = (True, True, True),
    core_electrons: Optional[Sequence[Tuple[int, int]]] = None,
    exe: Optional[str] = None,
    atomic_densities: Optional[str] = None,
    compute_bond_orders: bool = True,
    spin: Optional[bool] = None,
    cube: Optional[str] = None,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    lat0: float = ANG_TO_BOHR,
    workdir: Optional[str | os.PathLike] = None,
    keep: bool = False,
    threads: Optional[int] = None,
) -> DdecAnalysis:
    """Run a full DDEC analysis on an ABACUS job directory.

    The valence density is taken from ``SPIN*_CHG.cube`` when the job wrote it,
    otherwise from ``*-CHARGE-DENSITY.restart``. An ``nspin 2`` job also writes
    the magnetization density, which makes Chargemol report atomic spin
    moments.

    Args:
        job: ABACUS job directory.
        charge_type: ``"DDEC6"`` or ``"DDEC3"``.
        net_charge: Net charge of the cell; taken from ``nelec`` of INPUT when
            the keyword is present, and zero otherwise.
        periodicity: Periodicity along the three lattice vectors. A molecule in
            a box wants ``(False, False, False)``.
        core_electrons: ``(Z, ncore)`` overrides of the values that the valence
            charges imply.
        exe: Chargemol executable.
        atomic_densities: ``atomic_densities`` directory of the distribution.
        compute_bond_orders: Whether Chargemol computes bond orders.
        spin: Write the magnetization density of an ``nspin 2`` job. Defaults
            to ``True`` for ``nspin 2``.
        cube: Explicit charge-density cube or directory.
        grid_shape: FFT grid for restart input.
        lat0: ``LATTICE_CONSTANT`` of the job in Bohr.
        workdir: Directory that keeps the cubes and the Chargemol output.
        keep: Keep a temporary working directory.
        threads: Value of ``OMP_NUM_THREADS``.

    Returns:
        The parsed :class:`DdecAnalysis`.

    Raises:
        DdecError: If the job cannot be analysed or Chargemol fails.
    """
    if charge_type not in CHARGE_TYPES:
        raise DdecError(f"unknown charge type: {charge_type!r}")
    job_path = Path(job).expanduser().absolute()
    inputs = ReadInput(str(job_path / "INPUT"))
    nspin = int(inputs.get("nspin", 1))
    if nspin not in (1, 2):
        raise DdecError(
            f"nspin={nspin} is not supported (only 1 and 2); ABACUS nspin 4 "
            "densities would have to be mapped onto spin_density_x/y/z.cube"
        )
    total, magnetization, source = assemble_density(
        job_path, inputs, cube=cube, grid_shape=grid_shape, lat0=lat0
    )

    entries = apply_core_electron_overrides(core_electron_counts(total), core_electrons)
    directory = atomic_densities_directory(atomic_densities)
    issues = reference_issues(directory, entries) + grid_issues(total)
    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue.message for issue in issues if issue.level == "warning"]
    if errors:
        raise DdecError(
            "Chargemol cannot analyse this job:\n"
            + "\n".join(f"  - [{issue.code}] {issue.message}" for issue in errors)
        )

    if net_charge is None:
        valences = [entry.z_valence for entry in entries]
        derived = _net_charge_from_input(inputs, valences)
        net_charge = 0.0 if derived is None else derived
    want_spin = (nspin == 2) if spin is None else bool(spin)
    if want_spin and magnetization is None:
        raise DdecError("a spin density needs an nspin 2 charge density")

    cleanup = False
    if workdir is not None:
        work = Path(workdir).expanduser().absolute()
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="abacustools-ddec-"))
        cleanup = not keep

    outputs: List[str] = []
    try:
        total.save_cube(str(work / "valence_density.cube"), format="abacus")
        if want_spin and magnetization is not None:
            magnetization.save_cube(str(work / "spin_density.cube"), format="abacus")
        write_job_control(
            work / "job_control.txt",
            net_charge=net_charge,
            periodicity=periodicity,
            core_electrons=entries,
            atomic_densities=directory,
            charge_type=charge_type,
            compute_bond_orders=compute_bond_orders,
        )
        run_chargemol(work, exe=exe, threads=threads, spin=want_spin)

        files = CHARGE_TYPE_FILES[charge_type]
        charges_path = work / files["charges"]
        if not charges_path.is_file():
            raise DdecError(
                f"Chargemol did not write {files['charges']}; see the output of "
                f"the run in {work}"
            )
        charge_records, cell = _read_xyz_table(charges_path)
        outputs.append(files["charges"])

        moments: Dict[str, List[Dict[str, Any]]] = {}
        for key, name in RADIAL_MOMENT_FILES.items():
            path = work / name
            if path.is_file():
                moments[key] = _read_xyz_table(path)[0]
                outputs.append(name)

        spin_records: List[Dict[str, Any]] = []
        total_spin: Optional[float] = None
        spin_path = work / files["spin"]
        if want_spin and spin_path.is_file():
            spin_records, _ = _read_xyz_table(spin_path)
            total_spin = _total_spin_moment(spin_path)
            if total_spin is None:
                total_spin = sum(record["value"] for record in spin_records)
            outputs.append(files["spin"])

        atoms: List[DdecAtom] = []
        for index, record in enumerate(charge_records, start=1):
            atom = DdecAtom(
                index=index,
                element=record["element"],
                position=record["position"],
                net_charge=record["value"],
                spin_moment=(
                    spin_records[index - 1]["value"] if index <= len(spin_records) else None
                ),
            )
            for key, records in moments.items():
                if index <= len(records):
                    setattr(atom, key, records[index - 1]["value"])
            atoms.append(atom)

        pairs: List[DdecPair] = []
        if compute_bond_orders:
            bond_path = work / files["bond_orders"]
            if bond_path.is_file():
                sums, records = read_bond_orders(bond_path)
                outputs.append(files["bond_orders"])
                for index, atom in enumerate(atoms):
                    if index < len(sums):
                        atom.bond_order_sum = sums[index]
                positions = [atom.position for atom in atoms]
                pairs = [
                    DdecPair(
                        atom1=record["atom1"],
                        atom2=record["atom2"],
                        element1=record["element1"],
                        element2=record["element2"],
                        image=record["image"],
                        bond_order=record["bond_order"],
                        spin_polarization=record["spin_polarization"],
                        distance=_pair_distance(
                            record["atom1"],
                            record["atom2"],
                            record["image"],
                            positions,
                            cell,
                        ),
                    )
                    for record in records
                ]

        log_path = next(iter(sorted(work.glob("*_DDEC_analysis.output"))), None)
        accounting = read_analysis_log(log_path) if log_path is not None else {}
        if log_path is not None:
            outputs.append(log_path.name)
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)

    return DdecAnalysis(
        job=str(job_path),
        charge_type=charge_type,
        nspin=nspin,
        density_source=source,
        net_charge=float(net_charge),
        core_electrons=tuple(entries),
        atoms=tuple(atoms),
        pairs=tuple(pairs),
        total_net_charge=float(sum(atom.net_charge for atom in atoms)),
        total_spin_moment=total_spin,
        electron_accounting=accounting,
        warnings=tuple(warnings),
        workdir=str(work),
        outputs=tuple(outputs),
        log=None if log_path is None else log_path.name,
    )
