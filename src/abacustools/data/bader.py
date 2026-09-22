"""Bader charge analysis driven by the Henkelman grid-based ``bader`` program.

The external executable (https://theory.cm.utexas.edu/henkelman/code/bader/)
reads a charge density in Gaussian cube format, partitions space into Bader
volumes and writes the assigned electrons to ``ACF.dat``. This module builds
the required cube inputs from ABACUS charge densities (``SPIN*_CHG.cube`` or
``*-CHARGE-DENSITY.restart``), runs the program and parses its output.

Only collinear densities (``nspin=1`` and ``nspin=2``) are supported. For
``nspin=2`` the total density (up + down) is used for the Bader partitioning
and the magnetization density (up - down) is integrated over the same volumes
to obtain per-atom spin moments.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ase.data import chemical_symbols

from abacustools.core.config import CONFIG
from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.data.charge import (
    ChargeDensityError,
    combine,
    fft_grid_from_log,
    find_density_source,
    read_cube_charges,
    read_restart_charges,
    total_charge,
    valence_electrons as _valence_electrons,
)
from abacustools.data.grid import Charge
from abacustools.io.abacus import ReadInput
from abacustools.io.stru import AbacusSTRU


A2BOHR = ANG_TO_BOHR
BOHR3_TO_ANG3 = BOHR_TO_ANG**3

class BaderError(RuntimeError):
    """Raised when a Bader analysis cannot be completed."""


@dataclass
class BaderAtom:
    """Bader analysis result for a single atom.

    Attributes:
        index: One-based atom index.
        element: Element symbol.
        position: Cartesian position in Angstrom.
        z_valence: Number of valence electrons of the pseudopotential.
        bader_charge: Electrons inside the Bader volume of the atom.
        min_distance: Distance from the atom to the nearest point of its Bader
            surface in Angstrom, as reported by the backend.
        atomic_volume: Volume of the Bader volume in Angstrom**3.
        spin_moment: Up minus down electrons inside the Bader volume, ``None``
            for ``nspin 1``.
    """

    index: int
    element: str
    position: Tuple[float, float, float]
    z_valence: float
    bader_charge: float
    min_distance: float
    atomic_volume: float
    spin_moment: Optional[float] = None

    @property
    def net_charge(self) -> float:
        return self.z_valence - self.bader_charge


@dataclass
class BaderAnalysis:
    """Complete Bader analysis of one ABACUS job."""

    job: Path
    nspin: int
    atoms: List[BaderAtom]
    vacuum_charge: float
    vacuum_volume: float
    number_of_electrons: float
    charge_source: str
    workdir: Path
    bader_stdout: str = ""
    reference: Optional[Path] = None
    backend: str = "bader"

    @property
    def total_net_charge(self) -> float:
        return float(sum(atom.net_charge for atom in self.atoms))

    @property
    def total_z_valence(self) -> float:
        return float(sum(atom.z_valence for atom in self.atoms))

    def to_dict(self) -> dict:
        return {
            "job": str(self.job),
            "nspin": self.nspin,
            "backend": self.backend,
            "charge_source": self.charge_source,
            "reference": str(self.reference) if self.reference else None,
            "workdir": str(self.workdir),
            "number_of_electrons": self.number_of_electrons,
            "vacuum_charge": self.vacuum_charge,
            "vacuum_volume": self.vacuum_volume,
            "total_net_charge": self.total_net_charge,
            "total_valence_electrons": self.total_z_valence,
            "atoms": [
                {
                    "index": atom.index,
                    "element": atom.element,
                    "position": list(atom.position),
                    "z_valence": atom.z_valence,
                    "bader_charge": atom.bader_charge,
                    "net_charge": atom.net_charge,
                    "spin_moment": atom.spin_moment,
                    "min_distance": atom.min_distance,
                    "atomic_volume": atom.atomic_volume,
                }
                for atom in self.atoms
            ],
        }


def read_acf(path: str | os.PathLike) -> Tuple[List[dict], float, float, float]:
    """Parse an ``ACF.dat`` file written by the Bader program.

    Returns:
        A tuple ``(records, vacuum_charge, vacuum_volume, number_of_electrons)``
        where every record is a dict with ``index``, ``position``, ``charge``,
        ``min_distance`` and ``atomic_volume``. The file stores lengths in Bohr
        and volumes in Bohr**3; :func:`_build_atoms` converts them.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    records: List[dict] = []
    vacuum_charge = 0.0
    vacuum_volume = 0.0
    number_of_electrons = 0.0
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "VACUUM CHARGE" in stripped:
            vacuum_charge = float(stripped.split(":")[1])
            continue
        if "VACUUM VOLUME" in stripped:
            vacuum_volume = float(stripped.split(":")[1])
            continue
        if "NUMBER OF ELECTRONS" in stripped:
            number_of_electrons = float(stripped.split(":")[1])
            continue
        if set(stripped) <= {"-"}:
            continue
        if stripped.startswith("#"):
            in_table = True
            continue
        if not in_table:
            continue
        fields = stripped.split()
        if len(fields) < 7:
            continue
        records.append(
            {
                "index": int(fields[0]),
                "position": (float(fields[1]), float(fields[2]), float(fields[3])),
                "charge": float(fields[4]),
                "min_distance": float(fields[5]),
                "atomic_volume": float(fields[6]),
            }
        )
    return records, vacuum_charge, vacuum_volume, number_of_electrons


def find_bader_executable(explicit: Optional[str] = None) -> str:
    """Resolve the Bader executable from an argument, the environment or config."""
    candidate = (
        explicit
        or os.environ.get("BADER_EXE")
        or CONFIG.get("bader", {}).get("exe")
        or "bader"
    )
    resolved = shutil.which(candidate)
    if resolved is None:
        raise BaderError(
            f"Bader executable not found: {candidate!r}. Install it or pass --bader-exe."
        )
    return resolved


def run_bader(
    charge_file: str | os.PathLike,
    *,
    reference: Optional[str | os.PathLike] = None,
    exe: Optional[str] = None,
    workdir: Optional[str | os.PathLike] = None,
    extra_args: Sequence[str] = (),
) -> str:
    """Run the Bader program and return its standard output.

    The program writes ``ACF.dat``/``BCF.dat``/``AVF.dat`` into ``workdir``.
    """
    executable = find_bader_executable(exe)
    command = [executable]
    if reference is not None:
        command += ["-ref", str(reference)]
    command += list(extra_args)
    command += [str(charge_file)]
    completed = subprocess.run(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise BaderError(
            f"bader exited with code {completed.returncode}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _vacuum_arguments(vacuum: Optional[object]) -> List[str]:
    if vacuum is None:
        return []
    if isinstance(vacuum, (int, float)):
        return ["-vac", f"{float(vacuum):.6e}"]
    text = str(vacuum)
    if text in {"off", "auto"}:
        return ["-vac", text]
    return ["-vac", f"{float(text):.6e}"]


def _build_atoms(
    records: Sequence[dict],
    elements: Sequence[Optional[str]],
    valences: Sequence[float],
) -> List[BaderAtom]:
    atoms = []
    for record, element, valence in zip(records, elements, valences):
        atoms.append(
            BaderAtom(
                index=record["index"],
                element=element or "",
                position=tuple(
                    float(value) * BOHR_TO_ANG for value in record["position"]
                ),
                z_valence=float(valence),
                bader_charge=record["charge"],
                min_distance=float(record["min_distance"]) * BOHR_TO_ANG,
                atomic_volume=float(record["atomic_volume"]) * BOHR3_TO_ANG3,
            )
        )
    return atoms


@dataclass
class BaderDensity:
    """Charge density of one job, ready for a Bader partition.

    Attributes:
        total: Total density (up + down) that defines the partition.
        magnetization: Up minus down density, ``None`` for ``nspin 1``.
        elements: Element symbol of every atom.
        valences: Number of valence electrons of every atom.
        charge_source: Description of the files the density was read from.
        nspin: Number of spin channels the job declares.
    """

    total: Charge
    magnetization: Optional[Charge]
    elements: List[str]
    valences: List[float]
    charge_source: str
    nspin: int


def read_bader_density(
    job: str | os.PathLike,
    *,
    cube: Optional[str] = None,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    lat0: Optional[float] = None,
) -> BaderDensity:
    """Assemble the density that every Bader backend partitions.

    The density is taken from ``SPIN*_CHG.cube`` when present, otherwise from
    ``*-CHARGE-DENSITY.restart`` (which is converted with an inverse FFT). The
    valence electron count of every atom comes from the cube header or, for a
    restart file, from the pseudopotentials the STRU names.

    Args:
        job: ABACUS job directory.
        cube: Explicit charge-density cube file or directory, relative to
            ``job``.
        grid_shape: FFT grid of a restart file, read from the log when omitted.
        lat0: ``LATTICE_CONSTANT`` of the job in Bohr, taken from the STRU when
            omitted.

    Returns:
        The assembled density.

    Raises:
        BaderError: If the job has no usable charge density or declares an
            ``nspin`` that no Bader partition supports.
    """
    job_path = Path(job).expanduser().absolute()
    inputs = ReadInput(str(job_path / "INPUT"))
    suffix = str(inputs.get("suffix", "ABACUS"))
    nspin = int(inputs.get("nspin", 1))
    if nspin not in (1, 2):
        raise BaderError(f"nspin={nspin} is not supported (only 1 and 2)")

    outdir = job_path / f"OUT.{suffix}"
    if not outdir.is_dir():
        raise BaderError(f"output directory not found: {outdir}")

    try:
        source = find_density_source(job_path, inputs, cube=cube)
        if source.kind == "cube":
            spin_charges = read_cube_charges(source)
            charge_source = source.describe()
            valences = [float(charge) for charge in spin_charges[0].atom_charges]
            elements = [
                chemical_symbols[int(number)] for number in spin_charges[0].atom_types
            ]
        else:
            stru = AbacusSTRU.read(str(job_path / "STRU"))
            valences = _valence_electrons(stru, inputs.get("pseudo_dir"), job_path)
            elements = list(stru.elements)
            shape = grid_shape or fft_grid_from_log(outdir / "running_scf.log")
            if shape is None:
                raise BaderError(
                    "could not determine the FFT grid; pass --grid nx ny nz"
                )
            constant = lat0
            if constant is None:
                constant = float(stru.metadata.get("lattice_constant", 1.0) or 1.0)
            spin_charges = read_restart_charges(
                source,
                structure=stru,
                valences=valences,
                grid_shape=shape,
                lat0=constant,
            )
            charge_source = source.describe(grid=shape)
        total = total_charge(spin_charges)
        magnetization = None
        if nspin == 2:
            magnetization = combine(spin_charges[0], spin_charges[1], -1.0)
    except ChargeDensityError as error:
        raise BaderError(str(error)) from error

    return BaderDensity(
        total=total,
        magnetization=magnetization,
        elements=elements,
        valences=valences,
        charge_source=charge_source,
        nspin=nspin,
    )


def analyze_bader(
    job: str | os.PathLike,
    *,
    cube: Optional[str] = None,
    reference: Optional[str] = None,
    exe: Optional[str] = None,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    lat0: Optional[float] = None,
    vacuum: Optional[object] = None,
    workdir: Optional[str | os.PathLike] = None,
    keep: bool = False,
) -> BaderAnalysis:
    """Run a full Bader analysis on an ABACUS job directory.

    The charge density is taken from ``SPIN*_CHG.cube`` when present, otherwise
    from ``*-CHARGE-DENSITY.restart`` (which is converted with an inverse FFT).
    The partition itself is done by the external Henkelman ``bader`` program;
    :func:`abacustools.integrations.baderkit.analyze_baderkit` runs the same
    partition with the ``baderkit`` library instead.
    """
    job_path = Path(job).expanduser().absolute()
    density = read_bader_density(job_path, cube=cube, grid_shape=grid_shape, lat0=lat0)

    if workdir is not None:
        work = Path(workdir).expanduser().absolute()
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="abacustools-bader-"))
        cleanup = not keep

    try:
        total_cube = work / "charge_total.cube"
        density.total.save_cube(str(total_cube), format="abacus")
        reference_path = None
        if reference is not None:
            reference_path = Path(reference)
            if not reference_path.is_absolute():
                reference_path = job_path / reference_path
        stdout = run_bader(
            total_cube.name,
            reference=reference_path,
            exe=exe,
            workdir=work,
            extra_args=_vacuum_arguments(vacuum),
        )
        records, vacuum_charge, vacuum_volume, number_of_electrons = read_acf(work / "ACF.dat")
        atoms = _build_atoms(records, density.elements, density.valences)

        if density.magnetization is not None:
            spin_cube = work / "charge_spin.cube"
            density.magnetization.save_cube(str(spin_cube), format="abacus")
            run_bader(
                spin_cube.name,
                reference=total_cube.name,
                exe=exe,
                workdir=work,
                extra_args=_vacuum_arguments(vacuum),
            )
            spin_records, *_ = read_acf(work / "ACF.dat")
            for atom, record in zip(atoms, spin_records):
                atom.spin_moment = record["charge"]
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)

    return BaderAnalysis(
        job=job_path,
        nspin=density.nspin,
        atoms=atoms,
        vacuum_charge=vacuum_charge,
        vacuum_volume=vacuum_volume,
        number_of_electrons=number_of_electrons,
        charge_source=density.charge_source,
        workdir=work,
        bader_stdout=stdout,
        reference=reference_path,
        backend="bader",
    )
