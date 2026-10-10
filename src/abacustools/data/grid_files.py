"""Locate the volumetric (grid) files that ABACUS writes.

The two supported ABACUS branches name their cube files differently. The
3.10-LTS branch writes ``SPIN1_CHG.cube``, ``SPIN1_POT.cube``, ``ELF.cube`` and
``SPIN1_TAU.cube``, while the develop branch writes ``chgs1.cube``,
``pots1.cube``, ``elftot.cube`` and ``taus1.cube``, and adds a ``g{step}``
token when ``out_freq_ion`` writes one file per geometry step. The
electrostatic potential of ``out_pot 2`` is ``ElecStaticPot.cube`` in the LTS
branch and ``potes.cube`` in develop, whose manual still mentions the unused
``pot_es.cube``; all three names are recognised here.

This module only answers "which files exist and what are they"; reading them
into grids stays with :mod:`abacustools.data.grid`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple


#: Names of the two supported naming conventions.
LTS = "lts"
DEVELOP = "develop"

#: Quantities this module can locate.
QUANTITIES = (
    "charge",
    "tau",
    "elf",
    "elf_total",
    "potential",
    "potential_es",
    "ldos",
    "partial_charge",
)

#: Quantities with one file per spin channel.
SPIN_RESOLVED = ("charge", "tau", "elf", "potential")


class GridFileError(RuntimeError):
    """Raised when the volumetric files of a job cannot be identified."""


@dataclass(frozen=True)
class GridFile:
    """One volumetric file of an ABACUS job.

    Attributes:
        path: The file itself.
        quantity: ``"charge"``, ``"tau"``, ``"elf"``, ``"elf_total"``,
            ``"potential"``, ``"potential_es"``, ``"ldos"`` or
            ``"partial_charge"``.
        naming: ``"lts"`` or ``"develop"``.
        spin: One-based spin channel, or 0 when the quantity is not spin
            resolved.
        step: Geometry step of a develop file, ``None`` when the file is not
            tied to a geometry step.
        initial: Whether the file holds the initial density or potential.
        energy: Energy of an LDOS file in eV, ``None`` otherwise.
        band: Band index of a partial-charge file, ``None`` otherwise.
        kpoint: k-point index of a partial-charge file, or 0 for the Gamma
            point of a gamma-only calculation, ``None`` otherwise.
    """

    path: Path
    quantity: str
    naming: str
    spin: int = 0
    step: Optional[int] = None
    initial: bool = False
    energy: Optional[float] = None
    band: Optional[int] = None
    kpoint: Optional[int] = None

    def describe(self) -> str:
        """Return the file name together with the convention behind it."""
        details = [self.naming]
        if self.step is not None:
            details.append(f"step {self.step}")
        if self.initial:
            details.append("initial")
        if self.energy is not None:
            details.append(f"{self.energy:g} eV")
        return f"{self.path.name} ({', '.join(details)})"


_LTS_CHARGE = re.compile(r"^SPIN(\d+)_CHG(_INI)?\.cube$")
_DEVELOP_CHARGE = re.compile(r"^chg(?:s(\d+))?(?:g(\d+))?(_ini)?\.cube$")
_LTS_TAU = re.compile(r"^SPIN(\d+)_TAU\.cube$")
_DEVELOP_TAU = re.compile(r"^tau(?:s(\d+))?(?:g(\d+))?\.cube$")
_LTS_ELF = re.compile(r"^ELF\.cube$")
_LTS_ELF_SPIN = re.compile(r"^ELF_SPIN(\d+)\.cube$")
_DEVELOP_ELF = re.compile(r"^elftot(?:g(\d+))?\.cube$")
_DEVELOP_ELF_SPIN = re.compile(r"^elfs(\d+)(?:g(\d+))?\.cube$")
_LTS_POTENTIAL = re.compile(r"^SPIN(\d+)_POT(_INI)?\.cube$")
_DEVELOP_POTENTIAL = re.compile(r"^pot(?:s(\d+))?(?:g(\d+))?(_ini)?\.cube$")
_LTS_POTENTIAL_ES = re.compile(r"^ElecStaticPot\.cube$")
_DEVELOP_POTENTIAL_ES = re.compile(r"^pot(?:es|_es)(?:g(\d+))?\.cube$")
_DEVELOP_LDOS = re.compile(r"^LDOS_(.+?)eV\.cube$")
_LTS_PARTIAL_CHARGE = re.compile(r"^BAND(\d+)_(GAMMA|K(\d+))_SPIN(\d+)_CHG\.cube$")
_DEVELOP_PARTIAL_CHARGE = re.compile(r"^pchgi(\d+)s(\d+)(?:k(\d+))?\.cube$")


def output_directory(job: Path, inputs: Mapping[str, Any]) -> Path:
    """Return the ``OUT.<suffix>`` directory that INPUT points at."""
    return Path(job) / f"OUT.{inputs.get('suffix', 'ABACUS')}"


def spin_channels(nspin: int) -> Tuple[int, ...]:
    """Return the one-based spin channels of a calculation."""
    if nspin <= 1:
        return (1,)
    return tuple(range(1, nspin + 1))


def _match_name(name: str) -> Optional[GridFile]:
    """Classify one file name, or return ``None`` for an unknown name."""
    match = _LTS_CHARGE.match(name)
    if match:
        return GridFile(
            Path(name), "charge", LTS, spin=int(match.group(1)),
            initial=bool(match.group(2)),
        )
    match = _DEVELOP_CHARGE.match(name)
    if match:
        return GridFile(
            Path(name), "charge", DEVELOP, spin=int(match.group(1) or 1),
            step=int(match.group(2)) if match.group(2) else None,
            initial=bool(match.group(3)),
        )
    match = _LTS_TAU.match(name)
    if match:
        return GridFile(Path(name), "tau", LTS, spin=int(match.group(1)))
    match = _DEVELOP_TAU.match(name)
    if match:
        return GridFile(
            Path(name), "tau", DEVELOP, spin=int(match.group(1) or 1),
            step=int(match.group(2)) if match.group(2) else None,
        )
    match = _LTS_ELF_SPIN.match(name)
    if match:
        return GridFile(Path(name), "elf", LTS, spin=int(match.group(1)))
    if _LTS_ELF.match(name):
        return GridFile(Path(name), "elf_total", LTS)
    match = _DEVELOP_ELF_SPIN.match(name)
    if match:
        return GridFile(
            Path(name), "elf", DEVELOP, spin=int(match.group(1)),
            step=int(match.group(2)) if match.group(2) else None,
        )
    match = _DEVELOP_ELF.match(name)
    if match:
        return GridFile(
            Path(name), "elf_total", DEVELOP,
            step=int(match.group(1)) if match.group(1) else None,
        )
    match = _LTS_POTENTIAL.match(name)
    if match:
        return GridFile(
            Path(name), "potential", LTS, spin=int(match.group(1)),
            initial=bool(match.group(2)),
        )
    match = _DEVELOP_POTENTIAL.match(name)
    if match:
        return GridFile(
            Path(name), "potential", DEVELOP, spin=int(match.group(1) or 1),
            step=int(match.group(2)) if match.group(2) else None,
            initial=bool(match.group(3)),
        )
    if _LTS_POTENTIAL_ES.match(name):
        return GridFile(Path(name), "potential_es", LTS)
    match = _DEVELOP_POTENTIAL_ES.match(name)
    if match:
        return GridFile(
            Path(name), "potential_es", DEVELOP,
            step=int(match.group(1)) if match.group(1) else None,
        )
    match = _DEVELOP_LDOS.match(name)
    if match:
        try:
            energy = float(match.group(1))
        except ValueError:
            return None
        return GridFile(Path(name), "ldos", DEVELOP, energy=energy)
    match = _LTS_PARTIAL_CHARGE.match(name)
    if match:
        return GridFile(
            Path(name), "partial_charge", LTS,
            spin=int(match.group(4)),
            band=int(match.group(1)),
            kpoint=0 if match.group(2) == "GAMMA" else int(match.group(3)),
        )
    match = _DEVELOP_PARTIAL_CHARGE.match(name)
    if match:
        return GridFile(
            Path(name), "partial_charge", DEVELOP,
            spin=int(match.group(2)),
            band=int(match.group(1)),
            kpoint=int(match.group(3)) if match.group(3) else 0,
        )
    return None


def scan_grid_files(
    outdir: Path,
    *,
    quantity: Optional[str] = None,
) -> List[GridFile]:
    """List the volumetric files of an output directory.

    Args:
        outdir: ``OUT.<suffix>`` directory of a job.
        quantity: Restrict the result to one quantity.

    Returns:
        Every recognised file, sorted by quantity, spin, step and name.

    Raises:
        GridFileError: If the directory does not exist or the quantity is
            unknown.
    """
    if quantity is not None and quantity not in QUANTITIES:
        raise GridFileError(f"unknown volumetric quantity: {quantity!r}")
    outdir = Path(outdir)
    if not outdir.is_dir():
        raise GridFileError(f"output directory not found: {outdir}")
    files = []
    for path in sorted(outdir.glob("*.cube")):
        found = _match_name(path.name)
        if found is None:
            continue
        if quantity is not None and found.quantity != quantity:
            continue
        files.append(
            GridFile(
                path,
                found.quantity,
                found.naming,
                spin=found.spin,
                step=found.step,
                initial=found.initial,
                energy=found.energy,
                band=found.band,
                kpoint=found.kpoint,
            )
        )
    files.sort(key=lambda item: (item.quantity, item.spin, item.step or 0, item.path.name))
    return files


def grid_files(
    outdir: Path,
    quantity: str,
    *,
    spins: Sequence[int] = (),
    step: Optional[int] = None,
    initial: bool = False,
) -> List[GridFile]:
    """Select the files of one quantity from either naming convention.

    Args:
        outdir: ``OUT.<suffix>`` directory of a job.
        quantity: Quantity to select, such as ``"charge"``.
        spins: One-based spin channels to return; every channel when empty.
        step: Geometry step to select. By default a file without a step token
            wins, and the last step is used when the job only wrote step files.
        initial: Select the initial rather than the self-consistent data.

    Returns:
        One file per requested spin channel, in channel order.

    Raises:
        GridFileError: If a job mixes the two naming conventions, or a
            requested channel or step does not exist.
    """
    if quantity not in QUANTITIES:
        raise GridFileError(f"unknown volumetric quantity: {quantity!r}")
    files = scan_grid_files(outdir, quantity=quantity)
    requested = tuple(spins) if spins else tuple(sorted({item.spin for item in files}))
    selected: List[GridFile] = []
    for spin in requested:
        candidates = [
            item for item in files if item.spin == spin and item.initial == initial
        ]
        if step is None:
            plain = [item for item in candidates if item.step is None]
            if plain:
                chosen = plain
            elif candidates:
                chosen = [max(candidates, key=lambda item: item.step or 0)]
            else:
                chosen = []
        else:
            chosen = [item for item in candidates if item.step == step]
        if not chosen:
            detail = "initial" if initial else "final"
            where = "" if step is None else f" at step {step}"
            raise GridFileError(
                f"no {detail} {quantity} file for spin {spin}{where} in {outdir}"
            )
        selected.extend(chosen)
    namings = {item.naming for item in selected}
    if len(namings) > 1:
        names = ", ".join(sorted(item.path.name for item in selected))
        raise GridFileError(
            f"the {quantity} files mix the LTS and develop naming conventions: {names}"
        )
    return selected
