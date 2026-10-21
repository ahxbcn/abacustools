"""Collect selected results from ABACUS calculation output."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from abacustools.core.constant import BOHR_TO_ANG, RY_TO_EV
from abacustools.data.abacus_json import latest_ionic_output, read_abacus_json
from abacustools.data.versions import resolve_version
from abacustools.io.abacus import ReadInput


grouped_params = {
    "scf_results": [
        "energy",
        "drho",
        "denergy",
        "scf_steps",
        "converged",
        "normal_end",
        "efermi",
        "force",
        "stress",
    ],
    "vdw_results": [
        "vdw_energy",
    ],
    "magnetic_results": [
        "total_mag",
        "absolute_mag",
        "atom_mag_mulliken",
        "atom_orb_mag",
    ],
    "relax_results": [
        "largest_force",
        "largest_stress",
        "relax_steps",
        "relax_converged",
    ],
}

_RELAX_CALCULATIONS = {"relax", "cell-relax", "md"}
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_RELAX_ENERGY_DIFF_RE = re.compile(rf"etot\s+diff\s*\(\s*eV\s*\)\s*:\s*({_FLOAT})", re.IGNORECASE)
_COORDINATE_HEADER_RE = re.compile(
    rf"^\s*cartesian\s+coordinates(?:\s*\(\s*unit\s*=\s*({_FLOAT})\s*bohr\s*\))?",
    re.IGNORECASE,
)
_DIRECT_COORDINATE_HEADER_RE = re.compile(
    r"^\s*direct\s+coordinates\s*$",
    re.IGNORECASE,
)
_LATTICE_VECTOR_HEADER_RE = re.compile(
    r"lattice\s+vectors\s*:?\s*\(\s*cartesian",
    re.IGNORECASE,
)
_LATTICE_CONSTANT_ANGSTROM_RE = re.compile(
    rf"lattice\s+constant\s*\(\s*angstrom\s*\)\s*=\s*({_FLOAT})",
    re.IGNORECASE,
)
_LATTICE_CONSTANT_BOHR_RE = re.compile(
    rf"lattice\s+constant\s*\(\s*bohr\s*\)\s*=\s*({_FLOAT})",
    re.IGNORECASE,
)
_STRESS_COMPONENTS = (
    ("xx", "xy", "xz"),
    ("xy", "yy", "yz"),
    ("xz", "yz", "zz"),
)
# Develop-branch output files: the LTS ``DOS<n>_smearing.dat`` files are
# renamed to ``doss<spin>[g<step>]_nao.txt`` (LCAO) or ``doss<spin>.txt`` /
# ``dos.txt`` (PW), and the XML ``PDOS`` file is replaced by text files
# ``pdoss<spin>[g<step>]_<basis>.txt``. ``g<step>`` appears when the writer
# names one file per ionic step; without it all ionic steps are appended to
# one file as repeated blocks.
_DOS_NAO_FILE_RE = re.compile(r"^dos(?:s(\d+))?(?:g(\d+))?(?:_nao)?\.txt$", re.IGNORECASE)
_PDOS_NAO_FILE_RE = re.compile(r"^pdoss(\d+)(?:g(\d+))?_\w+\.txt$", re.IGNORECASE)
_IONIC_STEP_HEADER_RE = re.compile(r"^\s*(\d+)\s*#\s*ionic step", re.IGNORECASE)


_FLOAT_RE = re.compile(_FLOAT)
_NEVER_MATCH = re.compile(r"(?!)")
_GLOBAL_FLAG_RE = re.compile(r"\(\?[aiLmsux]+\)")


def _as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float(value.replace("D", "E").replace("d", "e"))


def _numbers(line: str) -> List[float]:
    return [_as_float(value) for value in _FLOAT_RE.findall(line)]


@lru_cache(maxsize=None)
def _lowered(keywords: tuple[str, ...]) -> tuple[str, ...]:
    """Return the keywords lowercased once for case-insensitive ``in`` tests."""
    return tuple(keyword.lower() for keyword in keywords)


def _contains_any(line: str, keywords: Sequence[str]) -> bool:
    """Return whether a lowercased log line contains one of the keywords."""
    return any(keyword in line for keyword in _lowered(tuple(keywords)))


def _normalize_pattern(pattern: str) -> tuple[str, bool]:
    """Lowercase the literal parts of a regex for a lowercased log line.

    A gate is matched against ``line.lower()``, which lets it stay
    case-sensitive and fast.  Literal uppercase letters in a pattern therefore
    have to be lowercased too.  Escapes such as ``\\S`` and character classes
    that only list uppercase letters (``[A-Z]``) change meaning when
    lowercased, so they are left alone; the caller wraps such a pattern in a
    scoped ``(?i:...)`` group instead.

    Returns:
        The normalized pattern and whether it still needs case-insensitive
        matching.
    """
    if "(?P" in pattern or "\\N" in pattern:
        # Named groups and Unicode names are case-sensitive; do not touch them.
        return pattern, True

    normalized: List[str] = []
    needs_ignorecase = False
    index = 0
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "\\":
            normalized.append(pattern[index : index + 2])
            index += 2
            continue
        if char == "[":
            # Copy the whole class unchanged; only lower it when every
            # uppercase letter it lists has a lowercase counterpart.
            end = index + 1
            if end < length and pattern[end] == "^":
                end += 1
            if end < length and pattern[end] == "]":
                end += 1
            while end < length and pattern[end] != "]":
                if pattern[end] == "\\":
                    end += 1
                end += 1
            end = min(end + 1, length)
            content = pattern[index:end]
            if any(
                letter.isupper() and letter.lower() not in content
                for letter in content
                if letter.isalpha()
            ):
                needs_ignorecase = True
            normalized.append(content)
            index = end
            continue
        normalized.append(char.lower() if char.isupper() else char)
        index += 1
    return "".join(normalized), needs_ignorecase


def _strip_global_flags(pattern: str) -> str:
    """Move a leading ``(?flags)`` group into a scoped group.

    Python only accepts global inline flags at the start of a whole pattern,
    so a pattern that is one alternative of a gate has to spell them as a
    scoped ``(?flags:...)`` group.  ``i`` is dropped because the fallback gate
    is compiled case-insensitively anyway.
    """
    match = re.match(r"^\(\?([aiLmsux]+)\)", pattern)
    if match is None:
        return pattern
    scoped = "".join(flag for flag in match.group(1) if flag in "msx")
    body = pattern[match.end() :]
    return f"(?{scoped}:{body})" if scoped else body


@lru_cache(maxsize=None)
def _gate(
    literals: tuple[str, ...] = (),
    patterns: tuple[str, ...] = (),
) -> re.Pattern[str]:
    """Compile one marker gate for a lowercased log line.

    Log readers test many marker families against every line.  One
    alternation is much cheaper than a Python loop over every keyword.  The
    gate is matched against ``line.lower()``: keywords and the literal parts
    of the patterns are lowercased here, and only a pattern that cannot be
    lowercased safely keeps a scoped ``(?i:...)`` group.

    Args:
        literals: Plain keywords.
        patterns: Regular expressions that expose their captured value as the
            first group; they only decide whether a line may carry a marker.

    Returns:
        A compiled pattern that matches when any marker may be present.
    """
    literals = _lowered(tuple(literals))
    if any(_GLOBAL_FLAG_RE.search(pattern) for pattern in patterns):
        # A global inline flag cannot move into a scoped group, so fall back
        # to one case-insensitive gate.  Configured profiles are the only
        # source of such patterns, so the slower path stays rare.
        parts = [re.escape(keyword) for keyword in literals]
        parts.extend(f"(?:{_strip_global_flags(pattern)})" for pattern in patterns)
        return re.compile("|".join(parts), re.IGNORECASE) if parts else _NEVER_MATCH

    parts = [re.escape(keyword) for keyword in literals]
    for pattern in patterns:
        normalized, needs_ignorecase = _normalize_pattern(pattern)
        if needs_ignorecase:
            parts.append(f"(?i:{normalized})")
        else:
            parts.append(f"(?:{normalized})")
    return re.compile("|".join(parts)) if parts else _NEVER_MATCH


@lru_cache(maxsize=None)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile and cache the regular expressions of one version profile."""
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def _first_compiled(
    patterns: Sequence[re.Pattern[str]], line: str
) -> Optional[float]:
    """Return the value captured by the first matching compiled pattern."""
    for pattern in patterns:
        match = pattern.search(line)
        if match is not None:
            return _as_float(match.group(1))
    return None


def _first_match(patterns: tuple[str, ...], line: str) -> Optional[float]:
    """Return the value captured by the first matching pattern, if any."""
    return _first_compiled(_compiled(patterns), line)


def _lines_after(lines: Sequence[str], start: int):
    """Yield the lines after ``start`` without copying the remaining list."""
    for index in range(start + 1, len(lines)):
        yield lines[index]


def _step_patterns(profile: Any, text: str) -> tuple[str, ...]:
    """Select the ionic-step dialect that a log actually uses.

    ABACUS branches number the ionic steps with ``STEP OF RELAXATION`` or with
    the ``ION=`` field of the electronic loop.  One ``re.search`` over the log
    text is enough to choose; without it every reader scans the whole log once
    more before parsing.
    """
    patterns = profile.relax_step_patterns
    if not patterns:
        return profile.ion_step_patterns
    gate = re.compile(
        "|".join(f"(?:{pattern})" for pattern in patterns),
        re.IGNORECASE | re.MULTILINE,
    )
    return patterns if gate.search(text) else profile.ion_step_patterns


def _force_extremes(
    block: List[List[float]],
) -> Optional[tuple[float, int, str]]:
    """Return the magnitude, one-based atom index and component of the largest force."""
    if not block:
        return None
    vectors = np.asarray(block, dtype=float)
    magnitudes = np.sqrt((vectors * vectors).sum(axis=1))
    index = int(magnitudes.argmax())
    axis = int(np.abs(vectors[index]).argmax())
    return float(magnitudes[index]), index + 1, "xyz"[axis]


def _atom_label(labels: List[str], atom: int) -> Optional[str]:
    """Return the log label of a one-based atom index, such as ``H1``."""
    if 0 < atom <= len(labels):
        label = labels[atom - 1].strip()
        if label:
            return label
    return None


def _stress_extremes(
    tensor: List[List[float]],
) -> Optional[tuple[float, str]]:
    """Return the magnitude and Voigt label of the largest stress component."""
    best: Optional[tuple[float, str]] = None
    for row_index, row in enumerate(tensor[:3]):
        for column_index, value in enumerate(row[:3]):
            magnitude = abs(float(value))
            if best is None or magnitude > best[0]:
                best = (magnitude, _STRESS_COMPONENTS[row_index][column_index])
    return best


def read_relaxation_history(
    log_file: Union[str, Path],
    version: Optional[str] = None,
    *,
    lattice: Any = None,
) -> List[Dict[str, Any]]:
    """Read per-ionic-step geometry-optimization metrics from an ABACUS log.

    Every record holds the step number, the total energy and its change, the
    largest force and stress, and whether the step met the convergence
    criteria.  When the log contains the force or stress table of a step, the
    record also names the atom, its log label such as ``H1`` and the Cartesian
    component of the largest force, plus the Voigt component of the largest
    stress.  Logs of older branches that mark the ionic step only through the
    ``ION=`` field of the electronic loop are read the same way.  The raw
    ``forces`` vectors and the ``stress`` tensor are kept as well, so a caller
    can count the components beyond a convergence threshold.  When the log
    prints Cartesian coordinates, ``rms_displacement`` and
    ``max_displacement`` compare each structure with the preceding one in
    Angstrom, and ``max_displacement_atom`` names the atom that moved the
    most.  Those metrics need the cell to unfold the periodic wrapping that
    ABACUS applies after every ionic step, so they are reported only when a
    pymatgen ``lattice`` is given.

    Args:
        log_file: ABACUS running log.
        version: ABACUS version hint, as in :func:`read_relaxation_history`.
        lattice: Pymatgen lattice of the job, used for the periodic
            displacements.  Pass the result of
            :func:`abacustools.io.stru.periodic_lattice`; without it the
            displacement metrics are omitted.
    """
    path = Path(log_file)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    profile = resolve_version(version, job_dir=path.parent, text="\n".join(lines[:128]))
    step_patterns = _step_patterns(profile, text)
    del text
    records, force_blocks, stress_blocks = _collect_ionic_steps(
        lines, profile, step_patterns
    )

    # Coordinates only feed the periodic displacement metrics; without a
    # lattice the blocks are never used, so do not pay for parsing them.
    initial_positions: Optional[List[List[float]]] = None
    positions: Dict[int, List[List[float]]] = {}
    if lattice is not None:
        initial_positions, positions = _collect_coordinate_steps(
            lines, profile, step_patterns
        )
    previous_positions = initial_positions
    for item in [records[step] for step in sorted(records)]:
        current_positions = positions.get(item["step"])
        if current_positions is None:
            previous_positions = None
            continue
        displacement = _displacement_metrics(
            previous_positions, current_positions, lattice
        )
        if displacement is not None:
            rms, maximum, atom = displacement
            item["rms_displacement"] = rms
            item["max_displacement"] = maximum
            item["max_displacement_atom"] = atom
        previous_positions = current_positions

    for step, (labels, block) in force_blocks.items():
        extremes = _force_extremes(block)
        if extremes is None:
            continue
        magnitude, atom, component = extremes
        item = records[step]
        if item["max_force"] is None:
            item["max_force"] = magnitude
        item["force_atom"] = atom
        item["force_component"] = component
        item["force_atom_label"] = _atom_label(labels, atom)
        displacement_atom = item.get("max_displacement_atom")
        if displacement_atom is not None:
            item["max_displacement_atom_label"] = _atom_label(labels, displacement_atom)
        item["forces"] = block

    for step, tensor in stress_blocks.items():
        extremes = _stress_extremes(tensor)
        if extremes is None:
            continue
        magnitude, component = extremes
        item = records[step]
        if item["max_stress"] is None:
            item["max_stress"] = magnitude
        item["stress_component"] = component
        item["stress"] = tensor

    history = [records[step] for step in sorted(records)]
    _fill_energy_change(history)
    return history


def _lattice_constant_angstrom(lines: Sequence[str]) -> Optional[float]:
    """Return the lattice constant in Angstrom when the log prints it."""
    bohr_value: Optional[float] = None
    for line in lines:
        match = _LATTICE_CONSTANT_ANGSTROM_RE.search(line)
        if match is not None:
            return _as_float(match.group(1))
        if bohr_value is None:
            match = _LATTICE_CONSTANT_BOHR_RE.search(line)
            if match is not None:
                bohr_value = _as_float(match.group(1))
    return None if bohr_value is None else bohr_value * BOHR_TO_ANG


def _parse_matrix_block(
    lines: Sequence[str], start: int, size: int = 3
) -> Optional[List[List[float]]]:
    """Read the ``size`` numeric rows that follow a matrix header."""
    rows: List[List[float]] = []
    for line in lines[start + 1 : start + 1 + size]:
        values = _numbers(line)
        if len(values) < size:
            return None
        rows.append(values[:size])
    return rows if len(rows) == size else None


def _parse_coordinate_rows(
    lines: Sequence[str], start: int
) -> tuple[List[Optional[str]], List[List[float]]]:
    """Read the labelled coordinate rows that follow a coordinate header."""
    labels: List[Optional[str]] = []
    positions: List[List[float]] = []
    started = False
    for line in _lines_after(lines, start):
        parts = line.split()
        values: Optional[List[float]] = None
        label: Optional[str] = None
        if len(parts) >= 4:
            try:
                values = [_as_float(value) for value in parts[1:4]]
                label = parts[0]
            except ValueError:
                values = None
        if values is None and len(parts) >= 3:
            try:
                values = [_as_float(value) for value in parts[:3]]
                label = None
            except ValueError:
                values = None
        if values is not None:
            labels.append(label)
            positions.append(values)
            started = True
            continue
        if started:
            break
    return labels, positions


def _symbols_from_labels(labels: Sequence[Optional[str]]) -> Optional[List[str]]:
    """Turn the atom labels of a coordinate table into element symbols."""
    if not labels or any(label is None for label in labels):
        return None
    symbols: List[str] = []
    for label in labels:
        cleaned = re.sub(r"^(?:tauc_|taud_)", "", str(label))
        match = re.match(r"[A-Za-z]{1,2}", cleaned)
        if match is None:
            return None
        symbols.append(match.group(0))
    return symbols


def read_relax_structures(
    log_file: Union[str, Path],
    version: Optional[str] = None,
    *,
    cell: Any = None,
    symbols: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Read the per-step structures printed in a geometry-optimization log.

    ABACUS prints the atomic coordinates of every ionic step to the running
    log, and the cell whenever it changes, so the trajectory of a ``relax`` or
    ``cell-relax`` job can be rebuilt from the log alone.  The coordinate block
    that precedes the ``STEP OF RELAXATION`` marker of a step holds the geometry
    evaluated at that step, so the step numbering matches
    :func:`read_relaxation_history`.

    Args:
        log_file: ABACUS running relaxation log.
        version: ABACUS version hint, as in :func:`read_relaxation_history`.
        cell: Initial cell in Angstrom, used when the log prints no lattice
            vectors; the rows are the three cell vectors.
        symbols: Element symbols used when the coordinate rows carry no label.

    Returns:
        One record per ionic step, ordered by step, with the step number, the
        positions in Angstrom, the element symbols, the cell in Angstrom, the
        per-atom forces in eV/Angstrom, the stress tensor in kBar as ABACUS
        prints it, and the total energy in eV; a field is ``None`` when the log
        does not provide it.
    """
    path = Path(log_file)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    profile = resolve_version(
        version, job_dir=path.parent, text="\n".join(lines[:128])
    )
    step_patterns = _step_patterns(profile, text)
    del text
    step_res = _compiled(step_patterns)
    gate = _gate(
        (),
        step_patterns
        + (
            _COORDINATE_HEADER_RE.pattern,
            _DIRECT_COORDINATE_HEADER_RE.pattern,
            _LATTICE_VECTOR_HEADER_RE.pattern,
        ),
    )
    lat0 = _lattice_constant_angstrom(lines)

    markers: List[tuple[int, int]] = []
    coordinates: List[tuple[int, str, List[Optional[str]], List[List[float]]]] = []
    lattices: List[tuple[int, List[List[float]]]] = []
    for index, line in enumerate(lines):
        if gate.search(line.lower()) is None:
            continue
        step = _first_compiled(step_res, line)
        if step is not None:
            markers.append((index, int(step)))
            continue
        if _LATTICE_VECTOR_HEADER_RE.search(line) is not None:
            block = _parse_matrix_block(lines, index)
            if block is not None and lat0 is not None:
                lattices.append(
                    (index, [[value * lat0 for value in row] for row in block])
                )
            continue
        if _DIRECT_COORDINATE_HEADER_RE.match(line) is not None:
            labels, positions = _parse_coordinate_rows(lines, index)
            if positions:
                coordinates.append((index, "direct", labels, positions))
            continue
        header = _COORDINATE_HEADER_RE.search(line)
        if header is None:
            continue
        unit = 1.0 if header.group(1) is None else _as_float(header.group(1))
        labels, positions = _parse_coordinate_rows(lines, index)
        if positions:
            coordinates.append(
                (
                    index,
                    "cartesian",
                    labels,
                    [
                        [value * unit * BOHR_TO_ANG for value in row]
                        for row in positions
                    ],
                )
            )

    if not markers:
        return []

    history = {
        int(item["step"]): item for item in read_relaxation_history(path, version)
    }
    fallback_cell = None if cell is None else [[float(value) for value in row] for row in cell]
    structures: List[Dict[str, Any]] = []
    for marker_index, step in markers:
        geometry = next(
            (item for item in reversed(coordinates) if item[0] < marker_index), None
        )
        if geometry is None:
            continue
        _, mode, labels, positions = geometry
        block_cell = fallback_cell
        for lattice_index, lattice in reversed(lattices):
            if lattice_index < marker_index:
                block_cell = lattice
                break
        if mode == "direct":
            if block_cell is None:
                continue
            matrix = np.asarray(block_cell, dtype=float)
            positions = (np.asarray(positions, dtype=float) @ matrix).tolist()
        resolved = _symbols_from_labels(labels)
        if resolved is None and symbols is not None:
            resolved = list(symbols)
        if resolved is None or len(resolved) != len(positions):
            continue
        item = history.get(step)
        structures.append(
            {
                "step": step,
                "cell": block_cell,
                "symbols": list(resolved),
                "positions": positions,
                "forces": None if item is None else item.get("forces"),
                "stress": None if item is None else item.get("stress"),
                "energy": None if item is None else item.get("energy"),
            }
        )
    return structures


def _collect_ionic_steps(
    lines: List[str],
    profile: Any,
    step_patterns: tuple[str, ...],
) -> tuple[
    Dict[int, Dict[str, Any]],
    Dict[int, tuple[List[str], List[List[float]]]],
    Dict[int, List[List[float]]],
]:
    """Group the lines of one relaxation log into ionic steps."""
    records: Dict[int, Dict[str, Any]] = {}
    force_blocks: Dict[int, tuple[List[str], List[List[float]]]] = {}
    stress_blocks: Dict[int, List[List[float]]] = {}
    scf_energies: Dict[int, float] = {}
    current_step: Optional[int] = None

    def record(step: int) -> Dict[str, Any]:
        return records.setdefault(
            step,
            {
                "step": step,
                "energy": None,
                "energy_change": None,
                "max_force": None,
                "force_atom": None,
                "force_component": None,
                "force_atom_label": None,
                "forces": None,
                "max_stress": None,
                "stress_component": None,
                "stress": None,
                "rms_displacement": None,
                "max_displacement": None,
                "max_displacement_atom": None,
                "max_displacement_atom_label": None,
                "converged": False,
            },
        )

    step_res = _compiled(step_patterns)
    energy_res = _compiled(profile.relax_energy_patterns)
    force_res = _compiled(profile.relax_force_patterns)
    stress_res = _compiled(profile.relax_stress_patterns)
    diff_res = _compiled((_RELAX_ENERGY_DIFF_RE.pattern,))
    gate = _gate(
        profile.energy_keywords
        + profile.relax_converged_keywords
        + profile.force_header_keywords
        + profile.stress_header_keywords,
        step_patterns
        + profile.relax_energy_patterns
        + profile.relax_force_patterns
        + profile.relax_stress_patterns
        + (_RELAX_ENERGY_DIFF_RE.pattern,),
    )

    for index, line in enumerate(lines):
        low = line.lower()
        if gate.search(low) is None:
            continue
        step = _first_compiled(step_res, line)
        if step is not None:
            current_step = int(step)
            record(current_step)
            continue
        if current_step is None:
            continue
        item = record(current_step)
        if _contains_any(low, profile.energy_keywords):
            values = _numbers(line)
            if values:
                scf_energies[current_step] = values[-1]
        energy = _first_compiled(energy_res, line)
        if energy is not None:
            item["energy"] = energy
        energy_change = _first_compiled(diff_res, line)
        if energy_change is not None:
            item["energy_change"] = energy_change
        force = _first_compiled(force_res, line)
        if force is not None:
            item["max_force"] = force
        stress = _first_compiled(stress_res, line)
        if stress is not None:
            item["max_stress"] = stress
        if _contains_any(low, profile.force_header_keywords):
            labels, block = _parse_force_block(lines, index)
            if block:
                force_blocks[current_step] = (labels, block)
        elif _contains_any(low, profile.stress_header_keywords):
            tensor = _parse_stress_block(lines, index)
            if tensor:
                stress_blocks[current_step] = tensor
        if _contains_any(low, profile.relax_converged_keywords):
            item["converged"] = True

    # The ``final etot is`` report line is rounded, while the Kohn-Sham energy
    # of the last electronic iteration carries the full precision of the step.
    for step, energy in scf_energies.items():
        records[step]["energy"] = energy

    return records, force_blocks, stress_blocks


def _parse_coordinate_block(
    lines: List[str], start: int, unit_to_angstrom: float
) -> List[List[float]]:
    """Read Cartesian coordinates following one log coordinate header."""
    coordinates: List[List[float]] = []
    started = False
    for line in _lines_after(lines, start):
        parts = line.split()
        values: Optional[List[float]] = None
        if len(parts) >= 4:
            try:
                values = [_as_float(value) for value in parts[1:4]]
            except ValueError:
                pass
        if values is None and len(parts) >= 3:
            try:
                values = [_as_float(value) for value in parts[:3]]
            except ValueError:
                pass
        if values is not None:
            coordinates.append([value * unit_to_angstrom for value in values])
            started = True
            continue
        if started:
            break
    return coordinates


def _collect_coordinate_steps(
    lines: List[str],
    profile: Any,
    step_patterns: tuple[str, ...],
) -> tuple[Optional[List[List[float]]], Dict[int, List[List[float]]]]:
    """Collect the initial and per-step Cartesian structures from a log."""
    step_res = _compiled(step_patterns)
    gate = _gate(
        (),
        step_patterns + (_COORDINATE_HEADER_RE.pattern,),
    )
    current_step: Optional[int] = None
    initial_positions: Optional[List[List[float]]] = None
    positions: Dict[int, List[List[float]]] = {}

    for index, line in enumerate(lines):
        if gate.search(line.lower()) is None:
            continue
        step = _first_compiled(step_res, line)
        if step is not None:
            current_step = int(step)
            continue
        header = _COORDINATE_HEADER_RE.search(line)
        if header is None:
            continue
        unit = 1.0 if header.group(1) is None else _as_float(header.group(1))
        coordinates = _parse_coordinate_block(
            lines, index, unit * BOHR_TO_ANG
        )
        if not coordinates:
            continue
        if current_step is None:
            initial_positions = coordinates
        else:
            positions[current_step] = coordinates
    return initial_positions, positions


def _displacement_metrics(
    previous: Optional[List[List[float]]],
    current: List[List[float]],
    lattice: Any = None,
) -> Optional[tuple[float, float, int]]:
    """Return the atomic RMS and maximum displacement between structures.

    ABACUS wraps fractional coordinates into the cell after every ionic step,
    so an atom that crosses a cell boundary jumps by almost a full lattice
    vector in the log.  The per-atom distances therefore have to be taken as
    periodic minimum images, which needs the cell; without a ``lattice`` the
    displacements are left unreported instead of reporting a wrong jump.
    """
    if previous is None or len(previous) != len(current) or not current:
        return None
    if lattice is None:
        return None
    previous_cartesian = np.asarray(previous, dtype=float)
    current_cartesian = np.asarray(current, dtype=float)
    previous_fractional = lattice.get_fractional_coords(previous_cartesian)
    current_fractional = lattice.get_fractional_coords(current_cartesian)
    distances = np.array(
        [
            lattice.get_distance_and_image(first, second)[0]
            for first, second in zip(previous_fractional, current_fractional)
        ]
    )
    return (
        float(np.sqrt(np.mean(distances**2))),
        float(np.max(distances)),
        int(np.argmax(distances)) + 1,  # one-based, as in _force_extremes
    )


def _fill_energy_change(history: List[Dict[str, Any]]) -> None:
    """Fill missing energy changes from consecutive total energies."""
    previous_energy = None
    for item in history:
        if (
            item["energy_change"] is None
            and item["energy"] is not None
            and previous_energy is not None
        ):
            item["energy_change"] = item["energy"] - previous_energy
        if item["energy"] is not None:
            previous_energy = item["energy"]


def read_scf_history(
    log_file: Union[str, Path],
    version: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read the per-iteration metrics of an electronic (SCF) loop.

    Args:
        log_file: ABACUS running log.
        version: ABACUS version hint, as in :func:`read_relaxation_history`.

    Returns:
        One record per electronic iteration with the iteration number, its
        Kohn-Sham energy, the energy change to the previous iteration and the
        density error.
    """
    path = Path(log_file)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    profile = resolve_version(version, job_dir=path.parent, text="\n".join(lines[:128]))
    records: Dict[int, Dict[str, Any]] = {}
    current_step: Optional[int] = None

    def record(step: int) -> Dict[str, Any]:
        return records.setdefault(
            step,
            {"step": step, "energy": None, "energy_change": None, "drho": None},
        )

    step_res = _compiled(profile.scf_step_patterns)
    gate = _gate(
        profile.energy_keywords + profile.density_error_keywords,
        profile.scf_step_patterns,
    )
    for line in lines:
        lower_line = line.lower()
        if gate.search(lower_line) is None:
            continue
        step = _first_compiled(step_res, line)
        if step is not None:
            current_step = int(step)
            record(current_step)
            continue
        if current_step is None:
            continue
        item = record(current_step)
        if _contains_any(lower_line, profile.energy_keywords):
            values = _numbers(line)
            if values:
                item["energy"] = values[-1]
        elif _contains_any(lower_line, profile.density_error_keywords):
            values = _numbers(line)
            if values:
                item["drho"] = values[-1]

    history = [records[step] for step in sorted(records)]
    _fill_energy_change(history)
    return history


def read_md_history(
    log_file: Union[str, Path],
    version: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read the per-step metrics of a molecular-dynamics run.

    ABACUS prints the total, potential and kinetic energy, the temperature and
    the pressure of every MD step in one table; the records hold one entry per
    MD step, starting at step 0.  The table reports its energies in eV or in
    Rydberg and may omit the pressure column, so the energies are converted to
    eV and the pressure is left empty when it is not printed.

    Args:
        log_file: ABACUS running log of a ``calculation md`` job.
        version: ABACUS version hint, as in :func:`read_relaxation_history`.

    Returns:
        One record per MD step with the step number, the energies in eV, the
        temperature in K and the pressure in kBar.
    """
    path = Path(log_file)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    profile = resolve_version(version, job_dir=path.parent, text="\n".join(lines[:128]))
    records: Dict[int, Dict[str, Any]] = {}
    current_step: Optional[int] = None

    def record(step: int) -> Dict[str, Any]:
        return records.setdefault(
            step,
            {
                "step": step,
                "energy": None,
                "potential": None,
                "kinetic": None,
                "temperature": None,
                "pressure": None,
            },
        )

    step_res = _compiled(profile.md_step_patterns)
    gate = _gate(
        ("total-pressure", "temperature"),
        profile.md_step_patterns,
    )
    for index, line in enumerate(lines):
        lower_line = line.lower()
        if gate.search(lower_line) is None:
            continue
        step = _first_compiled(step_res, line)
        if step is not None:
            current_step = int(step)
            record(current_step)
            continue
        if current_step is None:
            continue
        if "total-pressure" in lower_line:
            values = _numbers(line)
            if values:
                record(current_step)["pressure"] = values[-1]
            continue
        if not _is_md_table_header(line):
            continue
        energy_factor = RY_TO_EV if "(ry)" in lower_line else 1.0
        has_pressure = "pressure" in lower_line
        for following in _lines_after(lines, index):
            values = _numbers(following)
            if not values:
                continue
            if len(values) >= (5 if has_pressure else 4):
                item = record(current_step)
                item["energy"] = values[0] * energy_factor
                item["potential"] = values[1] * energy_factor
                item["kinetic"] = values[2] * energy_factor
                item["temperature"] = values[3]
                if has_pressure:
                    item["pressure"] = values[4]
            break

    return [records[step] for step in sorted(records)]


def _is_md_table_header(line: str) -> bool:
    """Return whether a log line is the header of the MD energy table."""
    lower_line = line.lower()
    return all(
        column in lower_line
        for column in ("energy", "potential", "kinetic", "temperature")
    )


def read_convergence_thresholds(
    log_file: Union[str, Path],
    version: Optional[str] = None,
) -> Dict[str, float]:
    """Read the force and stress thresholds printed by a relaxation log.

    ABACUS prints the effective criteria along the largest force and stress,
    which also covers the runs whose INPUT leaves them at their defaults.
    """
    path = Path(log_file)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    profile = resolve_version(version, job_dir=path.parent, text="\n".join(lines[:128]))
    force_res = _compiled(profile.relax_force_threshold_patterns)
    stress_res = _compiled(profile.relax_stress_threshold_patterns)
    gate = _gate(
        (),
        profile.relax_force_threshold_patterns + profile.relax_stress_threshold_patterns,
    )
    thresholds: Dict[str, float] = {}
    for line in lines:
        if gate.search(line.lower()) is None:
            continue
        if "force_thr_ev" not in thresholds:
            value = _first_compiled(force_res, line)
            if value is not None:
                thresholds["force_thr_ev"] = value
        if "stress_thr" not in thresholds:
            value = _first_compiled(stress_res, line)
            if value is not None:
                thresholds["stress_thr"] = value
        if len(thresholds) == 2:
            break
    return thresholds


def _read_tail_lines(
    path: Path, *, max_bytes: int = 262144, max_lines: int = 80
) -> List[str]:
    """Read the last lines of a log without loading the whole file."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="replace").splitlines()[-max_lines:]


def read_normal_end(
    log_file: Union[str, Path],
    version: Optional[str] = None,
) -> bool:
    """Return whether a running log reached the normal-ending footer."""
    path = Path(log_file)
    lines = _read_tail_lines(path)
    with path.open(encoding="utf-8", errors="replace") as handle:
        head = handle.read(8192)
    profile = resolve_version(version, job_dir=path.parent, text=head)
    return any(
        _contains_any(line.lower(), profile.normal_end_keywords)
        for line in lines
    )


def _is_separator(line: str) -> bool:
    """Return whether a log line is made only of separator characters."""
    stripped = line.strip()
    return bool(stripped) and not stripped.strip("-")


def _job_input(job_path: Path) -> Dict[str, Any]:
    """Read the INPUT file from a job directory."""
    input_file = job_path / "INPUT"
    if not input_file.is_file():
        raise FileNotFoundError(f"Could not find INPUT in ABACUS job {job_path}")
    return ReadInput(input_file)


def _output_directory(job_path: Path, inputs: Dict[str, Any]) -> Path:
    """Find the output directory, including jobs without suffix in INPUT."""
    output_directories = sorted(
        path for path in job_path.glob("OUT.*") if path.is_dir()
    )
    suffix = inputs.get("suffix")
    if suffix:
        expected = job_path / f"OUT.{suffix}"
        if expected.is_dir():
            return expected
    if len(output_directories) == 1:
        return output_directories[0]
    if not output_directories:
        raise FileNotFoundError(f"Could not find an OUT.* directory in {job_path}")
    raise FileNotFoundError(
        f"Could not determine the output directory for ABACUS job {job_path}"
    )


def read_dos_from_job(job_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read total DOS data from an ABACUS job output.

    Reads the ``DOS<n>_smearing.dat`` files of the LTS branches and, when
    they are absent, the ``doss<spin>[g<step>][_nao].txt`` files of the
    develop branch. For a geometry run that writes one file per ionic
    step, the data of the last ionic step is returned.

    Args:
        job_dir: ABACUS job directory containing ``INPUT`` and ``OUT.*``.

    Returns:
        The shared energy grid in eV and the DOS of every spin channel as
        columns of one array, shape ``(nenergy, nspin)``.
    """
    job_path = Path(job_dir)
    inputs = _job_input(job_path)
    output_path = _output_directory(job_path, inputs)
    dos_files = sorted(output_path.glob("DOS*_smearing.dat"))
    if not dos_files:
        return _read_dos_nao_files(output_path)
    return _read_dos_smearing_files(dos_files)


def _read_dos_grid(
    files: List[Path], read_one: Callable[[Path], np.ndarray]
) -> Dict[str, Any]:
    """Stack the ``dos`` column of every spin-channel file on one grid."""
    energy = None
    dos_channels = []
    for dos_file in files:
        data = read_one(dos_file)
        if data.shape[1] < 2:
            raise ValueError(f"DOS file has fewer than two columns: {dos_file}")
        file_energy = data[:, 0]
        if energy is None:
            energy = file_energy
        elif energy.shape != file_energy.shape or not np.allclose(energy, file_energy):
            raise ValueError(f"DOS energy grids do not match: {dos_file}")
        dos_channels.append(data[:, 1])

    return {
        "energy": energy,
        "data": np.column_stack(dos_channels),
    }


def _read_dos_smearing_files(dos_files: List[Path]) -> Dict[str, Any]:
    """Read the LTS-branch ``DOS<n>_smearing.dat`` files."""
    return _read_dos_grid(dos_files, lambda dos_file: np.loadtxt(dos_file, ndmin=2))


def _select_latest_spin_files(output_path: Path, pattern: re.Pattern) -> Dict[int, Path]:
    """Pick the newest ionic-step file of every spin channel."""
    spin_files: Dict[int, tuple[int, Path]] = {}
    for path in sorted(output_path.iterdir()):
        match = pattern.match(path.name)
        if match is None or not path.is_file():
            continue
        spin = int(match.group(1) or 1)
        step = int(match.group(2) or 0)
        if spin not in spin_files or step >= spin_files[spin][0]:
            spin_files[spin] = (step, path)
    return spin_files


def _read_dos_nao_file(dos_file: Path) -> np.ndarray:
    """Read the last ionic-step block of one develop-branch DOS file.

    A file starts with ``<istep> # ionic step``, a point count and a
    column header, and holds the rows ``(energy, dos, dos_int,
    dos_smear, dos_smear_int)``; a run with ``out_app_flag 1`` appends
    one such block per ionic step, so the last block is the latest one.
    """
    blocks: Dict[int, List[List[float]]] = {}
    step: Optional[int] = None
    for line in dos_file.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = _IONIC_STEP_HEADER_RE.match(line)
        if step_match is not None:
            step = int(step_match.group(1))
            continue
        if step is None or line.lstrip().startswith("#"):
            continue
        try:
            values = [_as_float(field) for field in line.split()]
        except ValueError:
            continue
        blocks.setdefault(step, []).append(values)
    if not blocks:
        raise ValueError(f"DOS file has no data rows: {dos_file}")
    data = np.asarray(blocks[max(blocks)], dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"DOS file has fewer than two columns: {dos_file}")
    return data


def _read_dos_nao_files(output_path: Path) -> Dict[str, Any]:
    """Read the develop-branch ``doss<spin>...txt`` total-DOS files."""
    spin_files = _select_latest_spin_files(output_path, _DOS_NAO_FILE_RE)
    if not spin_files:
        raise FileNotFoundError(f"Could not find DOS files in {output_path}")
    return _read_dos_grid(
        [spin_files[spin][1] for spin in sorted(spin_files)],
        _read_dos_nao_file,
    )


def read_orbital_xml(xml_file: Union[str, Path]) -> Dict[str, Any]:
    """Read orbital metadata and data from an ABACUS orbital XML file."""
    xml_path = Path(xml_file)
    root = ET.parse(xml_path).getroot()

    energy = None
    energy_node = root.find("energy_values")
    if energy_node is not None and energy_node.text:
        energy = np.asarray(
            [_as_float(value) for value in energy_node.text.split()]
        )

    orbitals = []
    for orbital in root.findall("orbital"):
        data_node = orbital.find("data")
        if data_node is None or not data_node.text:
            raise ValueError(f"Orbital has no data: {xml_path}")
        data = np.loadtxt(StringIO(data_node.text), ndmin=2)
        try:
            orbital_info = {
                "index": int(orbital.get("index")),
                "atom_index": int(orbital.get("atom_index")),
                "species": orbital.get("species"),
                "l": int(orbital.get("l")),
                "m": int(orbital.get("m")),
                "z": int(orbital.get("z")),
                "data": data,
            }
        except (TypeError, ValueError) as error:
            raise ValueError(f"Orbital metadata is invalid: {xml_path}") from error
        orbitals.append(orbital_info)

    return {"energy": energy, "orbitals": orbitals}


def _pdos_nao_channels(ncolumns: int) -> List[Tuple[int, int]]:
    """Return the ``(l, m)`` of every PDOS text column of one atom.

    ABACUS writes one column per ``(l, m)`` of the basis, shells in the
    order s, p, d, f and ``m`` as 0, +1, -1, +2, -2 within a shell, each
    column summed over the zeta radii; ``m`` therefore follows the same
    convention as the ``m`` attribute of the XML PDOS file.
    """
    channels: List[Tuple[int, int]] = []
    remaining = ncolumns
    shell = 0
    while remaining > 0:
        width = 2 * shell + 1
        if remaining < width:
            raise ValueError(
                f"PDOS column count {ncolumns} does not fill complete shells"
            )
        channels.extend((shell, m) for m in range(width))
        remaining -= width
        shell += 1
    return channels


def _read_pdos_nao_file(pdos_file: Path) -> Dict[str, Any]:
    """Read one develop-branch ``pdoss<spin>...txt`` projected-DOS file.

    After three comment lines, the file holds one row
    ``(energy, atom, species, pdos columns...)`` per atom and energy
    point; every atom occupies ``npoints`` consecutive rows and carries
    its own number of PDOS columns.
    """
    energy: List[float] = []
    atom_order: List[int] = []
    species_by_atom: Dict[int, str] = {}
    values_by_atom: Dict[int, List[List[float]]] = {}
    for line in pdos_file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 4:
            continue
        try:
            row_energy = _as_float(fields[0])
            atom_index = int(fields[1])
            values = [_as_float(field) for field in fields[3:]]
        except ValueError:
            continue
        if atom_index not in values_by_atom:
            values_by_atom[atom_index] = []
            species_by_atom[atom_index] = fields[2]
            atom_order.append(atom_index)
        if atom_index == atom_order[0]:
            energy.append(row_energy)
        values_by_atom[atom_index].append(values)
    if not atom_order:
        raise ValueError(f"PDOS file has no data rows: {pdos_file}")
    return {
        "energy": np.asarray(energy, dtype=float),
        "atom_order": atom_order,
        "species": species_by_atom,
        "values": {
            atom_index: np.asarray(rows, dtype=float)
            for atom_index, rows in values_by_atom.items()
        },
    }


def _read_pdos_nao_files(output_path: Path) -> Dict[str, Any]:
    """Read the develop-branch ``pdoss<spin>...txt`` projected-DOS files.

    The ``nspin 2`` spin channels are written as two files; their columns
    are combined into one up/down column pair per ``(l, m)`` orbital, as
    the XML PDOS file of the LTS branches reports it.
    """
    spin_files = _select_latest_spin_files(output_path, _PDOS_NAO_FILE_RE)
    if not spin_files:
        raise FileNotFoundError(f"Could not find PDOS files in {output_path}")

    parsed = {spin: _read_pdos_nao_file(spin_files[spin][1]) for spin in spin_files}
    first = parsed[min(parsed)]
    for spin in sorted(parsed)[1:]:
        other = parsed[spin]
        other_file = spin_files[spin][1]
        if other["atom_order"] != first["atom_order"]:
            raise ValueError(f"PDOS spin files list different atoms: {other_file}")
        if other["energy"].shape != first["energy"].shape or not np.allclose(
            other["energy"], first["energy"]
        ):
            raise ValueError(f"PDOS energy grids do not match: {other_file}")

    orbitals: List[Dict[str, Any]] = []
    for atom_index in first["atom_order"]:
        atom_columns = [parsed[spin]["values"][atom_index] for spin in sorted(parsed)]
        species = first["species"][atom_index]
        for column, (l, m) in enumerate(_pdos_nao_channels(atom_columns[0].shape[1])):
            orbitals.append(
                {
                    "index": len(orbitals) + 1,
                    "atom_index": atom_index,
                    "species": species,
                    "l": l,
                    "m": m,
                    "z": 1,  # the text columns are summed over the zeta radii
                    "data": np.stack(
                        [column_data[:, column] for column_data in atom_columns],
                        axis=1,
                    ),
                }
            )
    return {"energy": first["energy"], "orbitals": orbitals}


def read_pdos_from_job(job_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read projected DOS data from an ABACUS job output.

    Reads the XML ``PDOS`` file of the LTS branches and, when it is
    absent, the ``pdoss<spin>[g<step>]_<basis>.txt`` text files of the
    develop branch; for a geometry run that writes one file per ionic
    step, the data of the last ionic step is returned.

    Args:
        job_dir: ABACUS job directory containing ``INPUT`` and ``OUT.*``.

    Returns:
        The shared energy grid in eV and one record per ``(l, m)`` orbital
        holding its atom, species and the PDOS of every spin channel.
    """
    job_path = Path(job_dir)
    inputs = _job_input(job_path)
    output_path = _output_directory(job_path, inputs)
    pdos_file = output_path / "PDOS"
    if pdos_file.is_file():
        result = read_orbital_xml(pdos_file)
    else:
        result = _read_pdos_nao_files(output_path)

    energy = result["energy"]
    if energy is None:
        raise ValueError("PDOS data has no energy grid")

    for orbital in result["orbitals"]:
        data = orbital["data"]
        if data.shape[0] != energy.shape[0]:
            raise ValueError(
                f"PDOS energy and data lengths do not match for orbital "
                f"{orbital.get('index')} of job {job_path}"
            )

    return result


def _calculation(inputs: Dict[str, Any], output_path: Path) -> str:
    """Get calculation type, preferring the effective INPUT when available.

    The effective-input copy is read in both spellings: ``INPUT`` on the
    LTS release and ``INPUT.info`` on develop.
    """
    for name in ("INPUT", "INPUT.info"):
        output_input = output_path / name
        if output_input.is_file():
            inputs = {**inputs, **ReadInput(output_input)}
            break
    return str(inputs.get("calculation", "scf")).lower()


def _machine_efermi(job_path: Path, log_value: Optional[float]) -> Optional[float]:
    """Prefer the machine-readable ``e_fermi`` of ``abacus.json``.

    Args:
        job_path: Job directory holding (or not) ``abacus.json``.
        log_value: The value parsed from the ``E_Fermi`` log keywords.

    Returns:
        The Fermi energy of the latest ionic step when the job summary
        provides one, else the log-parsed value.
    """
    summary = read_abacus_json(job_path)
    if summary is None:
        return log_value
    entry = latest_ionic_output(summary)
    if isinstance(entry, dict) and isinstance(entry.get("e_fermi"), (int, float)):
        return float(entry["e_fermi"])
    return log_value


def _log_file(
    output_path: Path,
    calculation: str,
    *,
    relax: bool = False,
) -> Path:
    """Find the ABACUS log matching an SCF or ionic calculation."""
    if relax:
        names = [
            f"running_{calculation}.log",
            "running_relax.log",
            "running_cell-relax.log",
        ]
    else:
        names = [
            "running_scf.log",
            f"running_{calculation}.log",
        ]
    for name in dict.fromkeys(names):
        path = output_path / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find an ABACUS log in {output_path}; tried {', '.join(names)}"
    )


def find_job_log(
    job_dir: str,
    *,
    ionic: bool = False,
) -> Optional[Path]:
    """Return the running log that holds the results of one job.

    The log is selected the same way the result collectors select it, so that
    monitoring and ``postprocess result`` always read the same file.

    Args:
        job_dir: ABACUS job directory containing ``INPUT`` and ``OUT.*``.
        ionic: Prefer the log of an ionic (relax/cell-relax/md) calculation.

    Returns:
        The running log, or ``None`` when the job has no readable output yet.
    """
    job_path = Path(job_dir)
    try:
        inputs = _job_input(job_path)
        output_path = _output_directory(job_path, inputs)
        calculation = _calculation(inputs, output_path)
        return _log_file(output_path, calculation, relax=ionic)
    except FileNotFoundError:
        return None


def split_param_by_group(param_names: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    """Split result names into their parser groups and reject unknown names."""
    if param_names is None:
        return {}

    param_groups: Dict[str, List[str]] = {}
    known_params = {
        parameter: group
        for group, parameters in grouped_params.items()
        for parameter in parameters
    }
    for name in param_names:
        parameter = name.lower()
        if parameter not in known_params:
            raise ValueError(f"Unknown result parameter: {name}")
        group = known_params[parameter]
        param_groups.setdefault(group, [])
        if parameter not in param_groups[group]:
            param_groups[group].append(parameter)
    return param_groups


def _empty_results(metrics: Sequence[str]) -> Dict[str, Any]:
    """Return empty values for metrics whose output is not available yet."""
    return {metric: None for metric in metrics}


def _parse_force_block(
    lines: List[str], start: int
) -> tuple[List[str], List[List[float]]]:
    """Read the atom labels and force vectors of one ``TOTAL-FORCE`` block."""
    labels: List[str] = []
    force: List[List[float]] = []
    for line in _lines_after(lines, start):
        if _is_separator(line):
            if force:
                return labels, force
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                force.append([_as_float(value) for value in parts[1:4]])
            except ValueError:
                if force:
                    return labels, force
            else:
                labels.append(parts[0])
    return labels, force


def _parse_stress_block(lines: List[str], start: int) -> List[List[float]]:
    stress: List[List[float]] = []
    for line in _lines_after(lines, start):
        if _is_separator(line):
            if stress:
                break
            continue
        values = _numbers(line)
        if len(values) >= 3:
            stress.append(values[-3:])
            if len(stress) == 3:
                break
    return stress


def collect_scf_results(
    job_dir: str,
    metrics: List[str],
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect SCF results from the final electronic iteration."""
    param_groups = split_param_by_group(metrics)
    if "scf_results" not in param_groups:
        return {}

    job_path = Path(job_dir)
    try:
        inputs = _job_input(job_path)
        output_path = _output_directory(job_path, inputs)
        calculation = _calculation(inputs, output_path)
        log_path = _log_file(output_path, calculation)
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return _empty_results(param_groups["scf_results"])

    profile = resolve_version(version, job_dir=job_path)
    last_line = next((line.strip() for line in reversed(lines) if line.strip()), None)
    normal_end = (
        None
        if last_line is None
        else _contains_any(last_line.lower(), profile.normal_end_keywords)
    )

    requested = set(param_groups["scf_results"])
    want_energy = bool({"energy", "denergy", "scf_steps"} & requested)
    want_final = "energy" in requested
    want_drho = "drho" in requested
    want_efermi = "efermi" in requested
    want_converged = "converged" in requested
    want_force = "force" in requested
    want_stress = "stress" in requested

    # Only the marker families a caller asked for take part in the scan; a
    # status update, for example, must not pay for the force and stress tables.
    literals: tuple[str, ...] = ()
    if want_energy:
        literals += profile.energy_keywords
    if want_final:
        literals += profile.final_energy_keywords
    if want_drho:
        literals += profile.density_error_keywords
    if want_efermi:
        literals += profile.fermi_keywords
    if want_converged:
        literals += profile.scf_converged_keywords
    if want_force:
        literals += profile.force_header_keywords
    if want_stress:
        literals += profile.stress_header_keywords
    gate = _gate(literals)

    energies: List[float] = []
    drhos: List[float] = []
    efermis: List[float] = []
    final_energies: List[float] = []
    forces: List[List[List[float]]] = []
    stresses: List[List[List[float]]] = []
    converged = False

    for line_number, line in enumerate(lines):
        lower_line = line.lower()
        if gate.search(lower_line) is None:
            continue
        if want_energy and _contains_any(lower_line, profile.energy_keywords):
            values = _numbers(line)
            if values:
                energies.append(values[-1])
        elif want_drho and _contains_any(lower_line, profile.density_error_keywords):
            values = _numbers(line)
            if values:
                drhos.append(values[-1])
        elif want_final and _contains_any(lower_line, profile.final_energy_keywords):
            values = _numbers(line)
            if values:
                final_energies.append(values[-1])
        elif want_efermi and _contains_any(lower_line, profile.fermi_keywords):
            values = _numbers(line)
            if values:
                efermis.append(values[-1])
        elif want_converged and _contains_any(
            lower_line, profile.scf_converged_keywords
        ):
            converged = True
        elif want_force and _contains_any(lower_line, profile.force_header_keywords):
            _, force = _parse_force_block(lines, line_number)
            if force:
                forces.append(force)
        elif want_stress and _contains_any(lower_line, profile.stress_header_keywords):
            stress = _parse_stress_block(lines, line_number)
            if stress:
                stresses.append(stress)

    energy_values = (final_energies or energies) if want_final else energies
    available = {
        "energy": energy_values[-1] if energy_values else None,
        "drho": drhos[-1] if drhos else None,
        "denergy": energies[-1] - energies[-2] if len(energies) > 1 else None,
        "scf_steps": len(energies),
        "converged": converged,
        "normal_end": normal_end,
        "efermi": (
            _machine_efermi(job_path, efermis[-1] if efermis else None)
            if want_efermi
            else None
        ),
        "force": forces[-1] if forces else None,
        "stress": stresses[-1] if stresses else None,
    }
    return {metric: available[metric] for metric in param_groups["scf_results"]}


def collect_vdw_results(
    job_dir: str,
    metrics: List[str],
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect the final DFT-D dispersion energy in eV."""
    param_groups = split_param_by_group(metrics)
    if "vdw_results" not in param_groups:
        return {}

    job_path = Path(job_dir)
    try:
        inputs = _job_input(job_path)
        output_path = _output_directory(job_path, inputs)
        calculation = _calculation(inputs, output_path)
        log_path = _log_file(output_path, calculation)
    except FileNotFoundError:
        return _empty_results(param_groups["vdw_results"])

    profile = resolve_version(version, job_dir=job_path)
    gate = _gate(profile.vdw_keywords)
    vdw_energies: List[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        lower_line = line.lower()
        if gate.search(lower_line) is None:
            continue
        if _contains_any(lower_line, profile.vdw_keywords):
            values = _numbers(line)
            if values:
                # ABACUS prints the dispersion energy in Rydberg and eV.
                vdw_energies.append(values[-1])

    available = {"vdw_energy": vdw_energies[-1] if vdw_energies else None}
    return {metric: available[metric] for metric in param_groups["vdw_results"]}


def _mulliken_magnetization(line: str) -> Optional[Union[float, List[float]]]:
    """Read the magnetization of one ``Total Magnetism on atom`` line.

    Noncollinear runs print the three components inside parentheses and
    collinear runs print a single number as the last field.
    """
    if "(" in line and ")" in line:
        inside = line[line.index("(") + 1 : line.index(")")]
        components = [_as_float(item) for item in inside.split(",") if item.strip()]
        if len(components) == 3:
            return components
    fields = line.split()
    if not fields:
        return None
    try:
        return _as_float(fields[-1])
    except ValueError:
        numbers = _numbers(line)
        return numbers[-1] if numbers else None


def read_mulliken_magnetization(
    mulliken_file: Union[str, Path],
) -> List[List[Union[float, List[float]]]]:
    """Read the per-atom magnetization of every Mulliken step.

    The Mulliken analysis writes one ``STEP:`` block per ionic step and one
    ``Total Magnetism on atom`` line per atom.  The result holds one entry per
    ionic step, each listing the atoms in file order; a noncollinear run
    stores the three Cartesian components of an atom as a list.

    Args:
        mulliken_file: ``mulliken.txt`` written below ``OUT.*``.

    Returns:
        One per-atom magnetization list per ionic step, empty when the file
        holds no magnetization.
    """
    text = Path(mulliken_file).read_text(encoding="utf-8", errors="replace")
    steps: List[List[Union[float, List[float]]]] = []
    current: List[Union[float, List[float]]] = []
    for line in text.splitlines():
        if line.startswith("STEP:"):
            if current:
                steps.append(current)
            current = []
            continue
        if "total magnetism on atom" not in line.lower():
            continue
        magnetization = _mulliken_magnetization(line)
        if magnetization is not None:
            current.append(magnetization)
    if current:
        steps.append(current)
    return steps


def read_orbital_magnetization(
    lines: Sequence[str],
    header_keywords: Sequence[str] = ("orbital charge analysis",),
) -> Optional[List[List[float]]]:
    """Read the per-atom magnetization of the last orbital charge analysis.

    ABACUS prints the orbital-projected charge and magnetization as numbered
    blocks that end with a ``Sum`` row per atom::

        Orbital Charge Analysis      Charge         Mag(x)         Mag(y)         Mag(z)
        Fe1
                           s         1.0799        -0.0000         0.0000         0.0034
                         Sum        13.6214        -0.0039         0.0013         3.0729

    Args:
        lines: Running-log lines.
        header_keywords: Markers introducing one orbital charge analysis.

    Returns:
        The Cartesian magnetization of every atom in the last block, or
        ``None`` when the log holds no such block.
    """
    blocks: List[List[List[float]]] = []
    gate = _gate(tuple(header_keywords))
    for index, line in enumerate(lines):
        lower_line = line.lower()
        if gate.search(lower_line) is None:
            continue
        if not _contains_any(lower_line, header_keywords):
            continue
        block: List[List[float]] = []
        for row in _lines_after(lines, index):
            if _is_separator(row):
                if block:
                    break
                continue
            fields = row.split()
            if not fields:
                continue
            if fields[0].lower() != "sum":
                continue
            try:
                values = [_as_float(value) for value in fields[2:]]
            except ValueError:
                continue
            if len(values) >= 3:
                block.append(values[-3:])
        if block:
            blocks.append(block)
    return blocks[-1] if blocks else None


def collect_magnetic_results(
    job_dir: str,
    metrics: List[str],
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect magnetic moments from the running log and Mulliken analysis."""
    param_groups = split_param_by_group(metrics)
    if "magnetic_results" not in param_groups:
        return {}
    requested = param_groups["magnetic_results"]

    job_path = Path(job_dir)
    available: Dict[str, Any] = _empty_results(requested)
    try:
        inputs = _job_input(job_path)
        output_path = _output_directory(job_path, inputs)
        calculation = _calculation(inputs, output_path)
        log_path = _log_file(output_path, calculation)
    except FileNotFoundError:
        return available

    profile = resolve_version(version, job_dir=job_path)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    want_total = "total_mag" in requested
    want_absolute = "absolute_mag" in requested
    literals: tuple[str, ...] = ()
    if want_total:
        literals += profile.total_mag_keywords
    if want_absolute:
        literals += profile.absolute_mag_keywords
    gate = _gate(literals)

    total_mags: List[Union[float, List[float]]] = []
    absolute_mags: List[float] = []
    for line in lines:
        lower_line = line.lower()
        if gate.search(lower_line) is None:
            continue
        if want_total and _contains_any(lower_line, profile.total_mag_keywords):
            values = _numbers(line)
            if values:
                # Noncollinear runs print the three Cartesian components.
                total_mags.append(values[-3:] if len(values) >= 3 else values[-1])
        elif want_absolute and _contains_any(
            lower_line, profile.absolute_mag_keywords
        ):
            values = _numbers(line)
            if values:
                absolute_mags.append(values[-1])

    available["total_mag"] = total_mags[-1] if total_mags else None
    available["absolute_mag"] = absolute_mags[-1] if absolute_mags else None
    if "atom_orb_mag" in requested:
        available["atom_orb_mag"] = read_orbital_magnetization(
            lines, profile.orbital_mag_header_keywords
        )
    if "atom_mag_mulliken" in requested:
        mulliken_file = output_path / "mulliken.txt"
        if mulliken_file.is_file():
            steps = read_mulliken_magnetization(mulliken_file)
            available["atom_mag_mulliken"] = steps[-1] if steps else None
    return {metric: available[metric] for metric in requested}


def collect_relax_results(
    job_dir: str,
    metrics: List[str],
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect ionic relaxation results from the final relaxation step."""
    param_groups = split_param_by_group(metrics)
    if "relax_results" not in param_groups:
        return {}

    job_path = Path(job_dir)
    try:
        inputs = _job_input(job_path)
        output_path = _output_directory(job_path, inputs)
        calculation = _calculation(inputs, output_path)
    except FileNotFoundError:
        return _empty_results(param_groups["relax_results"])
    if calculation not in _RELAX_CALCULATIONS:
        return _empty_results(param_groups["relax_results"])
    try:
        log_path = _log_file(output_path, calculation, relax=True)
    except FileNotFoundError:
        return _empty_results(param_groups["relax_results"])

    profile = resolve_version(version, job_dir=job_path)
    requested = set(param_groups["relax_results"])
    want_force = bool({"largest_force", "relax_steps"} & requested)
    want_stress = bool({"largest_stress", "relax_steps"} & requested)
    want_converged = "relax_converged" in requested

    force_res = _compiled(profile.relax_force_patterns) if want_force else ()
    stress_res = _compiled(profile.relax_stress_patterns) if want_stress else ()
    literals = profile.relax_converged_keywords if want_converged else ()
    patterns: tuple[str, ...] = ()
    if want_force:
        patterns += profile.relax_force_patterns
    if want_stress:
        patterns += profile.relax_stress_patterns
    gate = _gate(literals, patterns)

    largest_forces: List[float] = []
    largest_stresses: List[float] = []
    relax_converged = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        lower_line = line.lower()
        if gate.search(lower_line) is None:
            continue
        force = _first_compiled(force_res, line)
        if force is not None:
            largest_forces.append(force)
        stress = _first_compiled(stress_res, line)
        if stress is not None:
            largest_stresses.append(stress)
        if want_converged and _contains_any(
            lower_line, profile.relax_converged_keywords
        ):
            relax_converged = True

    available = {
        "largest_force": largest_forces[-1] if largest_forces else None,
        "largest_stress": largest_stresses[-1] if largest_stresses else None,
        "relax_steps": max(len(largest_forces), len(largest_stresses)),
        "relax_converged": relax_converged,
    }
    return {metric: available[metric] for metric in param_groups["relax_results"]}


def _default_params(job_dir: str) -> List[str]:
    """Select default results based on whether the job has ionic relaxation."""
    job_path = Path(job_dir)
    inputs = _job_input(job_path)
    calculation = str(inputs.get("calculation", "scf")).lower()
    try:
        output_path = _output_directory(job_path, inputs)
    except FileNotFoundError:
        output_path = None
    if output_path is not None:
        calculation = _calculation(inputs, output_path)
    params = list(grouped_params["scf_results"])
    if calculation in _RELAX_CALCULATIONS:
        params.extend(grouped_params["relax_results"])
    return params


def get_result_from_job(
    job_dir: str,
    param_names: Optional[Sequence[str]],
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect requested results from one ABACUS job directory.

    Args:
        job_dir: ABACUS job directory containing ``INPUT`` and ``OUT.*``.
        param_names: Result names to collect; all result names when ``None``.
        version: ABACUS version hint such as ``"develop"`` or
            ``"3.10.1LTS"``.  ``None``/``"auto"`` (the default) selects the
            version declared by the running log.

    Returns:
        Mapping of the requested result names to their values.
    """
    if param_names is not None:
        params = list(param_names)
    else:
        try:
            params = _default_params(job_dir)
        except FileNotFoundError:
            params = list(grouped_params["scf_results"])
    param_groups = split_param_by_group(params)
    results: Dict[str, Any] = {}

    if "scf_results" in param_groups:
        results.update(
            collect_scf_results(job_dir, param_groups["scf_results"], version)
        )
    if "vdw_results" in param_groups:
        results.update(
            collect_vdw_results(job_dir, param_groups["vdw_results"], version)
        )
    if "magnetic_results" in param_groups:
        results.update(
            collect_magnetic_results(job_dir, param_groups["magnetic_results"], version)
        )
    if "relax_results" in param_groups:
        results.update(
            collect_relax_results(job_dir, param_groups["relax_results"], version)
        )
    return results
