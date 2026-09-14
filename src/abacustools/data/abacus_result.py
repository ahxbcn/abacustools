"""Collect selected results from ABACUS calculation output."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from abacustools.core.constant import RY_TO_EV
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
_STRESS_COMPONENTS = (
    ("xx", "xy", "xz"),
    ("xy", "yy", "yz"),
    ("xz", "yz", "zz"),
)


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _numbers(line: str) -> List[float]:
    return [_as_float(value) for value in re.findall(_FLOAT, line)]


def _contains_any(line: str, keywords: Sequence[str]) -> bool:
    """Return whether a lowercased log line contains one of the keywords."""
    return any(keyword.lower() in line for keyword in keywords)


@lru_cache(maxsize=None)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile and cache the regular expressions of one version profile."""
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def _first_match(patterns: tuple[str, ...], line: str) -> Optional[float]:
    """Return the value captured by the first matching pattern, if any."""
    for pattern in _compiled(patterns):
        match = pattern.search(line)
        if match is not None:
            return _as_float(match.group(1))
    return None


def _force_extremes(
    block: List[List[float]],
) -> Optional[tuple[float, int, str]]:
    """Return the magnitude, one-based atom index and component of the largest force."""
    best: Optional[tuple[float, int, str]] = None
    for index, values in enumerate(block, start=1):
        vector = np.asarray(values, dtype=float)
        magnitude = float(np.linalg.norm(vector))
        if best is None or magnitude > best[0]:
            axis = int(np.argmax(np.abs(vector)))
            best = (magnitude, index, "xyz"[axis])
    return best


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
    can count the components beyond a convergence threshold.
    """
    path = Path(log_file)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    profile = resolve_version(version, job_dir=path.parent, text="\n".join(lines[:128]))
    records, force_blocks, stress_blocks = _collect_ionic_steps(
        lines, profile, profile.relax_step_patterns
    )
    if not records:
        # Some branches advance the ionic step without a relaxation marker and
        # only number it in the ``ION=`` field of the electronic loop.
        records, force_blocks, stress_blocks = _collect_ionic_steps(
            lines, profile, profile.ion_step_patterns
        )

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
                "converged": False,
            },
        )

    for index, line in enumerate(lines):
        step = _first_match(step_patterns, line)
        if step is not None:
            current_step = int(step)
            record(current_step)
            continue
        if current_step is None:
            continue
        item = record(current_step)
        if _contains_any(line.lower(), profile.energy_keywords):
            values = _numbers(line)
            if values:
                scf_energies[current_step] = values[-1]
        energy = _first_match(profile.relax_energy_patterns, line)
        if energy is not None:
            item["energy"] = energy
        energy_diff_match = _RELAX_ENERGY_DIFF_RE.search(line)
        if energy_diff_match:
            item["energy_change"] = _as_float(energy_diff_match.group(1))
        force = _first_match(profile.relax_force_patterns, line)
        if force is not None:
            item["max_force"] = force
        stress = _first_match(profile.relax_stress_patterns, line)
        if stress is not None:
            item["max_stress"] = stress
        if _contains_any(line.lower(), profile.force_header_keywords):
            labels, block = _parse_force_block(lines, index)
            if block:
                force_blocks[current_step] = (labels, block)
        elif _contains_any(line.lower(), profile.stress_header_keywords):
            tensor = _parse_stress_block(lines, index)
            if tensor:
                stress_blocks[current_step] = tensor
        if _contains_any(line.lower(), profile.relax_converged_keywords):
            item["converged"] = True

    # The ``final etot is`` report line is rounded, while the Kohn-Sham energy
    # of the last electronic iteration carries the full precision of the step.
    for step, energy in scf_energies.items():
        records[step]["energy"] = energy

    return records, force_blocks, stress_blocks


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

    for line in lines:
        step = _first_match(profile.scf_step_patterns, line)
        if step is not None:
            current_step = int(step)
            record(current_step)
            continue
        if current_step is None:
            continue
        item = record(current_step)
        lower_line = line.lower()
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

    for index, line in enumerate(lines):
        step = _first_match(profile.md_step_patterns, line)
        if step is not None:
            current_step = int(step)
            record(current_step)
            continue
        if current_step is None:
            continue
        if "total-pressure" in line.lower():
            values = _numbers(line)
            if values:
                record(current_step)["pressure"] = values[-1]
            continue
        if not _is_md_table_header(line):
            continue
        lower_line = line.lower()
        energy_factor = RY_TO_EV if "(ry)" in lower_line else 1.0
        has_pressure = "pressure" in lower_line
        for following in lines[index + 1 :]:
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
    thresholds: Dict[str, float] = {}
    for line in lines:
        if "force_thr_ev" not in thresholds:
            value = _first_match(profile.relax_force_threshold_patterns, line)
            if value is not None:
                thresholds["force_thr_ev"] = value
        if "stress_thr" not in thresholds:
            value = _first_match(profile.relax_stress_threshold_patterns, line)
            if value is not None:
                thresholds["stress_thr"] = value
    return thresholds


def read_normal_end(
    log_file: Union[str, Path],
    version: Optional[str] = None,
) -> bool:
    """Return whether a running log reached the normal-ending footer."""
    path = Path(log_file)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    profile = resolve_version(version, job_dir=path.parent, text="\n".join(lines[:128]))
    return any(
        _contains_any(line.lower(), profile.normal_end_keywords)
        for line in lines[-80:]
    )


def _is_separator(line: str) -> bool:
    """Return whether a log line is made only of separator characters."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"-"}


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
    """Read total DOS data from an ABACUS job output."""
    job_path = Path(job_dir)
    inputs = _job_input(job_path)
    output_path = _output_directory(job_path, inputs)
    dos_files = sorted(output_path.glob("DOS*_smearing.dat"))
    if not dos_files:
        raise FileNotFoundError(f"Could not find DOS files in {output_path}")

    energy = None
    dos_channels = []
    for dos_file in dos_files:
        data = np.loadtxt(dos_file, ndmin=2)
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


def read_pdos_from_job(job_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read projected DOS data from an ABACUS PDOS XML file."""
    job_path = Path(job_dir)
    inputs = _job_input(job_path)
    output_path = _output_directory(job_path, inputs)
    pdos_file = output_path / "PDOS"
    if not pdos_file.is_file():
        raise FileNotFoundError(f"Could not find PDOS file: {pdos_file}")

    result = read_orbital_xml(pdos_file)
    energy = result["energy"]
    if energy is None:
        raise ValueError(f"PDOS file has no energy grid: {pdos_file}")

    for orbital in result["orbitals"]:
        data = orbital["data"]
        if data.shape[0] != energy.shape[0]:
            raise ValueError(f"PDOS energy and data lengths do not match: {pdos_file}")

    return result


def _calculation(inputs: Dict[str, Any], output_path: Path) -> str:
    """Get calculation type, preferring the output INPUT when available."""
    output_input = output_path / "INPUT"
    if output_input.is_file():
        output_inputs = ReadInput(output_input)
        inputs = {**inputs, **output_inputs}
    return str(inputs.get("calculation", "scf")).lower()


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
    for line in lines[start + 1 :]:
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
    for line in lines[start + 1 :]:
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

    energies: List[float] = []
    drhos: List[float] = []
    efermis: List[float] = []
    final_energies: List[float] = []
    forces: List[List[List[float]]] = []
    stresses: List[List[List[float]]] = []
    converged = False

    for line_number, line in enumerate(lines):
        lower_line = line.lower()
        if _contains_any(lower_line, profile.energy_keywords):
            values = _numbers(line)
            if values:
                energies.append(values[-1])
        elif _contains_any(lower_line, profile.density_error_keywords):
            values = _numbers(line)
            if values:
                drhos.append(values[-1])
        elif _contains_any(lower_line, profile.final_energy_keywords):
            values = _numbers(line)
            if values:
                final_energies.append(values[-1])
        elif _contains_any(lower_line, profile.fermi_keywords):
            values = _numbers(line)
            if values:
                efermis.append(values[-1])
        elif _contains_any(lower_line, profile.scf_converged_keywords):
            converged = True
        elif _contains_any(lower_line, profile.force_header_keywords):
            _, force = _parse_force_block(lines, line_number)
            if force:
                forces.append(force)
        elif _contains_any(lower_line, profile.stress_header_keywords):
            stress = _parse_stress_block(lines, line_number)
            if stress:
                stresses.append(stress)

    energy_values = final_energies or energies
    available = {
        "energy": energy_values[-1] if energy_values else None,
        "drho": drhos[-1] if drhos else None,
        "denergy": energies[-1] - energies[-2] if len(energies) > 1 else None,
        "scf_steps": len(energies),
        "converged": converged,
        "normal_end": normal_end,
        "efermi": efermis[-1] if efermis else None,
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
    vdw_energies: List[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _contains_any(line.lower(), profile.vdw_keywords):
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
    for index, line in enumerate(lines):
        if not _contains_any(line.lower(), header_keywords):
            continue
        block: List[List[float]] = []
        for row in lines[index + 1 :]:
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

    total_mags: List[Union[float, List[float]]] = []
    absolute_mags: List[float] = []
    for line in lines:
        lower_line = line.lower()
        if _contains_any(lower_line, profile.total_mag_keywords):
            values = _numbers(line)
            if values:
                # Noncollinear runs print the three Cartesian components.
                total_mags.append(values[-3:] if len(values) >= 3 else values[-1])
        elif _contains_any(lower_line, profile.absolute_mag_keywords):
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
    largest_forces: List[float] = []
    largest_stresses: List[float] = []
    relax_converged = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        force = _first_match(profile.relax_force_patterns, line)
        if force is not None:
            largest_forces.append(force)
        stress = _first_match(profile.relax_stress_patterns, line)
        if stress is not None:
            largest_stresses.append(stress)
        if _contains_any(line.lower(), profile.relax_converged_keywords):
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
