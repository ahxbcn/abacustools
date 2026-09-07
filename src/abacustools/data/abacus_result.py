"""Collect selected results from ABACUS calculation output."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

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
    "relax_results": [
        "largest_force",
        "largest_stress",
        "relax_steps",
        "relax_converged",
    ],
}

_RELAX_CALCULATIONS = {"relax", "cell-relax", "md"}
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _numbers(line: str) -> List[float]:
    return [_as_float(value) for value in re.findall(_FLOAT, line)]


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
) -> List[List[float]]:
    force: List[List[float]] = []
    for line in lines[start + 1 :]:
        if _is_separator(line):
            if force:
                return force
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                force.append([_as_float(value) for value in parts[1:4]])
            except ValueError:
                if force:
                    return force
    return force


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


def collect_scf_results(job_dir: str, metrics: List[str]) -> Dict[str, Any]:
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

    last_line = next((line.strip() for line in reversed(lines) if line.strip()), None)
    normal_end = (
        None if last_line is None else "Total  Time  :" in last_line
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
        if "e_kohnsham" in lower_line:
            values = _numbers(line)
            if values:
                energies.append(values[-1])
        elif "density error" in lower_line:
            values = _numbers(line)
            if values:
                drhos.append(values[-1])
        elif "final etot" in lower_line:
            values = _numbers(line)
            if values:
                final_energies.append(values[-1])
        elif "efermi" in lower_line:
            values = _numbers(line)
            if values:
                efermis.append(values[-1])
        elif "charge density convergence is achieved" in lower_line:
            converged = True
        elif "total-force" in lower_line:
            force = _parse_force_block(lines, line_number)
            if force:
                forces.append(force)
        elif "total-stress" in lower_line:
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


def collect_vdw_results(job_dir: str, metrics: List[str]) -> Dict[str, Any]:
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

    vdw_energies: List[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "e_vdw" in line.lower():
            values = _numbers(line)
            if values:
                # ABACUS prints the dispersion energy in Rydberg and eV.
                vdw_energies.append(values[-1])

    available = {"vdw_energy": vdw_energies[-1] if vdw_energies else None}
    return {metric: available[metric] for metric in param_groups["vdw_results"]}


def collect_relax_results(job_dir: str, metrics: List[str]) -> Dict[str, Any]:
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

    largest_forces: List[float] = []
    largest_stresses: List[float] = []
    relax_converged = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        lower_line = line.lower()
        values = _numbers(line)
        if "largest gradient in force is" in lower_line and values:
            largest_forces.append(values[-1])
        elif "largest gradient in stress is" in lower_line and values:
            largest_stresses.append(values[-1])
        elif "relaxation is converged" in lower_line:
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
    version: str,
) -> Dict[str, Any]:
    """Collect requested results from one ABACUS job directory."""
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
        results.update(collect_scf_results(job_dir, param_groups["scf_results"]))
    if "vdw_results" in param_groups:
        results.update(collect_vdw_results(job_dir, param_groups["vdw_results"]))
    if "relax_results" in param_groups:
        results.update(collect_relax_results(job_dir, param_groups["relax_results"]))
    return results
