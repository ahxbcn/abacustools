"""Models and diagnostics for one ABACUS calculation directory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from abacustools.io.abacus import ReadInput


@dataclass(frozen=True)
class ValidationIssue:
    """One problem or warning found while inspecting a job directory."""

    level: str
    code: str
    message: str


@dataclass
class JobValidation:
    """Structured result of validating one ABACUS input directory."""

    valid: bool
    issues: list[ValidationIssue]
    inputs: dict[str, Any]


@dataclass
class JobStatus:
    """Current state and latest progress of one ABACUS job."""

    state: str
    progress: dict[str, Any]
    log: Optional[Path] = None


def _as_positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _resolve_job_path(job: Path, filename: str) -> Path:
    path = Path(str(filename)).expanduser()
    return path if path.is_absolute() else job / path


def _resource_path(job: Path, filename: str, directory: Any) -> Optional[Path]:
    path = Path(str(filename)).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(job / path)
        if directory:
            resource_dir = Path(str(directory)).expanduser()
            candidates.append(resource_dir / path if resource_dir.is_absolute() else job / resource_dir / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _known_input_keywords() -> set[str]:
    from importlib.resources import files

    try:
        values = json.loads((files("abacustools.io") / "input-params.json").read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return set()
    known = set()
    for item in values:
        known.update(name.strip().lower() for name in str(item.get("name", "")).split(","))
    return {name for name in known if name}


def validate_job(job_dir: Path, *, strict: bool = False) -> JobValidation:
    """Validate the input and referenced resources of one ABACUS job."""
    job = Path(job_dir).expanduser().absolute()
    issues: list[ValidationIssue] = []
    inputs: dict[str, Any] = {}

    if not job.is_dir():
        issues.append(ValidationIssue("error", "missing-job", f"job directory does not exist: {job}"))
        return JobValidation(False, issues, inputs)

    input_path = job / "INPUT"
    if not input_path.is_file():
        issues.append(ValidationIssue("error", "missing-input", f"missing INPUT: {input_path}"))
        return JobValidation(False, issues, inputs)
    try:
        inputs = ReadInput(input_path)
    except (OSError, ValueError) as error:
        issues.append(ValidationIssue("error", "invalid-input", f"cannot read INPUT: {error}"))
        return JobValidation(False, issues, inputs)

    unknown = sorted(set(inputs) - _known_input_keywords())
    if unknown:
        level = "error" if strict else "warning"
        issues.append(
            ValidationIssue(level, "unknown-input", f"unknown INPUT keyword(s): {', '.join(unknown)}")
        )

    structure_path = _resolve_job_path(job, inputs.get("stru_file", "STRU"))
    if not structure_path.is_file():
        issues.append(ValidationIssue("error", "missing-stru", f"missing STRU: {structure_path}"))
        return JobValidation(not any(item.level == "error" for item in issues), issues, inputs)

    from abacustools.io.stru import AbacusSTRU

    structure = AbacusSTRU.read(structure_path)
    if structure is None:
        issues.append(ValidationIssue("error", "invalid-stru", f"cannot read STRU: {structure_path}"))
        return JobValidation(False, issues, inputs)

    pseudo_dir = inputs.get("pseudo_dir")
    orbital_dir = inputs.get("orbital_dir")
    for atom in structure.atoms:
        if not atom.pp:
            issues.append(
                ValidationIssue("error", "missing-pseudopotential", f"pseudopotential is not set for {atom.element}")
            )
        elif _resource_path(job, atom.pp, pseudo_dir) is None:
            issues.append(
                ValidationIssue("error", "missing-pseudopotential", f"pseudopotential not found: {atom.pp}")
            )

        basis = str(inputs.get("basis_type", "pw")).lower()
        if basis.startswith("lcao"):
            if not atom.orb:
                issues.append(ValidationIssue("error", "missing-orbital", f"orbital is not set for {atom.element}"))
            elif _resource_path(job, atom.orb, orbital_dir) is None:
                issues.append(ValidationIssue("error", "missing-orbital", f"orbital not found: {atom.orb}"))
        if atom.paw and _resource_path(job, atom.paw, inputs.get("paw_dir")) is None:
            issues.append(ValidationIssue("error", "missing-paw", f"PAW file not found: {atom.paw}"))

    if not _as_positive(inputs.get("gamma_only")) and not _as_positive(inputs.get("kspacing")):
        kpoint_path = _resolve_job_path(job, inputs.get("kpoint_file", "KPT"))
        if not kpoint_path.is_file():
            issues.append(ValidationIssue("error", "missing-kpt", f"missing KPT: {kpoint_path}"))

    valid = not any(item.level == "error" for item in issues)
    return JobValidation(valid, issues, inputs)


def _output_directory(job: Path, inputs: dict[str, Any]) -> Optional[Path]:
    suffix = inputs.get("suffix")
    if suffix:
        expected = job / f"OUT.{suffix}"
        if expected.is_dir():
            return expected
    output_directories = sorted(path for path in job.glob("OUT.*") if path.is_dir())
    return output_directories[0] if len(output_directories) == 1 else None


def _log_path(output: Path, calculation: str) -> Optional[Path]:
    names = [
        f"running_{calculation}.log",
        "running_scf.log",
        "running_relax.log",
        "running_cell-relax.log",
        "running_md.log",
    ]
    for name in dict.fromkeys(names):
        path = output / name
        if path.is_file():
            return path
    return None


def status_job(job_dir: Path, validation: Optional[JobValidation] = None) -> JobStatus:
    """Determine the calculation state without requiring a scheduler."""
    job = Path(job_dir).expanduser().absolute()
    validation = validation or validate_job(job)
    if not validation.valid:
        return JobStatus("invalid", {}, None)

    output = _output_directory(job, validation.inputs)
    if output is None:
        return JobStatus("ready", {}, None)
    calculation = str(validation.inputs.get("calculation", "scf")).lower()
    log = _log_path(output, calculation)
    if log is None:
        return JobStatus("incomplete", {}, None)

    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    failed = any(
        any(marker in line.lower() for marker in ("segmentation fault", "error in", "aborting"))
        for line in lines
    )
    from abacustools.data.abacus_result import get_result_from_job

    parameters = ["energy", "drho", "denergy", "scf_steps", "converged"]
    if calculation in {"relax", "cell-relax", "cell_relax", "md"}:
        parameters.extend(["largest_force", "largest_stress", "relax_steps", "relax_converged"])
    try:
        parsed = get_result_from_job(job, parameters, version="")
    except (FileNotFoundError, ValueError):
        parsed = {}
    progress = {key: parsed[key] for key in parameters if key in parsed and parsed[key] is not None}
    converged = bool(
        parsed.get("relax_converged")
        if calculation in {"relax", "cell-relax", "cell_relax", "md"}
        else parsed.get("converged")
    )

    if converged:
        state = "converged"
    elif failed:
        state = "failed"
    else:
        state = "running"
    return JobStatus(state, progress, log)


def as_json(value: Any) -> Any:
    """Convert job diagnostics to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (JobValidation, JobStatus, ValidationIssue)):
        return as_json(asdict(value))
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [as_json(item) for item in value]
    return value
