"""Models and diagnostics for one ABACUS calculation directory."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from abacustools.io.abacus import IsEnabled, KnownInputKeywords, ReadInput

if TYPE_CHECKING:
    from abacustools.io.stru import AbacusSTRU


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
class InputCheck:
    """Structured result of checking an ABACUS input and its dependencies."""

    valid: bool
    issues: list[ValidationIssue]
    inputs: dict[str, Any]
    summary: dict[str, Any]


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


def _has_positive_value(value: Any) -> bool:
    """Return whether a scalar or vector setting contains positive values."""
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_as_positive(item) for item in value)
    return _as_positive(value)


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


def _input_syntax_issues(input_path: Path) -> list[ValidationIssue]:
    """Find non-comment INPUT lines that cannot be parsed as key/value pairs."""
    issues = []
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return issues
    for line_number, line in enumerate(lines, start=1):
        content = line.split("#", 1)[0].strip()
        if not content or content.upper() == "INPUT_PARAMETERS":
            continue
        if len(content.split(None, 1)) != 2:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid-input-line",
                    f"INPUT line {line_number} must contain a keyword and a value",
                )
            )
    return issues


def _kpoint_summary(kpoint_path: Path) -> tuple[dict[str, Any], Optional[str]]:
    """Return a compact KPT summary and an optional syntax error."""
    try:
        lines = []
        for line in kpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].split("//", 1)[0].strip()
            if line:
                lines.append(line)
    except (OSError, UnicodeError) as error:
        return {}, f"cannot read KPT: {error}"

    if len(lines) < 3:
        return {}, "KPT must contain a count and a k-point mode"

    mode = lines[2].lower()
    if mode.startswith("gamma"):
        mode = "gamma"
    elif mode.startswith("mp") or mode.startswith("monkhorst"):
        mode = "mp"
    elif mode.startswith("direct"):
        mode = "direct"
    elif mode.startswith("cart"):
        mode = "cartesian"
    elif mode.startswith("line_cartesian"):
        mode = "line_cartesian"
    elif mode.startswith("line"):
        mode = "line"
    else:
        return {}, f"unsupported KPT mode: {lines[2]}"

    try:
        count = int(lines[1].split()[0])
    except (IndexError, ValueError):
        return {}, "KPT point count must be an integer"

    if mode in {"gamma", "mp"}:
        if len(lines) < 4:
            return {}, f"{mode} KPT requires a mesh line"
        values = lines[3].split()
        if len(values) < 3:
            return {}, f"{mode} KPT mesh must contain three integers"
        try:
            mesh = [int(value) for value in values[:3]]
        except ValueError:
            return {}, f"{mode} KPT mesh must contain three integers"
        if any(value <= 0 for value in mesh):
            return {}, f"{mode} KPT mesh must be positive"
        return {"mode": mode, "mesh": mesh}, None

    if count <= 0:
        return {}, "KPT point count must be positive"
    if len(lines) < 3 + count:
        return {}, f"KPT declares {count} points but contains fewer entries"
    for line in lines[3 : 3 + count]:
        values = line.split()
        if len(values) < 4:
            return {}, "each explicit KPT point needs coordinates and a weight"
        try:
            [float(value) for value in values[:4]]
        except ValueError:
            return {}, "explicit KPT coordinates and weights must be numeric"
    return {"mode": mode, "count": count}, None


def _input_summary(
    inputs: dict[str, Any],
    *,
    structure=None,
    kpoints: Optional[dict[str, Any]] = None,
    resources: Optional[dict[str, list[str]]] = None,
) -> dict[str, Any]:
    """Build the user-facing summary for an input check."""
    summary: dict[str, Any] = {
        "calculation": str(inputs.get("calculation", "scf")).lower(),
        "basis_type": str(inputs.get("basis_type", "pw")).lower(),
        "esolver_type": str(inputs.get("esolver_type", "ksdft")).lower(),
        "nspin": inputs.get("nspin", 1),
    }
    for name in ("ecutwfc", "scf_thr", "smearing_method", "smearing_sigma"):
        if name in inputs:
            summary[name] = inputs[name]
    if IsEnabled(inputs.get("gamma_only")):
        summary["kpoints"] = {"mode": "gamma_only"}
    elif _has_positive_value(inputs.get("kspacing")):
        summary["kpoints"] = {"mode": "kspacing", "value": inputs["kspacing"]}
    elif kpoints is not None:
        summary["kpoints"] = kpoints
    if structure is not None:
        counts = Counter(
            atom.element or atom.label
            for atom in structure.atoms
        )
        from abacustools.data.unitcell import Unitcell

        cell = Unitcell(structure.cell)
        summary["structure"] = {
            "natoms": structure.natoms,
            "species": dict(sorted(counts.items())),
            "cell_parameters_ang_deg": cell.get_cell_param(),
            "cell_volume_ang3": cell.get_cell_volume(),
        }
    if resources is not None:
        summary["resources"] = resources
    return summary


def read_job_input(job: Path) -> dict[str, Any]:
    """Read an ABACUS job's INPUT file with a consistent error message.

    Args:
        job: ABACUS calculation directory that holds ``INPUT``.

    Returns:
        The parsed INPUT parameters.

    Raises:
        FileNotFoundError: When the directory has no ``INPUT`` file.
    """
    job = Path(job)
    input_path = job / "INPUT"
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not find INPUT in ABACUS job: {input_path}")
    return ReadInput(input_path)


def read_job_structure(job: Path) -> tuple[dict[str, Any], str, AbacusSTRU]:
    """Read an ABACUS job's INPUT and the structure it references.

    Args:
        job: ABACUS calculation directory that holds ``INPUT`` and ``STRU``.

    Returns:
        A ``(inputs, stru_filename, structure)`` tuple: the parsed INPUT
        parameters, the structure filename named by ``stru_file``, and the
        parsed structure.

    Raises:
        FileNotFoundError: When the directory has no ``INPUT`` file.
        RuntimeError: When the referenced structure file cannot be read.
    """
    from abacustools.io.stru import AbacusSTRU

    job = Path(job)
    inputs = read_job_input(job)
    stru_filename = str(inputs.get("stru_file", "STRU"))
    stru_path = job / stru_filename
    structure = AbacusSTRU.read(stru_path)
    if structure is None:
        raise RuntimeError(f"failed to read structure: {stru_path}")
    return inputs, stru_filename, structure


def check_input(job_dir: Path, *, strict: bool = False) -> InputCheck:
    """Check an ABACUS INPUT and the files it references, without reading output."""
    job = Path(job_dir).expanduser().absolute()
    issues: list[ValidationIssue] = []
    inputs: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    if not job.is_dir():
        issues.append(ValidationIssue("error", "missing-job", f"job directory does not exist: {job}"))
        return InputCheck(False, issues, inputs, summary)

    input_path = job / "INPUT"
    if not input_path.is_file():
        issues.append(ValidationIssue("error", "missing-input", f"missing INPUT: {input_path}"))
        return InputCheck(False, issues, inputs, summary)
    try:
        inputs = ReadInput(input_path)
    except (OSError, ValueError) as error:
        issues.append(ValidationIssue("error", "invalid-input", f"cannot read INPUT: {error}"))
        return InputCheck(False, issues, inputs, summary)

    issues.extend(_input_syntax_issues(input_path))
    summary = _input_summary(inputs)
    unknown = sorted(set(inputs) - KnownInputKeywords())
    if unknown:
        level = "error" if strict else "warning"
        issues.append(
            ValidationIssue(level, "unknown-input", f"unknown INPUT keyword(s): {', '.join(unknown)}")
        )

    calculation = summary["calculation"]
    valid_calculations = {
        "scf",
        "nscf",
        "relax",
        "cell-relax",
        "md",
        "get_pchg",
        "get_wf",
        "get_s",
        "gen_bessel",
        "test_memory",
        "test_neighbour",
    }
    if calculation not in valid_calculations:
        issues.append(ValidationIssue("error", "invalid-calculation", f"unsupported calculation: {calculation}"))

    basis_type = summary["basis_type"]
    if basis_type not in {"pw", "lcao"}:
        issues.append(ValidationIssue("error", "invalid-basis", f"unsupported basis_type: {basis_type}"))

    try:
        nspin = int(inputs.get("nspin", 1))
        if nspin not in {1, 2, 4}:
            raise ValueError
    except (TypeError, ValueError):
        issues.append(ValidationIssue("error", "invalid-nspin", "nspin must be one of 1, 2, or 4"))

    for name in ("ecutwfc", "scf_thr"):
        if name in inputs and not _as_positive(inputs[name]):
            issues.append(ValidationIssue("error", f"invalid-{name}", f"{name} must be positive"))
    if "kspacing" in inputs:
        kspacing = inputs["kspacing"]
        valid_kspacing = (
            _has_positive_value(kspacing)
            and not isinstance(kspacing, (list, tuple))
        ) or (
            isinstance(kspacing, (list, tuple))
            and len(kspacing) == 3
            and _has_positive_value(kspacing)
        )
        if not valid_kspacing:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid-kspacing",
                    "kspacing must be a positive number or three positive numbers",
                )
            )

    structure = None
    structure_path = _resolve_job_path(job, inputs.get("stru_file", "STRU"))
    if not structure_path.is_file():
        issues.append(ValidationIssue("error", "missing-stru", f"missing STRU: {structure_path}"))
    else:
        from abacustools.io.stru import AbacusSTRU

        try:
            structure = AbacusSTRU.read(structure_path)
        except Exception as error:
            structure = None
            issues.append(ValidationIssue("error", "invalid-stru", f"cannot read STRU: {error}"))
        if structure is None and not any(item.code == "invalid-stru" for item in issues):
            issues.append(ValidationIssue("error", "invalid-stru", f"cannot read STRU: {structure_path}"))

    resources = {"pseudopotentials": [], "orbitals": [], "paw": []}
    if structure is not None:
        pseudo_dir = inputs.get("pseudo_dir")
        orbital_dir = inputs.get("orbital_dir")
        for atom in structure.atoms:
            if not atom.pp:
                issues.append(ValidationIssue("error", "missing-pseudopotential", f"pseudopotential is not set for {atom.element or atom.label}"))
            else:
                resources["pseudopotentials"].append(str(atom.pp))
                if _resource_path(job, atom.pp, pseudo_dir) is None:
                    issues.append(ValidationIssue("error", "missing-pseudopotential", f"pseudopotential not found: {atom.pp}"))

            if basis_type == "lcao":
                if not atom.orb:
                    issues.append(ValidationIssue("error", "missing-orbital", f"orbital is not set for {atom.element or atom.label}"))
                else:
                    resources["orbitals"].append(str(atom.orb))
                    if _resource_path(job, atom.orb, orbital_dir) is None:
                        issues.append(ValidationIssue("error", "missing-orbital", f"orbital not found: {atom.orb}"))
            if atom.paw:
                resources["paw"].append(str(atom.paw))
                if _resource_path(job, atom.paw, inputs.get("paw_dir")) is None:
                    issues.append(ValidationIssue("error", "missing-paw", f"PAW file not found: {atom.paw}"))

    resources = {name: sorted(set(values)) for name, values in resources.items() if values}
    kpoints = None
    if not IsEnabled(inputs.get("gamma_only")) and not _has_positive_value(inputs.get("kspacing")):
        kpoint_path = _resolve_job_path(job, inputs.get("kpoint_file", "KPT"))
        if not kpoint_path.is_file():
            issues.append(ValidationIssue("error", "missing-kpt", f"missing KPT: {kpoint_path}"))
        else:
            kpoints, error = _kpoint_summary(kpoint_path)
            if error:
                issues.append(ValidationIssue("error", "invalid-kpt", f"cannot read KPT: {error}"))

    try:
        summary = _input_summary(inputs, structure=structure, kpoints=kpoints, resources=resources)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        issues.append(ValidationIssue("error", "invalid-stru", f"cannot summarize STRU: {error}"))
        summary = _input_summary(inputs, kpoints=kpoints, resources=resources)
    valid = not any(item.level == "error" for item in issues)
    return InputCheck(valid, issues, inputs, summary)


def validate_job(job_dir: Path, *, strict: bool = False) -> JobValidation:
    """Backward-compatible input validation facade."""
    report = check_input(job_dir, strict=strict)
    return JobValidation(report.valid, report.issues, report.inputs)


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
        parsed = get_result_from_job(job, parameters, version=None)
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
    if isinstance(value, (InputCheck, JobValidation, JobStatus, ValidationIssue)):
        return as_json(asdict(value))
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [as_json(item) for item in value]
    return value
