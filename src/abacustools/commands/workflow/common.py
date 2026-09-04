"""Shared helpers for multi-stage ABACUS workflows."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


def workflow_manifest_path(job: Path, workflow: str) -> Path:
    """Return the manifest path for a workflow in a job directory."""
    return job / f"workflow_{workflow}.json"


def register_stages(
    subparsers,
    command: str,
    help_text: str,
    prepare_handler: Callable,
    postprocess_handler: Callable,
    add_prepare_arguments: Callable,
    add_postprocess_arguments: Callable,
    *,
    aliases: Optional[list[str]] = None,
) -> None:
    """Register the standard prepare/postprocess workflow stages."""
    workflow_parser = subparsers.add_parser(
        command,
        aliases=aliases or [],
        help=help_text,
    )
    stages = workflow_parser.add_subparsers(
        dest=f"{command}_command",
        metavar="COMMAND",
        title=f"{command} commands",
        required=True,
    )

    prepare_parser = stages.add_parser(
        "prepare",
        help=f"Prepare jobs for the {command} workflow.",
    )
    add_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=prepare_handler)

    postprocess_parser = stages.add_parser(
        "postprocess",
        help=f"Postprocess the {command} workflow results.",
    )
    add_postprocess_arguments(postprocess_parser)
    postprocess_parser.set_defaults(handler=postprocess_handler)


def has_kpoint_setting(inputs: dict[str, Any]) -> bool:
    """Return whether INPUT can generate k points without a KPT file."""
    try:
        if float(inputs.get("gamma_only", 0)) > 0:
            return True
    except (TypeError, ValueError):
        pass

    kspacing = inputs.get("kspacing", 0)
    values = kspacing if isinstance(kspacing, (list, tuple)) else (kspacing,)
    for value in values:
        try:
            if float(value) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def kpoint_filename(job: Path, inputs: dict[str, Any]) -> Optional[str]:
    """Return the KPT filename only when explicit K points are required."""
    if has_kpoint_setting(inputs):
        return None

    filename = str(inputs.get("kpoint_file", "KPT"))
    if not (job / filename).is_file():
        raise RuntimeError(f"could not find KPT file: {job / filename}")
    return filename


def clear_generated_jobs(
    job: Path,
    names: Iterable[str],
    *,
    override: bool,
) -> None:
    """Refuse or remove existing generated workflow directories."""
    existing = [
        job / name
        for name in names
        if (job / name).exists() or (job / name).is_symlink()
    ]
    if existing and not override:
        paths = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            f"generated workflow directories already exist: {paths}; "
            "use --override to replace them"
        )

    for path in existing:
        print(f"  removing old directory: {path}")
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _link_file(
    source_dir: Path,
    destination_dir: Path,
    filename: Optional[str],
    file_type: str,
) -> None:
    if not filename:
        return
    source = source_dir / filename
    if not source.is_file():
        raise RuntimeError(f"{file_type} file not found: {source}")
    target = destination_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def copy_referenced_files(
    structure,
    source_dir: Path,
    destination_dir: Path,
    *,
    kpoint: Optional[str] = None,
) -> None:
    """Link all files referenced by a generated STRU and INPUT."""
    files = [
        *((filename, "pseudopotential") for filename in structure.pps),
        *((filename, "orbital") for filename in structure.orbs),
        *((filename, "PAW") for filename in structure.paws),
    ]
    copied = set()
    for filename, file_type in files:
        if filename in copied:
            continue
        copied.add(filename)
        _link_file(source_dir, destination_dir, filename, file_type)

    if kpoint:
        _link_file(source_dir, destination_dir, kpoint, "KPT")


def write_abacus_job(
    inputs: dict[str, Any],
    structure,
    source_dir: Path,
    destination_dir: Path,
    *,
    stru_filename: str,
    kpoint: Optional[str] = None,
) -> None:
    """Write one generated ABACUS job and its referenced input files."""
    from abacustools.io.abacus import WriteInput

    destination_dir.mkdir(parents=True, exist_ok=True)
    WriteInput(inputs, destination_dir / "INPUT")
    if not structure.write(destination_dir / stru_filename):
        raise RuntimeError(f"failed to write structure: {destination_dir / stru_filename}")
    copy_referenced_files(
        structure,
        source_dir,
        destination_dir,
        kpoint=kpoint,
    )


def write_manifest(job: Path, workflow: str, **metadata: Any) -> None:
    """Write metadata describing generated workflow jobs."""
    manifest = {"format": 1, "workflow": workflow, **metadata}
    path = workflow_manifest_path(job, workflow)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def read_manifest(
    job: Path,
    workflow: str,
    required_tasks: Iterable[str],
) -> dict[str, Any]:
    """Read and validate a workflow manifest."""
    path = workflow_manifest_path(job, workflow)
    if not path.is_file():
        raise FileNotFoundError(f"Could not find workflow manifest: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid workflow manifest: {path}") from error
    if manifest.get("workflow") != workflow:
        raise RuntimeError(
            f"workflow manifest is for {manifest.get('workflow')!r}, "
            f"not {workflow!r}: {path}"
        )
    actual_tasks = set(manifest.get("tasks", []))
    missing_tasks = set(required_tasks) - actual_tasks
    if missing_tasks:
        names = ", ".join(sorted(missing_tasks))
        raise RuntimeError(f"workflow manifest is missing tasks: {names}")
    return manifest


def completed_scf_output(job: Path):
    """Return INPUT and output directory after confirming SCF convergence."""
    from abacustools.io.abacus import ReadInput

    input_path = job / "INPUT"
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not find INPUT in workflow job: {job}")
    inputs = ReadInput(input_path)
    suffix = inputs.get("suffix", "ABACUS")
    output_dir = job / f"OUT.{suffix}"
    log_path = output_dir / "running_scf.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"Could not find SCF log in workflow job: {job}")
    log = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if "charge density convergence is achieved" not in log:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    return inputs, output_dir
