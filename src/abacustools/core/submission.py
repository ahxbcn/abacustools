"""Configured submission-script generation for workflow jobs."""

from __future__ import annotations

import re
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from .config import CONFIG


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _simple_filename(value: Any, name: str) -> str:
    """Validate a configured filename cannot escape its task directory."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{name} must be a simple filename")
    return value


def _shell_command(command: str, name: str) -> str:
    """Normalize a command line while preserving arguments and quoting."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"submission {name} must be a non-empty command")
    try:
        parts = shlex.split(command)
    except ValueError as error:
        raise ValueError(f"submission {name} is not a valid shell command") from error
    if not parts:
        raise ValueError(f"submission {name} must be a non-empty command")
    return shlex.join(parts)


def _safe_name(value: str) -> str:
    """Convert a workflow task path into a scheduler-friendly job name."""
    name = _SAFE_NAME.sub("_", value.strip("/"))
    return name or "abacus-workflow"


def resolve_submission(
    submission_type: Optional[str] = None,
    generate: Optional[bool] = None,
    abacus_command: Optional[str] = None,
    *,
    config: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Resolve a configured submission type and validate its template."""
    source_config = CONFIG if config is None else config
    settings = deepcopy(source_config.get("submission", {}))
    if not isinstance(settings, dict):
        raise ValueError("submission configuration must be a mapping")
    enabled = settings.get("generate", False) if generate is None else generate
    if not isinstance(enabled, bool):
        raise ValueError("submission.generate must be a boolean")
    if not enabled:
        return None

    selected = submission_type if submission_type is not None else settings.get("default", "slurm")
    templates = settings.get("templates", {})
    if not isinstance(templates, dict) or selected not in templates:
        available = ", ".join(sorted(str(key) for key in templates))
        raise ValueError(
            f"unknown submission type {selected!r}; available types: {available or 'none'}"
        )
    template = templates[selected]
    if not isinstance(template, dict):
        raise ValueError(f"submission.templates.{selected} must be a mapping")
    filename = _simple_filename(
        template.get("filename"), f"submission.templates.{selected}.filename"
    )
    body = template.get("template")
    if not isinstance(body, str) or not body.strip():
        raise ValueError(f"submission.templates.{selected}.template must be non-empty text")

    launcher = template.get("launcher", {})
    if not isinstance(launcher, dict):
        raise ValueError(f"submission.templates.{selected}.launcher must be a mapping")
    mode = launcher.get("mode", "scheduler")
    if mode not in {"local", "scheduler"}:
        raise ValueError(
            f"submission.templates.{selected}.launcher.mode must be local or scheduler"
        )
    result = {
        "type": str(selected),
        "filename": filename,
        "template": body,
        "launcher": launcher,
        "workflow_filename": _simple_filename(
            settings.get("workflow_filename", "submit_{workflow}.sh"),
            "submission.workflow_filename",
        ),
        "abacus_command": _shell_command(
            (
                abacus_command
                if abacus_command is not None
                else settings.get("abacus_command", "abacus")
            ),
            "abacus_command",
        ),
    }
    return result


def _render(template: str, context: dict[str, str]) -> str:
    """Render a configured template and report missing placeholders clearly."""
    try:
        return template.format(**context)
    except KeyError as error:
        raise ValueError(f"unknown submission-template placeholder: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid submission template: {error}") from error


def _task_context(workflow: str, task: str, command: str) -> dict[str, str]:
    return {
        "abacus_command": command,
        "job_name": _safe_name(f"{workflow}-{task}"),
        "task_name": task,
        "workflow": workflow,
    }


def _write_task_scripts(
    job: Path,
    workflow: str,
    tasks: list[str],
    settings: dict[str, Any],
) -> None:
    for task in tasks:
        path = job / task / settings["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _render(
                settings["template"],
                _task_context(workflow, task, settings["abacus_command"]),
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)


def _scheduler_submission(
    settings: dict[str, Any],
    task_script: str,
    task: str,
    dependency: Optional[str],
) -> str:
    launcher = settings["launcher"]
    submit_command = _shell_command(
        launcher.get("submit_command", "sbatch --parsable"), "submit_command"
    )
    command = f"{submit_command} {shlex.quote(task_script)}"
    if dependency:
        dependency_option = launcher.get("dependency_option")
        if not isinstance(dependency_option, str) or not dependency_option.strip():
            raise ValueError(
                f"submission.templates.{settings['type']}.launcher.dependency_option "
                "is required for dependent tasks"
            )
        command = (
            f"{submit_command} "
            f"{dependency_option.replace('{dependency_id}', dependency)} "
            f"{shlex.quote(task_script)}"
        )
    filter_command = launcher.get("id_filter", "cat")
    filter_command = _shell_command(filter_command, "id_filter")
    return f'(cd "$ROOT_DIR/{task}" && {command}) | {filter_command}'


def _write_workflow_launcher(
    job: Path,
    workflow: str,
    tasks: list[str],
    settings: dict[str, Any],
) -> Path:
    launcher = settings["launcher"]
    filename = settings["workflow_filename"].format(workflow=workflow)
    path = job / filename
    task_script = settings["filename"]
    task_command = shlex.quote(f"./{task_script}")
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "ROOT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "",
    ]
    if launcher.get("mode", "scheduler") == "local":
        equilibrium = tasks[0]
        lines.extend(
            [
                f'(cd "$ROOT_DIR/{equilibrium}" && {task_command})',
                "pids=()",
            ]
        )
        for task in tasks[1:]:
            lines.append(f'(cd "$ROOT_DIR/{task}" && {task_command}) &')
            lines.append("pids+=(\"$!\")")
        lines.extend(
            [
                "status=0",
                'for pid in "${pids[@]}"; do',
                '    wait "$pid" || status=$?',
                "done",
                'exit "$status"',
            ]
        )
    else:
        equilibrium = tasks[0]
        lines.append(
            f'equilibrium_id=$({_scheduler_submission(settings, task_script, equilibrium, None)})'
        )
        lines.extend(
            [
                'if [ -z "$equilibrium_id" ]; then',
                '    echo "failed to obtain the equilibrium job id" >&2',
                "    exit 1",
                "fi",
            ]
        )
        for task in tasks[1:]:
            lines.append(
                f'_scheduler_id=$({_scheduler_submission(settings, task_script, task, "${equilibrium_id}")})'
            )
            lines.extend(
                [
                    'if [ -z "$_scheduler_id" ]; then',
                    f'    echo "failed to submit {task}" >&2',
                    "    exit 1",
                    "fi",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def generate_workflow_submission(
    job: Path,
    workflow: str,
    tasks: list[str],
    *,
    submission_type: Optional[str] = None,
    generate: Optional[bool] = None,
    abacus_command: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Generate task scripts and a dependency-aware workflow launcher."""
    if not tasks:
        raise ValueError("at least one workflow task is required")
    settings = resolve_submission(submission_type, generate, abacus_command)
    if settings is None:
        return None
    _write_task_scripts(job, workflow, tasks, settings)
    launcher = _write_workflow_launcher(job, workflow, tasks, settings)
    return {
        "type": settings["type"],
        "task_script": settings["filename"],
        "workflow_script": launcher.name,
        "task_count": len(tasks),
    }
