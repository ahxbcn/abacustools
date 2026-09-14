"""Implementation of the ``abacustools job monitor`` command."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

from abacustools.core.job import status_job, validate_job
from abacustools.data.abacus_result import (
    find_job_log,
    read_convergence_thresholds,
    read_md_history,
    read_normal_end,
    read_relaxation_history,
    read_scf_history,
)
from abacustools.io.stru import AbacusSTRU


ENERGY_UNIT = "eV"
FORCE_UNIT = "eV/Angstrom"
STRESS_UNIT = "kBar"
TEMPERATURE_UNIT = "K"
PRESSURE_UNIT = "kBar"

#: MD thermostat types that control a target temperature.
_THERMOSTAT_MD_TYPES = {"nvt", "npt", "langevin", "andersen", "bussi", "csvr", "nhc"}
#: MD types that control a target pressure.
_PRESSURE_MD_TYPES = {"npt", "msst"}

#: Plot file written below JOB when ``--plot`` is used without a name.
_PLOT_FILENAMES = {
    "scf": "monitor_scf.png",
    "relax": "monitor_relax.png",
    "cell-relax": "monitor_cell-relax.png",
    "md": "monitor_md.png",
}
#: ``--plot`` value asking for the task-specific default file name.
_AUTO_PLOT = "auto"

#: Line style of a convergence curve.
_CURVE_STYLE = {
    "linestyle": "-",
    "color": "red",
    "marker": "s",
    "markeredgecolor": "black",
    "markerfacecolor": "black",
    "markersize": 4,
}
#: Dotted style of a convergence threshold line.
_THRESHOLD_STYLE = {"linestyle": ":", "color": "k", "linewidth": 1, "alpha": 0.5}
#: Color of a quantity drawn on the twin axis of a convergence panel.
_TWIN_COLOR = "deepskyblue"
#: History fields only the plots use, which stay out of the JSON payload.
_PLOT_ONLY_FIELDS = ("forces", "stress")

#: Steps printed by default when a job ran for many ionic or electronic steps.
DEFAULT_TAIL = 30

_CSV_FIELDS = {
    "scf": ["step", "energy", "energy_change", "drho"],
    "relax": [
        "step", "energy", "energy_change",
        "max_force", "force_atom", "force_component", "converged",
    ],
    "cell-relax": [
        "step", "energy", "energy_change",
        "max_force", "force_atom", "force_component",
        "max_stress", "stress_component", "converged",
    ],
    "md": ["step", "energy", "potential", "kinetic", "temperature", "pressure"],
}


def _job_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"job directory does not exist: {value}")
    return path


def register_parser(subparsers) -> None:
    """Register the ``job monitor`` parser."""
    parser = subparsers.add_parser(
        "monitor", help="Show one update of one ABACUS job's state and progress."
    )
    parser.add_argument("job", type=_job_directory, metavar="JOB")
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Deprecated: the monitor prints one update and exits without waiting.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Deprecated: the monitor always prints one update and exits.",
    )
    parser.add_argument(
        "--relax", "--geometry",
        dest="relaxation",
        action="store_true",
        help="Kept for compatibility; the task type is detected from INPUT.",
    )
    parser.add_argument(
        "--scf-steps",
        action="store_true",
        help="Print every SCF iteration of an SCF calculation.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=DEFAULT_TAIL,
        metavar="N",
        help=(
            "Print only the last N steps of the step table, default: 30. "
            "Use 0 to print every step; --json, --csv and --plot keep the full history."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument(
        "--csv",
        type=Path,
        metavar="FILE",
        help="Write the step history to CSV as well as the text report.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=_AUTO_PLOT,
        default=None,
        metavar="FILE",
        help=(
            "Write a step-history plot as well as the text report. Without FILE "
            "the name follows the task type, such as monitor_relax.png below JOB."
        ),
    )
    parser.set_defaults(handler=run)


def _task_type(calculation: str) -> str:
    """Map an ABACUS ``calculation`` to the monitoring mode."""
    calculation = calculation.lower()
    if calculation in {"relax", "cell-relax", "cell_relax"}:
        return "cell-relax" if calculation.startswith("cell") else "relax"
    if calculation == "md":
        return "md"
    return "scf"


def _read_history(task: str, log: Path) -> list[dict[str, Any]]:
    """Read the step history matching one monitoring mode."""
    if task == "md":
        return read_md_history(log)
    if task in {"relax", "cell-relax"}:
        return read_relaxation_history(log)
    return read_scf_history(log)


def _finished_state(state: str, normal_end: bool) -> str:
    """Report a normally ended job that carries no convergence marker."""
    if state in {"invalid", "failed", "converged"}:
        return state
    return "finished" if normal_end else state


def _plot_path(value: Any, job: Path, task: str) -> Optional[Path]:
    """Resolve the ``--plot`` value into the file to write."""
    if value is None:
        return None
    if str(value) == _AUTO_PLOT:
        return job / _PLOT_FILENAMES[task]
    return Path(value)


def _format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.8g}"


def _format_energy(value: Any) -> str:
    """Format a total energy with eight decimals."""
    return "-" if value is None else f"{float(value):.8f}"


def _format_energy_change(value: Any) -> str:
    """Format an energy difference in scientific notation."""
    return "-" if value is None else f"{float(value):.6e}"


def _format_force_site(atom: Any, component: Any, label: Any = None) -> str:
    """Format the atom and Cartesian component of the largest force as ``H1x``."""
    if atom is None:
        return "-"
    site = str(label) if label else str(atom)
    return f"{site}{component}" if component else site


def _print_table(header: list[str], rows: list[list[str]]) -> None:
    """Print a header and its rows aligned on the widest cell of every column."""
    widths = [len(title) for title in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    for line in [header, *rows]:
        print("  ".join(cell.rjust(width) for cell, width in zip(line, widths)))


def _convergence_criteria(inputs: dict[str, Any], log: Optional[Path], task: str) -> dict[str, Any]:
    """Collect the convergence criteria of a relaxation calculation."""
    thresholds = read_convergence_thresholds(log) if log is not None else {}
    criteria: dict[str, Any] = {}
    force_threshold = inputs.get("force_thr_ev")
    if force_threshold is None:
        force_threshold = thresholds.get("force_thr_ev")
    if force_threshold is not None:
        criteria["force_thr_ev"] = force_threshold
    if task == "cell-relax":
        stress_threshold = inputs.get("stress_thr")
        if stress_threshold is None:
            stress_threshold = thresholds.get("stress_thr")
        if stress_threshold is not None:
            criteria["stress_thr"] = stress_threshold
    for key in ("relax_method", "relax_nmax", "relax_cg_thr"):
        value = inputs.get(key)
        if value is not None:
            criteria[key] = value
    return criteria


def _md_settings(inputs: dict[str, Any]) -> dict[str, Any]:
    """Collect the MD settings that decide which quantities are reported."""
    md_type = str(inputs.get("md_type", "")).strip().lower()
    settings: dict[str, Any] = {"md_type": md_type} if md_type else {}
    for key in ("md_nstep", "md_dt"):
        value = inputs.get(key)
        if value is not None:
            settings[key] = value
    if md_type in _THERMOSTAT_MD_TYPES:
        for key in ("md_tfirst", "md_tlast"):
            value = inputs.get(key)
            if value is not None:
                settings[key] = value
    if md_type in _PRESSURE_MD_TYPES:
        for key in ("md_pfirst", "md_plast"):
            value = inputs.get(key)
            if value is not None:
                settings[key] = value
    return settings


def _scf_criteria(inputs: dict[str, Any]) -> dict[str, Any]:
    """Collect the electronic threshold of an SCF calculation."""
    threshold = inputs.get("scf_thr")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return {}
    threshold_type = inputs.get("scf_thr_type")
    if threshold_type is None:
        # ABACUS compares a plane-wave run against the energy change and a
        # localized-orbital run against the density error.
        basis = str(inputs.get("basis_type", "pw")).strip().lower()
        threshold_type = 2 if basis == "lcao" else 1
    return {"scf_thr": float(threshold), "scf_thr_type": int(threshold_type)}


def _structure_moves(
    job: Path, inputs: dict[str, Any]
) -> Optional[list[tuple[bool, bool, bool]]]:
    """Read the movement constraints of the job's STRU, when it can be read."""
    stru_file = Path(str(inputs.get("stru_file", "STRU")))
    if not stru_file.is_absolute():
        stru_file = job / stru_file
    if not stru_file.is_file():
        return None
    try:
        structure = AbacusSTRU.read(stru_file)
    except (OSError, ValueError, TypeError):
        return None
    return [tuple(bool(flag) for flag in move) for move in structure.moves]


def _payload(
    job: Path,
    state: str,
    log: Optional[Path],
    calculation: str,
    task: str,
    inputs: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the report of one monitoring update."""
    payload: dict[str, Any] = {
        "job": str(job.absolute()),
        "state": state,
        "log": None if log is None else str(log),
        "calculation": calculation,
        "task": task,
        "energy_unit": ENERGY_UNIT,
        "force_unit": FORCE_UNIT,
        "stress_unit": STRESS_UNIT,
        "temperature_unit": TEMPERATURE_UNIT,
        "pressure_unit": PRESSURE_UNIT,
        "steps": [
            {key: value for key, value in item.items() if key not in _PLOT_ONLY_FIELDS}
            for item in history
        ],
    }
    if task in {"relax", "cell-relax"}:
        payload["criteria"] = _convergence_criteria(inputs, log, task)
    elif task == "scf":
        payload["criteria"] = _scf_criteria(inputs)
    elif task == "md":
        payload["md"] = _md_settings(inputs)
    return payload


def _print_header(payload: dict[str, Any]) -> None:
    print(f"job: {payload['job']}")
    print(f"state: {payload['state']}")
    if payload["log"]:
        print(f"log: {payload['log']}")
    print(f"calculation: {payload['task']}")


def _print_criteria(criteria: dict[str, Any]) -> None:
    print("convergence criteria:")
    if not criteria:
        print("  -")
        return
    if "force_thr_ev" in criteria:
        print(f"  force_thr_ev: {_format_metric(criteria['force_thr_ev'])} {FORCE_UNIT}")
    if "stress_thr" in criteria:
        print(f"  stress_thr: {_format_metric(criteria['stress_thr'])} {STRESS_UNIT}")
    for key in ("relax_method", "relax_nmax", "relax_cg_thr"):
        if key in criteria:
            print(f"  {key}: {criteria[key]}")


def _print_md_settings(settings: dict[str, Any]) -> None:
    if "md_type" in settings:
        print(f"md_type: {settings['md_type']}")
    for key in ("md_nstep", "md_dt", "md_tfirst", "md_tlast", "md_pfirst", "md_plast"):
        if key in settings:
            print(f"{key}: {settings[key]}")


def _print_scf_steps(history: list[dict[str, Any]]) -> None:
    rows = [
        [
            str(item["step"]),
            _format_energy(item["energy"]),
            _format_energy_change(item["energy_change"]),
            _format_metric(item["drho"]),
        ]
        for item in history
    ]
    _print_table(["step", "energy(eV)", "dE(eV)", "drho"], rows)


def _print_geometry_steps(task: str, history: list[dict[str, Any]]) -> None:
    header = [
        "step", "energy(eV)", "dE(eV)",
        "max_force(eV/A)", "force_atom/component",
    ]
    if task == "cell-relax":
        header += ["max_stress(kBar)", "stress_component"]
    header.append("converged")
    rows = []
    for item in history:
        row = [
            str(item["step"]),
            _format_energy(item["energy"]),
            _format_energy_change(item["energy_change"]),
            _format_metric(item["max_force"]),
            _format_force_site(
                item["force_atom"],
                item["force_component"],
                item.get("force_atom_label"),
            ),
        ]
        if task == "cell-relax":
            row += [
                _format_metric(item["max_stress"]),
                item["stress_component"] or "-",
            ]
        row.append("yes" if item["converged"] else "no")
        rows.append(row)
    _print_table(header, rows)


def _print_md_steps(history: list[dict[str, Any]]) -> None:
    header = [
        "step",
        "energy(eV)",
        "potential(eV)",
        "kinetic(eV)",
        f"temperature({TEMPERATURE_UNIT})",
        f"pressure({PRESSURE_UNIT})",
    ]
    rows = [
        [
            str(item["step"]),
            _format_energy(item["energy"]),
            _format_energy(item["potential"]),
            _format_energy(item["kinetic"]),
            _format_metric(item["temperature"]),
            _format_metric(item["pressure"]),
        ]
        for item in history
    ]
    _print_table(header, rows)


def _print_report(
    payload: dict[str, Any],
    *,
    scf_steps: bool,
    tail: int,
    plain_status: str,
) -> None:
    """Print one monitoring update."""
    task = payload["task"]
    if task == "scf" and not scf_steps:
        print(plain_status)
        return
    _print_header(payload)
    if task in {"relax", "cell-relax"}:
        _print_criteria(payload["criteria"])
    elif task == "md":
        _print_md_settings(payload["md"])
    history = payload["steps"]
    if not history:
        print("no steps found yet")
        return
    steps, hidden = _tail_steps(history, tail)
    if hidden:
        print(f"showing the last {len(steps)} of {len(history)} steps")
    if task == "scf":
        _print_scf_steps(steps)
    elif task == "md":
        _print_md_steps(steps)
    else:
        _print_geometry_steps(task, steps)


def _tail_steps(
    history: list[dict[str, Any]],
    tail: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return the steps to print and how many earlier steps they hide."""
    if tail <= 0 or len(history) <= tail:
        return history, 0
    return history[-tail:], len(history) - tail


def _write_csv(path: Path, task: str, history: list[dict[str, Any]]) -> None:
    """Write a step history as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _CSV_FIELDS[task]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field) for field in fields} for item in history)


def _metric_points(
    history: list[dict[str, Any]], key: str, *, absolute: bool = False
) -> list[tuple[int, float]]:
    """Return the step and value of every record that carries a metric."""
    points: list[tuple[int, float]] = []
    for item in history:
        value = item.get(key)
        if value is None:
            continue
        value = abs(float(value)) if absolute else float(value)
        points.append((item["step"], value))
    return points


def _plot_metric(
    axis,
    history: list[dict[str, Any]],
    key: str,
    *,
    label: str,
    ylabel: str,
    threshold: Any = None,
    threshold_label: str = "",
    marker: str = "s",
    absolute: bool = False,
) -> None:
    """Draw one convergence curve, on a log axis while its values stay positive."""
    points = _metric_points(history, key, absolute=absolute)
    if any(value > 0.0 for _, value in points):
        points = [(step, value) for step, value in points if value > 0.0]
        axis.set_yscale("log")
    if points:
        axis.plot(
            [step for step, _ in points],
            [value for _, value in points],
            label=label,
            **{**_CURVE_STYLE, "marker": marker},
        )
    if threshold is not None:
        axis.axhline(float(threshold), label=threshold_label, **_THRESHOLD_STYLE)
    axis.set_ylabel(ylabel)


def _unconverged_counts(
    history: list[dict[str, Any]],
    key: str,
    threshold: Any,
    moves: Optional[list[tuple[bool, bool, bool]]],
) -> list[tuple[int, int]]:
    """Count the force or stress components beyond the convergence threshold.

    Args:
        history: Records of :func:`read_relaxation_history`.
        key: ``forces`` for the per-atom force vectors, ``stress`` for a tensor.
        threshold: Convergence threshold in the unit of the metric.
        moves: Movement flags per atom, which mask the fixed atoms of ``forces``.

    Returns:
        One ``(step, count)`` pair per step that carries the raw data.
    """
    if threshold is None:
        return []
    limit = float(threshold)
    counts: list[tuple[int, int]] = []
    for item in history:
        values = item.get(key)
        if not values:
            continue
        total = 0
        for index, row in enumerate(values):
            for axis, value in enumerate(row):
                if abs(float(value)) <= limit:
                    continue
                if moves is not None and key == "forces":
                    flags = moves[index] if index < len(moves) else (True, True, True)
                    if not flags[axis]:
                        continue
                total += 1
        counts.append((item["step"], total))
    return counts


def _plot_twin(
    axis,
    points: list[tuple[int, float]],
    label: str,
    *,
    counts: bool = False,
) -> Optional[Any]:
    """Draw a secondary quantity on the twin axis of a convergence panel."""
    if not points:
        return None
    twin = axis.twinx()
    twin.plot(
        [step for step, _ in points],
        [value for _, value in points],
        color=_TWIN_COLOR,
        linestyle="-",
        marker="o",
        markersize=4,
        label=label,
    )
    if counts:
        twin.set_ylim(bottom=-0.5)
    twin.set_ylabel(label)
    return twin


def _merge_legend(axis, twin: Optional[Any]) -> None:
    """Draw one legend that also lists the curves of a twin axis."""
    handles, labels = axis.get_legend_handles_labels()
    if twin is not None:
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        handles = handles + twin_handles
        labels = labels + twin_labels
    if handles:
        axis.legend(handles, labels, loc="best")


def _share_x(axes, history: list[dict[str, Any]], xlabel: str) -> None:
    """Label the shared x axis and limit it to the steps of the history."""
    axes[-1].set_xlabel(xlabel)
    steps = [item["step"] for item in history]
    if len(steps) > 1:
        for axis in axes:
            axis.set_xlim(min(steps), max(steps))


def _plot_geometry(
    axes,
    task: str,
    history: list[dict[str, Any]],
    criteria: dict[str, Any],
    moves: Optional[list[tuple[bool, bool, bool]]],
) -> None:
    """Draw the energy, force and stress convergence of a relaxation."""
    _plot_metric(
        axes[0],
        history,
        "energy",
        label="Total energy",
        ylabel=f"Total energy ({ENERGY_UNIT})",
    )
    axes[0].ticklabel_format(useOffset=False, axis="y")
    _merge_legend(axes[0], None)

    force_threshold = criteria.get("force_thr_ev")
    _plot_metric(
        axes[1],
        history,
        "max_force",
        label="Max force",
        ylabel=f"Max force ({FORCE_UNIT})",
        threshold=force_threshold,
        threshold_label="Force threshold",
    )
    force_twin = _plot_twin(
        axes[1],
        _unconverged_counts(history, "forces", force_threshold, moves),
        "Unconverged components",
        counts=True,
    )
    _merge_legend(axes[1], force_twin)

    if task == "cell-relax":
        stress_threshold = criteria.get("stress_thr")
        _plot_metric(
            axes[2],
            history,
            "max_stress",
            label="Max stress",
            ylabel=f"Max stress ({STRESS_UNIT})",
            threshold=stress_threshold,
            threshold_label="Stress threshold",
            marker="D",
        )
        stress_twin = _plot_twin(
            axes[2],
            _unconverged_counts(history, "stress", stress_threshold, None),
            "Unconverged components",
            counts=True,
        )
        _merge_legend(axes[2], stress_twin)

    _share_x(axes, history, "Step")


def _plot_scf(axes, history: list[dict[str, Any]], criteria: dict[str, Any]) -> None:
    """Draw the energy and convergence history of an SCF run."""
    _plot_metric(
        axes[0],
        history,
        "energy",
        label="Total energy",
        ylabel=f"Total energy ({ENERGY_UNIT})",
    )
    axes[0].ticklabel_format(useOffset=False, axis="y")
    _merge_legend(axes[0], None)

    threshold = criteria.get("scf_thr")
    # ``scf_thr_type`` decides whether the threshold belongs to the energy
    # change (1) or to the density error (2).
    density_threshold = int(criteria.get("scf_thr_type", 1)) == 2
    _plot_metric(
        axes[1],
        history,
        "energy_change",
        label="|Energy change|",
        ylabel=f"|Energy change| ({ENERGY_UNIT})",
        absolute=True,
        threshold=None if density_threshold else threshold,
        threshold_label="scf_thr",
    )

    drhos = [
        (step, value) for step, value in _metric_points(history, "drho") if value > 0.0
    ]
    drho_axis = None
    if drhos:
        drho_axis = _plot_twin(axes[1], drhos, "Density error")
        drho_axis.set_yscale("log")
        if density_threshold and threshold is not None:
            drho_axis.axhline(float(threshold), label="scf_thr", **_THRESHOLD_STYLE)
    _merge_legend(axes[1], drho_axis)
    _share_x(axes, history, "SCF iteration")


def _plot_md(axes, history: list[dict[str, Any]]) -> None:
    """Draw the energy and temperature/pressure history of an MD run."""
    styles = (
        ("energy", "Total", {"linewidth": 2.0, "marker": "o"}),
        ("potential", "Potential", {"linestyle": "--", "marker": "s", "linewidth": 1.2}),
        ("kinetic", "Kinetic", {"marker": "^"}),
    )
    for key, label, style in styles:
        points = [(item["step"], item[key]) for item in history if item[key] is not None]
        if points:
            axes[0].plot(
                [step for step, _ in points],
                [value for _, value in points],
                label=label,
                **style,
            )
    axes[0].set_ylabel(f"Energy ({ENERGY_UNIT})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    temperatures = [
        (item["step"], item["temperature"])
        for item in history
        if item["temperature"] is not None
    ]
    if temperatures:
        axes[1].plot(
            [step for step, _ in temperatures],
            [value for _, value in temperatures],
            "o-",
            color="tab:orange",
        )
    axes[1].set_ylabel(f"Temperature ({TEMPERATURE_UNIT})")
    axes[1].grid(True, alpha=0.3)

    pressures = [
        (item["step"], item["pressure"])
        for item in history
        if item["pressure"] is not None
    ]
    if pressures:
        axes[2].plot(
            [step for step, _ in pressures],
            [value for _, value in pressures],
            "s-",
            color="tab:blue",
        )
    axes[2].set_xlabel("MD step")
    axes[2].set_ylabel(f"Pressure ({PRESSURE_UNIT})")
    axes[2].grid(True, alpha=0.3)


def _write_plot(
    path: Path,
    task: str,
    history: list[dict[str, Any]],
    criteria: dict[str, Any],
    moves: Optional[list[tuple[bool, bool, bool]]] = None,
) -> None:
    """Save the step-history plot of one task type."""
    if not history:
        raise RuntimeError("cannot plot the step history: no steps found")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 3 if task in {"cell-relax", "md"} else 2
    figure, axes = plt.subplots(rows, 1, figsize=(8, 4 * rows), sharex=True)
    if task == "md":
        _plot_md(axes, history)
    elif task == "scf":
        figure.suptitle("SCF convergence")
        _plot_scf(axes, history, criteria)
    else:
        figure.suptitle("Geometry relaxation convergence")
        _plot_geometry(axes, task, history, criteria, moves)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> int:
    """Print one update of an ABACUS job's state and progress."""
    job = Path(args.job)
    validation = validate_job(job)
    task = _task_type(str(validation.inputs.get("calculation", "scf")))
    plot = _plot_path(args.plot, job, task)
    moves = (
        _structure_moves(job, validation.inputs)
        if plot is not None and task in {"relax", "cell-relax"}
        else None
    )
    status = status_job(job, validation)
    log = find_job_log(job, ionic=task in {"relax", "cell-relax", "md"})
    state = _finished_state(
        status.state,
        read_normal_end(log) if log is not None else False,
    )
    history = _read_history(task, log) if log is not None else []
    payload = _payload(
        job,
        state,
        log,
        str(validation.inputs.get("calculation", "scf")).lower(),
        task,
        validation.inputs,
        history,
    )

    if args.csv:
        _write_csv(args.csv, task, history)
    if plot is not None:
        _write_plot(plot, task, history, payload.get("criteria", {}), moves)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if state in {"invalid", "failed"} else 0

    progress = " ".join(
        f"{key}={value}"
        for key, value in status.progress.items()
        if value is not None
    )
    _print_report(
        payload,
        scf_steps=args.scf_steps,
        tail=args.tail,
        plain_status=f"{status.state}: {progress}".rstrip(),
    )
    if args.csv:
        print(f"step history: {args.csv}")
    if plot is not None:
        print(f"step history plot: {plot}")

    return 1 if state in {"invalid", "failed"} else 0
