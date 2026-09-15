"""Magnetic exchange coupling constants from the four-state method.

The workflow prepares one reference calculation and, for every requested
magnetic pair and tilt angle, three magnetization configurations: only the
first moment tilted, only the second one tilted, and both moments tilted by
half the angle. The three configurations share the same pair angle, so the
four-state energy difference

    dE = (E_both - E_original) - (E_atom1 - E_original) - (E_atom2 - E_original)

is fitted against ``1 - cos(theta)``. The slope of that line is the exchange
coupling constant ``J``, which is positive for antiferromagnetic coupling.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.core.submission import generate_workflow_submission
from abacustools.data.exchange import (
    CASES,
    ExchangeError,
    angle_between,
    fit_exchange_coupling,
    four_state_energy,
    moment_vector,
    tilted_moments,
)
from abacustools.data.versions import default_version

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_job_structure,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)


_ROOT = "magj"
_ORIGINAL_TASK = f"{_ROOT}/original"
_DEFAULT_INFO_FILE = "magj.txt"
_SCF_THRESHOLD = 1.0e-7


def _format_angle(value: float) -> str:
    """Format a tilt angle compactly for a task name."""
    return f"{value:g}"


def _validate_step(step: float) -> None:
    """Validate the tilt step in degrees."""
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("the tilt step must be a positive number of degrees")


def _validate_number(number: int) -> None:
    """Validate the number of tilt angles."""
    if isinstance(number, bool) or int(number) != number or number < 2:
        raise ValueError("at least two tilt angles are required for a linear fit")


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the exchange coupling preparation stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="ABACUS job directory with a noncollinear INPUT and the magnetic structure.",
    )
    parser.add_argument(
        "-p", "--pair",
        dest="pairs",
        action="append",
        nargs=4,
        metavar=("LABEL1", "INDEX1", "LABEL2", "INDEX2"),
        default=None,
        help=(
            "Magnetic pair as label and one-based index within that label, "
            "such as '--pair Fe 1 Fe 2'; repeat the option for several pairs."
        ),
    )
    parser.add_argument(
        "-f", "--file",
        dest="info_file",
        default=_DEFAULT_INFO_FILE,
        help=(
            "File listing one magnetic pair per line, relative to JOB by "
            f"default; used when --pair is omitted, default: {_DEFAULT_INFO_FILE}."
        ),
    )
    parser.add_argument(
        "-s", "--step",
        type=float,
        default=1.0,
        help="Tilt step in degrees, default: 1.0.",
    )
    parser.add_argument(
        "-n", "--number",
        type=int,
        default=5,
        help="Number of tilt angles, from STEP to NUMBER*STEP, default: 5.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Replace existing generated exchange coupling directories.",
    )
    submission = parser.add_mutually_exclusive_group()
    submission.add_argument(
        "--submit-script",
        dest="generate_scripts",
        action="store_true",
        help="Generate configured task and workflow submission scripts.",
    )
    submission.add_argument(
        "--no-submit-script",
        dest="generate_scripts",
        action="store_false",
        help="Do not generate submission scripts, overriding the config default.",
    )
    parser.set_defaults(generate_scripts=None)
    parser.add_argument(
        "--submission-type",
        "--submit-type",
        dest="submission_type",
        help="Submission template type from the config, such as local, slurm, pbs, or lsf.",
    )
    parser.add_argument(
        "--abacus-command",
        help="ABACUS command used in generated scripts; otherwise use the config default.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the exchange coupling postprocessing stage."""
    parser.add_argument(
        "-j", "--job",
        type=Path,
        required=True,
        help="Directory containing the prepared exchange coupling calculations.",
    )
    parser.add_argument(
        "-v", "--version",
        default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-o", "--output",
        default="exchange_results.json",
        help="Output JSON filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--plot",
        default="exchange_fit.png",
        help="Output fit plot filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help=(
            "Fit calculations whose SCF stopped without reaching scf_thr; such a "
            "state is reported instead of stopping the postprocessing."
        ),
    )


def _require_noncollinear(inputs: dict[str, Any]) -> None:
    """Require the noncollinear magnetization that the four-state method needs."""
    from abacustools.io.abacus import IsEnabled

    value = inputs.get("nspin", 1)
    try:
        nspin = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"cannot read nspin from INPUT: {value!r}") from error
    # ABACUS raises nspin to 4 by itself when noncolin or lspinorb is enabled.
    if nspin == 4 or IsEnabled(inputs.get("noncolin")) or IsEnabled(inputs.get("lspinorb")):
        return
    raise RuntimeError(
        "the four-state method needs vector magnetic moments, so INPUT must set "
        f"nspin 4, noncolin or lspinorb; found nspin {nspin}"
    )


def _read_pairs_file(path: Path) -> list[tuple[str, int, str, int]]:
    """Read the reference-style ``magj.txt`` list of magnetic pairs.

    Every non-empty line holds ``LABEL1 INDEX1 LABEL2 INDEX2``, where the
    indices are one-based within the atoms of that label.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"could not find the magnetic pair file: {path}; create it or pass "
            "--pair LABEL1 INDEX1 LABEL2 INDEX2"
        )
    pairs = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 4:
            raise ValueError(f"invalid magnetic pair on line {line_number}: {line}")
        try:
            ordinal1 = int(fields[1])
            ordinal2 = int(fields[3])
        except ValueError as error:
            raise ValueError(f"invalid atom index on line {line_number}: {line}") from error
        if ordinal1 < 1 or ordinal2 < 1:
            raise ValueError(f"atom indices are one-based on line {line_number}: {line}")
        pairs.append((fields[0], ordinal1, fields[2], ordinal2))
    if not pairs:
        raise ValueError(f"the magnetic pair file is empty: {path}")
    return pairs


def _resolve_atom(structure, label: str, ordinal: int) -> int:
    """Resolve one label and ordinal to a zero-based atom index."""
    matches = [
        index
        for index, atom in enumerate(structure.atoms)
        if atom.label == label or atom.element == label
    ]
    if not matches:
        raise ValueError(f"no atom matches the label {label!r} in the structure")
    if ordinal > len(matches):
        raise ValueError(
            f"{label} has only {len(matches)} atoms, but index {ordinal} was requested"
        )
    return matches[ordinal - 1]


def _atom_moment(structure, index: int) -> np.ndarray:
    """Return the magnetic moment of one atom as a vector."""
    atom = structure.atoms[index]
    try:
        value = atom.atommag
    except (TypeError, ValueError) as error:
        raise ExchangeError(
            f"cannot read the magnetic moment of atom {index + 1} ({atom.label}): {error}"
        ) from error
    moment = moment_vector(value)
    if float(np.linalg.norm(moment)) <= 0.0:
        raise ExchangeError(
            f"atom {index + 1} ({atom.label}) has a zero magnetic moment; the "
            "four-state method needs a moment on both atoms of the pair"
        )
    return moment


def _requested_pairs(args: argparse.Namespace, job: Path) -> list[tuple[str, int, str, int]]:
    """Return the magnetic pairs from the command line or from the info file."""
    if getattr(args, "pairs", None):
        requested = []
        for values in args.pairs:
            try:
                ordinal1 = int(values[1])
                ordinal2 = int(values[3])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid atom index in --pair {values}") from error
            if ordinal1 < 1 or ordinal2 < 1:
                raise ValueError(f"atom indices are one-based in --pair {values}")
            requested.append((str(values[0]), ordinal1, str(values[2]), ordinal2))
        return requested

    path = Path(args.info_file)
    if not path.is_absolute():
        path = job / path
    return _read_pairs_file(path)


def _collect_pairs(structure, requested) -> list[dict[str, Any]]:
    """Resolve pairs to atoms, drop duplicates and read their moments."""
    pairs = []
    seen: set[tuple[int, int]] = set()
    for label1, ordinal1, label2, ordinal2 in requested:
        atom1 = _resolve_atom(structure, label1, ordinal1)
        atom2 = _resolve_atom(structure, label2, ordinal2)
        if atom1 == atom2:
            raise ValueError(
                f"the pair {label1} {ordinal1} - {label2} {ordinal2} resolves to the "
                "same atom, which cannot be coupled to itself"
            )
        key = (min(atom1, atom2), max(atom1, atom2))
        if key in seen:
            print(
                f"  skipping duplicate pair {label1} {ordinal1} - "
                f"{label2} {ordinal2} (atoms {key[0] + 1} and {key[1] + 1})"
            )
            continue
        seen.add(key)

        if atom1 <= atom2:
            first, second = (label1, ordinal1, atom1), (label2, ordinal2, atom2)
        else:
            first, second = (label2, ordinal2, atom2), (label1, ordinal1, atom1)
        moment1 = _atom_moment(structure, first[2])
        moment2 = _atom_moment(structure, second[2])
        pairs.append(
            {
                "name": f"{first[0]}{first[1]}_{second[0]}{second[1]}",
                "label1": first[0],
                "ordinal1": first[1],
                "atom1": first[2] + 1,
                "label2": second[0],
                "ordinal2": second[1],
                "atom2": second[2] + 1,
                "moment1": moment1.tolist(),
                "moment2": moment2.tolist(),
                "reference_angle": angle_between(moment1, moment2),
            }
        )
    return pairs


def _case_metadata(pairs: list[dict[str, Any]], step: float, number: int) -> list[dict[str, Any]]:
    """Build the magnetic moments of every generated four-state calculation."""
    cases = []
    for pair in pairs:
        for index in range(1, number + 1):
            tilt = float(round(index * step, 10))
            for case in CASES:
                moment1, moment2 = tilted_moments(
                    pair["moment1"], pair["moment2"], tilt, case
                )
                cases.append(
                    {
                        "task": f"{_ROOT}/{pair['name']}_angle{_format_angle(tilt)}_{case}",
                        "pair": pair["name"],
                        "tilt": tilt,
                        "case": case,
                        "atom1": pair["atom1"],
                        "atom2": pair["atom2"],
                        "moment1": [float(value) for value in moment1],
                        "moment2": [float(value) for value in moment2],
                        "pair_angle": angle_between(moment1, moment2),
                    }
                )
    return cases


def _scf_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return the INPUT of the generated single-point calculations."""
    prepared = deepcopy(inputs)
    prepared["calculation"] = "scf"
    try:
        threshold = float(prepared.get("scf_thr", _SCF_THRESHOLD))
    except (TypeError, ValueError):
        threshold = _SCF_THRESHOLD
    if not np.isfinite(threshold) or threshold > _SCF_THRESHOLD:
        prepared["scf_thr"] = _SCF_THRESHOLD
    return prepared


def prepare(args: argparse.Namespace) -> int:
    """Prepare the reference and four-state calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")
    _validate_step(args.step)
    _validate_number(args.number)

    inputs, stru_filename, structure = read_job_structure(job)
    _require_noncollinear(inputs)
    requested = _requested_pairs(args, job)
    pairs = _collect_pairs(structure, requested)
    if not pairs:
        raise ValueError("no magnetic pair is left to calculate")

    cases = _case_metadata(pairs, args.step, args.number)
    names = [_ORIGINAL_TASK] + [case["task"] for case in cases]
    clear_generated_jobs(job, [_ROOT], override=args.override)

    scf_inputs = _scf_inputs(inputs)
    requested_threshold = inputs.get("scf_thr")
    if "scf_thr" in scf_inputs and scf_inputs["scf_thr"] != requested_threshold:
        print(
            "  scf_thr tightened to "
            f"{scf_inputs['scf_thr']:g} because the four-state fit compares energies"
        )
    kpoint = kpoint_filename(job, inputs)
    write_abacus_job(
        scf_inputs,
        structure,
        job,
        job / _ORIGINAL_TASK,
        stru_filename=stru_filename,
        kpoint=kpoint,
    )
    print(f"  prepared {_ORIGINAL_TASK}")

    for case in cases:
        tilted = deepcopy(structure)
        tilted.atoms[case["atom1"] - 1].set_atommag(case["moment1"])
        tilted.atoms[case["atom2"] - 1].set_atommag(case["moment2"])
        write_abacus_job(
            scf_inputs,
            tilted,
            job,
            job / case["task"],
            stru_filename=stru_filename,
            kpoint=kpoint,
        )
    print(f"  prepared {len(cases)} four-state calculations")

    submission = generate_workflow_submission(
        job,
        _ROOT,
        names,
        submission_type=getattr(args, "submission_type", None),
        generate=getattr(args, "generate_scripts", None),
        abacus_command=getattr(args, "abacus_command", None),
    )
    if submission is not None:
        print(f"  submission type: {submission['type']}")
        print(f"  workflow script: {submission['workflow_script']}")

    manifest = {
        "tasks": names,
        "original_task": _ORIGINAL_TASK,
        "step": float(args.step),
        "number": int(args.number),
        "pairs": pairs,
        "cases": cases,
    }
    if submission is not None:
        manifest["submission"] = submission
    write_manifest(job, _ROOT, **manifest)

    print(f"  job: {job}")
    tilts = ", ".join(_format_angle(index * args.step) for index in range(1, args.number + 1))
    for pair in pairs:
        print(
            f"  pair {pair['name']} (atoms {pair['atom1']} and {pair['atom2']}): "
            f"reference angle {pair['reference_angle']:.2f} deg"
        )
        if pair["reference_angle"] + args.number * args.step > 180.0:
            print(
                "  warning: the largest tilt passes a pair angle of 180 degrees; "
                "consider a smaller step or number"
            )
    print(f"  tilt angles: {tilts} deg")
    print(f"  generated calculations: {len(names)}")
    return 0


def _read_state(job: Path, version: str) -> dict[str, Any]:
    """Read the total energy and the SCF convergence flag of one state.

    The four-state method compares energies, so a calculation that stopped
    without reaching ``scf_thr`` is still usable when its energy stopped
    moving; postprocessing reports those states and only refuses them unless
    ``--allow-unconverged`` asks for the fit.
    """
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(
        str(job),
        param_names=["energy", "converged"],
        version=version,
    )
    energy = result["energy"]
    if energy is None or not np.isfinite(energy):
        raise RuntimeError(f"energy was not found in the output: {job}")
    return {"energy": float(energy), "converged": bool(result["converged"])}


def _manifest_pairs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and return the atom pairs of a workflow manifest."""
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise RuntimeError("the exchange workflow manifest holds no atom pairs")
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("name"), str):
            raise RuntimeError("the exchange workflow manifest has an invalid atom pair")
    return pairs


def _manifest_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and return the four-state cases of a workflow manifest."""
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("the exchange workflow manifest holds no four-state calculations")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("task"), str):
            raise RuntimeError("the exchange workflow manifest has an invalid four-state case")
        if case.get("case") not in CASES:
            raise RuntimeError("the exchange workflow manifest has an invalid four-state case")
    return cases


def _pair_points(
    pair: dict[str, Any],
    cases: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    original_energy: float,
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    """Collect the four-state points of one magnetic pair."""
    grouped: dict[float, dict[str, dict[str, Any]]] = {}
    for case in cases:
        if case.get("pair") != pair["name"]:
            continue
        grouped.setdefault(float(case["tilt"]), {})[str(case["case"])] = case
    if not grouped:
        raise RuntimeError(f"the workflow manifest holds no four-state cases for {pair['name']}")

    angles: list[float] = []
    delta_energies: list[float] = []
    points: list[dict[str, Any]] = []
    for tilt in sorted(grouped):
        grouped_states = grouped[tilt]
        missing = [case for case in CASES if case not in grouped_states]
        if missing:
            raise RuntimeError(
                f"pair {pair['name']} is missing the {', '.join(missing)} case "
                f"at a tilt of {tilt:g} deg"
            )
        state_energies = {
            case: states[grouped_states[case]["task"]]["energy"] for case in CASES
        }
        converged = all(
            states[grouped_states[case]["task"]]["converged"] for case in CASES
        )
        angle = float(
            np.mean(
                [
                    angle_between(
                        grouped_states[case]["moment1"], grouped_states[case]["moment2"]
                    )
                    for case in CASES
                ]
            )
        )
        delta_energy = four_state_energy(
            original_energy,
            state_energies["atom1"],
            state_energies["atom2"],
            state_energies["both"],
        )
        angles.append(angle)
        delta_energies.append(delta_energy)
        points.append(
            {
                "tilt": tilt,
                "pair_angle": angle,
                "cos_angle": float(np.cos(np.radians(angle))),
                "x": float(1.0 - np.cos(np.radians(angle))),
                "energy_original_ev": original_energy,
                "energy_atom1_ev": state_energies["atom1"],
                "energy_atom2_ev": state_energies["atom2"],
                "energy_both_ev": state_energies["both"],
                "delta_energy_ev": delta_energy,
                "converged": converged,
            }
        )
    return angles, delta_energies, points


def _plot(reports: list[dict[str, Any]], path: Path) -> None:
    """Plot the four-state energy differences together with their fits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for report in reports:
        coupling = report["coupling"]
        x = np.asarray(coupling["x"], dtype=float)
        energies = np.asarray(coupling["delta_energy_ev"], dtype=float)
        axis.plot(
            x,
            energies,
            "o",
            label=f"{report['name']}: J = {coupling['j_mev']:.2f} meV",
        )
        line_x = np.linspace(min(0.0, float(x.min())), max(1.0e-6, float(x.max())), 100)
        axis.plot(
            line_x,
            coupling["slope_ev"] * line_x + coupling["intercept_ev"],
            "-",
            linewidth=1.2,
        )
    axis.set_xlabel("1 - cos(theta)")
    axis.set_ylabel("Delta E (eV)")
    axis.set_title("Four-state exchange coupling")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def postprocess(args: argparse.Namespace) -> int:
    """Fit the exchange coupling constants and write the report and plot."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    manifest = read_manifest(job, _ROOT, required_tasks=[_ORIGINAL_TASK])
    pairs = _manifest_pairs(manifest)
    cases = _manifest_cases(manifest)
    original_task = str(manifest.get("original_task", _ORIGINAL_TASK))
    read_manifest(job, _ROOT, [original_task] + [case["task"] for case in cases])

    original_state = _read_state(job / original_task, args.version)
    states = {
        case["task"]: _read_state(job / case["task"], args.version) for case in cases
    }
    unconverged = sorted(
        task for task, state in states.items() if not state["converged"]
    )
    if unconverged and not getattr(args, "allow_unconverged", False):
        raise RuntimeError(
            "SCF calculations did not converge: "
            f"{', '.join(unconverged)}; pass --allow-unconverged to fit them anyway"
        )
    if unconverged:
        print(f"  warning: unconverged SCF calculations: {', '.join(unconverged)}")
    original_energy = original_state["energy"]

    reports = []
    for pair in pairs:
        try:
            moment1 = pair["moment1"]
            moment2 = pair["moment2"]
        except KeyError as error:
            raise RuntimeError(
                "the exchange workflow manifest has an incomplete atom pair"
            ) from error
        reference_angle = angle_between(moment1, moment2)
        angles, delta_energies, points = _pair_points(pair, cases, states, original_energy)
        fit = fit_exchange_coupling(angles, delta_energies)
        reports.append(
            {
                "name": pair["name"],
                "atom1": int(pair["atom1"]),
                "atom2": int(pair["atom2"]),
                "label1": pair.get("label1"),
                "ordinal1": pair.get("ordinal1"),
                "label2": pair.get("label2"),
                "ordinal2": pair.get("ordinal2"),
                "reference_angle": reference_angle,
                "reference_moment1": [float(value) for value in moment1],
                "reference_moment2": [float(value) for value in moment2],
                "points": points,
                "coupling": fit.to_dict(),
            }
        )

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "workflow": _ROOT,
        "step": manifest.get("step"),
        "number": manifest.get("number"),
        "original_energy_ev": original_energy,
        "unconverged_tasks": unconverged,
        "convention": (
            "dE = J (1 - cos(theta)) with the reference magnetic moment magnitudes; "
            "J > 0 is antiferromagnetic"
        ),
        "pairs": reports,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot = Path(args.plot)
    if not plot.is_absolute():
        plot = job / plot
    plot.parent.mkdir(parents=True, exist_ok=True)
    _plot(reports, plot)

    for item in reports:
        coupling = item["coupling"]
        print(
            f"  {item['name']} (atoms {item['atom1']} and {item['atom2']}): "
            f"reference angle {item['reference_angle']:.2f} deg"
        )
        print(
            f"    J = {coupling['j_mev']:.4f} meV, R^2 = {coupling['r_squared']:.6f}, "
            f"rms residual = {coupling['rms_residual_ev']:.3e} eV, "
            f"points = {coupling['points']}"
        )
    print(f"  results: {output}")
    print(f"  plot: {plot}")
    return 0


def register_parser(subparsers) -> None:
    """Register the exchange coupling preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "exchange",
        "Calculate magnetic exchange coupling constants with the four-state method.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
        aliases=["magj"],
    )
