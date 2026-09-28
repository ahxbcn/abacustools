"""The ``abacustools workflow eos`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from abacustools.data.eos import EosFit, fit_birch_murnaghan
from abacustools.data.versions import default_version
from abacustools.core.job import read_job_structure

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_manifest,
    register_stages,
    write_abacus_job,
    write_manifest,
)


def _volume_scales(start: float, end: float, step: float) -> list[float]:
    """Build the list of volume scale factors from a start/end/step range."""
    for name, value in (("start", start), ("end", end), ("step", step)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if start <= 0 or end <= 0:
        raise ValueError("volume scales must be positive")
    if start > end:
        raise ValueError("start must not be larger than end")
    if step <= 0:
        raise ValueError("step must be positive")

    count = int(round((end - start) / step)) + 1
    scales = [round(start + index * step, 6) for index in range(count)]
    if scales[-1] > end + 1e-9:
        scales.pop()
    return scales


def _task_name(scale: float) -> str:
    return f"eos_{scale:.3f}"


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the EOS preparation stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory used to prepare the EOS calculations.",
    )
    parser.add_argument(
        "--start", type=float, default=0.90,
        help="Smallest volume scale, default: 0.90.",
    )
    parser.add_argument(
        "--end", type=float, default=1.10,
        help="Largest volume scale, default: 1.10.",
    )
    parser.add_argument(
        "--step", type=float, default=0.025,
        help="Volume-scale step, default: 0.025.",
    )
    parser.add_argument(
        "--relax", action="store_true",
        help="Relax the ions at each fixed volume instead of running SCF.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace existing generated EOS directories.",
    )


def _register_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the EOS postprocessing stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory containing the prepared EOS calculations.",
    )
    parser.add_argument(
        "-v", "--version", default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-o", "--output", default="eos_results.json",
        help="Output JSON filename, relative to JOB by default.",
    )
    parser.add_argument(
        "--plot", default="eos.png",
        help="Output plot filename, relative to JOB by default.",
    )


def prepare(args: argparse.Namespace) -> int:
    """Prepare volume-scaled SCF (or fixed-volume relax) calculations."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    scales = _volume_scales(args.start, args.end, args.step)
    if len(scales) < 4:
        raise ValueError("at least four volume points are required for an EOS fit")

    inputs, stru_filename, structure = read_job_structure(job)
    cell = np.asarray(structure.cell, dtype=float)
    if cell.shape != (3, 3):
        raise RuntimeError(f"invalid cell in structure: {stru_filename}")
    equilibrium_volume = abs(float(np.linalg.det(cell)))

    eos_inputs = deepcopy(inputs)
    eos_inputs["calculation"] = "relax" if args.relax else "scf"
    kpoint_file = kpoint_filename(job, inputs)

    names = [_task_name(scale) for scale in scales]
    clear_generated_jobs(job, names, override=args.override)

    print(f"  job: {job}")
    print(f"  equilibrium volume: {equilibrium_volume:.6f} Ang^3")
    print(f"  calculation: {eos_inputs['calculation']}")
    print(f"  volume scales: {', '.join(f'{scale:.3f}' for scale in scales)}")

    points = []
    for scale, name in zip(scales, names):
        scaled = deepcopy(structure)
        scaled.cell = (cell * scale ** (1.0 / 3.0)).tolist()
        write_abacus_job(
            eos_inputs,
            scaled,
            job,
            job / name,
            stru_filename=stru_filename,
            kpoint=kpoint_file,
        )
        points.append({"name": name, "scale": scale, "volume": equilibrium_volume * scale})
        print(f"  prepared {name} (V = {equilibrium_volume * scale:.6f} Ang^3)")

    write_manifest(
        job,
        "eos",
        tasks=names,
        points=points,
        equilibrium_volume=equilibrium_volume,
        relax=bool(args.relax),
    )
    return 0


def _read_energy(job: Path, version: str) -> float:
    """Read one converged total energy in eV."""
    from abacustools.data.abacus_result import get_result_from_job

    result = get_result_from_job(
        str(job),
        param_names=["energy", "converged"],
        version=version,
    )
    if not result["converged"]:
        raise RuntimeError(f"SCF calculation did not converge: {job}")
    energy = result["energy"]
    if energy is None or not np.isfinite(energy):
        raise RuntimeError(f"energy was not found in the output: {job}")
    return float(energy)


def _plot(fit: EosFit, path: Path) -> None:
    """Plot the calculated points together with the fitted EOS curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(fit.volumes, fit.energies, "o", label="calculated")
    axis.plot(fit.fit_volumes, fit.fit_energies, "-", linewidth=1.2, label="Birch-Murnaghan fit")
    axis.axvline(fit.volume, color="gray", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Volume (Ang^3)")
    axis.set_ylabel("Energy (eV)")
    axis.set_title(
        f"Equation of state (V0 = {fit.volume:.3f} Ang^3, B0 = {fit.bulk_modulus:.2f} GPa)"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def postprocess(args: argparse.Namespace) -> int:
    """Fit the equation of state and write the report and plot."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise RuntimeError(f"job directory does not exist: {job}")

    manifest = read_manifest(job, "eos", required_tasks=[])
    points = manifest.get("points", [])
    if len(points) < 4:
        raise RuntimeError("the EOS manifest does not contain enough volume points")

    volumes = []
    energies = []
    collected = []
    for point in points:
        name = point["name"]
        volume = float(point["volume"])
        energy = _read_energy(job / name, args.version)
        volumes.append(volume)
        energies.append(energy)
        collected.append(
            {"name": name, "scale": point.get("scale"), "volume": volume, "energy": energy}
        )
        print(f"  {name}: V = {volume:.6f} Ang^3, E = {energy:.8f} eV")

    fit = fit_birch_murnaghan(volumes, energies)

    output = Path(args.output)
    if not output.is_absolute():
        output = job / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"workflow": "eos", "points": collected, **fit.to_dict()}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    plot = Path(args.plot)
    if not plot.is_absolute():
        plot = job / plot
    plot.parent.mkdir(parents=True, exist_ok=True)
    _plot(fit, plot)

    print(f"  equilibrium volume: {fit.volume:.6f} Ang^3")
    print(f"  equilibrium energy: {fit.energy:.8f} eV")
    print(f"  bulk modulus: {fit.bulk_modulus:.4f} GPa")
    print(f"  bulk modulus derivative: {fit.bulk_modulus_derivative:.4f}")
    print(f"  residual: {fit.residual:.3e} eV")
    print(f"  results: {output}")
    print(f"  plot: {plot}")
    return 0


def register_parser(subparsers) -> None:
    """Register the EOS preparation and postprocessing stages."""
    register_stages(
        subparsers,
        "eos",
        "Fit the equation of state from volume-scaled calculations.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
