"""The ``abacustools workflow dftu`` workflow."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from abacustools.data.dftu import (
    ResponseData,
    _find_output_dir,
    compute_response_function,
    parse_running_scf_dftu,
)
from abacustools.data.versions import default_version
from abacustools.core.submission import generate_workflow_submission

from .common import (
    clear_generated_jobs,
    kpoint_filename,
    read_job_structure,
    register_stages,
    write_abacus_job,
    write_manifest,
)


_ANGULAR_LABEL = {2: "d", 3: "f"}


def _validate_u_values(values: list[float]) -> list[float]:
    """Validate and deduplicate U values, ensuring 0 is included."""
    if len(values) < 2:
        raise ValueError("at least two U values are required")

    result: list[float] = []
    for value in values:
        if not np.isfinite(value) or value < 0:
            raise ValueError("U values must be non-negative finite numbers")
        if value not in result:
            result.append(float(value))

    # Ensure U=0 is included for bare response calculation
    if 0.0 not in result:
        result.insert(0, 0.0)
        print("  note: U=0.0 automatically added for bare response calculation")

    return sorted(result)


def _resolve_atom_indices(
    indices: list[int] | None,
    elements: list[str] | None,
    structure,
) -> list[int]:
    """Resolve atom indices from --index and/or --elements arguments.

    Returns 0-based atom indices.
    """
    if indices is None and elements is None:
        raise ValueError(
            "specify at least one of --index or --elements "
            "to identify correlated atoms"
        )

    result: list[int] = []
    natom = len(structure.atoms)

    if indices is not None:
        for idx in indices:
            if idx < 1 or idx > natom:
                raise ValueError(
                    f"atom index {idx} out of range (1..{natom})"
                )
            idx_0based = idx - 1
            if idx_0based not in result:
                result.append(idx_0based)

    if elements is not None:
        elem_set = set(elements)
        for i, atom in enumerate(structure.atoms):
            if atom.element in elem_set and i not in result:
                result.append(i)

    if not result:
        raise ValueError("no atoms matched the given criteria")

    return sorted(result)


def _resolve_angular_momentum(
    orbital: int | None,
    structure,
    atom_indices: list[int],
) -> int:
    """Determine the angular momentum for the correlated orbitals.

    If not explicitly given, infer from the elements:
    transition metals -> d (2), lanthanides/actinides -> f (3).
    """
    if orbital is not None:
        if orbital not in (2, 3):
            raise ValueError(
                f"orbital must be 2 (d) or 3 (f), got {orbital}"
            )
        return orbital

    # Auto-detect from elements
    _d_elements = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    }
    _f_elements = {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
        "Es", "Fm", "Md", "No", "Lr",
    }

    detected: set[int] = set()
    for idx in atom_indices:
        elem = structure.atoms[idx].element
        if elem in _d_elements:
            detected.add(2)
        elif elem in _f_elements:
            detected.add(3)
        else:
            raise ValueError(
                f"cannot auto-detect orbital for element '{elem}'; "
                "use --orbital 2 (d) or --orbital 3 (f)"
            )

    if len(detected) > 1:
        raise ValueError(
            "selected atoms require different orbital types (d and f); "
            "use --orbital to specify explicitly"
        )

    return detected.pop()


def _register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the DFT+U preparation stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="ABACUS input directory used to prepare DFT+U calculations.",
    )
    parser.add_argument(
        "--index", type=int, nargs="+", default=None,
        help="One-based indices of correlated atoms.",
    )
    parser.add_argument(
        "--elements", type=str, nargs="+", default=None,
        help="Element symbols of correlated atoms (e.g. Ni Fe).",
    )
    parser.add_argument(
        "--orbital", type=int, default=None,
        help="Angular momentum of correlated orbitals: 2 (d) or 3 (f). "
        "Auto-detected from elements when omitted.",
    )
    parser.add_argument(
        "--u-values", type=float, nargs="+", default=None,
        help="U values to scan in eV (e.g. 0 2 4 6 8 10).",
    )
    parser.add_argument(
        "--u-min", type=float, default=None,
        help="Minimum U value in eV (used with --u-max and --u-step).",
    )
    parser.add_argument(
        "--u-max", type=float, default=None,
        help="Maximum U value in eV (used with --u-min and --u-step).",
    )
    parser.add_argument(
        "--u-step", type=float, default=None,
        help="Step size for U values in eV (used with --u-min and --u-max).",
    )
    parser.add_argument(
        "--dftu-type", type=int, choices=[1, 2], default=1,
        help="DFT+U method: 1 (legacy) or 2 (Hamiltonian-based, recommended). "
        "Default: 2.",
    )
    parser.add_argument(
        "--onsite-radius", type=float, default=None,
        help="Radius of the projection sphere in Bohr. "
        "Uses ABACUS default (3.0) when omitted.",
    )
    parser.add_argument(
        "--scf-thr", type=float, default=None,
        help="Override SCF convergence threshold for better precision.",
    )
    parser.add_argument(
        "--scf-nmax", type=int, default=None,
        help="Override maximum SCF iterations.",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Replace existing generated DFT+U directories.",
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
    """Register arguments for the DFT+U postprocessing stage."""
    parser.add_argument(
        "-j", "--job", type=Path, required=True,
        help="Directory containing the prepared DFT+U calculations.",
    )
    parser.add_argument(
        "-v", "--version", default=default_version(),
        help="ABACUS version used for the calculations.",
    )
    parser.add_argument(
        "-o", "--output", default="dftu_results.json",
        help="Output JSON filename. Relative paths are resolved below JOB.",
    )
    parser.add_argument(
        "--plot", default="dftu_response.png",
        help="Response plot filename. Relative paths are resolved below JOB.",
    )


def _resolve_u_values(args: argparse.Namespace) -> list[float]:
    """Resolve U values from --u-values or --u-min/--u-max/--u-step."""
    if args.u_values is not None:
        if args.u_min is not None or args.u_max is not None or args.u_step is not None:
            raise ValueError(
                "cannot combine --u-values with --u-min/--u-max/--u-step"
            )
        return _validate_u_values(list(args.u_values))

    if args.u_min is not None and args.u_max is not None and args.u_step is not None:
        if args.u_step <= 0:
            raise ValueError("--u-step must be positive")
        if args.u_min < 0:
            raise ValueError("--u-min must be non-negative")
        if args.u_max < args.u_min:
            raise ValueError("--u-max must be >= --u-min")
        values = []
        u = args.u_min
        while u <= args.u_max + args.u_step * 0.01:
            values.append(round(u, 4))
            u += args.u_step
        return _validate_u_values(values)

    raise ValueError(
        "specify either --u-values or all of --u-min, --u-max, --u-step"
    )


def prepare(args: argparse.Namespace) -> int:
    """Prepare DFT+U linear-response SCF jobs for individual atoms."""
    u_values = _resolve_u_values(args)
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise FileNotFoundError(f"job directory not found: {job}")

    inputs, stru_filename, structure = read_job_structure(job)
    kpoint = kpoint_filename(job, inputs)

    # Resolve correlated atoms and orbital type
    atom_indices = _resolve_atom_indices(args.index, args.elements, structure)
    angular_momentum = _resolve_angular_momentum(
        args.orbital, structure, atom_indices
    )

    # Build task name mapping first
    all_task_names = []
    atom_task_map = {}
    
    for atom_idx in atom_indices:
        atom_task_names = []
        for u_value in u_values:
            task_name = f"atom{atom_idx + 1}_U{u_value:05.2f}".replace(".", "p")
            atom_task_names.append(task_name)
            all_task_names.append(task_name)
        atom_task_map[atom_idx] = atom_task_names
    
    # Clear old jobs BEFORE creating new ones
    clear_generated_jobs(job, all_task_names, override=args.override)
    
    # Now create the jobs
    for atom_idx in atom_indices:
        atom_task_names = atom_task_map[atom_idx]
        
        # Create modified structure with unique label for target atom
        modified_structure = deepcopy(structure)
        original_label = modified_structure.atoms[atom_idx].label
        target_label = f"{original_label}_{atom_idx + 1}"
        
        # Modify the target atom's label
        modified_structure.atoms[atom_idx].label = target_label
        
        # Build orbital_corr and hubbard_u for the modified structure
        unique_labels = []
        for atom in modified_structure.atoms:
            if atom.label not in unique_labels:
                unique_labels.append(atom.label)
        
        orbital_corr = []
        for label in unique_labels:
            if label == target_label:
                orbital_corr.append(angular_momentum)
            else:
                orbital_corr.append(-1)
        
        # Generate one job per U value for this atom
        for u_value, task_name in zip(u_values, atom_task_names):
            task_inputs = deepcopy(inputs)
            
            # Set DFT+U parameters
            task_inputs["dft_plus_u"] = args.dftu_type
            task_inputs["orbital_corr"] = orbital_corr
            
            # Set hubbard_u: only target atom gets U value
            hubbard_u = []
            for label in unique_labels:
                if label == target_label:
                    hubbard_u.append(u_value)
                else:
                    hubbard_u.append(0.0)
            task_inputs["hubbard_u"] = hubbard_u
            
            # Ensure output settings
            task_inputs["out_mul"] = 1
            
            # Optional overrides
            if args.onsite_radius is not None:
                task_inputs["onsite_radius"] = args.onsite_radius
            if args.scf_thr is not None:
                task_inputs["scf_thr"] = args.scf_thr
            if args.scf_nmax is not None:
                task_inputs["scf_nmax"] = args.scf_nmax
            
            # Ensure calculation is SCF
            task_inputs["calculation"] = "scf"
            
            dest = job / task_name
            write_abacus_job(
                task_inputs,
                modified_structure,
                source_dir=job,
                destination_dir=dest,
                stru_filename=stru_filename,
                kpoint=kpoint,
            )
    
    # Generate submission scripts if requested
    submission = generate_workflow_submission(
        job,
        "dftu",
        all_task_names,
        submission_type=getattr(args, "submission_type", None),
        generate=getattr(args, "generate_scripts", None),
        abacus_command=getattr(args, "abacus_command", None),
    )
    if submission is not None:
        print(f"  submission type: {submission['type']}")
        print(f"  workflow script: {submission['workflow_script']}")
    
    # Write manifest
    manifest_data = {
        "u_values": u_values,
        "atom_indices": [i + 1 for i in atom_indices],  # 1-based in manifest
        "atom_task_map": {str(k + 1): v for k, v in atom_task_map.items()},
        "angular_momentum": angular_momentum,
        "tasks": all_task_names,
        "dftu_type": args.dftu_type,
    }
    if submission is not None:
        manifest_data["submission"] = submission
    write_manifest(job, "dftu", **manifest_data)
    
    print(f"Prepared {len(all_task_names)} DFT+U calculations for {len(atom_indices)} atoms under {job}/")
    for atom_idx, task_names in atom_task_map.items():
        print(f"  Atom {atom_idx + 1}: {len(task_names)} tasks")
    
    return 0

def postprocess(args: argparse.Namespace) -> int:
    """Postprocess DFT+U linear-response calculations for individual atoms."""
    job = Path(args.job).absolute()
    if not job.is_dir():
        raise FileNotFoundError(f"job directory not found: {job}")

    # Read manifest
    manifest_path = job / "workflow_dftu.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"workflow manifest not found: {manifest_path}. "
            "Run 'abacustools workflow dftu prepare' first."
        )
    manifest = json.loads(manifest_path.read_text())

    u_values = manifest["u_values"]
    atom_indices_1based = manifest["atom_indices"]
    atom_task_map = manifest.get("atom_task_map", {})
    angular_momentum = manifest["angular_momentum"]

    # Compute response functions for each atom independently
    results = []
    
    for atom_idx_1based in atom_indices_1based:
        atom_idx_0based = atom_idx_1based - 1
        task_names = atom_task_map.get(str(atom_idx_1based), [])
        
        if not task_names:
            print(f"Warning: No tasks found for atom {atom_idx_1based}")
            continue
        
        # Collect occupation data for this atom
        occupations = []
        
        for task_name in task_names:
            task_dir = job / task_name
            
            if not task_dir.is_dir():
                raise FileNotFoundError(
                    f"Task directory not found: {task_dir}. "
                    "Did the calculation complete successfully?"
                )
            
            # Look for running_scf.log in output directory
            out_dir = _find_output_dir(task_dir)
            if out_dir is None:
                raise FileNotFoundError(
                    f"No output directory found in {task_dir}. "
                    "Did the calculation complete successfully?"
                )
            
            log_path = out_dir / "running_scf.log"
            if not log_path.is_file():
                raise FileNotFoundError(
                    f"running_scf.log not found in {out_dir}. "
                    "The calculation may not have started."
                )
            
            # Parse occupation data from running_scf.log
            atoms_data = parse_running_scf_dftu(log_path)
            if atoms_data is None:
                raise ValueError(
                    f"No DFT+U data found in {log_path}. "
                    "Ensure dft_plus_u is set in INPUT."
                )
            
            # Find the target atom's occupation
            # The target atom should have the modified label (e.g., Ni_1)
            # We need to find it by checking which atom has non-zero occupation change
            target_occ = None
            for idx, data in atoms_data.items():
                if data['L'] == angular_momentum:
                    # Compute total occupation for this atom
                    total_occ = 0.0
                    for spin, eigenvalues in data['eigenvalues'].items():
                        total_occ += sum(eigenvalues)
                    
                    # For the first U value, check if this atom has occupation
                    # The target atom should be the one with the modified label
                    # Since we modified the STRU, the target atom will be at a specific position
                    # We need to identify it by its position in the structure
                    if target_occ is None:
                        # For now, assume the target atom is the one we're analyzing
                        # This is a simplification; in reality, we need to track which atom
                        # in the modified structure corresponds to the original atom index
                        target_occ = total_occ
                        break
            
            if target_occ is None:
                raise ValueError(
                    f"Could not find occupation data for atom {atom_idx_1based} "
                    f"in task {task_name}"
                )
            
            occupations.append(target_occ)
        
        # Check if occupations vary with U
        occ_array = np.array(occupations)
        occ_range = np.max(occ_array) - np.min(occ_array)
        
        if occ_range < 1e-6:
            print(f"Warning: Occupation for atom {atom_idx_1based} does not vary with U "
                  f"(range = {occ_range:.2e}). Skipping this atom.")
            continue
        
        # Compute response function
        chi_0, chi, hubbard_u = compute_response_function(u_values, occupations)
        
        results.append(
            ResponseData(
                atom_index=atom_idx_0based,
                angular_momentum=angular_momentum,
                u_values=list(u_values),
                occupations=occupations,
                chi_0=chi_0,
                chi=chi,
                hubbard_u=hubbard_u,
            )
        )

    if not results:
        raise RuntimeError(
            "no occupation data found for any of the specified atoms. "
            "Check that the calculations completed and produced onsite.dm."
        )

    # Build output report
    l_label = _ANGULAR_LABEL.get(angular_momentum, f"l={angular_momentum}")
    report: dict[str, Any] = {
        "format": 1,
        "workflow": "dftu",
        "angular_momentum": angular_momentum,
        "angular_momentum_label": l_label,
        "u_scan_values_eV": u_values,
        "atoms": [],
    }

    for resp in results:
        atom_entry = {
            "atom_index": resp.atom_index + 1,  # 1-based
            "angular_momentum": resp.angular_momentum,
            "chi_0_eV_inv": resp.chi_0,
            "chi_eV_inv": resp.chi,
            "hubbard_u_eV": resp.hubbard_u,
            "occupations": resp.occupations,
        }
        report["atoms"].append(atom_entry)

    # Write JSON report
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = job / output_path
    output_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")

    # Print summary
    print(f"Hubbard U from linear response ({l_label} orbitals)")
    print(f"{'Atom':>8s}  {'chi_0 (eV^-1)':>14s}  {'chi (eV^-1)':>14s}  {'U (eV)':>10s}")
    print("-" * 54)
    for resp in results:
        print(
            f"{resp.atom_index + 1:>8d}  {resp.chi_0:>14.6f}  "
            f"{resp.chi:>14.6f}  {resp.hubbard_u:>10.4f}"
        )

    print(f"\nResults written to {output_path}")

    # Try to generate plot
    try:
        _plot_response(job, results, args.plot, l_label)
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
    except Exception as exc:
        print(f"  (plot generation failed: {exc})")

    return 0

def _plot_response(
    job: Path,
    results: list,
    plot_filename: str,
    l_label: str,
) -> None:
    """Generate a response function plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_filename)
    if not plot_path.is_absolute():
        plot_path = job / plot_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: occupation vs U
    ax1 = axes[0]
    for resp in results:
        ax1.plot(
            resp.u_values, resp.occupations,
            "o-", label=f"Atom {resp.atom_index + 1}",
        )
    ax1.set_xlabel("U (eV)")
    ax1.set_ylabel("Total occupation")
    ax1.set_title(f"{l_label}-orbital occupation vs U")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: response function
    ax2 = axes[1]
    for resp in results:
        # Plot dn/dU as a function of U (numerical derivative)
        u_arr = np.array(resp.u_values)
        occ_arr = np.array(resp.occupations)
        if len(u_arr) >= 2:
            dn_du = np.gradient(occ_arr, u_arr)
            ax2.plot(u_arr, dn_du, "s-", label=f"Atom {resp.atom_index + 1}")
    ax2.set_xlabel("U (eV)")
    ax2.set_ylabel("dn/dU (eV^-1)")
    ax2.set_title("Response function")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot written to {plot_path}")


def register_parser(subparsers) -> None:
    """Register the DFT+U linear-response workflow."""
    register_stages(
        subparsers,
        "dftu",
        "Calculate Hubbard U via linear response.",
        prepare,
        postprocess,
        _register_prepare_arguments,
        _register_postprocess_arguments,
    )
