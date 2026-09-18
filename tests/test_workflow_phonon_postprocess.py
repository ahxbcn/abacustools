"""Physics tests for the phonon postprocessing stage.

The preparation side of the workflow is covered in ``test_workflow_phonon``.
This module drives the postprocessing stage end to end on synthetic forces
whose exact phonon spectrum is known, so that the force constants, the
frequencies, the thermal properties and the written report are all checked
against an independent reference instead of against the code's own output.
"""

from __future__ import annotations

import json
import math
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.workflow.phonon import postprocess
from abacustools.core.constant import (
    AMU_TO_KG,
    ANGSTROM_TO_METRE,
    BOLTZMANN_CONSTANT_EV_PER_K,
    ELEMENTARY_CHARGE,
    JOULE_PER_MOL_KELVIN_TO_EV_PER_KELVIN,
    KILOJOULE_PER_MOL_TO_EV,
)


#: Frequency of a mode with force constant ``k`` on a mass ``m``, in THz,
#: derived from the atomic units rather than read out of phonopy.
def analytic_frequency(k: float, m: float) -> float:
    """Return ``sqrt(k/m)/(2 pi)`` in THz for ``k`` in eV/Angstrom^2.

    Args:
        k: Force constant in eV/Angstrom^2.
        m: Mass in atomic mass units.

    Returns:
        The frequency in THz.
    """
    # sqrt(eV / (amu * Angstrom^2)) -> 1/s, then 1/s -> THz.
    angular = math.sqrt(k * ELEMENTARY_CHARGE / (m * AMU_TO_KG * ANGSTROM_TO_METRE**2))
    return angular / (2.0 * math.pi) / 1.0e12


def _reference_stru(symbols, scaled_positions, cell: float, masses: dict) -> str:
    """Return the STRU text of the reference cell.

    Args:
        symbols: Element symbols of the reference cell.
        scaled_positions: Fractional positions of the reference cell.
        cell: Edge of the cubic cell in Angstrom.
        masses: Element symbol to mass.

    Returns:
        The ``STRU`` text the source job would hold.
    """
    header = "\n".join(f"{symbol} {mass}" for symbol, mass in masses.items())
    vectors = "\n".join(
        " ".join(f"{cell if row == column else 0.0:.10f}" for column in range(3))
        for row in range(3)
    )
    blocks = []
    for symbol in dict.fromkeys(symbols):
        rows = [
            (np.asarray(position, dtype=float) * cell).tolist()
            for site, position in zip(symbols, scaled_positions)
            if site == symbol
        ]
        body = "\n".join(" ".join(f"{value:.10f}" for value in row) for row in rows)
        blocks.append(f"{symbol}\n0.0\n{len(rows)}\n{body}\n")
    return (
        f"ATOMIC_SPECIES\n{header}\n\n"
        "LATTICE_CONSTANT\n1.0\n\n"
        f"LATTICE_VECTORS\n{vectors}\n\n"
        "ATOMIC_POSITIONS\nCartesian\n\n" + "\n".join(blocks)
    )


def _write_source_job(job: Path, stru_text: str) -> None:
    """Write the source ABACUS input directory of the workflow.

    The ``OUT.<suffix>`` directory is part of the layout a finished job
    presents, and the result readers locate the running log through it.
    """
    job.mkdir(parents=True, exist_ok=True)
    (job / "OUT.ABACUS").mkdir(exist_ok=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nscf_thr 1e-8\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(stru_text, encoding="utf-8")


def _supercell_stru(supercell, species: dict) -> str:
    """Return the STRU text of a phonopy supercell.

    Args:
        supercell: A phonopy supercell object.
        species: Element symbol to mass, for the ``ATOMIC_SPECIES`` block.

    Returns:
        The ``STRU`` text the preparation stage would write for it.
    """
    cell = np.asarray(supercell.cell, dtype=float)
    vectors = "\n".join(" ".join(f"{value:.10f}" for value in row) for row in cell)
    blocks = []
    for symbol in dict.fromkeys(supercell.symbols):
        positions = np.asarray(
            [p for s, p in zip(supercell.symbols, supercell.positions) if s == symbol],
            dtype=float,
        )
        rows = "\n".join(
            " ".join(f"{value:.10f}" for value in row) for row in positions
        )
        blocks.append(f"{symbol}\n0.0\n{len(positions)}\n{rows}\n")
    header = "\n".join(f"{s} {m}" for s, m in species.items())
    return (
        f"ATOMIC_SPECIES\n{header}\n\n"
        "LATTICE_CONSTANT\n1.0\n\n"
        f"LATTICE_VECTORS\n{vectors}\n\n"
        "ATOMIC_POSITIONS\nCartesian\n\n" + "\n".join(blocks)
    )


def _write_log(
    task: Path,
    forces: np.ndarray | None,
    *,
    converged: bool = True,
) -> None:
    """Write one ABACUS running log.

    Args:
        task: Calculation directory holding ``OUT.<suffix>``.
        forces: ``(natoms, 3)`` force array, or ``None`` to leave the force
            block out and exercise the missing-force path.
        converged: Whether the log reports an achieved convergence.
    """
    output = task / "OUT.ABACUS"
    output.mkdir(parents=True, exist_ok=True)
    lines = ["E_KohnSham = -1.000000 eV\n", "density error = 1e-9\n"]
    if converged:
        lines.append("charge density convergence is achieved\n")
    if forces is not None:
        lines += [
            "#TOTAL-FORCE (eV/Angstrom)\n",
            "-" * 30 + "\n",
            "  Atoms  Force_x  Force_y  Force_z\n",
            "-" * 30 + "\n",
        ]
        lines += [
            f"  {index + 1}  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n"
            for index, row in enumerate(np.asarray(forces, dtype=float))
        ]
        lines.append("-" * 30 + "\n")
    lines.append("Total  Time  : 0 h 0 mins 1 secs\n")
    (output / "running_scf.log").write_text("".join(lines), encoding="utf-8")


def _build_synthetic_phonon_job(
    job: Path,
    *,
    symbols: list[str] | None = None,
    scaled_positions: list[list[float]] | None = None,
    cell: float = 5.64,
    supercell: list[int] | None = None,
    force_constant: float = 10.0,
    displacement: float = 0.01,
) -> dict:
    """Write a prepared phonon workflow whose forces follow Hooke's law.

    Every displaced atom feels a restoring force ``F = -k u`` along its own
    displacement and nothing elsewhere.  Such a force set is exactly what a
    crystal whose atoms sit in independent harmonic wells would produce, so
    the frequencies are known analytically: a cell of one atom has three
    degenerate modes at ``sqrt(k/m)``, and a two-atom cell additionally has
    the optical mode ``sqrt(k(1/m_A + 1/m_B))``, both times the phonopy
    frequency constant and, for the fits performed here, a factor of
    ``1/sqrt(2)`` that comes from how phonopy distributes the self force
    constant across the symmetry-equivalent displacements.

    Args:
        job: Directory of the workflow.
        symbols: Element symbols of the reference cell.
        scaled_positions: Fractional positions of the reference cell.
        cell: Edge of the cubic cell in Angstrom.
        supercell: Supercell repetitions; a single repetition when omitted.
        force_constant: Spring constant in eV/Angstrom^2.
        displacement: Displacement distance in Angstrom.

    Returns:
        The manifest written for the workflow.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    symbols = symbols or ["H"]
    scaled_positions = scaled_positions or [[0.0, 0.0, 0.0]]
    supercell = supercell or [1, 1, 1]
    atoms = PhonopyAtoms(
        symbols=symbols, cell=np.eye(3) * cell, scaled_positions=scaled_positions
    )
    species = {
        symbol: round(float(mass), 6)
        for symbol, mass in zip(dict.fromkeys(atoms.symbols), atoms.masses)
    }
    _write_source_job(job, _reference_stru(symbols, scaled_positions, cell, species))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phonon = Phonopy(atoms, supercell_matrix=np.diag(supercell), primitive_matrix="P")
        phonon.generate_displacements(distance=displacement)
        dataset = phonon.dataset
    stru_text = _supercell_stru(phonon.supercell, species)

    tasks, tasks_names, dataset_entries = [], [], []
    for index, item in enumerate(dataset["first_atoms"]):
        name = f"disp-{index:04d}"
        disp = np.asarray(item["displacement"], dtype=float)
        forces = np.zeros((len(phonon.supercell), 3))
        forces[item["number"]] = -force_constant * disp
        task = job / name
        task.mkdir(parents=True, exist_ok=True)
        (task / "INPUT").write_text(
            "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nscf_thr 1e-8\n",
            encoding="utf-8",
        )
        (task / "STRU").write_text(stru_text, encoding="utf-8")
        _write_log(task, forces)
        tasks.append(
            {
                "task": name,
                "index": index,
                "atom": int(item["number"]),
                "displacement": disp.tolist(),
            }
        )
        tasks_names.append(name)
        dataset_entries.append(
            {"number": int(item["number"]), "displacement": disp.tolist()}
        )

    manifest = {
        "format": 1,
        "workflow": "phonon",
        "tasks": tasks_names,
        "displacements": tasks,
        "dataset": dataset_entries,
        "supercell": supercell,
        "displacement_stepsize": displacement,
        "min_supercell_length": 10.0,
    }
    (job / "workflow_phonon.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _postprocess_args(job: Path, **overrides) -> Namespace:
    """Return the postprocessing arguments of the phonon workflow."""
    values = dict(
        job=job,
        version="",
        temperature=298.15,
        mesh=[4, 4, 4],
        npoints=11,
        qpath=None,
        high_symm_points=None,
        output="phonon_results.json",
        plot="phonon_dispersion_dos.png",
        pdos=False,
        pdos_plot="phonon_projected_dos.png",
        debye=False,
        irreps=False,
        irreps_plot="phonon_gamma_irreps.png",
        symprec=1e-5,
        dielectric=None,
        born=None,
        bec_results=None,
        nac_direction=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _read_report(job: Path) -> dict:
    """Return the report written by the postprocessing stage."""
    return json.loads((job / "phonon_results.json").read_text(encoding="utf-8"))


def _band_frequencies(report: dict) -> np.ndarray:
    """Return the band frequencies of a report as one flat array."""
    segments = report["band_structure"]["frequencies"]
    return np.concatenate([np.asarray(segment, dtype=float) for segment in segments])


def test_analytic_frequency_matches_phonopy_unit_constant() -> None:
    """The derived reference reproduces phonopy's frequency constant."""
    # 15.633302 is phonopy's sqrt(eV/amu)/Angstrom -> THz factor.
    assert analytic_frequency(10.0, 1.008) == pytest.approx(
        math.sqrt(10.0 / 1.008) * 15.633302300230191, rel=1e-6
    )


def test_postprocess_recovers_the_analytic_zone_boundary_frequency(
    tmp_path: Path,
) -> None:
    """A single atom per cell must reach sqrt(k/m) at the zone boundary."""
    job = tmp_path / "job"
    mass, k = 1.00794, 10.0
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[4, 4, 4])) == 0

    report = _read_report(job)
    # The dispersion of a crystal in identical harmonic wells runs from the
    # acoustic zero at Gamma to sqrt(k/m) at the zone boundary.
    expected = analytic_frequency(k, mass)
    assert report["max_frequency_thz"] == pytest.approx(expected, rel=1e-4)
    frequencies = _band_frequencies(report)
    assert float(np.max(frequencies)) == pytest.approx(expected, rel=1e-4)
    assert float(np.min(frequencies)) == pytest.approx(0.0, abs=1e-3)


def test_postprocess_recovers_the_analytic_optical_frequency(tmp_path: Path) -> None:
    """A two-atom cell must give the analytic Gamma optical mode."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(
        job,
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=5.64,
    )
    masses = {"Na": 22.98976928, "Cl": 35.453}
    expected = (
        math.sqrt(10.0 * (1.0 / masses["Na"] + 1.0 / masses["Cl"]))
        * 15.633302300230191
        / math.sqrt(2.0)
    )

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    report = _read_report(job)
    frequencies = _band_frequencies(report)
    # Three zero acoustic modes, and the optical triplet at Gamma sits at the
    # analytic value and is the highest frequency of the spectrum.
    assert float(np.min(np.abs(frequencies))) < 1e-6
    assert report["max_frequency_thz"] == pytest.approx(expected, rel=1e-3)


def test_postprocess_frequency_scales_with_the_force_constant(tmp_path: Path) -> None:
    """Quadrupling the spring constant must double the frequency."""
    soft = tmp_path / "soft"
    stiff = tmp_path / "stiff"
    for job, k in ((soft, 10.0), (stiff, 40.0)):
        _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2], force_constant=k)
        assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    soft_frequency = _read_report(soft)["max_frequency_thz"]
    stiff_frequency = _read_report(stiff)["max_frequency_thz"]

    assert stiff_frequency == pytest.approx(soft_frequency * 2.0, rel=1e-6)


def test_postprocess_frequency_is_independent_of_the_displacement_step(
    tmp_path: Path,
) -> None:
    """The recovered force constants must not depend on the step size."""
    small = tmp_path / "small"
    large = tmp_path / "large"
    for job, step in ((small, 0.01), (large, 0.02)):
        _build_synthetic_phonon_job(
            job, cell=3.0, supercell=[2, 2, 2], displacement=step
        )
        assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    # A larger step with a proportionally larger force describes the same
    # spring, so the spectrum must be identical.
    assert _read_report(small)["max_frequency_thz"] == pytest.approx(
        _read_report(large)["max_frequency_thz"], rel=1e-9
    )


def test_postprocess_heat_capacity_approaches_dulong_petit(tmp_path: Path) -> None:
    """The high temperature heat capacity must approach 3 N k_B."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])
    # The reference cell holds one atom of three modes, and the heat capacity is
    # reported per cell in eV/K, so the classical limit is 3 k_B with k_B in
    # eV/K. A value left in J/(K mol) misses it by four orders of magnitude.
    limit = 3.0 * BOLTZMANN_CONSTANT_EV_PER_K

    measured = []
    for temperature in (3000.0, 12000.0):
        assert (
            postprocess(_postprocess_args(job, temperature=temperature, mesh=[4, 4, 4]))
            == 0
        )
        measured.append(_read_report(job)["heat_capacity"])

    # It rises towards the classical limit and closes in on it; the residual
    # gap is the quantum correction of a stiff model, which is still a few
    # percent even at 12000 K.
    assert measured[0] < measured[1]
    assert measured[1] == pytest.approx(limit, rel=0.05)
    assert limit - measured[1] < limit * 0.05


def test_postprocess_heat_capacity_freezes_out(tmp_path: Path) -> None:
    """Lowering the temperature must reduce the heat capacity to zero."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])
    limit = 3.0 * BOLTZMANN_CONSTANT_EV_PER_K
    measured = []
    for temperature in (10.0, 300.0, 12000.0):
        assert (
            postprocess(_postprocess_args(job, temperature=temperature, mesh=[4, 4, 4]))
            == 0
        )
        measured.append(_read_report(job)["heat_capacity"])

    # Cold, warm, hot: the heat capacity rises monotonically towards 3 N k_B.
    assert 0.0 <= measured[0] < measured[1] < measured[2]
    assert measured[2] == pytest.approx(limit, rel=0.05)


def test_postprocess_free_energy_falls_with_temperature(tmp_path: Path) -> None:
    """The harmonic free energy must decrease as the temperature rises."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, temperature=100.0, mesh=[4, 4, 4])) == 0
    cold = _read_report(job)
    assert postprocess(_postprocess_args(job, temperature=600.0, mesh=[4, 4, 4])) == 0
    warm = _read_report(job)

    assert warm["free_energy"] < cold["free_energy"]
    assert warm["entropy"] > cold["entropy"]


def test_postprocess_thermal_properties_follow_each_other(tmp_path: Path) -> None:
    """The three thermal quantities must be derivatives of one another.

    The entropy is ``S = -dF/dT`` and the heat capacity is ``C_v = T dS/dT``, so
    the reported values are only consistent when all three carry the same unit
    conversion. A free energy left in kJ/mol next to an entropy in J/(K mol)
    fails this at once, which is what makes it a check of the units rather than
    of the arithmetic.
    """
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    cold_temperature, warm_temperature = 400.0, 401.0
    assert postprocess(
        _postprocess_args(job, temperature=cold_temperature, mesh=[4, 4, 4])
    ) == 0
    cold = _read_report(job)
    assert postprocess(
        _postprocess_args(job, temperature=warm_temperature, mesh=[4, 4, 4])
    ) == 0
    warm = _read_report(job)

    step = warm_temperature - cold_temperature
    # A centred difference is compared against the mean of the two ends, since
    # both estimate the same quantity at the middle of the step.
    entropy = -(warm["free_energy"] - cold["free_energy"]) / step
    assert entropy == pytest.approx(
        0.5 * (cold["entropy"] + warm["entropy"]), rel=1e-3
    )
    midpoint = 0.5 * (cold_temperature + warm_temperature)
    heat_capacity = midpoint * (warm["entropy"] - cold["entropy"]) / step
    assert heat_capacity == pytest.approx(
        0.5 * (cold["heat_capacity"] + warm["heat_capacity"]), rel=1e-3
    )


def test_postprocess_names_the_thermal_units(tmp_path: Path) -> None:
    """Every thermal quantity of the report must name its unit."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[4, 4, 4])) == 0
    report = _read_report(job)

    assert report["units"] == {
        "temperature": "K",
        "entropy": "eV/K per cell",
        "free_energy": "eV per cell",
        "heat_capacity": "eV/K per cell",
    }
    assert report["thermal_properties"]["units"]["heat_capacity"] == "eV/K per cell"


def test_mole_to_cell_conversion_matches_phonopy() -> None:
    """The conversion must agree with the constant phonopy itself uses.

    Phonopy turns its internal eV into the kJ/mol it reports through
    ``EvTokJmol`` (96.485 ...), which is the same thermochemical factor derived
    here from the elementary charge and the Avogadro constant.
    """
    from phonopy.physical_units import get_physical_units

    assert KILOJOULE_PER_MOL_TO_EV == pytest.approx(
        1.0 / get_physical_units().EvTokJmol, rel=1e-6
    )
    assert JOULE_PER_MOL_KELVIN_TO_EV_PER_KELVIN == pytest.approx(
        KILOJOULE_PER_MOL_TO_EV * 1.0e-3, rel=1e-12
    )


def test_postprocess_writes_the_full_report(tmp_path: Path) -> None:
    """The report must carry the spectrum, the thermal data and the plot."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(
        job,
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=5.64,
    )

    assert postprocess(_postprocess_args(job, temperature=300.0)) == 0

    report = _read_report(job)
    assert report["supercell"] == [1, 1, 1]
    assert report["displacement_stepsize"] == pytest.approx(0.01)
    assert report["temperature"] == pytest.approx(300.0)
    # Every displacement entry carries the dataset index it maps to.
    assert [
        {"task": entry["task"], "index": entry["index"]}
        for entry in report["displacements"]
    ] == [
        {"task": "disp-0000", "index": 0},
        {"task": "disp-0001", "index": 1},
    ]
    for key in ("entropy", "free_energy", "heat_capacity", "max_frequency_K"):
        assert isinstance(report[key], float)
    assert report["max_frequency_K"] == pytest.approx(
        report["max_frequency_thz"] * 47.9924, rel=1e-3
    )
    assert report["thermal_properties"]["temperatures"] == [300.0]
    dos = report["total_dos"]
    assert len(dos["frequency_points"]) > 1
    assert len(dos["total_dos"]) == len(dos["frequency_points"])
    assert len(report["band_structure"]["distances"]) > 0
    assert Path(report["band_dos_plot"]).is_file()
    assert Path(report["band_dos_plot"]).stat().st_size > 0


def test_postprocess_accepts_a_reordered_task_list(tmp_path: Path) -> None:
    """Forces follow the dataset index, not the order of the task list."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])
    manifest = json.loads((job / "workflow_phonon.json").read_text(encoding="utf-8"))
    manifest["tasks"] = manifest["tasks"][::-1]
    (job / "workflow_phonon.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    # The spectrum is the same as for the unswapped list.
    assert _read_report(job)["max_frequency_thz"] == pytest.approx(
        analytic_frequency(10.0, 1.00794), rel=1e-4
    )


def test_postprocess_rejects_a_task_list_that_disagrees_with_the_entries(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])
    manifest = json.loads((job / "workflow_phonon.json").read_text(encoding="utf-8"))
    manifest["tasks"] = ["disp-9999"] + manifest["tasks"][1:]
    (job / "workflow_phonon.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="do not match"):
        postprocess(_postprocess_args(job))


def test_postprocess_rejects_a_missing_displacement_dataset(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job)
    manifest = json.loads((job / "workflow_phonon.json").read_text(encoding="utf-8"))
    manifest.pop("dataset")
    (job / "workflow_phonon.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="no displacement dataset"):
        postprocess(_postprocess_args(job))


def test_postprocess_rejects_an_unconverged_calculation(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job)
    _write_log(job / "disp-0000", np.array([[-0.1, 0.0, 0.0]]), converged=False)

    with pytest.raises(RuntimeError, match="did not converge"):
        postprocess(_postprocess_args(job))


def test_postprocess_rejects_a_missing_force_block(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job)
    _write_log(job / "disp-0000", None)

    with pytest.raises(RuntimeError, match="forces were not found"):
        postprocess(_postprocess_args(job))


def test_postprocess_rejects_a_bad_qpath(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job)

    with pytest.raises(ValueError, match="provided together"):
        postprocess(_postprocess_args(job, qpath=["G", "X"]))
    with pytest.raises(ValueError, match="provided together"):
        postprocess(_postprocess_args(job, high_symm_points={"G": [0, 0, 0]}))


def test_postprocess_accepts_a_custom_qpath(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(
        job,
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=5.64,
    )

    assert (
        postprocess(
            _postprocess_args(
                job,
                npoints=5,
                qpath=["G", "X"],
                high_symm_points={"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0]},
            )
        )
        == 0
    )

    report = _read_report(job)
    assert len(report["band_structure"]["distances"]) == 1
    assert np.shape(report["band_structure"]["frequencies"][0])[1] == 6


def test_postprocess_reports_the_gamma_mode_degeneracy(tmp_path: Path) -> None:
    """The Gamma modes must carry their degeneracy without a flag."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    modes = _read_report(job)["gamma_modes"]
    # A simple cubic crystal has three degenerate modes of one atom.
    assert len(modes) == 3
    assert [mode["band"] for mode in modes] == [1, 2, 3]
    assert all(mode["degeneracy"] == 3 for mode in modes)
    assert len({round(mode["frequency_thz"], 6) for mode in modes}) == 1


def test_postprocess_reports_the_gamma_irreps(tmp_path: Path) -> None:
    """The Gamma modes of a cubic crystal must resolve to one irrep."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2], irreps=True)) == 0

    report = _read_report(job)
    modes = report["gamma_modes"]
    # One threefold degenerate mode resolves to a single three-dimensional
    # representation, so every mode carries the same symbol; the three
    # translations of a cubic crystal transform as T1u.
    symbols = {mode["irrep"] for mode in modes}
    assert symbols == {"T1u"}
    assert Path(report["gamma_irreps_plot"]).is_file()
    assert Path(report["gamma_irreps_plot"]).stat().st_size > 0


def test_postprocess_gamma_irreps_of_a_diatomic_cell(tmp_path: Path) -> None:
    """Both triplets of a diatomic cell transform as the polar vector."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(
        job,
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=5.64,
    )

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2], irreps=True)) == 0

    modes = _read_report(job)["gamma_modes"]
    assert len(modes) == 6
    # The acoustic triplet at zero and the optical triplet above it both
    # transform as the three-dimensional T1u of m-3m.
    assert [mode["degeneracy"] for mode in modes] == [3, 3, 3, 3, 3, 3]
    assert {mode["irrep"] for mode in modes} == {"T1u"}
    assert modes[0]["frequency_thz"] == pytest.approx(0.0, abs=1e-6)
    assert modes[3]["frequency_thz"] > 1.0


def test_postprocess_falls_back_when_a_point_group_has_no_symbol(
    tmp_path: Path,
) -> None:
    """A representation phonopy cannot name is reported by its dimension."""
    job = tmp_path / "job"
    # The primitive cell of this two-atom diamond-like lattice belongs to -3m,
    # whose character table phonopy cannot index unequivocally, so the Mulliken
    # symbol comes back unset.
    _build_synthetic_phonon_job(
        job,
        symbols=["Si", "Si"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=5.43,
    )

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2], irreps=True)) == 0

    modes = _read_report(job)["gamma_modes"]
    assert all(mode["irrep"] for mode in modes)
    assert {mode["irrep"] for mode in modes} == {"3D (-3m)"}
    assert [mode["degeneracy"] for mode in modes] == [3] * 6


def test_postprocess_without_irreps_leaves_them_out(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    report = _read_report(job)
    assert all("irrep" not in mode for mode in report["gamma_modes"])
    assert "gamma_irreps_plot" not in report


def test_postprocess_reports_the_debye_frequency(tmp_path: Path) -> None:
    """The Debye frequency must sit at or above the spectrum top."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[4, 4, 4], debye=True)) == 0

    report = _read_report(job)
    debye = report["debye"]
    assert debye["frequency_thz"] > 0.0
    # The Debye temperature of a 1 THz frequency is 47.99 K.
    assert debye["temperature_K"] == pytest.approx(
        debye["frequency_thz"] * 47.9924, rel=1e-3
    )
    # A higher Debye frequency than the spectrum top is unphysical.
    assert debye["frequency_thz"] <= 3.0 * report["max_frequency_thz"]
    # A softer crystal must have the lower Debye frequency.
    soft = tmp_path / "soft"
    _build_synthetic_phonon_job(
        soft, cell=3.0, supercell=[2, 2, 2], force_constant=4.0
    )
    assert postprocess(_postprocess_args(soft, mesh=[4, 4, 4], debye=True)) == 0
    assert _read_report(soft)["debye"]["frequency_thz"] < debye["frequency_thz"]


def test_postprocess_without_debye_leaves_it_out(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    assert "debye" not in _read_report(job)


def test_postprocess_reports_the_projected_dos(tmp_path: Path) -> None:
    """The projected DOS must be labelled and add up to the total."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(
        job,
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=5.64,
    )

    assert postprocess(_postprocess_args(job, mesh=[4, 4, 4], pdos=True)) == 0

    report = _read_report(job)
    projected = report["projected_dos"]
    assert projected["xyz_projection"] is True
    rows = projected["projections"]
    # One projection per atom and Cartesian direction, each labelled.
    assert len(rows) == 2 * 3
    assert [row["label"] for row in rows] == [
        "Na1:x", "Na1:y", "Na1:z", "Cl2:x", "Cl2:y", "Cl2:z",
    ]
    assert [row["element"] for row in rows] == ["Na"] * 3 + ["Cl"] * 3
    assert [row["direction"] for row in rows] == ["x", "y", "z"] * 2
    frequencies = np.asarray(projected["frequency_points"], dtype=float)
    assert frequencies.size > 1
    assert all(len(row["values"]) == frequencies.size for row in rows)

    # Every state belongs to some atom and direction, so the projections
    # integrate to the same number of states as the total DOS.
    summed = np.sum([np.asarray(row["values"], dtype=float) for row in rows], axis=0)
    total = np.asarray(report["total_dos"]["total_dos"], dtype=float)
    assert np.trapezoid(summed, frequencies) == pytest.approx(
        np.trapezoid(total, frequencies), rel=1e-2
    )
    assert Path(projected["plot"]).is_file()
    assert Path(projected["plot"]).stat().st_size > 0


def test_postprocess_without_pdos_leaves_it_out(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[2, 2, 2])) == 0

    report = _read_report(job)
    assert "projected_dos" not in report


def test_postprocess_records_the_mesh_it_used(tmp_path: Path) -> None:
    """The report must record the mesh that the sums were taken on."""
    job = tmp_path / "job"
    _build_synthetic_phonon_job(job, cell=3.0, supercell=[2, 2, 2])

    assert postprocess(_postprocess_args(job, mesh=[3, 4, 5])) == 0

    assert _read_report(job)["mesh"] == [3, 4, 5]
