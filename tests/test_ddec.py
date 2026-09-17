"""Tests for DDEC analysis driven by Chargemol."""

from __future__ import annotations

import json
import stat
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from abacustools.commands.postprocess.ddec import run
from abacustools.core.constant import ANG_TO_BOHR
from abacustools.data.ddec import (
    CHARGE_TYPE_FILES,
    CoreElectrons,
    DdecError,
    DdecPair,
    analyze_ddec,
    apply_core_electron_overrides,
    chargemol_executable,
    core_electron_counts,
    core_reference_path,
    grid_issues,
    read_analysis_log,
    read_bond_orders,
    reference_ion_path,
    reference_issues,
    select_pairs,
    write_job_control,
)
from abacustools.data.grid import Charge, RestartCharge


UPF_FIXTURE = """\
<UPF version="2.0.1">
  <PP_HEADER element="Si" z_valence="4.0" l_max="1" mesh_size="3"/>
  <PP_MESH>
    <PP_R type="real" size="3">0.0 1.0 2.0</PP_R>
    <PP_RAB type="real" size="3">1.0 1.0 1.0</PP_RAB>
  </PP_MESH>
  <PP_LOCAL size="3">-4.0 -2.0 -1.0</PP_LOCAL>
  <PP_RHOATOM size="3">0.1 0.2 0.3</PP_RHOATOM>
</UPF>
"""


STRU_FIXTURE = """\
ATOMIC_SPECIES
Si 28.0855 Si.upf

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0

ATOMIC_POSITIONS
Cartesian
Si
0.0
2
0.0 0.0 0.0 1 1 1
1.0 1.0 1.0 1 1 1
"""


#: Stand-in for Chargemol. It checks the inputs it received, keeps the control
#: file for the tests and writes the result files that the parser reads. The
#: spin and bond-order files are only written when the matching input or flag
#: was passed in, and the jmol cell is built from the cube header so that the
#: distances of the parsed bonds follow the job.
FAKE_CHARGEMOL = '''#!/usr/bin/env python3
# Stand-in for the Chargemol program.
import re
from pathlib import Path

work = Path.cwd()
if not (work / "valence_density.cube").is_file():
    raise SystemExit("valence_density.cube is missing")

control = (work / "job_control.txt").read_text()
(work / "captured_job_control.txt").write_text(control)
charge_type = re.search(r"<charge type>\\s*(\\S+)", control).group(1)
compute_bos = ".true." in control.split("<compute BOs>")[1]

NAMES = {
    "DDEC6": (
        "DDEC6_even_tempered_net_atomic_charges.xyz",
        "DDEC6_even_tempered_atomic_spin_moments.xyz",
        "DDEC6_even_tempered_bond_orders.xyz",
    ),
    "DDEC3": (
        "DDEC3_net_atomic_charges.xyz",
        "DDEC3_atomic_spin_moments.xyz",
        "DDEC3_bond_orders.xyz",
    ),
}
charge_file, spin_file, bond_file = NAMES[charge_type]

lines = (work / "valence_density.cube").read_text().splitlines()
natoms = int(lines[2].split()[0])
vectors = [
    [float(value) * 0.529177 for value in lines[3 + index].split()[1:4]]
    for index in range(3)
]
positions = [
    [float(value) * 0.529177 for value in lines[6 + index].split()[2:5]]
    for index in range(natoms)
]
cell_text = ", ".join(
    "{ %12.6f %12.6f %12.6f }" % tuple(vector) for vector in vectors
)
HEADER = "%5d\\n" % natoms + (
    'jmolscript: load "" {1 1 1} spacegroup "x,y,z" unitcell ['
    + cell_text
    + "]\\n"
)


def body(values):
    """One line per atom: symbol, position in Angstrom and the quantity."""
    return "".join(
        "Si  %12.6f %12.6f %12.6f %14.6f\\n" % (position[0], position[1], position[2], value)
        for position, value in zip(positions, values)
    )


(work / charge_file).write_text(HEADER + body([-0.125, 0.125]), encoding="utf-8")
for name, value in (
    ("DDEC_atomic_Rsquared_moments.xyz", 35.824291),
    ("DDEC_atomic_Rcubed_moments.xyz", 88.295321),
    ("DDEC_atomic_Rfourth_moments.xyz", 250.0),
):
    (work / name).write_text(HEADER + body([value] * natoms), encoding="utf-8")

if (work / "spin_density.cube").is_file():
    (work / spin_file).write_text(
        HEADER
        + body([0.5] * natoms)
        + "\\nCollinear spin population analysis was performed\\n"
        + "The total spin magnetic moment of the unit cell is    1.000000\\n",
        encoding="utf-8",
    )

if compute_bos:
    (work / bond_file).write_text(
        HEADER
        + "Si      0.000000      0.000000      0.000000       0.873500\\n"
        + "Si      1.000000      1.000000      1.000000       0.873500\\n"
        + "\\n Printing BOs for ATOM #      1 ( Si ) in the reference unit cell.\\n"
        + " Bonded to the (  0,   0,   0) translated image of atom number     2 ( Si )"
        + " with bond order =     0.8735    The average spin polarization of this"
        + " bonding =     0.0000\\n"
        + " The sum of bond orders for this atom is SBO =        0.873500\\n"
        + " Printing BOs for ATOM #      2 ( Si ) in the reference unit cell.\\n"
        + " Bonded to the (  0,   0,   0) translated image of atom number     1 ( Si )"
        + " with bond order =     0.8735    The average spin polarization of this"
        + " bonding =     0.0000\\n",
        encoding="utf-8",
    )

(work / "valence_cube_DDEC_analysis.output").write_text(
    " ncore =       10.0000\\n"
    " nvalence =        4.0000\\n"
    " numerically integrated valence density =    4.000000E+00\\n"
    " sum_valence_occupancy_correction =    0.000000E+00\\n"
    " checkme =    1.0000E-05\\n"
    " The grid spacing in your electron density input file is adequate.\\n"
    " The grid spacing is adequate and all electrons are properly accounted for.\\n",
    encoding="utf-8",
)
'''


def _fake_chargemol(tmp_path: Path) -> Path:
    path = tmp_path / "chargemol"
    path.write_text(FAKE_CHARGEMOL, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _cube(path: Path, points: int = 20, valence: float = 4.0) -> None:
    cell = np.diag([2.0, 2.0, 2.0])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    charge = Charge(
        np.full((points, points, points), 0.05),
        cell,
        positions,
        [14, 14],
        [valence, valence],
    )
    charge.save_cube(str(path), format="abacus")


def _atomic_densities(directory: Path, *, z: int = 14, ncore: int = 10) -> Path:
    """Create a directory that holds the reference files of one element."""
    directory.mkdir(exist_ok=True)
    core_reference_path(directory, z, ncore).write_text("", encoding="utf-8")
    reference_ion_path(directory, z, z).write_text("", encoding="utf-8")
    return directory


def _job(tmp_path: Path, *, nspin: int = 1, points: int = 20) -> Path:
    job = tmp_path / "job"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "pp").mkdir()
    (job / "pp" / "Si.upf").write_text(UPF_FIXTURE, encoding="utf-8")
    (job / "INPUT").write_text(
        f"INPUT_PARAMETERS\nsuffix ABACUS\nnspin {nspin}\npseudo_dir ./pp\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(STRU_FIXTURE, encoding="utf-8")
    _cube(output / "SPIN1_CHG.cube", points=points)
    if nspin == 2:
        _cube(output / "SPIN2_CHG.cube", points=points)
    return job


def _namespace(job: Path, **overrides) -> Namespace:
    values = dict(
        job=str(job),
        output=None,
        chargemol_exe=None,
        charge_type="DDEC6",
        atomic_densities=None,
        core_electrons=None,
        net_charge=None,
        periodicity=(True, True, True),
        spin=None,
        bos=True,
        cube=None,
        grid=None,
        lat0=ANG_TO_BOHR,
        threads=None,
        workdir=None,
        keep=False,
        cutoff=None,
        pairs=None,
        pairs_file=None,
        threshold=0.2,
        json=True,
    )
    values.update(overrides)
    return Namespace(**values)


def test_write_job_control_matches_the_chargemol_format(tmp_path: Path) -> None:
    path = tmp_path / "job_control.txt"
    write_job_control(
        path,
        net_charge=-0.5,
        periodicity=(True, False, True),
        core_electrons=[
            CoreElectrons("Si", 14, 4.0, 10),
            CoreElectrons("Na", 11, 9.0, 2),
        ],
        atomic_densities=tmp_path,
        charge_type="DDEC6",
    )
    text = path.read_text(encoding="utf-8")
    assert "<net charge>\n-0.500000\n</net charge>" in text
    assert ".true.\n.false.\n.true." in text
    assert f"{tmp_path}/" in text
    assert "<number of core electrons>\n14 10\n11 2\n</number of core electrons>" in text
    assert "<charge type>\nDDEC6\n</charge type>" in text
    assert "<compute BOs>\n.true.\n</compute BOs>" in text


def test_write_job_control_rejects_an_unknown_charge_type(tmp_path: Path) -> None:
    with pytest.raises(DdecError):
        write_job_control(
            tmp_path / "job_control.txt",
            net_charge=0.0,
            periodicity=(True, True, True),
            core_electrons=[],
            atomic_densities=tmp_path,
            charge_type="DDEC9",
        )


def test_core_electron_counts_from_cube_columns(tmp_path: Path) -> None:
    path = tmp_path / "Si_CHG.cube"
    _cube(path)
    density = Charge.from_cube(str(path), format="abacus")
    entries = core_electron_counts(density)
    assert [entry.element for entry in entries] == ["Si"]
    assert entries[0].z == 14
    assert entries[0].z_valence == pytest.approx(4.0)
    assert entries[0].ncore == 10


def test_apply_core_electron_overrides_rejects_an_unknown_element() -> None:
    entries = [CoreElectrons("Si", 14, 4.0, 10)]
    assert apply_core_electron_overrides(entries, [(14, 4)])[0].ncore == 4
    with pytest.raises(DdecError):
        apply_core_electron_overrides(entries, [(26, 10)])


def test_reference_issues_reports_missing_files(tmp_path: Path) -> None:
    directory = _atomic_densities(tmp_path / "atomic_densities")
    entries = [CoreElectrons("Si", 14, 4.0, 10), CoreElectrons("Ce", 58, 11.0, 47)]
    issues = reference_issues(directory, entries)
    codes = {issue.code for issue in issues}
    assert "missing-core-density" in codes
    message = "\n".join(issue.message for issue in issues)
    assert "core_058_058_047_500_100.txt" in message
    silicon = [
        issue
        for issue in issues
        if issue.code == "missing-core-density" and "Si" in issue.message
    ]
    assert silicon == []


def test_reference_issues_accepts_zero_core_electrons(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    issues = reference_issues(directory, [CoreElectrons("Li", 3, 3.0, 0)])
    codes = [issue.code for issue in issues]
    assert "missing-reference-ion" in codes
    assert "missing-core-density" not in codes


def test_reference_issues_warns_about_an_incomplete_reference_ion_window(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "atomic_densities"
    directory.mkdir()
    core_reference_path(directory, 17, 10).write_text("", encoding="utf-8")
    for electrons in range(9, 20):  # the shipped tables stop at 19 electrons
        reference_ion_path(directory, 17, electrons).write_text("", encoding="utf-8")
    issues = reference_issues(directory, [CoreElectrons("Cl", 17, 7.0, 10)])
    warnings = [issue for issue in issues if issue.level == "warning"]
    assert warnings and warnings[0].code == "incomplete-reference-ion-window"
    assert "[20]" in warnings[0].message


def test_grid_issues_rejects_a_coarse_grid() -> None:
    charge = Charge(
        np.ones((4, 4, 4)),
        np.diag([4.0, 4.0, 4.0]),
        np.array([[0.0, 0.0, 0.0]]),
        [14],
        [4.0],
    )
    issues = grid_issues(charge)
    assert [issue.level for issue in issues] == ["error"]
    assert issues[0].code == "coarse-grid"


def test_grid_issues_accepts_an_abacus_grid() -> None:
    cell = np.array([[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]])
    charge = Charge(
        np.ones((54, 54, 54)),
        cell,
        np.array([[0.0, 0.0, 0.0]]),
        [14],
        [4.0],
    )
    assert grid_issues(charge) == []


def test_chargemol_executable_missing_raises() -> None:
    with pytest.raises(DdecError):
        chargemol_executable("definitely-not-a-chargemol-executable")


def test_read_bond_orders_reads_images_and_sums(tmp_path: Path) -> None:
    path = tmp_path / "bond_orders.xyz"
    path.write_text(
        "    1\n"
        'jmolscript: load "" {1 1 1} spacegroup "x,y,z" unitcell '
        "[{ 0.0 2.0 2.0 }, { 2.0 0.0 2.0 }, { 2.0 2.0 0.0 }]\n"
        "Si      0.000000      0.000000      0.000000       3.803777\n"
        "\n Printing BOs for ATOM #      1 ( Si ) in the reference unit cell.\n"
        " Bonded to the (  0,   0,  -1) translated image of atom number     1 ( Si )"
        " with bond order =     0.0209    The average spin polarization of this"
        " bonding =     0.0000\n",
        encoding="utf-8",
    )
    sums, records = read_bond_orders(path)
    assert sums == pytest.approx([3.803777])
    assert records[0]["image"] == (0, 0, -1)
    assert records[0]["bond_order"] == pytest.approx(0.0209)
    assert records[0]["atom2"] == 1


def test_read_analysis_log_reports_the_verdicts(tmp_path: Path) -> None:
    path = tmp_path / "valence_cube_DDEC_analysis.output"
    path.write_text(
        " ncore =       10.0000\n nvalence =       16.0000\n"
        " numerically integrated valence density =    1.6000E+01\n"
        " checkme =    2.4781E-04\n"
        " The grid spacing in your electron density input file is adequate.\n"
        " The grid spacing is adequate and all electrons are properly accounted for.\n",
        encoding="utf-8",
    )
    accounting = read_analysis_log(path)
    assert accounting["ncore"] == pytest.approx(10.0)
    assert accounting["nvalence"] == pytest.approx(16.0)
    assert accounting["integrated_valence"] == pytest.approx(16.0)
    assert accounting["checkme"] == pytest.approx(2.4781e-4)
    assert accounting["grid_adequate"] is True
    assert accounting["electrons_accounted"] is True


def test_select_pairs_filters_by_selection_and_threshold() -> None:
    pairs = [
        DdecPair(1, 2, "Si", "Si", (0, 0, 0), 0.87, 0.0, 1.73),
        DdecPair(1, 1, "Si", "Si", (0, 0, 1), 0.02, 0.0, 1.0),
    ]
    assert len(select_pairs(pairs, threshold=0.2)) == 1
    assert len(select_pairs(pairs, selection={(1, 1)})) == 1
    assert len(select_pairs(pairs, cutoff=1.5)) == 1


def test_analyze_ddec_on_a_cube_job(tmp_path: Path) -> None:
    job = _job(tmp_path)
    directory = _atomic_densities(tmp_path / "atomic_densities")
    work = tmp_path / "work"
    analysis = analyze_ddec(
        job,
        exe=str(_fake_chargemol(tmp_path)),
        atomic_densities=str(directory),
        workdir=work,
    )
    assert analysis.charge_type == "DDEC6"
    assert analysis.nspin == 1
    assert analysis.density_source == "cube"
    assert [atom.element for atom in analysis.atoms] == ["Si", "Si"]
    assert analysis.atoms[0].net_charge == pytest.approx(-0.125)
    assert analysis.total_net_charge == pytest.approx(0.0)
    assert analysis.atoms[0].spin_moment is None
    assert analysis.atoms[0].r_squared == pytest.approx(35.824291)
    assert analysis.atoms[0].bond_order_sum == pytest.approx(0.8735)
    assert len(analysis.pairs) == 2
    assert analysis.pairs[0].image == (0, 0, 0)
    assert analysis.pairs[0].distance == pytest.approx(np.sqrt(3.0), rel=1e-4)
    assert analysis.electron_accounting["nvalence"] == pytest.approx(4.0)
    control = (work / "captured_job_control.txt").read_text(encoding="utf-8")
    assert "14 10" in control
    assert str(directory) + "/" in control
    assert not (work / "spin_density.cube").exists()


def test_analyze_ddec_needs_no_pseudopotential_for_cube_input(tmp_path: Path) -> None:
    job = tmp_path / "cube-only"
    output = job / "OUT.ABACUS"
    output.mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 1\n", encoding="utf-8"
    )
    _cube(output / "SPIN1_CHG.cube")
    analysis = analyze_ddec(
        job,
        exe=str(_fake_chargemol(tmp_path)),
        atomic_densities=str(_atomic_densities(tmp_path / "atomic_densities")),
        workdir=tmp_path / "work2",
    )
    assert analysis.core_electrons[0].ncore == 10


def test_analyze_ddec_converts_a_restart_file(tmp_path: Path) -> None:
    job = _job(tmp_path)
    output = job / "OUT.ABACUS"
    (output / "SPIN1_CHG.cube").unlink()
    shape = (32, 32, 32)
    fractions = [np.fft.fftfreq(n) * n for n in shape]
    mesh = np.meshgrid(*fractions, indexing="ij")
    miller = np.stack([axis.ravel() for axis in mesh], axis=1).astype(np.int64)
    reciprocal = np.linalg.inv(np.diag([2.0, 2.0, 2.0]))
    rng = np.random.default_rng(0)
    rhog = rng.standard_normal((1, miller.shape[0])) + 1j * rng.standard_normal((1, miller.shape[0]))
    RestartCharge(rhog, miller, reciprocal).write(
        str(output / "ABACUS-CHARGE-DENSITY.restart")
    )
    (output / "running_scf.log").write_text(
        "fft grid for charge/potential = [ 32, 32, 32 ]\n", encoding="utf-8"
    )
    analysis = analyze_ddec(
        job,
        exe=str(_fake_chargemol(tmp_path)),
        atomic_densities=str(_atomic_densities(tmp_path / "atomic_densities")),
        workdir=tmp_path / "work",
    )
    assert analysis.density_source.startswith("restart")
    assert analysis.core_electrons[0].ncore == 10
    assert analysis.atoms[0].net_charge == pytest.approx(-0.125)


def test_analyze_ddec_writes_the_spin_density(tmp_path: Path) -> None:
    job = _job(tmp_path, nspin=2)
    analysis = analyze_ddec(
        job,
        exe=str(_fake_chargemol(tmp_path)),
        atomic_densities=str(_atomic_densities(tmp_path / "atomic_densities")),
        workdir=tmp_path / "work",
    )
    assert analysis.nspin == 2
    assert analysis.atoms[0].spin_moment == pytest.approx(0.5)
    assert analysis.total_spin_moment == pytest.approx(1.0)


def test_analyze_ddec_without_bond_orders(tmp_path: Path) -> None:
    job = _job(tmp_path)
    analysis = analyze_ddec(
        job,
        exe=str(_fake_chargemol(tmp_path)),
        atomic_densities=str(_atomic_densities(tmp_path / "atomic_densities")),
        compute_bond_orders=False,
        workdir=tmp_path / "work",
    )
    assert analysis.pairs == ()
    assert analysis.atoms[0].bond_order_sum is None


def test_analyze_ddec_reads_ddec3_output(tmp_path: Path) -> None:
    job = _job(tmp_path)
    analysis = analyze_ddec(
        job,
        charge_type="DDEC3",
        exe=str(_fake_chargemol(tmp_path)),
        atomic_densities=str(_atomic_densities(tmp_path / "atomic_densities")),
        workdir=tmp_path / "work",
    )
    assert analysis.charge_type == "DDEC3"
    assert CHARGE_TYPE_FILES["DDEC3"]["charges"] in analysis.outputs
    assert analysis.atoms[0].net_charge == pytest.approx(-0.125)


def test_analyze_ddec_rejects_an_unknown_charge_type(tmp_path: Path) -> None:
    with pytest.raises(DdecError):
        analyze_ddec(_job(tmp_path), charge_type="DDEC1")


def test_analyze_ddec_rejects_nspin4(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "OUT.ABACUS").mkdir(parents=True)
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nnspin 4\n", encoding="utf-8"
    )
    with pytest.raises(DdecError):
        analyze_ddec(job)


def test_analyze_ddec_needs_the_core_reference_density(tmp_path: Path) -> None:
    job = _job(tmp_path)
    directory = tmp_path / "atomic_densities"
    directory.mkdir()
    reference_ion_path(directory, 14, 14).write_text("", encoding="utf-8")
    with pytest.raises(DdecError) as error:
        analyze_ddec(
            job,
            exe=str(_fake_chargemol(tmp_path)),
            atomic_densities=str(directory),
            workdir=tmp_path / "work",
        )
    assert "core_014_014_010_500_100.txt" in str(error.value)


def test_ddec_command_cube_nspin1(tmp_path: Path) -> None:
    job = _job(tmp_path)
    output = tmp_path / "report.json"
    code = run(
        _namespace(
            job,
            output=str(output),
            chargemol_exe=str(_fake_chargemol(tmp_path)),
            atomic_densities=str(_atomic_densities(tmp_path / "atomic_densities")),
            workdir=str(tmp_path / "work"),
            json=False,
        )
    )
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["charge_type"] == "DDEC6"
    assert report["atoms"][0]["net_charge"] == pytest.approx(-0.125)
    assert report["pairs"][0]["bond_order"] == pytest.approx(0.8735)
    assert report["core_electrons"][0]["ncore"] == 10


def test_ddec_command_honours_core_electron_overrides(tmp_path: Path) -> None:
    job = _job(tmp_path)
    work = tmp_path / "work"
    directory = _atomic_densities(tmp_path / "atomic_densities", z=14, ncore=10)
    core_reference_path(directory, 14, 4).write_text("", encoding="utf-8")
    code = run(
        _namespace(
            job,
            chargemol_exe=str(_fake_chargemol(tmp_path)),
            atomic_densities=str(directory),
            core_electrons=[(14, 4)],
            workdir=str(work),
        )
    )
    assert code == 0
    control = (work / "captured_job_control.txt").read_text(encoding="utf-8")
    assert "14 4" in control


def test_ddec_command_reports_a_failure(tmp_path: Path) -> None:
    job = _job(tmp_path, points=4)
    code = run(
        _namespace(
            job,
            chargemol_exe=str(_fake_chargemol(tmp_path)),
            atomic_densities=str(_atomic_densities(tmp_path / "atomic_densities")),
        )
    )
    assert code == 1
