# Workflow catalogue

Common options on every `prepare`: `-j/--job JOB` (the reference ABACUS input
directory) and `--override`. Common options on every `postprocess`:
`-j/--job JOB`, `-v/--version` (ABACUS log profile), and usually `-o/--output`
for the result JSON. Every run also writes `workflow_<name>.json`. Only
`band postprocess` accepts `--json`.

## Electronic structure

**`band`** - prepare writes two jobs: `band_scf/` (an SCF that stores the
charge density) and `band_nscf/` (an NSCF with the line-mode KPT along the
path, which reads the density back and writes `BANDS_1.dat`). Run the SCF
first. The path follows the dimensionality as in `file kpt --path`: seekpath
for a bulk, the in-plane path for a slab with the vacuum direction at `k = 0`,
the periodic axis for a wire; `--path-mode auto|bulk|slab|wire` and
`--min-vacuum` control the choice, `--npoints` sets points per segment and
`--bands` the NSCF `nbands`. Postprocess reports the gap (VBM, CBM,
direct/indirect), writes `band_results.json` and plots `band.png`; `--json`,
`--plot`, `--no-plot`, `--emin/--emax` and `--efermi` tune the report.

## Vibrational and thermal

**`phonon`** - prepare generates displaced supercell SCF jobs `disp-NNNN`
numbered by their index in the Phonopy displacement dataset; `--supercell A B C`
(or an automatic supercell at least `--min-supercell-length` Angstrom,
default 10) and `--displacement-stepsize` (default 0.01 A). Postprocess reads
their forces and reports dispersion, thermal properties at `--temperature`
(default 298.15 K), and Gamma-point modes with degeneracy, plus optional
`--debye`, `--pdos` (`--pdos-plot`), and `--irreps` (`--irreps-plot`,
`--symprec`). `--mesh`/`--npoints` set the DOS mesh, `--qpath`/
`--high-symm-points` supply a custom path. Thermal properties are reported per
cell in eV and eV/K, converted from Phonopy's per-mole units.

**`gruneisen`** - prepare calls the phonon prepare three times at volumes
`(1 +/- strain)^(1/3)`; `--strain` is limited to 0.1-5 percent because the
parameter is a central difference of the frequencies. Postprocess rebuilds each
volume with the phonon loader and writes `gruneisen_mesh.yaml`,
`gruneisen_band.yaml`, and `gruneisen_results.json` (q-weighted mean, range and
count of the mode parameters, and `gamma(T)` over `--tmin/--tmax/--tstep`).
The three volumes must share supercell, k mesh and displacement step; the
non-analytical correction is not applied here.

**`thermal-conductivity`** (alias `kappa`) - needs `phono3py`, imported lazily.
Prepare writes displaced supercells under `fc3-*` plus an optional independent
`--supercell-fc2` set and `phono3py_disp.yaml`. Postprocess fits `fc2`/`fc3`
and solves the Boltzmann transport equation: `--mesh`, `--tmin/--tmax/--tstep`,
`--lbte` for the full linearized solution alongside the RTA reference. Writes
the conductivity tensor per temperature, `thermal_conductivity.png`, and the
phono3py `kappa-*.hdf5`. Missing or unconverged forces stop it with an error.

**`vibration`** - prepare writes equilibrium and displaced force calculations
below `vib/`; `-i/--index` selects one-based atoms (default all) and
`-s/--stepsize` the displacement. Postprocess reports frequencies in cm^-1,
zero-point energy, and thermochemical corrections, marking imaginary modes with
`i`, and writes mode displacements below `vib/modes/`. `--traj` writes one
vibration period per mode below `vib/mode_trajectories/`
(`--traj-format extxyz|traj`, `--frames`), and `--output-stru`/`--no-output-stru`
with `--stru-format extxyz|poscar` control the mode structures. `--backend ase`
(default) uses ASE; `--backend builtin` uses the abacustools analysis, which
adds reduced masses in amu and force constants in eV/Angstrom^2 per mode.
`--mass ELEMENT=MASS` (or `--element-mass`) sets isotope masses, matched against
element first and atom label second. `--gaussian-log [FILE]` writes a fake
Gaussian frequency log (default `gaussian_fake.log`) that GaussView can open to
animate the modes; `--no-cell` omits the periodic cell from it.

## Mechanical and elastic

**`elastic`** - prepare generates one unstrained job plus strained jobs;
`--norelax` switches from ionic relaxation at fixed cell shape to fixed-ion
SCF. `--norm`/`--shear` set the strain magnitudes, `--dimension auto|2d|3d`
decides whether the vacuum direction takes part (2D results are reported in
N/m with in-plane Young's modulus and Poisson ratio), and `--strains
independent` generates only the symmetry-distinct strain directions (nine jobs
instead of twenty-five for cubic). Postprocess writes `elastic_results.json`
with both the raw 6x6 fit and the symmetry-projected tensor, the largest
projection change, the independent constants and the Voigt moduli; `--fit
independent` fits the symmetry basis directly, `--no-symmetrize` keeps the raw
fit, and `--symprec` (default 0.01 A) sets the cell's symmetry tolerance.

**`energy-strain`** - the same constants from the energy curvature:
`E(e) = E0 + (V0/2) sum_k a_k (e B_k e)`. Prepare generates strained cells with
symmetric amplitudes that keep the reference stress out of the curvature;
postprocess writes `energy_strain_results.json` with the fit, the independent
constants, the moduli, the RMS energy residual and the reference stress
projected on the patterns.

**`fdforce`** / **`fdstress`** - validate analytic derivatives by central finite
differences. `fdforce`: `-i/--index` (one-based) or `--info FILE` with
`abacus-test`-style `C 2 x y z` lines, `-s/--stepsize`, `-n/--number`,
`--dir x y z`. `fdstress`: `-s`, `-n`, six components by default,
`--component` to choose them, `--all-components` for all nine. Calculations go
below `fdforce/`/`fdstress/`; postprocess writes JSON with analytic values,
finite-difference values, deviations, and step-size convergence. Missing or
unconverged results stop it.

## Polar and dielectric properties

**`bec`** - prepare writes SCF plus three Berry-phase NSCF calculations for each
selected atom displacement (`--index`, `--dir x y z`, `--type c`, `--stepsize`).
`run_bec.sh` in each `bec_*` directory is a local runner, not a cluster script.
Postprocess writes `bec_results.json` with the tensors and per-task diagnostics;
missing directions are retained as missing entries so other directions are
still reported.

**`dielectric`** - clamped-ion `epsilon_inf` from a Kubo-Greenwood sum over the
LCAO Hamiltonian/overlap/position matrices, evaluated by the optional `pyatb`
package (needs an MPI runtime; `--pyatb-command 'mpirun -np 4 pyatb'` runs it
out of process). Prepare writes one self-consistent calculation (default name
`dielectric`, `-n/--name`) that enables `out_mat_hs2`, `out_mat_r` and
`symmetry 0`. Postprocess writes `dielectric_results.json` with the tensor,
dense grid, photon window, spin channels, occupied bands, Fermi energy, and
pyatb version. The photon window must start at zero and reach well above the
gap; `--omega 0 80` is the default and a suspicious window is warned about.

**`piezoelectric`** - prepare generates the six Voigt strain modes (or only the
independent ones with `--strains independent`) at `--strain`, with
`--type f|b|c` for forward/backward/central differences and `--relax` to relax
ions at each strained cell (which turns the clamped-ion tensor into the
relaxed-ion one). Postprocess writes `piezoelectric_results.json` with the
fitted tensor, the symmetrized tensor, the largest change, and the independent
components; `--fit independent`, `--no-symmetrize` behave as in `elastic`.
Avoid `--use-k-continuity` on LTSv3.10.1: every Berry-phase step aborts.

**`workfunc`** - prepare enables `out_pot 2` and writes `workfunc_job`;
`--vacuum a|b|c|auto`, `--dipole-corr`, and the empty-atom options
`--use-empty-atom`, `--empty-atom-elem`, `--empty-atom-height`,
`--empty-atom-dist`. Postprocess identifies the vacuum plateaus and writes the
work function and potential profile. Reads `ElecStaticPot.cube` (LTS) and
`potes.cube`/`pot_es.cube` (develop).

## Energies, defects, and responses

**`eos`** - prepare generates one fixed-volume SCF (or `--relax` ionic
relaxation) job per volume scale over `--start/--end/--step`; postprocess fits a
third-order Birch-Murnaghan equation of state and writes `eos_results.json`
(equilibrium volume, bulk modulus, B0') plus `eos.png`.

**`vacancy`** - prepare builds the pristine supercell, one defective supercell
per `-i/--index` (one-based; atoms become `empty`), and reference elemental
crystal jobs under `ref_element/` (`--no-cal-reference`, `--ref-dir` to
change). Postprocess computes `E_f = (E_defect + mu) - E_original * N_cells`
and writes `vacancy_results.json`.

**`dftu`** - prepare scans `--u-values` or `--u-min/--u-max/--u-step`, treating
`--index`/`--elements` atoms as correlated and giving them their own species so
only they carry the perturbation; `--orbital 2|3` selects d/f and
`--dftu-type` the flavour. Postprocess reads the local occupation matrices from
`running_scf.log`, fits the bare and screened responses, and writes
`dftu_results.json` plus `dftu_response.png`.

**`exchange`** (alias `magj`) - four-state method for magnetic coupling.
Prepare takes pairs via `-p/--pair LABEL1 INDEX1 LABEL2 INDEX2` or `-f FILE`
(`LABEL1 INDEX1 LABEL2 INDEX2` per line), with `-s/--step` in degrees and
`-n/--number` tilt angles; the reference job must be `nspin 4`. Calculations go
below `magj/` (reference in `magj/original`). Postprocess fits
`dE = J (1 - cos(theta))` per pair and writes `exchange_results.json` with the
fit quality and every point; `--allow-unconverged` fits anyway and lists the
affected calculations. J is the coefficient for the reference moment
magnitudes, so divide by their product for a per-unit-moment constant.

## Convergence, validation, and other

**`ecutwfc`** / **`kspacing`** - prepare writes one SCF per `--values` entry;
postprocess reports energy per atom, convergence deltas, incomplete tasks, the
first value within `--energy-tol`, a JSON report and a convergence plot.

**`chgdiff`** - prepare writes the full-system and two-subsystem SCF jobs
(`-i` selects the subsystem atoms); postprocess writes the difference cube.

**`bsse`** - preparation and postprocessing framework for basis-set
superposition error; the prepare stage takes `-i/--index` for the first
subsystem and the manifest-driven structure is the same as the others.
