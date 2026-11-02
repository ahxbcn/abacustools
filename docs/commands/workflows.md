# Workflow commands

[Documentation home](../index.md)

The `workflow` family uses `prepare` and `postprocess` stages. A prepare stage
writes a `workflow_<name>.json` manifest beside the generated calculations; the
postprocess stage reads that manifest instead of inferring the setup from file
or directory order. The available workflows are `band`, `bsse`, `bec`,
`chgdiff`, `dftu`, `dielectric`, `ecutwfc`, `energy-strain`, `eos`,
`exchange`/`magj`, `elastic`, `fdforce`, `fdstress`, `gruneisen`,
`kspacing`, `phonon`, `piezoelectric`, `thermal-conductivity`/`kappa`,
`vacancy`, `vibration`, and `workfunc`.

## workflow ecutwfc

Convergence tests can generate independent SCF calculations for cutoff energy:

```text
abacustools workflow ecutwfc prepare -j JOB --values 30 40 50 60
abacustools workflow ecutwfc postprocess -j JOB --energy-tol 1e-4
```

## workflow kspacing

The same workflow can test k-point spacing:

```text
abacustools workflow kspacing prepare -j JOB --values 0.4 0.3 0.2 0.1
abacustools workflow kspacing postprocess -j JOB
```

The postprocessing stage reports total energy per atom, convergence deltas,
incomplete tasks, a recommended first value within the tolerance, a JSON
report, and a convergence plot.

## workflow band

A band structure along the seekpath high-symmetry path is prepared from a
reference input directory:

```text
abacustools workflow band prepare -j JOB --npoints 20
abacustools workflow band postprocess -j JOB
```

The prepare stage writes `band_scf/`, an SCF job that also stores the charge
density, and `band_nscf/`, an NSCF job whose line-mode KPT follows the path and
which reads the SCF density back and writes `BANDS_1.dat`; run the SCF first.
`--nbands` sets the number of bands of the NSCF step. The path itself follows
the dimensionality of the structure as in `file kpt --path`: seekpath for a 3D
bulk, the in-plane path of the 2D lattice for a slab with the vacuum direction
at `k = 0`, and the periodic axis for a 1D wire. `--min-vacuum` and
`--path-mode auto|bulk|slab|wire` control that choice, and the workflow manifest
records the dimensionality and the method that produced the path. The postprocess stage
reports the band gap with its VBM and CBM, writes `band_results.json` next to
the workflow manifest, and plots the bands to `band.png`.

## workflow eos

The equation of state can be fitted from volume-scaled calculations:

```text
abacustools workflow eos prepare -j JOB --start 0.90 --end 1.10 --step 0.025
abacustools workflow eos postprocess -j JOB
```

The prepare stage generates one fixed-volume SCF (or ionic-relaxation) job per
volume scale; the postprocess stage reads their energies and fits a third-order
Birch-Murnaghan equation of state, writing the equilibrium volume, bulk
modulus, and its pressure derivative to `eos_results.json` and `eos.png`.

## workflow vacancy

Vacancy formation energies can be computed from a supercell with an empty atom:

```text
abacustools workflow vacancy prepare -j JOB -s 2 2 2 -i 1 5
abacustools workflow vacancy postprocess -j JOB
```

The prepare stage builds the pristine supercell, one defective supercell per
requested atom index (the atom becomes an `empty` species), and reference
elemental crystal jobs under `ref_element/`. After the cell-relaxation jobs and
their `final_scf` follow-up have run, the postprocess stage computes
`E_f = (E_defect + mu) - E_original * N_cells` and writes
`vacancy_results.json`.

## workflow bsse

Complex calculation workflows are organized by task and stage. The BSSE
workflow currently provides the preparation and postprocessing framework:

```text
abacustools workflow bsse prepare
abacustools workflow bsse postprocess
```

## workflow chgdiff

Charge-density difference calculations are available as `chgdiff`. Prepare
the full-system and two-subsystem SCF jobs, submit them with
`sbatch runabacus.sh`, and then generate the difference cube:

```text
abacustools workflow chgdiff prepare -j JOB -i 1 3 4
abacustools workflow chgdiff postprocess -j JOB
```

## workflow elastic

Elastic constants can be calculated with the stress-strain workflow. It
generates one unstrained job and 24 independently strained jobs. By default,
the jobs use ionic relaxation at fixed cell shape; use `--norelax` for fixed-ion
SCF calculations:

```text
abacustools workflow elastic prepare -j JOB
sbatch runabacus.sh  # submit from each generated job directory
abacustools workflow elastic postprocess -j JOB
```

The preparation stage analyses the reference cell and records its point group,
space group and the number of independent elastic constants in
`workflow_elastic.json`. The postprocessing stage fits the unconstrained 6x6
tensor from the stresses and then projects it onto the subspace the crystal
symmetry allows, so the numerical noise of the stresses no longer shows up as
components the symmetry forbids or as a tensor that is not symmetric in its
two index pairs. Both tensors, the largest change the projection made, the
independent constants and the Voigt moduli are written to
`elastic_results.json` under `JOB`:

```text
symmetrization residual: <largest component the projection changed>, in GPa
independent constants (GPa): C11 = <...>, C12 = <...>, C44 = <...>
```

Use `--no-symmetrize` to keep the raw fit; `--symprec` sets the tolerance of
the symmetry analysis, which defaults to 0.01 Angstrom so that a relaxed cell
is still recognised as symmetric.

Add `--pymatgen` to evaluate the same final tensor through pymatgen's
`ElasticTensor` API without replacing the custom fit. The JSON report then
gains a `pymatgen` block with the Voigt, Reuss and Voigt-Reuss-Hill moduli, the
universal anisotropy, the homogeneous Poisson ratio, longitudinal and
transverse sound velocities, Debye temperature, and the Snyder, Clarke, Cahill
and Agne thermal-conductivity estimates. The custom tensor, symmetrisation,
independent constants and moduli remain in place; the pymatgen block is a
second, independently implemented view of the same elastic tensor:

```text
abacustools workflow elastic postprocess -j JOB --pymatgen
```

The structure-dependent pymatgen properties are offered for three-dimensional
cells only. A slab, wire or molecule cell is reported as unavailable, because
pymatgen computes its velocities and Debye temperature per cell volume, and
that volume contains the vacuum. A slab keeps the custom in-plane block in N/m
instead.

A two dimensional material is recognised automatically: when the reference
cell has vacuum along one direction, only the strain components of the two
periodic axes are prepared and fitted, because the components that involve the
vacuum direction are set by the cell rather than by the material. The report
then carries the in-plane block in `elastic_tensor_2d`, converted to the two
dimensional unit N/m with the cell height along the vacuum direction, its
independent constants, and the directional in-plane Young's modulus and
Poisson ratio:

```text
abacustools workflow elastic prepare -j JOB --dimension auto --strains independent
abacustools workflow elastic postprocess -j JOB
```

`--dimension 3d` forces the three dimensional treatment, and `--dimension 2d`
refuses a cell without vacuum. For a hexagonal sheet the in-plane symmetry
leaves two independent constants, so a single strain direction is enough and
the run needs five calculations instead of twenty five.

For a crystal with symmetry there is a second, cheaper route: strain only one
representative of every symmetry orbit of strain directions and fit the
independent constants directly, instead of straining all six directions and
fitting the full tensor:

```text
abacustools workflow elastic prepare -j JOB --strains independent
abacustools workflow elastic postprocess -j JOB --fit independent
```

The preparation stage picks the directions whose information raises the rank
of the fit, so a cubic crystal needs two of them (`xx` and `yz`, nine jobs
instead of twenty five), a hexagonal or trigonal one three, and a tetragonal
one four; when no symmetry relates the directions all six are kept. The
postprocessing stage then writes the stiffness matrix as a combination of the
symmetry allowed basis tensors and fits its coefficients in one least squares,
which reports the independent constants without a separate symmetrisation
step.

## workflow energy-strain

Elastic constants can also be obtained from the curvature of the total energy
instead of from the stresses. The `energy-strain` workflow strains the cell
along a set of patterns and fits

```text
E(e) = E0 + (V0 / 2) sum_k a_k (e B_k e)
```

```text
abacustools workflow energy-strain prepare -j JOB
sbatch runabacus.sh  # submit from each generated job directory
abacustools workflow energy-strain postprocess -j JOB
```

The patterns are picked so that their curvature covers every independent
constant: three for a cubic crystal (`xx`, `yz` and `xx + yy`, twelve strained
calculations), and one per constant in the lower symmetry classes. Every
pattern is strained with amplitudes that are symmetric about zero, which keeps
the stress of the reference cell out of the curvature; that stress is reported
separately, projected on the patterns, as a check of the reference. The fit,
the independent constants, the moduli and the root mean square energy residual
are written to `energy_strain_results.json` under `JOB`.

## workflow phonon

Phonon spectra can be calculated with finite differences using Phonopy. The
prepare stage generates displaced supercell SCF jobs, and the postprocess
stage reads their forces to produce thermal properties, DOS, and a combined
dispersion/DOS plot:

```text
abacustools workflow phonon prepare -j JOB
abacustools workflow phonon postprocess -j JOB
```

Use `--supercell A B C` to set the supercell explicitly. Without it, the
supercell is selected so each lattice vector is at least 10 Angstrom long.
Custom paths can be passed as JSON with `--qpath` and
`--high-symm-points`.

The report carries the entropy, the free energy and the heat capacity at
`--temperature`, which defaults to 298.15 K. They are written in eV and eV/K
per cell of the reference structure, the unit the vibration workflow reports
its thermochemistry in, and the `units` block of the JSON names them. Phonopy
states its thermal properties per mole of cells, a free energy in kJ/mol and an
entropy and heat capacity in J/(K mol), so they are converted on the way into
the report rather than passed through with the wrong label.

Every report carries the Gamma point modes with their degeneracy. Three
optional analyses extend it, and each one is off by default because it either
costs time or enlarges the report:

```text
abacustools workflow phonon postprocess -j JOB --debye
abacustools workflow phonon postprocess -j JOB --pdos --pdos-plot PDOS.png
abacustools workflow phonon postprocess -j JOB --irreps
```

`--debye` fits a Debye frequency to the total DOS and reports it in THz and as
a temperature. The Debye model is fitted below the quarter point of the
spectrum and extrapolated to the `3 N` modes of the cell, so for a material
whose optical branches carry much of the DOS the cut off can exceed the highest
calculated frequency; the value is then a thermodynamic Debye temperature
rather than the low temperature calorimetric one. `--pdos` reports the DOS projected onto every atom and
Cartesian direction as one labelled record per projection, and plots the
projections against the total DOS. `--irreps` resolves the space-group
irreducible representation of each Gamma point mode by its Mulliken symbol,
using `--symprec` as the symmetry tolerance, and plots the modes labelled with
their symbols. `--irreps` needs a structure whose symmetry can be found, so it
fails with an explicit message when the tolerance does not match the geometry.
Phonopy leaves the symbol unset for point groups whose character table it
cannot index unequivocally, such as the `-3m` of a diamond-like primitive
cell; those modes are reported by their dimension and point group instead,
for example `3D (-3m)`. `--debye` warns and omits the value when the fit does
not converge, which happens when the DOS has no Debye-like low-frequency
region.

The projections are summed on the same mesh as the total DOS, and the mesh is
recorded in the report so that a reader can tell which sampling produced them.
Note that without Born effective charges the Gamma point longitudinal optical
modes carry no non-analytical correction, so a polar material is described
without its LO-TO splitting.

Polar materials need the non-analytical correction, which the long range
Coulomb field of a longitudinal optical vibration adds. Pass the Born effective
charges and the dielectric tensor, both as JSON, and the limit of the q to zero
is taken along a direction:

```text
abacustools workflow phonon postprocess -j JOB --irreps \
  --dielectric "[2.34,0,0,0,2.34,0,0,0,2.34]" \
  --born "[[[1.12,0,0],[0,1.12,0],[0,0,1.12]],[[-1.12,0,0],[0,-1.12,0],[0,0,-1.12]]]"
```

`--dielectric` accepts a scalar, three diagonal values, a flat nine value matrix
or a 3x3 matrix, and `--born` holds one 3x3 tensor per atom of the reference
cell in its atom order. The charges can also be read from the `bec_results.json`
that `workflow bec` writes, with `--bec-results`, which avoids transcribing
tensors by hand: that file already stores them with the rows as the displacement
directions and the columns as the Cartesian polarization directions, which is
the layout the correction expects. In the same way `--dielectric-results` reads
the dielectric tensor of the `dielectric_results.json` that `workflow
dielectric` writes, and the two files together describe a polar material
without transcribing a single number. Born charges and dielectric tensor must
both be given; either one alone is refused, and so is a dielectric tensor that
is given both as JSON and as a results file. The correction is applied to the dispersion, to the
total and projected DOS and to the thermal properties, because the mesh takes
the limit with the direction of each of its own q points. The Gamma point modes
need an explicit direction, which `--nac-direction` sets and which defaults to
the first lattice vector; without it the longitudinal mode keeps the transverse
frequency and the two stay degenerate, so the correction would be reported but
invisible. The report records the tensors and the direction that was used.

The largest frequency of the spectrum is taken over the dispersion rather than
over the commensurate points of the supercell, because a polar material reaches
it in the longitudinal optical mode at Gamma, which the supercell does not
carry. Each Gamma point mode is labelled `LO`, `TO` or `acoustic`: the
longitudinal one is the mode the correction moves, which distinguishes it from
an optical mode that is merely non-degenerate in a low symmetry crystal.

The displaced calculations live in `disp-NNNN`, numbered by their index in the
Phonopy displacement dataset, and `workflow_phonon.json` records both that
index and the displaced atom of every task. Postprocessing therefore maps each
force set onto its displacement through the manifest rather than through the
order of the task list, and refuses a manifest whose task list and displacement
entries disagree. A displaced supercell is written in the Phonopy atom order,
because `AbacusSTRU.supercell` orders atoms by lattice point and mixing the two
orders would attach every force to the wrong atom.

## workflow gruneisen

The mode Grueneisen parameters, which measure how the frequency of each mode
follows the volume, come from the same displaced calculations at three volumes.
The prepare stage writes three ordinary phonon workflows, one per volume, whose
lattice vectors are scaled by `(1 ± strain) ** (1/3)` so that their volume
differs from the reference one by `±strain`:

```text
abacustools workflow gruneisen prepare -j JOB --strain 0.01 --supercell 4 4 4
```

Submit and postprocess the displaced calculations of the three volumes exactly
like a plain phonon workflow, then average them:

```text
abacustools workflow gruneisen postprocess -j JOB
```

`gruneisen_mesh.yaml` and `gruneisen_band.yaml` hold the phonopy mesh and band
results with their plots, and `gruneisen_results.json` adds the q weighted mean,
range and count of the mode parameters and the thermodynamic parameter
`gamma(T) = sum(C_v gamma) / sum(C_v)` at `--temperature`, together with a curve
over `--tmin/--tmax/--tstep`. Every volume has to be computed with the same
supercell, k mesh and displacement step, because the three force constant fits
are compared with each other, and the strain has to stay where the central
difference of the frequencies is meaningful, which is why `--strain` is limited
to 0.1 to 5 percent. The zero frequency translations at Gamma carry no parameter
at all; modes whose parameter comes out as a non-finite number are left out of
the averages and counted in the report. The non-analytical correction is not
applied, since its parameters would have to be computed for every volume as
well, and it only affects the longitudinal optical mode at Gamma.

## workflow thermal-conductivity

Lattice thermal conductivities can be calculated with phono3py using third-order
force constants. The prepare stage generates the displaced supercells below
`fc3-*`, plus an optional independent `fc2-*` set, and writes the exact
phono3py displacement dataset to `phono3py_disp.yaml`; the postprocess stage
reads the forces, fits `fc2`/`fc3` and solves the phonon Boltzmann transport
equation. phono3py is required for this workflow and is imported only when it
runs:

```text
abacustools workflow thermal-conductivity prepare -j JOB
abacustools workflow thermal-conductivity prepare -j JOB \
  --supercell-fc3 2 2 2 --supercell-fc2 4 4 4 \
  --displacement-stepsize-fc3 0.03 --displacement-stepsize-fc2 0.01
abacustools workflow thermal-conductivity postprocess -j JOB --mesh 15 15 15
abacustools workflow thermal-conductivity postprocess -j JOB --lbte \
  --tmin 200 --tmax 400 --tstep 50
```

`kappa` is an alias of `thermal-conductivity`. Without `--supercell-fc3` the
supercell is selected so each lattice vector is at least `--min-supercell-length`
(10 Angstrom) long; without `--supercell-fc2`, `fc2` reuses the `fc3` supercell.
An explicit gamma/MP `KPT` mesh is divided by the supercell repetitions so the
k-point density stays roughly constant, while `kspacing` and `gamma_only` input
are used unchanged. The postprocess stage writes the conductivity tensor per
temperature, the RTA reference values when `--lbte` is used, a
`thermal_conductivity.png` plot, and the phono3py `kappa-*.hdf5` below `JOB`,
and it fails with an explicit error when a force calculation is missing or did
not converge.

## workflow vibration

Molecular vibration frequencies can be calculated with selected atoms using
central finite differences. The workflow writes equilibrium and displaced
force calculations below `vib/`:

```text
abacustools workflow vibration prepare -j JOB
abacustools workflow vibration postprocess -j JOB
```

Use `--index 1 2 ...` to select atoms and `--traj` to write mode trajectories.
Postprocessing reports the frequencies in `cm^-1`, the zero-point energy and
the thermochemical corrections as JSON, marks imaginary modes with `i` in the
mode output and writes the displacements of every mode below `vib/modes/`,
with `--traj` also writing one full vibration period per mode as `extxyz` or
ASE `traj` below `vib/mode_trajectories/`. The mode velocities are scaled for
visualization, so a reference mode of 2500 `cm^-1` moves its atoms with a peak
velocity of 0.5, which keeps every animation in a readable range instead of
growing with the frequency.

Postprocessing uses ASE by default (`--backend ase`): the Hessian is analyzed
by `ase.vibrations.data.VibrationsData`, the thermochemistry by
`ase.thermochemistry.HarmonicThermo`, and the mode structures are written by
ASE. With `--backend builtin` the same workflow runs on the built-in analysis
of `abacustools.data.vibration`, which is the extension point for features that
ASE does not provide:

```text
abacustools workflow vibration postprocess -j JOB --backend builtin
```

Both backends mass-weight the Hessian with the relative atomic masses declared
in `ATOMIC_SPECIES`, which `AbacusSTRU.masses` reports for every structure read
from a STRU file, so the two backends agree to the precision of their
constants. `--mass ELEMENT=MASS` (also `--element-mass`) replaces these masses
for one or more elements, which accounts for isotope effects:

```text
abacustools workflow vibration postprocess -j JOB --mass H=2.014
abacustools workflow vibration postprocess -j JOB --mass H=2.014 O=18.0
```

Keys are matched against the element of every atom first and against the ABACUS
atom label afterwards, so structures with custom labels can be addressed as
well. An assignment that matches no atom of the structure is an error. The
masses that enter the analysis are reported as `masses` in the result JSON.
The built-in backend reports unstable modes as negative frequencies, follows
the harmonic oscillator partition function for the thermochemistry and writes
mode structures with the built-in extended XYZ writer, which carries the cell,
the magnetic moments and the pseudopotential/orbital metadata of the
equilibrium structure. Its JSON adds one entry per mode with the reduced mass
in `amu` and the force constant in `eV/Angstrom^2`, and, per temperature, the
internal energy, the heat capacity and the number of imaginary modes next to
the entropy and the free energy. In both backends the reported zero-point
energy keeps the convention of adding the magnitude of an imaginary mode, while
the thermochemistry of a temperature is evaluated from the stable modes only.

`--gaussian-log [FILE]` writes a fake Gaussian frequency output next to the
results (`gaussian_fake.log` by default), in the spirit of OfakeG and
CP2KfakeG, so that GaussView can open the file and animate the modes:

```text
abacustools workflow vibration postprocess -j JOB --gaussian-log
abacustools workflow vibration postprocess -j JOB --gaussian-log modes.log --no-cell
```

Because an ABACUS structure is periodic, the geometry follows Gaussian's
periodic convention: an `Input orientation` block whose last three centers are
the translation vectors of the cell, written as atomic-number `-2` pseudo-atoms
and followed by the `Lengths of translation vectors` and `Angles of translation
vectors` lines, which is how GaussView reads the unit cell. `--no-cell` leaves
the cell out and writes a plain `Standard orientation` block instead. The
frequency block carries the signed frequencies in `cm^-1`, the reduced masses
in `amu`, the force constants in `mDyne/Angstrom` and the Cartesian normal
coordinates of every atom. The IR intensities are written as zero, because the
workflow does not compute the dipole derivatives, and the thermal section is
filled from the zero-point energy and the thermochemistry of the results.

## Submission scripts

Workflow submission scripts can be generated from `~/.abacustools/config.yaml`.
The packaged defaults support local execution and Slurm, PBS, and LSF
submission. Script generation is disabled by default; enable it globally with
`submission.generate: true`, or for one preparation use `--submit-script`.
Use `--no-submit-script` to override an enabled default, and
`--submission-type local|slurm|pbs|lsf` to choose a configured template:

```yaml
submission:
  generate: true
  default: slurm
  abacus_command: "mpirun -np 8 abacus"
```

For the vibration workflow, the generated `submit_vibration.sh` runs the
equilibrium task first and submits all displacement tasks in parallel after it
finishes. The task template is applied to each generated task directory, so
site-specific scheduler directives can be customized without changing the
workflow implementation. Templates support `{abacus_command}`, `{job_name}`,
`{task_name}`, and `{workflow}` placeholders; literal shell braces must be
written as doubled braces.

## workflow fdforce and fdstress

Analytic forces and stresses can be checked independently with central finite
differences. Force validation accepts one-based atom indices, or the
`abacus-test`-compatible `info.txt` format (`C 2 x y z`):

```text
abacustools workflow fdforce prepare -j JOB --index 1 2 --dir x y z
abacustools workflow fdforce postprocess -j JOB
```

Stress validation defaults to the six independent cell components and keeps
fractional coordinates fixed while deforming the cell. Use `--all-components`
to include all nine components:

```text
abacustools workflow fdstress prepare -j JOB --step 0.0001 --number 5
abacustools workflow fdstress postprocess -j JOB
```

Both workflows write their generated calculations below `fdforce/` or
`fdstress/`, record the exact finite-difference cases in a manifest, and write
JSON results containing analytic values, finite-difference values, deviations,
and step-size convergence. Missing or unconverged ABACUS results stop
postprocessing with an explicit error.

## workflow workfunc

Surface work functions can be calculated from the averaged electrostatic
potential. The prepare stage enables `out_pot=2` and writes a calculation
under `workfunc_job`; the postprocess stage identifies vacuum plateaus and
writes the work-function results and potential profile. Both branch names of
that cube are read, `ElecStaticPot.cube` in the LTS branch and `potes.cube` in
develop, together with the `pot_es.cube` that the develop manual mentions:

```text
abacustools workflow workfunc prepare -j JOB
abacustools workflow workfunc postprocess -j JOB
```

Use `--vacuum a|b|c|auto` to select the vacuum direction, and
`--dipole-corr` to enable dipole correction.

## workflow bec

Born effective charges can be calculated with Berry-phase finite differences.
The workflow prepares SCF and three Berry-phase NSCF calculations for each
selected atom displacement:

```text
abacustools workflow bec prepare -j JOB --index 1 --dir x y z --type c
```

Run `run_bec.sh` in each generated `bec_*` directory, then postprocess the
polarization differences:

```text
abacustools workflow bec postprocess -j JOB
```

The BEC tensors and task diagnostics are written to `bec_results.json` under
`JOB`. Missing or incomplete Berry-phase task output is retained as missing
tensor entries so other completed displacement directions can still be reported.

## workflow dielectric

The clamped-ion (electronic) dielectric tensor `epsilon_inf` is the other half
of a polar material's non-analytical correction, and ABACUS does not write it.
It is obtained by a tight-binding Kubo-Greenwood sum over the Hamiltonian,
overlap and position matrices of an LCAO calculation, which the `pyatb` package
evaluates on a dense Brillouin zone grid. The prepare stage generates the one
self consistent calculation that has to be rerun for those matrices to appear:

```text
abacustools workflow dielectric prepare -j JOB
```

The generated `dielectric` directory is the source job with `out_mat_hs2`,
`out_mat_r` and `symmetry 0`, and `workflow_dielectric.json` records the task
and the keywords that were switched on, as every other workflow manifest does.
Run it, bring the `OUT.*` matrices back, and postprocess them:

```text
abacustools workflow dielectric postprocess -j JOB
```

The tensor is written to `dielectric_results.json` under `JOB` together with
the dense grid, the photon energy window, the spin channels, the occupied band
count and Fermi energy of the source calculation and the `pyatb` version, so
the number can be traced back to what produced it. `--grid`, `--omega`,
`--domega` and `--eta` set the sum, and `--workdir` moves the pyatb working
directory that holds the copied matrices and its own output. The sum runs
locally, on the machine that holds the matrices and the `abacustools`
environment, which needs the optional `pyatb` package and an MPI runtime:
`pip install 'abacustools[pyatb]'` and a conda MPI such as `mpich` provide both,
and `--pyatb-command 'mpirun -np 4 pyatb'` runs the sum out of process on
several ranks instead of in process.

The occupation of the sum is pinned to the `occupied bands` count that ABACUS
autosets and prints in its running log, not to a Fermi level. For an insulator
any level inside the gap gives the same occupation, ABACUS places its `EFERMI`
at one such value, and the transition energies the sum is built from are
differences that do not depend on the energy reference at all; the Fermi energy
is still written into the pyatb input, because pyatb asks for it, but it does
not enter the result. The photon energy window has to start at zero, since its
first row is the static limit, and to reach well above the band gap, since the
sum covers every transition: a window that sits inside the gap captures no
transition and leaves an empty spectrum, and for NaCl a 20 eV window gives a
tensor about one percent low. `--omega 0 80` is the default and a window that
starts above zero or ends below 40 eV is reported as a warning. The dense grid
of the sum is what the tensor converges with; the k mesh of the source SCF only
has to converge the density, and for NaCl 20x20x20 already agrees with
50x50x50 to five digits.

## workflow piezoelectric

Piezoelectric stress tensors can be calculated from finite-strain changes in
the Berry-phase polarization. The workflow generates the six independent
Voigt strain modes (`xx`, `yy`, `zz`, `yz`, `xz`, and `xy`), with forward,
backward, or central differences:

```text
abacustools workflow piezoelectric prepare -j JOB --strain 0.01 --type c
```

Run `run.sh` in each generated `piezoelectric_*` directory. Use `--relax` to
relax ionic positions at each strained cell before the Berry-phase
calculation. The tensor and per-task diagnostics are written under `JOB`:

```text
abacustools workflow piezoelectric postprocess -j JOB
```

The tensor is a polar third-rank tensor, so the point group of the reference
cell fixes how many of its 18 components are independent: one for `-43m`,
three for `6mm`, four for `3m`, none for a centrosymmetric crystal. The
preparation stage records the point group and the number of independent
components in `workflow_piezoelectric.json`, and `--strains independent`
prepares only the strain modes those components need - three of the six for a
wurtzite cell. The postprocessing stage symmetrises the fitted tensor with the
point group, which removes the components the symmetry forbids and enforces
the relations between the ones that survive:

```text
abacustools workflow piezoelectric prepare -j JOB --strains independent
abacustools workflow piezoelectric postprocess -j JOB --fit independent
```

Both the fitted tensor, the symmetrised tensor, the largest change the
symmetrisation made and the independent components are written to
`piezoelectric_results.json` under `JOB`; `--no-symmetrize` keeps the raw
components.

The strain is a Cartesian deformation of the lattice vectors with the
fractional coordinates held fixed, so the six modes are the strain tensor
components of the IEEE convention: a shear of nominal size `s` puts `s/2` in
each off-diagonal element, and `S_4 = 2 eps_yz` equals `s`. The reported shear
columns are therefore `dP/dS` with the same meaning as in the literature and in
DFPT codes, and a requested one percent shear is a one percent shear in the
same sense the elastic workflow uses. `--relax` adds one self consistent
relaxation of the ionic positions at each strained cell, which is what turns
the clamped-ion tensor into the relaxed-ion one; it respects the `force_thr_ev`
of the source `INPUT`, and tightening that value (the ABACUS default is 0.01
eV/Angstrom) matters because the internal strain it produces is exactly the
difference between the two.

ABACUS accepts `use_k_continuity` only for plane wave calculations that are not
a non self consistent run, and the Berry-phase steps of this workflow are
exactly such a run, so the option is off by default. `--use-k-continuity` is
kept for versions that lift the restriction; on LTSv3.10.1 it makes every
generated calculation stop with `use_k_continuity only works for PW basis`.

## workflow dftu

Hubbard `U` parameters can be derived from first principles with the linear
response method, which screens the occupation of the correlated orbitals
against a small applied `U`. The prepare stage writes one SCF calculation per
scanned `U` value, treating the atoms named by `--index` or `--elements` as the
correlated ones and giving them their own species label so that only that atom
carries the perturbation:

```text
abacustools workflow dftu prepare -j JOB --index 1 --u-min 0 --u-max 0.5 --u-step 0.1
abacustools workflow dftu postprocess -j JOB
```

`--u-values` lists the scanned values directly, `--orbital 2|3` selects d or f
orbitals instead of inferring the angular momentum from the element, and
`--dftu-type` picks the DFT+U flavour, defaulting to the recommended method 1.
The postprocess stage reads the local occupation matrices from every
`running_scf.log`, fits the bare and screened responses, and writes the
resulting `U` per atom to `dftu_results.json` together with a `dftu_response.png`
plot. Submit scripts are generated on request with `--submit-script` and
`--submission-type`.

## workflow exchange

Magnetic exchange coupling constants can be calculated with the four-state
method, which maps the total energy of a magnetic pair onto the Heisenberg
form. Prepare one reference calculation together with three tilted
magnetization configurations per tilt angle: only the first moment tilted,
only the second one tilted, and both moments tilted by half the angle, so all
three share the same pair angle. Pairs are given on the command line or in a
`magj.txt` file that holds one `LABEL1 INDEX1 LABEL2 INDEX2` line per pair,
where the indices are one-based within the atoms of that label:

```text
abacustools workflow exchange prepare -j JOB --pair Fe 1 Fe 2
abacustools workflow exchange prepare -j JOB --step 10 --number 5
abacustools workflow exchange prepare -j JOB -f pairs.txt
abacustools workflow exchange postprocess -j JOB
```

The `exchange` workflow is also available as `magj`, the name of the
`abacus-test` model it follows. The input job needs noncollinear magnetism
(`nspin 4`), and both atoms of a pair need a magnetic moment: a scalar moment
counts as a moment along `z`, a vector `mag x y z` keeps its direction. The
generated `STRU` files keep the magnitudes of the reference moments and tilt
them in the plane of the reference pair. `--step` is the tilt step in degrees
and `--number` the number of tilt angles, so the defaults of 1 degree and five
steps test 1 to 5 degrees, as the reference implementation does; larger tilts
raise the energy difference that is fitted and therefore give a better
conditioned result. Repeating `--pair` calculates several pairs in one go, and
a pair that is requested in both orders, or twice, is calculated once because
the two atoms are symmetric in the method.

Postprocessing combines the four energies into

```text
dE = (E_both - E_atom1) - (E_atom2 - E_original)
```

and fits `dE = J (1 - cos(theta))`, so the slope `J` is positive for
antiferromagnetic coupling. The fit is reported per pair together with its
coefficient of determination and its residual, which show whether the energy
difference follows the Heisenberg form. Results are written to
`exchange_results.json`, with the energies, angles and four-state differences
of every point, and the fit is plotted to `exchange_fit.png`. The fit uses the
angles of the configurations that were generated, so a run whose moments
rotate away from the prescribed directions is best checked with
`postprocess result -p atom_orb_mag` before its coupling constant is used.
Postprocessing stops when a calculation did not reach `scf_thr`; some magnetic
configurations converge their energy while the density error keeps oscillating,
so `--allow-unconverged` fits them anyway and lists the affected calculations
in `unconverged_tasks` next to a `converged` flag on every fitted point.
Because the moments keep their reference magnitudes, `J` is the coefficient of
the unit directions of the two moments times the product of their magnitudes;
divide by that product to obtain a coupling constant per unit moment.
Tilted configurations converge more slowly than the reference one, so the
per-point `delta_energy_ev` and `x` of the JSON are worth a look: every tilt
should give the same slope, and a tilt that does not is a calculation that has
not reached the configuration it was given. The generated calculations live
below `magj/`, with one directory per tilt angle and path, and the reference
calculation in `magj/original`.

## Workflow manifests and generated files

Generated calculation directories are protected by default. Use `--override`
when intentionally replacing them. The BEC workflow's `run_bec.sh` is only a
local four-step runner for one generated task; cluster submission scripts are
not generated because their contents depend on the target computing environment.

Each prepared workflow records its task names and atom partition in a
workflow-specific manifest such as `workflow_phonon.json`. Postprocessing
validates this manifest and checks that the required SCF calculations converged
before reading their outputs.
