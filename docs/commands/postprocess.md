# Postprocess commands

[Documentation home](../index.md)

The `postprocess` family reads completed ABACUS jobs. The aliases `post` and `pp` are equivalent to `postprocess`.

```text
abacustools postprocess bader -j JOB
abacustools post bader -j JOB
abacustools pp bader -j JOB
```

## postprocess result

Calculation results can be collected from one or more completed ABACUS jobs:

```text
abacustools postprocess result -j JOB -p energy force stress
```

Use `abacustools postprocess result --help` to see all supported result
parameters. If `--param` is omitted, the command selects the results relevant
to the job's calculation type and displays scalar results as compact,
borderless tables. If the result columns do not fit the terminal, they are
split into multiple tables with repeated headers rather than being truncated
or wrapped by the terminal. A result that does not apply to a job is shown as
`-`. Large array results such as `force` and `stress` are omitted from this
summary. Use `--json` to print the collected results, including the array
results, as JSON.

Magnetic moments are collected with their own parameters, which are not part
of the default selection:

```text
abacustools postprocess result -j JOB -p total_mag absolute_mag
abacustools postprocess result -j JOB -p atom_mag_mulliken atom_orb_mag --json
```

`total_mag` and `absolute_mag` are the total and absolute magnetization of the
cell in Bohr magneton, taken from the last electronic iteration of the running
log; a noncollinear calculation reports the three Cartesian components of
`total_mag` as a list. `atom_mag_mulliken` lists the magnetization of every
atom of the last ionic step as written by the Mulliken analysis
(`mulliken.txt`), and `atom_orb_mag` lists the orbital-projected magnetization
of every atom from the last orbital charge analysis block of the log. Both are
`None` when the run prints no such analysis.

## postprocess band

Band structures from ABACUS NSCF calculations can be processed and plotted
from a job directory containing `BANDS_1.dat`:

```text
abacustools postprocess band -j JOB --emin -8 --emax 8
```

The command reads the line-mode `KPT`, shifts the bands by the Fermi energy
from the NSCF log, and writes `band.png`, `band.dat` (or spin-resolved files),
and `KPATH.txt` below `JOB`. Use `--efermi` to provide an explicit Fermi
energy when it is not present in the log. Only `calculation nscf` jobs are
recommended; other calculation types emit a warning and are still processed.

Band-gap, effective-mass, and projected (fat) band analysis use the same
command:

```text
abacustools postprocess band -j JOB --gap
abacustools postprocess band -j JOB --gap --json
abacustools postprocess band -j JOB --effective-mass cbm --direction G X
abacustools postprocess band -j JOB --fat-band species
```

`--gap` reports the band gap, VBM, CBM, and whether the gap is direct or
indirect (`--spin-resolved` adds per-spin gaps; `--json` prints the report as
JSON). `--effective-mass cbm|vbm` with `--direction START END` fits the band
curvature at the edge and is available for nspin=1 only.
`--fat-band species|species-shell|species-orbital|atoms` plots the projected
band structure from the `PBANDS_*` output.

## postprocess cohp

COHP and COOP curves can be calculated for atom pairs when the LCAO output
contains `data-*-H`, `data-*-S`, `kpoints`, and `WFC_NAO_K*.txt` files. Atom
indices are one-based, and every pair is automatically expanded to the complete
NAO range of its two atoms from `STRU` and the numerical orbital files:

```text
abacustools postprocess cohp -j JOB --pairs 1-2 \
  --method COHP --de 0.1 --output cohp.png --data-output cohp.dat
abacustools postprocess cohp -j JOB --pairs 1-2,1-3
abacustools postprocess cohp -j JOB --pairs-file pairs.txt
abacustools postprocess cohp -j JOB --cutoff 3.0
```

`--pairs`, `--pairs-file` and `--cutoff` have the same atom-pair semantics as
`postprocess mayer`. A cutoff uses the shortest periodic distance in Angstrom.
When several pairs are selected, each pair gets its own plot and data file with
a `_1-2` style suffix, and the printed ICOHP/ICOOP labels identify the atom
pair. The resolved zero-based NAO indices are also recorded on the data result
and printed with the curve.

The older advanced interface remains available for arbitrary orbital groups
through `--atom-i-orbs` and `--atom-j-orbs`; those indices are zero-based
global NAO indices. The atom-pair selectors and the explicit orbital groups are
mutually exclusive. Use `--method COOP` for overlap-weighted curves,
`--spin up|down` for a spin channel, and `--invert` to invert only the plotted
curve. The data file keeps the computed sign. Develop-version output is read as
well: `hk*_nao.txt` and `sk*_nao.txt` replace the `data-*-H`/`data-*-S` pair,
`wf*_nao.txt` replaces `WFC_NAO_K*.txt`, and the k-point weights come from the
`dm*_nao.txt` density-matrix headers. The WFC occupations already contain the
k-point weight and, for `nspin 1`, the spin degeneracy, so the curve is weighted
by those occupations without applying the k-point weight a second time. COHP
uses the Hamiltonian in Ry and is converted to eV before writing; COOP uses the
dimensionless overlap and stays dimensionless. `ICOHP` (eV) and `ICOOP` are
evaluated as the occupied-band sum up to the Fermi level, independently of the
plotting grid. This is an ABACUS-NAO COHP/COOP implementation and is not a
standard LOBSTER pCOHP projection.

## postprocess qtaim

An approximate QTAIM bond-critical-point search is available for total-density
cubes and the valence density of an ABACUS job:

```text
abacustools postprocess qtaim -j JOB
abacustools postprocess qtaim -j JOB --core-radius 0.4 --cutoff 3.5
abacustools postprocess qtaim -j JOB --density all_electron.cube --json
```

The command samples the minimum-image segment of every atom pair within
`--cutoff` (4.0 Angstrom by default), takes density minima outside spherical
core-exclusion regions as seeds, and refines them by a three-dimensional
Newton search for `grad(rho) = 0`. A point is reported only when its Hessian
has the `(3,-1)` BCP signature and it remains outside every core sphere.
The report contains the BCP position, `rho`, gradient, Hessian eigenvalues,
Laplacian and ellipticity in atomic units. `--core-radius` defaults to 0.3
Angstrom; `auto` uses the largest UPF non-local-projector cutoff as a more
conservative radius.

This is deliberately an approximate QTAIM mode. Norm-conserving ABACUS
pseudopotentials normally write a valence pseudo-density, so the missing core
density is excluded from the search rather than reconstructed. Use
`--density` with an all-electron or validated core-augmented cube for
publication-quality QTAIM. The command does not compute ring/cage critical
points, bond paths, atomic basins or basin-integrated properties.

## postprocess molden

An ABACUS LCAO wavefunction can be exported to the Molden format, for example
to visualize the orbitals in Molden, Multiwfn or Avogadro:

```text
abacustools postprocess molden -j JOB
abacustools postprocess molden -j JOB -o orbitals.molden --gto-primitives 8
abacustools postprocess molden -j JOB --atoms-unit angstrom --json
```

The command reads the LCAO coefficients from `WFC_NAO_GAMMA*` (or the
develop `wf*_nao.txt` names and the `WFC_NAO_K*` files), expands every
numerical atomic orbital into a contracted Gaussian fit, and writes
`wfc.molden` below `JOB`. Valence counts for the `[Nval]` block come from the
pseudopotentials, and `[5D7F]`/`[9G]` are written whenever the basis reaches
that angular momentum, so the pure spherical harmonics of ABACUS are kept.
`nspin 2` jobs write both spin channels into one file.

The Molden format stores a single real set of orbitals, so the job must be a
`gamma_only 1` run, or a non-gamma run whose selected k-point is real; use
`--kpoint` to pick one k-point of the latter. `--gto-primitives` sets the
number of Gaussians per numerical orbital (default 6) and the printed report
quotes the largest relative radial fit error.

## postprocess dos

DOS and projected DOS can be processed from an ABACUS output directory:

```text
abacustools postprocess dos -j JOB --emin -20 --emax 10
abacustools postprocess dos -j JOB --pdos species
```

The command reads `DOS*_smearing.dat` (ABACUS LTS) or the `doss<spin>...txt`
and `pdoss<spin>...txt` text files of ABACUS develop, shifts the energy axis
by the Fermi energy, and writes `DOS.png` and `DOS.dat`. Use `--pdos
species-shell`, `--pdos species-orbital`, or `--pdos atoms --atom-index 1 2`
for projected DOS plots. Output paths can be changed with `-o` and
`--data-output`; use `--efermi` when the Fermi energy cannot be read from the
ABACUS output.

Use `--combined` to overlay the total DOS with the species-projected DOS
(`DOS_PDOS.png`/`DOS_PDOS.dat`), `--pdos atom-shell` or `--pdos atom-orbital`
with `--atom-index` for single-atom projections, and `--list` (optionally with
`--json`) to list the available species, shells, orbitals and atoms.

## postprocess chg

Charge densities can be inspected without leaving the command line. The
command reads the density of a job, reports the integrated electrons next to
the valence electrons of the atoms, and writes it as a cube, a planar profile
or a two-dimensional slice:

```text
abacustools postprocess chg -j JOB
abacustools postprocess chg -j JOB --cube charge.cube
abacustools postprocess chg -j JOB --spin difference --cube magnetization.cube
abacustools postprocess chg -j JOB --difference OTHER_JOB --cube bonding.cube
abacustools postprocess chg -j JOB --profile c
abacustools postprocess chg -j JOB --slice c --slice-index 0.5 --slice-plot
abacustools postprocess chg -j JOB --json
```

`--spin` selects what every action works on: the total density, the up or down
channel of an `nspin 2` calculation, or their difference, whose integral is the
magnetization in Bohr magneton. `--difference OTHER_JOB` subtracts another job
before the analysis and turns the command into a general density-difference
tool, as long as both grids match, so a bonding or adsorption difference no
longer needs a prepared `workflow chgdiff` set.

The density is taken from the cubes that an SCF calculation writes with
`out_chg 1`, and both ABACUS naming conventions are read: the LTS branch writes
`SPIN1_CHG.cube` while develop writes `chg.cube` or `chgs1.cube`, and a
geometry-step token such as `chgs1g3.cube` is recognised, with the last step
used when the job wrote one file per step. A job that only stores the
`*-CHARGE-DENSITY.restart` backup, which is what `out_chg 0` leaves behind, is
converted from `rho(G)` instead: the structure and its pseudopotentials give
the cell and the valence charges, and the FFT grid comes from the log of the
current calculation, with `--grid NX NY NZ` as an override. A `gamma_only`
restart file, the usual case at the Gamma point, stores one G-vector of every
`+G`/`-G` pair, and the missing half is rebuilt with `rho(-G) = conj(rho(G))`.
The summary compares the
integrated charge with the valence charge of the atoms that the cube stores, so
a deviation reports either a charged cell or the truncation of the real-space
grid; the comparison is left out for a spin channel or a difference, where it
has no meaning.

`--quantity` moves the analysis to a field derived from the selected density,
with the definitions of Quantum ESPRESSO's `pp.x`: `rdg` is the reduced density
gradient (`plot_num=19`), `sl2rho` is `sign(lambda_2) rho` built from the middle
eigenvalue of the density Hessian (`plot_num=20`), and `dori` is the density
overlap regions indicator (`plot_num=123`). The derivatives are evaluated in
reciprocal space, as `pp.x` does, so the two codes agree on the same grid, and
`--cube`, `--profile` and `--slice` work on the derived field as well.
`--nci-plot` draws the non-covalent interaction plot of the density, the
reduced density gradient against `sign(lambda_2) rho`, keeping the grid points
below `--nci-rho-max` (`0.05` e/Bohr^3 by default) so that the cores and the
bonds stay out of the plot.

`iri` is the interaction region indicator of Multiwfn, `|grad rho| / rho**1.1`,
which shows covalent and non-covalent interactions in one function; points
below `5e-5` e/Bohr^3 are replaced by zero instead of the placeholder Multiwfn
uses. The `-promolecular` variants and `dg` build the promolecular reference
from the `PP_RHOATOM` table of the same UPF files that the calculation used:
the reference is the superposition of the pseudoatomic densities of the atoms,
and `dg` is the independent gradient model function `sum_A |grad rho_A| -
|grad rho|`, whose plot against `sign(lambda_2) rho` comes from `--igm-plot`.
These three need the structure and its pseudopotentials, so they cannot be
combined with `--difference`, and `--promolecular-plot` draws the NCI plot of
the reference density itself.

The Hirshfeld-partitioned independent gradient model, IGMH, replaces the frozen
promolecular atomic densities with `rho_A = w_A rho`, where the weights come
from the Hirshfeld partition of the calculated density. The `igmh-i` variant
uses the self-consistent Hirshfeld-I weights instead:

```text
abacustools postprocess chg -j JOB --quantity igmh --igmh-plot
abacustools postprocess chg -j JOB --quantity igmh-i --igmh-plot
```

Both quantities are written in the same units as `dg` and can be exported with
`--cube`, `--profile` or `--slice`. The partition uses the total density of the
job and the same pseudopotential references as `postprocess hirshfeld`, so these
analyses require `--spin total`, cannot be combined with `--difference`, and
need a pseudopotential with `PP_RHOATOM` for Hirshfeld; the Hirshfeld-I variant
additionally needs `PP_PSWFC`. The data API also accepts explicit reference
densities for that variant.
Without `--igmh-plot`, `igmh` and `igmh-i` are still available as fields through
`--quantity`.

`--profile AXIS` writes the in-plane average of every plane in e/Angstrom^3 to
`chg_profile_<axis>_average.dat`, or the charge of every plane in e with
`--profile-kind integral`, whose sum is the total number of electrons.
`--slice AXIS` writes the plane closest to the fractional `--slice-index` (0.5
by default) as three columns with the two in-plane axes and the value, and
`--slice-plot` draws it as a colour map together with the atoms that the plane
crosses; `--no-atoms` leaves those markers out and `--vmin`/`--vmax` fix the
colour range. `--data-output`, `--slice-output`, `--plot` and `--slice-plot`
change the file names, and a plot flag without a name writes
`chg_profile_<axis>_<kind>.png` or `chg_slice_<axis>_<position>.png`.

## postprocess md

Molecular-dynamics trajectories are written to standard formats. ABACUS
appends one block per dumped step to `OUT.<suffix>/MD_dump`, holding the cell,
the positions and, when `dump_force`, `dump_vel` and `dump_virial` are enabled,
the forces, the velocities and the virial; `postprocess md` turns those blocks
into a trajectory file, with the energy, temperature and pressure of every step
attached from the running log:

```text
abacustools postprocess md -j JOB
abacustools postprocess md -j JOB -o traj.extxyz --first 100 --last 2000 --stride 5
abacustools postprocess md -j JOB -o trajectory.traj --json
```

The suffix of the output selects the format, which is any format ASE writes,
such as `extxyz`, `xyz`, `traj` or `xsf`, and `--format` sets it explicitly.
Positions are in Angstrom, forces in eV/Angstrom, velocities in Angstrom/fs and
the virial in kBar, as in `MD_dump` itself. A job that wrote no `MD_dump` is
read from its per-step `STRU_MD_*` structures instead, which is what `out_stru 1`
produces in the job directory of the LTS branch and in a directory per step of
develop; those frames carry positions and velocities but no forces or virial.

## postprocess traj

Geometry optimizations are written from the per-step structures that
`out_stru` produces. The 3.10 LTS branch writes `STRU_ION<step>_D` for every
ionic step of a `relax` or `cell-relax` job, while the develop branch writes
`STRU<step>` every `out_freq_ion` steps, so set `out_stru 1` and a positive
`out_freq_ion` to keep them. `postprocess traj` collects those structures in
step order and attaches the energy of every step from `running_relax.log` or
`running_cell-relax.log`; when the per-step files are absent it rebuilds the
frames from the coordinates, cell, forces and stress that the running log
prints instead:

```text
abacustools postprocess traj -j JOB
abacustools postprocess traj -j JOB -o relax.extxyz --stride 2
abacustools postprocess traj -j JOB --first 1 --last 50 --json
```

The forces and stress of every step are read from the `TOTAL-FORCE` and
`TOTAL-STRESS` tables of the running log, so even the LTS files, which hold
positions and the cell only, end up with both; the stress is written into the
trajectory in ASE's units and sign. The develop per-step files also carry
forces themselves, which the log fills in only when a frame has none.
`--no-energy` skips only the energy, and `postprocess md` reads a
molecular-dynamics job the same way.

## postprocess mayer

Mayer bond orders can be analyzed from an ABACUS LCAO calculation. For the
develop layout the density-matrix outputs are preferred: `out_dmk=1` writes
DM(k), and `out_dmr=1` writes DM(R). The older LTS layout uses
`out_mat_hs=1` (and `out_dm=1` for gamma-only jobs). H(R)/S(R) remains a
fallback through `out_mat_hs2=1` or `out_hsr=1`:

```text
abacustools postprocess mayer -j JOB --cutoff 3.0
abacustools postprocess mayer -j JOB --pairs 1-2,1-3 --json -o mayer.json
abacustools postprocess mayer -j JOB --pairs 1-2 --shells
```

With `--shells` every bond order is decomposed into the contributions of the
s/p/d/f shells of the two atoms, for example `s-p 0.0512  p-p 0.9031`. The
same breakdown is always part of the JSON report as `shell_bond_orders`
(shell of the first atom, shell of the second atom, contribution). The
contributions of radial functions with the same angular momentum are summed,
and all values add up to the pair bond order.

The analyzer reads numerical orbitals, overlap matrices, density matrices or
NAO wavefunctions, validates their dimensions, and reports atom indices,
elements, periodic distances, and Mayer bond orders. It supports gamma-only
and multi-k calculations with `nspin=1` or `nspin=2`, for the LTS output
layout (`data-*-S` with `SPIN1_DM`/`SPIN2_DM` or `WFC_NAO_K*.txt`), the
develop density-matrix layout (`dm*_nao.txt` with `out_dmk`, or
`dmrs*_nao.csr` with `out_dmr`), and gamma-only ABACUS v3.11 H(R)/S(R) CSR
output (`sr_nao.csr` with `hrs*_nao.csr` and `eig_occ.txt`). For the H(R)/S(R)
fallback the density matrices are rebuilt from the Hamiltonian, overlap and
occupations; DM(R) is used directly when it contains a single R block.

A run that reduced the k-point mesh (`symmetry 1`) is analyzed on the full
mesh. The analyzer reads the mesh and the irreducible k-points from
`OUT.*/kpoints`, rebuilds the space-group operations from the structure, and
evaluates every star member through the atom permutation of the operation that
reaches it. A bond order is quadratic in the k-resolved density matrix, so the
star cannot be folded into a k-point weight; expanding it is what makes a
`symmetry 1` run agree with the `symmetry 0` and `symmetry -1` runs of the same
calculation, which the analyzer reads directly. The weights are rebuilt from
the star sizes as well, because the `kpoints` file prints them with four
decimals only, which would bias the totals by about 0.2%.

An output that does not record how the mesh was reduced - an older format, or a
magnetic `symmetry 2`/`symmetry 3` run whose reduction is not the
crystallographic one - cannot be expanded; the analyzer says so and asks for a
`symmetry 0` or `symmetry -1` calculation instead.

## postprocess bader

Bader charges are integrated over the Bader volumes of a job with the external
[Henkelman](https://theory.cm.utexas.edu/henkelman/code/bader/) program or with
the Python [baderkit](https://github.com/SWeavz/baderkit) library:

```text
abacustools postprocess bader -j JOB
abacustools postprocess bader -j JOB --backend baderkit
abacustools postprocess bader -j JOB --reference OTHER.cube --vacuum auto --keep-cubes
abacustools postprocess bader -j JOB --json -o bader.json
```

Both backends partition the same density, which comes from the `SPIN*_CHG.cube`
of `out_chg 1` or from the `*-CHARGE-DENSITY.restart` backup, converted from
`rho(G)` with the FFT grid of the running log (`--grid NX NY NZ` overrides it)
and the lattice constant of the STRU (`--lat0`). `--cube` selects another cube
and `--reference` partitions with another density. `--vacuum off|auto|DENSITY`
follows the flag of the external program, where `auto` is the 1e-3
e/Angstrom^3 cutoff. An `nspin 2` job integrates the magnetization over the
Bader volumes of the total density, which is what gives the per-atom spin
moments in Bohr magneton.

The external program is resolved from `--bader-exe`, the `BADER_EXE`
environment variable or `bader.exe` in `~/.abacustools/config.yaml`, and writes
its `ACF.dat` into the working directory that `--workdir` or `--keep-cubes`
preserves. The `baderkit` backend needs the optional package
(`pip install abacustools[baderkit]`) and runs the partition in the Python
process instead, so it needs no executable. `--baderkit-method` selects one of
its `neargrid`, `neargrid-weight`, `ongrid` and `weight` algorithms; the
default `neargrid` is the partitioning the external program applies, which
makes the two backends agree to a few 1e-3 e on the same cube.

Charges are reported next to the valence electron count of the pseudopotential,
so `net charge = z_valence - bader_charge` is positive for an electron-poor
atom. Positions and distances are in Angstrom and volumes in Angstrom^3, for
both backends.

## postprocess hirshfeld

Hirshfeld (stockholder) atomic charges are computed directly from the
charge density of a job, using the spherically averaged free-atom densities of
the pseudopotentials as the promolecule:

```text
abacustools postprocess hirshfeld -j JOB
abacustools postprocess hirshfeld -j JOB --json
```

The density is read from the charge-density cube or the `*-CHARGE-DENSITY.restart`
file, and the proatoms are summed over lattice images so the promolecule is
periodic. CM5 charges (Marenich, Jerome, Cramer and Truhlar, *J. Chem. Theory
Comput.* 2012, 8, 527) add Pauling-bond-order weighted pairwise corrections to
the Hirshfeld charges; the parameters of the paper's Table 1 and its covalent
radii are built in, so no extra file is needed:

```text
abacustools postprocess hirshfeld -j JOB
abacustools postprocess hirshfeld -j JOB --no-cm5
```

Hirshfeld-I (Bultinck, Van Alsenoy, Ayers and Carbo-Dorca, *J. Chem. Phys.*
2007, 126, 144111) makes the promolecule self-consistent instead of fixing it
to the neutral atoms.  Each iteration rebuilds every atomic reference density at
the population the previous iteration assigned to the atom, which removes the
dependence on an arbitrary reference and gives charges that track the
electrostatic-potential ones much better.  The reference densities at integer
populations come from the pseudo-atomic wavefunctions (`PP_PSWFC`) of the
pseudopotentials, filled by Aufbau around the neutral configuration and summed
over lattice images like the plain promolecule:

```text
abacustools postprocess hirshfeld -j JOB --hirshfeld-i
abacustools postprocess hirshfeld -j JOB --hirshfeld-i --json
abacustools postprocess hirshfeld -j JOB --hirshfeld-i --max-iter 300 --tol 1e-4
```

A pseudopotential without `PP_PSWFC` (many ONCV files) has no charged reference
states, so pass a directory of reference densities with `--references DIR`
instead.  Every `<element>_<population>.dat` file there holds two columns, `r`
in Angstrom and `rho(r)` in `e/Angstrom^3`, with one file per integer valence
population the iteration may need.

## postprocess ddec

DDEC6 and DDEC3 net atomic charges, spin moments and bond orders are
computed with the external [Chargemol](https://ddec.sourceforge.net) program,
which partitions the valence density of a job:

```text
abacustools postprocess ddec -j JOB
abacustools postprocess ddec -j JOB --charge-type DDEC3 --json -o ddec.json
abacustools postprocess ddec -j JOB --no-spin --threshold 0.1
abacustools postprocess ddec -j JOB --pairs 1-2,1-3 --cutoff 3.0
abacustools postprocess ddec -j JOB --no-bos --threads 16
abacustools postprocess ddec -j JOB --core-electrons "26 10" --workdir ddec
```

The command writes the `valence_density.cube` (and the `spin_density.cube` of
an `nspin 2` job) that the program expects, together with its
`job_control.txt`, into a working directory, runs Chargemol there and parses
the `*.xyz` results, so the tables list the net charge, the sum of bond orders
and, for a spin-polarized job, the DDEC spin moment of every atom next to the
bond orders and their periodic images. `--workdir` and `--keep` preserve that
directory, `--output` writes the complete report as JSON and `--json` prints
it. The executable and the reference density tables are set by
`chargemol.exe` and `chargemol.atomic_densities` in
`~/.abacustools/config.yaml`, and can also come from the `CHARGEMOL_EXE` and
`CHARGEMOL_ATOMIC_DENSITIES` environment variables or from
`--chargemol-exe` and `--atomic-densities`.

A valence-only cube makes Chargemol insert the core electrons from its
`atomic_densities` tables, which fixes the number of core electrons of every
element to `Z - z_valence` of the pseudopotential; the command derives that
number from the atom columns of the density and checks the tables before the
program starts, so the trivalent lanthanide pseudopotentials of the APNS
library, whose 4f electrons sit in the core, or any other element whose core
count the distribution does not ship, are reported instead of failing inside
Chargemol. `--core-electrons "26 10"` overrides the count of one element. The
charge density should come from `out_chg 1 10` so that the cube carries enough
digits, its grid has to be finer than 0.25 Bohr per direction (0.14 Bohr is
the recommended spacing), and a charge-density cube of a PAW or ultrasoft
calculation does not integrate to the valence charge, so only norm-conserving
pseudopotentials work. A molecule in a box wants
`--periodicity false false false`.

The net charge of the cell follows the `nelec` and `nelec_delta` keywords of
INPUT: `nelec 0`, the default, means that the electrons are the sum of the
valence charges of the atoms, so the cell is neutral, while a positive value
describes a charged cell, and `--net-charge` overrides the derived value. The
command also compares the electrons of the cube with the charge of the cell
before Chargemol runs and stops with an explanation when they disagree, which
happens when the cube and INPUT belong to different calculations or when the
density comes from another grid or another `--cube`.

A job that kept no cube is read from its `*-CHARGE-DENSITY.restart` file with
the same conversion that `postprocess chg` uses: the finest FFT grid the
running log reports, the valence charges of the pseudopotentials and the
`LATTICE_CONSTANT` of STRU, with `--grid` and `--lat0` as overrides. Both
routes agree to about 1e-4 e, so a job whose `out_chg` was turned off can still
be analysed.

Bond orders and overlap populations are the expensive part of a Chargemol run:
the time grows with the number of atoms times the number of grid points. On 16
threads, 64 atoms on a 135^3 grid (2.5 million points) took 15 s, 224 atoms on
a 256x250x150 grid (9.6 million points) 66 s and 224 atoms on a 320x320x180
grid (18 million points) more than two minutes, while the same job with
`--no-bos` finished in 109 s. The charges and spin moments are identical with
and without bond orders, so a large system is best analysed with `--no-bos`
first. `--threads` sets `OMP_NUM_THREADS` of the OpenMP binary, and one
Chargemol instance should run at a time.

Chargemol 3.5 crashes on a valence-only cube when a spin density is present,
because `module_format_valence_cube_density` never allocates the arrays that
its spin reader uses. The command recognises the crash, explains it and
suggests either a patched build of Chargemol or `--no-spin`.
