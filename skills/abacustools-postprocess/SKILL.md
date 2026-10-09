---
name: abacustools-postprocess
description: Extract and analyse results from finished ABACUS calculations with the abacustools postprocess family. Covers energy/force/stress and convergence parameters, band structures, band gaps and effective masses, fat bands, DOS and projected DOS, COHP/COOP, Mayer bond orders, Bader/DDEC/Hirshfeld/CM5/Hirshfeld-I charges, charge-density cubes/profiles/slices, NCI/IGM/IGMH fields, Molden export, and MD trajectories. Use when OUT directories, logs, or density/matrix/wavefunction files already exist; use abacustools-workflows when a property needs several new calculations.
---

# Post-processing ABACUS results

Every command takes a job directory with `-j/--job`, writes its products below
that directory unless a path is given, and accepts `--json` where a structured
report makes sense. The banner caveat applies: an `--json` payload starts at
line 10 of stdout. Paths given to `-o`, `--data-output`, `--cube`, and friends
are resolved relative to the job directory.

## Scalar results from the logs

```text
abacustools postprocess result -j JOB
abacustools postprocess result -j JOB -p energy force stress drho efermi converged
abacustools postprocess result -j JOB -p total_mag absolute_mag --json
abacustools postprocess result -j JOB -v 3.10.1LTS
```

`-j` accepts several directories. Without `-p/--param` the parameters relevant
to each job's calculation type are selected and printed as compact tables;
large arrays such as `force` and `stress` are left out of that summary but are
included in `--json`. Available parameters: `energy`, `denergy`, `drho`,
`efermi`, `converged`, `normal_end`, `relax_converged`, `relax_steps`,
`scf_steps`, `force`, `largest_force`, `stress`, `largest_stress`,
`vdw_energy`, `total_mag`, `absolute_mag`, `atom_mag_mulliken`,
`atom_orb_mag`. A parameter that does not apply is shown as `-`.
`-v/--version` selects the ABACUS log profile (see the
`abacustools-job-lifecycle` skill); `auto` detects it from the running log.

## Band structure, gap, effective mass, fat bands

```text
abacustools postprocess band -j JOB --emin -8 --emax 8
abacustools postprocess band -j JOB --gap --json
abacustools postprocess band -j JOB --effective-mass cbm --direction G X
abacustools postprocess band -j JOB --fat-band species
abacustools postprocess band -j JOB --fat-band atoms --atom-index 1 2
```

Reads `BANDS_1.dat` (spin-resolved files for `nspin 2`) and the line-mode
`KPT`, shifts by the Fermi energy from the NSCF log (`--efermi` overrides), and
writes `band.png`, `band.dat`, and `KPATH.txt` (paths settable with
`-o`/`--data-output`/`--kpath-output`). It is meant for `calculation nscf`
jobs; other types warn and are still processed. `--gap` reports gap, VBM, CBM
and direct/indirect character (`--spin-resolved` adds per-spin gaps), and
`--fat-band species|species-shell|species-orbital|atoms` needs the `PBANDS_*`
output. `--effective-mass cbm|vbm` with `--direction START END` fits the band
curvature (`--fit-points`, default 5) and works for `nspin 1` only.

## DOS and projected DOS

```text
abacustools postprocess dos -j JOB --emin -20 --emax 10
abacustools postprocess dos -j JOB --pdos species --combined
abacustools postprocess dos -j JOB --pdos atoms --atom-index 1 2
abacustools postprocess dos -j JOB --list --json
```

Reads `DOS*_smearing.dat`, shifts by the Fermi energy, writes `DOS.png` and
`DOS.dat`. `--pdos` accepts `species`, `species-shell`, `species-orbital`,
`atom-shell`, `atom-orbital`, and `atoms`; `--combined` overlays total and
projected DOS in `DOS_PDOS.*`. `--list` reports which species, shells, orbitals
and atoms are available, which is the reliable way to build the selectors.

## COHP / COOP

```text
abacustools postprocess cohp -j JOB --atom-i-orbs 0,1,2 --atom-j-orbs 13,14,15 \
  --method COHP --de 0.1 --output cohp.png --data-output cohp.dat
```

Orbital indices are zero-based global NAO indices. `--method COOP` gives the
overlap-weighted curve, `--spin up|down` a single channel, `--invert` flips the
plotted curve only (the data file keeps the computed sign), and
`--no-smooth`/`--smooth-nstddev` control the Gaussian smoothing (`--width` sets
the horizontal plot limit). Needs an LCAO calculation with the
Hamiltonian/overlap and wavefunction outputs (`data-*-H`/`data-*-S`/
`WFC_NAO_K*.txt`, or the develop names `hk*_nao.txt`, `sk*_nao.txt`,
`wf*_nao.txt`). This is ABACUS-NAO COHP/COOP, not a LOBSTER projection.
`ICOHP`/`ICOOP` is the integral up to the Fermi level.

## Bond orders

```text
abacustools postprocess mayer -j JOB --cutoff 3.0
abacustools postprocess mayer -j JOB --pairs 1-2,1-3 --json -o mayer.json
```

Needs an LCAO run with `out_mat_hs 1` (and `out_dm 1` for gamma-only). Pairs
are one-based; `--pairs-file` reads one pair per line. A `symmetry 1` run is
expanded to the full mesh with the star members of every irreducible k-point,
so it agrees with `symmetry 0`. An old or magnetic (`symmetry 2`/`3`) output
that does not record the reduction cannot be expanded and is reported as such;
ask for a `symmetry 0`/`-1` calculation in that case.

## Charges: Bader, DDEC

```text
abacustools postprocess bader -j JOB --json
abacustools postprocess bader -j JOB --backend baderkit --baderkit-method neargrid
abacustools postprocess bader -j JOB --reference OTHER.cube --vacuum auto --keep-cubes
abacustools postprocess ddec -j JOB --charge-type DDEC3 --json -o ddec.json
abacustools postprocess ddec -j JOB --no-bos --threads 16
abacustools postprocess ddec -j JOB --core-electrons "26 10" --workdir ddec --keep
abacustools postprocess ddec -j JOB --net-charge 2 --periodicity true true true
```

`bader --backend` chooses the external Henkelman program (default) or the
optional `baderkit` library, whose `--baderkit-method` (`neargrid`,
`neargrid-weight`, `ongrid`, `weight`) matches the external partition by
default. Both need a density cube from `out_chg 1` or the
`*-CHARGE-DENSITY.restart` backup; `--grid`/`--lat0` override the FFT grid and
lattice constant when they cannot be read.
`ddec` runs Chargemol, generates `valence_density.cube` (+ spin) and
`job_control.txt` in a working directory, and parses its `*.xyz`. Both take the
executable and reference tables from `~/.abacustools/config.yaml`, the
`BADER_EXE`/`CHARGEMOL_EXE`/`CHARGEMOL_ATOMIC_DENSITIES` environment variables,
or their own flags. Bond orders dominate the DDEC cost, so `--no-bos` first is
the cheap route to charges and spin moments. DDEC needs norm-conserving
pseudopotentials, a grid finer than 0.25 Bohr (0.14 recommended), and a cube
that integrates to the cell charge; mismatch or unsupported core-electron
counts are reported instead of failing inside Chargemol. `--net-charge` and
`--periodicity` handle a charged cell and a molecule in a box.

## Hirshfeld, CM5, and Hirshfeld-I

```text
abacustools postprocess hirshfeld -j JOB
abacustools postprocess hirshfeld -j JOB --no-cm5 --json
abacustools postprocess hirshfeld -j JOB --hirshfeld-i --max-iter 300 --tol 1e-4
abacustools postprocess hirshfeld -j JOB --hirshfeld-i --references DIR
```

Hirshfeld (stockholder) charges use the spherically averaged free-atom
densities of the pseudopotentials (`PP_RHOATOM`) as the promolecule, summed
over lattice images. CM5 adds Pauling-bond-order weighted pairwise corrections
with built-in parameters, so `--no-cm5` gives the plain Hirshfeld charges.
`--hirshfeld-i` iterates the reference densities at the populations the previous
step assigned; the charged references come from `PP_PSWFC`, or from
`--references DIR` of `<element>_<population>.dat` files (two columns, `r` in
Angstrom and `rho` in e/Angstrom^3) when the pseudopotential has no
`PP_PSWFC`. `--max-iter`, `--tol` and `--mixing` control the iteration. The
density is read from the cube or `*-CHARGE-DENSITY.restart`; `--grid`/`--lat0`
override the restart conversion. The command prints its tables and takes no
`-o`.

## Charge densities and derived fields

```text
abacustools postprocess chg -j JOB --cube charge.cube
abacustools postprocess chg -j JOB --spin difference --cube magnetization.cube
abacustools postprocess chg -j JOB --difference OTHER_JOB --cube bonding.cube
abacustools postprocess chg -j JOB --profile c --slice c --slice-index 0.5 --slice-plot
abacustools postprocess chg -j JOB --quantity rdg --nci-plot
abacustools postprocess chg -j JOB --quantity igmh --igmh-plot
```

`--spin total|up|down|difference` selects the field; `--difference OTHER_JOB`
subtracts the density of another job with a matching grid. Densities come from
the cubes of `out_chg 1` or are reconstructed from
`*-CHARGE-DENSITY.restart` (`rho(G)`, the Gamma-only half is rebuilt). The report
compares the integrated charge with the valence charge of the atoms.
`--quantity` (default `density`) derives a field with Quantum ESPRESSO
`pp.x` definitions: `rdg`,
`sl2rho`, `dori`, plus Multiwfn-style `iri`, the promolecular `rdg-promolecular`
and `sl2rho-promolecular`, the independent gradient model `dg`, and the
Hirshfeld/Hirshfeld-I `igmh`/`igmh-i`. The promolecular and IGMH quantities need
the structure and its UPFs (`igmh-i` also `PP_PSWFC`), require `--spin total`,
and cannot be combined with `--difference`. `--nci-plot`, `--igm-plot`,
`--igmh-plot` and `--promolecular-plot` draw the corresponding scatter plots
(`--nci-rho-max`, default 0.05 e/Bohr^3, drops cores and bonds). `--cube`,
`--profile` (with `--profile-kind average|integral`), `--slice` (with
`--slice-index`, `--slice-output`, `--slice-plot`, `--vmin`/`--vmax`,
`--no-atoms`) work on either the raw or the derived field.

## Molden export

```text
abacustools postprocess molden -j JOB
abacustools postprocess molden -j JOB -o orbitals.molden --gto-primitives 8
abacustools postprocess molden -j JOB --kpoint 2 --atoms-unit angstrom --json
```

Reads the LCAO coefficients from `WFC_NAO_GAMMA*` (or the develop
`wf*_nao.txt`/`WFC_NAO_K*` names), expands every numerical atomic orbital into a
contracted Gaussian fit, and writes `wfc.molden` (valence counts from the
pseudopotentials, pure spherical harmonics kept). The format holds one real
orbital set, so the job must be `gamma_only 1` or the selected k-point real;
`--kpoint` picks one k-point (one-based, defaults to Gamma or the first).
`--gto-primitives` sets the Gaussians per NAO (default 6) and the report quotes
the largest relative radial fit error; `--atoms-unit` sets the `[Atoms]` block
(bohr by default). `nspin 2` writes both channels into one file.

## MD trajectories

```text
abacustools postprocess md -j JOB -o traj.extxyz
abacustools postprocess md -j JOB --first 100 --last 2000 --stride 5 --json
```

Reads `OUT.<suffix>/MD_dump` (or the per-step `STRU_MD_*` files when the dump
is absent, which then carries no forces or virial), attaches energy,
temperature, and pressure from the running log, and writes any ASE format; the
output suffix selects it (`--format` forces it). Units follow `MD_dump`:
positions Angstrom, forces eV/Angstrom, velocities Angstrom/fs, virial kBar.
`file traj` converts between trajectory formats with the same machinery.

## Reference

- [references/postprocess-commands.md](references/postprocess-commands.md) -
  per-command inputs, outputs, and options, including what each analysis needs
  in `INPUT` to have been written.
