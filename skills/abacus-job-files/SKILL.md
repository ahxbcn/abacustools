---
name: abacus-job-files
description: List the input files an ABACUS calculation needs and the output files it writes, for the 3.10 LTS branch and the current develop branch, including the names that differ between them. Use when setting up or inspecting an ABACUS job directory, or when a structure, k-point, pseudopotential, orbital, charge-density, potential, wavefunction, matrix, band, DOS, PDOS or trajectory file cannot be found. Not for choosing physics or numerical parameters, and not for running the calculation.
---

# ABACUS job files

An ABACUS calculation runs in one working directory and reads its files from
there. The set of files is the same on the LTS and develop branches; what
changes between the two is the naming of the volumetric (cube) files, of the
LCAO matrices and wavefunctions, and of the per-step structures. The branch is
declared by the `ABACUS v...` banner in the running log, and the file names are
a reliable fallback.

Everything named below is a default. The `INPUT` file itself is fixed, but the
structure, k-point, pseudopotential, orbital and PAW files are named by
keywords inside it (`stru_file`, `kpoint_file`, `pseudo_dir`, `orbital_dir`,
`paw_dir`), and the output directory by `suffix`. See
[references/input-files.md](references/input-files.md) for the keyword-to-file
mapping.

## Input files

| file | role | when needed |
| --- | --- | --- |
| `INPUT` | keyword deck, one `name value` pair per line under `INPUT_PARAMETERS` | always; the name is fixed |
| `STRU` | structure: `ATOMIC_SPECIES`, `LATTICE_CONSTANT`, `LATTICE_VECTORS`, `ATOMIC_POSITIONS`, optional `NUMERICAL_ORBITAL`, `PAW_FILES` | always; renamed by `stru_file` |
| pseudopotential (`*.upf`) | per element, named in `ATOMIC_SPECIES` | always, one per element |
| `KPT` | k-point mesh (Gamma/MP) or explicit list / band path | unless `kspacing` fixes the mesh or an LCAO run sets `gamma_only 1`; renamed by `kpoint_file` |
| numerical orbital (`*.orb`) | per element, named in `NUMERICAL_ORBITAL` | only for `basis_type lcao` |
| `PAW_FILES` datasets | per element, named in `PAW_FILES` | only for PAW calculations |

`INPUT` and `STRU` are mandatory. The pseudopotential is mandatory for every
element and is resolved relative to the working directory, or below
`pseudo_dir` when that keyword is set; the same holds for orbitals below
`orbital_dir` and PAW data below `paw_dir`. For LCAO, an orbital is required
for every element. A KPT file is what makes a multi-k run possible; it is not
needed when `kspacing` fixes the mesh or in an LCAO `gamma_only 1` run.

For continuation, `read_file_dir` (default `OUT.<suffix>`) points at the
directory holding a previous charge density or wavefunction, and
`init_chg`/`init_wfc` decide whether it is read; `md_restart`/`md_restartfreq`
continue a molecular-dynamics run from `STRU_MD_<step>`.

## Output files

All output goes below `OUT.<suffix>` (the `suffix` keyword, default `ABACUS`),
which is created in the working directory. The driver log is written there,
not into the job directory:

- `running_<calculation>.log`, with `calculation` one of `scf`, `relax`,
  `cell-relax`, `md`, `nscf`. With `out_alllog` there is one file per rank,
  `running_<calculation>_<rank>.log`, otherwise only rank 0 writes.
- `warning.log` (rank 0) and `math_info_<rank>.log`, plus `device*.log` on a
  GPU build.
- A copy of the effective input: `INPUT` in the LTS branch, `INPUT.info` in
  develop.

Subdirectories are created only when needed: `STRU/` holds the MD restart
structures, `matrix/` the real-space matrices of an MD run, and develop adds
`WFC/` when `out_wfc_lcao` is on with `out_app_flag 0`.

Written on request by the `out_*` keywords, with branch-dependent names:

| quantity | trigger | LTS 3.10 | develop |
| --- | --- | --- | --- |
| charge density | `out_chg 1` | `SPIN<n>_CHG.cube` | `chg.cube` (nspin 1), `chgs<n>.cube` (nspin 2/4) |
| initial charge density | `out_chg 2` | `SPIN<n>_CHG_INI.cube` | `chg_ini.cube`, `chgs<n>_ini.cube` |
| local potential | `out_pot 1` | `SPIN<n>_POT.cube` | `pot.cube` (nspin 1), `pots<n>.cube` (nspin 2/4) |
| initial potential | `out_pot 3` | `SPIN<n>_POT_INI.cube` | `pot_ini.cube`, `pots<n>_ini.cube` |
| kinetic energy density | Meta-GGA | `SPIN<n>_TAU.cube` | `tau.cube`, `taus<n>.cube` |
| ELF | `out_elf 1` | `ELF.cube`, `ELF_SPIN<n>.cube` | `elftot.cube`, `elfs<n>.cube` |
| electrostatic potential | `out_pot 2` | `ElecStaticPot.cube` | `potes.cube` |
| LDOS | `out_ldos` | not written | `LDOS_<E>eV.cube` (and `LDOS.txt` along a line) |
| band partial charge | `out_band 1` | `BAND<n>_GAMMA_SPIN<n>_CHG.cube`, `BAND<n>_K<m>_SPIN<n>_CHG.cube` | `pchgi<n>s<n>.cube`, `pchgi<n>s<n>k<m>.cube` |

In develop a geometry-step token `g<step>` (step counted from 1) is inserted
before the extension when `out_freq_ion` writes one file per step, for example
`chgs1g3.cube`; the LTS cubes carry no step token. The charge-density backup is
`<suffix>-CHARGE-DENSITY.restart` (default `ABACUS-CHARGE-DENSITY.restart`) on
both branches; `out_chg -1` disables it.

Electronic-structure outputs use the same names on both branches:

| file | trigger | content |
| --- | --- | --- |
| `BANDS_<n>.dat` | NSCF with `out_band 1` | band energies along the line-mode path, in eV |
| `PBANDS_<n>` | `out_proj_band 1` | projected (fat) band weights, XML |
| `DOS<n>_smearing.dat` | `out_dos 1` | total DOS per spin channel |
| `PDOS` | `out_dos 1` (LCAO) | projected DOS, XML |
| `kpoints` | symmetry-reduced run | the k-point mesh and weights used |
| `mulliken.txt` | `out_mul 1` | Mulliken population analysis |

LCAO matrices and wavefunctions differ by branch:

| quantity | trigger | LTS 3.10 | develop |
| --- | --- | --- | --- |
| H(k), S(k) | `out_mat_hs 1` | `data-<ik>-H`, `data-<ik>-S` | `hk<ik>_nao.txt`, `sk<ik>_nao.txt` |
| density matrix in k space | `out_dm` / `out_dmk` | `SPIN<n>_DM` | `dm_nao.txt`, `dmk<ik>_nao.txt`, `dms<n>_nao.txt` |
| density matrix in real space | `out_dm1` / `out_dmr` | `data-DMR-sparse_SPIN<n>.csr` | `dmrs<n>_nao.csr` |
| NAO wavefunctions | `out_wfc_lcao 1` | `WFC_NAO_GAMMA<n>.TXT`, `WFC_NAO_K<n>.TXT` (+ `_ION<step>`) | `wf_nao.txt`, `wfk<ik>_nao.txt`, `wfs<n>_nao.txt` |
| H(R), S(R) | `out_mat_hs2 1` / `out_hsr 1` | `data-HR-sparse_SPIN<n>.csr`, `data-SR-sparse_SPIN0.csr` | `hrs<n>_nao.csr`, `sr_nao.csr` |
| position matrix r(R) | `out_mat_r 1` | `data-rR-sparse.csr` | `rxrs1_nao.csr`, `ryrs1_nao.csr`, `rzrs1_nao.csr` |
| kinetic matrix T(R) | `out_mat_t 1` / `out_mat_tk` | `data-TR-sparse_SPIN0.csr` | `tr_nao.csr` |

The develop names carry an optional `g<step>` when `out_app_flag 0`, and the
NAO wavefunctions go to `OUT.<suffix>/WFC/` in that case. The LTS H(k)/S(k)
files carry an optional `<istep>_` prefix per ionic step.

Trajectory and per-step structure outputs:

- `OUT.<suffix>/MD_dump`: one appended block per dumped MD step, controlled by
  `md_dumpfreq`, holding the cell, positions, and forces/velocities/virial when
  the corresponding `dump_*` flags are on.
- `OUT.<suffix>/STRU/STRU_MD_<step>`: the per-step structures written by
  `md_restartfreq`, used to restart an MD run. Both branches use this path.
- Relaxation structures: LTS always writes `STRU_ION_D` and `STRU_NOW.cif`,
  and `STRU_ION<step>_D` with `out_stru`; develop writes `STRU_NOW` and
  `STRU_FINAL` (or the `.cif` forms with `out_stru 2`), plus `STRU<step+1>`
  when `out_stru` is 1 or 2.

Berry-phase polarization and DFT+U occupations are not separate files: they are
parsed from `running_nscf1/2/3.log` and `running_scf.log` respectively.

## LTS versus develop

| what | LTS 3.10 | develop (3.11+) |
| --- | --- | --- |
| effective input copy | `OUT.<suffix>/INPUT` | `OUT.<suffix>/INPUT.info` |
| charge / potential / tau / ELF cubes | `SPIN<n>_<QUANT>.cube` | `chg`/`pot`/`tau`/`elf` with an optional `s<n>` spin token |
| electrostatic potential of `out_pot 2` | `ElecStaticPot.cube` | `potes.cube` |
| per-step cube token | none | `g<step>` before the extension |
| H(k)/S(k) | `data-<ik>-H`, `data-<ik>-S` | `hk<ik>_nao.txt`, `sk<ik>_nao.txt` |
| H(R)/S(R) | `data-HR-sparse_SPIN<n>.csr`, `data-SR-sparse_SPIN0.csr` | `hrs<n>_nao.csr`, `sr_nao.csr` |
| NAO wavefunctions | `WFC_NAO_GAMMA<n>.TXT`, `WFC_NAO_K<n>.TXT` | `wf*_nao.txt`, in `WFC/` when `out_app_flag 0` |
| per-step structures | `STRU_ION_D`, `STRU_ION<step>_D` | `STRU_NOW`, `STRU_FINAL`, `STRU<step+1>` |
| MD restart structures | `OUT.<suffix>/STRU/STRU_MD_<step>` | same |
| density-error log line | `density error = ...` | `Electron density deviation ...` |
| SCF-converged log line | `charge density convergence is achieved` | `#SCF IS CONVERGED#` |
| final-energy log line | `final etot is ...` | `!FINAL_ETOT_IS ...` |

To decide which branch produced a job, read the `ABACUS v...` banner in
`OUT.<suffix>/running_*.log`; the file names alone are a reliable fallback
(`SPIN1_CHG.cube` means LTS, `chgs1.cube` means develop). The full naming and
log-marker tables are in
[references/version-dialects.md](references/version-dialects.md).

## References

- [references/input-files.md](references/input-files.md) - every input file,
  the keyword that names it, the `STRU` and `KPT` block structure, and the
  resource directories.
- [references/output-files.md](references/output-files.md) - the full output
  inventory by quantity, with the `out_*` keyword that produces it and the LTS
  and develop names.
- [references/version-dialects.md](references/version-dialects.md) - how the two
  branches name the same product and how to detect the branch from the log.
