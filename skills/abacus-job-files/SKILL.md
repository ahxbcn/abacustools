---
name: abacus-job-files
description: List the input files an ABACUS calculation needs and the output files it writes, for the 3.10 LTS branch and the current develop branch, including the names that differ between them. Use when setting up or inspecting an ABACUS job directory, or when a structure, k-point, pseudopotential, orbital, charge-density, potential, wavefunction, matrix, band, DOS, PDOS or trajectory file cannot be found. Not for choosing physics or numerical parameters, and not for running the calculation.
---

# ABACUS job files

An ABACUS calculation runs in one working directory and reads its files from
there. The set of files is the same on the LTS and develop branches; what
changes between the two is the naming of the volumetric (cube) files and of the
LCAO matrices and wavefunctions. The version is declared by the `ABACUS v...`
banner in the running log.

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
| `KPT` | k-point mesh (gamma/mp) or explicit list / band path | unless `kspacing` fixes the mesh or an LCAO run sets `gamma_only 1`; renamed by `kpoint_file` |
| numerical orbital (`*.orb`) | per element, named in `NUMERICAL_ORBITAL` | only for `basis_type lcao` |
| `PAW_FILES` datasets | per element, named in `PAW_FILES` | only for PAW calculations |

`INPUT` and `STRU` are mandatory. The pseudopotential is mandatory for every
element and is resolved relative to the working directory, or below
`pseudo_dir` when that keyword is set; the same holds for orbitals below
`orbital_dir` and PAW data below `paw_dir`. A KPT file is not needed when
`kspacing` fixes the mesh or in an LCAO run with `gamma_only 1`. For LCAO, an orbital
is required for every element.

Two further inputs matter for continuation rather than for a fresh run:
`read_file_dir` points at the directory holding a previous charge density or
wavefunction, and `init_chg`/`init_wfc` decide whether it is read. A restarted
job can therefore run with the old `OUT.<suffix>` contents but no fresh
pseudopotential-independent input beyond `INPUT` and `STRU`.

## Output files

All output goes below `OUT.<suffix>` (the `suffix` keyword, default `ABACUS`),
which is created in the working directory. ABACUS also copies the effective
`INPUT` to `OUT.<suffix>/INPUT`. Older layouts can leave `running_*.log` in the
job directory instead of inside the output directory.

Always present once the calculation has started:

- `OUT.<suffix>/running_<calculation>.log`: the driver log, with
  `calculation` one of `scf`, `relax`, `cell-relax`, `md`, `nscf`. The three
  Berry-phase steps of a polarization run write `running_nscf1.log`,
  `running_nscf2.log`, `running_nscf3.log`.
- `OUT.<suffix>/INPUT`: the copy of the input that was actually used.

Written on request by the `out_*` keywords, with branch-dependent names:

| quantity | trigger | LTS name | develop name |
| --- | --- | --- | --- |
| charge density | `out_chg 1` | `SPIN<n>_CHG.cube` (`SPIN<n>_CHG_INI.cube` for the initial density) | `chg.cube` / `chgs<n>.cube`, with `g<step>` when one file per step is written |
| charge-density restart | `out_chg 0` or `1` | `<name>-CHARGE-DENSITY.restart` | same |
| electrostatic potential | `out_pot 1` | `SPIN<n>_POT.cube` | `pot.cube` / `pots<n>.cube` |
| electrostatic potential (averaged) | `out_pot 2` | `ElecStaticPot.cube` | `potes.cube` (the manual also names `pot_es.cube`) |
| kinetic energy density | `out_tau 1` | `SPIN<n>_TAU.cube` | `tau.cube` / `taus<n>.cube` |
| ELF | `out_elf 1` | `ELF.cube`, `ELF_SPIN<n>.cube` | `elftot.cube`, `elfs<n>.cube` |
| LDOS | `out_ldos` | not written | `LDOS_<E>eV.cube` |
| band partial charge | `out_band 1` | `BAND<n>_GAMMA_SPIN<n>_CHG.cube`, `BAND<n>_K<m>_SPIN<n>_CHG.cube` | `pchgi<n>s<n>[k<m>].cube` |

Electronic-structure and LCAO outputs:

| quantity | trigger | LTS name | develop name |
| --- | --- | --- | --- |
| band energies | NSCF with `out_band 1` | `BANDS_1.dat` (`BANDS_2.dat` for `nspin 2`) | same |
| projected (fat) bands | `out_proj_band 1` | `PBANDS_1`, `PBANDS_2` | same |
| total DOS | `out_dos 1` | `DOS<n>_smearing.dat` | same |
| projected DOS | `out_dos 1` with PDOS | `PDOS` (XML) | same |
| k-point table | any symmetry-reduced run | `kpoints` | same |
| Mulliken population | `out_mul 1` | `mulliken.txt` | same |
| Hamiltonian / overlap | `out_mat_hs 1` | `data-<ik>-H`, `data-<ik>-S` | `hk<ik>_nao.txt`, `sk<ik>_nao.txt` (index omitted at Gamma; spin adds `s<n>`) |
| density matrix | `out_dm 1` | `SPIN1_DM`, `SPIN2_DM` (gamma-only) | `dm*_nao.txt` |
| NAO wavefunctions | `out_wfc_lcao 1` | `WFC_NAO_GAMMA<n>.TXT`, `WFC_NAO_K<n>.TXT` | `wf_nao.txt`, `wfs<n>_nao.txt`, `wfk<n>_nao.txt`, `wfk<n>s<n>_nao.txt` |
| sparse matrices for `pyatb` | `out_mat_hs2 1`, `out_mat_r 1` | `data-HR-sparse_SPIN0.csr`, `data-SR-sparse_SPIN0.csr`, `data-rR-sparse.csr` | same |

Trajectory outputs:

- `OUT.<suffix>/MD_dump`: one appended block per dumped MD step (cell,
  positions, and forces/velocities/virial when the corresponding `dump_*`
  flags are on).
- `STRU_MD_<step>`: per-step structures written by `out_stru 1`. LTS keeps them
  in the job directory; develop writes a directory per step below
  `OUT.<suffix>` (`STRU_MD_<step>/STRU`).

Berry-phase polarization and DFT+U occupations are not separate files: they are
parsed from `running_nscf1/2/3.log` and `running_scf.log` respectively.

## LTS versus develop

The two branches differ in naming, not in the set of calculations:

| what | LTS 3.10 | develop (3.11+) |
| --- | --- | --- |
| charge / potential / tau / ELF cubes | `SPIN<n>_<QUANT>.cube` | `chg`/`pot`/`tau`/`elf` with an optional `s<n>` spin token |
| electrostatic potential of `out_pot 2` | `ElecStaticPot.cube` | `potes.cube` |
| per-step cubes | not step-tagged | `g<step>` inserted before the extension |
| LCAO H/S, density, wavefunctions | `data-<ik>-H/S`, `SPIN<n>_DM`, `WFC_NAO_*` | `*_nao.txt` (`hk`/`sk`/`dm`/`wf`/`wfk`/`wfs`) |
| per-step structures | `STRU_MD_<step>` in the job directory | `STRU_MD_<step>/STRU` below `OUT.<suffix>` |
| density-error log line | `density error = ...` | `Electron density deviation ...` |
| SCF-converged log line | `charge density convergence achieved` | `#SCF IS CONVERGED#` |
| final-energy log line | `final etot is ...` | `!FINAL_ETOT_IS ...` |

The full table, including the geometry-optimization markers, is in
[references/version-dialects.md](references/version-dialects.md). To decide
which branch produced a job, read the `ABACUS v...` banner in
`OUT.<suffix>/running_*.log`; the file names alone are a reliable fallback
(`SPIN1_CHG.cube` means LTS, `chgs1.cube` means develop).

## References

- [references/input-files.md](references/input-files.md) - every input file,
  the keyword that names it, the `STRU` and `KPT` block structure, and the
  resource directories.
- [references/output-files.md](references/output-files.md) - the full output
  inventory by quantity, with the `out_*` keyword that produces it and the LTS
  and develop names.
- [references/version-dialects.md](references/version-dialects.md) - how the two
  branches name the same product and how to detect the branch from the log.
