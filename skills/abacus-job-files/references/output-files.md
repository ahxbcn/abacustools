# Output files of an ABACUS job

All output is written below `OUT.<suffix>`, where `suffix` is the INPUT
keyword (default `ABACUS`), so the usual directory is `OUT.ABACUS`. The
`out_*` keywords decide which products exist; the branch decides how they are
named (see [version-dialects.md](version-dialects.md)).

## Directory layout and lifecycle files

| path | content |
| --- | --- |
| `OUT.<suffix>/running_<calculation>.log` | driver log: `scf`, `relax`, `cell-relax`, `md`, `nscf` (rank 0) |
| `OUT.<suffix>/running_<calculation>_<rank>.log` | per-rank logs, only with `out_alllog` |
| `OUT.<suffix>/running_nscf1.log`, `...nscf2.log`, `...nscf3.log` | the three Berry-phase steps of a polarization run |
| `OUT.<suffix>/warning.log` | warnings (rank 0) |
| `OUT.<suffix>/math_info_<rank>.log` | math/library diagnostic log |
| `OUT.<suffix>/INPUT` | copy of the effective input (LTS 3.10) |
| `OUT.<suffix>/INPUT.info` | copy of the effective input (develop) |
| `OUT.<suffix>/kpoints` | k-point mesh and weights actually used |
| `OUT.<suffix>/mulliken.txt` | Mulliken population, with `out_mul 1` |

Subdirectories are created only when needed: `OUT.<suffix>/STRU/` holds the MD
restart structures, `OUT.<suffix>/matrix/` the real-space matrices of an MD run
with `out_app_flag 0`, and develop adds `OUT.<suffix>/WFC/` when
`out_wfc_lcao` is on with `out_app_flag 0`.

## Volumetric (cube) files

Written by `out_chg`, `out_pot`, `out_tau`/Meta-GGA, `out_elf`, `out_ldos` and
`out_band`. The LTS and develop branches name the same product differently:

| quantity | trigger | LTS 3.10 | develop |
| --- | --- | --- | --- |
| charge density | `out_chg 1` | `SPIN<n>_CHG.cube` | `chg.cube` (nspin 1), `chgs<n>.cube` (nspin 2/4) |
| initial charge | `out_chg 2` | `SPIN<n>_CHG_INI.cube` | `chg_ini.cube`, `chgs<n>_ini.cube` |
| local potential | `out_pot 1` | `SPIN<n>_POT.cube` | `pot.cube` (nspin 1), `pots<n>.cube` (nspin 2/4) |
| initial potential | `out_pot 3` | `SPIN<n>_POT_INI.cube` | `pot_ini.cube`, `pots<n>_ini.cube` |
| kinetic energy density | Meta-GGA | `SPIN<n>_TAU.cube` | `tau.cube`, `taus<n>.cube` |
| ELF | `out_elf 1` | `ELF.cube` (nspin 1), `ELF_SPIN<n>.cube` (nspin 2) | `elftot.cube` (nspin 1/4), `elfs<n>.cube` (nspin 2) |
| electrostatic potential | `out_pot 2` | `ElecStaticPot.cube` | `potes.cube` |
| local DOS | `out_ldos` | not written | `LDOS_<E>eV.cube`, and `LDOS.txt` along a line |
| band partial charge | `out_band 1` | `BAND<n>_GAMMA_SPIN<n>_CHG.cube`, `BAND<n>_K<m>_SPIN<n>_CHG.cube` | `pchgi<n>s<n>.cube`, `pchgi<n>s<n>k<m>.cube` |

Notes on the two naming schemes:

- In the LTS scheme the spin index is always present (`SPIN1_...` even for
  `nspin 1`); in develop it appears as `s<n>` only for `nspin 2/4`.
- develop inserts `g<step>` before the extension when `out_freq_ion` writes one
  file per geometry step (`chgs1g3.cube`). The step counter starts at 1. The
  LTS cubes are overwritten and carry no step token.
- The develop input reference text prints `pots1.cube` for `nspin 1` and
  `pot_es.cube` for `out_pot 2`, but the writer emits `pot.cube` and
  `potes.cube`. Match on both spellings when globbing.

The charge-density backup is a separate binary file,
`<suffix>-CHARGE-DENSITY.restart` (default `ABACUS-CHARGE-DENSITY.restart`),
written even when the cube is suppressed. `out_chg -1` disables it. It stores
`rho(G)` and is converted back to a real-space density with the FFT grid and
lattice constant of the run.

## Electronic-structure files

| file | trigger | content |
| --- | --- | --- |
| `BANDS_<n>.dat` | NSCF with `out_band 1` | band energies along the line-mode path, in eV; `n` is the spin channel |
| `PBANDS_<n>` | `out_proj_band 1` | projected (fat) band weights, XML |
| `DOS<n>_smearing.dat` | `out_dos 1` | total DOS per spin channel |
| `PDOS` | `out_dos 1` (LCAO) | projected DOS, XML with an energy grid |

## LCAO matrices and wavefunctions

Written for `basis_type lcao`. The names and the indexing differ by branch:

| quantity | trigger | LTS 3.10 | develop |
| --- | --- | --- | --- |
| H(k), S(k) | `out_mat_hs 1` | `data-<ik>-H`, `data-<ik>-S`, `<ik>` zero-based | `hk<ik>_nao.txt`, `sk<ik>_nao.txt`, `<ik>` one-based |
| density matrix (k space) | `out_dm` / `out_dmk` | `SPIN<n>_DM` | `dm_nao.txt`, `dmk<ik>_nao.txt`, `dms<n>_nao.txt` |
| density matrix (real space) | `out_dm1` / `out_dmr` | `data-DMR-sparse_SPIN<n>.csr` | `dmrs<n>_nao.csr` |
| NAO wavefunctions | `out_wfc_lcao 1` | `WFC_NAO_GAMMA<n>.TXT`, `WFC_NAO_K<n>.TXT` | `wf_nao.txt`, `wfk<ik>_nao.txt`, `wfs<n>_nao.txt` |
| H(R), S(R) | `out_mat_hs2 1` / `out_hsr 1` | `data-HR-sparse_SPIN<n>.csr`, `data-SR-sparse_SPIN0.csr` | `hrs<n>_nao.csr`, `sr_nao.csr` |
| kinetic matrix T(R) | `out_mat_t 1` / `out_mat_tk` | `data-TR-sparse_SPIN0.csr` (k-space `data-<ik>-T`) | `tr_nao.csr` (k-space `tk..._nao.txt`) |
| position matrix r(R) | `out_mat_r 1` | `data-rR-sparse.csr` | `rr_nao.txt` (the input reference text documents `rxrs1_nao.csr`, `ryrs1_nao.csr`, `rzrs1_nao.csr`; the writer emits `rr_nao.txt`) |

Format and step conventions:

- LTS H(k)/S(k) gets an `<istep>_` prefix when `out_app_flag 0`
  (`2_data-3-H`); the k index is zero-based.
- develop inserts `g<step>` when `out_app_flag 0` (`hk1s1g1_nao.txt`) and can
  write `.dat` instead of `.txt` for a binary wavefunction (`out_wfc_lcao 2`).
- develop `out_hsr 0|1|2|3` selects disabled, text CSR, binary CSR `.dat`, or
  `.npz`; `out_mat_hs2` is the legacy alias for `out_hsr 1`.
- develop NAO wavefunctions go to `OUT.<suffix>/` with `out_app_flag 1` (the
  default) and to `OUT.<suffix>/WFC/` with `out_app_flag 0`.
- In the develop spin/spinor names, `s<n>` is the spin index; a non-gamma run
  adds `k<ik>` (for example `wfk1s1_nao.txt`).

## Structures and trajectories

| file | trigger | content |
| --- | --- | --- |
| `OUT.<suffix>/STRU/STRU_MD_<step>` | `md_restartfreq` | per-step structure for restarting MD; both branches use this path |
| `OUT.<suffix>/MD_dump` | `md_dumpfreq` | appended blocks with cell, positions, and forces/velocities/virial |
| LTS `STRU_ION_D`, `STRU_NOW.cif` | always in relax/cell-relax | final structure and cell |
| LTS `STRU_ION<step>_D` | `out_stru` | per-step structure during relaxation |
| develop `STRU_NOW`, `STRU_FINAL` | `out_stru 1` | overwritten current structure and final structure |
| develop `STRU_NOW.cif`, `STRU_FINAL.cif` | `out_stru 2` | same in CIF format |
| develop `STRU<step+1>` | `out_stru 1|2` with `out_freq_ion` | numbered per-step structure |

Positions in `MD_dump` are Angstrom, forces eV/Angstrom, velocities
Angstrom/fs and the virial kBar. An `md` run with no `MD_dump` can still be
read from its restart structures, which carry positions but no forces.

## Values that are not separate files

- Berry-phase polarization is parsed from `running_nscf1/2/3.log`.
- DFT+U local occupations are parsed from `running_scf.log`.
- Convergence, energies, Fermi level, forces and stresses are parsed from the
  `running_*.log` files; they are not written as standalone tables.

Cube files store density in e/Bohr^3 and dimensionless fields (ELF, LDOS,
`pchg`) as plain values on a Bohr grid; band and DOS files use eV.
