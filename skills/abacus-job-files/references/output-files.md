# Output files of an ABACUS job

All output is written below `OUT.<suffix>`, where `suffix` is the INPUT
keyword (default `ABACUS`), so the usual directory is `OUT.ABACUS`. The
`out_*` keywords decide which products exist; the branch decides how they are
named (see [version-dialects.md](version-dialects.md)).

## Lifecycle files

| file | content |
| --- | --- |
| `OUT.<suffix>/running_<calculation>.log` | the driver log: `scf`, `relax`, `cell-relax`, `md` or `nscf` |
| `OUT.<suffix>/running_nscf1.log`, `...nscf2.log`, `...nscf3.log` | the three Berry-phase steps of a polarization calculation |
| `OUT.<suffix>/INPUT` | a copy of the effective input, useful to see what actually ran |
| `OUT.<suffix>/kpoints` | the k-point mesh and weights actually used, including the symmetry reduction |
| `OUT.<suffix>/mulliken.txt` | Mulliken population analysis, with `out_mul 1` |

Some layouts leave `running_*.log` in the job directory instead of below
`OUT.<suffix>`; readers should check both.

## Volumetric (cube) files

Written by `out_chg`, `out_pot`, `out_tau`, `out_elf`, `out_ldos` and
`out_band`, optionally once per geometry step (`out_freq_ion`). The LTS and
develop branches name the same product differently:

| quantity | LTS 3.10 | develop |
| --- | --- | --- |
| charge density | `SPIN<n>_CHG.cube`, initial `SPIN<n>_CHG_INI.cube` | `chg.cube`, `chgs<n>.cube`, initial `..._ini.cube` |
| electrostatic potential | `SPIN<n>_POT.cube`, initial `SPIN<n>_POT_INI.cube` | `pot.cube`, `pots<n>.cube` |
| kinetic energy density | `SPIN<n>_TAU.cube` | `tau.cube`, `taus<n>.cube` |
| ELF | `ELF.cube`, `ELF_SPIN<n>.cube` | `elftot.cube`, `elfs<n>.cube` |
| local DOS | not written | `LDOS_<E>eV.cube` |
| band partial charge | `BAND<n>_GAMMA_SPIN<n>_CHG.cube`, `BAND<n>_K<m>_SPIN<n>_CHG.cube` | `pchgi<n>s<n>[k<m>].cube` |

In develop, a geometry-step index is inserted as `g<step>` before the
extension (for example `chgs1g3.cube`). The averaged electrostatic potential of
`out_pot 2` is `ElecStaticPot.cube` in LTS and `potes.cube` in develop; the
develop manual also mentions the unused `pot_es.cube`.

The restart density is a separate binary file,
`<suffix>-CHARGE-DENSITY.restart` (default `ABACUS-CHARGE-DENSITY.restart`),
written even when `out_chg` suppresses the cube. It stores `rho(G)`, so it can
be converted back to a real-space density with the FFT grid and lattice
constant of the run.

## Electronic-structure files

| file | trigger | content |
| --- | --- | --- |
| `BANDS_1.dat` (`BANDS_2.dat` for `nspin 2`) | NSCF with `out_band 1` | band energies along the line-mode path, in eV |
| `PBANDS_1`, `PBANDS_2` | `out_proj_band 1` | projected (fat) band weights |
| `DOS<n>_smearing.dat` | `out_dos 1` | total DOS per spin channel |
| `PDOS` | `out_dos 1` with the projection on | projected DOS, XML with an energy grid |

## LCAO matrices and wavefunctions

Written for `basis_type lcao`, with branch-dependent names:

| quantity | trigger | LTS 3.10 | develop |
| --- | --- | --- | --- |
| Hamiltonian, overlap | `out_mat_hs 1` | `data-<ik>-H`, `data-<ik>-S` | `hk<ik>_nao.txt`, `sk<ik>_nao.txt` |
| density matrix | `out_dm 1` | `SPIN1_DM`, `SPIN2_DM` (gamma-only) | `dm*_nao.txt` |
| NAO wavefunctions | `out_wfc_lcao 1` | `WFC_NAO_GAMMA<n>.TXT`, `WFC_NAO_K<n>.TXT` | `wf_nao.txt`, `wfs<n>_nao.txt`, `wfk<n>_nao.txt` |
| sparse H/S/position for `pyatb` | `out_mat_hs2 1`, `out_mat_r 1` | `data-HR-sparse_SPIN0.csr`, `data-SR-sparse_SPIN0.csr`, `data-rR-sparse.csr` | same |

In the develop names the k-point index is omitted for a gamma-only run
(`hk_nao.txt`, `wk_nao.txt`), a spin index is written as `s<n>`, and a geometry
step can appear as `g<n>`.

## Structures and trajectories

| file | trigger | content |
| --- | --- | --- |
| `STRU_ION<step>_D` below `OUT.<suffix>` | `out_stru 1` in a `relax`/`cell-relax` run | the structure of each ionic step |
| `STRU_MD_<step>` | `out_stru 1` in an `md` run | the structure of each MD step; LTS keeps it in the job directory, develop writes `OUT.<suffix>/STRU_MD_<step>/STRU` |
| `MD_dump` below `OUT.<suffix>` | `md_dumpfreq` | one appended block per dumped step: cell, positions, and forces/velocities/virial when `dump_force`/`dump_vel`/`dump_virial` are on |

Positions in `MD_dump` are Angstrom, forces eV/Angstrom, velocities
Angstrom/fs and the virial kBar. An `md` run with no `MD_dump` can still be
read from its `STRU_MD_*` files, which then carry no forces or virial.

## Values that are not separate files

- Berry-phase polarization is parsed from `running_nscf1/2/3.log`.
- DFT+U local occupations are parsed from `running_scf.log`.
- Convergence, energies, Fermi level, forces and stresses are parsed from the
  `running_*.log` files; they are not written as standalone tables.

Units follow the cube writer: `SPIN*_CHG.cube` and the derived density files
store density in e/Bohr^3 and Bohr coordinates, while `BANDS_*.dat` and
`DOS*_smearing.dat` use eV.
