# postprocess commands: inputs and outputs

| command | needs in the job | writes | key options |
| --- | --- | --- | --- |
| `result` | `OUT.*/running_*.log` | stdout (table or `--json`) | `-j` (repeatable), `-p/--param` (repeatable), `-v/--version`, `--json` |
| `band` | `BANDS_*.dat`, line-mode `KPT`, Fermi energy in the NSCF log | `band.png`, `band.dat`, `KPATH.txt` (spin-resolved variants) | `--emin/--emax`, `--efermi`, `--gap`, `--spin-resolved`, `--effective-mass cbm\|vbm`, `--direction`, `--fit-points`, `--fat-band`, `--atom-index`, `-o`, `--data-output`, `--kpath-output` |
| `dos` | `DOS*_smearing.dat`, Fermi energy | `DOS.png`, `DOS.dat` (`DOS_PDOS.*` with `--combined`) | `--pdos`, `--atom-index`, `--combined`, `--list`, `--emin/--emax`, `--efermi`, `-o`, `--data-output` |
| `cohp` | LCAO H/S and wavefunctions (`data-*-H`, `data-*-S`, `WFC_NAO_K*.txt`; develop: `hk*_nao.txt`, `sk*_nao.txt`, `wf*_nao.txt`) | plot and data file | `--atom-i-orbs`, `--atom-j-orbs` (zero-based, required), `--method COHP\|COOP`, `--spin`, `--de`, `--no-smooth`, `--smooth-nstddev`, `--emin/--emax`, `--width`, `--invert`, `--efermi`, `-o`, `--data-output` |
| `mayer` | LCAO `out_mat_hs 1` (+ `out_dm 1` for gamma-only) | table, `-o` JSON | `--cutoff`, `--pairs 1-2,1-3`, `--pairs-file`, `--threshold`, `--json` |
| `bader` | density cube (`out_chg 1`) or `*-CHARGE-DENSITY.restart`, `bader` binary or `baderkit` | table, `-o` JSON, optional cubes | `--backend bader\|baderkit`, `--baderkit-method`, `--bader-exe`, `--cube`, `--reference`, `--vacuum off\|auto\|VALUE`, `--grid`, `--lat0`, `--workdir`, `--keep-cubes`, `--json` |
| `ddec` | density cube or restart, Chargemol + `atomic_densities` | table, `-o` JSON | `--charge-type DDEC6\|DDEC3`, `--no-spin`, `--no-bos`, `--threads`, `--pairs`, `--cutoff`, `--core-electrons`, `--net-charge`, `--periodicity`, `--workdir`, `--keep`, `--json` |
| `hirshfeld` | density cube or restart, UPFs with `PP_RHOATOM` (Hirshfeld-I: `PP_PSWFC` or `--references`) | stdout table or `--json` | `--hirshfeld-i`, `--references`, `--max-iter`, `--tol`, `--mixing`, `--no-cm5`, `--grid`, `--lat0`, `--json` |
| `chg` | density cube or restart | cube/profile/slice data, plots, report | `--spin total\|up\|down\|difference`, `--difference`, `--quantity density\|rdg\|sl2rho\|dori\|iri\|rdg-promolecular\|sl2rho-promolecular\|dg\|igmh\|igmh-i`, `--cube`, `--profile a\|b\|c`, `--profile-kind average\|integral`, `--slice`, `--slice-index`, `--slice-output`, `--slice-plot`, `--vmin/--vmax`, `--no-atoms`, `--nci-plot`, `--igm-plot`, `--igmh-plot`, `--promolecular-plot`, `--nci-rho-max`, `--grid`, `--json` |
| `molden` | LCAO `WFC_NAO_*` (or develop `wf*_nao.txt`), pseudopotentials | `wfc.molden` | `-o`, `--kpoint` (one-based), `--gto-primitives`, `--atoms-unit bohr\|angstrom`, `--json` |
| `md` | `OUT.*/MD_dump` (fallback `STRU_MD_*`) and the running log | trajectory file in any ASE format | `-o`, `--format`, `--first`, `--last`, `--stride`, `-v`, `--json` |

## INPUT flags each analysis depends on

- `out_chg 1` - Bader, DDEC, Hirshfeld, `chg`. Use `out_chg 1 10` for DDEC
  precision.
- `out_mat_hs 1`, `out_dm 1` - Mayer bond orders (gamma-only needs the DM).
- `out_mat_hs2 1`, `out_mat_r 1`, `symmetry 0` - `workflow dielectric`.
- `out_wfc_lcao`, `data-*-H`/`data-*-S` - COHP/COOP.
- `out_band 1`, line-mode `KPT` - band structure; `out_dos 1` - DOS.
- `out_pot 2` - `workflow workfunc`.
- `dump_force`, `dump_vel`, `dump_virial`, `md_dumpfreq` - richer `MD_dump`
  frames for `postprocess md`.
- `out_stru 1` - per-step structures, the fallback trajectory source.

## Failure modes worth checking first

- A band or DOS shift by a wrong Fermi level: pass `--efermi` explicitly.
- `mayer` on an old or magnetic (`symmetry 2`/`3`) output cannot be expanded
  from a reduced mesh; rerun with `symmetry 0` or `-1`.
- DDEC refuses a cube that does not integrate to the cell charge: the cube and
  `INPUT` are from different calculations, different grids, or a different
  `--cube`. It also refuses PAW/ultrasoft densities and core-electron counts
  that the reference tables do not ship (override with `--core-electrons`).
- `hirshfeld --hirshfeld-i` needs charged reference densities: a `PP_PSWFC`
  table in the pseudopotential or explicit `--references` files.
- `chg --quantity igmh`/`igmh-i` and the promolecular quantities need the
  UPFs and `--spin total`; `igmh-i` additionally needs `PP_PSWFC`.
- `cohp` needs the LCAO matrix outputs; a PW run has nothing to read.
- `molden` writes a single real orbital set, so it needs `gamma_only 1` or a
  real selected k-point.
- All of these read a finished directory, so a non-converged or killed run
  produces partial or stale numbers rather than an error. Check
  `postprocess result -p converged normal_end` alongside any analysis.
