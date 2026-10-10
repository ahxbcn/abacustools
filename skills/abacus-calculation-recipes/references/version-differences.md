# LTS 3.10 versus develop

The recipe values are the same across the two current branches; the keyword
names and a few defaults differ. Confirm the keyword on the branch at hand
before copying a deck from the other one.

| area | LTS 3.10 | develop |
| --- | --- | --- |
| H(k)/S(k) output | `out_mat_hs`, files `data-<ik>-H/S` | `out_hsk`, files `hk*_nao.txt`/`sk*_nao.txt` |
| density-matrix output | `out_dm1`, `SPIN<n>_DM` | `out_dmr`, `dm*_nao.txt` |
| per-step structures | `out_stru` is a boolean; `STRU_ION<step>_D` | `out_stru` is an integer mode (final/STRU/CIF); `STRU_NOW`, `STRU_FINAL`, `STRU<step+1>` |
| overlap matrix | `calculation get_S` | `calculation get_s` |
| relaxation method | `bfgs_trad` for relax; `relax_new` requires `cg` | `bfgs` for relax (default variant 2); `cg 1`/`cg 2` (default `cg 2`); adds `lbfgs` |
| extra `esolver_type` | ksdft, sdft, ofdft, tddft, lj, dp, lr, ks-lr | adds tdofdft, nep, dfpt |
| extra functionals | - | SCANL; PW hybrids are refused on LTS but supported on develop |
| extra dispersion | D2, D3(0), D3(BJ) | adds D4 (external DFT-D4) |
| LCAO solver list | includes `cg_in_lcao` (under testing) | `cg_in_lcao` removed |

Defaults to confirm on the running version rather than assuming:

- `ecutwfc` and `scf_thr` defaults depend on the basis; set them explicitly.
- `md_nstep` defaults can be small (a smoke-test value); set a production value.
- The default LCAO `ks_solver` depends on the build (ELPA present or not), so
  the effective solver can differ between two machines running the same deck.

The `symmetry` rule is version-independent: `0` in general, `-1` for `nspin 4`
with SOC, and `1` only where the reduction is known to be safe.
