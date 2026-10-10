# ABACUS LTS and develop dialects

The current LTS line is 3.10 (`3.10.1LTS`); the develop branch is 3.11 and
onward. Both run the same calculations and write the same set of files, but
they name several products differently, use a different k-point index base for
some matrices, and print different markers into the running log. A reader that
only knows one dialect will fail to find files or parse results in the other.

## Detecting the branch

Read the `ABACUS v...` banner in `OUT.<suffix>/running_*.log`. The file names
are a reliable fallback: `SPIN1_CHG.cube` and `ElecStaticPot.cube` mean LTS,
`chgs1.cube` and `potes.cube` mean develop. A job can also be a mixture when
an LTS directory is postprocessed by a develop binary or the reverse, so the
banner is the primary signal.

## File naming

| product | LTS 3.10 | develop |
| --- | --- | --- |
| effective input copy | `INPUT` | `INPUT.info` |
| charge density | `SPIN<n>_CHG.cube` | `chg.cube` (nspin 1), `chgs<n>.cube` |
| initial charge density | `SPIN<n>_CHG_INI.cube` | `chg_ini.cube`, `chgs<n>_ini.cube` |
| local potential | `SPIN<n>_POT.cube` | `pot.cube` (nspin 1), `pots<n>.cube` |
| averaged potential (`out_pot 2`) | `ElecStaticPot.cube` | `potes.cube` |
| kinetic energy density | `SPIN<n>_TAU.cube` | `tau.cube`, `taus<n>.cube` |
| ELF | `ELF.cube`, `ELF_SPIN<n>.cube` | `elftot.cube`, `elfs<n>.cube` |
| local DOS | not written | `LDOS_<E>eV.cube`, `LDOS.txt` |
| band partial charge | `BAND<n>_GAMMA_SPIN<n>_CHG.cube`, `BAND<n>_K<m>_SPIN<n>_CHG.cube` | `pchgi<n>s<n>.cube`, `pchgi<n>s<n>k<m>.cube` |
| per-step cube | no step token | `g<step>` before the extension |
| H(k), S(k) | `data-<ik>-H`, `data-<ik>-S` (zero-based) | `hk<ik>_nao.txt`, `sk<ik>_nao.txt` (one-based) |
| density matrix (k space) | `SPIN<n>_DM` | `dm_nao.txt`, `dmk<ik>_nao.txt`, `dms<n>_nao.txt` |
| density matrix (real space) | `data-DMR-sparse_SPIN<n>.csr` | `dmrs<n>_nao.csr` |
| H(R), S(R) | `data-HR-sparse_SPIN<n>.csr`, `data-SR-sparse_SPIN0.csr` | `hrs<n>_nao.csr`, `sr_nao.csr` |
| kinetic matrix T(R) | `data-TR-sparse_SPIN0.csr` | `tr_nao.csr` |
| position matrix r(R) | `data-rR-sparse.csr` | `rr_nao.txt` |
| NAO wavefunctions | `WFC_NAO_GAMMA<n>.TXT`, `WFC_NAO_K<n>.TXT` | `wf_nao.txt`, `wfk<ik>_nao.txt`, `wfs<n>_nao.txt` |
| per-step structure (relax) | `STRU_ION_D`, `STRU_ION<step>_D` | `STRU_NOW`, `STRU_FINAL`, `STRU<step+1>` |
| per-step structure (MD restart) | `OUT.<suffix>/STRU/STRU_MD_<step>` | same |

Details that are easy to get wrong:

- develop wavefunction and matrix files carry an optional `g<step>` when
  `out_app_flag 0`, and the wavefunctions move to `OUT.<suffix>/WFC/` in that
  case; with the default `out_app_flag 1` they are in `OUT.<suffix>/`.
- The LTS H(k)/S(k) k index starts at 0 (`data-0-H`); the develop `hk`/`sk` index
  starts at 1 (`hk1_nao.txt`).
- The develop input reference text prints `pots1.cube` for `nspin 1` and
  `pot_es.cube` for `out_pot 2`, but the writer emits `pot.cube` and
  `potes.cube`; glob both spellings.
- develop `out_mat_r` is documented as `rxrs1_nao.csr`/`ryrs1_nao.csr`/
  `rzrs1_nao.csr`, while the writer emits a single `rr_nao.txt`. Keep the glob
  loose.

## Log markers

| quantity | LTS 3.10 | develop |
| --- | --- | --- |
| density error | `density error = ...` | `Electron density deviation ...` (also keeps the LTS marker) |
| SCF converged | `charge density convergence is achieved` | `#SCF IS CONVERGED#` |
| final energy | `final etot is ...` | `!FINAL_ETOT_IS ...` (also `#Total Energy#`) |
| Fermi level | `efermi` | `efermi` or `e_fermi` |
| relaxation step | `STEP OF RELAXATION : n` | `RELAX STEP: n` |
| largest force | `Largest gradient in force is ... eV/` | `Largest force is ... eV/Angstrom` |
| largest stress | `Largest gradient in stress is ... kbar` | `Largest stress is ... kbar` |
| relaxation converged | `Relaxation is converged!` | `end of geometry optimization` |
| normal end | `Total  Time  :` | `Total  Time  :` |

The develop keyword lists deliberately keep the LTS markers where the two
dialects do not conflict, so a develop reader also parses older logs; an LTS
reader does not understand the develop-only markers.

## Consequences for reading a job

- A reader that looks for one naming convention must try both. The
  abacustools post-processing commands do this internally; a hand-written
  script should not glob only `SPIN*_CHG.cube` or only `chg*.cube`.
- develop tags per-step cubes and matrices with `g<step>`, so a job with
  `out_freq_ion` or `out_app_flag 0` can hold several copies; the last step is
  the converged one.
- Keyword names and accepted values change between branches and releases too,
  for example the LTS `out_dm1` versus the develop `out_dmr`, or the boolean LTS
  `out_stru` versus the integer develop `out_stru`. Check the input reference
  of the running version rather than assuming the LTS names.
