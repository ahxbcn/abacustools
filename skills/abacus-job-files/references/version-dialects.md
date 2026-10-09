# ABACUS LTS and develop dialects

The current LTS line is 3.10 (`3.10.1LTS`); the develop branch is 3.11 and
onward. Both run the same calculations and write the same set of files, but
they name several products differently and print different markers into the
running log. A reader that only knows one dialect will fail to find files or
parse results in the other.

## Detecting the branch

Read the `ABACUS v...` banner in `OUT.<suffix>/running_*.log`. The file names
are a reliable fallback: `SPIN1_CHG.cube` means LTS, `chgs1.cube` means
develop. A job can also be a mixture when an LTS directory is postprocessed by
a develop binary or the reverse, so the banner is the primary signal.

## File naming

| product | LTS 3.10 | develop |
| --- | --- | --- |
| charge density | `SPIN<n>_CHG.cube` | `chg.cube`, `chgs<n>.cube` |
| initial charge density | `SPIN<n>_CHG_INI.cube` | `chg_ini.cube`, `chgs<n>_ini.cube` |
| electrostatic potential | `SPIN<n>_POT.cube` | `pot.cube`, `pots<n>.cube` |
| averaged potential (`out_pot 2`) | `ElecStaticPot.cube` | `potes.cube` |
| kinetic energy density | `SPIN<n>_TAU.cube` | `tau.cube`, `taus<n>.cube` |
| ELF | `ELF.cube`, `ELF_SPIN<n>.cube` | `elftot.cube`, `elfs<n>.cube` |
| local DOS | not written | `LDOS_<E>eV.cube` |
| band partial charge | `BAND<n>_GAMMA_SPIN<n>_CHG.cube`, `BAND<n>_K<m>_SPIN<n>_CHG.cube` | `pchgi<n>s<n>[k<m>].cube` |
| per-step cube | no step token | `g<step>` before the extension |
| Hamiltonian / overlap | `data-<ik>-H`, `data-<ik>-S` | `hk<ik>_nao.txt`, `sk<ik>_nao.txt` |
| density matrix | `SPIN1_DM`, `SPIN2_DM` | `dm*_nao.txt` |
| NAO wavefunctions | `WFC_NAO_GAMMA<n>.TXT`, `WFC_NAO_K<n>.TXT` | `wf_nao.txt`, `wfs<n>_nao.txt`, `wfk<n>_nao.txt` |
| per-step structure | `STRU_MD_<step>` in the job directory | `OUT.<suffix>/STRU_MD_<step>/STRU` |

The develop names omit the k-point index for a gamma-only run (`hk_nao.txt`),
add a spin index as `s<n>`, and can add a geometry step as `g<n>`. The restart
density, `BANDS_*.dat`, `PBANDS_*`, `DOS*_smearing.dat`, `PDOS`, `kpoints` and
`mulliken.txt` keep the same names on both branches.

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
- The develop branch tags per-step cubes with `g<step>`, so a job with
  `out_freq_ion` can hold several densities; the last step is the converged
  one.
- Keyword names and accepted values can change between a branch and a release
  too. `abacustools` validates keywords against a shipped parameter list; when
  a newer ABACUS adds a keyword, pass it through an INPUT template instead of
  assuming the validator knows it.
