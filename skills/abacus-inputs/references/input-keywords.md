# ABACUS INPUT keywords

The authoritative list of keywords, their types, options, and defaults is
ABACUS's own `input-main` reference:
https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html

The tables below are the keywords that decide most calculations in practice,
grouped by the task that makes you reach for them. They are a starting point,
not a substitute for the upstream page when a keyword is new or version
specific.

## System and files

| keyword | type | default | notes |
| --- | --- | --- | --- |
| `suffix` | string | `ABACUS` | Run output goes to `OUT.<suffix>`. |
| `calculation` | string | `scf` | `scf`, `nscf`, `relax`, `cell-relax`, `md`, `get_pchg`, `get_wf`, `get_S`, `gen_bessel`, `test_*`. |
| `esolver_type` | string | `ksdft` | `sdft`, `ofdft`, `tddft`, `lr` for the other solvers. |
| `symmetry` | int | | `-1`/`0` off (k-points as given), `1` use the space group. `1` reduces the mesh to the irreducible wedge, which any consumer of the full mesh has to expand again. |
| `symmetry_prec` | real | `1.0e-6` | Symmetry tolerance. |
| `init_wfc` | string | `atomic` | Wavefunction guess. |
| `init_chg` | string | `atomic` | `file` reads the density of the previous run, as `nscf` needs. |
| `kspacing` | real | `0.0` | Reciprocal-space spacing in 1/Bohr; `0` means use `KPT`. |
| `device` | string | `cpu` | `gpu` for a GPU build. |
| `stru_file` / `kpoint_file` | string | `STRU` / `KPT` | Structure and k-point file names. |
| `pseudo_dir` / `orbital_dir` | string | `""` | Where ABACUS looks for the files named in `STRU`. |
| `read_file_dir` | string | `OUT.$suffix` | Where `init_chg file`/`init_wfc file` read from. |

## Basis and cutoffs

| keyword | type | default | notes |
| --- | --- | --- | --- |
| `basis_type` | string | `pw` | `lcao` for numerical orbitals. |
| `ks_solver` | string | | Basis-dependent: `genelpa`/`scalapack_gvx` for LCAO, `dav_subspace`/`cg` for PW. |
| `ecutwfc` | real | 50 Ry (PW), 100 Ry (LCAO) | Plane-wave cutoff in Ry. LCAO jobs take it from the orbital file names unless set. |
| `ecutrho` | real | `4*ecutwfc` | Density cutoff. |
| `pw_diag_ndim` | int | `4` | PW Davidson subspace size. |
| `pw_diag_nmax` | int | `40` | PW diagonalization iterations. |
| `nbands` | int | | Number of bands; leave unset to let ABACUS choose. |

## Electronic structure and SCF

| keyword | type | default | notes |
| --- | --- | --- | --- |
| `smearing_method` | string | `gauss` | `gaussian`, `mp`, `mv`, `fd`, `fixed`. |
| `smearing_sigma` | real | `0.015` | Broadening in Ry. |
| `mixing_type` | string | `broyden` | Charge mixing scheme. |
| `mixing_beta` | real | `0.8` (`nspin 1`), `0.4` (`nspin 2`/`4`) | Lower it for oscillation. |
| `scf_thr` | real | `1e-9` (PW), `1e-7` (LCAO) | SCF convergence threshold. |
| `scf_nmax` | int | `100` | Maximum SCF iterations. |
| `scf_thr_type` | int | `1` (PW), `2` (LCAO) | `1` energy, `2` density. |
| `chg_extrap` | string | `first-order` (relax), `second-order` (MD), else `atomic` | Charge extrapolation between ionic steps. |
| `nelec` / `nelec_delta` | real | `0.0` / `0.0` | `nelec 0` means neutral; a positive `nelec` with `nelec_delta` gives a charged cell. |
| `nupdown` | real | `0.0` | Fixed total spin difference for `nspin 2`. |
| `nspin` | int | `1` | `1`, `2` (collinear), `4` (noncollinear/SOC). |
| `lspinorb` / `noncolin` | bool | `False` | Both needed for SOC. |
| `dft_functional` | string | from the pseudopotential | `pbe`, `pbesol`, `hse`, `scan`, `hf`, ... |

## Geometry relaxation and MD

| keyword | type | default | notes |
| --- | --- | --- | --- |
| `relax_method` | string | `cg` | `bfgs`, `cg`, `sd`, `fire`, `cell-relax` variants. |
| `relax_nmax` | int | `1` (scf), `50` (relax/cell-relax) | Ionic steps. |
| `cal_force` | bool | True for `relax`/`cell-relax`/`md` | Needed for any force consumer. |
| `cal_stress` | bool | True for `cell-relax` | Needed for `cell-relax`, workfunction dipole correction, elastic/energy-strain workflows. |
| `force_thr_ev` | real | `0.0257112` | Force convergence in eV/Angstrom. |
| `stress_thr` | real | `0.5` | Stress convergence in kBar. |
| `md_type` | string | `nvt` | `nve`, `nvt`, `npt`, `langevin`, `msst`, `fire`. |
| `md_nstep` | int | `10` | MD steps. |
| `md_dt` | real | `1.0` | Time step in fs. |
| `md_tfirst` / `md_tlast` | real | none | Target temperature(s). |
| `md_thermostat`, `md_tfreq`, `md_tchain` | | | Thermostat, its frequency, and chain length. |
| `md_pfirst`, `md_plast`, `md_pfreq`, `md_pcouple` | | | Barostat settings for `npt`/`msst`. |
| `md_dumpfreq` | int | `1` | How often `MD_dump` is appended; that file is the trajectory source. |

## DFT+U, vdW, and exact exchange

| keyword | type | default | notes |
| --- | --- | --- | --- |
| `dft_plus_u` | int | `0` | `1` enables the correction. |
| `orbital_corr` | int | `-1` | Per species in `STRU`; `-1` none, `1`/`2`/`3` = p/d/f. |
| `hubbard_u` | real | `0.0` | Per species, eV. |
| `uramping` | real | `-1.0` | Ramp the U up instead of applying it at once. |
| `mixing_dftu` | bool | `False` | Mix the U occupation between SCF steps. |
| `vdw_method` | string | `none` | `d2`, `d3_0`, `d3_bj`, `d3_zero`, `d3_bjm`. |
| `exx_hybrid_alpha` | real | 0.25 (or 1 with `hf`) | Exact-exchange fraction. |

## Outputs

| keyword | type | default | notes |
| --- | --- | --- | --- |
| `out_chg` | int | `0` | `1` writes charge cubes, `3` also spin. `1 10` keeps 10 digits for DDEC. |
| `out_pot` | int | `0` | `2` writes the electrostatic potential, which `workflow workfunc` reads. |
| `out_dos` | int | `0` | Write the DOS. |
| `out_band` | bool/int | `False` | Write the band data files (`BANDS_*.dat`) that band-structure tools read. |
| `out_dm` | bool | `False` | Density matrix; bond-order analysis needs it in gamma-only runs. |
| `out_mat_hs` / `out_mat_hs2` | bool | `False` | Overlap and Hamiltonian matrices. `hs2` is the sparse form `workflow dielectric` uses. |
| `out_mat_r` | bool | `False` | Position matrices for `workflow dielectric`. |
| `out_wfc_lcao` | int | `False` | LCAO wavefunctions, read by COHP and Mayer analysis. |
| `out_mul` | bool | `False` | Mulliken populations (`mulliken.txt`). |
| `out_stru` | bool | `False` | Write the structure of every ionic/MD step. |
| `out_elf` | int | `0` | Electron localization function. |
