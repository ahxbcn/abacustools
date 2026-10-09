# Algorithm and solver support

Two keywords decide how the electronic problem is solved: `basis_type` chooses
the representation and `esolver_type` the driver. `ks_solver` then chooses the
diagonalizer for a Kohn-Sham DFT run.

## Basis

| `basis_type` | meaning | notes |
| --- | --- | --- |
| `pw` | plane waves | needs a pseudopotential per element |
| `lcao` | numerical atomic orbitals | needs an orbital per element |
| `lcao_in_pw` | LCAO input expanded in PW | GPU is not supported for this mode |

## Diagonalizers (`ks_solver`)

The PW and LCAO solver sets are disjoint; the parser refuses a solver from the
wrong list.

| basis | solvers | gating |
| --- | --- | --- |
| PW | `cg`, `dav`, `dav_subspace`, `bpcg` | `dav_subspace` is the recommended fast choice (`pw_diag_ndim 2`); `bpcg` is tested, GPU-oriented |
| LCAO | `genelpa`, `elpa`, `lapack`, `scalapack_gvx`, `cusolver`, `cusolvermp`, `pexsi` | `genelpa` needs an ELPA build and refuses GPU; `elpa` supports CPU and GPU; `lapack` is serial; `cusolver`/`cusolvermp` need CUDA; `pexsi` needs a PEXSI build |

Default resolution when `ks_solver` is `default`: PW gets `cg`; LCAO gets
`genelpa` with an ELPA build, else `scalapack_gvx` under MPI, else `lapack`;
a GPU LCAO run gets `cusolver`.

LTS 3.10.1 additionally lists `cg_in_lcao` (under testing); develop removed it
from the accepted LCAO list.

## Drivers (`esolver_type`)

| `esolver_type` | LTS 3.10.1 | develop | notes |
| --- | --- | --- | --- |
| `ksdft` | yes | yes | standard Kohn-Sham DFT |
| `sdft` | yes | yes | stochastic DFT, PW only |
| `ofdft` | yes | yes | orbital-free DFT |
| `tddft` | yes | yes | time-dependent DFT |
| `tdofdft` | no | yes | time-dependent OFDFT |
| `lj` | yes | yes | Lennard-Jones MD |
| `dp` | yes | yes | DeePMD potential |
| `nep` | no | yes | NEP potential |
| `lr` / `ks-lr` | yes | yes | linear response; `lr` requires `calculation nscf` |
| `dfpt` | no | yes | DFPT driver |

## Other selection keywords

- `device cpu|gpu` and `precision double|single|mixing`.
- `gamma_only` (LCAO) trades k-points for a real-valued Gamma-only run; it
  cannot be combined with `noncolin`.
- `symmetry` controls k-point reduction; `symmetry 0` is required by some
  output paths (for example the dielectric workflow).
- `kpar` (k-point pools) and `bndpar` (band parallel) affect the parallel
  layout, not correctness; their use is covered by the execution skill.

## Practical sequence

1. Fix `basis_type` first; it decides which pseudopotential/orbital files and
   which `ks_solver` values are legal.
2. For LCAO confirm the build has the chosen solver: `genelpa` needs ELPA,
   `cusolver`/`cusolvermp` need CUDA, `pexsi` needs PEXSI.
3. For PW the default `cg` is always available; prefer `dav_subspace` for
   efficiency unless the build lacks it.
4. Pick `esolver_type` only when a non-Kohn-Sham driver is intended; the
   LTS/develop lists differ (develop adds `tdofdft`, `nep`, `dfpt`).

## Driver-specific boundaries

| driver | basis | notes |
| --- | --- | --- |
| `tddft` (RT-TDDFT) | PW and LCAO | most `td_*` keywords are gated on `esolver_type==tddft`; the linear-solver keywords are PW-only, several propagation keywords are LCAO-only |
| `tdofdft` | develop, PW-oriented | time-dependent orbital-free DFT; its keywords are gated on `esolver_type==tdofdft` |
| `dfpt` | develop only | adds `dfpt_qmesh`, `dfpt_qfile`, `dfpt_compute_q0`, `dfpt_loto`, `dfpt_conv_thr`, `dfpt_max_iter`, `dfpt_mix_beta`; `dfpt_loto` requires `dfpt_compute_q0` |
| `lr` / `ks-lr` | both branches | `lr` requires `calculation nscf`, because it reads the converged ground state before the response |
| `sdft` | PW only | stochastic DFT |

The ELF output (`out_elf`) is accepted only for `ksdft`, `ofdft` and `tddft`;
other drivers refuse it.

LTS 3.10.1 has no `tdofdft` and no `dfpt`, so the corresponding keywords and
drivers do not exist there.
