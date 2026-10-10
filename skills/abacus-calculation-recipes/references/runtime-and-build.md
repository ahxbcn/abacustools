# Runtime, hardware and build parameters

These are the INPUT parameters that interact with the machine and the build,
as opposed to the physics/numerics parameters of the calculation itself.

## Device and precision

| keyword | common value | comment |
| --- | --- | --- |
| `device` | `cpu` | `gpu` requires a CUDA build |
| `precision` | `double` | `single` and `mixing` are for testing or memory pressure |

On GPU the solver must be GPU-capable: `cusolver` or `cusolvermp` for LCAO,
`elpa` for either basis. `genelpa` refuses GPU and aborts with an explicit
message.

## Solver gating

| solver | basis | requires |
| --- | --- | --- |
| `dav_subspace` (+ `pw_diag_ndim 2`, `pw_diag_nmax 20`) | PW | always available; the recommended combination |
| `cg`, `dav`, `bpcg` | PW | always available |
| `genelpa` | LCAO | an ELPA build; CPU only; usually the fastest |
| `elpa` | LCAO | ELPA; supports CPU and GPU |
| `scalapack_gvx` | LCAO | MPI; the fallback when ELPA is absent |
| `lapack` | LCAO | serial build |
| `cusolver`, `cusolvermp` | LCAO | CUDA |
| `pexsi` | LCAO | a PEXSI build and the PEXSI library |

## MPI and OpenMP layout

- Launch with `mpirun -np N` (or the scheduler's equivalent); set
  `OMP_NUM_THREADS` to the cores available per rank.
- `kpar` divides the k points into pools; useful when the k set is large.
- `bndpar` divides the bands; useful when there are many bands per k point.
- Small jobs: `kpar 1`, `bndpar 1`, and put the parallelism into OpenMP or a
  single rank per GPU.
- A common rule of thumb is `kpar` no larger than the number of k points, and
  a total rank count that the memory per rank can support.

## Memory

- Run `calculation test_memory` before a large job; it estimates the peak
  memory for the current settings.
- Memory grows with `nbands`, the plane-wave count (`ecutwfc`) or the number of
  orbitals (LCAO), and the number of k-point pools.
- A larger `pw_diag_ndim` (the Davidson workspace) also costs memory; the
  recommended `2` with `dav_subspace` keeps it small.

## Debugging output

| keyword | comment |
| --- | --- |
| `out_alllog` | one running log per rank; use only when debugging MPI |
| `calculation test_neighbour` | LCAO: list neighbor information; needs a search radius |

## Build flags that gate the options

| flag | gates |
| --- | --- |
| `ENABLE_MPI` | multi-rank runs; without it LCAO falls back to `lapack` |
| `ENABLE_ELPA` | `genelpa` and `elpa` |
| `USE_CUDA` | `device gpu`, `cusolver`, `cusolvermp` |
| `ENABLE_LIBXC` | SCAN/SCAN0/SCANL, HSE, B3LYP and the other LIBXC hybrids |
| `ENABLE_PEXSI` | `ks_solver pexsi` |
| `USE_PAW` | LTS-only legacy PAW (see `abacus-model-selection`) |

Site facts - queue names, node memory, module loads, scratch paths - belong in
the runtime configuration of the machine, not in this skill.
