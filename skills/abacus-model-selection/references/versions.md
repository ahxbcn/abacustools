# Pinned source trees

The support facts in this skill are read from two local checkouts. Pin them
when citing support, and re-read the source before trusting a newer checkout.

| branch | version | commit | commit date | path |
| --- | --- | --- | --- | --- |
| LTS 3.10 | `v3.10.1` | `f71921fe848659deac8db319cd4311b55b5ad480` | 2025-11-21 | `~/software/abacus-develop-LTSv3.10.1` (`LTS` branch) |
| develop | `v3.11.0-beta10` | `260139d97786fa1ac3a416e72d1a19f7b221a5db` | 2026-10-06 | `~/software/abacus-develop` (`develop` branch) |

Reproduce the pin:

```bash
git -C ~/software/abacus-develop-LTSv3.10.1 log -1 --format='%H %ad %s' --date=short
git -C ~/software/abacus-develop  log -1 --format='%H %ad %s' --date=short
cat ~/software/abacus-develop-LTSv3.10.1/source/version.h
cat ~/software/abacus-develop/source/source_main/version.h
```

The LTS working tree carries unrelated modifications under `toolchain/` and
`examples/`, so `git describe` reports it dirty; the compiled behavior follows
`source/`, whose contents match the pinned commit. The develop tree is clean.
The version strings live in `source/version.h` (LTS) and
`source/source_main/version.h` (develop).

## Compile-time gating

Several supported features exist in the source but only at runtime if the build
enabled them. The support matrix in `SKILL.md` assumes a build with the option
available; a stripped build refuses the same input with a message.

| feature | build option | effect when off |
| --- | --- | --- |
| LIBXC functionals | `ENABLE_LIBXC` (CMake) | SCAN/SCAN0/SCANL, HSE, B3LYP, LC/LRC/CAM hybrids, and LIBXC expressions are refused |
| GEN-ELPA solver | `ENABLE_ELPA` | `ks_solver genelpa`/`elpa` refused; use `scalapack_gvx` |
| GPU solvers | `USE_CUDA` | `device gpu`, `cusolver`, `cusolvermp` unavailable |
| DFT-D4 | `dftd4` package (CMake `dftd4_VERSION >= 4.3.0`) | `vdw_method d4` refused |
| PEXSI | `ENABLE_PEXSI` (>= 2.0.0) | `ks_solver pexsi` unavailable |
| NPZ output | `ENABLE_CNPY` | `.npz` matrix/wavefunction output refused |
| MPI | `ENABLE_MPI` | LCAO defaults to `lapack`; only serial runs |

The input parser checks the algorithm lists at read time and quits with an
explicit message when a solver is missing, so an unsupported combination fails
before the first SCF step rather than silently falling back.
