# Pinned source versions

The support facts in this skill were read from two ABACUS source revisions.
Pin the revision when citing support, and re-read the source before trusting a
newer one.

| branch | version | commit | commit date |
| --- | --- | --- | --- |
| LTS 3.10 | `v3.10.1` | `f71921fe848659deac8db319cd4311b55b5ad480` | 2025-11-21 |
| develop | `v3.11.0-beta10` | `260139d97786fa1ac3a416e72d1a19f7b221a5db` | 2026-10-06 |

Reproduce the pin from any checkout of the revision:

```bash
git log -1 --format='%H %ad %s' --date=short
# LTS keeps the string in source/version.h; develop in source/source_main/version.h
sed -n 's/.*VERSION "\(v[^"]*\)".*/\1/p' source/version.h 2>/dev/null \
  || sed -n 's/.*VERSION "\(v[^"]*\)".*/\1/p' source/source_main/version.h
```

Attribute behavior to the commit, not to whatever is on a working disk: a
checkout can carry unrelated local changes, so read the tree the commit names.
The `LTS` and `develop` branch names above are the upstream branch of each
revision, not a local checkout path.

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
| PAW (LTS only) | `USE_PAW` | `use_paw true` refused |
| MPI | `ENABLE_MPI` | LCAO defaults to `lapack`; only serial runs |

The input parser checks the algorithm lists at read time and quits with an
explicit message when a solver is missing, so an unsupported combination fails
before the first SCF step rather than silently falling back.
