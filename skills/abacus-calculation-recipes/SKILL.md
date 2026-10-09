---
name: abacus-calculation-recipes
description: Parameter recipes for the most common ABACUS calculations - scf, nscf, relax, cell-relax and md - with the plane-wave and LCAO bases, plus the runtime, hardware and build parameters usually set for a real run. Use when writing or reviewing an INPUT deck and choosing values for a common calculation, or when a run is slower or less stable than expected. Covers ABACUS-supported keywords only; it does not choose the scientific method, the functional or the system-specific defaults.
---

# ABACUS calculation recipes

This skill lists, in order of setting type, the parameters that are usually set
for the common calculations, together with the values that are commonly used.
Treat every number as a starting point that still has to pass a convergence
check, not as a universal constant. Which functionals, pseudopotentials and
spin treatments are supported is answered by the `abacus-model-selection`
skill; file layout and output names by the `abacus-job-files` skill; STRU/KPT
syntax by `abacus-inputs`; pseudopotential/orbital choice by `abacus-basis`.

## Always set these first

| keyword | common value | comment |
| --- | --- | --- |
| `basis_type` | `pw` or `lcao` | decides the solver set and the resource files |
| `ks_solver` | PW `dav_subspace`, LCAO `genelpa` | see the note below |
| `ecutwfc` | PW 50-80 Ry; LCAO from the orbital | PW must converge this |
| `ecutrho` | NC 4x `ecutwfc`; USPP 8-12x | PW only |
| `kspacing` | about 0.14 (1/Bohr) for bulk | see the note below |
| `smearing_method` | `gaussian` | metals and insulators alike |
| `smearing_sigma` | 0.015 Ry (the shipped template); insulators 0.001-0.005 Ry | in Ry |
| `mixing_type` | `broyden` | |
| `mixing_beta` | 0.8 for `nspin 1`, 0.4 for `nspin 2/4` | lower it if the SCF oscillates |
| `scf_thr` | shipped template 1e-7; PW production 1e-8, LCAO 1e-7 | tighten for production |
| `scf_nmax` | 100-200 | |
| `symmetry` | 0 (the shipped scf template uses 1) | see the note below |
| `nspin` | 1, 2 or 4 | |

Two frequent mistakes:

- `kspacing` is not VASP's `KSPACING`. The ABACUS value is in 1/Bohr and the
  definition differs by a factor of 2*pi (the ABACUS value corresponds to the
  VASP value times 2*pi), so a VASP number cannot be pasted in. For a bulk
  crystal about 0.14 is a good starting point for the convergence test.
- An insulator is normally run with `smearing_method gaussian` and a small
  `smearing_sigma` (for example 0.001-0.005 Ry), not with `fixed` smearing.

## Solver and symmetry notes

- PW: `ks_solver dav_subspace` with `pw_diag_ndim 2` is the usually recommended
  combination.
- LCAO: the default `genelpa` is generally the fastest, but it needs a build
  with the ELPA library; without it use `scalapack_gvx`.
- `symmetry 0` is the general-purpose setting. Use `symmetry 1` only in the few
  cases where the symmetry reduction is wanted and known to be safe. For an
  `nspin 4` calculation that includes SOC, set `symmetry -1` so that
  time-reversal symmetry is not imposed on top of the spin treatment.

## 1. scf

The base run: total energy, density and magnetization.

```text
calculation      scf
basis_type       pw
ecutwfc          60
ecutrho          240            # NC; use 8-12x for USPP
kspacing         0.14
smearing_method  gaussian
smearing_sigma   0.015
mixing_type      broyden
mixing_beta      0.8
scf_thr          1e-8
scf_nmax         100
ks_solver        dav_subspace
pw_diag_ndim     2
symmetry         0
nspin            1
```

LCAO variant: `basis_type lcao`, `ks_solver genelpa`, `scf_thr 1e-7`,
`ecutwfc` at least the cutoff encoded in the orbital file; `gamma_only 1` is
allowed for a Gamma-only run. Details in [references/scf.md](references/scf.md).

## 2. nscf

Needs the charge density of a preceding SCF and a denser or path-like k set.

```text
calculation      nscf
read_file_dir    OUT.ABACUS
init_chg         file
nbands           40             # comfortably above the occupied count
kspacing         0.14           # or a line-mode KPT for a band path
out_band         1
# out_dos        1              # for DOS/PDOS instead of bands
symmetry         0
```

A line-mode `KPT` is required for a band path; a dense mesh is used for DOS.
Details in [references/nscf.md](references/nscf.md).

## 3. relax

Ionic relaxation at fixed cell.

```text
calculation      relax
cal_force        1
relax_method     bfgs
relax_nmax       100
force_thr_ev     0.02
# plus the scf parameters above
```

## 4. cell-relax

Ionic and cell relaxation together.

```text
calculation      cell-relax
cal_force        1
cal_stress       1
relax_method     cg
relax_nmax       100
force_thr_ev     0.02
stress_thr       0.5            # kBar
# plus the scf parameters above
```

`relax` and `cell-relax` share the thresholds and methods; details in
[references/relax-and-cell-relax.md](references/relax-and-cell-relax.md).

## 5. md

Molecular dynamics: the SCF parameters plus the ensemble and dump controls.

```text
calculation      md
md_type          nvt
md_nstep         1000
md_dt            1.0            # fs
md_tfirst        300            # K
md_tlast         300
md_dumpfreq      10
md_restartfreq   50
dump_force       1
dump_vel         1
dump_virial      1
# plus the scf parameters above
```

The default `md_nstep` is far too small for production; set it explicitly.
Details in [references/md.md](references/md.md).

## Shipped templates

The example `~/.abacustools/config.yaml` carries a recommended base deck per job
type under `input_templates`, plus the basis/solver defaults under
`basis_settings`. Those are the values used above unless a note says
otherwise. The full recommended decks are reproduced in
[references/shipped-templates.md](references/shipped-templates.md).

## Runtime, hardware and build

| keyword | common value | comment |
| --- | --- | --- |
| `device` | `cpu` | `gpu` needs a CUDA build and a GPU solver |
| `precision` | `double` | `single`/`mixing` only for testing |
| `kpar` | 1 for small jobs | number of k-point pools |
| `bndpar` | 1 for small jobs | band parallel groups |
| `out_alllog` | 0 | set to 1 only to debug MPI |

Launch with `mpirun -np N` (or the scheduler's equivalent) and set
`OMP_NUM_THREADS` to the cores per rank. Use `calculation test_memory` to size a
large job. `genelpa` refuses GPU; `cusolver`/`cusolvermp` need CUDA. The exact
solver and feature set depends on the build flags (`ENABLE_MPI`, `ENABLE_ELPA`,
`USE_CUDA`, `ENABLE_LIBXC`, `ENABLE_PEXSI`). Site details such as queue names,
node memory and module names belong in runtime configuration, not here. See
[references/runtime-and-build.md](references/runtime-and-build.md).

## Version differences

Keyword names and defaults changed between the LTS and develop branches; the
values above are the common ones, but confirm the keyword on the branch at
hand. See [references/version-differences.md](references/version-differences.md).

## References

- [references/scf.md](references/scf.md) - scf, PW and LCAO.
- [references/nscf.md](references/nscf.md) - nscf, bands and DOS.
- [references/relax-and-cell-relax.md](references/relax-and-cell-relax.md) -
  ionic and cell relaxation.
- [references/md.md](references/md.md) - molecular dynamics.
- [references/runtime-and-build.md](references/runtime-and-build.md) - solver
  gating, MPI/OpenMP layout, memory, GPU.
- [references/version-differences.md](references/version-differences.md) - LTS
  3.10 versus develop.
- [references/shipped-templates.md](references/shipped-templates.md) - the
  recommended base deck per job type and the basis/solver defaults.
