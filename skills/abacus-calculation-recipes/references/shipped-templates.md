# Shipped base decks and basis defaults

The example configuration carries a recommended base INPUT per job type under
`input_templates`, and the basis and solver defaults under `basis_settings`.
Copy the deck for the job type and add the system-specific keywords
(`ecutwfc`/`ecutrho` or the orbital cutoff, `kspacing`/KPT, `nspin`, and the
`out_*` flags). The paths in the configuration's `resources` block are
machine-specific and are not repeated here.

## basis_settings

| basis | keywords |
| --- | --- |
| `pw` | `basis_type pw`, `ks_solver dav_subspace`, `pw_diag_ndim 2`, `pw_diag_nmax 20` |
| `lcao` | `basis_type lcao`, `ks_solver genelpa` |

`genelpa` needs an ELPA build; without it the LCAO default falls back to
`scalapack_gvx` (MPI) or `lapack` (serial).

## input_templates

Common to every job type: `mixing_type broyden`, `mixing_beta 0.8`,
`scf_nmax 100`, `scf_thr 1e-7`, `smearing_method gaussian`,
`smearing_sigma 0.015`.

scf:

```text
calculation      scf
mixing_type      broyden
mixing_beta      0.8
scf_nmax         100
scf_thr          1e-7
smearing_method  gaussian
smearing_sigma   0.015
symmetry         1
```

relax:

```text
calculation      relax
cal_force        1
relax_method     bfgs
relax_nmax       100
force_thr_ev     0.02
mixing_type      broyden
mixing_beta      0.8
scf_nmax         100
scf_thr          1e-7
smearing_method  gaussian
smearing_sigma   0.015
symmetry         0
```

cell-relax:

```text
calculation      cell-relax
cal_force        1
cal_stress       1
relax_method     cg
relax_nmax       100
force_thr_ev     0.02
stress_thr       0.5
mixing_type      broyden
mixing_beta      0.8
scf_nmax         100
scf_thr          1e-7
smearing_method  gaussian
smearing_sigma   0.015
symmetry         0
```

md:

```text
calculation      md
md_type          nvt
md_nstep         1000
md_dt            1.0
md_tfirst        300.0
mixing_type      broyden
mixing_beta      0.8
scf_nmax         100
scf_thr          1e-7
smearing_method  gaussian
smearing_sigma   0.015
symmetry         0
```

## Where the recipe recommends deviating

Two points are worth overriding for a production run:

- `scf_thr 1e-7` is the generic template value. For a plane-wave run tighten
  to `1e-8`; for LCAO `1e-7` is usually enough.
- `symmetry 1` in the scf template enables the symmetry reduction. That is only
  safe when the reduction is known to apply to the intended output. Otherwise
  use `symmetry 0`, and use `symmetry -1` for an `nspin 4` SOC calculation so
  that time-reversal symmetry is not imposed.
