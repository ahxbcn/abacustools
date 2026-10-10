# relax and cell-relax

Both relax the ions; `cell-relax` also relaxes the cell. They reuse every scf
parameter and add a convergence criterion and an optimization method.

## relax (fixed cell)

| keyword | common value | comment |
| --- | --- | --- |
| `calculation` | `relax` | |
| `cal_force` | 1 | the default is already on for relax |
| `relax_method` | LTS `bfgs_trad`, develop `bfgs` | see the branch note below |
| `relax_nmax` | 100-200 | |
| `force_thr_ev` | 0.02 eV/Angstrom | 0.01-0.03 is the usual range |
| `scf_thr` | 1e-8, or 1e-6 for a large structure | the SCF is inside the loop |
| `symmetry`, `ks_solver`, `mixing_*`, `smearing_*` | as for scf | |

Branch note on the method:

- LTS 3.10.1 accepts `cg`, `bfgs`, `sd`, `cg_bfgs`, `bfgs_trad`. The recommended
  choice for a plain ionic relaxation is `bfgs_trad`.
- develop drops `bfgs_trad` and recommends `bfgs` (optionally `bfgs 2`, which is
  the default and updates the inverse Hessian directly). `lbfgs` is available
  and suits large systems.

## cell-relax

| keyword | common value | comment |
| --- | --- | --- |
| `calculation` | `cell-relax` | |
| `cal_force` | 1 | |
| `cal_stress` | 1 | the default is already on for cell-relax |
| `relax_method` | `cg` | the only method that optimizes cell and ions together |
| `relax_nmax` | 100-200 | |
| `force_thr_ev` | 0.02 eV/Angstrom | tighten for a simple crystal |
| `stress_thr` | 0.5 kBar | tighten for a simple crystal |

The algorithm caveat is the important one: **only `cg` updates the atomic
positions and the cell parameters simultaneously**. Any other method is
implemented as a two-level loop (an inner relaxation of the ions at fixed cell
and an outer update of the cell), which is much slower and is best avoided for
`cell-relax`.

For a simple, high-symmetry crystal the cell degrees of freedom are few and
well conditioned, so tighten the thresholds, for example
`force_thr_ev 0.005-0.01` and `stress_thr 0.1-0.3`. A loose threshold on an
easy crystal leaves a residual strain that shows up in a later phonon or
elastic calculation.

## Full examples

```text
calculation      relax
cal_force        1
relax_method     bfgs_trad      # LTS 3.10.1; bfgs on develop
relax_nmax       100
force_thr_ev     0.02
# scf parameters: ecutwfc/ecutrho, kspacing, smearing, mixing, scf_thr,
# ks_solver, symmetry, nspin
```

```text
calculation      cell-relax
cal_force        1
cal_stress       1
relax_method     cg
relax_nmax       100
force_thr_ev     0.02           # 0.005-0.01 for a simple crystal
stress_thr       0.5            # 0.1-0.3 for a simple crystal
# scf parameters as above
```

## Notes

- The k-point sampling of a `cell-relax` should be rechecked after the cell
  changes size, because a fixed mesh becomes a different sampling density.
- Constraints in `STRU` (fixed atoms or directions) are respected by both
  modes; use them instead of editing the cell by hand.
- For a large structure the inner SCF can run at `scf_thr 1e-6`; the final
  energy used for the force/stress is what matters, so keep the force and
  stress thresholds as the real convergence gate.
- In LTS, the new relaxation path (`relax_new`) requires the `cg` method; with
  any other method it is disabled and the older two-level path is used. The
  develop branch expresses the same distinction as `cg 2` (simultaneous) versus
  `cg 1` (two-level).
