# relax and cell-relax

Both relax the ions; `cell-relax` also relaxes the cell. They reuse every scf
parameter and add a convergence criterion.

## relax

| keyword | common value | comment |
| --- | --- | --- |
| `calculation` | `relax` | |
| `cal_force` | 1 | the default is already on for relax |
| `relax_method` | `bfgs` | `cg` is a common alternative |
| `relax_nmax` | 100-200 | |
| `force_thr_ev` | 0.01-0.03 eV/Angstrom | 0.02 is a common start |
| `symmetry`, `ks_solver`, `mixing_*`, `smearing_*`, `scf_thr` | as for scf | the shipped template uses 0.8, gaussian 0.015, 1e-7, 0 |

## cell-relax

| keyword | common value | comment |
| --- | --- | --- |
| `calculation` | `cell-relax` | |
| `cal_force` | 1 | |
| `cal_stress` | 1 | the default is already on for cell-relax |
| `relax_method` | `cg` | `bfgs` is also used |
| `relax_nmax` | 100-200 | |
| `force_thr_ev` | 0.01-0.03 eV/Angstrom | |
| `stress_thr` | 0.5-1 kBar | 0.5 is the shipped value |

## Full examples

```text
calculation      relax
cal_force        1
relax_method     bfgs
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
force_thr_ev     0.02
stress_thr       0.5
# scf parameters as above
```

## Notes

- The k-point sampling of a `cell-relax` should be rechecked after the cell
  changes size, because a fixed mesh becomes a different sampling density.
- A too-loose `force_thr_ev` gives a structure whose forces are still visible
  in a following phonon or elastic calculation; use the tighter end of the
  range for those.
- Constraints in `STRU` (fixed atoms or directions) are respected by both
  modes; use them instead of editing the cell by hand.
- For LCAO, keep the same basis and solver choices as the scf; the relaxed
  structure only changes the cell and positions.
