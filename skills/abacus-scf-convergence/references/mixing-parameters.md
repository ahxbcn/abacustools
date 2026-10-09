# Mixing parameters

All values below are the ABACUS defaults and the ranges that are useful when
the SCF does not converge on its own. Change one parameter at a time and keep
the run reproducible.

## Charge-density mixing

| keyword | default | effect and useful range |
| --- | --- | --- |
| `mixing_type` | `broyden` | `plain`, `pulay`, `broyden`; Broyden is usually a little faster than Pulay. Keep the default unless debugging. |
| `mixing_beta` | 0.8 (`nspin 1`), 0.4 (`nspin 2/4`) | `rho_new = rho_old + mixing_beta * drho`. Larger is faster but less stable. For hard cases go below 0.1; a progressive strategy is 0.4 -> 0.1 -> 0.025. |
| `mixing_ndim` | 8 | Number of previous densities used by Pulay/Broyden. Raise it for hard systems, typically to 20-30. |
| `mixing_gg0` | 1.0 | Kerker preconditioner for the charge density; suppresses long-wavelength sloshing. The implementation bypasses it when `mixing_beta <= 0.1`, so it has no effect in that regime. |
| `mixing_gg0_min` | 0.1 | Lower bound of the Kerker filter, evaluated as `mixing_gg0_min / mixing_beta`. |
| `mixing_restart` | 0 | If `drho` falls below this value, the mixer restarts once from the current density. Useful for a run that stalls near convergence. |
| `mixing_tau` | false | Also mix the kinetic energy density. Set to `1` for a meta-GGA that converges in the density but not in the energy. |

Source guidance worth keeping in mind: for a low-dimensional large system the
combination `mixing_beta 0.1`, `mixing_ndim 20`, `mixing_gg0 1.0` usually works
well.

## Magnetic-density mixing

| keyword | default | effect and useful range |
| --- | --- | --- |
| `mixing_beta_mag` | `4*mixing_beta`, capped at 1.6 | Mixing of the magnetization. When the moment oscillates, lower it toward 1.5-2 times `mixing_beta` (for example `mixing_beta 0.1` with `mixing_beta_mag 0.2`). |
| `mixing_gg0_mag` | 0.0 | Kerker preconditioner for the magnetic density; off by default, enable only for a magnetic density that is hard to converge. Bypassed when `mixing_beta_mag <= 0.1`. |
| `mixing_angle` | -10.0 (`nspin 4` only) | Angle mixing to relax the moment directions toward the ground-state configuration; only the value 1.0 is implemented. |

Setting the initial moments in `STRU` correctly is as important as the mixing:
a wrong starting configuration can look like a mixer failure.

## DFT+U

`mixing_dftu` (default false) additionally mixes the occupation matrices.
Experience in the source notes that it is not very helpful when a +U run does
not converge; lowering `mixing_beta` and fixing the initial occupations is the
usual route.

## Convergence criteria

| keyword | default | effect |
| --- | --- | --- |
| `scf_thr` | 1e-9 (PW), 1e-7 (LCAO) | Density error threshold. Unit depends on `scf_thr_type`. |
| `scf_thr_type` | 1 (PW), 2 (LCAO) | 1: reciprocal-space density error in Ry; 2: real-space density error (dimensionless). |
| `scf_ene_thr` | -1 (off) | Total-energy threshold in eV, checked in addition to `scf_thr`. Use it when the density converges but the energy does not. |
| `scf_nmax` | 100 | Maximum electronic iterations. Raise it before changing the mixer if the run is on a smooth, slow trajectory. |

## Starting point

- `init_chg` / `init_wfc`: `atomic` by default; `file` reads a previous
  density/wavefunction from `read_file_dir`, which is the strongest lever for a
  run that only converges from a good guess.
- `chg_extrap`: for `relax`/`cell-relax` the default is first-order and for
  `md` second-order, so the density is extrapolated between ionic steps; this
  is what keeps an optimization or MD run from re-converging from scratch each
  step.
- `smearing_method` / `smearing_sigma`: a metal needs a real smearing
  (gaussian or mp, sigma around 0.01-0.02 Ry); an insulator uses gaussian with
  a small sigma. A wrong smearing looks exactly like a mixer failure.
