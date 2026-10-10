# Reading an SCF that will not converge

Work from the per-iteration output, not from the final flag. The running log
prints, for each electronic step, the energy and the density error, and the
shape of those two curves tells you which parameter to change. The log names
are described in the `abacus-job-files` skill.

## What the numbers mean

- The density error is `drho`. Its unit depends on `scf_thr_type`: a
  reciprocal-space error in Ry for a PW run (`scf_thr_type 1`), a real-space
  error for LCAO (`scf_thr_type 2`). Compare it with `scf_thr` of the same
  type, and never with the other basis.
- The total energy is printed separately. `scf_ene_thr` (eV) is what checks it;
  by default it is off and only `scf_thr` gates convergence.

## Patterns

| pattern | what it looks like | interpretation |
| --- | --- | --- |
| smooth and slow | `drho` decreases by a small factor each step, no upturn | the mixer is too cautious or the history is too short; the setup is probably fine |
| oscillation | `drho` and the energy bounce between values, no net decay | the mixing is too aggressive for the charge or the magnetization |
| divergence | `drho` grows for several steps | usually a wrong smearing, a wrong `nspin`/moment, or a bad structure; mixing is the second suspect |
| charge stalls, moment moves | the total density is flat but the energy is not | the magnetic mixing is the problem, not the charge mixing |
| density converges, energy does not | `drho` below `scf_thr`, energy still drifting | meta-GGA tau not mixed, or `scf_ene_thr` not set |
| reconverges from scratch each ionic step | the first SCF step of every relaxation step starts far from a solution | `chg_extrap` or a previous density is missing |

## First checks before touching the mixer

1. Is the smearing right for the system? A metal with no real smearing, or an
   insulator with a large one, converges badly.
2. Is `nspin` consistent with the intended magnetic state, and are the initial
   moments set in `STRU`?
3. Was `ecutwfc`/k point actually converged? A too-small cutoff produces noisy
   energies that no mixer can fix.
4. Is `scf_thr_type` the default for the basis, and is `scf_thr` not absurdly
   tight for the system size?

## Then change one thing

- Oscillation, charge only: lower `mixing_beta` (0.4 -> 0.1 -> 0.025) and, if
  it is a long-wavelength problem, keep `mixing_gg0` at its default 1.0 while
  `mixing_beta` is above 0.1.
- Slow but stable: raise `mixing_ndim` (8 -> 20-30) and raise `scf_nmax`.
- Magnetic oscillation: lower `mixing_beta_mag` toward 1.5-2 times
  `mixing_beta`.
- meta-GGA: set `mixing_tau 1` and keep an eye on `scf_ene_thr`.
- Still stuck: restart from a good density with `init_chg file` and
  `read_file_dir`, or use `mixing_restart` for a run that stalls near the
  threshold.
