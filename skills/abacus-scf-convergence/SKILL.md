---
name: abacus-scf-convergence
description: Diagnose and fix ABACUS SCF convergence problems by tuning the mixing parameters, initial density and smearing. Covers oscillation, slow monotonic convergence, divergence, charged and magnetic systems, slabs with vacuum, meta-GGA (mixing_tau) and DFT+U. Use when an SCF oscillates, fails to reach scf_thr, or converges in the density but not in the energy. Not for a basis or k-mesh that was never converged, and not for choosing the functional or pseudopotential.
---

# ABACUS SCF convergence

Most SCF failures are a mixing problem, not a physics problem. Identify the
failure pattern first, then change one mixing parameter at a time. Do not
loosen `scf_thr` to make a run "converge": a density that is not converged
poisons every later step.

## Decide which failure it is

| pattern in the running log | usual cause | first move |
| --- | --- | --- |
| `drho` falls smoothly but slowly, `scf_nmax` hit | mixing too cautious, or `mixing_ndim` too small | raise `mixing_beta` slightly, or raise `mixing_ndim` |
| `drho` and the energy oscillate | mixing too aggressive | lower `mixing_beta`; enable Kerker |
| `drho` grows instead of falling | setup problem or far too aggressive mixing | check smearing, `nspin` and initial moments, structure; then lower `mixing_beta` |
| density converges but the energy keeps changing | meta-GGA tau not mixed | set `mixing_tau 1` |
| convergence depends on a good initial guess | poor starting density/magnetization | set initial moments; use `init_chg`/`init_wfc file` and `chg_extrap` |

Also confirm the units and type of the criterion: `scf_thr_type 1` is a
reciprocal-space density error in Ry (PW), `scf_thr_type 2` is a real-space
density error (LCAO). `scf_ene_thr` is a separate total-energy threshold in eV.

## Tuning order

1. Fix the physical setup before the mixer: correct `smearing_method` and
   `smearing_sigma` for a metal or an insulator, a consistent `nspin`, a
   sensible `ecutwfc`/k set, and a reasonable starting structure.
2. Keep `mixing_type broyden` (the default). `pulay` is a little slower;
   `plain` is for testing.
3. Lower `mixing_beta` if the run oscillates; raise it a little if it is
   stable but slow. Recommended starts: 0.8 (`nspin 1`), 0.4 (`nspin 2/4`).
   A progressive strategy is 0.4 -> 0.1 -> 0.025.
4. Raise `mixing_ndim` for hard systems (default 8; try 20-30).
5. Use `mixing_gg0` (Kerker) for long-wavelength charge sloshing, typical
   value 1.0. Note the implementation bypasses the charge Kerker
   preconditioner when `mixing_beta <= 0.1`, so it has no effect in that
   regime.
6. Magnetic density converges separately: `mixing_beta_mag` defaults to
   `4*mixing_beta` capped at 1.6. Lower it toward 1.5-2 times `mixing_beta`
   when the moment oscillates.
7. For meta-GGA, set `mixing_tau 1`.
8. Give a better start: correct initial magnetic moments, `init_chg`/`init_wfc`
   from a previous run, and `chg_extrap` during relax/cell-relax/md.
9. Last, adjust the criterion: `scf_thr`, `scf_ene_thr` and `scf_nmax`.

## The three common hard cases

- Slab or other low-dimensional large system: `mixing_beta 0.025`,
  `mixing_ndim 20-30`.
- Magnetic system: set the initial moments correctly in `STRU`, then lower
  `mixing_beta_mag` from the default `4*mixing_beta` to about 1.5-2 times
  `mixing_beta`.
- meta-GGA: `mixing_tau 1`, so the kinetic energy density is mixed together
  with the density and the energy converges with the density.

Concrete decks for these and other cases are in
[references/convergence-playbook.md](references/convergence-playbook.md).

## References

- [references/mixing-parameters.md](references/mixing-parameters.md) - every
  mixing keyword, its default, its effect and its unit.
- [references/diagnostics.md](references/diagnostics.md) - how to read the SCF
  log and tell oscillation from slow convergence.
- [references/convergence-playbook.md](references/convergence-playbook.md) -
  ready parameter sets for slab, magnetic, metallic, insulating, meta-GGA and
  DFT+U cases.

Further reading: the ABACUS SCF convergence tutorial at
https://mcresearch.github.io/abacus-user-guide/abacus-conv.html.
