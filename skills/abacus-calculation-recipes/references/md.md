# md

Molecular dynamics: the scf parameters plus the ensemble, time step and output
controls.

| keyword | common value | comment |
| --- | --- | --- |
| `calculation` | `md` | |
| `md_type` | `nvt` | `nve`, `nvt`, `npt`, `langevin`, `msst` |
| `md_nstep` | 1000-10000 | the default (10) is only a smoke test |
| `md_dt` | about 1.0 fs | smaller for light elements or high temperature |
| `md_tfirst` | 300 K | thermostat target |
| `md_tlast` | 300 K | ramp target if different from the first temperature |
| `md_dumpfreq` | 10-100 | how often a frame is written |
| `md_restartfreq` | 50-500 | how often a restart structure is written |
| `dump_force` / `dump_vel` / `dump_virial` | 1 | needed for a useful trajectory |
| `mixing_*`, `smearing_*`, `scf_thr`, `scf_nmax` | as for scf | shipped template: broyden 0.8, gaussian 0.015, 1e-7, 100 |
| `nspin` | 1 or 2 | SOC MD is possible but rarer |

For `nvt`/`npt` set the thermostat/barostat targets that match `md_type`; for
`npt` also set the target pressure. For `msst` set the shock parameters.

## Full example

```text
calculation      md
md_type          nvt
md_nstep         1000
md_dt            1.0
md_tfirst        300
md_tlast         300
md_dumpfreq      10
md_restartfreq   50
dump_force       1
dump_vel         1
dump_virial      1
# scf parameters: ecutwfc/ecutrho, kspacing, smearing, mixing, scf_thr,
# ks_solver, symmetry, nspin
```

## Notes

- Always discard an equilibration segment before averaging; the thermostat
  target is not reached from the first step.
- The time step is limited by the fastest vibration in the system; if the
  energy drifts, halve `md_dt`.
- `md_restartfreq` writes the per-step structures used by `md_restart`; set it
  to a finite value for a long run.
- The per-step output goes through `MD_dump` and `STRU_MD_<step>`; see the
  `abacus-job-files` skill for the names.
