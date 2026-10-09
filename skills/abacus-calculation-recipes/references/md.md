# md

Molecular dynamics: the scf parameters plus the ensemble, time step and output
controls. The two ensembles used most often are NVT and NPT.

## Common to every ensemble

| keyword | common value | comment |
| --- | --- | --- |
| `calculation` | `md` | |
| `md_nstep` | 1000-10000 | the default is only a smoke test |
| `md_dt` | about 1.0 fs | smaller for light elements or high temperature |
| `md_dumpfreq` | 10-100 | how often a trajectory frame is written |
| `md_restartfreq` | 50-500 | how often a restart structure is written |
| `dump_force` / `dump_vel` / `dump_virial` | 1 | needed for a useful trajectory |
| `mixing_*`, `smearing_*`, `scf_thr`, `scf_nmax` | as for scf | |

## NVT

| keyword | common value | comment |
| --- | --- | --- |
| `md_type` | `nvt` | |
| `md_thermostat` | `nhc` | Nose-Hoover chain; `anderson`, `berendsen`, `csvr`, ... also exist |
| `md_tfirst` | 300 K | initial and target temperature |
| `md_tlast` | 300 K | if different from `md_tfirst` the temperature is ramped |
| `md_tfreq` | 1/(40*`md_dt`) | NHC oscillation frequency, about 0.025 for `md_dt 1.0` |
| `md_tchain` | 1 | number of Nose-Hoover thermostats |
| `md_seed` | an integer | reproducible initial velocities |

```text
calculation      md
md_type          nvt
md_thermostat    nhc
md_nstep         1000
md_dt            1.0
md_tfirst        300
md_tlast         300
md_tfreq         0.025
md_dumpfreq      10
md_restartfreq   50
dump_force       1
dump_vel         1
dump_virial      1
# scf parameters: ecutwfc/ecutrho, kspacing, smearing, mixing, scf_thr,
# ks_solver, symmetry, nspin
```

## NPT

| keyword | common value | comment |
| --- | --- | --- |
| `md_type` | `npt` | Nose-Hoover style barostat |
| `md_pmode` | `iso` | `iso`, `aniso`, `tri` |
| `md_pfirst` | 1.0 bar | target pressure |
| `md_plast` | 1.0 bar | if different from `md_pfirst` the pressure is ramped |
| `md_pfreq` | 1/(400*`md_dt`) | barostat oscillation frequency, about 0.0025 for `md_dt 1.0` |
| `md_pchain` | 1 | number of thermostats coupled to the barostat |
| temperature controls | as for NVT | |

```text
calculation      md
md_type          npt
md_thermostat    nhc
md_pmode         iso
md_nstep         1000
md_dt            1.0
md_tfirst        300
md_tlast         300
md_tfreq         0.025
md_pfirst        1.0
md_plast         1.0
md_pfreq         0.0025
md_dumpfreq      10
md_restartfreq   50
dump_force       1
dump_vel         1
dump_virial      1
# scf parameters as above
```

## Notes

- Always discard an equilibration segment before averaging; the thermostat and
  barostat targets are not reached from the first step.
- The time step is limited by the fastest vibration in the system; if the
  energy drifts, halve `md_dt`.
- `md_tfreq` and `md_pfreq` are empirical and system dependent. The common
  starting values above (1/(40*md_dt) and 1/(400*md_dt)) are also the ABACUS
  defaults when they are left at 0.
- `md_restartfreq` writes the per-step structures used by `md_restart`; set it
  to a finite value for a long run.
- The per-step output goes through `MD_dump` and `STRU_MD_<step>`; see the
  `abacus-job-files` skill for the names.
