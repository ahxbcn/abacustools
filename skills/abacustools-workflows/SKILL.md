---
name: abacustools-workflows
description: Run the multi-step abacustools workflows that turn one ABACUS job into a derived property or tensor. Covers band structures, phonons, Gruneisen parameters, thermal conductivity, elastic and energy-strain constants, piezoelectricity, Born effective charges, the clamped-ion dielectric tensor, equation of state, cutoff and k-spacing convergence, vacancies, work function, vibrations, DFT+U, magnetic exchange, charge-density difference, finite-difference force/stress validation, and BSSE. Use when a property needs several coordinated ABACUS calculations; use abacustools-postprocess for analysis of a single existing run.
---

# Multi-step workflows

## The two-stage contract

Every workflow is `abacustools workflow NAME prepare -j JOB ...` followed, once
the generated calculations have run, by
`abacustools workflow NAME postprocess -j JOB`. `-j` is the parent job
directory holding the reference `INPUT`/`STRU`; the generated calculations live
below it in a workflow-specific subdirectory.

- `prepare` records its decisions - the task list, the atom partition or
  displacement index, the parameters - in `workflow_<name>.json` next to the
  output. `postprocess` reads that manifest instead of inferring the setup from
  result files or from a directory listing order.
- `postprocess` validates the manifest and checks that the required SCF
  calculations converged before reading their outputs; missing or unconverged
  results stop it with an explicit error rather than producing a number.
- Derived tensor/table results are written next to the manifest as
  `<name>_results.json`, together with the units, the method, and the
  parameters that produced them. Later workflows read those files, so results
  are never transcribed by hand.
- Generated directories are protected: replacing one requires `--override`,
  which is why a prepare can be inspected before it is rerun.

## Typical run

```text
abacustools job prepare -f STRU --library apns -o relax/     # converged reference job
# run relax/, then use it as -j for the workflow
abacustools workflow phonon prepare -j relax --supercell 2 2 2
cd relax/disp-0000 && sbatch submit.slurm    # if scripts were generated, else: abacus > abacus.log
...
abacustools job monitor relax/disp-0000 --json
abacustools workflow phonon postprocess -j relax
```

A band structure has the same two-stage shape, but its prepare stage already
writes the two calculations (`band_scf/`, `band_nscf/`); run the SCF first so
the NSCF can read its density back:

```text
abacustools workflow band prepare -j JOB --npoints 20 --bands 40
abacustools workflow band postprocess -j JOB --emin -6 --emax 6
```

`--submit-script`/`--no-submit-script`/`--submission-type` control generated
submission scripts on the workflows that implement them (`dftu`, `exchange`,
`thermal-conductivity`, `vibration`); see the `abacustools-job-lifecycle`
skill for the `submission` block of `~/.abacustools/config.yaml`.
`postprocess -v/--version` selects the ABACUS log profile and `-o/--output`
renames the result JSON. Most workflow commands do not take `--json`: the
terminal output is the report and the machine-readable result is the
`*_results.json` file the stage writes next to the manifest. The exception is
`workflow band postprocess`, which accepts `--json` to print its report.

Every invocation prints a 9-line ASCII banner to stdout before anything else,
including before the `--json` output of the other command families.

## Which workflow for which property

| Workflow | Property | Reference cell | Notes |
| --- | --- | --- | --- |
| `band` | Band structure along the seekpath path | SCF job | `--npoints`, `--bands`; writes `band_scf/` + `band_nscf/`, postprocess reports the gap and plots `band.png` |
| `phonon` | Phonon dispersion, thermal properties, DOS, irreps | SCF job | `--supercell`, `--displacement-stepsize`; NAC with `--bec-results` + `--dielectric-results` |
| `gruneisen` | Mode Grueneisen parameters `-dln w/dln V` | SCF job | Runs three phonon workflows at `1 +/- strain`; volumes must share supercell, mesh, and step |
| `thermal-conductivity` (alias `kappa`) | Lattice thermal conductivity | SCF job | Needs `phono3py`; `fc3-*` and optional `fc2-*` displacement sets |
| `elastic` | Stiffness tensor / elastic moduli | Relaxed cell | Stress-strain fit, symmetrized by point group; `--strains independent` is the cheap route; `--dimension auto|2d|3d` handles a slab |
| `energy-strain` | Stiffness from energy curvature | Relaxed cell | Use when stresses are noisy |
| `piezoelectric` | Piezoelectric stress tensor | Relaxed polar cell | Berry-phase polarization under strain |
| `bec` | Born effective charges | SCF job | Berry-phase finite differences per atom and direction |
| `dielectric` | Clamped-ion dielectric tensor | LCAO SCF job | Needs `pyatb` + MPI runtime; out-of-process with `--pyatb-command` |
| `eos` | Equilibrium volume, bulk modulus, B0' | SCF or relax job | Birch-Murnaghan fit over `--start/--end/--step` volume scales |
| `ecutwfc`, `kspacing` | Convergence tests | SCF job | `--values`, `--energy-tol` |
| `vacancy` | Vacancy formation energies | Supercell SCF job | Empty-atom supercells plus `ref_element/` references |
| `workfunc` | Surface work function | Slab SCF job | Needs `out_pot 2`, vacuum detection; `--dipole-corr`, empty atoms |
| `vibration` | Molecular vibration frequencies | SCF job | `--index`, ASE or builtin backend, isotope `--mass`, `--gaussian-log` |
| `dftu` | Hubbard U by linear response | SCF job | `--index`/`--elements`, `--u-values` or `--u-min/--u-max/--u-step` |
| `exchange` (alias `magj`) | Magnetic exchange J, four-state method | `nspin 4` job | `--pair`, `--step`, `--number` |
| `chgdiff` | Charge-density difference cube | SCF job | `-i` selects the subsystem atoms |
| `fdforce`, `fdstress` | Validate analytic forces/stresses | SCF job | Finite differences against the analytic values |
| `bsse` | Basis-set superposition error | SCF job | Preparation and postprocessing framework |

## Chaining polar materials

The phonon non-analytical correction needs both halves, and both are produced by
other workflows:

```text
abacustools workflow bec prepare -j JOB --index 1 --dir x y z
abacustools workflow bec postprocess -j JOB          # writes bec_results.json
abacustools workflow dielectric prepare -j JOB       # rerun the generated dielectric/ job
abacustools workflow dielectric postprocess -j JOB   # writes dielectric_results.json
abacustools workflow phonon postprocess -j JOB --irreps \
  --bec-results --dielectric-results --nac-direction c
```

`--bec-results`/`--dielectric-results` read the two JSON files instead of
hand-typed tensors; the raw `--born`/`--dielectric` arguments exist for values
computed elsewhere. Born charges alone, or a dielectric tensor alone, are
refused: the correction needs both. Without `--nac-direction` the Gamma-point
longitudinal mode keeps its transverse frequency and the correction, though
reported, is invisible.

## Details per workflow

- [references/workflow-catalogue.md](references/workflow-catalogue.md) - for
  each workflow: what prepare writes, its options, the directories it
  generates, the postprocess options, and the result files. Read the entry for
  the workflow at hand instead of the whole file.
