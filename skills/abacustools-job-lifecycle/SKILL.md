---
name: abacustools-job-lifecycle
description: Prepare, validate, and monitor ABACUS job directories with the abacustools job family. Covers building self-contained jobs from a structure, switching their pseudopotential/orbital library, pre-flight checks that need no output, single-job state, step-by-step convergence monitoring for SCF/relax/cell-relax/MD, batch monitoring, and the submission-script and ABACUS-version configuration behind them. Use while a calculation is being set up or is running; use abacustools-postprocess once results exist.
---

# ABACUS job lifecycle

A "job" is one directory holding `INPUT`, `STRU` (or the file named by
`stru_file`), optionally `KPT`, the referenced pseudopotential/orbital files,
and afterwards `OUT.<suffix>/`, `running_<calculation>.log`, and the submit
script. Everything in this skill reads or writes such a directory; none of it
parses finished scientific results.

## Banner and exit codes

`abacustools` prints a 9-line ASCII banner to stdout before every command, so
the JSON of an `--json` flag starts at line 10; strip it before parsing.
`job validate` exits `1` for an invalid job and `0` for a valid one, argparse
errors exit `2`, and any other failure surfaces as a Python traceback on stderr.

## Prepare

```text
abacustools job prepare -f STRU --library apns -o runs/
abacustools job prepare -f 'structures/*.cif' --job-type cell-relax --nspin 2
abacustools job prepare -f STRU --input template/INPUT --set scf_thr 1e-8
abacustools job prepare -f 'structures/*.cif' -o runs/ --submit-config
abacustools job prepare -f 'structures/*.cif' -o runs/ --submit-config \
    --abacus-command 'mpirun -np 32 abacus'
```

Each generated job is self-contained: resources are symlinked into it (copied
with `--copy-resources`), `STRU` refers to them by file name, and `INPUT` comes
from the configured template for the job type. `--folder-syntax` names the
folders from an f-string over `{x}` (source file name) and `{i}` (index);
`--override` is required to replace an existing folder. `--lcao`, `--soc`,
`--dftu`/`--dftu-param`, `--init-mag`/`--afm`, and `--basis` control the
electronic-structure setup. `--submit-config`/`--no-submit-config` write (or
suppress) the batch-runner file next to the generated directories. For the
input keywords themselves, see the `abacus-inputs` skill (and `abacus-basis`
for the resource files).

## Switch the resources of an existing job

```text
abacustools job setpporb JOB --library sg15
abacustools job setpporb JOB1 JOB2 --library apns --variant precision
abacustools job setpporb JOB --library sg15 --variant SZ --copy-resources
abacustools job setpporb JOB --library sg15 --dry-run
```

`setpporb` re-resolves every element of the job's `STRU` from the selected
library, rewrites the `ATOMIC_SPECIES` pseudopotential name and the
`NUMERICAL_ORBITAL` entry of each species, installs the new files, and removes
the resource files the old `STRU` referenced. With no `JOB` it updates the
current directory. New files are copied when the job already held copies and
symlinked when it held symlinks; `--copy-resources`/`--symlink` force either,
and `--dry-run` reports the resolved names and changes without writing. It also
warns when the new orbitals carry a higher plane-wave cutoff than the job's
`ecutwfc`.

## Pre-flight checks

```text
abacustools job checkinput JOB            # INPUT + STRU + KPT + resource files
abacustools job checkinput JOB --strict   # unknown keywords are errors
abacustools job checkinput JOB --json
abacustools job validate JOB --json       # same diagnostics, job-directory view
abacustools job status JOB --json         # state + latest progress, one job
```

`checkinput` does not read `OUT.*` or logs, so it works before the first run.
It reports the calculation settings, structure size and composition, cell
parameters, k-point mode, and the resource files, with issues carrying a level
and a code. `status` adds validation plus the state and the latest progress of
a job that has started.

## Monitor a running job

`monitor` prints one update and returns; it never waits for the job to finish,
so it is safe to call repeatedly.

```text
abacustools job monitor JOB
abacustools job monitor JOB --scf-steps
abacustools job monitor JOB --tail 0 --csv steps.csv --plot
abacustools job monitor JOB --json
abacustools job monitor-many -j JOB1 JOB2 JOB3 --json
```

- SCF: current electronic progress; `--scf-steps` adds energy, energy change,
  and density error per iteration.
- `relax`: criteria first, then per ionic step the energy, energy change,
  largest force with its atom and Cartesian component, and the RMS and maximum
  atomic displacement from the previous step (empty when `STRU` cannot be read).
- `cell-relax`: also the largest stress with its Voigt component.
- `md`: total/potential/kinetic energy, temperature, pressure per step, plus
  the thermostat or barostat target for the `md_type`.
- Step tables show the last 30 steps; `--tail N` changes that and `--tail 0`
  prints all. `--json`, `--csv`, and `--plot` always carry the full history.
  Energies are eV, forces eV/Angstrom, stresses and pressures kBar,
  temperatures K. The current incomplete step is shown while the run is live.
- `--interval` and `--once` are accepted but have no effect.
- `monitor-many` takes `-j` plus several directories and gives the same single
  update for all of them.

A job whose forces, SCF cycle, or stress did not converge still prints its last
step; convergence is reported, not enforced.

## Running ABACUS

abacustools does not launch the binary itself for a plain job: you run
`abacus` (or `mpirun -np N abacus`) in the job directory, or submit a generated
script. Workflow prepare stages can generate those scripts when
`submission.generate: true` is set in `~/.abacustools/config.yaml` or
`--submit-script` is passed; the packaged templates cover local execution and
Slurm, PBS, and LSF. `job prepare --submit-config` writes the separate batch
file (`submission.batch.filename`, `job.json` by default) that a runner such as
`abacustest` or Bohrium reads, with the generated directory names filled in.
See [references/config-and-submission.md](references/config-and-submission.md)
for the template syntax, the launcher block, the batch placeholders, and the
ABACUS version profiles that decide how logs are read.

## Reference

- [references/config-and-submission.md](references/config-and-submission.md) -
  `~/.abacustools/config.yaml`: `abacus.version` and custom version profiles,
  `submission` templates/placeholders and the `submission.batch` block, the
  `bader`/`chargemol` executables, and where the file is generated from.
- [references/basis-libraries.md](references/basis-libraries.md) - the
  `resources:` block: library paths, orbital variants, `element.json` and
  `ecutwfc.json` indexes, and how a job is made self-contained (also used by
  `job setpporb`).
