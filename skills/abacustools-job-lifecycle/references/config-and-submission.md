# Configuration, submission, and version profiles

`~/.abacustools/config.yaml` is generated from the packaged
`core/default_config.yaml` the first time the package is imported. It is
deep-merged over the packaged defaults, so a user file only needs the values it
changes. Keys used across the toolkit:

- `constants` - unit conversions used by the analysis code.
- `abacus` - `version` and the log-marker profiles (below).
- `resources` - pseudopotential/orbital libraries (see
  [basis-libraries.md](basis-libraries.md)).
- `input_templates` - the `INPUT` bodies `job prepare` writes, per job type
  (`scf`, `relax`, `cell-relax`, `md`), plus `basis_settings` per basis.
- `submission` - submission-script generation and the batch file (below).
- `bader.exe` - Henkelman Bader binary, also `BADER_EXE` or `--bader-exe`.
- `chargemol.exe`, `chargemol.atomic_densities` - DDEC binaries and tables,
  also `CHARGEMOL_EXE` / `CHARGEMOL_ATOMIC_DENSITIES`.

## ABACUS version profiles

Results are read through a version profile of log markers, so the same command
reads both the LTS branch and develop output. The profile is detected from the
`ABACUS v...` banner of `OUT.*/running_*.log`; when the log declares a version
that differs from `-v/--version`, the log wins and a warning is printed.

```yaml
abacus:
  version: auto           # or "3.10.1LTS" / "develop"
  versions:
    develop:
      density_error_keywords: ["electron density deviation"]
    my-branch:
      aliases: ["mybranch"]
      version_prefixes: ["6."]
      scf_converged_keywords: ["#SCF DONE#"]
```

Profile fields: `aliases`, `version_prefixes`, `energy_keywords`,
`final_energy_keywords`, `density_error_keywords`, `scf_converged_keywords`,
`fermi_keywords`, `normal_end_keywords`, `vdw_keywords`, `total_mag_keywords`,
`absolute_mag_keywords`, `orbital_mag_header_keywords`, `force_header_keywords`,
`stress_header_keywords`, `scf_step_patterns`, `ion_step_patterns`,
`md_step_patterns`, `relax_step_patterns`, `relax_energy_patterns`,
`relax_force_patterns`, `relax_stress_patterns`,
`relax_force_threshold_patterns`, `relax_stress_threshold_patterns`,
`relax_converged_keywords`. Keywords are matched case-insensitively;
regular-expression fields must capture the value in their first group.

## Submission scripts

Disabled by default. Enable globally or per preparation:

```yaml
submission:
  generate: true
  default: slurm
  abacus_command: "mpirun -np 8 abacus"
```

```text
abacustools workflow phonon prepare -j JOB --submit-script --submission-type slurm
abacustools workflow phonon prepare -j JOB --no-submit-script
```

The packaged templates are `local` (`run.sh`), `slurm` (`submit.slurm`), `pbs`
(`submit.pbs`), and `lsf` (`submit.lsf`). Placeholders available in a template:
`{abacus_command}`, `{job_name}`, `{task_name}`, `{workflow}`; literal shell
braces must be doubled. Each template also carries a `launcher` block:

```yaml
launcher:
  mode: scheduler
  submit_command: "sbatch --parsable"
  dependency_option: "--dependency=afterok:{dependency_id}"
  id_filter: "cat"
```

`mode: local` runs in place. A scheduler launcher records the submit command,
how to express a dependency on an earlier task, and how to extract the job id
from the command's output. `workflow_filename` names the top-level script
(`submit_<workflow>.sh` by default); the workflow schedules its tasks in
dependency order, for example the vibration workflow runs the equilibrium task
first and submits the displacements after it.

## Batch runner file

`abacustools job prepare --submit-config` writes one configuration file next to
the generated job directories for a batch runner.
The file is named by `submission.batch.filename` (`job.json` by default) and is
either rendered from the inline `submission.batch.template` or copied and
rendered from `submission.batch.template_file`.

```yaml
submission:
  batch:
    generate: false          # --submit-config turns it on for one run
    filename: "job.json"
    template_file: /path/to/my/batch.json
```

Placeholders: `{examples}` (the directory names as a JSON array), `{count}`,
`{job_type}` and `{abacus_command}`. Braces that are not one of them, such as
the ones of the JSON object itself, are left alone, so a JSON template needs no
escaping; an unknown placeholder is an error rather than a silently broken
file. `--no-submit-config` overrides a configuration that enables the file, and
`job prepare --abacus-command 'mpirun -np 32 abacus'` sets the command the
batch file records. The packaged template targets one cloud runner;
review its image, machine type, account and command before submitting, or point
`template_file` at your own job file.
