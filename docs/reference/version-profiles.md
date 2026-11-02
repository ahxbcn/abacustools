# Version profiles

[Documentation home](../index.md)

## Detection and extension

Results are read with a *version profile* describing the markers each ABACUS
branch writes into its running log. The profile of a job is detected from the
`ABACUS v...` banner of `OUT.*/running_*.log`, so the same command reads LTS
3.10 and develop (3.11) output:

```text
abacustools postprocess result -j JOB -p energy drho efermi converged
abacustools postprocess result -j JOB -v develop -p energy efermi
abacustools job monitor JOB
```

`-v/--version` selects a dialect explicitly. When the log declares a different
version, the version found in the output wins and a warning is printed. The
default comes from `abacus.version` in `~/.abacustools/config.yaml`, whose
packaged value is `auto`. Profiles can be extended from that file without
changing the code:

```yaml
abacus:
  version: develop
  versions:
    develop:
      density_error_keywords: ["electron density deviation"]
    my-branch:
      aliases: ["mybranch"]
      version_prefixes: ["6."]
      scf_converged_keywords: ["#SCF DONE#"]
```

The marker fields of a profile are `aliases`, `version_prefixes`,
`energy_keywords`, `final_energy_keywords`, `density_error_keywords`,
`scf_converged_keywords`, `fermi_keywords`, `normal_end_keywords`,
`vdw_keywords`, `total_mag_keywords`, `absolute_mag_keywords`,
`orbital_mag_header_keywords`, `force_header_keywords`, `stress_header_keywords`,
`scf_step_patterns`, `ion_step_patterns`, `md_step_patterns`, `relax_step_patterns`,
`relax_energy_patterns`, `relax_force_patterns`, `relax_stress_patterns`,
`relax_force_threshold_patterns`, `relax_stress_threshold_patterns`, and
`relax_converged_keywords`. Keywords are matched case-insensitively; the
regular-expression fields must capture the value as their first group.
