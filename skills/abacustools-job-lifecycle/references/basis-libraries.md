# Basis libraries

How `abacustools` resolves pseudopotentials and numerical orbitals for
`job prepare`. Which files to choose - family, functional, relativity, orbital
zeta - is an ABACUS modelling decision and lives in the `abacus-basis` skill;
this page only covers the wiring.

`abacustools job prepare` never takes a pseudopotential path on the command
line. It resolves files from a *library* entry of
`~/.abacustools/config.yaml`, which is generated on first import from the
packaged `core/default_config.yaml`.

```yaml
resources:
  default: apns
  orb_variant: "DZP"
  libraries:
    apns:
      pp: /path/to/apns-pseudopotentials
      orb: /path/to/apns-orbitals
    sg15:
      pp: /path/to/SG15_ONCV_v1.0_upf
      orb: /path/to/SG15_v1.0/Orbitals
      orb_variant: "TZDP"
    custom:
      pp: ~/.abacustools/pseudopotentials
      orb: ~/.abacustools/orbitals
```

## How a file is chosen

1. `--library NAME` selects the entry; the default comes from
   `resources.default`. `--variant` overrides `resources.orb_variant`.
2. The library directory is searched recursively for a file whose name starts
   with the element symbol, preferring the `.upf`/`.orb` suffix that matches the
   resource type.
3. `element.json` inside the library directory pins the mapping explicitly:
   `{"Si": "Si_ONCV_PBE-1.0.upf", "O": "O_ONCV_PBE-1.0.upf"}`. A broken entry
   stops the preparation instead of falling back to the name search.
4. `ecutwfc.json` gives the recommended PW cutoffs, `{"Si": 60, "O": 80}`; the
   largest value of the structure's elements is applied to a PW job that does
   not set `ecutwfc` itself.

## Orbital variants

Two layouts exist and both are supported:

- Variant subdirectories, as in SG15/Dojo (`Si_SZ/`, `Si_DZP/`, `Si_TZDP/`):
  resolved through `orb_variant`. When an element has variant directories but
  not the requested one, preparation stops and lists the available variants
  instead of silently picking another basis.
- One directory per orbital set (APNS efficiency/precision):

  ```yaml
  apns:
    pp: /path/to/apns-pseudopotentials-v1
    orb: /path/to/apns-orbitals-efficiency-v1
    orb_variants:
      efficiency: /path/to/apns-orbitals-efficiency-v1
      precision: /path/to/apns-orbitals-precision-v1
  ```

  `--variant precision` selects the mapped directory; an unmapped name keeps the
  configured `orb`.

Upstream also publishes a recommended cutoff radius next to the orbital
directory as `<orbital directory>_<VARIANT>_..._StandardRcut.json` with an
`Others` fallback. When present, the matching `<radius>au` orbital is chosen;
otherwise the first candidate of the variant is used and the alternatives are
reported. A library that offers several radii without an index (a locally
collected `custom` set) reports which file it picked and how to pin the choice.

## Fallbacks and failures

- `ABACUS_PP_PATH` and `ABACUS_ORB_PATH` are honoured when neither an explicit
  path nor the selected library defines one.
- A configured path that no longer exists is resolved to the uniquely matching
  sibling directory of the same resource type (upstream renames `Orbitals` to
  `Orbitals_v2.0`), with a warning; several candidates stop the preparation.
- Generated jobs are self-contained: resources are symlinked into the job
  directory, or copied with `--copy-resources`/`--copy-pp-orb`.
- PW jobs never ship or reference numerical orbitals, even when the source
  `STRU` has a `NUMERICAL_ORBITAL` block; LCAO jobs require an orbital for every
  element.
- SOC/noncollinear jobs need UPF files with `relativistic="full"` or `has_so`;
  a scalar-relativistic selection is still prepared but warned about by file
  name.
- PAW files are not supported by `job prepare`; a source `STRU` with a
  `PAW_FILES` block is rejected rather than producing a job with missing files.

## Switching an existing job

`job setpporb` re-resolves the resources of a job that already exists, using the
same library/variant lookup as `job prepare`:

```text
abacustools job setpporb JOB --library sg15 --variant SZ
abacustools job setpporb JOB1 JOB2 --library apns
abacustools job setpporb JOB --library sg15 --copy-resources
abacustools job setpporb JOB --library sg15 --dry-run
```

It rewrites the `ATOMIC_SPECIES` pseudopotential name and the
`NUMERICAL_ORBITAL` entry of every species, installs the new files and removes
the old ones the `STRU` no longer references (only those). New files keep the
job's existing convention by default - copied when the job held copies,
symlinked when it held symlinks - and `--copy-resources`/`--symlink` force one.
`--dry-run` reports what would change without writing. A new orbital that
carries a higher plane-wave cutoff than the job's `ecutwfc` triggers a warning.
