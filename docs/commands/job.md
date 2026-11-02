# Job commands

[Documentation home](../index.md)

The `job` family prepares ABACUS input directories and inspects their inputs, state, and progress.

## job prepare

Complete ABACUS input directories can be prepared with a resource library
selected from `~/.abacustools/config.yaml`:

```text
abacustools job prepare -f STRUCTURE --library apns
```

The configuration contains the `resources.default` library and paths for each
library's `pp` and `orb` directories. The command uses the default library
when `--library` is omitted. For example:

```yaml
resources:
  default: apns
  libraries:
    apns:
      pp: /path/to/apns-pseudopotentials
      orb: /path/to/apns-orbitals
    dojo-nc-sr:
      pp: /path/to/Dojo-NC-SR/Pseudopotential
      orb: /path/to/Dojo-NC-SR/Orbitals
```

The K-point mesh written by `job prepare` is controlled by `--kpt` and
`--kpt-model`, which support the two mesh models gamma and MP with three or six
values:

```text
abacustools job prepare -f STRUCTURE --kpt 9 9 9
abacustools job prepare -f STRUCTURE --kpt 9 9 9 0 0 0 --kpt-model mp
```

Band paths and explicit k-point lists are prepared with `abacustools file kpt`
or by placing a KPT file next to the structure.

`--job-type` defaults to `scf`. The basis is selected with `--basis pw|lcao`
and defaults to `lcao` (the configured `abacus.default_basis`); pass
`--basis pw` for a plane-wave job.

Magnetic and DFT+U settings are prepared with `--nspin`, `--soc`,
`--init-mag`, `--afm` and `--dftu-param`. `--dftu-param ELEMENT [ORBITAL] U`
enables DFT+U for an element: set U in eV, and optionally the correlated
orbital (`p`, `d` or `f`) before it; the orbital is inferred from the element
type when omitted. It repeats for several elements. Initial magnetic moments
require a spin-polarized run (`--nspin 2` or `--nspin 4`), and `--soc`
requires `--nspin 4`; a conflicting choice is an error rather than a silent
override. `--kpt-model` applies only together with `--kpt` and warns when it
is given alone.

Generated jobs are self-contained: the referenced pseudopotentials and orbitals
are symlinked into the job directory (copied with `--copy-resources`), and the
written `STRU` refers to them by file name. PAW files are not supported when
preparing directories and a source `STRU` containing a `PAW_FILES` block is
rejected instead of producing a job with missing files. A
plane-wave job (`--basis pw`) never ships or references numerical orbitals,
even when the source `STRU` contains a `NUMERICAL_ORBITAL` block; LCAO jobs
require an orbital for every element. The packaged templates set
`kspacing 0.14`, so a prepared job gets a real k-point mesh instead of a
single Gamma point: no `KPT` file is written and ABACUS builds the mesh from
`kspacing`. An explicit `--kpt` overrides the template `kspacing`, and a `KPT`
file next to the structure is used when `kspacing` was not set explicitly.
When the k sampling is disabled (`kspacing 0`) and no KPT file is available, a
1x1x1 Gamma mesh is written and a warning is issued. A structure with a vacuum
layer (slab, wire or molecule) triggers a warning that suggests the
three-value `kspacing` form with a large value along the vacuum.

The basis defaults of `basis_settings` (solver, diagonalization settings) are
applied for the basis the job ends up using, so `--set basis_type pw` also
selects the plane-wave solver. The basis may be given only once: `--basis` and
`--set basis_type`, or `--basis` and a template with another `basis_type`, are
rejected instead of producing a mixed INPUT. A `ks_solver` that does not belong
to the selected basis (for example `genelpa` with `--basis pw`) is rejected
before any directory is created.

The names given to `--set` are checked against the ABACUS parameter list shipped
in `input-params.json`, so a mistyped parameter is reported with a suggestion
before any directory is created. A parameter of a newer ABACUS than the shipped
list can still be passed through an INPUT template (`--input`).

Generated folder names default to a zero-padded index. `--folder-syntax` builds
them from an f-string over `{x}` (the source file name) and `{i}` (the index),
such as `{x[:-5]}` or `{i:03d}`; any other expression, conversion or path that
escapes the output directory is rejected.

A whole batch can be handed to a runner such as `abacustest` with one
configuration file next to the generated directories:

```text
abacustools job prepare -f 'structures/*.cif' -o runs/ --submit-config
abacustools job prepare -f 'structures/*.cif' -o runs/ --submit-config \
    --abacus-command 'mpirun -np 32 abacus'
```

`--submit-config` writes `submission.batch.filename` (`job.json`) into the
output directory, with the generated directory names listed in the `{examples}`
placeholder, so the `run_dft` entry of an abacustest or Bohrium job points at
exactly the directories that were prepared. The template is the inline
`submission.batch.template`, or the file named by
`submission.batch.template_file`:

```yaml
submission:
  batch:
    generate: false          # --submit-config turns it on for one run
    filename: "job.json"
    template_file: /path/to/my/abacustest.json
```

The placeholders are `{examples}` (the directory names as a JSON array),
`{count}`, `{job_type}` and `{abacus_command}`. Braces that are not one of them,
such as the ones of the JSON itself, are left alone, so a JSON template needs no
escaping and an unknown placeholder is an error rather than a silently broken
file. `--submit-config` forces the file on for one run; when it is absent the
configured `submission.batch.generate` decides, and there is no command-line
switch to turn an enabled default off. The
packaged template is an abacustest job for Bohrium; review its image, machine
type, account and command before submitting.

Pseudopotential and orbital paths are configured through libraries rather than
through command-line paths: `--library NAME` selects one entry of
`resources.libraries`, and `~/.abacustools/config.yaml` can hold any number of
them, so adding an entry is how a different set of files is used.
The environment variables `ABACUS_PP_PATH` and `ABACUS_ORB_PATH` are still
honoured as fallbacks when neither an explicit path nor the selected library
defines one.

Alongside the upstream libraries the configuration ships a `custom` entry for
pseudopotentials and orbitals from any other source:

```yaml
resources:
  libraries:
    custom:
      pp: /path/to/your/pseudopotentials
      orb: /path/to/your/orbitals
```

It assumes the APNS layout: one file per element whose name starts with the
element symbol, such as `Si.upf` and `Si_gga_7au_100Ry_2s2p1d.orb`, searched
recursively and without variant directories. The generated configuration points
it at `~/.abacustools/pseudopotentials` and `~/.abacustools/orbitals`, and the
packaged default leaves both empty.

A library directory is searched for files whose names start with the element
symbol, preferring the `.upf`/`.orb` suffix matching the resource type;
subdirectories are searched too. Two optional index files make the mapping
explicit:

Libraries that store one directory per element and orbital variant, such as
SG15 and Dojo (`Si_SZ/`, `Si_DZP/`, `Si_TZDP/`), are resolved to the variant
selected by `resources.orb_variant`, which defaults to `DZP`. A library can
override it, and `--variant` overrides both for a single run:

```yaml
resources:
  orb_variant: "DZP"
  libraries:
    sg15:
      pp: /path/to/SG15_ONCV_v1.0_upf
      orb: /path/to/SG15_v1.0/Orbitals
      orb_variant: "TZDP"
```

```text
abacustools job prepare -f STRUCTURE --library sg15 --variant SZ
```

When an element has variant directories but not the requested one, the
preparation stops and lists the variants that are available instead of
silently using a different basis. Libraries without variant directories are
unaffected, and an `element.json` mapping takes precedence over variant
selection.

Some libraries ship one directory per orbital set instead of variant
subdirectories; APNS provides an `efficiency` and a `precision` set. Such
directories are mapped with `orb_variants`, and the same `--variant` and
`orb_variant` settings select them:

```yaml
resources:
  libraries:
    apns:
      orb: /path/to/apns-orbitals-efficiency-v1
      orb_variants:
        efficiency: /path/to/apns-orbitals-efficiency-v1
        precision: /path/to/apns-orbitals-precision-v1
      pp: /path/to/apns-pseudopotentials-v1
```

```text
abacustools job prepare -f STRUCTURE --library apns                    # efficiency
abacustools job prepare -f STRUCTURE --library apns --variant precision
```

A variant name that is not mapped keeps the configured `orb` directory, so the
SG15-style default (`DZP`) leaves an APNS library on its configured set. Set
`orb_variant` inside a library entry to change which set that library uses by
default. Requesting an unavailable variant with `--variant` reports which
library could not provide it and which set is used instead, while a variant
inherited from the configuration falls back silently.

Spin-orbit and noncollinear calculations (`--nspin 4`, or `--soc`) need
pseudopotentials that explicitly support them, which UPF files declare as
`relativistic="full"` or through tabulated spin-orbit projectors (`has_so`).
The job is still prepared when a selected pseudopotential only declares
scalar-relativistic support, but the warning names the files, so a setup such
as the SG15 set (scalar only) cannot be used for spin-orbit work unnoticed.
Selected pseudopotentials that cannot be read as UPF are reported as unreadable
rather than assumed usable.

Structures that name several species of one element, such as `Si1` and `Si2`,
receive one `orbital_corr`/`hubbard_u` entry per species, looked up by species
label first and by element second. Repeating the same structure file through
overlapping `-f` patterns produces one job instead of duplicates.

Upstream libraries also publish the recommended cutoff radius of every element
as `<orbital directory>_<VARIANT>_..._StandardRcut.json` next to the orbital
directory, with an `Others` fallback. When such an index is present, the
matching `<radius>au` orbital is selected; otherwise the first candidate of the
chosen variant is used, as before. A library without an index whose elements
offer several radii, such as a locally collected `custom` set, reports the radii
and the file that was picked, and points at the index format for choosing them
explicitly.

Because upstream sometimes renames these directories (`Orbitals` became
`Orbitals_v2.0`), a configured path that no longer exists is resolved to the
uniquely matching sibling directory of the same resource type, with a warning.
If several candidates match, the preparation stops and lists them.

```yaml
# element.json: element -> file name, resolved below the library directory
{"Si": "Si_ONCV_PBE-1.0.upf", "O": "O_ONCV_PBE-1.0.upf"}
```

Every entry of `element.json` must point to an existing file; a broken entry
stops the preparation with an explicit error instead of silently falling back
to the file-name search, and so does an element that the index does not list
even though the structure needs it.

```yaml
# ecutwfc.json: element -> recommended plane-wave cutoff in Ry
{"Si": 60, "O": 80}
```

`ecutwfc.json` is only used for plane-wave jobs that do not set `ecutwfc`
themselves; the largest recommended cutoff of the structure's elements is
applied.

LCAO jobs take `ecutwfc` from the selected numerical orbitals instead: the
largest cutoff encoded in their file names (such as the `150Ry` of
`Cu_gga_9au_150Ry_4s2p2d1f.orb`) is applied, so a structure mixing a 150 Ry
orbital with a 100 Ry one gets 150 Ry. An explicitly requested `ecutwfc` is
kept, but a value below the orbital cutoff is reported as a warning.

## job setpporb

An existing job can be moved to another library, or to another orbital variant
of the same library, without preparing it again. `job setpporb`
re-resolves every element of the job's `STRU` from the selected library,
rewrites the `ATOMIC_SPECIES` pseudopotential name and the `NUMERICAL_ORBITAL`
entry of each species, installs the new files, and drops the resource files the
`STRU` no longer references:

```text
abacustools job setpporb JOB --library sg15
abacustools job setpporb JOB1 JOB2 --library apns --variant precision
abacustools job setpporb JOB --library sg15 --variant SZ --copy-resources
abacustools job setpporb JOB --library sg15 --dry-run
```

`--library` and `--variant` read the same `resources` configuration as
`job prepare`, and `--dry-run` reports the resolved names and the files it would
install or remove without writing anything. New files are copied when the job
already held copies and symlinked when it held symlinks; `--copy-resources` and
`--symlink` force either. Only files whose name the old `STRU` referenced are
removed, and when the new numerical orbitals carry a higher plane-wave cutoff
than the job's `ecutwfc`, a warning reports the shortfall.

## job checkinput

Input files can be checked independently of calculation output:

```text
abacustools job checkinput JOB
abacustools job checkinput JOB --strict
```

The command checks `INPUT`, `STRU`, `KPT`, and referenced pseudopotential,
orbital, and PAW files. It reports the calculation settings, structure size
and composition, cell parameters, k-point mode, and resource files. It does
not read `OUT.*` directories or calculation logs. Use `--json` for a
machine-readable report.

## job validate

`job validate JOB` validates the input side of one job directory: `INPUT`,
`STRU`, `KPT`, and every referenced pseudopotential, orbital, and PAW file.
Unknown INPUT keywords are warnings by default and errors with `--strict`.
The command exits nonzero when the job is invalid and supports `--json`.

```text
abacustools job validate JOB
abacustools job validate JOB --strict --json
```

## job status

`job status JOB` combines validation with the current calculation state and
returns one of the states reported by the job reader, such as `ready`,
`incomplete`, `running`, `completed`, `invalid`, or `failed`. The text output
includes available progress fields and the log that supplied them; `--json`
returns the same state, progress, and log path as a machine-readable object.

```text
abacustools job status JOB
abacustools job status JOB --json
```

## CLI help

Every level of the command tree exposes argparse help:

```text
abacustools --help
abacustools job --help
abacustools job prepare --help
```

## job monitor

Running jobs can be inspected step by step. The monitor reads the
`calculation` of `INPUT`, prints one update of what the running log holds for
that task type and exits. It never waits for the job to finish, so a slow DFT
run can be checked at any time without blocking the shell:

```text
abacustools job monitor JOB
abacustools job monitor JOB --json
abacustools job monitor JOB --csv steps.csv
abacustools job monitor JOB --plot
abacustools job monitor JOB --plot steps.png
```

An SCF calculation shows the current electronic progress, and
`--scf-steps` adds the energy, energy change and density error of every SCF
iteration. A `relax` calculation prints its convergence criteria first and
then every ionic step with the step number, total energy, energy change, the
largest force, and the atom it belongs to, written as its log label plus the
Cartesian component such as `H1x` for the `x` component of atom `H1`. Each
relaxation step also reports the atomic RMS displacement, the maximum atomic
displacement from the preceding structure, and the atom that moved the most,
in Angstrom; the displacements are taken as periodic minimum images, which
needs the cell of the job, so they are left empty when the `STRU` of the job
cannot be read. A
`cell-relax` calculation adds the largest stress with its Voigt component. An
`md` calculation prints the total, potential and kinetic energy, the
temperature and the pressure of every MD step, together with the settings of
its `md_type`: the target temperature of a thermostat (`nvt`, `npt`,
`langevin`) and the target pressure of a barostat (`npt`, `msst`). The MD
energies are converted to eV when a branch prints them in Rydberg, and the
pressure is left empty when the branch does not print it.

`--plot` draws the convergence history as stacked panels that share the step
axis. A relaxation shows the total energy, the largest force on a logarithmic
axis with its threshold, and, for a `cell-relax` job, the largest stress the
same way. The force and stress panels carry the number of components beyond
the threshold on a second axis, where atoms fixed by `STRU` are left out. An
SCF calculation shows its total energy and the absolute energy change together
with the density error on logarithmic axes, together with the `scf_thr` line
that `scf_thr_type` assigns to the energy or to the density. An MD run keeps
its energy, temperature and pressure panels.
Without a file name the plot is named after the task type below JOB, that is
`monitor_scf.png`, `monitor_relax.png`, `monitor_cell-relax.png` or
`monitor_md.png`.
`--csv` writes the same history as a table. `--csv` and `--plot` are additions
to the text report: the screen output stays the same and the path of every
written file is printed below it. With `--json` only the JSON is printed, so
the report stays machine-readable. A step table prints the last 30 steps
because an optimization or a long MD run accumulates far more than a terminal
shows; `--tail N` changes that limit and `--tail 0` prints every step. The
`--json`, `--csv` and `--plot` output always keeps the full history. The
history keeps the current incomplete step while the calculation is still
running, so an unfinished job still shows its latest step. Step tables align
their columns on the widest cell, energies carry eight decimals and energy
changes use scientific notation so that small steps stay readable. Energies
are in eV, forces in eV/Angstrom, stresses and pressures in the native ABACUS
kBar unit, and temperatures in K.

## job monitor-many

Several jobs can be inspected together in the same single update.
Geometry-optimization jobs include their current step and latest force/stress
and structure-displacement metrics, while SCF jobs show their current
electronic progress:

```text
abacustools job monitor-many -j JOB1 JOB2 JOB3
abacustools job monitor-many -j JOB1 JOB2 --json
```

`--once` and `--interval` are accepted for compatibility but have no effect:
the monitor always prints one update and returns.
