# Input files of an ABACUS job

ABACUS is started in a working directory and resolves its input files from
there. The `INPUT` file itself has a fixed name; every other input file is
named by a keyword inside it, so the names below are defaults, not
requirements. The block syntax of `INPUT`, `STRU` and `KPT` is documented in
the `abacus-inputs` skill; this page only says which files a job needs and
which keyword selects each one.

## Required for every run

### INPUT

Fixed name, read from the working directory. One `name value` pair per line
under a leading `INPUT_PARAMETERS` line; `#` starts a comment. It carries both
the physics/numerics settings and the names of the other input files:

| keyword | default | selects |
| --- | --- | --- |
| `stru_file` | `STRU` | the structure file |
| `kpoint_file` | `KPT` | the k-point file |
| `pseudo_dir` | (working directory) | directory searched for pseudopotentials |
| `orbital_dir` | (working directory) | directory searched for numerical orbitals |
| `paw_dir` | (working directory) | directory searched for PAW data |
| `read_file_dir` | `OUT.$suffix` | directory holding a restart density / wavefunction |
| `suffix` | `ABACUS` | names the output directory `OUT.<suffix>` |

### Structure file (`stru_file`, default `STRU`)

The structure is required and holds, in this order:

- `ATOMIC_SPECIES`: one line per species as `label mass pseudopotential_file`
  (develop also accepts a `pp_type` column). The pseudopotential name on this
  line is the link to the resource files.
- `LATTICE_CONSTANT`: in Bohr, not Angstrom; it scales `LATTICE_VECTORS`.
- `LATTICE_VECTORS`: three rows of cell vectors in units of the constant.
- `ATOMIC_POSITIONS`: the coordinate mode (`Direct`, `Cartesian`,
  `Cartesian_angstrom`, `Cartesian_au`, `Cartesian_angstrom_center_*`), then
  one per-species block with mass, magnetic moments and the atom lines.
- `NUMERICAL_ORBITAL`: optional; required for `basis_type lcao`, one orbital
  file per species.
- `PAW_FILES`: optional; PAW datasets, one per species.
- Per-atom magnetic moments, movement constraints, velocities and forces are
  part of the position blocks, not separate files.

The unit convention is the trap: `LATTICE_CONSTANT` is Bohr while
`Cartesian_angstrom` positions are Angstrom, and `Cartesian` positions are in
units of the lattice constant.

### Pseudopotential files

One file per element, referenced by name from `ATOMIC_SPECIES`, resolved from
the working directory or below `pseudo_dir`. They are normally UPF files
(`.upf`), with the element and functional encoded in the file. They are not
optional: a run without a resolvable pseudopotential for every element stops
during setup.

## Conditionally required

### KPT (`kpoint_file`, default `KPT`)

Not needed when `kspacing` (1/Bohr) fixes the mesh, or in an LCAO run with
`gamma_only 1`. Otherwise it is mandatory. The first line selects the model: `Gamma`,
`MP`, `Direct`, `Cartesian`, `Line` or `Line_Cartesian`. Gamma and MP take a
mesh; Direct/Cartesian take one line per k-point; Line and Line_Cartesian take
one line per high-symmetry node and are what a band calculation uses.

### Numerical orbitals (LCAO)

For `basis_type lcao`, `NUMERICAL_ORBITAL` must list one orbital file per
species, resolved from the working directory or below `orbital_dir`. A
plane-wave job never needs them, even if the source `STRU` mentions orbitals.
The orbital file name encodes the angular momentum content and the recommended
cutoff (for example `..._100Ry_2s2p1d.orb`).

### PAW data

Only for PAW calculations; the `PAW_FILES` block lists the datasets, resolved
below `paw_dir`.

## Continuation and restart

A restarted run reuses the density or wavefunction of a previous one:

- `read_file_dir` (default `OUT.$suffix`) points at the directory that holds
  the starting files, such as `SPIN1_CHG.cube` or its restart form.
- `init_chg` selects the starting density; its `file` option reads it from
  `read_file_dir`.
- `init_wfc` selects the starting wavefunction; its `file` option is mainly for
  LCAO runs that continue a `set_wf`/`get_pchg` calculation.
- `md_restart` continues a molecular-dynamics run; while it is `true`,
  `stru_file` is ignored and the structure is taken from the restart data.

Because these files live in `OUT.<suffix>`, a continuation job needs `INPUT`,
`STRU` (unless `md_restart`), the pseudopotentials (and orbitals for LCAO),
and the previous output directory. It does not need a fresh `KPT` when the
previous run's mesh is reused through `read_file_dir`.
