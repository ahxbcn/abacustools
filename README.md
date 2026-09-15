Collection of tools used for performing DFT calculation with ABACUS.

## Command-line subcommands

Subcommands are registered in the package with Python's
`argparse.add_subparsers()`. The top-level command provides nested `file`
subcommands:

```text
abacustools file input INPUT
abacustools file stru STRU
abacustools file kpt KPT
abacustools file stru INPUT OUTPUT
```

The `postprocess` command family also has the shorter aliases `post` and `pp`:

```text
abacustools postprocess bader -j JOB
abacustools post bader -j JOB
abacustools pp bader -j JOB
```

The top-level families are `file`, `job`, `postprocess`, `workflow`, and `mp`.
The `mp` family searches and downloads Materials Project entries as described
at the end of this document.

## Interactive menu

Running `abacustools` with no arguments in a terminal opens an interactive,
Multiwfn/VASPKIT-style menu (an ASCII-art banner with the name and version is
printed first). The same menu is available explicitly:

```text
abacustools menu
```

The menu mirrors the command tree: navigate with an entry number (or name /
alias), press `b` to go back and `q` to quit. Selecting a command prompts for
its arguments, prints the equivalent non-interactive command, runs it, and
waits for ENTER before returning. When input is not a terminal (for example in
a pipeline or CI), `abacustools` prints the help instead.

`file stru` converts between ABACUS `STRU` and common structure formats
supported by ASE, including POSCAR/VASP, CIF, XYZ, EXTXYZ, and XSF. The input
and output formats are inferred from filenames, or can be set explicitly with
`--input-format` and `--output-format`:

```text
abacustools file stru STRU POSCAR
abacustools file stru POSCAR STRU
abacustools file stru structure.xyz STRU --cell 10 0 0 0 10 0 0 0 10
```

Basic structure information can be inspected without converting the file:

```text
abacustools file info STRU
abacustools file info POSCAR --json
abacustools file info structure.xyz --cell 10 0 0 0 10 0 0 0 10
abacustools file info slab.STRU --coordination
abacustools file info STRU --coordination voronoi --json
```

The report includes cell parameters, volume, density, element and label counts,
the formula unit with the number of formula units per cell and the prototype
formula, space group, crystal system, point group with its Schoenflies symbol,
Bravais lattice with the Pearson symbol, inversion symmetry, polar point group,
the symmetry tolerances that were used, symmetry operation count, per-atom
Wyckoff positions with their multiplicity and site symmetry, a separate list of
symmetry-inequivalent atomic positions, and the ABACUS pseudopotential and
orbital file of every label. The inequivalent list contains one representative
atom per symmetry-equivalent group together with the equivalent atom indices
and multiplicity; when no two atoms are related by symmetry, as in `P1`, the
list only repeats the per-atom table and is left out of the report.
The per-atom table holds the index, label, element, fractional and Cartesian
coordinates, the Wyckoff position with its multiplicity and the site symmetry
symbol, followed by the magnetic moments (with their polar angles when the
structure sets them), the movement constraints and the velocities; those last
columns appear only when the structure actually defines them, so an
unconstrained structure keeps a compact table. A Wyckoff symbol counts the
atoms of its orbit in the conventional cell, as the International Tables write
it, so it does not change when the same crystal is handed in as a supercell;
the equivalent atom column counts the atoms of the orbit in the given cell.
Three more items are reported from the same structure when they are defined:
the magnetic space group with its BNS and UNI numbers and its type (a structure
without magnetic moments keeps the grey group), the magnetic ordering (FM, AFM
or FiM with the net moment and the number of magnetic sites) and the layer group
of a slab. The layer group needs the non-periodic direction, which is detected
from the vacuum of the cell and can be set explicitly with `--layer-direction`;
slabs show it next to the space group, bulk structures stay without it.
Use `--symprec` and `--angle-tolerance` when the input coordinates require
different symmetry tolerances. Structures without a three-dimensional cell
are still summarized, but symmetry and Wyckoff positions are reported as
unavailable.

The same vacuum that carries the layer group also gives the dimensionality:
an empty span of at least `--min-vacuum` Angstrom, 5 by default, makes a
lattice direction non-periodic, so the report names the structure a 3D bulk,
a 2D slab, a 1D wire or a 0D molecule or cluster. `--coordination` adds the
coordination number of every atom to the per-atom table and, for a bulk, the
coordination geometry symbols of ChemEnv such as `T:4` or `O:6`, together with
their continuous symmetry measure. The analysis follows the dimensionality by
default: ChemEnv describes bulk crystals, but its Voronoi tessellation is
undefined in the vacuum of a slab, so slabs, wires and molecules are analysed
with the CrystalNN nearest neighbours instead, and a note explains the choice.
`--coordination METHOD` forces `crystalnn`, `chemenv`, `voronoi` or
`minimum-distance`; an explicit method that cannot handle the structure is
reported as unavailable instead of being replaced silently. The work-function
workflow uses the same vacuum analysis to find the direction of the
electrostatic vacuum.

Structures can also be edited into a new file. Every action reads a structure,
writes a separate OUTPUT (existing files are only replaced with `--override`)
and keeps the atom attributes it does not touch, such as pseudopotential and
orbital file names, movement constraints and magnetic moments:

```text
abacustools file editstru supercell STRU -o SUPER -n 2 2 1
abacustools file editstru vacuum    STRU -o SLAB  -t 15 --direction c --center
abacustools file editstru slab      STRU -o SLAB  --miller 1 0 0 --layers 3 --vacuum 15
abacustools file editstru slab      STRU -o SLAB  --miller 1 1 0 --layers 4 --fix
abacustools file editstru select    STRU -o SUB   --elements Si O
abacustools file editstru select    STRU -o FREE  --indices 1 3 --remove
abacustools file editstru fix       STRU -o FIXED --coords 0 0.2 --direction c --direct
abacustools file editstru fix       STRU -o FIXED --elements O --move x y --free-others
abacustools file editstru direct    STRU -o DIRECT
abacustools file editstru cartesian STRU -o CART
abacustools file editstru primitive   STRU -o PRIM
abacustools file editstru conventional STRU -o CONV
abacustools file editstru standardize STRU -o STD --to-primitive
abacustools file editstru all-slabs   STRU --miller 1 1 0 --output-prefix SLAB
```

`supercell` replicates along the lattice vectors. `vacuum` extends one lattice
vector by the given thickness in Angstrom, optionally shifting the atoms so the
extra space is split between both sides. `select` keeps, or with `--remove`
drops, the atoms matching `--indices`, `--elements` or a `--coords` window.
`fix` sets the movement constraint of the selection: the atoms may only move
along the axes given to `--move`, and `--free-others` releases the rest.
Selections are additive filters (an atom must match every filter), while
`--indices` are one-based, as in the other abacustools commands. `--coords`
uses Cartesian coordinates unless `--direct` requests fractional ones, and
`--direction` accepts `a`/`b`/`c` or `x`/`y`/`z`.

`direct` and `cartesian` rewrite the same structure with an
`ATOMIC_POSITIONS Direct` or `ATOMIC_POSITIONS Cartesian` block. Atoms, cell and
every other attribute stay untouched, so converting forth and back returns the
original structure, and asking for the representation the file already uses
changes nothing. Unlike `file stru --direct`, these actions refuse to replace an
existing output unless `--override` is given.

`slab` cuts a surface with `ase.build.surface`: `--miller` selects the surface,
`--layers` counts repeating units along the surface normal, `--supercell` sets
the in-plane repetitions and `--vacuum` the empty space between the slab and its
periodic image in Angstrom. The normal is placed along `c` and can be moved to
`a` or `b` with `--vacuum-direction`. Pseudopotential, orbital and magnetic data
are restored per element afterwards, because rebuilding the in-plane supercell
through ASE drops that mapping; the zero moments and velocities ASE fills in for
structures that had none are removed again. Movement constraints are not carried
over by the cut, so constrain the slab with `--fix` or with the `fix` action.
Structures with empty atoms are rejected, because ASE cannot represent those
labels.

`--fix` constrains the slab right away: the atoms within a fraction of the slab
thickness above its lowest atom stay fixed and every other atom is released. The
window follows the vacuum direction and, unlike the plain `fix` action, is
measured from the slab itself, so the default `--fix` fixes the bottom half
regardless of where the slab sits inside the cell. Pass a value such as
`--fix 0.25` to fix only the lowest quarter.

`primitive`, `conventional` and `standardize` use spglib to change the cell
without touching the atoms. `primitive` reduces the structure to its smallest
symmetric cell, `conventional` returns the cell that follows the space-group
conventions, and `standardize` rewrites the structure in the standard setting,
optionally reducing it with `--to-primitive`. `--no-idealize` keeps small
deviations of the standardize action instead of rounding them away, and
`--symprec`/`--angle-tolerance` set the tolerances of the symmetry search.
Pseudopotential, orbital and magnetic data travel with the atoms.

`all-slabs` asks pymatgen for every symmetrically distinct termination of one
set of Miller indices, which matters for polar or mixed-terminated surfaces
where only one cut is not enough. `--min-slab-size` and `--min-vacuum-size`
size the slabs, `--in-unit-planes` switches the slab thickness from Angstrom to
repeating units, and `--symmetrize`/`--repair` clean up the cut. Every
termination is written to its own file named after `--output-prefix`, such as
`SLAB_0.STRU` and `SLAB_1.STRU`.

Conversions issue a `StructureConversionWarning` when ABACUS-specific data
such as pseudopotential/orbital filenames, spin settings, velocities,
movement constraints, or `NUMERICAL_DESCRIPTOR` cannot be represented by the
target format. Standard XYZ files do not contain a periodic cell; provide
`--cell` when converting one to `STRU`.

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

The KPT file shipped with each generated job is controlled by `--kpt` and
`--kpt-model`. The gamma and MP models take three or six mesh values, while the
direct, cartesian and line models take one group per k-point or node, so the
option is repeated for every group:

```text
abacustools job prepare -f STRUCTURE --kpt 9 9 9
abacustools job prepare -f STRUCTURE --kpt 0 0 0 --kpt 0.5 0 0 --kpt-model direct
abacustools job prepare -f STRUCTURE \
  --kpt 0 0 0 10 G --kpt 0.5 0.5 0 1 X --kpt-model line
```

Generated jobs are self-contained: the referenced pseudopotentials and orbitals
are symlinked into the job directory (copied with `--copy-resources`), and the
written `STRU` refers to them by file name. PAW files are not supported when
preparing directories and a source `STRU` containing a `PAW_FILES` block is
rejected instead of producing a job with missing files. A
plane-wave job (`--basis pw`) never ships or references numerical orbitals,
even when the source `STRU` contains a `NUMERICAL_ORBITAL` block; LCAO jobs
require an orbital for every element. When neither `--kpt` nor a KPT file,
`kspacing` or `gamma_only` is available, a 1x1x1 Gamma mesh is written and a
warning is issued.

The basis defaults of `basis_settings` (solver, diagonalization settings) are
applied for the basis the job ends up using, so `--set basis_type pw` also
selects the plane-wave solver. The basis may be given only once: `--basis` and
`--set basis_type`, or `--basis` and a template with another `basis_type`, are
rejected instead of producing a mixed INPUT.

The names given to `--set` are checked against the ABACUS parameter list shipped
in `input-params.json`, so a mistyped parameter is reported with a suggestion
before any directory is created. A parameter of a newer ABACUS than the shipped
list can still be passed through an INPUT template (`--input`).

Generated folder names default to a zero-padded index. `--folder-syntax` builds
them from an f-string over `{x}` (the source file name) and `{i}` (the index),
such as `{x[:-5]}` or `{i:03d}`; any other expression, conversion or path that
escapes the output directory is rejected.

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

Both the top-level parser and each subcommand provide their own help text:

```text
abacustools --help
abacustools file --help
abacustools file input --help
```

Calculation results can be collected from one or more completed ABACUS jobs:

```text
abacustools postprocess result -j JOB -p energy force stress
```

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
relaxation step also reports the atomic RMS displacement and maximum
atomic displacement from the preceding structure, in Angstrom; they are taken
as periodic minimum images, which needs the cell of the job, so they are left
empty when the `STRU` of the job cannot be read. A
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

Convergence tests can generate independent SCF calculations for cutoff energy:

```text
abacustools workflow ecutwfc prepare -j JOB --values 30 40 50 60
abacustools workflow ecutwfc postprocess -j JOB --energy-tol 1e-4
```

The same workflow can test k-point spacing:

```text
abacustools workflow kspacing prepare -j JOB --values 0.4 0.3 0.2 0.1
abacustools workflow kspacing postprocess -j JOB
```

The postprocessing stage reports total energy per atom, convergence deltas,
incomplete tasks, a recommended first value within the tolerance, a JSON
report, and a convergence plot.

The equation of state can be fitted from volume-scaled calculations:

```text
abacustools workflow eos prepare -j JOB --start 0.90 --end 1.10 --step 0.025
abacustools workflow eos postprocess -j JOB
```

The prepare stage generates one fixed-volume SCF (or ionic-relaxation) job per
volume scale; the postprocess stage reads their energies and fits a third-order
Birch-Murnaghan equation of state, writing the equilibrium volume, bulk
modulus, and its pressure derivative to `eos_results.json` and `eos.png`.

Vacancy formation energies can be computed from a supercell with an empty atom:

```text
abacustools workflow vacancy prepare -j JOB -s 2 2 2 -i 1 5
abacustools workflow vacancy postprocess -j JOB
```

The prepare stage builds the pristine supercell, one defective supercell per
requested atom index (the atom becomes an `empty` species), and reference
elemental crystal jobs under `ref_element/`. After the cell-relaxation jobs and
their `final_scf` follow-up have run, the postprocess stage computes
`E_f = (E_defect + mu) - E_original * N_cells` and writes
`vacancy_results.json`.

Use `abacustools postprocess result --help` to see all supported result
parameters. If `--param` is omitted, the command selects the results relevant
to the job's calculation type and displays scalar results as compact,
borderless tables. If the result columns do not fit the terminal, they are
split into multiple tables with repeated headers rather than being truncated
or wrapped by the terminal. A result that does not apply to a job is shown as
`-`. Large array results such as `force` and `stress` are omitted from this
summary. Use `--json` to print the collected results, including the array
results, as JSON.

Magnetic moments are collected with their own parameters, which are not part
of the default selection:

```text
abacustools postprocess result -j JOB -p total_mag absolute_mag
abacustools postprocess result -j JOB -p atom_mag_mulliken atom_orb_mag --json
```

`total_mag` and `absolute_mag` are the total and absolute magnetization of the
cell in Bohr magneton, taken from the last electronic iteration of the running
log; a noncollinear calculation reports the three Cartesian components of
`total_mag` as a list. `atom_mag_mulliken` lists the magnetization of every
atom of the last ionic step as written by the Mulliken analysis
(`mulliken.txt`), and `atom_orb_mag` lists the orbital-projected magnetization
of every atom from the last orbital charge analysis block of the log. Both are
`None` when the run prints no such analysis.

Band structures from ABACUS NSCF calculations can be processed and plotted
from a job directory containing `BANDS_1.dat`:

```text
abacustools postprocess band -j JOB --emin -8 --emax 8
```

The command reads the line-mode `KPT`, shifts the bands by the Fermi energy
from the NSCF log, and writes `band.png`, `band.dat` (or spin-resolved files),
and `KPATH.txt` below `JOB`. Use `--efermi` to provide an explicit Fermi
energy when it is not present in the log. Only `calculation nscf` jobs are
recommended; other calculation types emit a warning and are still processed.

Band-gap, effective-mass, and projected (fat) band analysis use the same
command:

```text
abacustools postprocess band -j JOB --gap
abacustools postprocess band -j JOB --gap --json
abacustools postprocess band -j JOB --effective-mass cbm --direction G X
abacustools postprocess band -j JOB --fat-band species
```

`--gap` reports the band gap, VBM, CBM, and whether the gap is direct or
indirect (`--spin-resolved` adds per-spin gaps; `--json` prints the report as
JSON). `--effective-mass cbm|vbm` with `--direction START END` fits the band
curvature at the edge and is available for nspin=1 only.
`--fat-band species|species-shell|species-orbital|atoms` plots the projected
band structure from the `PBANDS_*` output.

COHP and COOP curves can be calculated for selected ABACUS NAO orbital groups
when the LCAO output contains `data-*-H`, `data-*-S`, `kpoints`, and
`WFC_NAO_K*.txt` files:

```text
abacustools postprocess cohp -j JOB \
  --atom-i-orbs 0,1,2 --atom-j-orbs 13,14,15 \
  --method COHP --de 0.1 --output cohp.png --data-output cohp.dat
```

Orbital indices are zero-based global NAO indices. Use `--method COOP` for
overlap-weighted curves, `--spin up|down` for a spin channel, and `--invert`
to invert only the plotted curve. The data file keeps the computed sign.
Develop-version output is read as well: `hk*_nao.txt` and `sk*_nao.txt` replace
the `data-*-H`/`data-*-S` pair, `wf*_nao.txt` replaces `WFC_NAO_K*.txt`, and
the k-point weights come from the `dm*_nao.txt` density-matrix headers.
`ICOHP`/`ICOOP` printed by the command is the integral up to the Fermi level.
This is an ABACUS-NAO COHP/COOP implementation and is not a standard LOBSTER
pCOHP projection.

DOS and projected DOS can be processed from an ABACUS output directory:

```text
abacustools postprocess dos -j JOB --emin -20 --emax 10
abacustools postprocess dos -j JOB --pdos species
```

The command reads `DOS*_smearing.dat`, shifts the energy axis by the Fermi
energy, and writes `DOS.png` and `DOS.dat`. Use `--pdos species-shell`,
`--pdos species-orbital`, or `--pdos atoms --atom-index 1 2` for projected DOS
plots. Output paths can be changed with `-o` and `--data-output`; use
`--efermi` when the Fermi energy cannot be read from the ABACUS output.

Use `--combined` to overlay the total DOS with the species-projected DOS
(`DOS_PDOS.png`/`DOS_PDOS.dat`), `--pdos atom-shell` or `--pdos atom-orbital`
with `--atom-index` for single-atom projections, and `--list` (optionally with
`--json`) to list the available species, shells, orbitals and atoms.

Mayer bond orders can be analyzed from an ABACUS LCAO calculation with
`out_mat_hs=1` (and `out_dm=1` for gamma-only jobs):

```text
abacustools postprocess mayer -j JOB --cutoff 3.0
abacustools postprocess mayer -j JOB --pairs 1-2,1-3 --json -o mayer.json
```

The analyzer reads numerical orbitals, overlap matrices, density matrices or
NAO wavefunctions, validates their dimensions, and reports atom indices,
elements, periodic distances, and Mayer bond orders. It supports gamma-only
and multi-k calculations with `nspin=1` or `nspin=2`, for both the LTS output
layout (`data-*-S` with `SPIN1_DM`/`SPIN2_DM` or `WFC_NAO_K*.txt`) and the
develop layout (`sk*_nao.txt` with `dm*_nao.txt`), including the gamma-only
names that omit the k-point index. The develop density matrices are used
directly, so no wavefunction reconstruction is needed there.

Complex calculation workflows are organized by task and stage. The BSSE
workflow currently provides the preparation and postprocessing framework:

```text
abacustools workflow bsse prepare
abacustools workflow bsse postprocess
```

Charge-density difference calculations are available as `chgdiff`. Prepare
the full-system and two-subsystem SCF jobs, submit them with
`sbatch runabacus.sh`, and then generate the difference cube:

```text
abacustools workflow chgdiff prepare -j JOB -i 1 3 4
abacustools workflow chgdiff postprocess -j JOB
```

Elastic constants can be calculated with the stress-strain workflow. It
generates one unstrained job and 24 independently strained jobs. By default,
the jobs use ionic relaxation at fixed cell shape; use `--norelax` for fixed-ion
SCF calculations:

```text
abacustools workflow elastic prepare -j JOB
sbatch runabacus.sh  # submit from each generated job directory
abacustools workflow elastic postprocess -j JOB
```

The fitted elastic tensor and Voigt moduli are written to
`elastic_results.json` under `JOB`.

Phonon spectra can be calculated with finite differences using Phonopy. The
prepare stage generates displaced supercell SCF jobs, and the postprocess
stage reads their forces to produce thermal properties, DOS, and a combined
dispersion/DOS plot:

```text
abacustools workflow phonon prepare -j JOB
abacustools workflow phonon postprocess -j JOB
```

Use `--supercell A B C` to set the supercell explicitly. Without it, the
supercell is selected so each lattice vector is at least 10 Angstrom long.
Custom paths can be passed as JSON with `--qpath` and
`--high-symm-points`.

Lattice thermal conductivities can be calculated with phono3py using third-order
force constants. The prepare stage generates the displaced supercells below
`fc3-*`, plus an optional independent `fc2-*` set, and writes the exact
phono3py displacement dataset to `phono3py_disp.yaml`; the postprocess stage
reads the forces, fits `fc2`/`fc3` and solves the phonon Boltzmann transport
equation. phono3py is required for this workflow and is imported only when it
runs:

```text
abacustools workflow thermal-conductivity prepare -j JOB
abacustools workflow thermal-conductivity prepare -j JOB \
  --supercell-fc3 2 2 2 --supercell-fc2 4 4 4 \
  --displacement-stepsize-fc3 0.03 --displacement-stepsize-fc2 0.01
abacustools workflow thermal-conductivity postprocess -j JOB --mesh 15 15 15
abacustools workflow thermal-conductivity postprocess -j JOB --lbte \
  --tmin 200 --tmax 400 --tstep 50
```

`kappa` is an alias of `thermal-conductivity`. Without `--supercell-fc3` the
supercell is selected so each lattice vector is at least `--min-supercell-length`
(10 Angstrom) long; without `--supercell-fc2`, `fc2` reuses the `fc3` supercell.
An explicit gamma/MP `KPT` mesh is divided by the supercell repetitions so the
k-point density stays roughly constant, while `kspacing` and `gamma_only` input
are used unchanged. The postprocess stage writes the conductivity tensor per
temperature, the RTA reference values when `--lbte` is used, a
`thermal_conductivity.png` plot, and the phono3py `kappa-*.hdf5` below `JOB`,
and it fails with an explicit error when a force calculation is missing or did
not converge.

Molecular vibration frequencies can be calculated with selected atoms using
central finite differences. The workflow writes equilibrium and displaced
force calculations below `vib/`:

```text
abacustools workflow vibration prepare -j JOB
abacustools workflow vibration postprocess -j JOB
```

Use `--index 1 2 ...` to select atoms and `--traj` to write mode trajectories.
Postprocessing reports the frequencies in `cm^-1`, the zero-point energy and
the thermochemical corrections as JSON, marks imaginary modes with `i` in the
mode output and writes the displacements of every mode below `vib/modes/`,
with `--traj` also writing one full vibration period per mode as `extxyz` or
ASE `traj` below `vib/mode_trajectories/`. The mode velocities are scaled for
visualization, so a reference mode of 2500 `cm^-1` moves its atoms with a peak
velocity of 0.5, which keeps every animation in a readable range instead of
growing with the frequency.

Postprocessing uses ASE by default (`--backend ase`): the Hessian is analyzed
by `ase.vibrations.data.VibrationsData`, the thermochemistry by
`ase.thermochemistry.HarmonicThermo`, and the mode structures are written by
ASE. With `--backend builtin` the same workflow runs on the built-in analysis
of `abacustools.data.vibration`, which is the extension point for features that
ASE does not provide:

```text
abacustools workflow vibration postprocess -j JOB --backend builtin
```

Both backends mass-weight the Hessian with the relative atomic masses declared
in `ATOMIC_SPECIES`, which `AbacusSTRU.masses` reports for every structure read
from a STRU file, so the two backends agree to the precision of their
constants. `--mass ELEMENT=MASS` (also `--element-mass`) replaces these masses
for one or more elements, which accounts for isotope effects:

```text
abacustools workflow vibration postprocess -j JOB --mass H=2.014
abacustools workflow vibration postprocess -j JOB --mass H=2.014 O=18.0
```

Keys are matched against the element of every atom first and against the ABACUS
atom label afterwards, so structures with custom labels can be addressed as
well. An assignment that matches no atom of the structure is an error. The
masses that enter the analysis are reported as `masses` in the result JSON.
The built-in backend reports unstable modes as negative frequencies, follows
the harmonic oscillator partition function for the thermochemistry and writes
mode structures with the built-in extended XYZ writer, which carries the cell,
the magnetic moments and the pseudopotential/orbital metadata of the
equilibrium structure. Its JSON adds one entry per mode with the reduced mass
in `amu` and the force constant in `eV/Angstrom^2`, and, per temperature, the
internal energy, the heat capacity and the number of imaginary modes next to
the entropy and the free energy. In both backends the reported zero-point
energy keeps the convention of adding the magnitude of an imaginary mode, while
the thermochemistry of a temperature is evaluated from the stable modes only.

Workflow submission scripts can be generated from `~/.abacustools/config.yaml`.
The packaged defaults support local execution and Slurm, PBS, and LSF
submission. Script generation is disabled by default; enable it globally with
`submission.generate: true`, or for one preparation use `--submit-script`.
Use `--no-submit-script` to override an enabled default, and
`--submission-type local|slurm|pbs|lsf` to choose a configured template:

```yaml
submission:
  generate: true
  default: slurm
  abacus_command: "mpirun -np 8 abacus"
```

For the vibration workflow, the generated `submit_vibration.sh` runs the
equilibrium task first and submits all displacement tasks in parallel after it
finishes. The task template is applied to each generated task directory, so
site-specific scheduler directives can be customized without changing the
workflow implementation. Templates support `{abacus_command}`, `{job_name}`,
`{task_name}`, and `{workflow}` placeholders; literal shell braces must be
written as doubled braces.

Analytic forces and stresses can be checked independently with central finite
differences. Force validation accepts one-based atom indices, or the
`abacus-test`-compatible `info.txt` format (`C 2 x y z`):

```text
abacustools workflow fdforce prepare -j JOB --index 1 2 --dir x y z
abacustools workflow fdforce postprocess -j JOB
```

Stress validation defaults to the six independent cell components and keeps
fractional coordinates fixed while deforming the cell. Use `--all-components`
to include all nine components:

```text
abacustools workflow fdstress prepare -j JOB --step 0.0001 --number 5
abacustools workflow fdstress postprocess -j JOB
```

Both workflows write their generated calculations below `fdforce/` or
`fdstress/`, record the exact finite-difference cases in a manifest, and write
JSON results containing analytic values, finite-difference values, deviations,
and step-size convergence. Missing or unconverged ABACUS results stop
postprocessing with an explicit error.

Surface work functions can be calculated from the averaged electrostatic
potential. The prepare stage enables `out_pot=2` and writes a calculation
under `workfunc_job`; the postprocess stage identifies vacuum plateaus and
writes the work-function results and potential profile:

```text
abacustools workflow workfunc prepare -j JOB
abacustools workflow workfunc postprocess -j JOB
```

Use `--vacuum a|b|c|auto` to select the vacuum direction, and
`--dipole-corr` to enable dipole correction.

Born effective charges can be calculated with Berry-phase finite differences.
The workflow prepares SCF and three Berry-phase NSCF calculations for each
selected atom displacement:

```text
abacustools workflow bec prepare -j JOB --index 1 --dir x y z --type c
```

Run `run_bec.sh` in each generated `bec_*` directory, then postprocess the
polarization differences:

```text
abacustools workflow bec postprocess -j JOB
```

The BEC tensors and task diagnostics are written to `bec_results.json` under
`JOB`. Missing or incomplete Berry-phase task output is retained as missing
tensor entries so other completed displacement directions can still be reported.

Piezoelectric stress tensors can be calculated from finite-strain changes in
the Berry-phase polarization. The workflow generates the six independent
Voigt strain modes (`xx`, `yy`, `zz`, `yz`, `xz`, and `xy`), with forward,
backward, or central differences:

```text
abacustools workflow piezoelectric prepare -j JOB --strain 0.01 --type c
```

Run `run.sh` in each generated `piezoelectric_*` directory. Use `--relax` to
relax ionic positions at each strained cell before the Berry-phase
calculation. The tensor and per-task diagnostics are written under `JOB`:

```text
abacustools workflow piezoelectric postprocess -j JOB
```

Hubbard `U` parameters can be derived from first principles with the linear
response method, which screens the occupation of the correlated orbitals
against a small applied `U`. The prepare stage writes one SCF calculation per
scanned `U` value, treating the atoms named by `--index` or `--elements` as the
correlated ones and giving them their own species label so that only that atom
carries the perturbation:

```text
abacustools workflow dftu prepare -j JOB --index 1 --u-min 0 --u-max 0.5 --u-step 0.1
abacustools workflow dftu postprocess -j JOB
```

`--u-values` lists the scanned values directly, `--orbital 2|3` selects d or f
orbitals instead of inferring the angular momentum from the element, and
`--dftu-type` picks the DFT+U flavour, defaulting to the recommended method 1.
The postprocess stage reads the local occupation matrices from every
`running_scf.log`, fits the bare and screened responses, and writes the
resulting `U` per atom to `dftu_results.json` together with a `dftu_response.png`
plot. Submit scripts are generated on request with `--submit-script` and
`--submission-type`.

Generated calculation directories are protected by default. Use `--override`
when intentionally replacing them. The BEC workflow's `run_bec.sh` is only a
local four-step runner for one generated task; cluster submission scripts are
not generated because their contents depend on the target computing environment.

Each prepared workflow records its task names and atom partition in a
workflow-specific manifest such as `workflow_phonon.json`. Postprocessing
validates this manifest and checks that the required SCF calculations converged
before reading their outputs.

## ABACUS version profiles

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

## ABACUS ASE interface

The separate `abacuslite` ASE interface can be used as an optional calculation
backend while ABACUSTools continues to provide job and structure management.
It is loaded only when requested, so it is not required for the normal command
line workflows.

```python
from ase.optimize import BFGS
from abacustools.io.stru import AbacusSTRU
from abacustools.integrations.abacuslite import (
    attach_calculator,
    calculator_from_structure,
    make_profile,
    structure_to_atoms,
)

structure = AbacusSTRU.read("STRU")
profile = make_profile(
    "mpirun -np 8 abacus",
    pseudo_dir="/path/to/pseudopotentials",
    orbital_dir="/path/to/orbitals",
)
atoms = structure_to_atoms(structure)
calculator = calculator_from_structure(
    structure,
    profile,
    directory="ase-relax",
    inp={"calculation": "scf", "basis_type": "lcao", "cal_force": 1},
)
attach_calculator(atoms, calculator)
BFGS(atoms).run(fmax=0.05)
```

The same adapter can build a calculator from an existing ABACUS job with
`calculator_from_job`. The returned ASE calculator can then be used by ASE
relaxation, cell-relaxation, NEB, MD, or `fixed_density()` band workflows.
Use `result_to_dict()` or `write_result()` to place ASE results in the
repository's JSON-compatible result format.

## Materials Project database

The `mp` command family searches the Materials Project and downloads structures
as ABACUS or common structure files. It needs the optional `mp-api` client and
an API key:

```bash
pip install 'abacustools[mp]'
export MP_API_KEY="your_key_here"   # https://materialsproject.org/api
```

`mp search` accepts the usual Materials Project selectors; at least one of
`--formula`, `--chemsys`, `--elements`, or `--material-id` is required:

```text
abacustools mp search --formula Fe2O3 --limit 10
abacustools mp search --chemsys Li-Fe-O --stable --limit 5 --json
abacustools mp search --elements Li Fe O --output li-fe-o.json
abacustools mp search --material-id mp-149 --material-id mp-22862
```

The table reports the material id, formula, chemical system, number of sites,
energy above the convex hull, band gap, stability, and whether the entry is
theoretical. `--json` prints the same records as JSON and `--output` also
writes them to a file. The command returns a non-zero exit status when nothing
matches the query.

`mp download` writes one directory per material, which can be used as an
ABACUS job directory directly:

```text
abacustools mp download mp-149
abacustools mp download mp-149 mp-22862 --output structures --format poscar
abacustools mp download mp-149 --format cif --json
```

Every structure goes to `OUTPUT/<material_id>/`, named `STRU`, `POSCAR`,
`structure.cif`, `structure.xyz`, `structure.extxyz`, or `structure.xsf`
according to `--format`. Materials Project structures carry no pseudopotential
or numerical-orbital information, so the `ATOMIC_SPECIES` files required by
ABACUS still have to be filled in, for example with the library settings
described above or `abacustools file stru`.

The same operations are available from Python, and every call accepts a
pre-constructed `client` so that no connection is made when one is supplied:

```python
from pathlib import Path
from abacustools.integrations.materials_project import (
    material_directory,
    search_materials,
    download_material,
    write_material_structure,
)

for summary in search_materials(chemsys="Li-Fe-O", is_stable=True, limit=5):
    print(summary.material_id, summary.formula, summary.energy_above_hull)

material = download_material("mp-149")
write_material_structure(
    material,
    material_directory(Path("structures"), material.material_id),
)
```

`load_materials_project()` and `materials_project_available()` report whether
the optional client can be imported. The API key is taken from the first of
`MP_API_KEY`, `PMG_MAPI_KEY`, and `MAPI_KEY` that is set, or from an explicit
`--api-key` / `api_key=` argument.
