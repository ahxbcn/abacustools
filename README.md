Collection of tools used for performing DFT calculation with ABACUS.

## Python data types

The core data types are importable straight from the package, so a script can
work with structures, inputs and results without reaching into the `io`/`data`
submodules:

```python
from abacustools import AbacusSTRU, AbacusATOM, ReadInput, Unitcell, UPF

structure = AbacusSTRU.read("STRU")
inputs = ReadInput("INPUT")
```

The exported set covers the file-handling types: the structures
(`AbacusSTRU`, `AbacusATOM`, `AbacusAtomType`, `StructureConversionWarning`),
the cell helper `Unitcell`, `INPUT` (`ReadInput`, `WriteInput`), the
pseudopotential and orbital readers (`UPF`, `AbacusNAO`) and the Molden types
(`MoldenShell`, `MoldenAtom`, `MoldenOrbital`). The names resolve lazily, so
`import abacustools` stays cheap and a defining module is loaded only when its
name is first used; `dir(abacustools)` lists the whole set.

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

The top-level families are `file`, `job`, `database`, `postprocess`,
`workflow`, and `mp`. The `database` family searches and downloads structures
from many materials databases as described at the end of this document, and
`mp` is the short spelling of `database --database mp`.

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
abacustools file struinfo STRU
abacustools file struinfo POSCAR --json
abacustools file struinfo structure.xyz --cell 10 0 0 0 10 0 0 0 10
abacustools file struinfo slab.STRU --coordination
abacustools file struinfo STRU --coordination voronoi --json
abacustools file struinfo *.vasp POSCAR --json
abacustools file struinfo STRU --summary
```

The report includes cell parameters, volume, density, element and label counts,
the formula unit with the number of formula units per cell and the prototype
formula, space group, crystal system, point group with its Schoenflies symbol,
Bravais lattice with the Pearson symbol, inversion symmetry, polar point group,
the symmetry tolerances that were used, symmetry operation count, per-atom
Wyckoff positions with their multiplicity and site symmetry, a separate list of
symmetry-inequivalent atomic positions, and, when the input is read as an
ABACUS `STRU`, the ABACUS pseudopotential and orbital file of every label. The inequivalent list contains one representative
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

Several structures can be listed at once, which reports only the fields that
make them comparable: the file, the formula, the number of atoms, the space
group with its number, the crystal system and the cell parameters, with the
lengths in Angstrom, the angles in degree and the volume in Angstrom^3.
`--summary` asks for that listing for a single structure as well. The listing
leaves the rest of the report out, so it does not pay for the Wyckoff
positions, the point group, the Bravais lattice, the magnetic and layer
symmetry, the dimensionality, the coordination or the per-atom table;
`--coordination` and `--layer-direction` are rejected together with it, and
`--json` returns one summary object per structure instead of a report.

`file kpt` inspects a KPT file, or writes a new one. Without `--output` it
reports the model, the mesh or the k-point list, and validates the values;
adding `--structure` also reports the k-spacing in 1/Angstrom that the mesh
realizes. With `--output` it writes a mesh (`--mesh 9 9 9`, or `--spacing 0.03`
together with `--structure`, which expands the target spacing into a mesh the
way ABACUS does) or, with `--path` and `--structure`, the seekpath
high-symmetry path as a line-mode KPT with `--npoints` points per segment:

```text
abacustools file kpt KPT --structure STRU
abacustools file kpt --structure STRU --spacing 0.03 -o KPT.scf
abacustools file kpt --structure STRU --path --npoints 20 -o KPT.band
```

The path follows the dimensionality of the structure: a bulk uses the seekpath
high-symmetry path, a slab uses the in-plane path of its 2D Bravais lattice
(hexagonal, square, rectangular or a generic oblique loop) with the vacuum
direction pinned to `k = 0`, and a wire uses the single periodic direction.
`--min-vacuum` sets the empty span that counts as vacuum (5 Angstrom by
default), and `--path-mode auto|bulk|slab|wire` forces one of them; a
zero-dimensional structure has no band path and is rejected.

Two structures can be stacked into a heterojunction interface with pymatgen's
coherent interface builder, which searches the Zur-McGuire lattice matches of
the two surfaces and strains the film onto the substrate:

```text
abacustools file interface FILM SUBSTRATE -o HET --film-miller 0 0 1 --substrate-miller 0 0 1
abacustools file interface FILM SUBSTRATE --list --max-strain 0.03 --max-area 200
abacustools file interface FILM SUBSTRATE -o HET --film-thickness 3 --substrate-thickness 3 --gap 2.5 --vacuum 15
```

`--list` prints the candidate lattice matches (supercell area, length and angle
mismatch, supercell size) without writing anything; otherwise the best match
that satisfies `--max-atoms` is written. `--film-thickness` and
`--substrate-thickness` size the two films in layers, or in Angstrom with
`--in-angstrom`, `--gap` and `--vacuum` place the stack, `--termination` picks
one of the surface terminations, and `--max-strain`, `--max-angle` and
`--max-area` bound the lattice search. The film sits on top of the substrate
along `c`. The pseudopotential and orbital of every element are inherited from
the structure that contains it, film or substrate, with the configured resource
library as the fallback, and the report names each file together with the reason
for choosing it.

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
abacustools file editstru substitute STRU -o DOPED --element Fe --elements Si
abacustools file editstru fix       STRU -o FIXED --coords 0 0.2 --direction c --direct
abacustools file editstru fix       STRU -o FIXED --elements O --move x y --free-others
abacustools file editstru direct    STRU -o DIRECT
abacustools file editstru cartesian STRU -o CART
abacustools file editstru primitive   STRU -o PRIM
abacustools file editstru conventional STRU -o CONV
abacustools file editstru standardize STRU -o STD --to-primitive
abacustools file editstru symmetrize  STRU -o CLEAN --symprec 0.001
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

`substitute` replaces the atoms selected by `--indices`, `--elements` or
`--coords`/`--direction` with `--element`. The new atoms take the mass of that
element, and their pseudopotential and orbital are chosen in a fixed order: when
the input structure already contains the element, the files that element uses
there are reused, so a doped cell stays consistent with its host; otherwise they
come from the configured resource library of `job prepare` (`--library` and
`--variant` select it, with the `ABACUS_PP_PATH`/`ABACUS_ORB_PATH` fallbacks),
and `--pp`/`--orb` override either choice. `--basis auto|lcao|pw` decides
whether an orbital is needed at all; `auto` follows the input structure, and a
plane-wave structure drops the orbital. The command reports the chosen
pseudopotential and orbital together with the reason for each, `--label` names
the new species, and `--keep-moments` keeps the magnetic moments of the
replaced atoms, which are cleared otherwise.

The STRU reader follows the newer ABACUS conventions as well: the
`Cartesian_angstrom`, `Cartesian_au` and
`Cartesian_angstrom_center_{xy,xz,yz,xyz}` coordinate modes, `#` comment
annotations after the block keywords, the per-atom force field
(`f`/`force`/`forces`, in eV/Angstrom) and the `pp_type` column of
`ATOMIC_SPECIES` are all parsed, kept on the atoms and written back.

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

`symmetrize` cleans up a structure instead of changing its cell setting. spglib
finds the space group, the atomic positions are averaged with their symmetry
images and the lattice is strained onto the metric that space group requires, so
the small numerical errors disappear and the symmetry of the result is exact.
The cell setting, the number and order of the atoms and every atom attribute are
kept, which distinguishes the action from `standardize` and `conventional`.
`--symprec` sets which deviations count as noise, so raise it above the errors
you want to remove (1e-5 would round away almost nothing), and `--keep-cell`
idealizes the positions only and leaves the lattice as it is.

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
target format. Trajectories are converted with `file traj`, which reads and
writes any multi-frame format that ASE supports, so an `extxyz` file becomes
an ASE `.traj`, a plain `.xyz` or an `.xsf` for a viewer of choice. Standard XYZ files do not contain a periodic cell; provide
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

Magnetic and DFT+U settings are prepared with `--nspin`, `--soc`,
`--init-mag`, `--afm` and `--dftu-param`. `--dftu-param ELEMENT U` enables
DFT+U by itself and repeats for several elements. Initial magnetic moments
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
file. `--no-submit-config` overrides a configuration that enables the file. The
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

A band structure along the seekpath high-symmetry path is prepared from a
reference input directory:

```text
abacustools workflow band prepare -j JOB --npoints 20
abacustools workflow band postprocess -j JOB
```

The prepare stage writes `band_scf/`, an SCF job that also stores the charge
density, and `band_nscf/`, an NSCF job whose line-mode KPT follows the path and
which reads the SCF density back and writes `BANDS_1.dat`; run the SCF first.
`--nbands` sets the number of bands of the NSCF step. The path itself follows
the dimensionality of the structure as in `file kpt --path`: seekpath for a 3D
bulk, the in-plane path of the 2D lattice for a slab with the vacuum direction
at `k = 0`, and the periodic axis for a 1D wire. `--min-vacuum` and
`--path-mode auto|bulk|slab|wire` control that choice, and the workflow manifest
records the dimensionality and the method that produced the path. The postprocess stage
reports the band gap with its VBM and CBM, writes `band_results.json` next to
the workflow manifest, and plots the bands to `band.png`.

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

An ABACUS LCAO wavefunction can be exported to the Molden format, for example
to visualize the orbitals in Molden, Multiwfn or Avogadro:

```text
abacustools postprocess molden -j JOB
abacustools postprocess molden -j JOB -o orbitals.molden --gto-primitives 8
abacustools postprocess molden -j JOB --atoms-unit angstrom --json
```

The command reads the LCAO coefficients from `WFC_NAO_GAMMA*` (or the
develop `wf*_nao.txt` names and the `WFC_NAO_K*` files), expands every
numerical atomic orbital into a contracted Gaussian fit, and writes
`wfc.molden` below `JOB`. Valence counts for the `[Nval]` block come from the
pseudopotentials, and `[5D7F]`/`[9G]` are written whenever the basis reaches
that angular momentum, so the pure spherical harmonics of ABACUS are kept.
`nspin 2` jobs write both spin channels into one file.

The Molden format stores a single real set of orbitals, so the job must be a
`gamma_only 1` run, or a non-gamma run whose selected k-point is real; use
`--kpoint` to pick one k-point of the latter. `--gto-primitives` sets the
number of Gaussians per numerical orbital (default 6) and the printed report
quotes the largest relative radial fit error.

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

Charge densities can be inspected without leaving the command line. The
command reads the density of a job, reports the integrated electrons next to
the valence electrons of the atoms, and writes it as a cube, a planar profile
or a two-dimensional slice:

```text
abacustools postprocess chg -j JOB
abacustools postprocess chg -j JOB --cube charge.cube
abacustools postprocess chg -j JOB --spin difference --cube magnetization.cube
abacustools postprocess chg -j JOB --difference OTHER_JOB --cube bonding.cube
abacustools postprocess chg -j JOB --profile c
abacustools postprocess chg -j JOB --slice c --slice-index 0.5 --slice-plot
abacustools postprocess chg -j JOB --json
```

`--spin` selects what every action works on: the total density, the up or down
channel of an `nspin 2` calculation, or their difference, whose integral is the
magnetization in Bohr magneton. `--difference OTHER_JOB` subtracts another job
before the analysis and turns the command into a general density-difference
tool, as long as both grids match, so a bonding or adsorption difference no
longer needs a prepared `workflow chgdiff` set.

The density is taken from the cubes that an SCF calculation writes with
`out_chg 1`, and both ABACUS naming conventions are read: the LTS branch writes
`SPIN1_CHG.cube` while develop writes `chg.cube` or `chgs1.cube`, and a
geometry-step token such as `chgs1g3.cube` is recognised, with the last step
used when the job wrote one file per step. A job that only stores the
`*-CHARGE-DENSITY.restart` backup, which is what `out_chg 0` leaves behind, is
converted from `rho(G)` instead: the structure and its pseudopotentials give
the cell and the valence charges, and the FFT grid comes from the log of the
current calculation, with `--grid NX NY NZ` as an override. A `gamma_only`
restart file, the usual case at the Gamma point, stores one G-vector of every
`+G`/`-G` pair, and the missing half is rebuilt with `rho(-G) = conj(rho(G))`.
The summary compares the
integrated charge with the valence charge of the atoms that the cube stores, so
a deviation reports either a charged cell or the truncation of the real-space
grid; the comparison is left out for a spin channel or a difference, where it
has no meaning.

`--quantity` moves the analysis to a field derived from the selected density,
with the definitions of Quantum ESPRESSO's `pp.x`: `rdg` is the reduced density
gradient (`plot_num=19`), `sl2rho` is `sign(lambda_2) rho` built from the middle
eigenvalue of the density Hessian (`plot_num=20`), and `dori` is the density
overlap regions indicator (`plot_num=123`). The derivatives are evaluated in
reciprocal space, as `pp.x` does, so the two codes agree on the same grid, and
`--cube`, `--profile` and `--slice` work on the derived field as well.
`--nci-plot` draws the non-covalent interaction plot of the density, the
reduced density gradient against `sign(lambda_2) rho`, keeping the grid points
below `--nci-rho-max` (`0.05` e/Bohr^3 by default) so that the cores and the
bonds stay out of the plot.

`iri` is the interaction region indicator of Multiwfn, `|grad rho| / rho**1.1`,
which shows covalent and non-covalent interactions in one function; points
below `5e-5` e/Bohr^3 are replaced by zero instead of the placeholder Multiwfn
uses. The `-promolecular` variants and `dg` build the promolecular reference
from the `PP_RHOATOM` table of the same UPF files that the calculation used:
the reference is the superposition of the pseudoatomic densities of the atoms,
and `dg` is the independent gradient model function `sum_A |grad rho_A| -
|grad rho|`, whose plot against `sign(lambda_2) rho` comes from `--igm-plot`.
These three need the structure and its pseudopotentials, so they cannot be
combined with `--difference`, and `--promolecular-plot` draws the NCI plot of
the reference density itself.

The Hirshfeld-partitioned independent gradient model, IGMH, replaces the frozen
promolecular atomic densities with `rho_A = w_A rho`, where the weights come
from the Hirshfeld partition of the calculated density. The `igmh-i` variant
uses the self-consistent Hirshfeld-I weights instead:

```text
abacustools postprocess chg -j JOB --quantity igmh --igmh-plot
abacustools postprocess chg -j JOB --quantity igmh-i --igmh-plot
```

Both quantities are written in the same units as `dg` and can be exported with
`--cube`, `--profile` or `--slice`. The partition uses the total density of the
job and the same pseudopotential references as `postprocess hirshfeld`, so these
analyses require `--spin total`, cannot be combined with `--difference`, and
need a pseudopotential with `PP_RHOATOM` for Hirshfeld; the Hirshfeld-I variant
additionally needs `PP_PSWFC`. The data API also accepts explicit reference
densities for that variant.
Without `--igmh-plot`, `igmh` and `igmh-i` are still available as fields through
`--quantity`.

`--profile AXIS` writes the in-plane average of every plane in e/Angstrom^3 to
`chg_profile_<axis>_average.dat`, or the charge of every plane in e with
`--profile-kind integral`, whose sum is the total number of electrons.
`--slice AXIS` writes the plane closest to the fractional `--slice-index` (0.5
by default) as three columns with the two in-plane axes and the value, and
`--slice-plot` draws it as a colour map together with the atoms that the plane
crosses; `--no-atoms` leaves those markers out and `--vmin`/`--vmax` fix the
colour range. `--data-output`, `--slice-output`, `--plot` and `--slice-plot`
change the file names, and a plot flag without a name writes
`chg_profile_<axis>_<kind>.png` or `chg_slice_<axis>_<position>.png`.

Molecular-dynamics trajectories are written to standard formats. ABACUS
appends one block per dumped step to `OUT.<suffix>/MD_dump`, holding the cell,
the positions and, when `dump_force`, `dump_vel` and `dump_virial` are enabled,
the forces, the velocities and the virial; `postprocess md` turns those blocks
into a trajectory file, with the energy, temperature and pressure of every step
attached from the running log:

```text
abacustools postprocess md -j JOB
abacustools postprocess md -j JOB -o traj.extxyz --first 100 --last 2000 --stride 5
abacustools postprocess md -j JOB -o trajectory.traj --json
```

The suffix of the output selects the format, which is any format ASE writes,
such as `extxyz`, `xyz`, `traj` or `xsf`, and `--format` sets it explicitly.
Positions are in Angstrom, forces in eV/Angstrom, velocities in Angstrom/fs and
the virial in kBar, as in `MD_dump` itself. A job that wrote no `MD_dump` is
read from its per-step `STRU_MD_*` structures instead, which is what `out_stru 1`
produces in the job directory of the LTS branch and in a directory per step of
develop; those frames carry positions and velocities but no forces or virial.

`file traj` converts a trajectory between formats with the same machinery:

```text
abacustools file traj trajectory.extxyz trajectory.xyz
abacustools file traj dump.traj dump.extxyz --stride 10
```

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

A run that reduced the k-point mesh (`symmetry 1`) is analyzed on the full
mesh. The analyzer reads the mesh and the irreducible k-points from
`OUT.*/kpoints`, rebuilds the space-group operations from the structure, and
evaluates every star member through the atom permutation of the operation that
reaches it. A bond order is quadratic in the k-resolved density matrix, so the
star cannot be folded into a k-point weight; expanding it is what makes a
`symmetry 1` run agree with the `symmetry 0` and `symmetry -1` runs of the same
calculation, which the analyzer reads directly. The weights are rebuilt from
the star sizes as well, because the `kpoints` file prints them with four
decimals only, which would bias the totals by about 0.2%.

An output that does not record how the mesh was reduced - an older format, or a
magnetic `symmetry 2`/`symmetry 3` run whose reduction is not the
crystallographic one - cannot be expanded; the analyzer says so and asks for a
`symmetry 0` or `symmetry -1` calculation instead.

Bader charges are integrated over the Bader volumes of a job with the external
[Henkelman](https://theory.cm.utexas.edu/henkelman/code/bader/) program or with
the Python [baderkit](https://github.com/SWeavz/baderkit) library:

```text
abacustools postprocess bader -j JOB
abacustools postprocess bader -j JOB --backend baderkit
abacustools postprocess bader -j JOB --reference OTHER.cube --vacuum auto --keep-cubes
abacustools postprocess bader -j JOB --json -o bader.json
```

Both backends partition the same density, which comes from the `SPIN*_CHG.cube`
of `out_chg 1` or from the `*-CHARGE-DENSITY.restart` backup, converted from
`rho(G)` with the FFT grid of the running log (`--grid NX NY NZ` overrides it)
and the lattice constant of the STRU (`--lat0`). `--cube` selects another cube
and `--reference` partitions with another density. `--vacuum off|auto|DENSITY`
follows the flag of the external program, where `auto` is the 1e-3
e/Angstrom^3 cutoff. An `nspin 2` job integrates the magnetization over the
Bader volumes of the total density, which is what gives the per-atom spin
moments in Bohr magneton.

The external program is resolved from `--bader-exe`, the `BADER_EXE`
environment variable or `bader.exe` in `~/.abacustools/config.yaml`, and writes
its `ACF.dat` into the working directory that `--workdir` or `--keep-cubes`
preserves. The `baderkit` backend needs the optional package
(`pip install abacustools[baderkit]`) and runs the partition in the Python
process instead, so it needs no executable. `--baderkit-method` selects one of
its `neargrid`, `neargrid-weight`, `ongrid` and `weight` algorithms; the
default `neargrid` is the partitioning the external program applies, which
makes the two backends agree to a few 1e-3 e on the same cube.

Charges are reported next to the valence electron count of the pseudopotential,
so `net charge = z_valence - bader_charge` is positive for an electron-poor
atom. Positions and distances are in Angstrom and volumes in Angstrom^3, for
both backends.

Hirshfeld (stockholder) atomic charges are computed directly from the
charge density of a job, using the spherically averaged free-atom densities of
the pseudopotentials as the promolecule:

```text
abacustools postprocess hirshfeld -j JOB
abacustools postprocess hirshfeld -j JOB --json
```

The density is read from the charge-density cube or the `*-CHARGE-DENSITY.restart`
file, and the proatoms are summed over lattice images so the promolecule is
periodic. CM5 charges (Marenich, Jerome, Cramer and Truhlar, *J. Chem. Theory
Comput.* 2012, 8, 527) add Pauling-bond-order weighted pairwise corrections to
the Hirshfeld charges; the parameters of the paper's Table 1 and its covalent
radii are built in, so no extra file is needed:

```text
abacustools postprocess hirshfeld -j JOB
abacustools postprocess hirshfeld -j JOB --no-cm5
```

Hirshfeld-I (Bultinck, Van Alsenoy, Ayers and Carbo-Dorca, *J. Chem. Phys.*
2007, 126, 144111) makes the promolecule self-consistent instead of fixing it
to the neutral atoms.  Each iteration rebuilds every atomic reference density at
the population the previous iteration assigned to the atom, which removes the
dependence on an arbitrary reference and gives charges that track the
electrostatic-potential ones much better.  The reference densities at integer
populations come from the pseudo-atomic wavefunctions (`PP_PSWFC`) of the
pseudopotentials, filled by Aufbau around the neutral configuration and summed
over lattice images like the plain promolecule:

```text
abacustools postprocess hirshfeld -j JOB --hirshfeld-i
abacustools postprocess hirshfeld -j JOB --hirshfeld-i --json
abacustools postprocess hirshfeld -j JOB --hirshfeld-i --max-iter 300 --tol 1e-4
```

A pseudopotential without `PP_PSWFC` (many ONCV files) has no charged reference
states, so pass a directory of reference densities with `--references DIR`
instead.  Every `<element>_<population>.dat` file there holds two columns, `r`
in Angstrom and `rho(r)` in `e/Angstrom^3`, with one file per integer valence
population the iteration may need.

DDEC6 and DDEC3 net atomic charges, spin moments and bond orders are
computed with the external [Chargemol](https://ddec.sourceforge.net) program,
which partitions the valence density of a job:

```text
abacustools postprocess ddec -j JOB
abacustools postprocess ddec -j JOB --charge-type DDEC3 --json -o ddec.json
abacustools postprocess ddec -j JOB --no-spin --threshold 0.1
abacustools postprocess ddec -j JOB --pairs 1-2,1-3 --cutoff 3.0
abacustools postprocess ddec -j JOB --no-bos --threads 16
abacustools postprocess ddec -j JOB --core-electrons "26 10" --workdir ddec
```

The command writes the `valence_density.cube` (and the `spin_density.cube` of
an `nspin 2` job) that the program expects, together with its
`job_control.txt`, into a working directory, runs Chargemol there and parses
the `*.xyz` results, so the tables list the net charge, the sum of bond orders
and, for a spin-polarized job, the DDEC spin moment of every atom next to the
bond orders and their periodic images. `--workdir` and `--keep` preserve that
directory, `--output` writes the complete report as JSON and `--json` prints
it. The executable and the reference density tables are set by
`chargemol.exe` and `chargemol.atomic_densities` in
`~/.abacustools/config.yaml`, and can also come from the `CHARGEMOL_EXE` and
`CHARGEMOL_ATOMIC_DENSITIES` environment variables or from
`--chargemol-exe` and `--atomic-densities`.

A valence-only cube makes Chargemol insert the core electrons from its
`atomic_densities` tables, which fixes the number of core electrons of every
element to `Z - z_valence` of the pseudopotential; the command derives that
number from the atom columns of the density and checks the tables before the
program starts, so the trivalent lanthanide pseudopotentials of the APNS
library, whose 4f electrons sit in the core, or any other element whose core
count the distribution does not ship, are reported instead of failing inside
Chargemol. `--core-electrons "26 10"` overrides the count of one element. The
charge density should come from `out_chg 1 10` so that the cube carries enough
digits, its grid has to be finer than 0.25 Bohr per direction (0.14 Bohr is
the recommended spacing), and a charge-density cube of a PAW or ultrasoft
calculation does not integrate to the valence charge, so only norm-conserving
pseudopotentials work. A molecule in a box wants
`--periodicity false false false`.

The net charge of the cell follows the `nelec` and `nelec_delta` keywords of
INPUT: `nelec 0`, the default, means that the electrons are the sum of the
valence charges of the atoms, so the cell is neutral, while a positive value
describes a charged cell, and `--net-charge` overrides the derived value. The
command also compares the electrons of the cube with the charge of the cell
before Chargemol runs and stops with an explanation when they disagree, which
happens when the cube and INPUT belong to different calculations or when the
density comes from another grid or another `--cube`.

A job that kept no cube is read from its `*-CHARGE-DENSITY.restart` file with
the same conversion that `postprocess chg` uses: the finest FFT grid the
running log reports, the valence charges of the pseudopotentials and the
`LATTICE_CONSTANT` of STRU, with `--grid` and `--lat0` as overrides. Both
routes agree to about 1e-4 e, so a job whose `out_chg` was turned off can still
be analysed.

Bond orders and overlap populations are the expensive part of a Chargemol run:
the time grows with the number of atoms times the number of grid points. On 16
threads, 64 atoms on a 135^3 grid (2.5 million points) took 15 s, 224 atoms on
a 256x250x150 grid (9.6 million points) 66 s and 224 atoms on a 320x320x180
grid (18 million points) more than two minutes, while the same job with
`--no-bos` finished in 109 s. The charges and spin moments are identical with
and without bond orders, so a large system is best analysed with `--no-bos`
first. `--threads` sets `OMP_NUM_THREADS` of the OpenMP binary, and one
Chargemol instance should run at a time.

Chargemol 3.5 crashes on a valence-only cube when a spin density is present,
because `module_format_valence_cube_density` never allocates the arrays that
its spin reader uses. The command recognises the crash, explains it and
suggests either a patched build of Chargemol or `--no-spin`.

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

The preparation stage analyses the reference cell and records its point group,
space group and the number of independent elastic constants in
`workflow_elastic.json`. The postprocessing stage fits the unconstrained 6x6
tensor from the stresses and then projects it onto the subspace the crystal
symmetry allows, so the numerical noise of the stresses no longer shows up as
components the symmetry forbids or as a tensor that is not symmetric in its
two index pairs. Both tensors, the largest change the projection made, the
independent constants and the Voigt moduli are written to
`elastic_results.json` under `JOB`:

```text
symmetrization residual: <largest component the projection changed>, in GPa
independent constants (GPa): C11 = <...>, C12 = <...>, C44 = <...>
```

Use `--no-symmetrize` to keep the raw fit; `--symprec` sets the tolerance of
the symmetry analysis, which defaults to 0.01 Angstrom so that a relaxed cell
is still recognised as symmetric.

A two dimensional material is recognised automatically: when the reference
cell has vacuum along one direction, only the strain components of the two
periodic axes are prepared and fitted, because the components that involve the
vacuum direction are set by the cell rather than by the material. The report
then carries the in-plane block in `elastic_tensor_2d`, converted to the two
dimensional unit N/m with the cell height along the vacuum direction, its
independent constants, and the directional in-plane Young's modulus and
Poisson ratio:

```text
abacustools workflow elastic prepare -j JOB --dimension auto --strains independent
abacustools workflow elastic postprocess -j JOB
```

`--dimension 3d` forces the three dimensional treatment, and `--dimension 2d`
refuses a cell without vacuum. For a hexagonal sheet the in-plane symmetry
leaves two independent constants, so a single strain direction is enough and
the run needs five calculations instead of twenty five.

For a crystal with symmetry there is a second, cheaper route: strain only one
representative of every symmetry orbit of strain directions and fit the
independent constants directly, instead of straining all six directions and
fitting the full tensor:

```text
abacustools workflow elastic prepare -j JOB --strains independent
abacustools workflow elastic postprocess -j JOB --fit independent
```

The preparation stage picks the directions whose information raises the rank
of the fit, so a cubic crystal needs two of them (`xx` and `yz`, nine jobs
instead of twenty five), a hexagonal or trigonal one three, and a tetragonal
one four; when no symmetry relates the directions all six are kept. The
postprocessing stage then writes the stiffness matrix as a combination of the
symmetry allowed basis tensors and fits its coefficients in one least squares,
which reports the independent constants without a separate symmetrisation
step.

Elastic constants can also be obtained from the curvature of the total energy
instead of from the stresses. The `energy-strain` workflow strains the cell
along a set of patterns and fits

```text
E(e) = E0 + (V0 / 2) sum_k a_k (e B_k e)
```

```text
abacustools workflow energy-strain prepare -j JOB
sbatch runabacus.sh  # submit from each generated job directory
abacustools workflow energy-strain postprocess -j JOB
```

The patterns are picked so that their curvature covers every independent
constant: three for a cubic crystal (`xx`, `yz` and `xx + yy`, twelve strained
calculations), and one per constant in the lower symmetry classes. Every
pattern is strained with amplitudes that are symmetric about zero, which keeps
the stress of the reference cell out of the curvature; that stress is reported
separately, projected on the patterns, as a check of the reference. The fit,
the independent constants, the moduli and the root mean square energy residual
are written to `energy_strain_results.json` under `JOB`.

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

The report carries the entropy, the free energy and the heat capacity at
`--temperature`, which defaults to 298.15 K. They are written in eV and eV/K
per cell of the reference structure, the unit the vibration workflow reports
its thermochemistry in, and the `units` block of the JSON names them. Phonopy
states its thermal properties per mole of cells, a free energy in kJ/mol and an
entropy and heat capacity in J/(K mol), so they are converted on the way into
the report rather than passed through with the wrong label.

Every report carries the Gamma point modes with their degeneracy. Three
optional analyses extend it, and each one is off by default because it either
costs time or enlarges the report:

```text
abacustools workflow phonon postprocess -j JOB --debye
abacustools workflow phonon postprocess -j JOB --pdos --pdos-plot PDOS.png
abacustools workflow phonon postprocess -j JOB --irreps
```

`--debye` fits a Debye frequency to the total DOS and reports it in THz and as
a temperature. The Debye model is fitted below the quarter point of the
spectrum and extrapolated to the `3 N` modes of the cell, so for a material
whose optical branches carry much of the DOS the cut off can exceed the highest
calculated frequency; the value is then a thermodynamic Debye temperature
rather than the low temperature calorimetric one. `--pdos` reports the DOS projected onto every atom and
Cartesian direction as one labelled record per projection, and plots the
projections against the total DOS. `--irreps` resolves the space-group
irreducible representation of each Gamma point mode by its Mulliken symbol,
using `--symprec` as the symmetry tolerance, and plots the modes labelled with
their symbols. `--irreps` needs a structure whose symmetry can be found, so it
fails with an explicit message when the tolerance does not match the geometry.
Phonopy leaves the symbol unset for point groups whose character table it
cannot index unequivocally, such as the `-3m` of a diamond-like primitive
cell; those modes are reported by their dimension and point group instead,
for example `3D (-3m)`. `--debye` warns and omits the value when the fit does
not converge, which happens when the DOS has no Debye-like low-frequency
region.

The projections are summed on the same mesh as the total DOS, and the mesh is
recorded in the report so that a reader can tell which sampling produced them.
Note that without Born effective charges the Gamma point longitudinal optical
modes carry no non-analytical correction, so a polar material is described
without its LO-TO splitting.

Polar materials need the non-analytical correction, which the long range
Coulomb field of a longitudinal optical vibration adds. Pass the Born effective
charges and the dielectric tensor, both as JSON, and the limit of the q to zero
is taken along a direction:

```text
abacustools workflow phonon postprocess -j JOB --irreps \
  --dielectric "[2.34,0,0,0,2.34,0,0,0,2.34]" \
  --born "[[[1.12,0,0],[0,1.12,0],[0,0,1.12]],[[-1.12,0,0],[0,-1.12,0],[0,0,-1.12]]]"
```

`--dielectric` accepts a scalar, three diagonal values, a flat nine value matrix
or a 3x3 matrix, and `--born` holds one 3x3 tensor per atom of the reference
cell in its atom order. The charges can also be read from the `bec_results.json`
that `workflow bec` writes, with `--bec-results`, which avoids transcribing
tensors by hand: that file already stores them with the rows as the displacement
directions and the columns as the Cartesian polarization directions, which is
the layout the correction expects. In the same way `--dielectric-results` reads
the dielectric tensor of the `dielectric_results.json` that `workflow
dielectric` writes, and the two files together describe a polar material
without transcribing a single number. Born charges and dielectric tensor must
both be given; either one alone is refused, and so is a dielectric tensor that
is given both as JSON and as a results file. The correction is applied to the dispersion, to the
total and projected DOS and to the thermal properties, because the mesh takes
the limit with the direction of each of its own q points. The Gamma point modes
need an explicit direction, which `--nac-direction` sets and which defaults to
the first lattice vector; without it the longitudinal mode keeps the transverse
frequency and the two stay degenerate, so the correction would be reported but
invisible. The report records the tensors and the direction that was used.

The largest frequency of the spectrum is taken over the dispersion rather than
over the commensurate points of the supercell, because a polar material reaches
it in the longitudinal optical mode at Gamma, which the supercell does not
carry. Each Gamma point mode is labelled `LO`, `TO` or `acoustic`: the
longitudinal one is the mode the correction moves, which distinguishes it from
an optical mode that is merely non-degenerate in a low symmetry crystal.

The displaced calculations live in `disp-NNNN`, numbered by their index in the
Phonopy displacement dataset, and `workflow_phonon.json` records both that
index and the displaced atom of every task. Postprocessing therefore maps each
force set onto its displacement through the manifest rather than through the
order of the task list, and refuses a manifest whose task list and displacement
entries disagree. A displaced supercell is written in the Phonopy atom order,
because `AbacusSTRU.supercell` orders atoms by lattice point and mixing the two
orders would attach every force to the wrong atom.

The mode Grueneisen parameters, which measure how the frequency of each mode
follows the volume, come from the same displaced calculations at three volumes.
The prepare stage writes three ordinary phonon workflows, one per volume, whose
lattice vectors are scaled by `(1 ± strain) ** (1/3)` so that their volume
differs from the reference one by `±strain`:

```text
abacustools workflow gruneisen prepare -j JOB --strain 0.01 --supercell 4 4 4
```

Submit and postprocess the displaced calculations of the three volumes exactly
like a plain phonon workflow, then average them:

```text
abacustools workflow gruneisen postprocess -j JOB
```

`gruneisen_mesh.yaml` and `gruneisen_band.yaml` hold the phonopy mesh and band
results with their plots, and `gruneisen_results.json` adds the q weighted mean,
range and count of the mode parameters and the thermodynamic parameter
`gamma(T) = sum(C_v gamma) / sum(C_v)` at `--temperature`, together with a curve
over `--tmin/--tmax/--tstep`. Every volume has to be computed with the same
supercell, k mesh and displacement step, because the three force constant fits
are compared with each other, and the strain has to stay where the central
difference of the frequencies is meaningful, which is why `--strain` is limited
to 0.1 to 5 percent. The zero frequency translations at Gamma carry no parameter
at all; modes whose parameter comes out as a non-finite number are left out of
the averages and counted in the report. The non-analytical correction is not
applied, since its parameters would have to be computed for every volume as
well, and it only affects the longitudinal optical mode at Gamma.

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

`--gaussian-log [FILE]` writes a fake Gaussian frequency output next to the
results (`gaussian_fake.log` by default), in the spirit of OfakeG and
CP2KfakeG, so that GaussView can open the file and animate the modes:

```text
abacustools workflow vibration postprocess -j JOB --gaussian-log
abacustools workflow vibration postprocess -j JOB --gaussian-log modes.log --no-cell
```

Because an ABACUS structure is periodic, the geometry follows Gaussian's
periodic convention: an `Input orientation` block whose last three centers are
the translation vectors of the cell, written as atomic-number `-2` pseudo-atoms
and followed by the `Lengths of translation vectors` and `Angles of translation
vectors` lines, which is how GaussView reads the unit cell. `--no-cell` leaves
the cell out and writes a plain `Standard orientation` block instead. The
frequency block carries the signed frequencies in `cm^-1`, the reduced masses
in `amu`, the force constants in `mDyne/Angstrom` and the Cartesian normal
coordinates of every atom. The IR intensities are written as zero, because the
workflow does not compute the dipole derivatives, and the thermal section is
filled from the zero-point energy and the thermochemistry of the results.

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
writes the work-function results and potential profile. Both branch names of
that cube are read, `ElecStaticPot.cube` in the LTS branch and `potes.cube` in
develop, together with the `pot_es.cube` that the develop manual mentions:

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

The clamped-ion (electronic) dielectric tensor `epsilon_inf` is the other half
of a polar material's non-analytical correction, and ABACUS does not write it.
It is obtained by a tight-binding Kubo-Greenwood sum over the Hamiltonian,
overlap and position matrices of an LCAO calculation, which the `pyatb` package
evaluates on a dense Brillouin zone grid. The prepare stage generates the one
self consistent calculation that has to be rerun for those matrices to appear:

```text
abacustools workflow dielectric prepare -j JOB
```

The generated `dielectric` directory is the source job with `out_mat_hs2`,
`out_mat_r` and `symmetry 0`, and `workflow_dielectric.json` records the task
and the keywords that were switched on, as every other workflow manifest does.
Run it, bring the `OUT.*` matrices back, and postprocess them:

```text
abacustools workflow dielectric postprocess -j JOB
```

The tensor is written to `dielectric_results.json` under `JOB` together with
the dense grid, the photon energy window, the spin channels, the occupied band
count and Fermi energy of the source calculation and the `pyatb` version, so
the number can be traced back to what produced it. `--grid`, `--omega`,
`--domega` and `--eta` set the sum, and `--workdir` moves the pyatb working
directory that holds the copied matrices and its own output. The sum runs
locally, on the machine that holds the matrices and the `abacustools`
environment, which needs the optional `pyatb` package and an MPI runtime:
`pip install 'abacustools[pyatb]'` and a conda MPI such as `mpich` provide both,
and `--pyatb-command 'mpirun -np 4 pyatb'` runs the sum out of process on
several ranks instead of in process.

The occupation of the sum is pinned to the `occupied bands` count that ABACUS
autosets and prints in its running log, not to a Fermi level. For an insulator
any level inside the gap gives the same occupation, ABACUS places its `EFERMI`
at one such value, and the transition energies the sum is built from are
differences that do not depend on the energy reference at all; the Fermi energy
is still written into the pyatb input, because pyatb asks for it, but it does
not enter the result. The photon energy window has to start at zero, since its
first row is the static limit, and to reach well above the band gap, since the
sum covers every transition: a window that sits inside the gap captures no
transition and leaves an empty spectrum, and for NaCl a 20 eV window gives a
tensor about one percent low. `--omega 0 80` is the default and a window that
starts above zero or ends below 40 eV is reported as a warning. The dense grid
of the sum is what the tensor converges with; the k mesh of the source SCF only
has to converge the density, and for NaCl 20x20x20 already agrees with
50x50x50 to five digits.

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

The tensor is a polar third-rank tensor, so the point group of the reference
cell fixes how many of its 18 components are independent: one for `-43m`,
three for `6mm`, four for `3m`, none for a centrosymmetric crystal. The
preparation stage records the point group and the number of independent
components in `workflow_piezoelectric.json`, and `--strains independent`
prepares only the strain modes those components need - three of the six for a
wurtzite cell. The postprocessing stage symmetrises the fitted tensor with the
point group, which removes the components the symmetry forbids and enforces
the relations between the ones that survive:

```text
abacustools workflow piezoelectric prepare -j JOB --strains independent
abacustools workflow piezoelectric postprocess -j JOB --fit independent
```

Both the fitted tensor, the symmetrised tensor, the largest change the
symmetrisation made and the independent components are written to
`piezoelectric_results.json` under `JOB`; `--no-symmetrize` keeps the raw
components.

The strain is a Cartesian deformation of the lattice vectors with the
fractional coordinates held fixed, so the six modes are the strain tensor
components of the IEEE convention: a shear of nominal size `s` puts `s/2` in
each off-diagonal element, and `S_4 = 2 eps_yz` equals `s`. The reported shear
columns are therefore `dP/dS` with the same meaning as in the literature and in
DFPT codes, and a requested one percent shear is a one percent shear in the
same sense the elastic workflow uses. `--relax` adds one self consistent
relaxation of the ionic positions at each strained cell, which is what turns
the clamped-ion tensor into the relaxed-ion one; it respects the `force_thr_ev`
of the source `INPUT`, and tightening that value (the ABACUS default is 0.01
eV/Angstrom) matters because the internal strain it produces is exactly the
difference between the two.

ABACUS accepts `use_k_continuity` only for plane wave calculations that are not
a non self consistent run, and the Berry-phase steps of this workflow are
exactly such a run, so the option is off by default. `--use-k-continuity` is
kept for versions that lift the restriction; on LTSv3.10.1 it makes every
generated calculation stop with `use_k_continuity only works for PW basis`.

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

Magnetic exchange coupling constants can be calculated with the four-state
method, which maps the total energy of a magnetic pair onto the Heisenberg
form. Prepare one reference calculation together with three tilted
magnetization configurations per tilt angle: only the first moment tilted,
only the second one tilted, and both moments tilted by half the angle, so all
three share the same pair angle. Pairs are given on the command line or in a
`magj.txt` file that holds one `LABEL1 INDEX1 LABEL2 INDEX2` line per pair,
where the indices are one-based within the atoms of that label:

```text
abacustools workflow exchange prepare -j JOB --pair Fe 1 Fe 2
abacustools workflow exchange prepare -j JOB --step 10 --number 5
abacustools workflow exchange prepare -j JOB -f pairs.txt
abacustools workflow exchange postprocess -j JOB
```

The `exchange` workflow is also available as `magj`, the name of the
`abacus-test` model it follows. The input job needs noncollinear magnetism
(`nspin 4`), and both atoms of a pair need a magnetic moment: a scalar moment
counts as a moment along `z`, a vector `mag x y z` keeps its direction. The
generated `STRU` files keep the magnitudes of the reference moments and tilt
them in the plane of the reference pair. `--step` is the tilt step in degrees
and `--number` the number of tilt angles, so the defaults of 1 degree and five
steps test 1 to 5 degrees, as the reference implementation does; larger tilts
raise the energy difference that is fitted and therefore give a better
conditioned result. Repeating `--pair` calculates several pairs in one go, and
a pair that is requested in both orders, or twice, is calculated once because
the two atoms are symmetric in the method.

Postprocessing combines the four energies into

```text
dE = (E_both - E_atom1) - (E_atom2 - E_original)
```

and fits `dE = J (1 - cos(theta))`, so the slope `J` is positive for
antiferromagnetic coupling. The fit is reported per pair together with its
coefficient of determination and its residual, which show whether the energy
difference follows the Heisenberg form. Results are written to
`exchange_results.json`, with the energies, angles and four-state differences
of every point, and the fit is plotted to `exchange_fit.png`. The fit uses the
angles of the configurations that were generated, so a run whose moments
rotate away from the prescribed directions is best checked with
`postprocess result -p atom_orb_mag` before its coupling constant is used.
Postprocessing stops when a calculation did not reach `scf_thr`; some magnetic
configurations converge their energy while the density error keeps oscillating,
so `--allow-unconverged` fits them anyway and lists the affected calculations
in `unconverged_tasks` next to a `converged` flag on every fitted point.
Because the moments keep their reference magnitudes, `J` is the coefficient of
the unit directions of the two moments times the product of their magnitudes;
divide by that product to obtain a coupling constant per unit moment.
Tilted configurations converge more slowly than the reference one, so the
per-point `delta_energy_ev` and `x` of the JSON are worth a look: every tilt
should give the same slope, and a tilt that does not is a calculation that has
not reached the configuration it was given. The generated calculations live
below `magj/`, with one directory per tilt angle and path, and the reference
calculation in `magj/original`.

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

## Structure databases

`abacustools database` searches and downloads crystal structures from the
materials databases that are open to everyone, and writes them as ABACUS or
common structure files:

```text
abacustools database list                     # what can be queried, and how
abacustools database providers                # the OPTIMADE providers behind -d optimade
abacustools database search -d cod --formula Fe2O3 --limit 5
abacustools database download -d aflow aflow:608f86003961ee94
```

Two access routes cover the databases that need no subscription. The first is
the Materials Project client (`mp-api`, an API key, the full set of summary
fields), registered as `mp`. The second is
[OPTIMADE](https://www.optimade.org), a REST protocol spoken by most other open
structure databases and implemented once here: every deployment is a database
name of its own, and `-d optimade` queries all of them and stops as soon as the
answer is full.

`ABACUSTOOLS_DATABASE` sets the database used when `--database` is omitted;
without it the default is `mp`.

### Available databases

| database | content | access route |
| --- | --- | --- |
| `mp` | Materials Project: DFT energies, stability, structures | `mp-api`, API key |
| `mp-optimade` | the public OPTIMADE endpoint of the Materials Project | OPTIMADE |
| `aflow` | AFLOW: calculated alloys and compounds | OPTIMADE |
| `oqmd` | OQMD: formation energies and thermodynamic stability | OPTIMADE |
| `nomad` | NOMAD: parsed ab initio calculations | OPTIMADE |
| `jarvis` | NIST JARVIS-DFT: optoelectronic and elastic data | OPTIMADE |
| `cod` | Crystallography Open Database: experimental structures | OPTIMADE |
| `tcod` | Theoretical Crystallography Open Database | OPTIMADE |
| `alexandria` | Alexandria materials database (PBE+SOL) | OPTIMADE |
| `c2db` | Computational 2D Materials Database (DTU), with its computed data | query table + OPTIMADE |
| `c2db-optimade` | the same structures over OPTIMADE only, without the computed data | OPTIMADE |
| `mc3d`, `mc2d` | Materials Cloud three- and two-dimensional crystals | OPTIMADE |
| `twodmatpedia` | 2DMatPedia: 2D materials exfoliated from the Materials Project | OPTIMADE |
| `matterverse` | Matterverse: machine-learning property predictions | OPTIMADE |
| `odbx` | Open Database of Xtals | OPTIMADE |
| `mpds` | Materials Platform for Data Science | OPTIMADE, token |
| `optimade` | every catalogued provider at once | OPTIMADE |

`abacustools database list` shows which databases are ready, which need an API
key, and which selectors each one accepts. Databases reached over OPTIMADE
accept `--formula`, `--chemsys`, `--elements` and `--id`; the Materials Project
also accepts `--stable`, `--theoretical` and `--fields`, and C2DB adds
`--where` for its own property expressions and `--show` for extra columns (see
below). Asking a database for a selector it does not know is an error rather
than a silent partial match.

The OPTIMADE catalogue follows the official index at
`https://providers.optimade.org/providers.json`; `abacustools database
providers --refresh` prints the live list, and `--base-url` sends a query to an
OPTIMADE endpoint that the catalogue does not contain. Providers differ in what
they publish: `cod` and `tcod` report cell parameters but no atomic
coordinates, so they answer searches while a download of one of their entries
reports that there is no structure to write; `c2db-optimade` ignores filters on
the entry id and does not publish the computed data, which is why the `c2db`
database reads the query table of the C2DB web application instead.

### C2DB computed data

C2DB stores much more than the geometry: PBE, HSE06 and G0W0 band gaps, the
energy above the convex hull, the heat of formation, effective masses, elastic
and piezoelectric constants, magnetic states, optical properties and so on.
`abacustools database fields -d c2db` lists all 88 keys with their units, and
`--where` filters on them with the expression language of the C2DB search page:

```text
abacustools database fields -d c2db
abacustools database search -d c2db --formula MoS2 --limit 5
abacustools database search -d c2db --elements Mo S --where 'gap>1.5' --limit 5
abacustools database search -d c2db --where 'is_magnetic=True' --where 'ehull<0.05'
abacustools database search -d c2db --formula MoS2 --show gap_hse,emass_cbm
```

A `--where` expression compares one key, as in `gap>1.5`, `ehull<0.05`,
`xc=PBE` or `nspecies=3`; several expressions and the standard selectors are
combined with "and", `|` combines alternatives and `~` negates a term, exactly
as on the search page. `--show` adds the named keys as columns, and every
search record also carries them under `extra` in `--json`. Entries are
identified by their C2DB uid, such as `1MoS2-1`:

```text
abacustools database download -d c2db 1MoS2-1 --format cif
abacustools database download -d c2db 1MoS2-1 --json
```

A download joins the property row of the query table with the geometry of the
OPTIMADE endpoint, so the JSON record reports the formula, the site count, the
PBE gap, the energy above the hull, and the tabulated properties.

### Materials Project

The `mp` command family is the short spelling of `database --database mp`, kept
so that existing scripts keep working. It needs the optional `mp-api` client
and an API key:

```bash
pip install 'abacustools[mp]'
export MP_API_KEY="your_key_here"   # https://materialsproject.org/api
```

Searches need at least one of `--formula`, `--chemsys`, `--elements`, or
`--id` (also spelled `--material-id`):

```text
abacustools mp search --formula Fe2O3 --limit 10
abacustools mp search --chemsys Li-Fe-O --stable --limit 5 --json
abacustools database search -d mp --elements Li Fe O --output li-fe-o.json
abacustools database search -d cod --formula Fe2O3 --limit 5
```

The table reports the database, the identifier, the formula, the chemical
system, the number of sites, the energy above the convex hull, the band gap,
stability, and whether the entry is theoretical. `--json` prints the same
records as JSON and `--output` also writes them to a file. The command returns
a non-zero exit status when nothing matches the query.

Downloading writes one directory per entry, which can be used as an ABACUS job
directory directly:

```text
abacustools mp download mp-149
abacustools mp download mp-149 mp-22862 --output structures --format poscar
abacustools database download -d cod 1000000 --format cif
abacustools database download -d aflow aflow:608f86003961ee94 --group-by-database
```

Every structure goes to `OUTPUT/<id>/`, named `STRU`, `POSCAR`,
`structure.cif`, `structure.xyz`, `structure.extxyz`, or `structure.xsf`
according to `--format`; `--group-by-database` adds the database name as
another directory level, which keeps identifiers of different databases apart.
Downloaded structures carry no pseudopotential or numerical-orbital
information, so the `ATOMIC_SPECIES` files required by ABACUS still have to be
filled in, for example with the library settings described above or
`abacustools file stru`.

Each JSON record holds `database`, `id`, `formula`, `chemsys`, `nsites`,
`volume`, `energy_above_hull`, `band_gap`, `is_stable`, `theoretical`, and
`extra`, where `extra` keeps the provider-specific values of the entry.

### Python API

Every database is reached through the same interface, and each call accepts a
pre-constructed client or transport so that no connection is made when one is
supplied:

```python
from pathlib import Path
from abacustools.integrations.databases import (
    DatabaseQuery,
    get_database,
    structure_path,
    write_structure,
)

database = get_database("cod")
for summary in database.search(DatabaseQuery(formula="Fe2O3", limit=5)):
    print(summary.database, summary.identifier, summary.formula, summary.chemsys)

structure = database.fetch(summary.identifier)
write_structure(
    structure,
    structure_path(Path("structures"), structure.identifier, database="cod"),
)
```

`get_database(name)` returns the adapter of any registered database, and
`databases()`, `database_names()`, `describe_databases()` and
`default_database()` describe the registry. The Materials Project adapter is
also available on its own, as before:

```python
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
