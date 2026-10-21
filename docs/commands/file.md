# File commands

[Documentation home](../index.md)

The `file` family converts, inspects, and edits ABACUS file formats.

## file input

`abacustools file input INPUT` reads and rewrites one ABACUS `INPUT` file. With
`--set PARAMETER VALUE` it updates an existing keyword or adds a new one before
writing the file back; without `--set` it round-trips the parsed input. The
command modifies the file in place.

```text
abacustools file input INPUT
abacustools file input INPUT --set ecutwfc 80
abacustools file input --help
```

## file stru

`file stru` converts between ABACUS `STRU` and common structure formats
supported by ASE, including POSCAR/VASP, CIF, XYZ, EXTXYZ, and XSF. The input
and output formats are inferred from filenames, or can be set explicitly with
`--input-format` and `--output-format`:

```text
abacustools file stru STRU POSCAR
abacustools file stru INPUT OUTPUT
abacustools file stru POSCAR STRU
abacustools file stru structure.xyz STRU --cell 10 0 0 0 10 0 0 0 10
```

## file struinfo

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

## file kpt

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

## file interface

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

## file editstru

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

## file traj

`file traj` converts a trajectory between formats with the same machinery:

```text
abacustools file traj trajectory.extxyz trajectory.xyz
abacustools file traj dump.traj dump.extxyz --stride 10
```

## Structure conversion notes

Conversions issue a `StructureConversionWarning` when ABACUS-specific data
such as pseudopotential/orbital filenames, spin settings, velocities,
movement constraints, or `NUMERICAL_DESCRIPTOR` cannot be represented by the
target format. Trajectories are converted with `file traj`, which reads and
writes any multi-frame format that ASE supports, so an `extxyz` file becomes
an ASE `.traj`, a plain `.xyz` or an `.xsf` for a viewer of choice. Standard XYZ files do not contain a periodic cell; provide
`--cell` when converting one to `STRU`.
