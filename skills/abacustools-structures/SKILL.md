---
name: abacustools-structures
description: Read, convert, inspect, edit, and combine crystal structure files with the abacustools file family, and search or download structures from materials databases. Handles ABACUS STRU, POSCAR, CIF, XYZ, EXTXYZ, XSF, supercells, slabs, vacuum layers, heterojunction interfaces, element selection and substitution, movement constraints, symmetry and coordination reports, k-point meshes and band paths, and database queries. Use when the task is about structure geometry or obtaining a structure, not about ABACUS inputs, running jobs, or results analysis.
---

# Structure files with abacustools

## CLI conventions that bite

Every `abacustools` invocation prints a 9-line ASCII banner to stdout before
anything else, including before `--json` output. The JSON payload starts at
line 10. Never pipe the raw stdout of an `--json` command into a JSON parser;
strip the banner first, for example `abacustools ... --json | tail -n +10`.
Progress and error messages can follow the JSON on stderr; exit code `2` is an
argument error from argparse, `0` is success.

Formats are inferred from file names; `--input-format`/`--output-format` force
them. Conversions drop data the target cannot hold and say so through
`StructureConversionWarning` (pseudopotential/orbital names, spin settings,
velocities, movement constraints, `NUMERICAL_DESCRIPTOR`).

## Convert and inspect

```text
abacustools file stru STRU POSCAR
abacustools file stru POSCAR STRU
abacustools file stru structure.xyz STRU --cell 10 0 0 0 10 0 0 0 10
abacustools file stru STRU OUT --empty2x
abacustools file info STRU --json
abacustools file info slab.STRU --coordination --json
abacustools file info STRU --symprec 1e-3 --angle-tolerance 5
abacustools file info *.STRU --json          # one summary object per file
abacustools file traj dump.traj dump.extxyz --stride 10
abacustools file input INPUT --set scf_thr 1e-8   # parse or rewrite a keyword
abacustools file kpt KPT --structure STRU
abacustools file kpt --structure STRU --spacing 0.03 -o KPT.scf
abacustools file kpt --structure STRU --path --npoints 20 -o KPT.band
```

`file info` is the one command to reach for before any calculation: it reports
cell parameters, volume, density, element/label counts, formula unit and
prototype, space group, crystal system, point group with Schoenflies symbol,
Bravais lattice and Pearson symbol, inversion symmetry, polarity, the symmetry
tolerances used, Wyckoff positions with multiplicity and site symmetry,
symmetry-inequivalent atoms, magnetic space group (BNS/UNI and type), magnetic
ordering, layer group for a slab, dimensionality (3D/2D/1D/0D from the vacuum),
coordination numbers and ChemEnv symbols, and the pseudopotential/orbital file
of every label. Fields that a structure cannot provide (no cell, failed
symmetry search) are reported as unavailable rather than guessed.

Several structures can be listed at once, which reports only the fields that
make them comparable: the file, formula, atom count, space group with its
number, crystal system and cell parameters. `--summary` asks for that listing
for a single structure as well, and with several inputs or `--summary`,
`--json` returns one summary object per structure instead of a full report.
The listing is much cheaper because it skips Wyckoff positions, the point
group, the Bravais lattice, the magnetic and layer symmetry, the
dimensionality, the coordination and the per-atom table.

`--coordination` follows the dimensionality by default: ChemEnv for bulk,
CrystalNN for slabs/wires/molecules because Voronoi is undefined in vacuum.
Force a method with `--coordination crystalnn|chemenv|voronoi|minimum-distance`;
an explicit method that cannot handle the structure is reported as unavailable
instead of being substituted. `--min-vacuum` sets the empty span in Angstrom (5
by default) that makes a lattice direction non-periodic, and
`--layer-direction a|b|c` pins the aperiodic direction of a slab.

`file kpt` inspects a KPT file, validates it and, with `--structure`, reports
the k-spacing the mesh realizes. With `--output` it instead writes one:
`--mesh 9 9 9` (with `--model gamma|mp`), a target `--spacing 0.03` plus
`--structure`, or `--path` plus `--structure` for the line-mode seekpath path
with `--npoints` points per segment. The path follows the dimensionality
(`--path-mode auto|bulk|slab|wire`), with `--min-vacuum`, `--symprec` and
`--angle-tolerance` controlling the choice. An existing output needs
`--override`.

## Edit structures

Every action reads a structure and writes a separate output; an existing output
is replaced only with `--override`, which makes these commands safe to plan
without side effects. Atom attributes that the action does not touch
(pseudopotentials, orbitals, magnetic moments, constraints) travel along.

```text
abacustools file editstru supercell STRU -o SUPER -n 2 2 1
abacustools file editstru vacuum    STRU -o SLAB -t 15 --direction c --center
abacustools file editstru slab      STRU -o SLAB --miller 1 1 0 --layers 4 --vacuum 15 --fix
abacustools file editstru all-slabs STRU --miller 1 1 0 --output-prefix SLAB
abacustools file editstru select    STRU -o SUB  --elements Si O
abacustools file editstru select    STRU -o FREE --indices 1 3 --remove
abacustools file editstru substitute STRU -o DOPED --element Fe --elements Si
abacustools file editstru fix       STRU -o FIXED --elements O --move x y --free-others
abacustools file editstru direct    STRU -o DIRECT
abacustools file editstru cartesian STRU -o CART
abacustools file editstru primitive STRU -o PRIM
abacustools file editstru conventional STRU -o CONV
abacustools file editstru standardize STRU -o STD --to-primitive
abacustools file editstru symmetrize STRU -o CLEAN --symprec 0.001
```

`--indices` are one-based everywhere in abacustools. Selections are additive
filters (an atom must match every filter given). `slab` places its normal along
`c` unless `--vacuum-direction` says otherwise, rejects empty atoms, and does not
carry movement constraints over, so constrain it with `--fix` or the `fix`
action. `primitive`/`conventional`/`standardize` use spglib and keep the atoms'
element mapping.

`substitute` replaces the selected atoms with `--element`. When the input
already contains that element its pseudopotential/orbital files are reused;
otherwise they come from the configured resource library (`--library`,
`--variant`), and `--pp`/`--orb` override either choice. `--basis auto|lcao|pw`
decides whether an orbital is needed, `--label` names the new species, and
`--keep-moments` keeps the replaced atoms' magnetic moments (cleared
otherwise). The report names every chosen file and the reason.

`direct`/`cartesian` rewrite only the position block, so converting forth and
back returns the original structure. `symmetrize` cleans numerical noise
instead of changing the cell setting: spglib averages positions with their
symmetry images and strains the lattice onto the required metric, keeping the
cell setting, atom order and every atom attribute (unlike `standardize` and
`conventional`). `--symprec` sets which deviations count as noise and
`--keep-cell` idealizes the positions only.

The STRU reader follows the newer ABACUS conventions: `Cartesian_angstrom`,
`Cartesian_au` and `Cartesian_angstrom_center_{xy,xz,yz,xyz}` coordinate modes,
`#` annotations after block keywords, the per-atom force field
(`f`/`force`/`forces`, eV/Angstrom) and the `pp_type` column of
`ATOMIC_SPECIES` are parsed and written back.

## Build an interface

Two structures can be stacked into a heterojunction with pymatgen's coherent
interface builder, which searches the Zur-McGuire lattice matches of the two
surfaces and strains the film onto the substrate:

```text
abacustools file interface FILM SUBSTRATE -o HET --film-miller 0 0 1 --substrate-miller 0 0 1
abacustools file interface FILM SUBSTRATE --list --max-strain 0.03 --max-area 200
abacustools file interface FILM SUBSTRATE -o HET --film-thickness 3 --substrate-thickness 3 --gap 2.5 --vacuum 15
```

`--list` prints the candidate lattice matches without writing anything;
otherwise the best match that satisfies `--max-atoms` is written, with the film
on top of the substrate along `c`. `--film-thickness`/`--substrate-thickness`
size the films in layers (or Angstrom with `--in-angstrom`), and `--gap`,
`--vacuum`, `--termination`, `--max-strain`, `--max-angle` and `--max-area`
control the stack and the search. Pseudopotential and orbital files are
inherited from whichever structure contains the element, with the configured
library as fallback; the report names each file and the reason for choosing it.

## Materials databases

```text
abacustools database list                      # what is ready, which selectors it takes
abacustools database providers --json          # OPTIMADE providers
abacustools database search -d cod --formula Fe2O3 --limit 5 --json
abacustools database search -d c2db --elements Mo S --where 'gap>1.5' --show gap_hse
abacustools database download -d c2db 1MoS2-1 --format cif
abacustools mp search --chemsys Li-Fe-O --stable --limit 5   # = database -d mp
```

`--database` defaults to `ABACUSTOOLS_DATABASE`, else `mp`. OPTIMADE databases
take `--formula`, `--chemsys`, `--elements`, `--id`; MP adds `--stable`,
`--theoretical`, `--fields`; C2DB adds `--where` (its search-page expression
language with `|` or `~`) and `--show`. Asking for a selector a database does
not support is an error, not a partial match. A search with no hit exits
non-zero.

`download` writes one directory per entry, which can serve as an ABACUS job
directory, named by `--format`. Downloaded structures carry no pseudopotential
or orbital information, so the `ATOMIC_SPECIES` block still has to be filled
in, for example with `job prepare --library` or `editstru substitute`.

## References

- [references/editstru-actions.md](references/editstru-actions.md) - every
  `file editstru` action with its options and the invariants it preserves.
- [references/databases.md](references/databases.md) - database registry, MP
  API key setup, C2DB property queries, and the Python API.
