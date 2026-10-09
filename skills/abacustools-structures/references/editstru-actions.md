# file editstru actions

Common to every action: a positional `STRUCTURE`, a required `-o/--output`
(except `all-slabs`, which writes several prefixed files), `--input-format`/
`--output-format` (inferred from the file names when omitted), `--override` to
replace an existing output, and `--json` for the summary. Without `--override`
an existing output is left alone, which makes these commands safe to plan
before running them.

| action | purpose | key options |
| --- | --- | --- |
| `supercell` | Replicate along the lattice vectors | `-n/--repeat A B C` (required) |
| `vacuum` | Extend one lattice vector by empty space | `-t/--thickness ANG`, `--direction a|b|c|x|y|z` (default `c`), `--center` |
| `slab` | Cut a surface with `ase.build.surface` | `--miller H K L` (default `1 0 0`), `--layers N` (3), `--surface-supercell A B` (1 1), `--vacuum ANG` (15), `--vacuum-direction a|b|c`, `--fix [FRACTION]` (0.5) |
| `all-slabs` | Every symmetrically distinct termination of one Miller set | `--miller H K L` (required), `--output-prefix PREFIX` (default `slab`), `--min-slab-size`, `--min-vacuum-size`, `--in-unit-planes`, `--center-slab`/`--no-center-slab`, `--symmetrize`, `--repair`, `--tol`, `--max-broken-bonds` |
| `select` | Keep, or with `--remove` drop, matching atoms | `--indices`, `--elements`, `--coords MIN MAX` with `--direction`, `--direct` |
| `substitute` | Replace matching atoms with another element and pick its files | `--element` (required), plus the selection options; `--label`, `--library`, `--variant`, `--pp`, `--orb`, `--basis auto|lcao|pw`, `--keep-moments` |
| `fix` | Set movement constraints of the selection | same selection options plus `--move x y z` (axes still free; fully fixed by default) and `--free-others` |
| `direct` / `cartesian` | Rewrite the position block in the other representation | none beyond the common ones |
| `primitive` | Reduce to the primitive cell | `--symprec ANG` (1e-5), `--angle-tolerance DEG` (5) |
| `conventional` | Return the conventional cell | `--symprec`, `--angle-tolerance` |
| `standardize` | Rewrite in the standard setting | `--to-primitive`, `--no-idealize`, `--symprec`, `--angle-tolerance` |
| `symmetrize` | Remove numerical noise and enforce the found symmetry | `--symprec` (1e-5), `--angle-tolerance`, `--keep-cell` |

## Invariants

- Selection filters are additive: an atom must match every filter given.
  `--indices` are one-based, as in every other abacustools command.
- `--coords` is an inclusive window along `--direction`, Cartesian unless
  `--direct` asks for fractional coordinates.
- Pseudopotential/orbital names, magnetic moments, velocities and constraints
  are carried through the actions that do not touch them, so `direct` followed
  by `cartesian` returns the original structure.
- `slab` re-applies pseudopotential, orbital and magnetic data per element after
  the ASE cut, because the in-plane rebuild loses the mapping; constraints are
  not carried over, so use `--fix` or the `fix` action. Empty atoms are
  rejected because ASE cannot represent them.
- `--fix FRACTION` on `slab` measures from the slab itself, so the default 0.5
  fixes the bottom half wherever the slab sits in the cell.
- `substitute` inherits the pseudopotential/orbital of an element already
  present in the structure, otherwise it resolves them from the resource
  library (`--library`/`--variant`) with `--pp`/`--orb` as explicit overrides;
  `--basis auto` follows whether the input structure is LCAO or plane wave.
- `symmetrize` keeps the cell setting, the number and order of atoms and every
  atom attribute, which distinguishes it from `standardize` and `conventional`;
  its `--symprec` is the size of the deviation that counts as noise, so raise it
  above the errors to remove.
- `all-slabs` writes `PREFIX_0.<ext>`, `PREFIX_1.<ext>`, ... and refuses to
  overwrite without `--override`.
