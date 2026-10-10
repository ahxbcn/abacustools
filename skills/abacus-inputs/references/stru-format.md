# STRU file reference

Source: ABACUS documentation, `docs/advanced/input_files/stru.md`
(https://abacus.deepmodeling.com). Sections start with a keyword line; each
keyword below is a section.

## ATOMIC_SPECIES

One line per element label:

```text
label  mass  pseudopotential_file  [pseudo_type]
```

- `label` is what `ATOMIC_POSITIONS` refers to; `Si1`/`Si2` are legal and are
  the way to give one element two different pseudopotentials or DFT+U values.
- `mass` is only used by molecular dynamics; it does not matter for
  electronic-structure runs, but a dummy value such as `1.0` is still required.
- The pseudopotential path is resolved against the working directory unless an
  explicit path is given.
- `pseudo_type` is one of `upf`, `upf201`, `vwr`, `blps`, `auto` (default
  `auto`). It can be omitted entirely.
- Every pseudopotential of the structure must use the same XC functional. If
  they do not, `dft_functional` in `INPUT` must state the functional
  explicitly.

## NUMERICAL_ORBITAL

One line per element label, needed only for `basis_type lcao`; ignored for
plane-wave runs. The file name alone is enough when the file sits in the
working directory. See the `abacus-basis` skill for how to choose these files.

## LATTICE_CONSTANT and LATTICE_VECTORS

- `LATTICE_CONSTANT` is in **Bohr**. Output structures carry a `# in Bohr`
  comment on that line.
- `LATTICE_VECTORS` is three lines of three numbers, **in units of
  `LATTICE_CONSTANT`**: the real cell is `LATTICE_VECTORS * LATTICE_CONSTANT`.
  Output structures mark the section `# in units of lat0`.

## Bravais shortcut: `latname` + LATTICE_PARAMETERS

When `INPUT` sets `latname` (`sc`, `fcc`, `bcc`, `hexagonal`, `trigonal`, `st`,
`bct`, `so`, `baco`, `fco`, `bco`, `sm`, `bacm`, `triclinic`), omit
`LATTICE_VECTORS`. `sc`/`fcc`/`bcc` need no `LATTICE_PARAMETERS`; the others
need one line with:

| latname | parameters |
| --- | --- |
| `hexagonal`, `st`, `bct` | `c/a` |
| `trigonal` | `cos(gamma)` |
| `so`, `baco`, `fco`, `bco` | `b/a`, `c/a` |
| `sm`, `bacm` | `b/a`, `c/a`, `cos(ab)` |
| `triclinic` | `b/a`, `c/a`, `cos(ab)`, `cos(ac)`, `cos(bc)` |

## ATOMIC_POSITIONS

Header line selects the coordinate mode:

| mode | meaning |
| --- | --- |
| `Direct` | fractional coordinates |
| `Cartesian` | Cartesian, in units of `LATTICE_CONSTANT` (Bohr by default) |
| `Cartesian_au` | Cartesian in Bohr, equivalent to `Cartesian` with `LATTICE_CONSTANT 1.0` |
| `Cartesian_angstrom` | Cartesian in Angstrom, equivalent to `Cartesian` with `LATTICE_CONSTANT 1.889726125457828` |
| `Cartesian_angstrom_center_xy` (also `_xz`, `_yz`, `_xyz`) | as above, with the direct coordinate `(0.5, 0.5, 0.0)` (etc.) as origin - convenient for slabs |

Then, per element label, three lines: the label, the **default initial magnetic
moment** for that label (Bohr magneton; it is an ordinary moment, not a spin
flag), and the number of atoms of that label. Every atom then gets one line with
three coordinates followed by optional keyword suffixes.

Per-atom suffixes (any order, each at most once):

| keyword | meaning |
| --- | --- |
| `m 0/1 0/1 0/1` | movement allowed per Cartesian direction; `0 0 0` fixes the atom, `1 1 1` frees it. No keyword means free. Used by `relax`/`cell-relax`. |
| `v` / `vel` / `velocity x y z` | initial velocity, for restarting MD |
| `mag` / `magmom` | collinear: one value; noncollinear: three Cartesian components, or a norm together with `angle1`/`angle2` |
| `angle1`, `angle2` | polar angle from `z` and azimuth from `x` in the xy plane, in degrees |
| `lambda` | Lagrange multiplier for spin-constrained DFT, one or three values in eV (converted to Ry internally; needs `sc_mag_switch` in INPUT) |
| `sc` | spin-constraint target magnetization, one or three values |

Magnetization defaults: if no atom of a magnetic run sets a finite moment,
ABACUS autosets `1.0` for `nspin 2` and `(1, 1, 1)` for `nspin 4`. The autoset
is disabled as soon as any single atom specifies a finite value, so a partially
specified structure is not completed for you.

## Mistakes that silently change the structure or the physics

- Mixing `Direct` and `Cartesian` lines inside one `ATOMIC_POSITIONS` block.
- Giving the element line a moment that does not match the intended spin
  state: `1.0` is one Bohr magneton, not "spin up".
- Specifying both `mag x y z` and `angle1`/`angle2` for the same atom.
- Writing a number of coordinate lines that disagrees with the element header
  counts; ABACUS reads the declared count, not the file length.
- Setting `m` only for a subset of atoms in a slab relaxation, which leaves the
  rest free.
- Assuming a moment in the element header overrides a per-atom `mag`; it is the
  other way round.
- Using Angstrom numbers with a `LATTICE_CONSTANT` left at `1.0`: the cell comes
  out 1.89 times too large. See
  [units-and-conventions.md](units-and-conventions.md).

## What ABACUS writes back

Structure files written by ABACUS (for example `STRU_ION_D`, `STRU_MD_*`, or
the `STRU` next to a restart) annotate the units: the `LATTICE_CONSTANT` value
carries a trailing `# in Bohr` comment and the `LATTICE_VECTORS` header is
marked `# in units of lat0`. Those files also use `m`, `v` and `mag` suffixes
to carry the current constraints, velocities, and moments, so they can be fed
back as the next run's input.

ABACUS does not currently support PAW (projector-augmented wave)
pseudopotentials, so a PAW dataset is not a usable alternative to the
`ATOMIC_SPECIES` pseudopotential entry; only norm-conserving and ultrasoft files
are usable.
