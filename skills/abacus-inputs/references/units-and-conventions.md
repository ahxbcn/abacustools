# Units and the LATTICE_CONSTANT convention

## Units by quantity

| quantity | unit | where |
| --- | --- | --- |
| lengths in `STRU` | Bohr | `LATTICE_CONSTANT`, and `Cartesian` coordinates as multiples of it |
| `LATTICE_VECTORS` | multiples of `LATTICE_CONSTANT` | real cell = `LATTICE_VECTORS * LATTICE_CONSTANT` |
| plane-wave cutoff, smearing width | Ry | `ecutwfc`, `ecutrho`, `smearing_sigma` |
| forces, force thresholds | eV, eV/Angstrom | `force_thr_ev`, force output |
| stress, pressure | kBar | `stress_thr`, monitor output |
| magnetic moments | Bohr magneton | element header and `mag` |
| temperature | K | `md_tfirst`, `md_tlast` |
| k-spacing | 1/Bohr | `kspacing` |

Those units are per-quantity, not per-file: one `STRU` mixes Bohr lengths with
Angstrom-free fractions, and one `INPUT` mixes Ry cutoffs with eV force
thresholds. Output logs print the energy in Ry or eV depending on the branch
and the report, which is why the running log is the place to confirm what a
given number actually is.

## The Angstrom convention

ABACUS reads `LATTICE_CONSTANT` as a Bohr value, so a `STRU` written with
Angstrom numbers and `LATTICE_CONSTANT 1.0` describes a cell 1.89 times too
large. Two ways to avoid that, and they differ:

1. **Set `LATTICE_CONSTANT` to the Bohr-per-Angstrom factor**
   (`1.889726125457828`). `LATTICE_VECTORS` and `Cartesian` coordinates are then
   all Angstrom. This is the common shortcut; it changes the meaning of the whole
   file, so do not mix it with Bohr-valued vectors.
2. **Keep the physical constant and switch the coordinate mode** to
   `ATOMIC_POSITIONS Cartesian_angstrom`. ABACUS documents this as equivalent to
   `Cartesian` with `LATTICE_CONSTANT 1.889726125457828`; the cell vectors keep
   their own units.

ABACUS documents the factor as `1.889726125457828`, while `1 / 0.529177249`
gives `1.8897259885789233`. The two differ in the eighth digit - a relative
difference of about 1e-7, far below any DFT tolerance - but pick one value and
use it in every file of a study so round-trip comparisons stay free of noise.

## How to tell which convention a file uses

- Read the `LATTICE_CONSTANT` line of the `STRU`. A value near `1.0` means
  lengths in the file are Bohr; a value near `1.889726` means the file was
  written in Angstrom.
- Look at the `ATOMIC_POSITIONS` header. `Cartesian_angstrom` fixes the unit of
  the coordinates regardless of the constant, and `Cartesian_au` fixes it to
  Bohr; `Cartesian` inherits whatever `LATTICE_CONSTANT` says.
- Check magnitudes when the constant is absent. ABACUS defaults
  `LATTICE_CONSTANT` to 1 Bohr, so a cell written as `5.43 ...` with no
  constant line is a 5.43 Bohr cell, not a silicon lattice constant.
- On the way out, ABACUS annotates its own structure files: the constant line
  carries `# in Bohr` and the vector block is marked `# in units of lat0`.

## Keeping a deck consistent

Whichever route you take, keep the whole file on one convention. Mixing an
Angstrom-valued `LATTICE_VECTORS` with Bohr-valued `Cartesian` positions, or
copying coordinates from a Bohr deck into an Angstrom one, produces a
geometrically valid but physically wrong structure that ABACUS accepts without
complaint. State the convention in a comment on the `LATTICE_CONSTANT` line so
the next reader - or the next run - does not have to infer it.
