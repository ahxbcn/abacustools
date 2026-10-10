---
name: abacus-inputs
description: Write, review, and explain ABACUS input files - INPUT, STRU, and KPT - including calculation types, the STRU block structure, coordinate modes, magnetic moments, movement constraints, k-point meshes and band paths, and the unit conventions that decide whether lengths are read as Bohr or Angstrom. Use when authoring, reviewing, or debugging an ABACUS input deck, or when a run behaves differently from what the deck appears to say. Use abacus-basis instead for choosing pseudopotential or orbital files.
---

# ABACUS input files

## The three files

- `INPUT` - keywords, one `name value` pair per line under a leading
  `INPUT_PARAMETERS` line. `#` starts a comment. ABACUS documents every keyword
  in its `input-main` reference
  (https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html).
- `STRU` - the geometry, the pseudopotential/orbital file names, and per-atom
  attributes (movement constraints, magnetic moments, velocities).
- `KPT` - only read when the k-points are not set from `INPUT` through
  `kspacing` or `gamma_only`.

A run writes `OUT.<suffix>/` and `running_<calculation>.log` next to them.
Reference: [references/stru-format.md](references/stru-format.md),
[references/kpt-sampling.md](references/kpt-sampling.md),
[references/input-keywords.md](references/input-keywords.md).

## Units: check this before anything else

`LATTICE_CONSTANT` is in **Bohr**, and `LATTICE_VECTORS` are in units of that
constant. Two ways to work in Angstrom, and they are not equivalent:

- Set `LATTICE_CONSTANT` to `1.889726125457828` (Bohr per Angstrom). Then the
  cell vectors *and* `Cartesian` coordinates are all Angstrom. This is the
  common convention.
- Keep `LATTICE_CONSTANT` in Bohr and write the positions as
  `ATOMIC_POSITIONS Cartesian_angstrom`. Only the coordinates change unit; the
  cell stays Bohr.

ABACUS's own documentation quotes `1.889726125457828` for the second route;
the CODATA-derived value of 1 Bohr in Angstrom is `0.529177249`, so the factor
is `1.8897259885789233`. The two differ by about 1e-7 in relative terms, far
below any DFT tolerance, but pick one and use it consistently so round-trip
comparisons stay clean. Details in
[references/units-and-conventions.md](references/units-and-conventions.md).

## Choose the calculation type first

| `calculation` | Purpose | Must be present |
| --- | --- | --- |
| `scf` | Self-consistent ground state | basis, cutoffs, k-points, `scf_thr`/`scf_nmax` |
| `nscf` | Bands/DOS on a converged density | `init_chg file`, `out_band 1`/`out_dos 1`, line-mode `KPT` for bands |
| `relax` | Ions move, cell fixed | `cal_force 1`, `relax_method`, `relax_nmax`, `force_thr_ev` |
| `cell-relax` | Ions and cell move | also `cal_stress 1`, `stress_thr` |
| `md` | Molecular dynamics | `md_type`, `md_nstep`, `md_dt`, `md_tfirst`, masses in STRU |
| `get_pchg`/`get_wf`/`get_S` | Dump charges/wavefunctions/overlap | LCAO for the matrix dumps |

Minimal LCAO SCF deck:

```text
INPUT_PARAMETERS
suffix            ABACUS
calculation       scf
basis_type        lcao
ks_solver         genelpa
ecutwfc           100
scf_thr           1e-7
scf_nmax          100
smearing_method   gaussian
smearing_sigma    0.015
symmetry          1
kspacing          0.2
```

Per task add: `relax_method`/`relax_nmax`/`force_thr_ev` for `relax`;
`cal_stress 1` and `stress_thr` for `cell-relax`; `md_type`/`md_nstep`/`md_dt`/
`md_tfirst` for `md`; `out_chg 1` before any density, Bader, DDEC, or
work-function analysis; `out_mat_hs2`/`out_mat_r`/`out_dm`/`out_wfc_lcao` for
the LCAO matrix consumers. Keyword meanings, defaults and units:
[references/input-keywords.md](references/input-keywords.md).

## STRU in one screen

```text
ATOMIC_SPECIES
Si 28.0855 Si_ONCV_PBE-1.0.upf upf201

NUMERICAL_ORBITAL          # LCAO only
Si_gga_8au_60Ry_2s2p1d.orb

LATTICE_CONSTANT
1.889726125457828

LATTICE_VECTORS
5.43 0.0 0.0
0.0 5.43 0.0
0.0 0.0 5.43

ATOMIC_POSITIONS
Direct
Si
0.0
2
0.0 0.0 0.0 m 0 0 0
0.25 0.25 0.25 m 1 1 1 mag 1.0
```

Block grammar and the per-atom suffixes (`m`, `v`/`vel`, `mag`/`magmom`,
`angle1`/`angle2`, `lambda`, `sc`), the magnetization autoset rules, the
`latname`/`LATTICE_PARAMETERS` shortcut, and the mistakes that silently change
the structure are in [references/stru-format.md](references/stru-format.md).
Pick the pseudopotential and orbital files with the `abacus-basis` skill.

## KPT in one screen

```text
K_POINTS
0
Gamma            # or MP; the next line is "nx ny nz [sx sy sz]"
9 9 9 0 0 0
```

The three modes - automatic mesh, explicit list with weights, and line mode for
band paths (`Line`/`Line_Cartesian`) - plus the symmetry-reduction and weight
rules are in [references/kpt-sampling.md](references/kpt-sampling.md).
`gamma_only 1` overwrites the KPT file, so turn it off for multi-k runs.

## Checking a deck before the run

Read the deck against the list below before starting a run. These are the
things that decide whether ABACUS can start at all and whether the result means
what you think it means:

- The element counts in the `ATOMIC_POSITIONS` headers match the number of
  coordinate lines, and every label in that section also appears in
  `ATOMIC_SPECIES`.
- `LATTICE_CONSTANT` and the coordinate mode agree: Bohr vectors with
  Angstrom numbers, or the reverse, is the most common silent error.
- Exactly one k-point source is active: `kspacing`, a `KPT` file, or
  `gamma_only`.
- Every element has a pseudopotential, plus an orbital when `basis_type lcao`,
  and all pseudopotentials share one XC functional unless `dft_functional`
  says otherwise.
- The keywords the calculation type needs are present (`init_chg file` and
  `out_band`/`out_dos` for `nscf`; `cal_force`/`relax_*` for `relax`;
  `cal_stress`/`stress_thr` for `cell-relax`; `md_*` plus masses for `md`).

ABACUS reports an unknown keyword or a missing file when it starts, and the
message lands in `running_<calculation>.log` in the working directory.

## References

- [references/input-keywords.md](references/input-keywords.md) - keywords by
  task area with type, default, and units; how to look up any other keyword.
- [references/stru-format.md](references/stru-format.md) - every STRU block and
  per-atom suffix, coordinate modes, magnetization defaults, common mistakes.
- [references/kpt-sampling.md](references/kpt-sampling.md) - the three KPT
  modes, weights under symmetry, band paths, `gamma_only`.
- [references/units-and-conventions.md](references/units-and-conventions.md) -
  Bohr/Ry/eV/kBar, the LATTICE_CONSTANT convention, and how to tell which
  convention a file uses.
