# KPT file reference

Source: ABACUS documentation, `docs/advanced/input_files/kpt.md`
(https://abacus.deepmodeling.com). The first line may be `K_POINTS`, `KPOINTS`,
or `K`. The second line selects the mode.

## Automatic mesh

```text
K_POINTS
0
Gamma            # or MP
9 9 9 0 0 0
```

`0` means "generate automatically". `Gamma` samples a Gamma-centred
Monkhorst-Pack grid, `MP` the shifted Monkhorst-Pack scheme. The last line is
`nx ny nz` plus the three real-valued shifts.

## Explicit list with weights

```text
K_POINTS
8
Direct           # or Cartesian
0.0 0.0 0.0 0.125
0.5 0.0 0.0 0.125
...
```

`N` is the number of k-points; each following line is three coordinates and one
weight. With `symmetry 1`, ABACUS reduces the list to the irreducible Brillouin
zone: weights of symmetry-equivalent points are combined and the result is
normalised so the weights sum to `degspin` (2 for non-spin-polarised, 1 for
spin-polarised). Custom weights are preserved through that reduction, so an
inconsistent weight distribution survives and produces unexpected results.

## Band structure (line mode)

```text
K_POINTS
6
Line             # or Line_Cartesian
0.5 0.0 0.5 20   # X
0.0 0.0 0.0 20   # G
0.5 0.5 0.5 20   # L
0.5 0.25 0.75 20 # W
0.375 0.375 0.75 20 # K
0.0 0.0 0.0 1    # G
```

The count after each node is the number of points between it and the next node;
the count on the last node is unused. `Line` reads the nodes as fractional
coordinates, `Line_Cartesian` as Cartesian ones. A band path belongs to an
`nscf` (`calculation nscf`, `init_chg file`) run on a converged density;
The band data files are then read together with this file.

## Interaction with INPUT

- `gamma_only 1` (LCAO only) **overwrites** the KPT file; turn it off for
  multi-k runs.
- `kspacing` in `INPUT` is the alternative to a KPT file: a reciprocal-space
  spacing in 1/Bohr, from which ABACUS builds the mesh. Set it or the KPT file,
  not both.
- A single Gamma point is the right mesh for an isolated molecule in a box;
  a bulk crystal needs a real mesh or `kspacing`, and a missing setting is a
  common cause of a suspiciously fast, suspiciously wrong run.
- The mesh and the cell must be consistent for the intended sampling density:
  doubling every lattice vector halves the mesh you need for the same
  reciprocal-space spacing.
