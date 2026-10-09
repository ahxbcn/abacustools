# Implicit solvation, electric fields and dipole correction

## Implicit solvation

Both branches support an implicit solvent through `imp_sol`. The model is the
one of Andreussi, Dabo and Marzari: a diffuse dielectric cavity whose shape is
determined self-consistently by the electron density of the solute.

| keyword | meaning | default |
| --- | --- | --- |
| `imp_sol` | enable the implicit solvation correction | `false` |
| `eb_k` | relative permittivity of the bulk solvent | 80 (water) |
| `tau` | effective surface tension (cavitation, dispersion, repulsion) | 1.0798e-5 |
| `sigma_k` | width of the diffuse cavity | 0.6 |
| `nc_k` | charge density at which the cavity forms | 0.00037 |

`eb_k`, `tau`, `sigma_k` and `nc_k` are active only when `imp_sol` is true.
The correction enters the electrostatic potential written by `out_pot 2`; it is
not a continuum electrostatics solve on a fixed surface, so there is no
separate cavity file to provide. The develop tree lists the same keyword set as
LTS 3.10.1.

## Electric field and dipole correction

A slab can carry a sawtooth electric field and a dipole correction:

| keyword | meaning |
| --- | --- |
| `efield_flag` | enable the electric-field / dipole-correction machinery |
| `dip_cor_flag` | apply the dipole correction; requires `efield_flag` |
| `efield_dir` | direction (`0`, `1`, `2`) parallel to a reciprocal lattice vector |
| `efield_pos_max` | fractional position of the field discontinuity along `efield_dir` |
| `efield_pos_dec` | width over which the potential is ramped from max to zero |
| `efield_amp` | field amplitude; set to 0 for a pure dipole correction |

Constraints, checked while reading the input:

- `dip_cor_flag true` with `efield_flag false` aborts
  ("dipole correction is not active if efield_flag=false").
- An electric field does not support `symmetry 1`; the parser reports it and
  the run must use a lower symmetry.
- The discontinuity must fall inside the vacuum of a slab; the model is not
  meant for a bulk periodic cell.
- A gate field is the separate `gate_flag` mechanism (`zgate`, `relax`,
  `block`, `block_up`, `block_down`, `block_height`); `gate_flag` with
  `efield_flag` also requires `dip_cor_flag`.

Neither feature changes the cell charge. For a charged cell use
`nelec`/`nelec_delta` (see [capabilities.md](capabilities.md)).
