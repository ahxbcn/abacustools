# scf

The base run. Every other recipe reuses its parameters.

## Plane-wave basis

| keyword | common value | comment |
| --- | --- | --- |
| `basis_type` | `pw` | |
| `ecutwfc` | 50-80 Ry | converge the total energy per atom to about 1 meV |
| `ecutrho` | 4x for NC, 8-12x for USPP | only PW has a separate density cutoff |
| `kspacing` | about 0.14 (1/Bohr) for bulk; 1 or 3 values | not VASP's KSPACING; set the vacuum direction large |
| `smearing_method` | `gaussian` | |
| `smearing_sigma` | 0.015 Ry; insulators 0.001-0.005 Ry | do not use `fixed` for an insulator |
| `mixing_type` | `broyden` | |
| `mixing_beta` | 0.8 (`nspin 1`), 0.4 (`nspin 2/4`) | lower to 0.2-0.4 if the energy oscillates |
| `scf_thr` | 1e-8 | 1e-6 is acceptable inside a large relaxation |
| `scf_nmax` | 100-200 | |
| `ks_solver` | `dav_subspace` with `pw_diag_ndim 2` | recommended |
| `symmetry` | 0 | use -1 for nspin 4 SOC; 1 only in limited cases |
| `nspin` | 1 (or 2, 4) | |

## LCAO basis

| keyword | common value | comment |
| --- | --- | --- |
| `basis_type` | `lcao` | needs a numerical orbital per element |
| `ecutwfc` | at least the cutoff in the orbital name | the orbital recommends the value |
| `ks_solver` | `genelpa` | needs an ELPA build; otherwise `scalapack_gvx` |
| `scf_thr` | 1e-7 | 1e-6 is acceptable inside a large relaxation |
| `gamma_only` | 1 for a Gamma-only run | cannot be combined with `noncolin` |
| `mixing_*`, `smearing_*`, `symmetry` | as for PW | |

## Full example (PW)

```text
calculation      scf
basis_type       pw
ecutwfc          60
ecutrho          240
kspacing         0.14
smearing_method  gaussian
smearing_sigma   0.015
mixing_type      broyden
mixing_beta      0.8
scf_thr          1e-8
scf_nmax         100
ks_solver        dav_subspace
pw_diag_ndim     2
symmetry         0
nspin            1
```

## Systems with a vacuum layer

`kspacing` accepts one value or three values, one per reciprocal direction.
For a slab, wire or molecule in a box, give the vacuum direction a large
value, for example 1.0 or more in 1/Bohr, so the generated mesh is 1 along
that direction:

```text
kspacing         0.14 0.14 1.0     # third axis is the vacuum
```

This is the usual fix for a slab whose mesh is unnecessarily dense across the
vacuum; the two periodic directions keep the ordinary spacing. A Gaussian
smearing and the ordinary mixing settings are unchanged.

## Cost and tightening

- Inside a geometry optimization of a large structure, `scf_thr 1e-6` is a
  common compromise: each ionic step is cheaper and the final energy is still
  accurate enough to drive the forces. Keep 1e-8 for a small cell or when a
  delicate observable (gap, magnetic moment) is read from the run.
- The cutoff and k sampling still have to be converged before loosening the
  SCF threshold; a loose SCF threshold does not excuse an unconverged basis.

## Verify

- `scf_thr` reached and `drho` flat.
- The total energy no longer drifts with the last SCF steps.
- For a metal the smearing sigma does not change the energy or the moment
  beyond the target tolerance; for an insulator a too-large sigma shows up as
  an artificial occupation across the gap.
- The magnetic moment is stable if `nspin > 1`.
