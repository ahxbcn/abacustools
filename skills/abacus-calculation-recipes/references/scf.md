# scf

The base run. Every other recipe reuses its parameters.

## Plane-wave basis

| keyword | common value | comment |
| --- | --- | --- |
| `basis_type` | `pw` | |
| `ecutwfc` | 50-80 Ry | converge the total energy per atom to about 1 meV |
| `ecutrho` | 4x for NC, 8-12x for USPP | only PW has a separate density cutoff |
| `kspacing` | about 0.14 (1/Bohr) for bulk | not VASP's KSPACING; differ by 2*pi |
| `smearing_method` | `gaussian` | |
| `smearing_sigma` | 0.015 Ry (shipped template); insulators 0.001-0.005 Ry | do not use `fixed` for an insulator |
| `mixing_type` | `broyden` | |
| `mixing_beta` | 0.8 (`nspin 1`), 0.4 (`nspin 2/4`) | lower to 0.2-0.4 if the energy oscillates |
| `scf_thr` | shipped template 1e-7; 1e-8 for PW production | tighten for production |
| `scf_nmax` | 100-200 | |
| `ks_solver` | `dav_subspace` with `pw_diag_ndim 2` | recommended |
| `symmetry` | 0 (the shipped scf template uses 1) | use -1 for nspin 4 SOC |
| `nspin` | 1 (or 2, 4) | |

## LCAO basis

| keyword | common value | comment |
| --- | --- | --- |
| `basis_type` | `lcao` | needs a numerical orbital per element |
| `ecutwfc` | at least the cutoff in the orbital name | the orbital recommends the value |
| `ks_solver` | `genelpa` | needs an ELPA build; otherwise `scalapack_gvx` |
| `scf_thr` | 1e-7 | |
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

## Verify before trusting the result

- `scf_thr` reached and `drho` flat.
- The total energy no longer drifts with the last SCF steps.
- For a metal, the smearing sigma does not change the energy or the moment
  beyond the target tolerance; for an insulator a too-large sigma is visible as
  an artificial occupation across the gap.
- The magnetic moment is stable if `nspin > 1`.
