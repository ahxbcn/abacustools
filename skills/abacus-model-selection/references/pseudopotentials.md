# Pseudopotential support

ABACUS reads UPF files. The reader dispatches on the version string of the
first line: `2.0.1` selects the UPF v2.0.1 parser, anything else the legacy
parser. The `STRU` file names the file per species in `ATOMIC_SPECIES`, and
`pseudo_dir` locates it.

## Supported types

| pseudopotential | PW basis | LCAO basis | notes |
| --- | --- | --- | --- |
| norm-conserving UPF (`pseudo_type NC`) | yes | yes | the common case |
| ultrasoft UPF (`is_ultrasoft T`) | yes | no | the banner says USPP is "for plane wave basis set" |
| PAW (`is_paw T`) | no | no | `PAW POTENTIAL IS NOT SUPPORTED` |
| semi-local (`pseudo_type SL`) | no | no | `SEMI-LOCAL PSEUDOPOTENTIAL IS NOT SUPPORTED` |

The banner printed while reading pseudopotentials states the same rule:
"ABACUS supports norm-conserving (NC) pseudopotentials for both plane wave
basis set and numerical atomic orbital basis set. In addition, ABACUS supports
ultrasoft pseudopotentials (USPP) for plane wave basis set."

A legacy `BLPS`/`VWR` reader exists in the source but the dispatcher only
returns `upf201` or `upf`, so a run is driven by UPF in practice.

## Header fields that decide capability

The UPF `PP_HEADER` is parsed for:

- `pseudo_type`: `NC` or `US`; `SL` aborts.
- `relativistic`: `scalar` or `full`; the reader records it and uses it for the
  spin-orbit treatment.
- `is_ultrasoft` / `is_paw`: set the type and reject PAW.
- `has_so`: whether the file carries spin-orbit projectors.

## Spin-orbit coupling

- `lspinorb 1` requires `nspin 4`, and every element needs a UPF with
  `has_so = T` (a full-relativistic pseudopotential). A scalar-relativistic
  file aborts with `no soc upf used for lspinorb calculation`.
- `noncolin 1` (noncollinear magnetism) also requires `nspin 4` and cannot be
  combined with `gamma_only 1`.
- `soc_lambda` (0 to 1) interpolates between the scalar-relativistic and
  full-relativistic treatment of the same file: 0 is no SOC, 1 is full SOC.
- Full-relativistic ultrasoft pseudopotentials are supported for PW and require
  `lspinorb` to be on.

## Data that some analyses need

- `PP_RHOATOM` (spherical free-atom density): required by the Hirshfeld and
  promolecular/IGMH analyses of `postprocess hirshfeld`/`chg`.
- `PP_PSWFC` (pseudo atomic wavefunctions): required by Hirshfeld-I (and
  `igmh-i`) unless reference densities are supplied explicitly.

Many ONCV norm-conserving files carry `PP_RHOATOM` but not `PP_PSWFC`, so a
Hirshfeld calculation can work on a pseudopotential that cannot feed
Hirshfeld-I.

## Related keywords

`pseudo_rcut` and `pseudo_mesh` (PW) adjust the radial renormalization of the
projectors. The `STRU` `ATOMIC_SPECIES` block can carry a `pp_type` column, and
a `PAW_FILES` block is parsed for PAW datasets; the pseudopotential itself must
still be a supported UPF.
