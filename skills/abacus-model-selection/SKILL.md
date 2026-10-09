---
name: abacus-model-selection
description: Check whether a chosen physical model is supported by ABACUS before it is configured, for the 3.10 LTS branch and the current develop branch. Covers pseudopotential types, built-in and LIBXC functionals, the electronic-structure algorithms and diagonalizers, nspin 4 / spin-orbit coupling, DFT+U, dispersion corrections, and charged systems, including which options are compile-time gated. Use when selecting a functional, pseudopotential, basis or spin treatment, or when a run is refused for an unsupported combination. Not for choosing the scientific model or for estimating accuracy.
---

# ABACUS model selection

Before writing an input deck, check that the method it names is supported by
the ABACUS build at hand. Support depends on three layers:

1. the branch (3.10 LTS versus develop),
2. the pseudopotential file (norm-conserving or ultrasoft, scalar or
   full-relativistic, which functional it declares),
3. the build (LIBXC, ELPA, CUDA, DFT-D4, PEXSI, NPZ are compile-time options).

The support facts below were read from the source trees named in
[references/versions.md](references/versions.md). They are snapshots: verify a
newer checkout against its source before trusting an edge case.

## Pinned versions

| branch | version | commit | date |
| --- | --- | --- | --- |
| LTS | `v3.10.1` | `f71921fe848659deac8db319cd4311b55b5ad480` | 2025-11-21 |
| develop | `v3.11.0-beta10` | `260139d97786fa1ac3a416e72d1a19f7b221a5db` | 2026-10-06 |

The LTS working tree carries unrelated toolchain/example modifications; the
source under `source/` matches the commit. See
[references/versions.md](references/versions.md) for how to reproduce this.

## Support matrix

| area | LTS 3.10 | develop 3.11 |
| --- | --- | --- |
| norm-conserving UPF | PW and LCAO | PW and LCAO |
| ultrasoft UPF | PW only | PW only |
| PAW / semi-local UPF | rejected | rejected |
| scalar-relativistic UPF | yes | yes |
| full-relativistic UPF + SOC | yes (`has_so`) | yes (`has_so`) |
| built-in short-hand functionals | LDA, GGA, PBE0, HF | same, plus SCAN-L |
| LIBXC functionals | optional (LIBXC build) | optional (LIBXC build) |
| meta-GGA (SCAN) | requires LIBXC | requires LIBXC |
| hybrids (HSE, B3LYP, LC/CAM, ...) | require LIBXC | require LIBXC |
| PW diagonalizers | cg, dav, dav_subspace, bpcg | same |
| LCAO diagonalizers | genelpa, elpa, lapack, scalapack_gvx, cusolver, cusolvermp, pexsi, cg_in_lcao | without `cg_in_lcao` |
| `esolver_type` | ksdft, sdft, ofdft, tddft, lj, dp, lr, ks-lr | adds tdofdft, nep, dfpt |
| nspin 1/2/4 | yes | yes |
| SOC (`lspinorb`) | nspin 4 + `has_so` UPF | same |
| DFT+U (LCAO) | yes, nspin 1/2/4 | yes, nspin 1/2/4 |
| DFT+U (PW) | only nspin 4 | nspin 1, 2 or 4 |
| DFT+U method 1 (onsite_radius) / 2 (first zeta) | both | both |
| DFT+DMFT | LCAO only | LCAO only |
| dispersion | D2, D3(0), D3(BJ) | adds D4 (external library) |
| charged cell (`nelec`,`nelec_delta`) | yes | yes |
| implicit solvation (`imp_sol`) | yes | yes |
| electric field / dipole (`efield_flag`,`dip_cor_flag`) | yes, not with `symmetry 1` | yes, not with `symmetry 1` |
| RT-TDDFT (`tddft`) | yes (PW and LCAO) | yes (PW and LCAO) |
| TD-OFDFT (`tdofdft`) | no | yes |
| DFPT (`dfpt`) | no | yes |
| PAW | compile-gated `USE_PAW`, PW only | removed |
| noncollinear + gamma_only | not allowed | not allowed |

## How to choose

1. Pick the pseudopotential first: it fixes the element, the functional it
   declares, and whether it carries spin-orbit projectors. See
   [references/pseudopotentials.md](references/pseudopotentials.md).
2. Check the functional: a short-hand name that is in the built-in table, or a
   LIBXC expression that needs a LIBXC build. See
   [references/functionals.md](references/functionals.md).
3. Check the basis and solver: PW and LCAO have disjoint diagonalizer sets, and
   some solvers are compile-time gated. See
   [references/algorithms.md](references/algorithms.md).
4. Check the interacting switches: `nspin 4`, `lspinorb`, DFT+U, dispersion and
   charged-cell settings each restrict the others. See
   [references/capabilities.md](references/capabilities.md).
5. Check the environment terms: implicit solvation, an applied field, a dipole
   or gate correction. See
   [references/solvation-and-fields.md](references/solvation-and-fields.md).

## Boundary

This skill says what is supported, not what is accurate. It does not pick a
functional or a U for a material, does not set convergence parameters, and
does not replace the pseudopotential-choice reasoning in the `abacus-basis`
skill.
