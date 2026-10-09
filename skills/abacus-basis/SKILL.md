---
name: abacus-basis
description: Choose ABACUS pseudopotential and numerical orbital files. Covers what a pseudopotential file contains, core-valence and semicore choices, norm-conserving and ultrasoft formats, spin-orbit and functional compatibility, plane-wave versus LCAO requirements, the public libraries and their element coverage, and how the orbital naming and ordering decide accuracy and the indices of projected analyses. Use when deciding which UPF or ORB files a structure needs, or when a run fails or behaves oddly because of the basis.
---

# Pseudopotential and orbital selection

## Decide the basis first

| `basis_type` | needs | size controlled by |
| --- | --- | --- |
| `pw` (default) | one pseudopotential per element only | `ecutwfc` |
| `lcao` | pseudopotential **and** one numerical orbital per element | orbital files; `ecutwfc` is taken from their names |

`NUMERICAL_ORBITAL` in `STRU` is ignored by a plane-wave run, and an LCAO run
without an orbital for some element cannot start. `gamma_only` is LCAO-only.

## What a pseudopotential decides

A pseudopotential replaces the nucleus and the core electrons, and the file
carries that frozen core with it. Swapping the file changes the calculation even
when every keyword stays the same:

- **Valence/core split.** `z_valence` in the file's `PP_HEADER` is how many
  electrons are treated explicitly, and it is the reference every charge
  analysis is compared against. Freezing a shallow semicore state makes the run
  cheaper and can shift magnetic, elastic and defect energetics.
- **Non-local projectors.** The non-local part is a set of angular-momentum
  projectors; how hard they are is what sets the plane-wave cutoff an element
  needs, so the choice of file and the choice of `ecutwfc` are linked.
- **Non-linear core correction (NLCC).** Files that carry it reconstruct part of
  the core density, which matters when core and valence overlap, as in magnetic
  or bond-breaking situations.
- **Relativity and the functional** are baked into the file as well, and the
  rules below are mostly about keeping them consistent across the structure.

Reading the header before trusting a file is cheap and catches most selection
mistakes: [references/pseudopotential-selection.md](references/pseudopotential-selection.md).

## Pseudopotential rules that cause hard failures

- Supported norm-conserving formats: UPF, UPF2, VWR, BLPS. Supported ultrasoft
  formats: UPF, UPF2. The type is declared as the fourth field of the
  `ATOMIC_SPECIES` line (`upf`, `upf201`, `vwr`, `blps`, or `auto`); omitting it
  means `auto`.
- All pseudopotentials of one structure must share the same XC functional, or
  `dft_functional` in `INPUT` must name it explicitly. Note that
  `dft_functional` overrides the functional stored in the file rather than
  reconciling it.
- Spin-orbit (`lspinorb 1`) requires a file with `relativistic="full"` and
  `has_so="T"` (or `1`) in its `PP_HEADER`. ABACUS stops with
  `no soc upf used for lspinorb calculation` when that is missing.
- Without spin-orbit either scalar- or full-relativistic files work; ABACUS
  degrades a full-relativistic file to its scalar average automatically.
- A full-relativistic ultrasoft file must be used with `lspinorb=true`, else
  ABACUS warns `FR-USPP please use lspinorb=.true.`.
- Charges that integrate the density (DDEC in particular) need norm-conserving
  files; ultrasoft densities do not integrate to the valence charge the analysis
  assumes.

## Orbital rules

- One `.orb` per element label, LCAO only. The file name encodes the choice:
  `Si_gga_8au_60Ry_2s2p1d.orb` is element, functional, cutoff radius in Bohr,
  plane-wave cutoff in Ry, and radial functions per angular momentum.
- Larger cutoff radius and higher zeta mean better transferability and a more
  expensive run. The usual progression is SZ, DZP, TZDP; DZP is the common
  default, TZDP for high-precision work.
- The largest `...Ry...` in the selected file names sets the job's `ecutwfc`
  unless INPUT states one. An explicit value below the orbital cutoff is
  reported as a warning.
- Orbital ordering inside ABACUS is species, then atom, then ascending angular
  momentum, then zeta, with `m` ordered `0, 1, -1, 2, -2, ..., l, -l`. That
  ordering defines the global indices that projected analyses take (COHP
  orbital groups, projected DOS selectors), so a wrong assumption shifts the
  projections.
- Ghost/empty atoms for BSSE are labels ending in `empty` (`H_empty`), and must
  still have a pseudopotential and an orbital so the basis exists without the
  potential.

Details and the full naming rules:
[references/orbital-selection.md](references/orbital-selection.md).

## Where files come from

Pseudopotentials:

- **SG15** (norm-conserving, PBE): versions 1.0, 1.1, 1.2 for the non-SOC runs
  and a 1.0 FR set for SOC. Coverage is H-La and Hf-Bi, with gaps in the FR set.
- **Pseudo-Dojo** (norm-conserving): versions 0.3 to 0.5 are usable here, 0.4
  adds SOC support and +3 lanthanide files; standard and stringent flavours come
  with recommended cutoffs. The JTH sets are PAW and cannot be used.
- **PSlibrary** and **GBRV** (ultrasoft): PSlibrary covers H-Am with an SOC
  version and both 4f-in-core and 4f-in-valence lanthanides; GBRV is built for
  high-throughput work with cutoffs down to about 40 Ry.
- **PD03/PD04** (PWmat, norm-conserving): H-Po, with lanthanide options in PD04;
  neither supports SOC, and the `...SOC.zip` package is not usable in ABACUS.
- **SSSP** and **APNS** are tested sets: SSSP recommends one pseudopotential per
  element in efficiency and precision flavours with published test data, and
  APNS publishes the same kind of benchmark run inside ABACUS.

Orbitals come from the ABACUS download site and the ABACUS-orbitals repository:
SG15 V1.0 `StandardOrbitals-V2.0` and `AllOrbitals-V2.0`, Dojo-NC-SR,
Dojo-NC-FR (SOC), the Dojo-NC-SR lanthanide series, and the APNS-PPORBs-v1
efficiency (DZP) and precision (TZDP) sets with their lanthanide variant.

Two rules outrank the choice of library: an orbital must be used with the
pseudopotential it was generated from, and the pair must match the physics the
run needs - SOC, or magnetic shells kept in valence.

Library tables, coverage, download locations and test data:
[references/pseudopotential-selection.md](references/pseudopotential-selection.md),
[references/orbital-selection.md](references/orbital-selection.md).

## Putting the files into a calculation

- Name the pseudopotential as the third field of its `ATOMIC_SPECIES` line and,
  for LCAO, the orbital on the corresponding `NUMERICAL_ORBITAL` line. Both are
  looked up in the working directory unless an explicit path is given.
- `pseudo_dir` and `orbital_dir` in `INPUT` can point at the directories
  holding those files, which keeps a job directory free of copied data.
- A job is self-contained only if the files travel with it: ABACUS reads them
  at startup and stops if one is missing or, for LCAO, if an element has no
  orbital.
- ABACUS re-checks relativity when it reads a pseudopotential: a scalar file in
  a `lspinorb=1` run is a fatal error, a full-relativistic ultrasoft file
  without `lspinorb=1` is a warning. The message names the file, so the fix is
  to swap the file rather than to adjust the keywords.
- Masses on the `ATOMIC_SPECIES` line matter only for molecular dynamics; the
  pseudopotential and orbital file names matter for every run.

## References

- [references/pseudopotential-selection.md](references/pseudopotential-selection.md)
  - what the file contains, which header fields to audit, core/valence and
  semicore choices, hardness and cutoffs, the public libraries with their
  coverage and SOC support, and the published test data.
- [references/orbital-selection.md](references/orbital-selection.md) - naming,
  zeta and radius choice, ordering and global indices, empty atoms, generators.
