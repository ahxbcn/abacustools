# Numerical orbital selection

Sources: ABACUS documentation (`docs/advanced/pp_orb.md`, the
`NUMERICAL_ORBITAL` section of `docs/advanced/input_files/stru.md`) and the
ABACUS community talk "ABACUS 计算的赝势、轨道基组选择与 SCF 收敛策略" (2026).
Numerical atomic orbitals are needed only for `basis_type lcao`.

## Reading an orbital file name

```text
Fe_gga_7au_100Ry_4s2p2d1f.orb
 |   |    |    |     |
 |   |    |    |     `- radial functions per angular momentum
 |   |    |    `- plane-wave cutoff encoded for the basis, in Ry
 |   |    `- cutoff radius, in Bohr (au)
 |   `- functional tag
 `- element
```

The counts are the number of **radial functions** of each angular momentum, not
a list of occupied atomic orbitals. For the Fe example: the pseudopotential has
2 s, 1 p and 1 d valence shells; the orbital file adds a second zeta to each of
them and one f polarisation function, which makes it a DZP-level basis. The
number of basis functions per Fe atom is therefore

```text
4 s x 1 + 2 p x 3 + 2 d x 5 + 1 f x 7 = 27
```

So `4s2p2d1f` has no direct correspondence to atomic orbitals such as 4s, 2p,
or 2d. Reading the string as a shell occupation is a common mistake.

Consequences:

- A structure mixing `..._150Ry_...` and `..._100Ry_...` orbitals runs at the
  largest encoded cutoff unless INPUT sets `ecutwfc` itself; an explicit
  `ecutwfc` below the orbital cutoff is a warning.
- The canonical sizes are SZ, DZP and TZDP. DZP is the usual working basis,
  TZDP the high-precision one, and SZ is generally not recommended.
- The cutoff radius is a separate axis from the zeta count, and more radius is
  not monotonically better. Published PW-versus-LCAO comparisons show the
  deviation shrinking from 7 au (0.276) to 9 au (0.076) and rising again at
  10 au (0.109): transferability peaks and then degrades as the orbital is made
  softer. Pick the radius the library tested, not the largest available.

## Choosing a basis size

| purpose | basis |
| --- | --- |
| Geometry optimisation, phonons, elastic and mechanical properties, MD | DZP (for example APNS-PPORBs-v1 efficiency) |
| High-precision energies, electronic structure including empty states (bands, DOS), optical properties | TZDP (for example APNS-PPORBs-v1 precision) |
| SOC calculations | orbitals generated from the SOC pseudopotential of the same library |
| Magnetic properties | orbitals paired with a pseudopotential that keeps the magnetic shells in valence |

Cost and accuracy both increase from SZ to DZP to TZDP; the cutoff radius adds
cost linearly and accuracy non-monotonically.

## Orbital ordering (matters for projected analyses)

ABACUS orders orbitals lexicographically by species, atom, `l`, zeta, and `m`,
with one exception: within an atom `m` runs

```text
0, 1, -1, 2, -2, ..., l, -l
```

not the usual `-l ... +l`. The angular part uses real spherical harmonics
defined so that they differ from some published tables by a factor of `(-1)^m`.

Analyses that address orbitals by global index inherit this ordering, so a
projected COHP orbital group or a projected-DOS selector is an index into
exactly this sequence. Deriving the index of a given `(atom, l, m, zeta)` by
hand is a common source of wrong projections; prefer a listing produced by the
analysis itself when one is available.

## Where orbitals come from

Three places serve the orbital sets:

| orbital set | pairs with | SOC | sizes | elements |
| --- | --- | --- | --- | --- |
| SG15-V1.0 `StandardOrbitals-V2.0` | SG15 V1.0 pseudopotentials | no | SZ, DZP, TZDP | H-La, Hf-Bi |
| SG15-V1.0 `AllOrbitals-V2.0` | SG15 V1.0 pseudopotentials | no | DZP, TZDP (more radii) | H-La, Hf-Bi |
| Dojo-NC-SR | Pseudo-Dojo NC-SR | no | DZP, TZDP | H-La, Hf-Bi |
| Dojo-NC-FR | Pseudo-Dojo NC-FR | yes | SZ, DZP, TZDP | H-La, Hf-Bi |
| Dojo-NC-SR La series | Pseudo-Dojo +3 lanthanide pseudopotentials | no | DZP, TZDP | Ce-Lu |
| APNS-PPORBs-v1 efficiency | mixed sources (SG15, Pseudo-Dojo, PD04, ...) | no | DZP | H-La, Hf-Bi |
| APNS-PPORBs-v1 precision | mixed sources | no | TZDP (mostly 10 au) | H-La, Hf-Bi |
| APNS-PPORBs-v1 lanthanides | PD04 +3 lanthanide pseudopotentials | no | SZ, DZP, TZDP | Ce-Lu |

Also on the official site: the `SG15-V1.0_Pseudopotential` orbital package is an
older set and is no longer recommended; use the V2.0 orbital sets above. The
SIAB tool in the ABACUS-orbitals repository generates new orbitals.

### Where to download

| source | what it serves | download |
| --- | --- | --- |
| ABACUS official list | SG15 V1.0 standard/all orbital packages, the Dojo-NC-FR orbitals, and the pseudopotentials they pair with | https://abacus.ustc.edu.cn/pseudo/list.htm |
| ABACUS-orbitals repository | Dojo-NC-SR orbitals, the Dojo-NC-SR lanthanide series, and the SIAB generation tool | https://github.com/abacusmodeling/ABACUS-orbitals/tree/main |
| APNS dataset | APNS-PPORBs-v1 efficiency (DZP), precision (TZDP) and the lanthanide variant | https://www.aissquare.com/datasets/detail?pageType=datasets&name=ABACUS-APNS-PPORBs-v1%253Apre-release&id=326 |

On the official list page the packages are meant to be taken as pairs:
`SG15-V1.0_Pseudopotential` with `SG15-V1.0__StandardOrbitals-V2.0` or
`SG15-V1.0__AllOrbitals-V2.0`, and the Pseudo-Dojo pseudopotentials with
Dojo-NC-SR (non-SOC) or Dojo-NC-FR (SOC). Keep a record of the URL and date you
downloaded from.

Caveats that follow from the pairing rule:

- Orbitals must be used with the pseudopotential they were generated from.
  Mixing them arbitrarily changes the basis without an error message.
- The La-series and lanthanide sets are built on 4f-in-core +3 pseudopotentials,
  so they are a poor choice when the 4f magnetism matters.
- The APNS pre-release sets (`v1-pre-release`, `ALLDZP-v1-pre-release`) differ
  from the released files and are not recommended for production.

## Empty (ghost) atoms

A label whose name contains the `empty` suffix (`H_empty`, `O_empty`) keeps its
basis functions but contributes no ionic potential, which is how BSSE
corrections are set up. The label still needs a pseudopotential and, for LCAO,
an orbital file, because the basis must exist. A code that cannot represent an
`empty` label may expect a dummy element instead, which is a conversion problem
rather than an ABACUS one.

## Generating your own orbitals

The ABACUS documentation points at the spillage and PTG/DPSI constructions and
at the generation programs:

- ABACUS ORBGEN: https://github.com/kirk0830/ABACUS-ORBGEN
- SIAB, in the ABACUS-orbitals repository
- APNS project: https://kirk0830.github.io/ABACUS-Pseudopot-Nao-Square/
- Numerical atomic orbital tutorials (Chinese):
  https://mcresearch.github.io/abacus-user-guide/

Reusing a published set is the cheaper route; generate only when the element or
the accuracy target is not covered.
