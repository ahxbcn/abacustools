# Pseudopotential selection

Sources: ABACUS documentation (`docs/advanced/pp_orb.md`, the `ATOMIC_SPECIES`
section of `docs/advanced/input_files/stru.md`) and the ABACUS community talk
"ABACUS 计算的赝势、轨道基组选择与 SCF 收敛策略" (2026), whose library data is
drawn from the sites linked below.

## What ABACUS can use

| kind | formats | usable in ABACUS |
| --- | --- | --- |
| norm-conserving | UPF, UPF2, VWR, BLPS | yes |
| ultrasoft | UPF, UPF2 | yes |
| PAW (projector-augmented wave) | UPF (`is_paw`) | **no** - not supported at present |
| BLPS | BLPS | only for orbital-free DFT |

UPF/UPF2 is the common public format. A file declares its kind in the header
(`pseudo_type="NC"`, `is_ultrasoft="F"`, `is_paw="F"`), so a PAW file from a
mixed library can be recognised before it is used. Several libraries publish
PAW and ultrasoft files side by side; only the ultrasoft ones are relevant here.

## What the UPF file tells you

The sections that carry physics:

| section | content | why you care |
| --- | --- | --- |
| `PP_HEADER` | element, valence configuration, `z_valence`, `pseudo_type`, relativity, functional, suggested cutoff | the fields you audit before using the file |
| `PP_MESH` | radial grid `PP_R` and increments `PP_RAB` | defines the numerical support of everything else |
| `PP_LOCAL` | local part of the potential | the smooth channel the projectors act on |
| `PP_NONLOCAL` | `PP_BETA` projectors (angular momentum, cutoff radius) and the `PP_DIJ` coupling matrix | hardness, and therefore the plane-wave cutoff |
| `PP_NLCC` | non-linear core correction density, when present | core-valence overlap |
| `PP_PSWFC`, `PP_CHI` | pseudo-atomic wavefunctions | orbital and initial-wavefunction construction |
| `PP_RHOATOM` | pseudo-atomic charge density | starting density and reference densities for analyses |
| `PP_SPIN_ORB` | relativistic metadata | present in files that carry spin-orbit projectors |
| `PP_INFO`, `PP_INPUTFILE` | generator provenance and input | how the file was made, which is how you judge it |

Read from the header:

- **element**, and the **valence configuration** - which shells were pseudized
  and which are treated explicitly. Example: the SG15 Fe file pseudizes the
  first three atomic orbitals (1s, 2s, 2p) and keeps the last four (3s, 3p, 4s,
  3d) as valence, which is what `z_valence` counts.
- **`z_valence`** - the number of explicitly treated electrons. Charge
  analyses, densities and every electron-count comparison are relative to this
  number, so two files for one element with different `z_valence` are not
  interchangeable.
- **kind** - `pseudo_type`, `is_ultrasoft`, `is_paw` as above.
- **relativity** - `relativistic="scalar"` with `has_so="F"` cannot do SOC;
  `relativistic="full"` with `has_so="T"` (or `1`) can.
- **functional** - the functional the file was generated for.
- **suggested cutoff** - a starting point for the `ecutwfc` convergence test,
  not a substitute for it.

The remaining header fields are generator metadata (author, date, `l_max`,
projector count) and are informative rather than decisive.

## Norm-conserving versus ultrasoft

- Norm-conserving files have no augmentation charge: the density integrates to
  `z_valence`, the theory is simpler, and the pseudopotential is supported
  everywhere. They are typically harder, so they need a larger `ecutwfc`.
- Ultrasoft files carry augmentation charges that make them much softer and
  cheaper (GBRV files reach down to about 40 Ry). Their density does not
  integrate to `z_valence` on its own, so density-partitioning analyses that
  assume that property - DDEC in particular - need a norm-conserving file.
- BLPS is a bulk-derived local pseudopotential with no projectors, used by
  orbital-free DFT; it is not a drop-in equivalent for a standard calculation.

## Core-valence split and semicore states

How much of the atom is frozen is the choice with the largest silent effect:

- Short interatomic distances put the core in contact with the bonding region,
  which makes semicore states necessary to describe the bond. This is the
  standard reason to prefer a large-`z_valence` construction for a compressed
  or high-pressure structure.
- Magnetic properties need the shells that carry the magnetism inside the
  valence. For lanthanide magnetic materials this usually means keeping the 4f
  electrons in valence rather than using the 4f-in-core files that exist for the
  +3 ions.
- Freezing shallow semicore states (3s/3p for 3d metals, 4s/4p for 4d, 4f for
  lanthanides, 5f for actinides) buys speed and can bias magnetic moments,
  equilibrium volumes, elastic constants and defect formation energies.
- Consistency matters more than the individual choice: mixing semicore and
  frozen-core constructions within one compound is a chemically inconsistent
  description even when each file is fine on its own.

## Functionals

The functional is stored in the file, and ABACUS uses it unless
`dft_functional` is set in `INPUT`, which overrides it. Community practice is
lenient here - a calculation is not required to use the functional the file was
generated with - but two rules still hold: elements within one structure should
not silently come from different functional families, and the orbital set must
be the one generated for that pseudopotential. A PBE pseudopotential with a
differently generated orbital is a silently different basis, not an error.

## Hardness and `ecutwfc`

- The non-local projectors set how rapidly the pseudo-wavefunctions oscillate,
  and that sets the plane-wave cutoff the element needs.
- Use the suggested cutoff in the header as a starting point, then converge on
  the actual system; the requirement depends on which elements dominate the
  energy differences you care about.
- ABACUS exposes `pseudo_rcut` (radial integration cutoff) and `pseudo_mesh` as
  advanced knobs over how the pseudopotential is used internally. The defaults
  are adequate; change them only with a specific numerical reason.

## Public libraries

| library | kind | SOC | elements | functionals | valence options | notes |
| --- | --- | --- | --- | --- | --- | --- |
| SG15 1.0 / 1.2 | NC | no | H-La, Hf-Bi | PBE | single | 1.1 is incomplete; use 1.0 or 1.2 |
| SG15 1.0 FR | NC | yes | H-Bi, but no Li, Be, Ne, Ba, Ce-Lu | PBE | single | the SOC counterpart |
| Pseudo-Dojo NC | NC | 0.4 has SOC | H-La, Hf-Po, Rn (0.4 adds +3 lanthanides) | LDA, PBE, PBEsol | single | versions 0.3/0.4/0.4.1/0.5 are ABACUS-usable; standard and stringent flavours with recommended cutoffs; UPF/psp8/psml |
| Pseudo-Dojo JTH | PAW | - | - | - | - | not usable in ABACUS |
| PSlibrary | US (and PAW) | SOC versions exist | H-Am | PBE, PBEsol, revPBE, ... | lanthanides in- and out-of-valence | only the ultrasoft files are usable here; some files must be generated with QE `ld1.x` and then tested |
| GBRV | US | no | H-Bi, no lanthanides, no noble gases | LDA, PBE, PBEsol | single | designed for high-throughput; cutoffs as low as about 40 Ry |
| PD03 (PWmat) | NC | no | H-Po, no Ce-Lu | PBE | single | |
| PD04 (PWmat) | NC | no | H-Po | PBE | several elements have alternatives; 3 lanthanide sets (1 with 4f in valence, 2 for +3 ions) | `NCPP-PD04-PBE-SOC.zip` cannot be used by ABACUS |

Coverage is quoted for the PBE sets; check the site for the current list and
for the other functionals.

### Where to download

| library | download | pick |
| --- | --- | --- |
| SG15 | http://quantum-simulation.org/potentials/sg15_oncv/index.htm | the site is plain HTTP only, there is no HTTPS endpoint |
| Pseudo-Dojo | https://www.pseudo-dojo.org/index.html | the norm-conserving (NC) sets; the JTH sets are PAW and unusable in ABACUS |
| PSlibrary | https://github.com/dalcorso/pslibrary, mirrored at http://theossrv1.epfl.ch/Main/Pseudopotentials | the ultrasoft files; the PAW ones are unusable here |
| GBRV | https://www.physics.rutgers.edu/gbrv/ | ultrasoft, low cutoffs, no SOC |
| PWmat PD03 / PD04 | https://www.pwmat.com/potential-download | the norm-conserving packages; avoid anything advertising SOC, it is not usable in ABACUS |
| SSSP | https://sssp.materialscloud.org/, tables at https://www.materialscloud.org/discover/sssp/table/efficiency | the recommended file of the element, efficiency or precision flavour |
| Quantum ESPRESSO | https://www.quantum-espresso.org/pseudopotentials | UPF files are usable directly |
| BLPS | https://github.com/PrincetonUniversity/BLPSLibrary | local pseudopotentials for orbital-free DFT only |

For files hosted by the ABACUS project itself, including the APNS sets, see the
download list in [orbital-selection.md](orbital-selection.md); the official page
serves pseudopotentials and orbitals together.

Keep a record of the URL and date you downloaded from, so a published result can
be traced back to the exact files that produced it.

## How to choose

1. Filter by what the feature supports: only norm-conserving and ultrasoft
   files work, and UPF/UPF2 is the safest format to pick.
2. Filter by purpose: an SOC calculation needs a file with `has_so`; a
   lanthanide magnetic material needs the f electrons in valence.
3. Confirm the valence covers the physics: short interatomic distances argue for
   semicore states.
4. Keep the pseudopotential and the orbital from the same family and the same
   generation; do not mix sets arbitrarily.
5. Check published tests for the accuracy the property needs.

## Test data

- ACWF verification (https://acwf-verification.materialscloud.org/, summarized
  in Nat. Rev. Phys. 2024, 6, 45-58): when precision is sufficient, different
  pseudopotentials and different codes agree with each other. Useful as a
  sanity target rather than a per-element recommendation.
- SSSP (https://sssp.materialscloud.org): tests many pseudopotentials and
  recommends one per element, in an efficiency and a precision flavour, with
  EOS, cutoff and band-structure data published per element.
- APNS (https://kirk0830.github.io/ABACUS-Pseudopot-Nao-Square/): the same kind
  of benchmark run inside ABACUS, covering pseudopotentials and the orbitals
  built from them.
