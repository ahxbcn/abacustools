# Spin, correlation, dispersion and charge

These switches interact: a spin treatment constrains the functional and the
pseudopotential, DFT+U constrains the basis and nspin, and dispersion depends on
the functional. Check the combinations, not the keywords one by one.

## Spin and spin-orbit coupling

`nspin` selects the representation:

| `nspin` | meaning |
| --- | --- |
| 1 | single spin (non-magnetic) |
| 2 | collinear spin up/down |
| 4 | noncollinear and/or spin-orbit |

- `noncolin 1` or `lspinorb 1` forces `nspin 4`; a run with one of them and a
  different `nspin` is refused.
- `noncolin 1` cannot be combined with `gamma_only 1`; the wavefunction is
  doubled (`npol = 2`) and the density carries four Pauli components.
- `lspinorb 1` requires a full-relativistic pseudopotential (`has_so = T`) for
  every element; a scalar-relativistic file aborts with
  "no soc upf used for lspinorb calculation".
- The three combinations are meaningful:
  `noncolin 0/lspinorb 1` (SOC with z-axis magnetism), `noncolin 1/lspinorb 0`
  (noncollinear magnetism without SOC), `noncolin 1/lspinorb 1` (both).
- `soc_lambda` (0 to 1) interpolates the scalar and full-relativistic limits of
  the same full-relativistic file.
- meta-GGA (SCAN/SCAN0/SCANL) is not implemented for `nspin 4` on either branch.

## DFT+U

| keyword | meaning |
| --- | --- |
| `dft_plus_u` | 0 off; 1 radius-adjustable localized projections; 2 legacy first-zeta NAO projections |
| `orbital_corr` | one value per species: -1 none, 1 p, 2 d, 3 f |
| `hubbard_u` | one U in eV per species |
| `onsite_radius` | projection radius for method 1; auto-set to 3.0 Bohr when left at 0 |
| `dft_plus_dmft` | DFT+DMFT, LCAO only |

- If every `orbital_corr` is -1, DFT+U is switched off with a warning even when
  `dft_plus_u` is non-zero.
- Basis/nspin support: LCAO supports nspin 1, 2 and 4 on both branches. PW
  support differs: LTS 3.10.1 refuses PW DFT+U unless `nspin 4`
  ("only nspin2 with PW base is not supported now"), while develop accepts
  nspin 1, 2 or 4 with the PW basis.
- develop marks the legacy `dft_plus_u 2` force/stress path as broken and
  rejects it when `cal_force` or `cal_stress` is set; use method 1 for
  relaxations.
- `dft_plus_dmft` requires `basis_type lcao`.

## Dispersion correction

`vdw_method`:

| value | LTS 3.10.1 | develop |
| --- | --- | --- |
| `none` | yes | yes |
| `d2` | yes | yes |
| `d3_0` (zero damping) | yes | yes |
| `d3_bj` (Becke-Johnson damping) | yes | yes |
| `d4` | no | yes, needs the external DFT-D4 library |

- D3 and D4 load damping parameters for the chosen `dft_functional`
  automatically; `vdw_s6`, `vdw_s8`, `vdw_a1`, `vdw_a2` override them, and
  setting all four defines a fully custom set.
- D4 adds `vdw_d4_xc` (functional name for the D4 library; `default` infers it)
  and `vdw_d4_model` (`d4` or `d4s`).
- A dispersion correction needs a positive `vdw_cutoff_radius`; a zero cutoff
  with `vdw_method != none` is refused.
- There is no vdW-density-functional keyword in these trees; a VV10-type
  functional has to come through a LIBXC expression.

## Charged cells

- `nelec` is the total number of electrons: 0 means the neutral sum of the
  pseudopotential valence charges, a positive value sets it explicitly and must
  stay below `2 * nbands`.
- `nelec_delta` is added to `nelec`, which is the usual way to write a charged
  cell without recomputing the neutral count.
- `nupdown` constrains the spin difference for collinear calculations.
- The compensating background and any finite-size correction are the user's
  responsibility; ABACUS does not add a charged-defect correction.
- `efield` (and the dipole-correction options) are the separate mechanism for
  an applied field or a slab dipole, not for setting the cell charge.

## Compile-time dependencies

Each of these areas can additionally be gated by the build: LIBXC for
meta-GGA/hybrids, CUDA for GPU solvers, DFT-D4 for `vdw_method d4`, ELPA for
`genelpa`/`elpa`, PEXSI for `pexsi`. See
[versions.md](versions.md) for the option names and the failure mode.
