# Functional support

The functional is set by `dft_functional` in `INPUT`. If the keyword is absent,
ABACUS uses the functional declared in the pseudopotential (`xc_func` in the
UPF `PP_HEADER`). An explicit `dft_functional` overwrites the pseudopotential
value, so a mismatched pair can run but is physically inconsistent; check that
the pseudopotential covers the intended functional.

## Short-hand names

Both branches accept these short-hand names (case-insensitive):

| class | names |
| --- | --- |
| LDA | `LDA` (= `PZ` = `SLAPZNOGXNOGC`), `PWLDA` |
| GGA | `PBE` (= `SLAPWPBXPBC`), `PBESOL`, `REVPBE`, `WC`, `BLYP`, `BP` (BP86), `PW91`, `HCTH`, `OLYP`, `BLYP_LR` |
| meta-GGA | `SCAN` (LIBXC), `SCAN0` (LIBXC hybrid); develop adds `SCANL` (LIBXC) |
| hybrid | `PBE0`, `HF` |
| LIBXC hybrids | `HSE` (= HSE06), `B3LYP`, `LC_PBE`, `LC_WPBE`, `LRC_WPBE`, `LRC_WPBEH`, `CAM_PBEH`, `MULLER` (= `POWER`), `WP22`, `CWP22` |

The develop input reference lists exactly this set; LTS 3.10.1 accepts the same
short-hands through the same `set_xc_type` dispatch, minus `SCANL`.

## LIBXC

- SCAN, SCAN0, SCANL, HSE, and the long-range/CAM/MULLER/WP22/CWP22 hybrids
  are built from LIBXC ids. Without a `ENABLE_LIBXC` build, `set_xc_type`
  aborts with "to use SCAN, SCAN0, HSE, long-range corrected ... LIBXC is
  required".
- A LIBXC build also accepts an expression of LIBXC component keywords joined
  by `+`, for example
  `dft_functional='LDA_X_1D_EXPONENTIAL+LDA_C_1D_CSC'`. This is the escape
  hatch for a functional that has no short-hand.
- `B3LYP` is routed through the same hybrid machinery and its components come
  from LIBXC ids, so treat it as LIBXC-dependent.

## Branch differences

| capability | LTS 3.10.1 | develop |
| --- | --- | --- |
| `SCANL` | no | yes (LIBXC) |
| hybrid functional with PW basis | refused ("hybrid functional not realized for planewave yet") | supported (PW and LCAO) |
| meta-GGA (SCAN/SCAN0/SCANL) with `nspin 4` | refused | refused |
| GGA/LDA with `nspin 4` | supported | supported |

The meta-GGA restriction is in the gradient kernel ("meta-GGA has not been
implemented for nspin = 4 yet") and holds on both branches, so an SOC or
noncollinear calculation cannot use SCAN.

## Practical sequence

1. Read the functional from the pseudopotential (`xc_func`) or choose one.
2. If it is a hybrid, a meta-GGA or a long-range-corrected one, confirm the
   build has LIBXC.
3. If it is a hybrid on LTS and the basis is PW, expect a refusal; use LCAO or
   the develop branch.
4. If `nspin 4` is set, avoid meta-GGA.
5. For a dispersion-corrected calculation, the functional also selects the
   D3/D4 damping parameters (see
   [capabilities.md](capabilities.md)).

## Meta-GGA restrictions beyond nspin 4

- Meta-GGA functionals (SCAN, SCAN0, SCANL) are LIBXC-only on both branches, so
  a non-LIBXC build refuses them before the run starts.
- The one enforced physical restriction is `nspin 4`: the gradient kernel aborts
  with "meta-GGA has not been implemented for nspin = 4 yet" on both branches.
  A collinear (nspin 1 or 2) meta-GGA run is allowed.
- Both branches carry a dedicated meta-GGA stress path
  (`stress_func_mgga` / `stress_mgga`), so a stress or cell-relaxation run with
  a meta-GGA is supported for nspin 1/2. The LTS source keeps a commented-out
  historical guard about "mgga stress not implemented for polarized case"; it
  is not active, so the enforced limit stays the nspin-4 one.
- A meta-GGA run produces the kinetic-energy-density cube (`tau.cube` /
  `taus<n>.cube` on develop, `SPIN<n>_TAU.cube` on LTS) through the same
  `out_chg` path; `out_elf` can then use it.
- Hybrid meta-GGA (SCAN0) inherits both the LIBXC requirement and the nspin-4
  restriction.
