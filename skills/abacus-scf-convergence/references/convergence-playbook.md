# Convergence playbook

Ready parameter sets, from the easy cases to the hard ones. Each block shows
only the SCF-relevant keywords; the basis, cutoff and k-point settings stay as
in the calculation recipe.

## Well-behaved default

```text
mixing_type      broyden
mixing_beta      0.8            # nspin 1
mixing_ndim      8
mixing_gg0       1.0
scf_thr          1e-8
scf_nmax         100
```

## Metal

```text
smearing_method  gaussian
smearing_sigma   0.02
mixing_type      broyden
mixing_beta      0.4
mixing_ndim      10
mixing_gg0       1.0
scf_thr          1e-8
scf_nmax         150
```

A metal needs a real smearing; if the charge sloshes, keep `mixing_gg0 1.0` and
lower `mixing_beta`.

## Insulator or wide-gap system

```text
smearing_method  gaussian
smearing_sigma   0.001
mixing_type      broyden
mixing_beta      0.8
mixing_ndim      8
scf_thr          1e-8
scf_nmax         100
```

Do not use a metallic smearing; a small gaussian sigma is enough.

## Slab or low-dimensional large system

Low-dimensional systems with a lot of vacuum are the classic hard case for
charge sloshing. Lower the charge mixing and lengthen its history:

```text
mixing_type      broyden
mixing_beta      0.025
mixing_ndim      24
mixing_gg0       1.0            # Kerker is bypassed at mixing_beta <= 0.1
scf_thr          1e-8
scf_nmax         200
```

If it is stable but slow, try `mixing_beta 0.1` with `mixing_ndim 20` and
`mixing_gg0 1.0`, which the source notes usually works well for low-dimensional
large systems.

## Collinear magnetic system

Set the initial moments in `STRU` first, then reduce the magnetic mixing:

```text
nspin            2
mixing_type      broyden
mixing_beta      0.1
mixing_beta_mag  0.2            # 1.5-2 times mixing_beta; default is 4x, capped at 1.6
mixing_ndim      16
scf_thr          1e-8
scf_nmax         200
```

If the moment still oscillates, add the magnetic Kerker preconditioner
(`mixing_gg0_mag 1.0`) and lower `mixing_beta_mag` further. A correct initial
moment is more important than any of these.

## Noncollinear or SOC magnetic system

```text
nspin            4
noncolin         1
mixing_type      broyden
mixing_beta      0.1
mixing_beta_mag  0.2
mixing_angle     1.0            # relax the moment directions to the ground state
mixing_ndim      16
symmetry         -1             # do not impose time-reversal symmetry
scf_thr          1e-8
scf_nmax         200
```

`mixing_angle` only accepts 1.0. Use `symmetry -1` so the noncollinear spin
treatment is not combined with a time-reversal reduction.

## meta-GGA

The density can converge while the energy does not, because the kinetic energy
density is not mixed by default:

```text
dft_functional   SCAN           # or another meta-GGA
mixing_type      broyden
mixing_beta      0.1
mixing_tau       1
mixing_ndim      16
scf_thr          1e-8
scf_ene_thr      1e-5           # eV, checked in addition to scf_thr
scf_nmax         200
```

## DFT+U

```text
dft_plus_u       1
orbital_corr     -1 2          # per species
hubbard_u        0.0 4.0        # eV, per species
mixing_type      broyden
mixing_beta      0.1
mixing_ndim      16
scf_thr          1e-8
scf_nmax         200
```

`mixing_dftu 1` additionally mixes the occupation matrices, but experience in
the source says it is not very helpful; lowering `mixing_beta` is the first
move.

## Restart from a previous density

```text
init_chg         file
read_file_dir    OUT.ABACUS
mixing_restart   1e-4          # restart the mixer once drho falls below this
mixing_beta      0.4
scf_thr          1e-8
scf_nmax         100
```

Reusing a converged density from a related calculation is the strongest lever
for a system that only converges from a good guess. In a relaxation or MD run,
`chg_extrap` already extrapolates the density between ionic steps.

## Progressive tuning

When the first attempt oscillates, step the charge mixing down rather than
jumping straight to the smallest value:

```text
# 1) start
mixing_beta      0.4
mixing_ndim      8
# 2) oscillating
mixing_beta      0.1
mixing_ndim      16
# 3) still oscillating
mixing_beta      0.025
mixing_ndim      24
```

Keep `scf_thr` and the physics fixed while tuning; change only the mixer.
