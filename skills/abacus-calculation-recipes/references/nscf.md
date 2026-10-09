# nscf

A non-self-consistent run that reuses the density of a preceding SCF. Used for
band structures, DOS/PDOS and other eigenvalue-based outputs.

## What is different from scf

| keyword | common value | comment |
| --- | --- | --- |
| `calculation` | `nscf` | |
| `read_file_dir` | `OUT.<suffix>` of the SCF | where the density is read from |
| `init_chg` | `file` | start from the stored density |
| `nbands` | occupied + 20-50% | a band edge needs empty bands |
| k points | dense mesh for DOS, line-mode `KPT` for bands | |
| `symmetry` | 0 | a band path and most outputs assume no reduction |
| `out_band` | 1 (for bands) | |
| `out_dos` | 1 (for DOS/PDOS) | |
| `scf_thr` | as for scf | still used for the diagonalization |

`ecutwfc`/`ecutrho`, smearing, mixing and `ks_solver` keep the values of the
reference SCF; the mixing parameters have little effect because the density is
not updated.

## Full example (PW, band path)

```text
calculation      nscf
basis_type       pw
ecutwfc          60
ecutrho          240
nbands           40
read_file_dir    OUT.ABACUS
init_chg         file
scf_thr          1e-8
ks_solver        dav_subspace
pw_diag_ndim     2
symmetry         0
out_band         1
```

For DOS replace the line-mode `KPT` with a dense mesh and set `out_dos 1`
instead of `out_band 1`; the mesh for DOS should be considerably denser than
the SCF mesh.

## Verify

- The Fermi level from the NSCF log is near the one expected from the SCF.
- The band gap or DOS integral is stable when the mesh or `nbands` is increased.
- For a band path, the path matches the dimensionality of the structure
  (bulk seekpath, in-plane for a slab, periodic axis for a wire).
