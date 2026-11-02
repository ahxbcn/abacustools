# Python API

[Documentation home](index.md)

## Public data types

The core data types are importable straight from the package, so a script can
work with structures, inputs and results without reaching into the `io`/`data`
submodules:

```python
from abacustools import AbacusSTRU, AbacusATOM, ReadInput, Unitcell, UPF

structure = AbacusSTRU.read("STRU")
inputs = ReadInput("INPUT")
```

The exported set covers the file-handling types: the structures
(`AbacusSTRU`, `AbacusATOM`, `AbacusAtomType`, `StructureConversionWarning`),
the cell helper `Unitcell`, `INPUT` (`ReadInput`, `WriteInput`), the
pseudopotential and orbital readers (`UPF`, `AbacusNAO`) and the Molden types
(`MoldenShell`, `MoldenAtom`, `MoldenOrbital`). The names resolve lazily, so
`import abacustools` stays cheap and a defining module is loaded only when its
name is first used; `dir(abacustools)` lists the whole set.

## ABACUS ASE interface

The separate `abacuslite` ASE interface can be used as an optional calculation
backend while ABACUSTools continues to provide job and structure management.
It is loaded only when requested, so it is not required for the normal command
line workflows.

```python
from ase.optimize import BFGS
from abacustools.io.stru import AbacusSTRU
from abacustools.integrations.abacuslite import (
    attach_calculator,
    calculator_from_structure,
    make_profile,
    structure_to_atoms,
)

structure = AbacusSTRU.read("STRU")
profile = make_profile(
    "mpirun -np 8 abacus",
    pseudo_dir="/path/to/pseudopotentials",
    orbital_dir="/path/to/orbitals",
)
atoms = structure_to_atoms(structure)
calculator = calculator_from_structure(
    structure,
    profile,
    directory="ase-relax",
    inp={"calculation": "scf", "basis_type": "lcao", "cal_force": 1},
)
attach_calculator(atoms, calculator)
BFGS(atoms).run(fmax=0.05)
```

The same adapter can build a calculator from an existing ABACUS job with
`calculator_from_job`. The returned ASE calculator can then be used by ASE
relaxation, cell-relaxation, NEB, MD, or `fixed_density()` band workflows.
Use `result_to_dict()` or `write_result()` to place ASE results in the
repository's JSON-compatible result format.
