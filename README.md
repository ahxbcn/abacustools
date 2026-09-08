Collection of tools used for performing DFT calculation with ABACUS.

## Command-line subcommands

Subcommands are registered in the package with Python's
`argparse.add_subparsers()`. The top-level command provides `version` and
nested `file` subcommands:

```text
abacustools version
abacustools file input INPUT
abacustools file stru STRU
abacustools file kpt KPT
```

Complete ABACUS input directories can be prepared with a resource library
selected from `~/.abacustools/config.yaml`:

```text
abacustools job prepare -f STRUCTURE --library apns
```

The configuration contains the `resources.default` library and paths for each
library's `pp` and `orb` directories. The command uses the default library
when `--library` is omitted. For example:

```yaml
resources:
  default: apns
  libraries:
    apns:
      pp: /path/to/apns-pseudopotentials
      orb: /path/to/apns-orbitals
    dojo-nc-sr:
      pp: /path/to/Dojo-NC-SR/Pseudopotential
      orb: /path/to/Dojo-NC-SR/Orbitals
```

Both the top-level parser and each subcommand provide their own help text:

```text
abacustools --help
abacustools file --help
abacustools file input --help
```

Calculation results can be collected from one or more completed ABACUS jobs:

```text
abacustools postprocess result -j JOB -p energy force stress
```

Input files can be checked independently of calculation output:

```text
abacustools job checkinput JOB
abacustools job checkinput JOB --strict
```

The command checks `INPUT`, `STRU`, `KPT`, and referenced pseudopotential,
orbital, and PAW files. It reports the calculation settings, structure size
and composition, cell parameters, k-point mode, and resource files. It does
not read `OUT.*` directories or calculation logs. Use `--json` for a
machine-readable report.

Convergence tests can generate independent SCF calculations for cutoff energy:

```text
abacustools workflow ecutwfc prepare -j JOB --values 30 40 50 60
abacustools workflow ecutwfc postprocess -j JOB --energy-tol 1e-4
```

The same workflow can test k-point spacing:

```text
abacustools workflow kspacing prepare -j JOB --values 0.4 0.3 0.2 0.1
abacustools workflow kspacing postprocess -j JOB
```

The postprocessing stage reports total energy per atom, convergence deltas,
incomplete tasks, a recommended first value within the tolerance, a JSON
report, and a convergence plot. `cutoff` is an alias for `ecutwfc`.

Use `abacustools postprocess result --help` to see all supported result
parameters. If `--param` is omitted, the command selects the results relevant
to the job's calculation type and displays scalar results as compact,
borderless tables. If the result columns do not fit the terminal, they are
split into multiple tables with repeated headers rather than being truncated
or wrapped by the terminal. A result that does not apply to a job is shown as
`-`. Large array results such as `force` and `stress` are omitted from this
summary.

Band structures from ABACUS NSCF calculations can be processed and plotted
from a job directory containing `BANDS_1.dat`:

```text
abacustools postprocess band -j JOB --emin -8 --emax 8
```

The command reads the line-mode `KPT`, shifts the bands by the Fermi energy
from the NSCF log, and writes `band.png`, `band.dat` (or spin-resolved files),
and `KPATH.txt` below `JOB`. Use `--efermi` to provide an explicit Fermi
energy when it is not present in the log. Only `calculation nscf` jobs are
recommended; other calculation types emit a warning and are still processed.

COHP and COOP curves can be calculated for selected ABACUS NAO orbital groups
when the LCAO output contains `data-*-H`, `data-*-S`, `kpoints`, and
`WFC_NAO_K*.txt` files:

```text
abacustools postprocess cohp -j JOB \
  --atom-i-orbs 0,1,2 --atom-j-orbs 13,14,15 \
  --method COHP --de 0.1 --output cohp.png --data-output cohp.dat
```

Orbital indices are zero-based global NAO indices. Use `--method COOP` for
overlap-weighted curves, `--spin up|down` for a spin channel, and `--invert`
to invert only the plotted curve. The data file keeps the computed sign.
`ICOHP`/`ICOOP` printed by the command is the integral up to the Fermi level.
This is an ABACUS-NAO COHP/COOP implementation and is not a standard LOBSTER
pCOHP projection.

DOS and projected DOS can be processed from an ABACUS output directory:

```text
abacustools postprocess dos -j JOB --emin -20 --emax 10
abacustools postprocess dos -j JOB --pdos species
```

The command reads `DOS*_smearing.dat`, shifts the energy axis by the Fermi
energy, and writes `DOS.png` and `DOS.dat`. Use `--pdos species-shell`,
`--pdos species-orbital`, or `--pdos atoms --atom-index 1 2` for projected DOS
plots. Output paths can be changed with `-o` and `--data-output`; use
`--efermi` when the Fermi energy cannot be read from the ABACUS output.

Mayer bond orders can be analyzed from an ABACUS LCAO calculation with
`out_mat_hs=1` (and `out_dm=1` for gamma-only jobs):

```text
abacustools postprocess mayer -j JOB --cutoff 3.0
abacustools postprocess mayer -j JOB --pairs 1-2,1-3 --json -o mayer.json
```

The analyzer reads numerical orbitals, overlap matrices, density matrices or
NAO wavefunctions, validates their dimensions, and reports atom indices,
elements, periodic distances, and Mayer bond orders. It supports gamma-only
and multi-k calculations with `nspin=1` or `nspin=2`.

Complex calculation workflows are organized by task and stage. The BSSE
workflow currently provides the preparation and postprocessing framework:

```text
abacustools workflow bsse prepare
abacustools workflow bsse postprocess
```

Charge-density difference calculations are available as `chgdiff`. Prepare
the full-system and two-subsystem SCF jobs, submit them with
`sbatch runabacus.sh`, and then generate the difference cube:

```text
abacustools workflow chgdiff prepare -j JOB -i 1 3 4
abacustools workflow chgdiff postprocess -j JOB
```

Elastic constants can be calculated with the stress-strain workflow. It
generates one unstrained job and 24 independently strained jobs. By default,
the jobs use ionic relaxation at fixed cell shape; use `--norelax` for fixed-ion
SCF calculations:

```text
abacustools workflow elastic prepare -j JOB
sbatch runabacus.sh  # submit from each generated job directory
abacustools workflow elastic postprocess -j JOB
```

The fitted elastic tensor and Voigt moduli are written to
`elastic_results.json` under `JOB`.

Phonon spectra can be calculated with finite differences using Phonopy. The
prepare stage generates displaced supercell SCF jobs, and the postprocess
stage reads their forces to produce thermal properties, DOS, and a combined
dispersion/DOS plot:

```text
abacustools workflow phonon prepare -j JOB
abacustools workflow phonon postprocess -j JOB
```

Use `--supercell A B C` to set the supercell explicitly. Without it, the
supercell is selected so each lattice vector is at least 10 Angstrom long.
Custom paths can be passed as JSON with `--qpath` and
`--high-symm-points`.

Molecular vibration frequencies can be calculated with selected atoms using
central finite differences. The workflow writes equilibrium and displaced
force calculations below `vib/`:

```text
abacustools workflow vibration prepare -j JOB
abacustools workflow vibration postprocess -j JOB
```

Use `--index 1 2 ...` to select atoms and `--traj` to write mode trajectories.

Surface work functions can be calculated from the averaged electrostatic
potential. The prepare stage enables `out_pot=2` and writes a calculation
under `workfunc_job`; the postprocess stage identifies vacuum plateaus and
writes the work-function results and potential profile:

```text
abacustools workflow workfunc prepare -j JOB
abacustools workflow workfunc postprocess -j JOB
```

Use `--vacuum a|b|c|auto` to select the vacuum direction, and
`--dipole-corr` to enable dipole correction.

Born effective charges can be calculated with Berry-phase finite differences.
The workflow prepares SCF and three Berry-phase NSCF calculations for each
selected atom displacement:

```text
abacustools workflow bec prepare -j JOB --index 1 --dir x y z --type c
```

Run `run_bec.sh` in each generated `bec_*` directory, then postprocess the
polarization differences:

```text
abacustools workflow bec postprocess -j JOB
```

The BEC tensors and task diagnostics are written to `bec_results.json` under
`JOB`. Missing or incomplete Berry-phase task output is retained as missing
tensor entries so other completed displacement directions can still be reported.

Piezoelectric stress tensors can be calculated from finite-strain changes in
the Berry-phase polarization. The workflow generates the six independent
Voigt strain modes (`xx`, `yy`, `zz`, `yz`, `xz`, and `xy`), with forward,
backward, or central differences:

```text
abacustools workflow piezoelectric prepare -j JOB --strain 0.01 --type c
```

Run `run.sh` in each generated `piezoelectric_*` directory. Use `--relax` to
relax ionic positions at each strained cell before the Berry-phase
calculation. The tensor and per-task diagnostics are written under `JOB`:

```text
abacustools workflow piezoelectric postprocess -j JOB
```

Generated calculation directories are protected by default. Use `--override`
when intentionally replacing them. The BEC workflow's `run_bec.sh` is only a
local four-step runner for one generated task; cluster submission scripts are
not generated because their contents depend on the target computing environment.

Each prepared workflow records its task names and atom partition in a
workflow-specific manifest such as `workflow_phonon.json`. Postprocessing
validates this manifest and checks that the required SCF calculations converged
before reading their outputs.

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
