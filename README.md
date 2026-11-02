# abacustools

`abacustools` is a Python CLI toolkit for preparing, running, inspecting, and
postprocessing ABACUS DFT calculations.

## Highlights

- Convert, inspect, and edit `INPUT`, `STRU`, `KPT`, and common structure files.
- Prepare complete ABACUS job directories from pseudopotential/orbital libraries.
- Monitor SCF, relaxation, cell-relaxation, and MD jobs.
- Postprocess bands, DOS/PDOS, charge densities, trajectories, COHP/COOP,
  Mayer and DDEC bond orders, Bader/Hirshfeld charges, and Molden orbitals.
- Run multi-step workflows for convergence, elastic, phonon, dielectric,
  piezoelectric, BSSE, DFT+U, magnetic exchange, and related calculations.
- Search and download structures through OPTIMADE and the Materials Project.

## Install

```bash
python -m pip install -e .
```

Python 3.9 or newer is required. Without an editable install, run commands with
`PYTHONPATH=src python -m abacustools.main ...`.

## Quick start

```bash
abacustools --help
abacustools file stru STRU POSCAR
abacustools file struinfo STRU --coordination
abacustools job prepare -f STRUCTURE --library apns
abacustools postprocess result -j JOB
abacustools workflow phonon prepare -j JOB
abacustools database search -d cod --formula Fe2O3 --limit 5
```

The `postprocess` family also accepts `post`/`pp`; the database family accepts
`db`, and `mp` is the Materials Project spelling of `database --database mp`.
Running `abacustools` with no arguments in a terminal launches the interactive
menu.

## Documentation

The detailed documentation is under [`docs/`](https://github.com/ahxbcn/abacustools/tree/develop/docs):

- [Getting started](https://github.com/ahxbcn/abacustools/blob/develop/docs/getting-started.md)
- [File commands](https://github.com/ahxbcn/abacustools/blob/develop/docs/commands/file.md)
- [Job commands](https://github.com/ahxbcn/abacustools/blob/develop/docs/commands/job.md)
- [Postprocess commands](https://github.com/ahxbcn/abacustools/blob/develop/docs/commands/postprocess.md)
- [Workflow commands](https://github.com/ahxbcn/abacustools/blob/develop/docs/commands/workflows.md)
- [Structure databases](https://github.com/ahxbcn/abacustools/blob/develop/docs/commands/databases.md)
- [Python API and ASE adapter](https://github.com/ahxbcn/abacustools/blob/develop/docs/python-api.md)

The installed argparse tree remains the authoritative source for command
options; use `abacustools <family> <command> --help`.

## Development

```bash
python -m pytest tests
ruff check src tests
```

See [`AGENTS.md`](AGENTS.md) for repository architecture, contribution rules,
and documentation policy.

The package is licensed under LGPL-3.0-or-later.
