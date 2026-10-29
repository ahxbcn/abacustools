# Getting started

## Install

From a checkout, install the package in editable mode:

```bash
python -m pip install -e .
```

This provides the `abacustools` console script. If installation is not possible,
run the package with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m abacustools.main --help
```

Python 3.9 or newer is required. Optional features are installed through the
extras declared in `pyproject.toml`:

```bash
python -m pip install -e '.[mp]'
python -m pip install -e '.[baderkit]'
python -m pip install -e '.[pyatb]'
```

## Basic usage

The command families are `file`, `job`, `postprocess`, `workflow`,
`database`, and `mp`. `postprocess` has the aliases `post`/`pp`, the database
family has `db`, and `mp` is the Materials Project spelling of
`database --database mp`.

```bash
abacustools --help
abacustools file --help
abacustools file stru STRU POSCAR
abacustools file struinfo STRU --coordination
abacustools job prepare -f STRUCTURE --library apns
abacustools postprocess result -j JOB
```

Running `abacustools` with no arguments in a terminal opens the interactive
menu. The menu mirrors the argparse tree, prompts for the arguments of the
selected command, prints the equivalent non-interactive command, runs it, and
waits for ENTER. In a non-interactive process it prints help instead.

## Configuration

User configuration is read from `~/.abacustools/config.yaml` and merged with
the packaged defaults in `src/abacustools/core/default_config.yaml`. The main
settings cover resource libraries, job submission, external Bader/Chargemol
programs, and ABACUS version profiles.

## Where to go next

- File conversion and editing: [file commands](commands/file.md)
- Job preparation and monitoring: [job commands](commands/job.md)
- Results, bands, DOS, densities, trajectories, and bonding: [postprocess commands](commands/postprocess.md)
- Multi-step calculations: [workflow commands](commands/workflows.md)
- Structure search and download: [database commands](commands/databases.md)
- Python objects and the ASE adapter: [Python API](python-api.md)
