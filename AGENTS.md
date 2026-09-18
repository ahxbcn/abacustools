# AGENTS.md

Guidance for AI coding agents (and humans) working in **abacustools**, a CLI
toolkit for DFT calculations with [ABACUS](https://abacus.deepmodeling.com/).

## Project overview

- Pure-Python package `abacustools`, exposed as the `abacustools` console
  script (`abacustools.main:main`). No compiled extensions.
- Requires Python >= 3.9 (developed on 3.11).
- Runtime dependencies: `numpy`, `rich`, `pymatgen`, `phonopy`, `seekpath`,
  `matplotlib`, `ase`, `pydantic`.
- Four command families:
  - `abacustools file ...` — convert/inspect `INPUT`, `STRU`, `KPT`, structures.
  - `abacustools job ...` — prepare, check, validate, and monitor jobs.
  - `abacustools postprocess ...` — `result`, `band`, `dos`, `cohp`, `mayer`, `bader`.
  - `abacustools workflow ...` — multi-step workflows (elastic, phonon, ...).

## Environment setup

```bash
pip install -e .          # editable install; provides the `abacustools` script
```

The package is normally used from an editable install, so `import abacustools`
works without `PYTHONPATH`. If you cannot install, prefix commands with
`PYTHONPATH=src`.

## Common commands

```bash
abacustools --help                        # CLI help (also: <family> --help)
abacustools menu                          # interactive multi-level menu
python -m pytest tests                    # full test suite
python -m pytest tests/test_bader.py -q   # one module
ruff check src tests                      # lint (rule set pinned in pyproject.toml)
ruff format src tests                     # format
```

## Repository layout

```text
src/abacustools/
  main.py            # argparse entry point; registers the four families
  version.py         # __version__
  commands/          # CLI layer (thin): <family>/<command>.py
    file/ job/ postprocess/ workflow/
  data/              # parsing/analysis of ABACUS outputs -> arrays/dataclasses
  io/                # read/write file formats (STRU, INPUT, KPT, pseudo, NAO)
  core/              # config, constants, job/process handling, submission
  integrations/      # adapters to external tools (e.g. abacuslite)
  menu/              # interactive multi-level menu (argparse reflection)
tests/               # pytest suite (test_*.py)
```

Layering: `commands/` may import `data/`, `io/`, and `core/`; `data/` and `io/`
must not import `commands/`. Keep the CLI layer thin — argument parsing and
rendering only.

## Adding or changing a command

1. Implement the logic in `data/` (or `io/`, `core/`), not in the command file.
   Draw the module boundary along the computation, not along the commands: when
   two command modules need the same helper — a phonopy object, a force reader,
   a polarization conversion — give it a public home in `data/`. A command
   module must never import a private name out of a sibling command module, and
   shared helpers must not be copied into a second one.
2. Create `commands/<family>/<name>.py` containing:
   - `register_parser(subparsers)` that adds the subparser and calls
     `parser.set_defaults(handler=<func>)`;
   - a handler `def <func>(args: argparse.Namespace) -> int` returning the exit
     code (0 on success).
3. Register the module in `commands/<family>/__init__.py`.
4. Add `tests/test_<name>.py`.
5. Document the command in `README.md`.

New commands appear in the interactive menu automatically: `abacustools/menu/`
reflects the argparse tree, so keep argument definitions declarative and avoid
menu-specific branching in command modules.

## Workflows and manifests

- A workflow records the decisions of its prepare stage in a
  `workflow_<name>.json` manifest below the job directory, next to a `tasks`
  list naming the generated calculations. Postprocessing reads that manifest
  instead of inferring the setup from result files or from the order of a
  directory listing.
- A workflow that derives a tensor or a table writes it as JSON next to the
  manifest — `bec_results.json`, `dielectric_results.json` — including the
  units, the method and the parameters that produced it, so a later workflow can
  consume it instead of a user transcribing numbers.

## Optional dependencies

- A feature that needs a package the project does not require declares it as an
  extra in `pyproject.toml`, imports it lazily inside the function that runs it,
  and keeps the module importable without it: only the call fails, with a
  message naming the missing package and how to install it. `integrations/`
  holds those adapters, `abacuslite` and `pyatb` among them.
- `workflow dielectric` needs `pyatb`, whose compute core is C++ behind a
  compiled extension and which parallelises through `mpi4py`. The extra
  installs the Python package; the MPI runtime comes from the environment
  (`conda install -c conda-forge mpich`). The step runs where the ABACUS
  outputs are — locally, next to the `abacustools` environment — while the
  cluster only runs ABACUS.

## Code style

- Start modules with `from __future__ import annotations`.
- Type-annotate public functions; prefer `pathlib.Path` over string paths.
- Google-style docstrings (`Args:` / `Returns:`) on public APIs.
- Render CLI output with `rich`; support `--json` for structured reports.
- Never use a bare `except:`; avoid `# type: ignore` unless justified.
- Match the conventions of the file you are editing.

## Units and domain conventions

- ABACUS lengths are in Bohr, energies in Ry/eV. Conversion constants live in
  `core/constant.py` (`BOHR_TO_ANG`, `ANG_TO_BOHR`, `RY_TO_EV`, ...).
- `*-CHARGE-DENSITY.restart` stores `rho(G)`; ABACUS-format cube files store
  density in e/Bohr^3. The internal `Charge` class stores density in e/Ang^3 and
  cell vectors in Angstrom.
- When writing cube files, keep grid geometry at full precision — low-precision
  cell vectors corrupt the cell volume and make integrated charges non-integer.
- Phonopy 4 changed two defaults away from phonopy 3: `primitive_matrix`
  resolves `"auto"` with a symmetry search instead of the identity, and the
  `get_frequencies`/`get_*_dict` accessors are deprecated in favour of the
  result objects (`run_qpoints`, `thermal_properties`, `total_dos`,
  `band_structure`). Pin the matrix and read the objects, keeping a fallback
  only where an older phonopy has to keep working.
- The `occupied bands` count ABACUS autosets and prints in its running log is
  what pins the occupation of the `pyatb` Kubo-Greenwood sum that gives the
  electronic dielectric tensor. ABACUS's `EFERMI` for an insulator is one
  arbitrary level inside the gap; the sum is built from transition energies,
  which are differences, so the energy reference does not enter the result.
- A phonon non-analytical correction needs both the Born effective charges
  (`workflow bec`) and the clamped-ion dielectric tensor (`workflow
  dielectric`). The two results files are read by the phonon postprocessing
  stage with `--bec-results` and `--dielectric-results`.

## Testing

- Tests live in `tests/` as `test_*.py`; the suite mixes `unittest.TestCase`
  classes and plain pytest functions.
- Use `tmp_path` / `tempfile` for file I/O; never depend on network access or a
  real ABACUS binary. Prefer small synthetic fixtures over large reference files.
- Run the full suite (`python -m pytest tests`) before considering work done.

## Branching

- Never develop directly on `develop` or `main`. Before the first edit of a
  task, create a topic branch from the up-to-date base branch:
  `git switch -c <type>/<topic>` (for example `feat/ddec-postprocess`,
  `fix/bader-vacuum`).
- One branch carries one topic. If a new request is unrelated to the branch at
  hand, branch off the base branch again instead of stacking more work on top.
- State the branch in the final answer of a task so the user can check it out,
  and leave the branch checked out when the work is done.

## Git and commits

- Commit subjects are short and imperative: `Add <feature>` or `feat: ...`,
  `fix: ...`.
- Keep commits focused and do not mix unrelated changes. Prefer a separate
  `fix:` commit over folding a fix into a feature commit.
- Never commit generated artifacts: `OUT.*/`, `build/`, `__pycache__/`,
  `.pytest_cache/`, `.ruff_cache/`, or `.sisyphus/`.

## Agent checklist

- [ ] The work happened on a topic branch, not on `develop`/`main`.
- [ ] Logic lives outside the CLI layer; the command file only wires arguments.
- [ ] New behavior is covered by tests in `tests/`.
- [ ] `python -m pytest tests` passes.
- [ ] `ruff check src tests` is clean (or only pre-existing issues remain).
- [ ] CLI changes are reflected in `README.md`.
- [ ] No generated or large files staged.
