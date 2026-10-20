# AGENTS.md

Guidance for AI coding agents (and humans) working in **abacustools**, a CLI
toolkit for DFT calculations with [ABACUS](https://abacus.deepmodeling.com/).

## Project overview

- Pure-Python package `abacustools`, exposed as the `abacustools` console
  script (`abacustools.main:main`). No compiled extensions.
- Requires Python >= 3.9 (developed on 3.11).
- Runtime dependencies: `numpy`, `rich`, `pymatgen`, `phonopy`, `seekpath`,
  `matplotlib`, `ase`, `pydantic`, `spglib`. The config loader additionally
  imports `PyYAML`; it is not declared in `pyproject.toml` yet and only comes
  in transitively today — declare it when `pyproject.toml` is next touched.
- Command families:
  - `abacustools file ...` — convert/inspect `INPUT`, `STRU`, `KPT`, structures.
  - `abacustools job ...` — prepare, check, validate, and monitor jobs.
  - `abacustools database ...` — search and download structures from external
    databases (`list`, `fields`, `providers`, `search`, `download`); every
    database is a `StructureDatabase` in `integrations/databases/`, registered
    in that package's `__init__`. Prefer OPTIMADE when a database speaks it;
    C2DB keeps its own adapter because only its web query table publishes the
    computed properties.
  - `abacustools mp ...` — the Materials Project spelling of
    `abacustools database --database mp`.
  - `abacustools postprocess ...` — `result`, `band`, `dos`, `cohp`, `mayer`,
    `bader`, `chg`, `ddec`, `md`, `molden`.
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
  main.py            # argparse entry point; registers the command families
  version.py         # __version__, the single source of the version string
  __init__.py        # _LAZY_EXPORTS: PEP 562 top-level names (AbacusSTRU, ...)
  commands/          # CLI layer (thin): <family>/<command>.py
    file/ job/ database/ mp/ postprocess/ workflow/
  data/              # parsing/analysis of ABACUS outputs -> arrays/dataclasses
  io/                # read/write file formats (STRU, INPUT, KPT, pseudo,
                     # NAO, Molden, xyz); io/input-params.json is the packaged
                     # ABACUS parameter registry behind ReadInput
  core/              # config (default_config.yaml merged with the user's
                     # ~/.abacustools/config.yaml at import), constants, job
                     # and submission handling
  integrations/      # adapters to external tools: abacuslite, pyatb, ase,
                     # baderkit, materials_project; integrations/databases/
                     # holds one StructureDatabase per source, registered in
                     # that package's __init__.py
  menu/              # interactive multi-level menu (argparse reflection)
tests/               # pytest suite (test_*.py); tests/fixtures/ holds
                     # optional reference structures, skipped when absent
skills/              # ABACUS guidance packages; the LTS/develop output-file
                     # names live in skills/abacus-job-files/references/
                     # output-files.md
```

Layering: `commands/` may import `data/`, `io/`, and `core/`; `data/`, `io/`,
`core/`, and `integrations/` must not import `commands/`. Keep the CLI layer
thin — argument parsing and rendering only. A helper used by more than one
command module lives in `data/`, `io/`, or `core/` (for example the job
readers in `core/job.py`), never in a sibling command module.

## Reading ABACUS output (the version adapter layer)

Analysis and workflow code is version-blind: everything an ABACUS release may
change — file names, column layouts, log wording, the meaning of a printed
quantity — is confined to the reading layer, and every reader hands its
result on as one documented shape per result kind whose keys, array layout
and units do not depend on the ABACUS version. That shape, not an ABACUS
file, is what `DOSData`, `PDOSData`, `BandData`, the workflow stages and the
results-JSON files consume; changing it is an interface change and demands
the same care as a public API.

Where each kind of adaptation goes:

- Log dialects (keywords and per-step patterns of `running_*.log`) live in
  `data/versions.py`, extensible text-only from `~/.abacustools/config.yaml`
  (`abacus.versions`). Never re-match the running log with ad-hoc keywords
  or regexes somewhere else.
- Filename tables sit next to the reader function they serve — the
  `STRU_ION<step>_D`/`STRU<step>` regexes in `data/md.py`, the
  `doss*_nao.txt` matchers in `data/abacus_result.py` are the pattern.
- File-format codecs (STRU/INPUT/KPT/pseudo/...) belong in `io/`; a branch
  on version that is about *what the file says* goes next to the function
  that reads it, never inside an analysis or workflow class.

When an ABACUS version renames or reshapes an output:

1. Find the writer that produces the file in the ABACUS source tree first;
   never base a reader on the documentation alone.
2. Extend the existing reader of that result kind so both spellings work:
   glob the LTS name first and fall back to the new name; for outputs
   written per ionic step or appended per block, reject nothing silently
   and take the latest ionic step; normalise the new layout onto the old
   shape (spin channels as columns, `(l, m)` in the XML PDOS convention,
   text columns summed over zeta, single channels as shape `(n, 1)`
   columns). The reader accepts empty or partial output as missing.
3. Cover both spellings in the tests of that reader, and record the name
   pair in `skills/abacus-job-files/references/output-files.md`.
4. Consume machine-readable artefacts before reconstructing the same facts
   with regexes: the develop job-root `abacus.json` summary (its `comment`
   field states the units) and the effective-input copies below `OUT.*`
   (`INPUT` on LTS, `INPUT.info` on develop) are read when present, in
   both spellings.

Missing and unparseable are different situations: a file that the job has
not written yet raises `FileNotFoundError`; a file that exists but cannot be
understood raises `ValueError` naming the file and the spellings that were
looked for. An explicitly requested but unknown ABACUS version degrades to
the version-agnostic profile instead of raising — keep that tolerance when
touching the resolver.

## Adding or changing a command

1. Implement the logic in `data/` (or `io/`, `core/`), not in the command file.
   Draw the module boundary along the computation, not along the commands: when
   two command modules need the same helper — a phonopy object, a force reader,
   a polarization conversion — give it a public home in `data/`, `io/`, or
   `core/`. A command
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

The CLI surface has two more touch points: `postprocess` carries the pinned
aliases `post`/`pp` (`tests/test_main.py`), so keep new aliases out unless a
family really merges spellings; and a new public helper class must be added
to `_LAZY_EXPORTS` in `src/abacustools/__init__.py` so it stays importable as
`from abacustools import ...` (guarded by `tests/test_package_exports.py`).

## Adding a database provider

- Implement a `StructureDatabase` in `integrations/databases/`; use the
  shared OPTIMADE adapter unless the source publishes what OPTIMADE does
  not carry (that is why C2DB has its own).
- Register it in `integrations/databases/__init__.py`; commands select a
  database by name and never import a concrete provider.
- Ship the search/download field or key tables as package data, add cases
  in `tests/test_databases.py`, and document the database in `README.md`.

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
- Extras shipped today: `mp` (`mp-api`, the Materials Project family),
  `baderkit` (library backend of `postprocess bader`), `pyatb` (`workflow
  dielectric`). The comments in `pyproject.toml` are the source of truth for
  why each exists; note that `baderkit` pins `pymatgen <= 2025.10.7` and can
  downgrade the pymatgen of an environment that already has a newer one.
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
- Validate file contents and user input by raising `ValueError`/`TypeError`
  that name the offending file or option; reserve `assert` for internal
  invariants — assertions disappear under `python -O`.
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
- The Grueneisen workflow (`workflow gruneisen`) is three phonon workflows of
  the same crystal whose volumes differ by an isotropic strain, because the
  mode parameter is `-d ln omega / d ln V`. Its prepare stage therefore calls
  `prepare_phonon_jobs` of the phonon command, and its postprocessing stage
  rebuilds each volume with `load_workflow_phonon`. The three volumes have to
  share the supercell, the k mesh and the displacement step: the force
  constant fits are compared with each other, so anything that differs between
  them shows up as a spurious parameter.

## Testing

- Tests live in `tests/` as `test_*.py`; the suite mixes `unittest.TestCase`
  classes and plain pytest functions.
- Use `tmp_path` / `tempfile` for file I/O; never depend on network access or a
  real ABACUS binary. Prefer small synthetic fixtures over large reference files.
- A handful of skips are expected by design: optional packages not installed
  (pyatb, phono3py, baderkit) and optional reference structures under
  `tests/fixtures/`. "The suite passes" means no new failures *and* no new
  skips; never mute a new test with a skip that does not state what is
  missing and why.
- Reader/adapter tests carry both spellings of a version-dependent output,
  written as the files ABACUS actually emits, next to the reader they serve
  (see `tests/test_dos.py`, `tests/test_data_parsers.py`).
- Run the full suite (`python -m pytest tests`) before considering work done.

## Branching

- `develop` is the integration branch and the base branch for topic work;
  `main` is the release snapshot and may lag `develop` by many commits.
  Never develop directly on either. Before the first edit of a task, create
  a topic branch from an up-to-date `develop`:
  `git switch -c <type>/<topic> develop` (for example `feat/ddec-postprocess`,
  `fix/bader-vacuum`), and merge back into `develop` when the task lands.
- One branch carries one topic. If a new request is unrelated to the branch at
  hand, branch off `develop` again instead of stacking more work on top.
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
- [ ] Version-dependent file names and formats stay in the reading layer;
      both spellings are tested and recorded in
      `skills/abacus-job-files/references/output-files.md` (when ABACUS
      outputs are read or written).
- [ ] The test run adds no new failures and no new skips beyond the
      pre-existing optional-package and fixture skips.
- [ ] CLI changes are reflected in `README.md`.
- [ ] No generated or large files staged.
