# Structure databases

`abacustools database` reaches every open structure database through one
interface; `abacustools mp` is the short spelling of `database -d mp`.

| command | purpose |
| --- | --- |
| `database list` | Which databases exist, whether they are ready (API key present), which selectors each accepts |
| `database fields -d NAME` | Fields a database publishes, with units (C2DB: 88 keys) |
| `database providers` | OPTIMADE providers behind `-d optimade`; `--refresh` queries the live index, `--base-url` adds an endpoint |
| `database search` | Summaries (table, `--json`, `--output FILE`) |
| `database download` | Fetch structures into OUTPUT/ID/ (default `structures/`) |

## Access routes

- **Materials Project** (`mp`, needs the optional `mp-api` client and a key):
  the richest field set. Key is taken from the first of `MP_API_KEY`,
  `PMG_MAPI_KEY`, `MAPI_KEY`, or an explicit `--api-key`. Install with
  `pip install 'abacustools[mp]'`.
- **OPTIMADE** (`mp-optimade`, `aflow`, `oqmd`, `nomad`, `jarvis`, `cod`,
  `tcod`, `alexandria`, `c2db-optimade`, `mc3d`, `mc2d`, `twodmatpedia`,
  `matterverse`, `odbx`, `mpds`, and the aggregate `optimade`): no extra package,
  no key for most. The aggregate queries every catalogued provider and stops
  when the answer is full.
- **C2DB** (`c2db`): its own adapter, because only the web query table publishes
  the computed properties; the geometry still comes from OPTIMADE.

Providers differ in what they publish: `cod`/`tcod` report cell parameters but
no coordinates, so they answer searches while a download reports there is no
structure to write.

## Selectors

- Every OPTIMADE database: `--formula`, `--chemsys`, `--elements`, `--id`.
- Materials Project additionally: `--stable`, `--theoretical`, `--fields`.
- C2DB additionally: `--where EXPR` (its search-page language, repeatable,
  combined with "and"; `|` for alternatives, `~` to negate) and `--show key,key`
  for extra columns. Examples: `--where 'gap>1.5'`, `--where 'is_magnetic=True'`,
  `--where 'ehull<0.05'`, `--show gap_hse,emass_cbm`.

A selector a database does not know is an error rather than a silent partial
match, and a search with no hit exits non-zero.

## JSON record

Each search record holds `database`, `id`, `formula`, `chemsys`, `nsites`,
`volume`, `energy_above_hull`, `band_gap`, `is_stable`, `theoretical`, and
`extra` with the provider-specific values.

## Downloads

`--format` selects `STRU`, `POSCAR`, `structure.cif`, `structure.xyz`,
`structure.extxyz`, or `structure.xsf`; `--group-by-database` inserts the
database name as a directory level so equal identifiers from different
databases stay apart. Downloaded structures have no pseudopotential or orbital
information: fill in `ATOMIC_SPECIES` before running ABACUS, for example by
preparing the job with `--library`.

## Python API

```python
from pathlib import Path
from abacustools.integrations.databases import (
    DatabaseQuery, get_database, structure_path, write_structure,
)

database = get_database("cod")
for summary in database.search(DatabaseQuery(formula="Fe2O3", limit=5)):
    print(summary.database, summary.identifier, summary.formula)

structure = database.fetch(summary.identifier)
write_structure(structure, structure_path(Path("structures"), structure.identifier))
```

`get_database(name)` returns the adapter of any registered database;
`databases()`, `database_names()`, `describe_databases()`, and
`default_database()` describe the registry. `ABACUSTOOLS_DATABASE` sets the
database used when `--database` is omitted (default `mp`). Every query needs
network access; the Materials Project route additionally needs the `mp-api`
package and an API key.
