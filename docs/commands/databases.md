# Structure databases

[Documentation home](../index.md)

The `database` family is also available as `db`; `mp` is the Materials Project spelling of `database --database mp`.

`abacustools database` searches and downloads crystal structures from the
materials databases that are open to everyone, and writes them as ABACUS or
common structure files:

```text
abacustools database list                     # what can be queried, and how
abacustools database providers                # the OPTIMADE providers behind -d optimade
abacustools database search -d cod --formula Fe2O3 --limit 5
abacustools database download -d aflow aflow:608f86003961ee94
```

Two access routes cover the databases that need no subscription. The first is
the Materials Project client (`mp-api`, an API key, the full set of summary
fields), registered as `mp`. The second is
[OPTIMADE](https://www.optimade.org), a REST protocol spoken by most other open
structure databases and implemented once here: every deployment is a database
name of its own, and `-d optimade` queries all of them and stops as soon as the
answer is full.

`ABACUSTOOLS_DATABASE` sets the database used when `--database` is omitted;
without it the default is `mp`.

## Available databases

| database | content | access route |
| --- | --- | --- |
| `mp` | Materials Project: DFT energies, stability, structures | `mp-api`, API key |
| `mp-optimade` | the public OPTIMADE endpoint of the Materials Project | OPTIMADE |
| `aflow` | AFLOW: calculated alloys and compounds | OPTIMADE |
| `oqmd` | OQMD: formation energies and thermodynamic stability | OPTIMADE |
| `nomad` | NOMAD: parsed ab initio calculations | OPTIMADE |
| `jarvis` | NIST JARVIS-DFT: optoelectronic and elastic data | OPTIMADE |
| `cod` | Crystallography Open Database: experimental structures | OPTIMADE |
| `tcod` | Theoretical Crystallography Open Database | OPTIMADE |
| `alexandria` | Alexandria materials database (PBE+SOL) | OPTIMADE |
| `c2db` | Computational 2D Materials Database (DTU), with its computed data | query table + OPTIMADE |
| `c2db-optimade` | the same structures over OPTIMADE only, without the computed data | OPTIMADE |
| `mc3d`, `mc2d` | Materials Cloud three- and two-dimensional crystals | OPTIMADE |
| `twodmatpedia` | 2DMatPedia: 2D materials exfoliated from the Materials Project | OPTIMADE |
| `matterverse` | Matterverse: machine-learning property predictions | OPTIMADE |
| `odbx` | Open Database of Xtals | OPTIMADE |
| `mpds` | Materials Platform for Data Science | OPTIMADE, token |
| `optimade` | every catalogued provider at once | OPTIMADE |

`abacustools database list` shows which databases are ready, which need an API
key, and which selectors each one accepts. Databases reached over OPTIMADE
accept `--formula`, `--chemsys`, `--elements` and `--id`; the Materials Project
also accepts `--stable`, `--theoretical` and `--fields`, and C2DB adds
`--where` for its own property expressions and `--show` for extra columns (see
below). Asking a database for a selector it does not know is an error rather
than a silent partial match.

The OPTIMADE catalogue follows the official index at
`https://providers.optimade.org/providers.json`; `abacustools database
providers --refresh` prints the live list, and `--base-url` sends a query to an
OPTIMADE endpoint that the catalogue does not contain. Providers differ in what
they publish: `cod` and `tcod` report cell parameters but no atomic
coordinates, so they answer searches while a download of one of their entries
reports that there is no structure to write; `c2db-optimade` ignores filters on
the entry id and does not publish the computed data, which is why the `c2db`
database reads the query table of the C2DB web application instead.

## C2DB computed data

C2DB stores much more than the geometry: PBE, HSE06 and G0W0 band gaps, the
energy above the convex hull, the heat of formation, effective masses, elastic
and piezoelectric constants, magnetic states, optical properties and so on.
`abacustools database fields -d c2db` lists all 88 keys with their units, and
`--where` filters on them with the expression language of the C2DB search page:

```text
abacustools database fields -d c2db
abacustools database search -d c2db --formula MoS2 --limit 5
abacustools database search -d c2db --elements Mo S --where 'gap>1.5' --limit 5
abacustools database search -d c2db --where 'is_magnetic=True' --where 'ehull<0.05'
abacustools database search -d c2db --formula MoS2 --show gap_hse,emass_cbm
```

A `--where` expression compares one key, as in `gap>1.5`, `ehull<0.05`,
`xc=PBE` or `nspecies=3`; several expressions and the standard selectors are
combined with "and", `|` combines alternatives and `~` negates a term, exactly
as on the search page. `--show` adds the named keys as columns, and every
search record also carries them under `extra` in `--json`. Entries are
identified by their C2DB uid, such as `1MoS2-1`:

```text
abacustools database download -d c2db 1MoS2-1 --format cif
abacustools database download -d c2db 1MoS2-1 --json
```

A download joins the property row of the query table with the geometry of the
OPTIMADE endpoint, so the JSON record reports the formula, the site count, the
PBE gap, the energy above the hull, and the tabulated properties.

## Materials Project

The `mp` command family is the short spelling of `database --database mp`, kept
so that existing scripts keep working. It needs the optional `mp-api` client
and an API key:

```bash
pip install 'abacustools[mp]'
export MP_API_KEY="your_key_here"   # https://materialsproject.org/api
```

Searches need at least one of `--formula`, `--chemsys`, `--elements`, or
`--id` (also spelled `--material-id`):

```text
abacustools mp search --formula Fe2O3 --limit 10
abacustools mp search --chemsys Li-Fe-O --stable --limit 5 --json
abacustools database search -d mp --elements Li Fe O --output li-fe-o.json
abacustools database search -d cod --formula Fe2O3 --limit 5
```

The table reports the database, the identifier, the formula, the chemical
system, the number of sites, the energy above the convex hull, the band gap,
stability, and whether the entry is theoretical. `--json` prints the same
records as JSON and `--output` also writes them to a file. The command returns
a non-zero exit status when nothing matches the query.

Downloading writes one directory per entry, which can be used as an ABACUS job
directory directly:

```text
abacustools mp download mp-149
abacustools mp download mp-149 mp-22862 --output structures --format poscar
abacustools database download -d cod 1000000 --format cif
abacustools database download -d aflow aflow:608f86003961ee94 --group-by-database
```

Every structure goes to `OUTPUT/<id>/`, named `STRU`, `POSCAR`,
`structure.cif`, `structure.xyz`, `structure.extxyz`, or `structure.xsf`
according to `--format`; `--group-by-database` adds the database name as
another directory level, which keeps identifiers of different databases apart.
Downloaded structures carry no pseudopotential or numerical-orbital
information, so the `ATOMIC_SPECIES` files required by ABACUS still have to be
filled in, for example with the library settings described above or
`abacustools file stru`.

Each JSON record holds `database`, `id`, `formula`, `chemsys`, `nsites`,
`volume`, `energy_above_hull`, `band_gap`, `is_stable`, `theoretical`, and
`extra`, where `extra` keeps the provider-specific values of the entry.

## Python API

Every database is reached through the same interface, and each call accepts a
pre-constructed client or transport so that no connection is made when one is
supplied:

```python
from pathlib import Path
from abacustools.integrations.databases import (
    DatabaseQuery,
    get_database,
    structure_path,
    write_structure,
)

database = get_database("cod")
for summary in database.search(DatabaseQuery(formula="Fe2O3", limit=5)):
    print(summary.database, summary.identifier, summary.formula, summary.chemsys)

structure = database.fetch(summary.identifier)
write_structure(
    structure,
    structure_path(Path("structures"), structure.identifier, database="cod"),
)
```

`get_database(name)` returns the adapter of any registered database, and
`databases()`, `database_names()`, `describe_databases()` and
`default_database()` describe the registry. The Materials Project adapter is
also available on its own, as before:

```python
from abacustools.integrations.materials_project import (
    material_directory,
    search_materials,
    download_material,
    write_material_structure,
)

for summary in search_materials(chemsys="Li-Fe-O", is_stable=True, limit=5):
    print(summary.material_id, summary.formula, summary.energy_above_hull)

material = download_material("mp-149")
write_material_structure(
    material,
    material_directory(Path("structures"), material.material_id),
)
```

`load_materials_project()` and `materials_project_available()` report whether
the optional client can be imported. The API key is taken from the first of
`MP_API_KEY`, `PMG_MAPI_KEY`, and `MAPI_KEY` that is set, or from an explicit
`--api-key` / `api_key=` argument.
