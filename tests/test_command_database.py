"""Tests for the ``abacustools database`` command family."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest

from abacustools.commands.database.download import run as run_download
from abacustools.commands.database.fields import run as run_fields
from abacustools.commands.database.listing import run as run_list
from abacustools.commands.database.providers import run as run_providers
from abacustools.commands.database.search import run as run_search
from abacustools.integrations.databases import (
    DatabaseRequestError,
    DatabaseStructure,
    DatabaseSummary,
    StructureDatabase,
    register_database,
    unregister_database,
)
from abacustools.integrations.materials_project import (
    MaterialStructure,
    MaterialSummary,
    MaterialsProjectApiKeyError,
)
from abacustools.main import _create_parser


def _structure():
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


class DemoDatabase(StructureDatabase):
    """Database stand-in that never touches the network."""

    name = "demo"
    description = "demo database"
    protocol = "demo"
    capabilities = frozenset({"formula", "elements", "identifiers", "stability", "where"})
    options = frozenset({"base_url", "show"})

    def __init__(self):
        self.queries = []
        self.base_urls = []
        self.shows = []

    def search(self, query, *, api_key=None, client=None, **options):
        self.queries.append(query)
        self.base_urls.append(options.get("base_url"))
        self.shows.append(options.get("show"))
        if query.formula == "none":
            return []
        return [
            DatabaseSummary(
                database=self.name,
                identifier="demo-1",
                formula="Si2",
                chemsys="Si",
                nsites=2,
                band_gap=0.61,
                is_stable=True,
                extra={"uid": "demo-1", "gap_hse": "2.087"},
            )
        ]

    def fetch(self, identifier, *, api_key=None, client=None, **options):
        if identifier == "missing":
            raise LookupError(f"{self.name} has no entry {identifier!r}")
        return DatabaseStructure(
            summary=DatabaseSummary(
                database=self.name, identifier=identifier, formula="Si2", nsites=2
            ),
            structure=_structure(),
        )


@pytest.fixture
def demo():
    """Register the demo database for one test."""
    database = DemoDatabase()
    register_database(database)
    yield database
    unregister_database("demo")


def _parse(*argv: str):
    return _create_parser("abacustools").parse_args(list(argv))


def _search(*arguments: str):
    return _parse("database", "search", *arguments)


def _download(*arguments: str):
    return _parse("database", "download", *arguments)


def test_the_family_and_its_subcommands_are_registered():
    namespace = _search("--formula", "Si")
    assert (namespace.command, namespace.database_command) == ("database", "search")
    assert namespace.handler is not None

    assert _parse("db", "list").database_command == "list"
    alias = _parse("mp", "search", "--formula", "Si")
    assert (alias.command, alias.mp_command, alias.database) == ("mp", "search", "mp")

    for subcommand, extra in (
        ("list", ()),
        ("fields", ()),
        ("providers", ()),
        ("search", ()),
        ("download", ("demo-1",)),
    ):
        namespace = _parse("database", subcommand, *extra)
        assert namespace.database_command == subcommand

    with patch("sys.stderr", new_callable=StringIO):
        with pytest.raises(SystemExit):
            _parse("database", "download")


def test_search_prints_a_table_json_and_a_file(demo, tmp_path):
    namespace = _search("-d", "demo", "--formula", "Si")
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        status = run_search(namespace)
    assert status == 0
    assert "demo-1" in stdout.getvalue() and "0.610" in stdout.getvalue()

    destination = tmp_path / "results.json"
    namespace = _search("-d", "demo", "--formula", "Si", "--json", "--output", str(destination))
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        status = run_search(namespace)
    assert status == 0
    assert json.loads(stdout.getvalue())[0]["id"] == "demo-1"
    assert json.loads(destination.read_text())[0]["band_gap"] == 0.61

    with patch("sys.stdout", new_callable=StringIO) as stdout:
        status = run_search(_search("-d", "demo", "--formula", "none"))
    assert status == 1
    assert "no entries matched" in stdout.getvalue()


def test_search_passes_the_query_and_the_options_through(demo):
    namespace = _search(
        "-d",
        "demo",
        "--formula",
        "Si",
        "--elements",
        "Li",
        "O",
        "--stable",
        "--limit",
        "7",
        "--where",
        "gap>1.5",
        "--show",
        "gap_hse,magstate",
        "--base-url",
        "https://demo.example/optimade",
    )
    with patch("sys.stdout", new_callable=StringIO):
        status = run_search(namespace)

    assert status == 0
    query = demo.queries[0]
    assert (query.formula, query.elements, query.limit) == ("Si", ("Li", "O"), 7)
    assert query.is_stable is True
    assert query.where == ("gap>1.5",)
    assert demo.shows[0] == ["gap_hse", "magstate"]
    assert demo.base_urls[0] == "https://demo.example/optimade"

    namespace = _search("-d", "demo", "--formula", "Si", "--show", "gap_hse")
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_search(namespace) == 0
    assert "2.087" in stdout.getvalue()


def test_search_reports_bad_input(demo):
    cases = [
        (_search("-d", "demo"), "selector"),
        (_search("-d", "demo", "--formula", "Si", "--theoretical"), "theoretical"),
        (_search("-d", "nope", "--formula", "Si"), "unknown database"),
        (_search("-d", "mp", "--formula", "Si", "--where", "gap>1"), "where"),
    ]
    for namespace, message in cases:
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            assert run_search(namespace) == 1
        assert message in stderr.getvalue()
    assert demo.queries == []

    with patch.object(demo, "search", side_effect=DatabaseRequestError("boom")):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            assert run_search(_search("-d", "demo", "--formula", "Si")) == 1
    assert "boom" in stderr.getvalue()


def test_download_writes_the_files_it_reports(demo, tmp_path):
    namespace = _download("-d", "demo", "demo-1", "-o", str(tmp_path))
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        status = run_download(namespace)
    written = tmp_path / "demo-1" / "STRU"
    assert status == 0
    assert written.is_file() and "ATOMIC_SPECIES" in written.read_text()
    assert "demo-1" in stdout.getvalue()

    grouped = tmp_path / "grouped"
    namespace = _download(
        "-d", "demo", "demo-1", "-o", str(grouped), "-f", "poscar", "--group-by-database"
    )
    with patch("sys.stdout", new_callable=StringIO):
        assert run_download(namespace) == 0
    assert (grouped / "demo" / "demo-1" / "POSCAR").is_file()

    namespace = _download("-d", "demo", "demo-1", "-o", str(tmp_path), "--json")
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_download(namespace) == 0
    record = json.loads(stdout.getvalue())[0]
    assert (record["database"], record["id"]) == ("demo", "demo-1")
    assert record["file"].endswith("demo-1/STRU")


def test_download_keeps_going_after_one_failure(demo, tmp_path):
    namespace = _download("-d", "demo", "missing", "demo-1", "-o", str(tmp_path))
    with (
        patch("sys.stdout", new_callable=StringIO),
        patch("sys.stderr", new_callable=StringIO) as stderr,
    ):
        status = run_download(namespace)

    assert status == 1
    assert "missing" in stderr.getvalue()
    assert (tmp_path / "demo-1" / "STRU").is_file()
    assert not (tmp_path / "missing").exists()


def test_list_reports_the_databases():
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_list(_parse("database", "list")) == 0
    assert "mp" in stdout.getvalue() and "optimade" in stdout.getvalue()

    namespace = _parse("database", "list", "--json")
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_list(namespace) == 0
    records = json.loads(stdout.getvalue())
    assert "cod" in [record["name"] for record in records]
    assert all("status" in record for record in records)

    namespace = _parse("database", "list", "--available", "--json")
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        run_list(namespace)
    assert all(record["status"] == "ready" for record in json.loads(stdout.getvalue()))


def test_fields_lists_the_property_keys():
    namespace = _parse("database", "fields", "-d", "c2db", "--json")
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_fields(namespace) == 0
    records = json.loads(stdout.getvalue())
    assert len(records) >= 80
    assert {"key": "gap", "description": "Band gap (PBE) [eV]"} in records

    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_fields(_parse("database", "fields", "-d", "c2db")) == 0
    assert "ehull" in stdout.getvalue()

    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_fields(_parse("database", "fields", "-d", "mp")) == 0
    assert "does not document property keys" in stdout.getvalue()


def test_providers_lists_and_refreshes_the_catalogue():
    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_providers(_parse("database", "providers")) == 0
    assert "aflow" in stdout.getvalue() and "mp-optimade" in stdout.getvalue()

    with patch("sys.stdout", new_callable=StringIO) as stdout:
        assert run_providers(_parse("database", "providers", "--json")) == 0
    records = json.loads(stdout.getvalue())
    assert len(records) >= 10
    assert all(record["endpoint"].startswith("http") for record in records)

    index = {
        "data": [
            {
                "id": "demo",
                "attributes": {
                    "base_url": "https://demo.example/index",
                    "homepage": "https://demo.example",
                    "description": "Demo provider\nMore text",
                },
            }
        ]
    }
    namespace = _parse("database", "providers", "--refresh", "--json")
    with patch("abacustools.integrations.databases.optimade._http_get_json", return_value=index):
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            assert run_providers(namespace) == 0
    assert json.loads(stdout.getvalue())[0] == {
        "name": "demo",
        "endpoint": "https://demo.example/index",
        "homepage": "https://demo.example",
        "description": "Demo provider",
        "note": "index meta-database",
        "requires_api_key": False,
    }


def test_the_mp_alias_uses_the_materials_project(tmp_path):
    summary = MaterialSummary(
        material_id="mp-149",
        formula="Si2",
        chemsys="Si",
        nsites=2,
        volume=40.0,
        energy_above_hull=0.0,
        band_gap=0.61,
        is_stable=True,
        theoretical=False,
    )
    namespace = _parse("mp", "search", "--formula", "Si", "--limit", "5")
    with patch(
        "abacustools.integrations.databases.materials_project.search_materials",
        return_value=[summary],
    ) as search:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            assert run_search(namespace) == 0
    assert "mp-149" in stdout.getvalue()
    assert search.call_args.kwargs["formula"] == "Si"
    assert search.call_args.kwargs["limit"] == 5

    with patch(
        "abacustools.integrations.databases.materials_project.search_materials",
        side_effect=MaterialsProjectApiKeyError("no Materials Project API key found"),
    ):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            assert run_search(_parse("mp", "search", "--formula", "Si")) == 1
    assert "no Materials Project API key found" in stderr.getvalue()

    material = MaterialStructure(summary=summary, structure=_structure())
    namespace = _parse("mp", "download", "mp-149", "-o", str(tmp_path))
    with patch(
        "abacustools.integrations.databases.materials_project.download_material",
        return_value=material,
    ):
        with patch("sys.stdout", new_callable=StringIO):
            assert run_download(namespace) == 0
    assert (tmp_path / "mp-149" / "STRU").is_file()
