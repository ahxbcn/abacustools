"""Tests for the ``abacustools database`` command family."""

from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from abacustools.commands.database.download import run as run_download
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
    capabilities = frozenset({"formula", "elements", "identifiers", "stability"})
    options = frozenset({"base_url"})

    def __init__(self):
        self.queries = []
        self.base_urls = []

    def search(self, query, *, api_key=None, client=None, **options):
        self.queries.append(query)
        self.base_urls.append(options.get("base_url"))
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


def _search_namespace(*arguments: str):
    return _create_parser("abacustools").parse_args(["database", "search", *arguments])


def _download_namespace(*arguments: str):
    return _create_parser("abacustools").parse_args(["database", "download", *arguments])


class TestDatabaseParsers(unittest.TestCase):
    def test_database_family_is_registered(self):
        namespace = _search_namespace("--formula", "Si")

        self.assertEqual(namespace.command, "database")
        self.assertEqual(namespace.database_command, "search")
        self.assertIsNotNone(namespace.handler)

    def test_database_has_a_short_alias(self):
        namespace = _create_parser("abacustools").parse_args(["db", "list"])

        self.assertEqual(namespace.command, "db")
        self.assertEqual(namespace.database_command, "list")

    def test_mp_family_still_works_and_selects_the_materials_project(self):
        namespace = _create_parser("abacustools").parse_args(["mp", "search", "--formula", "Si"])

        self.assertEqual(namespace.command, "mp")
        self.assertEqual(namespace.mp_command, "search")
        self.assertEqual(namespace.database, "mp")

    def test_every_subcommand_is_available(self):
        parser = _create_parser("abacustools")

        for subcommand in ("list", "providers", "search", "download"):
            namespace = parser.parse_args(
                ["database", subcommand, *(("demo-1",) if subcommand == "download" else ())]
            )
            self.assertEqual(namespace.database_command, subcommand)

    def test_download_requires_an_identifier(self):
        parser = _create_parser("abacustools")
        with patch("sys.stderr", new_callable=StringIO):
            with self.assertRaises(SystemExit):
                parser.parse_args(["database", "download"])


class TestDatabaseSearchCommand(unittest.TestCase):
    def setUp(self):
        self.database = DemoDatabase()
        register_database(self.database)

    def tearDown(self):
        unregister_database("demo")

    def test_search_prints_a_table_of_matches(self):
        namespace = _search_namespace("-d", "demo", "--formula", "Si")
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            status = run_search(namespace)

        self.assertEqual(status, 0)
        self.assertIn("demo-1", stdout.getvalue())
        self.assertIn("Si2", stdout.getvalue())
        self.assertIn("0.610", stdout.getvalue())

    def test_search_passes_the_selectors_to_the_database(self):
        namespace = _search_namespace(
            "-d", "demo", "--formula", "Si", "--elements", "Li", "O", "--stable", "--limit", "7"
        )
        with patch("sys.stdout", new_callable=StringIO):
            run_search(namespace)

        query = self.database.queries[0]
        self.assertEqual(query.formula, "Si")
        self.assertEqual(query.elements, ("Li", "O"))
        self.assertTrue(query.is_stable)
        self.assertEqual(query.limit, 7)

    def test_search_accepts_repeated_identifiers(self):
        namespace = _search_namespace("-d", "demo", "--id", "demo-1", "--material-id", "demo-2")
        with patch("sys.stdout", new_callable=StringIO):
            run_search(namespace)

        self.assertEqual(self.database.queries[0].identifiers, ("demo-1", "demo-2"))

    def test_search_prints_json_and_writes_the_output_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "results.json"
            namespace = _search_namespace(
                "-d", "demo", "--formula", "Si", "--json", "--output", str(destination)
            )
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                status = run_search(namespace)

            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue())[0]["id"], "demo-1")
            self.assertEqual(json.loads(destination.read_text())[0]["band_gap"], 0.61)

    def test_search_without_matches_returns_one(self):
        namespace = _search_namespace("-d", "demo", "--formula", "none")
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            status = run_search(namespace)

        self.assertEqual(status, 1)
        self.assertIn("no entries matched", stdout.getvalue())

    def test_search_without_a_selector_fails_before_reaching_the_database(self):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            status = run_search(_search_namespace("-d", "demo"))

        self.assertEqual(status, 1)
        self.assertIn("selector", stderr.getvalue())
        self.assertEqual(self.database.queries, [])

    def test_search_reports_a_selector_the_database_does_not_support(self):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            status = run_search(_search_namespace("-d", "demo", "--formula", "Si", "--theoretical"))

        self.assertEqual(status, 1)
        self.assertIn("theoretical", stderr.getvalue())

    def test_search_reports_an_unknown_database(self):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            status = run_search(_search_namespace("-d", "nope", "--formula", "Si"))

        self.assertEqual(status, 1)
        self.assertIn("unknown database", stderr.getvalue())

    def test_search_passes_provider_options_through(self):
        namespace = _search_namespace(
            "-d", "demo", "--formula", "Si", "--base-url", "https://demo.example/optimade"
        )
        with patch("sys.stdout", new_callable=StringIO):
            status = run_search(namespace)

        self.assertEqual(status, 0)
        self.assertEqual(self.database.base_urls[0], "https://demo.example/optimade")

    def test_search_reports_an_error_from_the_database(self):
        with patch.object(self.database, "search", side_effect=DatabaseRequestError("boom")):
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                status = run_search(_search_namespace("-d", "demo", "--formula", "Si"))

        self.assertEqual(status, 1)
        self.assertIn("boom", stderr.getvalue())


class TestDatabaseDownloadCommand(unittest.TestCase):
    def setUp(self):
        self.database = DemoDatabase()
        register_database(self.database)

    def tearDown(self):
        unregister_database("demo")

    def test_download_writes_one_directory_per_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace = _download_namespace("-d", "demo", "demo-1", "-o", temporary, "-f", "poscar")
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                status = run_download(namespace)

            written = Path(temporary) / "demo-1" / "POSCAR"
            self.assertTrue(written.is_file())

        self.assertEqual(status, 0)
        self.assertIn("demo-1", stdout.getvalue())
        self.assertIn("POSCAR", stdout.getvalue())

    def test_download_defaults_to_stru(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace = _download_namespace("-d", "demo", "demo-1", "-o", temporary)
            with patch("sys.stdout", new_callable=StringIO):
                status = run_download(namespace)

            written = Path(temporary) / "demo-1" / "STRU"
            self.assertTrue(written.is_file())
            self.assertIn("ATOMIC_SPECIES", written.read_text())

        self.assertEqual(status, 0)

    def test_download_can_group_entries_by_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace = _download_namespace(
                "-d", "demo", "demo-1", "-o", temporary, "--group-by-database"
            )
            with patch("sys.stdout", new_callable=StringIO):
                run_download(namespace)

            self.assertTrue((Path(temporary) / "demo" / "demo-1" / "STRU").is_file())

    def test_download_prints_json_when_asked(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace = _download_namespace("-d", "demo", "demo-1", "-o", temporary, "--json")
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                status = run_download(namespace)

        self.assertEqual(status, 0)
        record = json.loads(stdout.getvalue())[0]
        self.assertEqual(record["id"], "demo-1")
        self.assertEqual(record["database"], "demo")
        self.assertTrue(record["file"].endswith("demo-1/STRU"))

    def test_download_reports_unknown_entries_and_returns_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace = _download_namespace("-d", "demo", "missing", "-o", temporary)
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                status = run_download(namespace)

        self.assertEqual(status, 1)
        self.assertIn("missing", stderr.getvalue())

    def test_download_keeps_going_after_one_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace = _download_namespace("-d", "demo", "missing", "demo-1", "-o", temporary)
            with (
                patch("sys.stdout", new_callable=StringIO),
                patch("sys.stderr", new_callable=StringIO),
            ):
                status = run_download(namespace)

            self.assertTrue((Path(temporary) / "demo-1" / "STRU").is_file())
            self.assertFalse((Path(temporary) / "missing").exists())

        self.assertEqual(status, 1)


class TestDatabaseListingCommand(unittest.TestCase):
    def test_list_prints_the_registered_databases(self):
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            status = run_list(_create_parser("abacustools").parse_args(["database", "list"]))

        self.assertEqual(status, 0)
        self.assertIn("mp", stdout.getvalue())
        self.assertIn("optimade", stdout.getvalue())

    def test_list_prints_json(self):
        namespace = _create_parser("abacustools").parse_args(["database", "list", "--json"])
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            status = run_list(namespace)

        records = json.loads(stdout.getvalue())
        names = [record["name"] for record in records]
        self.assertEqual(status, 0)
        self.assertIn("cod", names)
        self.assertTrue(all("status" in record for record in records))

    def test_list_can_keep_only_ready_databases(self):
        namespace = _create_parser("abacustools").parse_args(
            ["database", "list", "--available", "--json"]
        )
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            run_list(namespace)

        self.assertTrue(
            all(record["status"] == "ready" for record in json.loads(stdout.getvalue()))
        )


class TestDatabaseProvidersCommand(unittest.TestCase):
    def test_providers_prints_the_bundled_catalogue(self):
        namespace = _create_parser("abacustools").parse_args(["database", "providers"])
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            status = run_providers(namespace)

        self.assertEqual(status, 0)
        self.assertIn("aflow", stdout.getvalue())
        self.assertIn("mp-optimade", stdout.getvalue())

    def test_providers_prints_json(self):
        namespace = _create_parser("abacustools").parse_args(["database", "providers", "--json"])
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            status = run_providers(namespace)

        records = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertGreaterEqual(len(records), 10)
        self.assertTrue(all(record["endpoint"].startswith("http") for record in records))

    def test_refresh_reads_the_live_index(self):
        payload = {
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
        namespace = _create_parser("abacustools").parse_args(
            ["database", "providers", "--refresh", "--json"]
        )
        with patch(
            "abacustools.integrations.databases.optimade._http_get_json", return_value=payload
        ):
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                status = run_providers(namespace)

        records = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(records[0]["name"], "demo")
        self.assertEqual(records[0]["description"], "Demo provider")


class TestMaterialsProjectAlias(unittest.TestCase):
    def _summary(self) -> MaterialSummary:
        return MaterialSummary(
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

    def test_mp_search_uses_the_materials_project_database(self):
        namespace = _create_parser("abacustools").parse_args(
            ["mp", "search", "--formula", "Si", "--limit", "5"]
        )
        with patch(
            "abacustools.integrations.databases.materials_project.search_materials",
            return_value=[self._summary()],
        ) as search:
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                status = run_search(namespace)

        self.assertEqual(status, 0)
        self.assertIn("mp-149", stdout.getvalue())
        self.assertEqual(search.call_args.kwargs["formula"], "Si")
        self.assertEqual(search.call_args.kwargs["limit"], 5)

    def test_mp_search_reports_a_missing_api_key(self):
        namespace = _create_parser("abacustools").parse_args(["mp", "search", "--formula", "Si"])
        with patch(
            "abacustools.integrations.databases.materials_project.search_materials",
            side_effect=MaterialsProjectApiKeyError("no Materials Project API key found"),
        ):
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                status = run_search(namespace)

        self.assertEqual(status, 1)
        self.assertIn("no Materials Project API key found", stderr.getvalue())

    def test_mp_download_writes_the_structure(self):
        material = MaterialStructure(summary=self._summary(), structure=_structure())
        namespace = _create_parser("abacustools").parse_args(["mp", "download", "mp-149"])
        with tempfile.TemporaryDirectory() as temporary:
            namespace.output = Path(temporary)
            with patch(
                "abacustools.integrations.databases.materials_project.download_material",
                return_value=material,
            ):
                with patch("sys.stdout", new_callable=StringIO):
                    status = run_download(namespace)

            self.assertTrue((Path(temporary) / "mp-149" / "STRU").is_file())

        self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
