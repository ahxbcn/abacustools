"""Tests for the pluggable structure-database layer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

from abacustools.integrations.databases import (
    DatabaseQuery,
    c2db_keys,
    c2db_keys_source,
    DatabaseRequestError,
    DatabaseStructure,
    DatabaseSummary,
    OptimadeHttpError,
    OptimadeProvider,
    StructureDatabase,
    describe_databases,
    get_database,
    optimade_catalogue,
    optimade_catalogue_source,
    optimade_provider_from_name,
    register_database,
    structure_filename,
    structure_path,
    unregister_database,
    write_structure,
)
from abacustools.integrations.databases.c2db import (
    DEFAULT_COLUMNS,
    build_filter,
    parse_table,
    summary_from_row,
)
from abacustools.integrations.databases.optimade import (
    OptimadeDatabase,
    build_filters,
    build_strategies,
    formula_matches,
    reduced_formula,
    structure_from_entry,
    summary_from_entry,
)
from abacustools.integrations.materials_project import MaterialSummary


def _structure():
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _entry(identifier: str = "demo-1", **attributes):
    fields = {
        "chemical_formula_reduced": "Fe2O3",
        "chemical_formula_descriptive": "Fe2O3",
        "elements": ["Fe", "O"],
        "nsites": 5,
        "lattice_vectors": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "cartesian_site_positions": [[float(i)] * 3 for i in range(5)],
        "species_at_sites": ["Fe", "Fe", "O", "O", "O"],
    }
    fields.update(attributes)
    return {"id": identifier, "type": "structures", "attributes": fields}


def _response(entries, next_link=None):
    links = {"next": next_link} if next_link else {}
    return {"data": list(entries), "links": links, "meta": {"data_returned": len(entries)}}


def _query(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)


class RecordingTransport:
    """A stand-in HTTP transport that records the URLs it was asked for."""

    def __init__(self, responder):
        self.responder = responder
        self.urls: list[str] = []

    def __call__(self, url):
        self.urls.append(url)
        return self.responder(url)


class DemoDatabase(StructureDatabase):
    """Minimal database used to test the registry and the commands."""

    name = "demo"
    description = "demo database"
    protocol = "demo"
    capabilities = frozenset({"formula", "identifiers", "stability"})

    def search(self, query, *, api_key=None, client=None, **options):
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


class TestRegistry(unittest.TestCase):
    def test_bundled_databases_are_registered(self):
        from abacustools.integrations.databases import database_names

        names = database_names()
        for name in ("mp", "optimade", "aflow", "cod", "nomad", "jarvis"):
            self.assertIn(name, names)

    def test_aliases_resolve_to_the_same_adapter(self):
        self.assertIs(get_database("mp"), get_database("materials-project"))
        self.assertIs(get_database("optimade"), get_database("optimade-federation"))

    def test_unknown_database_lists_the_known_ones(self):
        with self.assertRaises(DatabaseRequestError) as error:
            get_database("nope")

        self.assertIn("mp", str(error.exception))

    def test_register_and_unregister(self):
        database = DemoDatabase()
        try:
            register_database(database, aliases=("demo-alias",))
            self.assertIs(get_database("demo-alias"), database)
        finally:
            unregister_database("demo")

        with self.assertRaises(DatabaseRequestError):
            get_database("demo-alias")

    def test_describe_databases_reports_every_database_once(self):
        records = describe_databases()
        names = [record["name"] for record in records]

        self.assertEqual(len(names), len(set(names)))
        self.assertIn("mp", names)
        for record in records:
            self.assertIn(record["status"], {"ready", "needs-api-key", "unavailable"})


class TestDatabaseQuery(unittest.TestCase):
    def test_selectors_report_what_the_query_uses(self):
        query = DatabaseQuery(formula="Si", elements=("Li", "O"), is_stable=True, limit=5)

        self.assertEqual(query.selectors(), ["formula", "elements", "stability"])

    def test_a_query_without_a_selector_is_rejected(self):
        with self.assertRaises(DatabaseRequestError):
            DatabaseQuery().require_selector()

    def test_unsupported_selectors_are_reported(self):
        query = DatabaseQuery(formula="Si", theoretical=True)

        self.assertEqual(query.unsupported(frozenset({"formula"})), ["theoretical"])


class TestDatabaseSummary(unittest.TestCase):
    def test_serialisation_keeps_the_common_fields(self):
        summary = DatabaseSummary(database="demo", identifier="demo-1", formula="Si2")

        record = summary.to_dict()

        self.assertEqual(record["database"], "demo")
        self.assertEqual(record["id"], "demo-1")
        self.assertEqual(record["extra"], {})

    def test_structure_converts_to_abacus_metadata(self):
        structure = DatabaseStructure(
            summary=DatabaseSummary(database="demo", identifier="demo-1"),
            structure=_structure(),
        )

        abacus = structure.to_abacus_structure()

        self.assertEqual(abacus.metadata["source"], "demo")
        self.assertEqual(abacus.metadata["identifier"], "demo-1")


class TestMaterialsProjectDatabase(unittest.TestCase):
    def test_search_maps_summaries_and_keeps_the_material_id(self):
        document = MaterialSummary(
            material_id="mp-149", formula="Si2", chemsys="Si", nsites=2, band_gap=0.61
        )
        database = get_database("mp")

        with patch(
            "abacustools.integrations.databases.materials_project.search_materials",
            return_value=[document],
        ) as search:
            summaries = database.search(DatabaseQuery(formula="Si", limit=5))

        self.assertEqual(summaries[0].database, "mp")
        self.assertEqual(summaries[0].identifier, "mp-149")
        self.assertEqual(summaries[0].extra["material_id"], "mp-149")
        self.assertEqual(search.call_args.kwargs["formula"], "Si")
        self.assertEqual(search.call_args.kwargs["limit"], 5)

    def test_search_needs_a_selector(self):
        with self.assertRaises(DatabaseRequestError):
            get_database("mp").search(DatabaseQuery())

    def test_a_stability_filter_reaches_the_adapter(self):
        database = get_database("mp")

        with patch(
            "abacustools.integrations.databases.materials_project.search_materials",
            return_value=[],
        ) as search:
            database.search(DatabaseQuery(chemsys="Li-O", is_stable=True))

        self.assertTrue(search.call_args.kwargs["is_stable"])

    def test_fetch_returns_a_database_structure(self):
        from abacustools.integrations.materials_project import MaterialStructure

        material = MaterialStructure(
            summary=MaterialSummary(material_id="mp-149", formula="Si2"),
            structure=_structure(),
        )

        with patch(
            "abacustools.integrations.databases.materials_project.download_material",
            return_value=material,
        ):
            structure = get_database("mp").fetch("mp-149")

        self.assertEqual(structure.identifier, "mp-149")
        self.assertEqual(len(structure.structure), 2)

    def test_missing_client_is_reported_as_unavailable(self):
        database = get_database("mp")
        with patch.dict(sys.modules, {"mp_api": None, "mp_api.client": None}):
            self.assertFalse(database.available())
            self.assertEqual(database.status(), "unavailable")
            self.assertIn("mp-api", database.unavailable_reason())


class TestOptimadeCatalogue(unittest.TestCase):
    def test_catalogue_lists_queryable_providers(self):
        providers = optimade_catalogue()

        self.assertGreaterEqual(len(providers), 10)
        for provider in providers:
            self.assertTrue(provider.base_url.startswith("http"))
        names = [provider.name for provider in providers]
        for name in ("aflow", "c2db-optimade", "cod", "mc2d", "twodmatpedia"):
            self.assertIn(name, names)

    def test_a_provider_can_narrow_the_selectors_it_answers(self):
        provider = optimade_provider_from_name("c2db-optimade")

        self.assertEqual(provider.selectors, ("formula", "chemsys", "elements", "where"))
        database = get_database("c2db-optimade")
        self.assertFalse(database.supports("identifiers"))
        self.assertTrue(database.supports("where"))
        with self.assertRaises(DatabaseRequestError) as error:
            database.search(DatabaseQuery(identifiers=("3680",)))

        self.assertIn("identifiers", str(error.exception))

    def test_catalogue_records_where_it_came_from(self):
        source = optimade_catalogue_source()

        self.assertIn("optimade.org", source["source"])
        self.assertTrue(source["retrieved"])

    def test_providers_resolve_by_name_and_alias(self):
        self.assertEqual(optimade_provider_from_name("aflow").name, "aflow")
        self.assertEqual(
            optimade_provider_from_name("materials-project-optimade").name, "mp-optimade"
        )

    def test_unknown_provider_lists_the_catalogue(self):
        with self.assertRaises(DatabaseRequestError) as error:
            optimade_provider_from_name("nope")

        self.assertIn("aflow", str(error.exception))


class TestOptimadeFilters(unittest.TestCase):
    def test_identifiers_and_elements_become_standard_terms(self):
        query = DatabaseQuery(identifiers=("cod-1", "cod-2"), elements=("Fe", "O"))

        expression = build_filters(query)[0]

        self.assertIn('(id="cod-1" OR id="cod-2")', expression)
        self.assertIn('elements HAS ALL "Fe", "O"', expression)

    def test_a_lone_element_uses_has(self):
        self.assertIn('elements HAS "Li"', build_filters(DatabaseQuery(elements=("Li",)))[0])

    def test_a_chemical_system_becomes_an_element_filter(self):
        self.assertIn('elements HAS ALL "Li", "O"', build_filters(DatabaseQuery(chemsys="Li-O"))[0])

    def test_a_formula_offers_a_reduced_description_and_element_filter(self):
        strategies = build_strategies(DatabaseQuery(formula="Fe2O3"))

        self.assertIn('chemical_formula_reduced="Fe2O3"', strategies[0].expression)
        self.assertIn("CONTAINS", strategies[1].expression)
        self.assertTrue(strategies[1].check_formula)
        self.assertIn('elements HAS ALL "Fe", "O"', strategies[2].expression)
        self.assertTrue(strategies[2].check_formula)

    def test_an_empty_query_filters_nothing(self):
        self.assertEqual(build_filters(DatabaseQuery()), [None])

    def test_wildcards_are_rejected(self):
        with self.assertRaises(DatabaseRequestError):
            reduced_formula("Li*O")

    def test_an_unparsable_formula_is_rejected(self):
        with self.assertRaises(DatabaseRequestError):
            reduced_formula("not a formula")

    def test_formula_matching_compares_compositions(self):
        self.assertTrue(formula_matches("Fe2O3", "Fe2O3"))
        self.assertTrue(formula_matches("Fe4O6", "Fe2O3"))
        self.assertFalse(formula_matches("Fe2O3Si", "Fe2O3"))
        self.assertFalse(formula_matches(None, "Fe2O3"))


class TestOptimadeEntries(unittest.TestCase):
    def test_summary_reads_the_standard_attributes(self):
        summary = summary_from_entry(_entry(), "cod")

        self.assertEqual(summary.identifier, "demo-1")
        self.assertEqual(summary.formula, "Fe2O3")
        self.assertEqual(summary.chemsys, "Fe-O")
        self.assertEqual(summary.nsites, 5)

    def test_summary_derives_nsites_and_keeps_private_attributes(self):
        entry = _entry(nsites=None, _cod_flags="has coordinates")

        summary = summary_from_entry(entry, "cod")

        self.assertEqual(summary.nsites, 5)
        self.assertEqual(summary.extra["_cod_flags"], "has coordinates")

    def test_summary_reads_the_materials_project_hull_energy(self):
        entry = _entry(
            _mp_stability={"gga_gga+u": {"energy_above_hull": 0.25}},
            _mp_chemical_system="Fe-O",
        )

        summary = summary_from_entry(entry, "mp-optimade")

        self.assertAlmostEqual(summary.energy_above_hull, 0.25)

    def test_structure_is_built_from_the_coordinates(self):
        structure = structure_from_entry(_entry())

        self.assertEqual(len(structure), 5)
        self.assertEqual(structure.composition.reduced_formula, "Fe2O3")

    def test_an_entry_without_coordinates_is_rejected(self):
        entry = _entry(cartesian_site_positions=None, species_at_sites=None)

        with self.assertRaises(DatabaseRequestError):
            structure_from_entry(entry)


class TestOptimadeSearch(unittest.TestCase):
    def _database(self):
        return OptimadeDatabase(
            OptimadeProvider(name="demo", base_url="https://demo.example/optimade")
        )

    def test_search_queries_the_structures_endpoint(self):
        transport = RecordingTransport(lambda url: _response([_entry()]))

        summaries = self._database().search(
            DatabaseQuery(formula="Fe2O3", limit=1), transport=transport
        )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].database, "demo")
        self.assertEqual(summaries[0].extra["provider"], "demo")
        url = transport.urls[0]
        self.assertIn("/v1/structures", url)
        self.assertIn("chemical_formula_reduced", _query(url)["filter"][0])

    def test_limit_is_sent_and_applied(self):
        payload = _response([_entry("demo-1"), _entry("demo-2")])
        transport = RecordingTransport(lambda url: payload)

        summaries = self._database().search(
            DatabaseQuery(identifiers=("demo-1",), limit=1), transport=transport
        )

        self.assertEqual([summary.identifier for summary in summaries], ["demo-1"])
        self.assertLessEqual(int(_query(transport.urls[0])["page_limit"][0]), 100)

    def test_pagination_follows_next_links(self):
        first = _response(
            [_entry("demo-1")],
            next_link={"href": "https://demo.example/optimade/v1/structures?page_offset=1"},
        )
        second = _response([_entry("demo-2")])
        transport = RecordingTransport(
            lambda url: first if url.count("page_offset") == 0 else second
        )

        summaries = self._database().search(
            DatabaseQuery(identifiers=("demo-1", "demo-2"), limit=2), transport=transport
        )

        self.assertEqual([summary.identifier for summary in summaries], ["demo-1", "demo-2"])
        self.assertEqual(len(transport.urls), 2)

    def test_a_rejected_formula_filter_falls_back(self):
        def responder(url):
            filter_expression = _query(url).get("filter", [""])[0]
            if "chemical_formula_reduced" in filter_expression:
                raise OptimadeHttpError("queries are not supported", status=501, url=url)
            return _response([_entry()])

        transport = RecordingTransport(responder)

        summaries = self._database().search(
            DatabaseQuery(formula="Fe2O3", limit=1), transport=transport
        )

        self.assertEqual(len(summaries), 1)
        self.assertIn("CONTAINS", _query(transport.urls[1])["filter"][0])

    def test_substring_matches_are_checked_against_the_formula(self):
        def responder(url):
            filter_expression = _query(url).get("filter", [""])[0]
            if "chemical_formula_reduced" in filter_expression:
                raise OptimadeHttpError("not supported", status=501, url=url)
            if "CONTAINS" in filter_expression:
                return _response([_entry("demo-2", chemical_formula_reduced="Fe2O3Si")])
            return _response([_entry("demo-3")])

        transport = RecordingTransport(responder)

        summaries = self._database().search(
            DatabaseQuery(formula="Fe2O3", limit=1), transport=transport
        )

        self.assertEqual([summary.identifier for summary in summaries], ["demo-3"])
        self.assertIn("elements HAS ALL", _query(transport.urls[-1])["filter"][0])

    def test_an_empty_result_still_reports_nothing(self):
        transport = RecordingTransport(lambda url: _response([]))

        summaries = self._database().search(
            DatabaseQuery(elements=("Fe",), limit=3), transport=transport
        )

        self.assertEqual(summaries, [])

    def test_a_broken_fallback_keeps_a_valid_empty_answer(self):
        def responder(url):
            filter_expression = _query(url).get("filter", [""])[0]
            if "chemical_formula_reduced" in filter_expression:
                return _response([])
            raise OptimadeHttpError("internal error", status=500, url=url)

        transport = RecordingTransport(responder)

        summaries = self._database().search(
            DatabaseQuery(formula="Fe2O3", limit=3), transport=transport
        )

        self.assertEqual(summaries, [])
        self.assertEqual(len(transport.urls), 3)

    def test_a_broken_filter_without_an_answer_is_reported(self):
        def responder(url):
            raise OptimadeHttpError(f"{url} returned HTTP 500", status=500, url=url)

        transport = RecordingTransport(responder)

        with self.assertRaises(DatabaseRequestError) as error:
            self._database().search(DatabaseQuery(formula="Fe2O3"), transport=transport)

        self.assertIn("500", str(error.exception))

    def test_a_missing_endpoint_falls_back_to_the_root_path(self):
        def responder(url):
            if "/v1/structures" in url:
                raise OptimadeHttpError("not found", status=404, url=url)
            return _response([_entry()])

        transport = RecordingTransport(responder)

        summaries = self._database().search(
            DatabaseQuery(elements=("Fe",), limit=1), transport=transport
        )

        self.assertEqual(len(summaries), 1)
        self.assertIn("https://demo.example/optimade/structures", transport.urls[-1])

    def test_unsupported_selectors_are_reported(self):
        database = get_database("cod")

        with self.assertRaises(DatabaseRequestError) as error:
            database.search(DatabaseQuery(formula="Si", is_stable=True))

        self.assertIn("stability", str(error.exception))

    def test_fetch_reads_a_single_structure(self):
        transport = RecordingTransport(lambda url: {"data": _entry("demo-1")})

        structure = self._database().fetch("demo-1", transport=transport)

        self.assertEqual(structure.identifier, "demo-1")
        self.assertEqual(len(structure.structure), 5)
        self.assertTrue(transport.urls[0].endswith("/v1/structures/demo-1"))

    def test_fetch_reports_a_missing_entry(self):
        def responder(url):
            raise OptimadeHttpError("not found", status=404, url=url)

        with self.assertRaises(LookupError):
            self._database().fetch("nope", transport=RecordingTransport(responder))

    def test_the_federation_needs_a_provider_to_fetch(self):
        with self.assertRaises(DatabaseRequestError) as error:
            get_database("optimade").fetch("demo-1")

        self.assertIn("provider", str(error.exception))

    def test_a_provider_can_be_named_for_a_federation_search(self):
        transport = RecordingTransport(lambda url: _response([_entry()]))
        database = get_database("optimade")

        summaries = database.search(
            DatabaseQuery(elements=("Fe",), limit=1),
            provider="aflow",
            transport=transport,
        )

        self.assertEqual(summaries[0].database, "aflow")
        self.assertIn("aflow.org", transport.urls[0])

    def test_options_are_rejected_by_databases_that_do_not_take_them(self):
        with self.assertRaises(DatabaseRequestError):
            get_database("mp").check_options(provider="aflow")


class TestStructureFiles(unittest.TestCase):
    def test_filenames_follow_the_format(self):
        self.assertEqual(structure_filename("stru"), "STRU")
        self.assertEqual(structure_filename(".cif"), "structure.cif")

    def test_unknown_formats_are_rejected(self):
        with self.assertRaises(ValueError):
            structure_filename("gen")

    def test_paths_can_be_grouped_by_database(self):
        self.assertEqual(structure_path(Path("out"), "mp-149"), Path("out") / "mp-149" / "STRU")
        self.assertEqual(
            structure_path(Path("out"), "1000000", fmt="poscar", database="cod"),
            Path("out") / "cod" / "1000000" / "POSCAR",
        )

    def test_writing_creates_directories_and_a_stru(self):
        structure = DatabaseStructure(
            summary=DatabaseSummary(database="demo", identifier="demo-1"),
            structure=_structure(),
        )

        with tempfile.TemporaryDirectory() as temporary:
            destination = structure_path(Path(temporary), "demo-1")
            written = write_structure(structure, destination)
            text = written.read_text()

        self.assertIn("ATOMIC_SPECIES", text)
        self.assertIn("LATTICE_VECTORS", text)


C2DB_LABELS = (
    "Formula",
    "Energy above hull [eV/atom]",
    "Heat of formation [eV/atom]",
    "Band gap (PBE) [eV]",
    "Magnetic",
    "Layer group (not Space group)",
)


def _c2db_page(rows, labels=C2DB_LABELS, total=None, error=""):
    """Build the markup of one C2DB query page."""
    total = len(rows) if total is None else total
    header = "".join(f"<th><a>{label}</a></th>" for label in labels)
    body = ""
    for uid, values in rows:
        cells = "".join(
            f'<th scope="row"><a href=/material/{uid} target="_blank">{value}</a></th>'
            for value in values
        )
        body += f"<tr>{cells}</tr>"
    return (
        '<div id="table-div">'
        "<!--  show the error message  -->"
        f'<div class="row"><p style="color: red;">{error}</p></div>'
        "<!--  show the summary message  -->"
        f'<div class="row"><p style="color: green;">Found {total} rows out of '
        f"17001, showing rows 1-{len(rows)}</p></div>"
        f"<table><thead><tr>{header}</tr></thead>{body}</table></div>"
    )


def _c2db_row(uid="1MoS2-1", **overrides):
    """Build one body row of the C2DB table."""
    values = {
        "formula": "MoS<sub>2</sub>",
        "ehull": "0.000",
        "hform": "-0.921",
        "gap": "1.580",
        "is_magnetic": "No",
        "layergroup": "p-6m2",
    }
    extra = [key for key in overrides if key not in values]
    values.update(overrides)
    return uid, [values[key] for key in [*DEFAULT_COLUMNS, *extra]]


class TestC2DBKeys(unittest.TestCase):
    def test_the_key_catalogue_documents_the_properties(self):
        keys = c2db_keys()

        self.assertGreaterEqual(len(keys), 80)
        for key in ("gap", "gap_hse", "gap_gw", "ehull", "hform", "is_magnetic", "magstate"):
            self.assertIn(key, keys)
        self.assertEqual(keys["gap"], "Band gap (PBE) [eV]")
        self.assertIn("c2db", c2db_keys_source()["source"])

    def test_the_database_documents_its_fields(self):
        fields = dict(get_database("c2db").fields())

        self.assertEqual(fields["ehull"], "Energy above hull [eV/atom]")
        self.assertEqual(len(fields), len(c2db_keys()))


class TestC2DBFilter(unittest.TestCase):
    def test_property_terms_come_first(self):
        query = DatabaseQuery(formula="Fe2O3", where=("gap>1",))

        self.assertEqual(build_filter(query), "gap>1,Fe2O3")

    def test_elements_and_chemical_systems_become_element_lists(self):
        self.assertEqual(build_filter(DatabaseQuery(elements=("Mo", "S"))), "Mo,S")
        self.assertEqual(build_filter(DatabaseQuery(chemsys="Li-Fe-O")), "Li,Fe,O")

    def test_identifiers_use_the_c2db_uid(self):
        self.assertEqual(build_filter(DatabaseQuery(identifiers=("1MoS2-1",))), "uid=1MoS2-1")
        self.assertEqual(
            build_filter(DatabaseQuery(identifiers=("1MoS2-1", "1MoS2-2"))),
            "(uid=1MoS2-1 | uid=1MoS2-2)",
        )

    def test_an_unknown_element_is_rejected(self):
        with self.assertRaises(DatabaseRequestError):
            build_filter(DatabaseQuery(elements=("Xx",)))

    def test_an_empty_query_is_rejected(self):
        with self.assertRaises(DatabaseRequestError):
            build_filter(DatabaseQuery())


class TestC2DBTable(unittest.TestCase):
    def test_rows_and_columns_are_read(self):
        page = parse_table(_c2db_page([_c2db_row()], total=167))

        self.assertEqual(page.total, 167)
        self.assertEqual(list(page.columns), list(DEFAULT_COLUMNS))
        self.assertEqual(len(page.rows), 1)
        self.assertEqual(page.rows[0].uid, "1MoS2-1")
        self.assertEqual(page.rows[0].values["formula"], "MoS2")
        self.assertEqual(page.rows[0].values["gap"], "1.580")

    def test_a_single_row_summary_is_read(self):
        page = parse_table(_c2db_page([_c2db_row()]))

        self.assertEqual(page.total, 1)

    def test_extra_columns_are_named_after_their_keys(self):
        labels = [*C2DB_LABELS, "Band gap (HSE06) [eV]"]

        page = parse_table(_c2db_page([_c2db_row(gap_hse="2.087")], labels))

        self.assertEqual(list(page.columns)[-1], "gap_hse")
        self.assertEqual(page.rows[0].values["gap_hse"], "2.087")

    def test_empty_cells_stay_empty(self):
        page = parse_table(_c2db_page([_c2db_row(gap="")]))

        self.assertEqual(page.rows[0].values["gap"], "")

    def test_a_rejected_filter_is_reported(self):
        with self.assertRaises(DatabaseRequestError) as error:
            parse_table(_c2db_page([], total=0, error="Bad filter string"))

        self.assertIn("Bad filter string", str(error.exception))

    def test_a_row_becomes_a_summary(self):
        page = parse_table(_c2db_page([_c2db_row()]))

        summary = summary_from_row(page.rows[0])

        self.assertEqual(summary.database, "c2db")
        self.assertEqual(summary.identifier, "1MoS2-1")
        self.assertEqual(summary.formula, "MoS2")
        self.assertEqual(summary.chemsys, "Mo-S")
        self.assertAlmostEqual(summary.energy_above_hull, 0.0)
        self.assertAlmostEqual(summary.band_gap, 1.58)
        self.assertTrue(summary.theoretical)
        self.assertEqual(summary.extra["layergroup"], "p-6m2")


class TestC2DBSearch(unittest.TestCase):
    def _transport(self, pages, optimade=None):
        def responder(url):
            if url.endswith("/"):
                return '<input class="form-control" name="sid" value="4321">'
            for match, page in pages.items():
                if match in url:
                    return page
            if optimade is not None and "optimade" in url:
                return optimade
            raise AssertionError(f"unexpected URL: {url}")

        return RecordingTransport(responder)

    def test_search_queries_the_table_and_returns_the_properties(self):
        page = _c2db_page([_c2db_row()], total=1)
        transport = self._transport({"filter=MoS2": page})

        summaries = get_database("c2db").search(
            DatabaseQuery(formula="MoS2", limit=5), transport=transport
        )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].identifier, "1MoS2-1")
        self.assertAlmostEqual(summaries[0].band_gap, 1.58)
        self.assertIn("filter=MoS2", transport.urls[-1])
        self.assertEqual(_query(transport.urls[-1])["sid"], ["4321"])

    def test_property_terms_reach_the_filter(self):
        page = _c2db_page([_c2db_row()], total=1)
        transport = self._transport({"filter=gap%3E1.5": page})

        get_database("c2db").search(DatabaseQuery(where=("gap>1.5",), limit=1), transport=transport)

        self.assertIn("filter=gap%3E1.5", transport.urls[-1])

    def test_extra_columns_are_toggled(self):
        page = _c2db_page([_c2db_row()], total=1)
        labels = [*C2DB_LABELS, "Band gap (HSE06) [eV]"]
        toggled = _c2db_page([_c2db_row(gap_hse="2.087")], labels, total=1)
        transport = self._transport({"toggle=gap_hse": toggled, "filter=MoS2": page})

        summaries = get_database("c2db").search(
            DatabaseQuery(formula="MoS2", limit=1),
            show=("gap_hse",),
            transport=transport,
        )

        self.assertEqual(summaries[0].extra["gap_hse"], "2.087")
        self.assertTrue(any("toggle=gap_hse" in url for url in transport.urls))

    def test_search_pages_through_the_results(self):
        first = _c2db_page([_c2db_row("1MoS2-1")], total=2)
        second = _c2db_page([_c2db_row("1MoS2-2")], total=2)
        transport = self._transport({"filter=MoS2": first, "page=1": second})

        summaries = get_database("c2db").search(
            DatabaseQuery(formula="MoS2", limit=2), transport=transport
        )

        self.assertEqual([summary.identifier for summary in summaries], ["1MoS2-1", "1MoS2-2"])
        self.assertIn("page=1", transport.urls[-1])

    def test_an_unknown_column_is_rejected(self):
        with self.assertRaises(DatabaseRequestError) as error:
            get_database("c2db").search(
                DatabaseQuery(formula="MoS2"),
                show=("nope",),
                transport=RecordingTransport(lambda url: ""),
            )

        self.assertIn("nope", str(error.exception))

    def test_a_changed_column_layout_is_reported(self):
        page = _c2db_page([_c2db_row()], labels=C2DB_LABELS[:-1], total=1)
        transport = self._transport({"filter=MoS2": page})

        with self.assertRaises(DatabaseRequestError) as error:
            get_database("c2db").search(DatabaseQuery(formula="MoS2"), transport=transport)

        self.assertIn("columns", str(error.exception))

    def test_a_selector_the_database_does_not_support_is_reported(self):
        with self.assertRaises(DatabaseRequestError) as error:
            get_database("c2db").search(DatabaseQuery(formula="MoS2", is_stable=True))

        self.assertIn("stability", str(error.exception))


class TestC2DBFetch(unittest.TestCase):
    def _transport(self, pages, optimade):
        def responder(url):
            if url.endswith("/"):
                return '<input name="sid" value="4321">'
            if "optimade" in url:
                return json.dumps(optimade)
            for match, page in pages.items():
                if match in url:
                    return page
            raise AssertionError(f"unexpected URL: {url}")

        return RecordingTransport(responder)

    def test_fetch_joins_the_row_and_the_optimade_structure(self):
        page = _c2db_page([_c2db_row()], total=1)
        transport = self._transport(
            {"filter=uid%3D1MoS2-1": page},
            {"data": [_entry("8192")]},
        )

        structure = get_database("c2db").fetch("1MoS2-1", transport=transport)

        self.assertEqual(structure.identifier, "1MoS2-1")
        self.assertEqual(structure.summary.formula, "MoS2")
        self.assertEqual(structure.summary.nsites, 5)
        self.assertEqual(structure.summary.extra["optimade_id"], "8192")
        self.assertEqual(structure.summary.extra["layergroup"], "p-6m2")
        self.assertEqual(len(structure.structure), 5)
        self.assertIn("optimade", transport.urls[-1])

    def test_fetch_reports_a_missing_entry(self):
        page = _c2db_page([], total=0)
        transport = self._transport({"filter=uid": page}, {"data": []})

        with self.assertRaises(LookupError):
            get_database("c2db").fetch("nope", transport=transport)

    def test_fetch_reports_an_entry_without_a_structure(self):
        page = _c2db_page([_c2db_row()], total=1)
        transport = self._transport({"filter=uid": page}, {"data": []})

        with self.assertRaises(LookupError):
            get_database("c2db").fetch("1MoS2-1", transport=transport)


if __name__ == "__main__":
    unittest.main()
