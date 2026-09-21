"""Tests for the structure-database layer: registry, OPTIMADE and C2DB."""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.parse
from pathlib import Path
from unittest.mock import patch

import pytest

from abacustools.integrations.databases import (
    DatabaseQuery,
    DatabaseRequestError,
    DatabaseStructure,
    DatabaseSummary,
    OptimadeHttpError,
    OptimadeProvider,
    c2db_keys,
    c2db_keys_source,
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
    build_filter as c2db_filter,
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
from abacustools.integrations.materials_project import (
    MaterialStructure,
    MaterialSummary,
)

OPTIMADE_URL = "https://demo.example/optimade"


def _structure():
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _entry(identifier: str = "demo-1", **attributes):
    """One OPTIMADE structure entry."""
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


def _failing(status: int, message: str = "failed"):
    """A transport that answers every request with an HTTP error."""

    def responder(url):
        raise OptimadeHttpError(f"{url} returned HTTP {status}", status=status, url=url)

    return responder


class RecordingTransport:
    """A stand-in HTTP transport that records the URLs it was asked for."""

    def __init__(self, responder):
        self.responder = responder
        self.urls: list[str] = []

    def __call__(self, url):
        self.urls.append(url)
        return self.responder(url)


def _optimade():
    return OptimadeDatabase(OptimadeProvider(name="demo", base_url=OPTIMADE_URL))


def test_registry_lists_every_database_once_with_aliases():
    records = describe_databases()
    names = [record["name"] for record in records]
    assert len(names) == len(set(names))
    assert {"mp", "optimade", "c2db", "c2db-optimade", "aflow", "cod"} <= set(names)
    assert all(record["status"] in {"ready", "needs-api-key", "unavailable"} for record in records)
    assert get_database("materials-project") is get_database("mp")
    assert get_database("optimade-federation") is get_database("optimade")


def test_registering_and_replacing_a_database():
    class Demo:
        name = "demo"

    demo = Demo()
    try:
        register_database(demo, aliases=("demo-alias",))
        assert get_database("demo-alias") is demo
        with pytest.raises(ValueError):
            register_database(Demo())
    finally:
        unregister_database("demo")
    with pytest.raises(DatabaseRequestError):
        get_database("demo-alias")


def test_unknown_databases_and_providers_list_the_known_ones():
    with pytest.raises(DatabaseRequestError, match="mp"):
        get_database("nope")
    with pytest.raises(DatabaseRequestError, match="aflow"):
        optimade_provider_from_name("nope")


def test_query_reports_and_validates_its_selectors():
    cases = [
        (DatabaseQuery(formula="Si"), ["formula"]),
        (DatabaseQuery(chemsys="Li-O", is_stable=True, limit=5), ["chemsys", "stability"]),
        (DatabaseQuery(elements=("Li", "O"), theoretical=True), ["elements", "theoretical"]),
        (DatabaseQuery(where=("gap>1",)), ["where"]),
    ]
    for query, selectors in cases:
        assert query.selectors() == selectors

    with pytest.raises(DatabaseRequestError, match="selector"):
        DatabaseQuery().require_selector()
    unsupported = DatabaseQuery(formula="Si", theoretical=True).unsupported(frozenset({"formula"}))
    assert unsupported == ["theoretical"]


def test_summaries_and_structures_serialise_to_the_common_shape():
    summary = DatabaseSummary(database="demo", identifier="demo-1", formula="Si2")
    record = summary.to_dict()
    assert (record["database"], record["id"], record["extra"]) == ("demo", "demo-1", {})

    abacus = DatabaseStructure(summary=summary, structure=_structure()).to_abacus_structure()
    assert abacus.metadata == {
        "lattice_constant": 1.0,
        "atom_type": "cartesian",
        "source": "demo",
        "identifier": "demo-1",
    }


def test_materials_project_maps_searches_and_structures():
    document = MaterialSummary(
        material_id="mp-149", formula="Si2", chemsys="Si", nsites=2, band_gap=0.61
    )
    database = get_database("mp")

    with patch(
        "abacustools.integrations.databases.materials_project.search_materials",
        return_value=[document],
    ) as search:
        summaries = database.search(
            DatabaseQuery(formula="Si", chemsys="Si-O", is_stable=True, limit=5)
        )
    assert [summary.identifier for summary in summaries] == ["mp-149"]
    assert summaries[0].database == "mp"
    assert summaries[0].extra["material_id"] == "mp-149"
    assert search.call_args.kwargs["is_stable"] is True
    assert search.call_args.kwargs["limit"] == 5

    material = MaterialStructure(summary=document, structure=_structure())
    with patch(
        "abacustools.integrations.databases.materials_project.download_material",
        return_value=material,
    ):
        structure = database.fetch("mp-149")
    assert structure.identifier == "mp-149"
    assert len(structure.structure) == 2

    with pytest.raises(DatabaseRequestError):
        database.search(DatabaseQuery())


def test_materials_project_without_its_client_is_unavailable():
    database = get_database("mp")
    with patch.dict(sys.modules, {"mp_api": None, "mp_api.client": None}):
        assert database.available() is False
        assert database.status() == "unavailable"
        assert "mp-api" in database.unavailable_reason()


def test_the_optimade_catalogue_documents_its_providers():
    providers = optimade_catalogue()
    assert len(providers) >= 10
    assert {"aflow", "c2db-optimade", "cod", "twodmatpedia"} <= {p.name for p in providers}
    assert all(provider.base_url.startswith("http") for provider in providers)
    assert "optimade.org" in optimade_catalogue_source()["source"]
    assert optimade_catalogue_source()["retrieved"]
    assert optimade_provider_from_name("materials-project-optimade").name == "mp-optimade"

    c2db = get_database("c2db-optimade")
    assert c2db.supports("where") and not c2db.supports("identifiers")


def test_optimade_filters_translate_the_selectors():
    cases = [
        (
            DatabaseQuery(identifiers=("a", "b"), elements=("Fe", "O")),
            ['(id="a" OR id="b")', 'elements HAS ALL "Fe", "O"'],
        ),
        (DatabaseQuery(elements=("Li",)), ['elements HAS "Li"']),
        (DatabaseQuery(chemsys="Li-O"), ['elements HAS ALL "Li", "O"']),
        (
            DatabaseQuery(formula="Fe2O3", where=("gap>1",)),
            ['chemical_formula_reduced="Fe2O3"', "gap>1"],
        ),
    ]
    for query, terms in cases:
        expression = build_filters(query)[0]
        for term in terms:
            assert term in expression
    assert build_filters(DatabaseQuery()) == [None]


def test_optimade_formula_strategies_and_validation():
    strategies = build_strategies(DatabaseQuery(formula="Fe2O3"))
    assert 'chemical_formula_reduced="Fe2O3"' in strategies[0].expression
    assert "CONTAINS" in strategies[1].expression and strategies[1].check_formula
    assert strategies[2].expression == 'elements HAS ALL "Fe", "O"'
    assert strategies[2].check_formula

    assert formula_matches("Fe4O6", "Fe2O3")
    assert not formula_matches("Fe2O3Si", "Fe2O3")
    assert not formula_matches(None, "Fe2O3")
    for formula in ("Li*O", "not a formula"):
        with pytest.raises(DatabaseRequestError):
            reduced_formula(formula)


def test_optimade_entries_become_summaries_and_structures():
    entry = _entry(
        nsites=None,
        _mp_stability={"gga_gga+u": {"energy_above_hull": 0.25}},
        _cod_flags="has coordinates",
    )
    summary = summary_from_entry(entry, "mp-optimade")
    assert (summary.identifier, summary.formula, summary.chemsys) == ("demo-1", "Fe2O3", "Fe-O")
    assert summary.nsites == 5
    assert summary.energy_above_hull == pytest.approx(0.25)
    assert summary.extra["_cod_flags"] == "has coordinates"

    structure = structure_from_entry(entry)
    assert len(structure) == 5
    assert structure.composition.reduced_formula == "Fe2O3"

    with pytest.raises(DatabaseRequestError, match="coordinates"):
        structure_from_entry(_entry(cartesian_site_positions=None, species_at_sites=None))


def test_optimade_search_queries_and_pages_the_endpoint():
    first = _response(
        [_entry("demo-1")], next_link={"href": f"{OPTIMADE_URL}/v1/structures?page_offset=1"}
    )
    second = _response([_entry("demo-2")])
    transport = RecordingTransport(lambda url: first if "page_offset" not in url else second)

    summaries = _optimade().search(DatabaseQuery(formula="Fe2O3", limit=2), transport=transport)

    assert [summary.identifier for summary in summaries] == ["demo-1", "demo-2"]
    assert summaries[0].extra["provider"] == "demo"
    filter_expression = _query(transport.urls[0])["filter"][0]
    assert "/v1/structures" in transport.urls[0]
    assert "chemical_formula_reduced" in filter_expression
    assert int(_query(transport.urls[0])["page_limit"][0]) <= 100


def test_optimade_search_falls_back_when_a_formula_filter_is_rejected():
    def responder(url):
        filter_expression = _query(url).get("filter", [""])[0]
        if "chemical_formula_reduced" in filter_expression:
            raise OptimadeHttpError("not supported", status=501, url=url)
        if "CONTAINS" in filter_expression:
            # A substring match, which the client has to check itself.
            return _response([_entry("demo-2", chemical_formula_reduced="Fe2O3Si")])
        return _response([_entry("demo-3")])

    transport = RecordingTransport(responder)
    summaries = _optimade().search(DatabaseQuery(formula="Fe2O3", limit=1), transport=transport)

    assert [summary.identifier for summary in summaries] == ["demo-3"]
    assert "elements HAS ALL" in _query(transport.urls[-1])["filter"][0]


def test_optimade_search_tolerates_empty_and_failing_fallbacks():
    empty = RecordingTransport(lambda url: _response([]))
    assert _optimade().search(DatabaseQuery(elements=("Fe",), limit=3), transport=empty) == []

    def empty_then_broken(url):
        if "chemical_formula_reduced" in _query(url).get("filter", [""])[0]:
            return _response([])
        return _failing(500)(url)

    summaries = _optimade().search(
        DatabaseQuery(formula="Fe2O3", limit=3),
        transport=RecordingTransport(empty_then_broken),
    )
    assert summaries == []

    with pytest.raises(DatabaseRequestError, match="500"):
        _optimade().search(
            DatabaseQuery(formula="Fe2O3"), transport=RecordingTransport(_failing(500))
        )


def test_optimade_search_falls_back_to_the_root_path():
    def responder(url):
        if "/v1/structures" in url:
            return _failing(404)(url)
        return _response([_entry()])

    transport = RecordingTransport(responder)
    summaries = _optimade().search(DatabaseQuery(elements=("Fe",), limit=1), transport=transport)

    assert len(summaries) == 1
    assert transport.urls[-1].split("?")[0].endswith("/optimade/structures")


def test_optimade_fetch_and_its_errors():
    transport = RecordingTransport(lambda url: {"data": _entry("demo-1")})
    structure = _optimade().fetch("demo-1", transport=transport)

    assert structure.identifier == "demo-1"
    assert len(structure.structure) == 5
    assert transport.urls[0].endswith("/v1/structures/demo-1")

    with pytest.raises(LookupError):
        _optimade().fetch("nope", transport=RecordingTransport(_failing(404)))

    with pytest.raises(DatabaseRequestError, match="provider"):
        get_database("optimade").fetch("demo-1")

    federation_transport = RecordingTransport(lambda url: _response([_entry()]))
    federation = get_database("optimade").search(
        DatabaseQuery(elements=("Fe",), limit=1),
        provider="aflow",
        transport=federation_transport,
    )
    assert federation[0].database == "aflow"
    assert "aflow.org" in federation_transport.urls[0]


def test_unsupported_selectors_and_options_are_rejected():
    with pytest.raises(DatabaseRequestError, match="stability"):
        get_database("cod").search(DatabaseQuery(formula="Si", is_stable=True))
    with pytest.raises(DatabaseRequestError, match="provider"):
        get_database("mp").check_options(provider="aflow")


def test_structure_files_are_named_and_written():
    assert structure_filename("stru") == "STRU"
    assert structure_filename(".cif") == "structure.cif"
    assert structure_path(Path("out"), "mp-149") == Path("out/mp-149/STRU")
    assert structure_path(Path("out"), "1MoS2-1", fmt="poscar", database="c2db") == Path(
        "out/c2db/1MoS2-1/POSCAR"
    )
    with pytest.raises(ValueError):
        structure_filename("gen")

    structure = DatabaseStructure(
        summary=DatabaseSummary(database="demo", identifier="demo-1"), structure=_structure()
    )
    with tempfile.TemporaryDirectory() as temporary:
        written = write_structure(structure, structure_path(Path(temporary), "demo-1"))
        text = written.read_text()
    assert "ATOMIC_SPECIES" in text and "LATTICE_VECTORS" in text


# ---------------------------------------------------------------- C2DB

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


def _c2db_transport(pages):
    """A transport that answers the session, table and OPTIMADE requests."""

    def responder(url):
        if url.endswith("/"):
            return '<input class="form-control" name="sid" value="4321">'
        for match, page in pages.items():
            if match in url:
                return page
        raise AssertionError(f"unexpected URL: {url}")

    return RecordingTransport(responder)


def test_c2db_documents_its_property_keys():
    keys = c2db_keys()
    assert len(keys) >= 80
    assert keys["gap"] == "Band gap (PBE) [eV]"
    assert "c2db" in c2db_keys_source()["source"]
    assert c2db_keys_source()["retrieved"]

    fields = dict(get_database("c2db").fields())
    assert fields["ehull"] == "Energy above hull [eV/atom]"
    assert len(fields) == len(keys)


def test_c2db_filters_translate_the_selectors():
    cases = [
        (DatabaseQuery(formula="Fe2O3", where=("gap>1",)), "gap>1,Fe2O3"),
        (DatabaseQuery(elements=("Mo", "S")), "Mo,S"),
        (DatabaseQuery(chemsys="Li-Fe-O"), "Li,Fe,O"),
        (DatabaseQuery(identifiers=("1MoS2-1",)), "uid=1MoS2-1"),
        (DatabaseQuery(identifiers=("a", "b")), "(uid=a | uid=b)"),
    ]
    for query, expected in cases:
        assert c2db_filter(query) == expected

    for query in (DatabaseQuery(elements=("Xx",)), DatabaseQuery()):
        with pytest.raises(DatabaseRequestError):
            c2db_filter(query)


def test_c2db_table_parsing():
    page = parse_table(_c2db_page([_c2db_row()], total=167))
    assert page.total == 167
    assert list(page.columns) == list(DEFAULT_COLUMNS)
    assert page.rows[0].uid == "1MoS2-1"
    assert page.rows[0].values["formula"] == "MoS2"
    assert page.rows[0].values["gap"] == "1.580"

    extra = parse_table(
        _c2db_page([_c2db_row(gap_hse="2.087")], [*C2DB_LABELS, "Band gap (HSE06) [eV]"])
    )
    assert list(extra.columns)[-1] == "gap_hse"
    assert extra.rows[0].values["gap_hse"] == "2.087"

    assert parse_table(_c2db_page([_c2db_row(gap="")])).rows[0].values["gap"] == ""
    assert parse_table(_c2db_page([_c2db_row()])).total == 1
    with pytest.raises(DatabaseRequestError, match="Bad filter"):
        parse_table(_c2db_page([], total=0, error="Bad filter string"))

    summary = summary_from_row(page.rows[0])
    assert (summary.database, summary.identifier) == ("c2db", "1MoS2-1")
    assert (summary.formula, summary.chemsys) == ("MoS2", "Mo-S")
    assert summary.band_gap == pytest.approx(1.58)
    assert summary.energy_above_hull == pytest.approx(0.0)
    assert summary.theoretical is True
    assert summary.extra["layergroup"] == "p-6m2"


def test_c2db_search_reads_the_query_table():
    first = _c2db_page([_c2db_row("1MoS2-1")], total=2)
    second = _c2db_page([_c2db_row("1MoS2-2")], total=2)
    toggled = _c2db_page(
        [_c2db_row(gap_hse="2.087")], [*C2DB_LABELS, "Band gap (HSE06) [eV]"], total=2
    )
    transport = _c2db_transport({"toggle=gap_hse": toggled, "page=1": second, "filter=": first})

    summaries = get_database("c2db").search(
        DatabaseQuery(formula="MoS2", where=("gap>1.5",), limit=2),
        show=("gap_hse",),
        transport=transport,
    )

    assert [summary.identifier for summary in summaries] == ["1MoS2-1", "1MoS2-2"]
    assert summaries[0].extra["gap_hse"] == "2.087"
    urls = " ".join(transport.urls)
    assert "filter=gap%3E1.5%2CMoS2" in urls
    assert "toggle=gap_hse" in urls
    assert "page=1" in urls
    assert "sid=4321" in urls


def test_c2db_search_rejects_bad_input():
    database = get_database("c2db")
    with pytest.raises(DatabaseRequestError, match="nope"):
        database.search(
            DatabaseQuery(formula="MoS2"),
            show=("nope",),
            transport=RecordingTransport(lambda url: ""),
        )
    with pytest.raises(DatabaseRequestError, match="stability"):
        database.search(DatabaseQuery(formula="MoS2", is_stable=True))

    changed = _c2db_page([_c2db_row()], labels=C2DB_LABELS[:-1], total=1)
    transport = _c2db_transport({"filter=": changed})
    with pytest.raises(DatabaseRequestError, match="columns"):
        database.search(DatabaseQuery(formula="MoS2"), transport=transport)


def test_c2db_fetch_joins_the_properties_and_the_structure():
    page = _c2db_page([_c2db_row()], total=1)
    transport = _c2db_transport(
        {"optimade": json.dumps({"data": [_entry("8192")]}), "filter=": page}
    )

    structure = get_database("c2db").fetch("1MoS2-1", transport=transport)

    assert structure.identifier == "1MoS2-1"
    assert structure.summary.formula == "MoS2"
    assert structure.summary.nsites == 5
    assert structure.summary.extra["optimade_id"] == "8192"
    assert structure.summary.extra["layergroup"] == "p-6m2"
    assert len(structure.structure) == 5

    for pages in (
        {"optimade": json.dumps({"data": []}), "filter=": _c2db_page([], total=0)},
        {"optimade": json.dumps({"data": []}), "filter=": page},
    ):
        with pytest.raises(LookupError):
            get_database("c2db").fetch("1MoS2-1", transport=_c2db_transport(pages))
