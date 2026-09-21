"""The Computational 2D Materials Database (C2DB) of DTU.

C2DB publishes its structures through OPTIMADE and its computed properties
through the query table of its web application: band gaps from PBE, HSE06 and
G0W0, energies above the convex hull, heat of formation, elastic and
piezoelectric constants, magnetic states, optical properties, and more than
eighty other keys. This backend reads the property table and takes the
geometry from the OPTIMADE endpoint, so a search reports the computed data and
a download writes the structure.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from importlib import resources
from typing import Any, Callable, Mapping, Optional, Sequence

from .base import (
    DatabaseQuery,
    DatabaseRequestError,
    DatabaseStructure,
    DatabaseSummary,
    StructureDatabase,
)
from .optimade import reduced_formula, structure_from_entry
from .registry import register_database

#: Web application that serves the query table.
WEB_BASE_URL = "https://c2db.fysik.dtu.dk"

#: Bundled catalogue of the property keys, taken from the ``help#keys`` page.
KEYS_RESOURCE = "c2db_keys.json"

#: Columns of the query table before any extra column is added.
DEFAULT_COLUMNS = ("formula", "ehull", "hform", "gap", "is_magnetic", "layergroup")

#: Rows the query table returns per page.
ROWS_PER_PAGE = 25

#: Upper bound on the pages read for one search.
MAX_PAGES = 20

#: Seconds before a request to the web application is abandoned.
REQUEST_TIMEOUT = 60.0

#: User agent sent with every request.
USER_AGENT = "abacustools"

#: A transport maps a URL to the body of the response.
Transport = Callable[[str], str]

#: Pattern of the session id in the query page.
_SESSION_PATTERN = re.compile(r'name="sid"[^>]*value="(\d+)"')

#: Pattern of the "Found N rows out of M" summary line.
_SUMMARY_PATTERN = re.compile(r"Found (\d+) rows? out of (\d+)")

#: Pattern of the link that identifies the material of a row.
_MATERIAL_PATTERN = re.compile(r"/material/([^/?#\s\"']+)")


@dataclass(frozen=True)
class C2DBRow:
    """One row of the C2DB query table."""

    uid: str
    values: Mapping[str, str]


@dataclass(frozen=True)
class TablePage:
    """One page of the C2DB query table."""

    total: Optional[int]
    columns: tuple[str, ...]
    rows: tuple[C2DBRow, ...]


def load_keys() -> dict[str, str]:
    """Return the documented C2DB property keys and their descriptions."""
    payload = json.loads(
        resources.files(__package__).joinpath(KEYS_RESOURCE).read_text(encoding="utf-8")
    )
    return {str(key): str(description) for key, description in payload["keys"].items()}


def keys_source() -> Mapping[str, str]:
    """Return where and when the bundled key catalogue was collected."""
    payload = json.loads(
        resources.files(__package__).joinpath(KEYS_RESOURCE).read_text(encoding="utf-8")
    )
    return {
        "source": str(payload.get("source") or ""),
        "retrieved": str(payload.get("retrieved") or ""),
    }


def column_labels() -> dict[str, str]:
    """Return the table column label of every property key."""
    return {key: description for key, description in load_keys().items()}


def label_to_key() -> dict[str, str]:
    """Return the property key of every table column label."""
    return {description: key for key, description in load_keys().items()}


def _clean_text(value: str) -> str:
    """Collapse the whitespace of a cell or message."""
    return re.sub(r"\s+", " ", value).strip()


def _message(text: str, kind: str) -> str:
    """Return the error or summary message of a query page."""
    match = re.search(
        rf"show the {kind} message.*?<p[^>]*>(.*?)</p>",
        text,
        re.DOTALL,
    )
    if not match:
        return ""
    return _clean_text(re.sub(r"<[^>]+>", " ", match.group(1)))


class _ResultsTableParser(HTMLParser):
    """Read the header labels and the cells of the C2DB results table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headers: list[str] = []
        self.rows: list[list[str]] = []
        self.links: list[str] = []
        self._in_table = False
        self._in_header = False
        self._cell: Optional[list[str]] = None
        self._cell_link: Optional[str] = None
        self._row: list[str] = []
        self._row_link: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        if tag == "table":
            self._in_table = True
            return
        if not self._in_table:
            return
        if tag == "thead":
            self._in_header = True
        elif tag in ("th", "td"):
            self._cell = []
            self._cell_link = None
        elif tag == "a" and self._cell is not None and self._cell_link is None:
            href = attributes.get("href")
            if href:
                self._cell_link = href.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._in_table = False
            self._in_header = False
            self._cell = None
            return
        if not self._in_table:
            return
        if tag == "thead":
            self._in_header = False
        elif tag in ("th", "td") and self._cell is not None:
            text = _clean_text("".join(self._cell))
            if self._in_header:
                self.headers.append(text)
            else:
                self._row.append(text)
                if self._row_link is None and self._cell_link:
                    self._row_link = self._cell_link
            self._cell = None
            self._cell_link = None
        elif tag == "tr" and not self._in_header:
            if self._row:
                self.rows.append(self._row)
                self.links.append(self._row_link or "")
            self._row = []
            self._row_link = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_table(text: str) -> TablePage:
    """Parse one page of the C2DB query table.

    Args:
        text: The HTML fragment returned by ``/table``.

    Returns:
        The reported number of matches, the columns and the rows.

    Raises:
        DatabaseRequestError: If the page reports a rejected query.
    """
    error = _message(text, "error")
    if error:
        raise DatabaseRequestError(f"C2DB rejected the query: {error}")
    match = _SUMMARY_PATTERN.search(text)
    total = int(match.group(1)) if match else None
    parser = _ResultsTableParser()
    parser.feed(text)
    labels = label_to_key()
    columns = tuple(labels.get(label, label) for label in parser.headers)
    rows = []
    for values, link in zip(parser.rows, parser.links):
        material = _MATERIAL_PATTERN.search(link)
        identifier = material.group(1) if material else ""
        rows.append(C2DBRow(uid=identifier, values=dict(zip(columns, values))))
    return TablePage(total=total, columns=columns, rows=tuple(rows))


def _elements(symbols: Sequence[str]) -> list[str]:
    """Return validated element symbols.

    Raises:
        DatabaseRequestError: If a symbol is not an element.
    """
    from pymatgen.core import Element

    validated = []
    for symbol in symbols:
        try:
            validated.append(str(Element(str(symbol))))
        except (ValueError, KeyError) as error:
            raise DatabaseRequestError(
                f"{symbol!r} is not an element symbol: {error}"
            ) from error
    return validated


def build_filter(query: DatabaseQuery) -> str:
    """Translate a query into a C2DB filter string.

    The C2DB table combines its terms with ``,``, a bare formula such as
    ``MoS2`` selects formula units, and a bare element symbol selects the
    materials that contain it.

    Args:
        query: The selectors to translate.

    Returns:
        The filter expression of the query table.

    Raises:
        DatabaseRequestError: If the query carries no selector.
    """
    terms = [str(term) for term in query.where or ()]
    if query.formula:
        terms.append(reduced_formula(query.formula))
    symbols = [str(symbol) for symbol in query.elements or ()]
    if not symbols and query.chemsys:
        symbols = [part for part in re.split(r"[-\s,]+", str(query.chemsys)) if part]
    if symbols:
        terms.append(",".join(_elements(symbols)))
    if query.identifiers:
        ids = [f"uid={identifier}" for identifier in query.identifiers]
        terms.append(ids[0] if len(ids) == 1 else "(" + " | ".join(ids) + ")")
    if not terms:
        raise DatabaseRequestError("a search needs at least one selector")
    return ",".join(terms)


def _number(value: Any) -> Optional[float]:
    """Return the number printed in a table cell, or None."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _chemsys(formula: Optional[str]) -> Optional[str]:
    """Return the chemical system of a formula, or None."""
    if not formula:
        return None
    from pymatgen.core import Composition

    try:
        return "-".join(sorted(str(element) for element in Composition(formula).elements))
    except (ValueError, TypeError):
        return None


def summary_from_row(row: C2DBRow, database: str = "c2db") -> DatabaseSummary:
    """Convert one table row into a database summary.

    The table prints its values with three decimals, so the properties stay
    strings in :attr:`DatabaseSummary.extra`; the energy above the hull and
    the PBE band gap are also parsed into the common fields.
    """
    values = {key: value for key, value in row.values.items() if value not in (None, "")}
    formula = values.get("formula") or None
    return DatabaseSummary(
        database=database,
        identifier=row.uid,
        formula=formula,
        chemsys=_chemsys(formula),
        energy_above_hull=_number(values.get("ehull")),
        band_gap=_number(values.get("gap")),
        theoretical=True,
        extra={"uid": row.uid, **values},
    )


def _web_base(base_url: Optional[str]) -> str:
    """Return the C2DB web application to query."""
    return (base_url or WEB_BASE_URL).rstrip("/")


def _requested_columns(show: Optional[Sequence[str]]) -> list[str]:
    """Return the extra columns asked for on the command line."""
    keys = []
    for value in show or ():
        for item in str(value).split(","):
            key = item.strip()
            if key:
                keys.append(key)
    return keys


class C2DBDatabase(StructureDatabase):
    """Search and download C2DB entries, including their computed data."""

    name = "c2db"
    description = "Computational 2D Materials Database (C2DB, DTU)"
    protocol = "C2DB table + OPTIMADE"
    capabilities = frozenset({"formula", "chemsys", "elements", "identifiers", "where"})
    options = frozenset({"base_url", "show"})
    homepage = "https://cmr.fysik.dtu.dk/c2db/c2db.html"

    def fields(self) -> list[tuple[str, str]]:
        """Return the C2DB property keys and their descriptions."""
        return sorted(load_keys().items())

    def _get(
        self,
        url: str,
        *,
        transport: Optional[Transport] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> str:
        """Return the body of one request."""
        if transport is not None:
            return str(transport(url))
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html, application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            raise DatabaseRequestError(
                f"{url} returned HTTP {error.code}"
            ) from error
        except (urllib.error.URLError, OSError) as error:
            raise DatabaseRequestError(f"cannot reach {url}: {error}") from error

    def _session(self, base: str, transport: Optional[Transport]) -> str:
        """Return a session id of the query page."""
        page = self._get(f"{base}/", transport=transport)
        match = _SESSION_PATTERN.search(page)
        if not match:
            raise DatabaseRequestError(
                f"{base} did not hand out a session id; the query page changed"
            )
        return match.group(1)

    def _page(
        self,
        session: str,
        base: str,
        params: Mapping[str, Any],
        transport: Optional[Transport],
    ) -> TablePage:
        """Return one page of the query table."""
        query = urllib.parse.urlencode({"sid": session, **params})
        return parse_table(self._get(f"{base}/table?{query}", transport=transport))

    def _check_columns(self, page: TablePage, expected: Sequence[str]) -> None:
        """Validate the columns of a page against the ones that were asked for."""
        if list(page.columns) != list(expected):
            raise DatabaseRequestError(
                "the C2DB query table changed its columns: expected "
                f"{list(expected)}, got {list(page.columns)}"
            )

    def _rows(
        self,
        session: str,
        base: str,
        expression: str,
        columns: Sequence[str],
        limit: int,
        transport: Optional[Transport],
    ) -> list[C2DBRow]:
        """Return the rows matching a filter, with the requested columns."""
        page = self._page(session, base, {"filter": expression}, transport)
        for key in list(columns)[len(DEFAULT_COLUMNS):]:
            page = self._page(session, base, {"toggle": key}, transport)
        self._check_columns(page, columns)
        rows = list(page.rows)
        total = page.total if page.total is not None else len(rows)
        index = 0
        while len(rows) < limit and len(rows) < total and index + 1 < MAX_PAGES:
            index += 1
            page = self._page(session, base, {"page": index}, transport)
            if not page.rows:
                break
            rows.extend(page.rows)
        return rows[:limit]

    def _columns(self, show: Optional[Sequence[str]]) -> list[str]:
        """Return the columns of a search, validating the requested keys."""
        keys = load_keys()
        columns = list(DEFAULT_COLUMNS)
        for key in _requested_columns(show):
            if key not in keys:
                raise DatabaseRequestError(
                    f"unknown C2DB key {key!r}; run 'abacustools database fields "
                    "-d c2db' to list the keys"
                )
            if key not in columns:
                columns.append(key)
        return columns

    def _optimade_entry(self, base: str, uid: str, transport: Optional[Transport]) -> Mapping[str, Any]:
        """Return the OPTIMADE entry of one C2DB uid."""
        query = urllib.parse.urlencode({"filter": f'uid="{uid}"', "page_limit": 1})
        body = self._get(f"{base}/optimade/v1/structures?{query}", transport=transport)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            raise DatabaseRequestError(
                f"the OPTIMADE endpoint of {base} did not return JSON: {error}"
            ) from error
        entries = payload.get("data") or []
        if not isinstance(entries, list) or not entries:
            raise LookupError(f"C2DB has no structure for {uid!r}")
        return entries[0]

    def search(
        self,
        query: DatabaseQuery,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        base_url: Optional[str] = None,
        show: Optional[Sequence[str]] = None,
        transport: Optional[Transport] = None,
        **options: Any,
    ) -> list[DatabaseSummary]:
        """Search C2DB by composition and by any computed property."""
        self.check_query(query)
        query.require_selector()
        columns = self._columns(show)
        base = _web_base(base_url)
        session = self._session(base, transport)
        rows = self._rows(session, base, build_filter(query), columns, query.limit, transport)
        return [summary_from_row(row, self.name) for row in rows]

    def fetch(
        self,
        identifier: str,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        base_url: Optional[str] = None,
        transport: Optional[Transport] = None,
        **options: Any,
    ) -> DatabaseStructure:
        """Download one C2DB structure together with its computed data."""
        base = _web_base(base_url)
        session = self._session(base, transport)
        rows = self._rows(
            session,
            base,
            f"uid={identifier}",
            list(DEFAULT_COLUMNS),
            1,
            transport,
        )
        if not rows:
            raise LookupError(f"C2DB has no entry {identifier!r}")
        row = rows[0]
        entry = self._optimade_entry(base, row.uid, transport)
        summary = summary_from_row(row, self.name)
        attributes = entry.get("attributes") or {}
        nsites = attributes.get("nsites")
        summary = replace(
            summary,
            nsites=int(nsites) if nsites else summary.nsites,
            extra={**summary.extra, "optimade_id": str(entry.get("id"))},
        )
        return DatabaseStructure(summary=summary, structure=structure_from_entry(entry))


def register() -> C2DBDatabase:
    """Register the C2DB database and return its adapter."""
    return register_database(C2DBDatabase(), replace=True)
