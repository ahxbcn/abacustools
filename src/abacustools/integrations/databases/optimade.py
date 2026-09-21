"""OPTIMADE databases: one protocol, many providers.

OPTIMADE (https://www.optimade.org) is a REST API for crystal structure
databases. Implementing it once gives abacustools access to every provider
that speaks it, which is why most entries of the bundled catalogue are thin
wrappers around :class:`OptimadeDatabase` rather than hand-written clients.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from importlib import resources
from typing import Any, Callable, Mapping, Optional, Sequence

from .base import (
    DatabaseApiKeyError,
    DatabaseQuery,
    DatabaseRequestError,
    DatabaseStructure,
    DatabaseSummary,
    StructureDatabase,
    api_key_from_environment,
)
from .registry import register_database

#: Official index of OPTIMADE providers, used by ``database providers``.
PROVIDERS_INDEX_URL = "https://providers.optimade.org/providers.json"

#: Bundled catalogue file, generated from the index above.
CATALOGUE_RESOURCE = "optimade_providers.json"

#: Seconds before an HTTP request to a provider is abandoned.
REQUEST_TIMEOUT = 30.0

#: User agent sent with every request.
USER_AGENT = "abacustools"

#: Largest page a provider is asked for at once.
MAX_PAGE_LIMIT = 100

#: Upper bound on the pages followed for one search.
MAX_PAGES = 20

#: Page size used when entries are checked against the query client-side.
CHECK_PAGE_LIMIT = 20

#: Most entries scanned for one client-side checked filter strategy.
MAX_CHECK_ENTRIES = 60

#: Most pages read for one client-side checked filter strategy.
MAX_CHECK_PAGES = 3

#: A transport maps a URL to an already decoded JSON payload.
Transport = Callable[[str], Mapping[str, Any]]

#: Attributes whose value is reported as the energy above the convex hull.
_HULL_PATTERNS = ("energy_above_hull", "e_above_hull", "delta_e")

#: Attributes whose value is reported as the band gap.
_GAP_PATTERNS = ("band_gap", "bandgap")


class OptimadeHttpError(DatabaseRequestError):
    """A request that failed with an HTTP status code.

    Attributes:
        status: The HTTP status, when the failure came from a response.
        url: The URL that was requested.
    """

    def __init__(self, message: str, *, status: Optional[int] = None, url: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.url = url


@dataclass(frozen=True)
class OptimadeProvider:
    """One OPTIMADE deployment with a queryable ``structures`` endpoint."""

    name: str
    base_url: str
    title: str = ""
    description: str = ""
    homepage: str = ""
    aliases: tuple[str, ...] = ()
    requires_api_key: bool = False
    api_key_environment_variables: tuple[str, ...] = ()
    note: str = ""
    #: Search selectors the deployment actually answers, when that is less
    #: than the ones every OPTIMADE API is supposed to implement.
    selectors: Optional[tuple[str, ...]] = None

    def structures_urls(self) -> tuple[str, ...]:
        """Return the candidate URLs of the ``structures`` endpoint.

        Providers publish their API below a version path in principle, but
        some serve ``/structures`` at the root, so both are tried in turn.
        """
        base = self.base_url.rstrip("/")
        return (f"{base}/v1/structures", f"{base}/structures")

    def entry_url(self, identifier: str) -> str:
        """Return the URL of one structure resource."""
        quoted = urllib.parse.quote(str(identifier), safe="")
        return f"{self.structures_urls()[0]}/{quoted}"

    def label(self) -> str:
        """Return a human-readable name."""
        return self.title or self.name


def _provider_from_record(record: Mapping[str, Any]) -> OptimadeProvider:
    """Build a provider from one catalogue record."""
    return OptimadeProvider(
        name=str(record["name"]),
        base_url=str(record["base_url"]),
        title=str(record.get("title") or record["name"]),
        description=str(record.get("description") or ""),
        homepage=str(record.get("homepage") or ""),
        aliases=tuple(str(alias) for alias in record.get("aliases") or ()),
        requires_api_key=bool(record.get("requires_api_key", False)),
        api_key_environment_variables=tuple(
            str(name) for name in record.get("api_key_environment_variables") or ()
        ),
        note=str(record.get("note") or ""),
        selectors=(
            tuple(str(selector) for selector in record["selectors"])
            if record.get("selectors")
            else None
        ),
    )


def catalogue() -> list[OptimadeProvider]:
    """Return the bundled OPTIMADE providers."""
    payload = json.loads(
        resources.files(__package__).joinpath(CATALOGUE_RESOURCE).read_text(encoding="utf-8")
    )
    return [_provider_from_record(record) for record in payload.get("providers", [])]


def catalogue_source() -> Mapping[str, str]:
    """Return where and when the bundled catalogue was collected."""
    payload = json.loads(
        resources.files(__package__).joinpath(CATALOGUE_RESOURCE).read_text(encoding="utf-8")
    )
    return {
        "source": str(payload.get("source") or PROVIDERS_INDEX_URL),
        "retrieved": str(payload.get("retrieved") or ""),
        "note": str(payload.get("note") or ""),
    }


def provider_from_name(name: str) -> OptimadeProvider:
    """Return the catalogued provider called ``name``.

    Args:
        name: Provider name or alias.

    Returns:
        The matching provider.

    Raises:
        DatabaseRequestError: If the catalogue has no such provider.
    """
    providers = catalogue()
    for provider in providers:
        if name == provider.name or name in provider.aliases:
            return provider
    known = ", ".join(provider.name for provider in providers)
    raise DatabaseRequestError(f"unknown OPTIMADE provider {name!r}; choose one of: {known}")


def load_live_providers(transport: Optional[Transport] = None) -> list[dict[str, Optional[str]]]:
    """Return the providers published by the live OPTIMADE index.

    Args:
        transport: Optional stand-in for the HTTP request, used by tests.

    Returns:
        One record per published provider, with the ``name``, the index
        ``base_url``, the ``homepage`` and the first line of the description.
    """
    payload = _http_get_json(PROVIDERS_INDEX_URL, transport=transport)
    records = []
    for entry in payload.get("data") or ():
        attributes = entry.get("attributes") or {}
        records.append(
            {
                "name": _optional_str(entry.get("id")),
                "base_url": _optional_str(attributes.get("base_url")),
                "homepage": _optional_str(attributes.get("homepage")),
                "description": _first_line(attributes.get("description")),
            }
        )
    return records


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _first_line(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip().splitlines()[0]


def _http_get_json(
    url: str,
    *,
    transport: Optional[Transport] = None,
    timeout: float = REQUEST_TIMEOUT,
) -> Mapping[str, Any]:
    """Return the JSON payload of ``url``.

    Args:
        url: Absolute URL to request.
        transport: Optional stand-in for the HTTP request, used by tests.
        timeout: Seconds before the request is abandoned.

    Returns:
        The decoded payload.

    Raises:
        DatabaseRequestError: If the request fails or does not answer JSON.
    """
    if transport is not None:
        payload = transport(url)
        if not isinstance(payload, Mapping):
            raise DatabaseRequestError(
                f"transport returned {type(payload).__name__}, expected a mapping"
            )
        return payload
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise OptimadeHttpError(
            f"{url} returned HTTP {error.code}{_error_detail(error)}",
            status=error.code,
            url=url,
        ) from error
    except (urllib.error.URLError, OSError) as error:
        raise OptimadeHttpError(f"cannot reach {url}: {error}", url=url) from error
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatabaseRequestError(f"{url} did not return JSON: {error}") from error


def _error_detail(error: urllib.error.HTTPError) -> str:
    """Return the message of an OPTIMADE error response, if it carries one."""
    try:
        body = error.read()
    except OSError:
        return ""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    errors = payload.get("errors") if isinstance(payload, Mapping) else None
    if not errors:
        return ""
    first = errors[0] if isinstance(errors, (list, tuple)) and errors else errors
    if isinstance(first, Mapping):
        for key in ("detail", "reason", "message", "title"):
            if first.get(key):
                return f": {first[key]}"
    return ""


def _next_link(payload: Mapping[str, Any]) -> Optional[str]:
    """Return the ``links.next`` URL of a response, in either published form."""
    link = (payload.get("links") or {}).get("next")
    if isinstance(link, Mapping):
        link = link.get("href")
    return None if not link else str(link)


def _quote(value: str) -> str:
    """Escape a string for an OPTIMADE filter literal."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def reduced_formula(formula: str) -> str:
    """Return the reduced form of a composition such as ``Fe2O3``.

    Args:
        formula: Composition given on the command line.

    Returns:
        The reduced formula used by the ``chemical_formula_reduced`` filter.

    Raises:
        DatabaseRequestError: If the formula carries a wildcard or cannot be
            parsed.
    """
    if "*" in str(formula):
        raise DatabaseRequestError(
            "OPTIMADE filters match an exact composition; wildcards such as "
            f"{formula!r} are not supported"
        )
    from pymatgen.core import Composition

    try:
        return Composition(formula).reduced_formula
    except Exception as error:
        raise DatabaseRequestError(
            f"cannot parse the formula {formula!r} as a composition: {error}"
        ) from error


@dataclass(frozen=True)
class FilterStrategy:
    """One way of expressing a search as an OPTIMADE filter.

    Attributes:
        expression: The filter to send, or None when nothing is filtered.
        check_formula: Whether the entries need a client-side formula check,
            because the filter is a superset of the query.
    """

    expression: Optional[str] = None
    check_formula: bool = False


def _elements_term(elements: Sequence[str]) -> str:
    """Return the filter term that requires every element to be present."""
    quoted = ", ".join(f'"{_quote(element)}"' for element in elements)
    return f"elements HAS ALL {quoted}" if len(elements) > 1 else f"elements HAS {quoted}"


def _common_terms(query: DatabaseQuery) -> list[str]:
    """Return the terms every strategy of a query shares."""
    terms = []
    if query.identifiers:
        quoted = [f'id="{_quote(identifier)}"' for identifier in query.identifiers]
        terms.append(f"({quoted[0]})" if len(quoted) == 1 else "(" + " OR ".join(quoted) + ")")
    elements = [str(element) for element in query.elements or ()]
    if not elements and query.chemsys:
        elements = [part for part in re.split(r"[-\s,]+", str(query.chemsys)) if part]
    if elements:
        terms.append(_elements_term(elements))
    terms.extend(str(term) for term in query.where or ())
    return terms


def _composition(formula: str):
    """Return the pymatgen composition of a formula, or None."""
    from pymatgen.core import Composition

    try:
        return Composition(formula)
    except (ValueError, TypeError):
        return None


def _formula_elements(formula: str) -> list[str]:
    """Return the elements of a formula, in a deterministic order."""
    composition = _composition(formula)
    if composition is None:
        return []
    return sorted(str(element) for element in composition.elements)


def formula_matches(formula: Optional[str], requested: str) -> bool:
    """Return whether an entry composition is the requested one.

    Used when a database can only filter by elements, so that a formula
    search does not report entries with a different stoichiometry.
    """
    if not formula:
        return False
    entry = _composition(str(formula))
    wanted = _composition(str(requested))
    if entry is None or wanted is None:
        return False
    return entry.reduced_formula == wanted.reduced_formula


def build_strategies(query: DatabaseQuery) -> list[FilterStrategy]:
    """Translate a query into the OPTIMADE filters to try, in order.

    ``chemical_formula_reduced`` is the standard formula field, the
    description search is the fallback databases such as COD implement, and
    an element filter is the fallback for databases such as AFLOW that
    reject both. The element filter is a superset, so its entries are
    checked against the requested formula before they are reported.
    """
    common = _common_terms(query)
    candidates: list[FilterStrategy] = []
    if query.formula:
        reduced = reduced_formula(query.formula)
        candidates.append(FilterStrategy(f'chemical_formula_reduced="{_quote(reduced)}"'))
        candidates.append(
            FilterStrategy(
                f'chemical_formula_descriptive CONTAINS "{_quote(query.formula)}"',
                check_formula=True,
            )
        )
        elements = _formula_elements(query.formula)
        if elements:
            candidates.append(FilterStrategy(_elements_term(elements), check_formula=True))
    else:
        candidates.append(FilterStrategy())
    strategies = []
    for candidate in candidates:
        terms = [*common, candidate.expression] if candidate.expression else list(common)
        strategies.append(FilterStrategy(" AND ".join(terms) or None, candidate.check_formula))
    return strategies


def build_filters(query: DatabaseQuery) -> list[Optional[str]]:
    """Return the OPTIMADE ``filter`` strings a query tries, in order."""
    return [strategy.expression for strategy in build_strategies(query)]


def build_filter(query: DatabaseQuery) -> Optional[str]:
    """Return the first OPTIMADE ``filter`` string of a query."""
    return build_filters(query)[0]


def _number(value: Any) -> Optional[float]:
    """Return a number found in a value, a mapping, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        for key in ("value", "energy_above_hull", "band_gap"):
            if key in value:
                found = _number(value[key])
                if found is not None:
                    return found
    return None


def _attribute_number(attributes: Mapping[str, Any], patterns: Sequence[str]) -> Optional[float]:
    """Return the first provider-specific number matching one of ``patterns``."""
    for name in sorted(attributes):
        lowered = name.lower()
        if any(pattern in lowered for pattern in patterns):
            value = _number(attributes[name])
            if value is not None:
                return value
    return None


def _mp_hull_energy(attributes: Mapping[str, Any]) -> Optional[float]:
    """Return the GGA convex-hull distance of a Materials Project entry."""
    stability = attributes.get("_mp_stability")
    if not isinstance(stability, Mapping):
        return None
    for functional in sorted(stability):
        record = stability[functional]
        if isinstance(record, Mapping):
            value = _number(record.get("energy_above_hull"))
            if value is not None:
                return value
    return None


def _nsites(attributes: Mapping[str, Any]) -> Optional[int]:
    """Return the site count, deriving it when the provider omits it."""
    nsites = attributes.get("nsites")
    if nsites is not None:
        try:
            return int(nsites)
        except (TypeError, ValueError):
            return None
    species = attributes.get("species_at_sites") or ()
    return len(species) or None


def _species_name(value: Any) -> str:
    """Return the element or label of one ``species_at_sites`` entry."""
    if isinstance(value, Mapping):
        value = value.get("name") or (value.get("chemical_symbols") or [""])[0]
    return str(value)


def structure_from_entry(entry: Mapping[str, Any]):
    """Build a pymatgen structure from one OPTIMADE structure entry.

    Args:
        entry: One element of the ``data`` list of a structures response.

    Returns:
        The corresponding :class:`pymatgen.core.Structure`.

    Raises:
        DatabaseRequestError: If the entry carries no lattice or positions.
    """
    from pymatgen.core import Lattice, Structure

    attributes = entry.get("attributes") or {}
    lattice = attributes.get("lattice_vectors")
    positions = attributes.get("cartesian_site_positions")
    species = attributes.get("species_at_sites")
    if not lattice or not positions or not species:
        raise DatabaseRequestError(
            f"entry {entry.get('id')!r} has no coordinates to build a structure"
        )
    if len(positions) != len(species):
        raise DatabaseRequestError(
            f"entry {entry.get('id')!r} has {len(positions)} positions but {len(species)} species"
        )
    return Structure(
        Lattice(lattice),
        [_species_name(site) for site in species],
        positions,
        coords_are_cartesian=True,
    )


def summary_from_entry(
    entry: Mapping[str, Any],
    database: str,
    provider: Optional[OptimadeProvider] = None,
) -> DatabaseSummary:
    """Convert one OPTIMADE structure entry into a database summary.

    Args:
        entry: One element of the ``data`` list of a structures response.
        database: Registry name to record in the summary.
        provider: The provider the entry came from, when it is known.

    Returns:
        The entry in the database-independent form.
    """
    attributes = entry.get("attributes") or {}
    elements = [str(element) for element in attributes.get("elements") or ()]
    extra: dict[str, Any] = {}
    if provider is not None:
        extra["provider"] = provider.name
    extra.update({name: value for name, value in attributes.items() if str(name).startswith("_")})
    hull = _mp_hull_energy(attributes)
    if hull is None:
        hull = _attribute_number(attributes, _HULL_PATTERNS)
    return DatabaseSummary(
        database=database,
        identifier=str(entry.get("id")),
        formula=(
            attributes.get("chemical_formula_reduced")
            or attributes.get("chemical_formula_descriptive")
        ),
        chemsys="-".join(sorted(elements)) or attributes.get("_mp_chemical_system"),
        nsites=_nsites(attributes),
        energy_above_hull=hull,
        band_gap=_attribute_number(attributes, _GAP_PATTERNS),
        extra=extra,
    )


class OptimadeDatabase(StructureDatabase):
    """One OPTIMADE deployment, or the federation of all of them.

    A database built without ``provider`` searches every catalogued
    provider; one built with ``provider`` queries that deployment only.
    """

    protocol = "OPTIMADE"
    capabilities = frozenset({"formula", "chemsys", "elements", "identifiers"})
    options = frozenset({"provider", "base_url"})

    def __init__(self, provider: Optional[OptimadeProvider] = None) -> None:
        self.provider = provider
        if provider is None:
            self.name = "optimade"
            self.description = "OPTIMADE federation: search every catalogued provider"
            self.aggregate = True
        else:
            self.name = provider.name
            self.description = provider.title or provider.name
            self.requires_api_key = provider.requires_api_key
            self.api_key_environment_variables = provider.api_key_environment_variables
            if provider.selectors:
                self.capabilities = frozenset(provider.selectors)
            self.aggregate = False

    def available(self) -> bool:
        """Return whether this database has a provider to query."""
        return bool(self.provider) or bool(catalogue())

    def unavailable_reason(self) -> Optional[str]:
        """Return why the federation is unavailable, or None."""
        if self.available():
            return None
        return "the bundled OPTIMADE provider catalogue is empty"

    def _targets(
        self,
        provider: Optional[str],
        base_url: Optional[str],
    ) -> list[OptimadeProvider]:
        """Resolve the providers one query has to visit."""
        if provider:
            target = provider_from_name(provider)
            return [replace(target, base_url=base_url) if base_url else target]
        if base_url:
            template = self.provider
            if template is None:
                return [
                    OptimadeProvider(
                        name="optimade-custom",
                        base_url=base_url,
                        title="custom OPTIMADE endpoint",
                    )
                ]
            return [replace(template, base_url=base_url)]
        if self.provider is not None:
            return [self.provider]
        return [
            entry
            for entry in catalogue()
            if not entry.requires_api_key
            or api_key_from_environment(entry.api_key_environment_variables)
        ]

    def _request_structures(
        self,
        provider: OptimadeProvider,
        params: Mapping[str, Any],
        *,
        transport: Optional[Transport],
    ) -> tuple[Mapping[str, Any], str]:
        """Return the first structures payload a provider answers with."""
        query = urllib.parse.urlencode(params)
        failures = []
        for url in provider.structures_urls():
            request_url = f"{url}?{query}" if query else url
            try:
                return _http_get_json(request_url, transport=transport), request_url
            except OptimadeHttpError as error:
                if error.status in (404, 405):
                    failures.append(str(error))
                    continue
                raise
        raise DatabaseRequestError(
            f"{provider.name}: no structures endpoint found ({'; '.join(failures)})"
        )

    def _collect_entries(
        self,
        payload: Mapping[str, Any],
        url: str,
        target: int,
        *,
        transport: Optional[Transport],
        max_pages: int = MAX_PAGES,
    ) -> list[Mapping[str, Any]]:
        """Return the entries of a response, following its ``next`` links."""
        entries = list(payload.get("data") or ())
        pages = 1
        while len(entries) < target and pages < max_pages:
            next_link = _next_link(payload)
            if not next_link:
                break
            url = urllib.parse.urljoin(url, next_link)
            payload = _http_get_json(url, transport=transport)
            fresh = list(payload.get("data") or ())
            if not fresh:
                break
            entries.extend(fresh)
            pages += 1
        return entries

    def _search_one(
        self,
        provider: OptimadeProvider,
        query: DatabaseQuery,
        *,
        api_key: Optional[str],
        transport: Optional[Transport],
    ) -> list[DatabaseSummary]:
        """Search one provider, trying each filter strategy in turn."""
        params: dict[str, Any] = {"page_limit": min(query.limit, MAX_PAGE_LIMIT)}
        if provider.requires_api_key:
            params["token"] = api_key or ""
        failure: Optional[DatabaseRequestError] = None
        empty_result: Optional[list[DatabaseSummary]] = None
        for strategy in build_strategies(query):
            attempt = dict(params)
            if strategy.expression:
                attempt["filter"] = strategy.expression
            if strategy.check_formula:
                attempt["page_limit"] = min(CHECK_PAGE_LIMIT, MAX_PAGE_LIMIT)
            try:
                payload, url = self._request_structures(provider, attempt, transport=transport)
            except OptimadeHttpError as error:
                # A rejected field moves on to the next strategy; a provider
                # that breaks on a fallback must not turn a valid, empty
                # answer into an error.
                rejected = error.status in (400, 501) or (
                    empty_result is not None and (error.status or 0) >= 500
                )
                if rejected:
                    failure = error
                    continue
                raise
            target = MAX_CHECK_ENTRIES if strategy.check_formula else query.limit
            entries = self._collect_entries(
                payload,
                url,
                target,
                transport=transport,
                max_pages=MAX_CHECK_PAGES if strategy.check_formula else MAX_PAGES,
            )
            summaries = [summary_from_entry(entry, provider.name, provider) for entry in entries]
            if strategy.check_formula:
                summaries = [
                    summary
                    for summary in summaries
                    if formula_matches(summary.formula, query.formula or "")
                ]
            if summaries:
                return summaries[: query.limit]
            if empty_result is None:
                empty_result = summaries
        if empty_result is not None:
            return empty_result[: query.limit]
        raise failure or DatabaseRequestError(f"{provider.name}: no filter strategy was accepted")

    def search(
        self,
        query: DatabaseQuery,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        transport: Optional[Transport] = None,
        **options: Any,
    ) -> list[DatabaseSummary]:
        """Search one OPTIMADE provider, or all of them at once."""
        self.check_query(query)
        query.require_selector()
        targets = self._targets(provider, base_url)
        if not targets:
            raise DatabaseRequestError("no OPTIMADE provider is available to query")
        resolved_key: dict[str, Optional[str]] = {}
        summaries: list[DatabaseSummary] = []
        failures: list[str] = []
        for target in targets:
            if len(summaries) >= query.limit:
                break
            if target.requires_api_key:
                names = target.api_key_environment_variables
                key = api_key or api_key_from_environment(names)
                if key is None:
                    failures.append(
                        f"{target.name}: no API key ({', '.join(names) or 'none known'})"
                    )
                    if not self.aggregate:
                        raise DatabaseApiKeyError(
                            f"database {target.name!r} needs an API key; set "
                            f"{', '.join(names)} or pass --api-key"
                        )
                    continue
                resolved_key[target.name] = key
            remaining = replace(query, limit=query.limit - len(summaries))
            try:
                summaries.extend(
                    self._search_one(
                        target,
                        remaining,
                        api_key=resolved_key.get(target.name),
                        transport=transport,
                    )
                )
            except DatabaseApiKeyError:
                raise
            except DatabaseRequestError as error:
                if not self.aggregate:
                    raise
                failures.append(f"{target.name}: {error}")
        if self.aggregate and failures and not summaries:
            raise DatabaseRequestError("every OPTIMADE provider failed: " + "; ".join(failures))
        return summaries[: query.limit]

    def fetch(
        self,
        identifier: str,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        transport: Optional[Transport] = None,
        **options: Any,
    ) -> DatabaseStructure:
        """Download one structure from a single OPTIMADE provider."""
        targets = self._targets(provider, base_url)
        if len(targets) != 1:
            raise DatabaseRequestError(
                "the OPTIMADE federation has no shared identifier space; name "
                "one provider, for example '-d cod' or '--provider cod'"
            )
        target = targets[0]
        key: Optional[str] = None
        if target.requires_api_key:
            names = target.api_key_environment_variables
            key = api_key or api_key_from_environment(names)
            if key is None:
                raise DatabaseApiKeyError(
                    f"database {target.name!r} needs an API key; set "
                    f"{', '.join(names) or 'its token'} or pass --api-key"
                )
        url = target.entry_url(identifier)
        if key:
            url = f"{url}?token={urllib.parse.quote(key)}"
        try:
            payload = _http_get_json(url, transport=transport)
        except OptimadeHttpError as error:
            if error.status == 404:
                raise LookupError(f"{target.name} has no entry {identifier!r}") from error
            raise
        data = payload.get("data")
        if isinstance(data, (list, tuple)):
            data = data[0] if data else None
        if not isinstance(data, Mapping):
            raise LookupError(f"{target.name} has no entry {identifier!r}")
        return DatabaseStructure(
            summary=summary_from_entry(data, target.name, target),
            structure=structure_from_entry(data),
        )


def register() -> list[OptimadeDatabase]:
    """Register the OPTIMADE federation and every catalogued provider."""
    registered = [
        register_database(
            OptimadeDatabase(),
            aliases=("optimade-federation",),
            replace=True,
        )
    ]
    for provider in catalogue():
        registered.append(
            register_database(
                OptimadeDatabase(provider),
                aliases=provider.aliases,
                replace=True,
            )
        )
    return registered
