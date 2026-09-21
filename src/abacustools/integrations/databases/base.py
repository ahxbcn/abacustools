"""Common types and errors for the pluggable structure-database layer.

Every external materials database is wrapped by a :class:`StructureDatabase`
subclass answering two questions: how to search for entries, and how to fetch
the structure of one entry. The CLI and the rest of abacustools only talk to
that interface, so adding a database never changes a command.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Mapping, Optional, Sequence

#: Environment variable that overrides the CLI's default database.
DEFAULT_DATABASE_ENVIRONMENT_VARIABLE = "ABACUSTOOLS_DATABASE"

#: Query selectors understood by :class:`DatabaseQuery`, in reporting order.
SELECTORS = ("formula", "chemsys", "elements", "identifiers")


class DatabaseError(RuntimeError):
    """Base class for database failures other than a missing client package."""


class DatabaseUnavailableError(ImportError):
    """Raised when a database is requested but its client package is missing."""


class DatabaseApiKeyError(DatabaseError):
    """Raised when a database needs an API key and none can be found."""


class DatabaseRequestError(DatabaseError):
    """Raised when a query is malformed, unsupported, or refused."""


def api_key_from_environment(
    names: Sequence[str],
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return the first API key found under ``names``, or None.

    Args:
        names: Environment variable names, in order of precedence.
        environ: Mapping to read instead of :data:`os.environ`.

    Returns:
        The first non-empty value, or None when none is set.
    """
    environment = os.environ if environ is None else environ
    for name in names:
        value = environment.get(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class DatabaseSummary:
    """One database entry, normalised across databases.

    Fields that a database does not report stay None; provider-specific
    values are kept verbatim in :attr:`extra`.
    """

    database: str
    identifier: str
    formula: Optional[str] = None
    chemsys: Optional[str] = None
    nsites: Optional[int] = None
    volume: Optional[float] = None
    energy_above_hull: Optional[float] = None
    band_gap: Optional[float] = None
    is_stable: Optional[bool] = None
    theoretical: Optional[bool] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a JSON-compatible dictionary."""
        return {
            "database": self.database,
            "id": self.identifier,
            "formula": self.formula,
            "chemsys": self.chemsys,
            "nsites": self.nsites,
            "volume": self.volume,
            "energy_above_hull": self.energy_above_hull,
            "band_gap": self.band_gap,
            "is_stable": self.is_stable,
            "theoretical": self.theoretical,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class DatabaseStructure:
    """A downloaded structure together with its summary."""

    summary: DatabaseSummary
    structure: Any

    @property
    def identifier(self) -> str:
        """Return the database identifier of this structure."""
        return self.summary.identifier

    @property
    def database(self) -> str:
        """Return the name of the database the structure came from."""
        return self.summary.database

    def to_abacus_structure(self):
        """Convert to an :class:`~abacustools.io.stru.AbacusSTRU`."""
        from abacustools.io.stru import AbacusSTRU

        return AbacusSTRU.from_pymatgen(
            self.structure,
            meta_data={
                "source": self.database,
                "identifier": self.identifier,
            },
        )


@dataclass(frozen=True)
class DatabaseQuery:
    """Database-independent selectors of a search.

    Attributes:
        formula: Composition such as ``Fe2O3``.
        chemsys: Chemical system such as ``Li-Fe-O``.
        elements: Elements that must all be present.
        identifiers: Restrict the search to specific entry identifiers.
        is_stable: Keep only entries on the convex hull, when supported.
        theoretical: Keep only entries without an experimental counterpart,
            when supported.
        limit: Maximum number of summaries to return.
        fields: Provider-specific summary fields to request; databases that
            have no such notion ignore it.
    """

    formula: Optional[str] = None
    chemsys: Optional[str] = None
    elements: Optional[tuple[str, ...]] = None
    identifiers: Optional[tuple[str, ...]] = None
    is_stable: Optional[bool] = None
    theoretical: Optional[bool] = None
    limit: int = 20
    fields: Optional[tuple[str, ...]] = None

    def selectors(self) -> list[str]:
        """Return the selector names this query actually uses."""
        used = []
        for name in SELECTORS:
            if getattr(self, name):
                used.append(name)
        if self.is_stable is not None:
            used.append("stability")
        if self.theoretical is not None:
            used.append("theoretical")
        return used

    def require_selector(self) -> None:
        """Raise if the query carries no selector at all.

        Raises:
            DatabaseRequestError: If no selector is set.
        """
        if not any(getattr(self, name) for name in SELECTORS):
            joined = ", ".join(SELECTORS)
            raise DatabaseRequestError(f"a search needs at least one selector: {joined}")

    def unsupported(self, capabilities: FrozenSet[str]) -> list[str]:
        """Return the selectors this query uses that a database does not support."""
        supported = set(capabilities)
        return [name for name in self.selectors() if name not in supported]


class StructureDatabase(ABC):
    """One searchable and downloadable structure database.

    Subclasses declare what they can do through the class attributes and
    implement :meth:`search` and :meth:`fetch`. Instances are stateless, so
    the registry keeps one instance per database.
    """

    #: Registry key, also used as the ``database`` field of summaries.
    name: str = ""
    #: One-line description shown by ``abacustools database list``.
    description: str = ""
    #: Name of the access route, such as ``mp-api`` or ``OPTIMADE``.
    protocol: str = ""
    #: Selectors from :data:`SELECTORS` this database understands.
    capabilities: FrozenSet[str] = frozenset()
    #: Provider-specific options this database accepts, such as ``provider``.
    options: FrozenSet[str] = frozenset()
    #: Whether a search needs an API key.
    requires_api_key: bool = False
    #: Environment variables that may hold an API key, in order of precedence.
    api_key_environment_variables: tuple[str, ...] = ()
    #: How to install the optional client, shown when it is missing.
    install_hint: str = ""

    def available(self) -> bool:
        """Return whether this database can be queried right now."""
        return True

    def unavailable_reason(self) -> Optional[str]:
        """Return why the database is unavailable, or None when it is fine."""
        return None

    def status(self) -> str:
        """Return ``ready``, ``needs-api-key``, or ``unavailable``."""
        if not self.available():
            return "unavailable"
        if self.requires_api_key and not self.api_key_from_environment():
            return "needs-api-key"
        return "ready"

    def api_key_from_environment(
        self,
        environ: Optional[Mapping[str, str]] = None,
    ) -> Optional[str]:
        """Return the first environment API key of this database, or None."""
        return api_key_from_environment(self.api_key_environment_variables, environ)

    def resolve_api_key(
        self,
        api_key: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> str:
        """Return an explicit or environment-provided API key.

        Args:
            api_key: Explicit key, which wins over the environment.
            environ: Mapping to read instead of :data:`os.environ`.

        Returns:
            The resolved API key.

        Raises:
            DatabaseApiKeyError: If no key is given and none is set.
        """
        if api_key:
            return str(api_key)
        key = self.api_key_from_environment(environ)
        if key is None:
            names = ", ".join(self.api_key_environment_variables)
            raise DatabaseApiKeyError(
                f"no API key found for database {self.name!r}; set one of "
                f"{names} or pass an explicit key."
            )
        return key

    def supports(self, capability: str) -> bool:
        """Return whether this database supports one selector."""
        return capability in self.capabilities

    def check_query(self, query: DatabaseQuery) -> None:
        """Validate a query against this database's capabilities.

        Raises:
            DatabaseRequestError: If the query uses an unsupported selector,
                or if the limit is not positive.
        """
        if query.limit < 1:
            raise DatabaseRequestError("limit must be a positive integer")
        if query.fields and not self.supports("fields"):
            raise DatabaseRequestError(f"database {self.name!r} has no summary fields to select")
        unsupported = query.unsupported(self.capabilities)
        if unsupported:
            supported = ", ".join(sorted(self.capabilities)) or "none"
            raise DatabaseRequestError(
                f"database {self.name!r} does not support the selector(s) "
                f"{', '.join(unsupported)}; it supports: {supported}"
            )

    def check_options(self, **options: Any) -> None:
        """Validate provider-specific options against this database.

        Raises:
            DatabaseRequestError: If an option is not supported by this
                database.
        """
        unsupported = sorted(
            name
            for name, value in options.items()
            if value is not None and name not in self.options
        )
        if unsupported:
            supported = ", ".join(sorted(self.options)) or "none"
            raise DatabaseRequestError(
                f"database {self.name!r} does not take the option(s) "
                f"{', '.join(unsupported)}; it accepts: {supported}"
            )

    @abstractmethod
    def search(
        self,
        query: DatabaseQuery,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        **options: Any,
    ) -> list[DatabaseSummary]:
        """Return the entries matching ``query``.

        Args:
            query: The selectors to search with.
            api_key: Explicit API key; otherwise the environment is used.
            client: Pre-constructed client, used to avoid the network.
            **options: Provider-specific options, such as ``provider`` or
                ``base_url`` for the OPTIMADE databases.

        Returns:
            The matching summaries, in the order returned by the database.
        """

    @abstractmethod
    def fetch(
        self,
        identifier: str,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        **options: Any,
    ) -> DatabaseStructure:
        """Return one structure and its summary.

        Args:
            identifier: Database entry identifier.
            api_key: Explicit API key; otherwise the environment is used.
            client: Pre-constructed client, used to avoid the network.
            **options: Provider-specific options.

        Returns:
            The downloaded structure and its summary.

        Raises:
            LookupError: If the database has no structure for ``identifier``.
        """
