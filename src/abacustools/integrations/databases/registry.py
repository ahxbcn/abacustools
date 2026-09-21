"""Look-up of the structure databases abacustools can talk to.

Databases register themselves when :mod:`abacustools.integrations.databases` is
imported, and commands look one up by name. The registry is the single place
that knows which databases exist, so ``abacustools database list`` and the
``--database`` option always agree.
"""

from __future__ import annotations

from typing import Optional

from .base import (
    DatabaseRequestError,
    StructureDatabase,
)

_REGISTRY: dict[str, StructureDatabase] = {}


def register_database(
    database: StructureDatabase,
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> StructureDatabase:
    """Add one database to the registry.

    Args:
        database: The database adapter to register.
        aliases: Extra names that resolve to the same adapter.
        replace: Overwrite an existing entry instead of raising.

    Returns:
        The registered database.

    Raises:
        ValueError: If the name or an alias is already taken and ``replace``
            is false.
    """
    names = (database.name, *aliases)
    for name in names:
        if not name:
            raise ValueError("a database needs a non-empty name")
        if name in _REGISTRY and not replace:
            raise ValueError(f"database {name!r} is already registered")
    _REGISTRY[database.name] = database
    for alias in aliases:
        _REGISTRY[alias] = database
    return database


def unregister_database(name: str) -> None:
    """Remove a database and its aliases from the registry.

    Args:
        name: Any registry name of the database.
    """
    database = _REGISTRY.get(name)
    if database is None:
        return
    for key, value in list(_REGISTRY.items()):
        if value is database:
            del _REGISTRY[key]


def get_database(name: str) -> StructureDatabase:
    """Return the database registered under ``name``.

    Args:
        name: Registry name of the database.

    Returns:
        The matching database adapter.

    Raises:
        DatabaseRequestError: If no database is registered under ``name``.
    """
    try:
        return _REGISTRY[str(name)]
    except KeyError:
        known = ", ".join(database_names())
        raise DatabaseRequestError(f"unknown database {name!r}; choose one of: {known}") from None


def databases() -> list[StructureDatabase]:
    """Return every registered database once, sorted by name."""
    unique: dict[int, StructureDatabase] = {}
    for database in _REGISTRY.values():
        unique.setdefault(id(database), database)
    return sorted(unique.values(), key=lambda database: database.name)


def database_names() -> list[str]:
    """Return the canonical database names, sorted."""
    return sorted(database.name for database in databases())


def available_databases() -> list[StructureDatabase]:
    """Return the registered databases that are ready to query."""
    return [database for database in databases() if database.available()]


def default_database_name(environ: Optional[dict] = None) -> str:
    """Return the CLI's default database name.

    The ``ABACUSTOOLS_DATABASE`` environment variable wins, then ``mp``.

    Args:
        environ: Mapping to read instead of :data:`os.environ`.

    Returns:
        The name of a registered database.
    """
    import os

    from .base import DEFAULT_DATABASE_ENVIRONMENT_VARIABLE

    environment = os.environ if environ is None else environ
    configured = environment.get(DEFAULT_DATABASE_ENVIRONMENT_VARIABLE)
    if configured:
        return str(configured)
    return "mp"
