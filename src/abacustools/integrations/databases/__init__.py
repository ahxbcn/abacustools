"""Structure databases abacustools can search and download from.

Two access routes cover the databases that are usable without buying a
subscription: the Materials Project client (``mp-api``), and OPTIMADE, the
REST protocol spoken by most other open crystal-structure databases. Both are
wrapped in the same :class:`~abacustools.integrations.databases.base.StructureDatabase`
interface and registered here, so commands never name a particular database.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .base import (
    DEFAULT_DATABASE_ENVIRONMENT_VARIABLE,
    SELECTORS,
    DatabaseApiKeyError,
    DatabaseError,
    DatabaseQuery,
    DatabaseRequestError,
    DatabaseStructure,
    DatabaseSummary,
    DatabaseUnavailableError,
    StructureDatabase,
    api_key_from_environment,
)
from .files import (
    STRUCTURE_FILENAMES,
    structure_filename,
    structure_path,
    write_structure,
)
from .c2db import (
    C2DBDatabase,
    keys_source as c2db_keys_source,
    load_keys as c2db_keys,
)
from .optimade import (
    PROVIDERS_INDEX_URL,
    OptimadeHttpError,
    OptimadeProvider,
    catalogue as optimade_catalogue,
    catalogue_source as optimade_catalogue_source,
    load_live_providers,
    provider_from_name as optimade_provider_from_name,
)
from .registry import (
    available_databases,
    database_names,
    databases,
    default_database_name,
    get_database,
    register_database,
    unregister_database,
)

from . import c2db as _c2db_provider
from . import materials_project as _materials_project_provider
from . import optimade as _optimade_provider

_materials_project_provider.register()
_optimade_provider.register()
_c2db_provider.register()


def default_database(environ: Optional[Mapping[str, str]] = None) -> StructureDatabase:
    """Return the database the CLI uses when none is named.

    Args:
        environ: Mapping to read instead of :data:`os.environ`.

    Returns:
        The default database adapter.
    """
    return get_database(default_database_name(dict(environ) if environ else None))


def describe_databases() -> list[dict[str, Any]]:
    """Return one report record per registered database.

    Returns:
        Records with the name, access route, capabilities, and status of
        every database.
    """
    return [
        {
            "name": database.name,
            "protocol": database.protocol,
            "description": database.description,
            "capabilities": sorted(database.capabilities),
            "requires_api_key": database.requires_api_key,
            "api_key_environment_variables": list(database.api_key_environment_variables),
            "status": database.status(),
            "detail": database.unavailable_reason() or "",
            "install_hint": database.install_hint,
        }
        for database in databases()
    ]


__all__ = [
    "DEFAULT_DATABASE_ENVIRONMENT_VARIABLE",
    "DatabaseApiKeyError",
    "DatabaseError",
    "DatabaseQuery",
    "DatabaseRequestError",
    "DatabaseStructure",
    "DatabaseSummary",
    "C2DBDatabase",
    "DatabaseUnavailableError",
    "OptimadeHttpError",
    "OptimadeProvider",
    "PROVIDERS_INDEX_URL",
    "SELECTORS",
    "STRUCTURE_FILENAMES",
    "StructureDatabase",
    "api_key_from_environment",
    "available_databases",
    "c2db_keys",
    "c2db_keys_source",
    "database_names",
    "databases",
    "default_database",
    "default_database_name",
    "describe_databases",
    "get_database",
    "load_live_providers",
    "optimade_catalogue",
    "optimade_catalogue_source",
    "optimade_provider_from_name",
    "register_database",
    "structure_filename",
    "structure_path",
    "unregister_database",
    "write_structure",
]
