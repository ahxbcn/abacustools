"""The Materials Project as a registered structure database.

The HTTP client and the summary documents stay in
:mod:`abacustools.integrations.materials_project`; this module only presents
them through the database interface of :mod:`abacustools.integrations.databases`.
"""

from __future__ import annotations

from typing import Any, Optional

from ..materials_project import (
    API_KEY_ENVIRONMENT_VARIABLES,
    MaterialSummary,
    MaterialsProjectApiKeyError,
    MaterialsProjectUnavailableError,
    download_material,
    materials_project_available,
    search_materials,
)
from .base import (
    DatabaseApiKeyError,
    DatabaseQuery,
    DatabaseRequestError,
    DatabaseStructure,
    DatabaseSummary,
    DatabaseUnavailableError,
    StructureDatabase,
)
from .registry import register_database


def summary_from_material(
    database: str,
    summary: MaterialSummary,
) -> DatabaseSummary:
    """Convert a Materials Project summary into a database summary.

    Args:
        database: Registry name to record in the summary.
        summary: Summary returned by the Materials Project adapter.

    Returns:
        The same entry in the database-independent form.
    """
    return DatabaseSummary(
        database=database,
        identifier=summary.material_id,
        formula=summary.formula,
        chemsys=summary.chemsys,
        nsites=summary.nsites,
        volume=summary.volume,
        energy_above_hull=summary.energy_above_hull,
        band_gap=summary.band_gap,
        is_stable=summary.is_stable,
        theoretical=summary.theoretical,
        extra={"material_id": summary.material_id},
    )


class MaterialsProjectDatabase(StructureDatabase):
    """Search and download entries of the Materials Project."""

    name = "mp"
    description = "Materials Project: DFT energies and structures"
    protocol = "mp-api"
    capabilities = frozenset(
        {
            "formula",
            "chemsys",
            "elements",
            "identifiers",
            "stability",
            "theoretical",
            "fields",
        }
    )
    requires_api_key = True
    api_key_environment_variables = tuple(API_KEY_ENVIRONMENT_VARIABLES)
    install_hint = "pip install 'abacustools[mp]'"

    def available(self) -> bool:
        """Return whether the optional ``mp-api`` client can be imported."""
        return materials_project_available()

    def unavailable_reason(self) -> Optional[str]:
        """Return why the Materials Project is unavailable, or None."""
        if self.available():
            return None
        return (
            "the optional 'mp-api' package is missing; install it with "
            f"{self.install_hint} and register a key at "
            "https://materialsproject.org/api"
        )

    def search(
        self,
        query: DatabaseQuery,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        **options: Any,
    ) -> list[DatabaseSummary]:
        """Search the Materials Project summary endpoint."""
        self.check_query(query)
        query.require_selector()
        try:
            summaries = search_materials(
                formula=query.formula,
                chemsys=query.chemsys,
                elements=list(query.elements) if query.elements else None,
                material_ids=list(query.identifiers) if query.identifiers else None,
                is_stable=query.is_stable,
                theoretical=query.theoretical,
                limit=query.limit,
                fields=list(query.fields) if query.fields else None,
                api_key=api_key,
                client=client,
            )
        except MaterialsProjectUnavailableError as error:
            raise DatabaseUnavailableError(str(error)) from error
        except MaterialsProjectApiKeyError as error:
            raise DatabaseApiKeyError(str(error)) from error
        except OSError as error:
            raise DatabaseRequestError(
                f"cannot reach the Materials Project: {error}"
            ) from error
        except RuntimeError as error:
            raise DatabaseRequestError(
                f"the Materials Project refused the query: {error}"
            ) from error
        return [summary_from_material(self.name, summary) for summary in summaries]

    def fetch(
        self,
        identifier: str,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        **options: Any,
    ) -> DatabaseStructure:
        """Download one Materials Project structure."""
        try:
            material = download_material(identifier, api_key=api_key, client=client)
        except MaterialsProjectUnavailableError as error:
            raise DatabaseUnavailableError(str(error)) from error
        except MaterialsProjectApiKeyError as error:
            raise DatabaseApiKeyError(str(error)) from error
        except OSError as error:
            raise DatabaseRequestError(
                f"cannot reach the Materials Project: {error}"
            ) from error
        except RuntimeError as error:
            raise DatabaseRequestError(
                f"the Materials Project refused the query: {error}"
            ) from error
        return DatabaseStructure(
            summary=summary_from_material(self.name, material.summary),
            structure=material.structure,
        )


def register() -> MaterialsProjectDatabase:
    """Register the Materials Project database and return its adapter."""
    return register_database(
        MaterialsProjectDatabase(),
        aliases=("materials-project", "materialsproject", "materials_project"),
        replace=True,
    )
