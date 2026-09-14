"""Adapters for searching and downloading from the Materials Project.

``mp-api`` owns the HTTP client and the pydantic summary documents. This module
only connects them to the structures, files, and reporting conventions that
already exist in :mod:`abacustools`, so the dependency stays optional and the
CLI layer stays thin.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

#: Environment variables that may hold an API key, in order of precedence.
API_KEY_ENVIRONMENT_VARIABLES = ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")

#: Summary fields requested by ``abacustools mp search`` by default.
DEFAULT_SUMMARY_FIELDS = (
    "material_id",
    "formula_pretty",
    "chemsys",
    "nsites",
    "volume",
    "energy_above_hull",
    "band_gap",
    "is_stable",
    "theoretical",
)

#: Structure format -> file name written below the per-material directory.
STRUCTURE_FILENAMES = {
    "stru": "STRU",
    "poscar": "POSCAR",
    "cif": "structure.cif",
    "xyz": "structure.xyz",
    "extxyz": "structure.extxyz",
    "xsf": "structure.xsf",
}


class MaterialsProjectUnavailableError(ImportError):
    """Raised when the Materials Project client is requested but unavailable."""


class MaterialsProjectApiKeyError(RuntimeError):
    """Raised when no Materials Project API key can be found."""


def load_materials_project():
    """Return the optional ``MPRester`` class from :mod:`mp_api.client`.

    The dependency is imported lazily so normal ABACUSTools workflows do not
    need to install the separate Materials Project client.
    """
    try:
        from mp_api.client import MPRester
    except ImportError as error:
        raise MaterialsProjectUnavailableError(
            "The optional 'mp-api' package is required for Materials Project "
            "queries. Install it with 'pip install abacustools[mp]' and "
            "register an API key at https://materialsproject.org/api."
        ) from error
    return MPRester


def materials_project_available() -> bool:
    """Return whether the optional Materials Project client can be imported."""
    try:
        load_materials_project()
    except MaterialsProjectUnavailableError:
        return False
    return True


def api_key_from_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return the first API key found in the environment, or None."""
    environment = os.environ if environ is None else environ
    for name in API_KEY_ENVIRONMENT_VARIABLES:
        value = environment.get(name)
        if value:
            return value
    return None


def resolve_api_key(
    api_key: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return an explicit or environment-provided API key.

    Raises:
        MaterialsProjectApiKeyError: If no key is given and none of
            :data:`API_KEY_ENVIRONMENT_VARIABLES` is set.
    """
    if api_key:
        return str(api_key)
    key = api_key_from_environment(environ)
    if key is None:
        names = ", ".join(API_KEY_ENVIRONMENT_VARIABLES)
        raise MaterialsProjectApiKeyError(
            "no Materials Project API key found; set one of "
            f"{names} (see https://materialsproject.org/api) or pass an "
            "explicit key."
        )
    return key


def _document_value(document: Any, name: str) -> Any:
    """Read one field from an ``mp-api`` document or a plain mapping."""
    if isinstance(document, Mapping):
        return document.get(name)
    return getattr(document, name, None)


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _optional_bool(value: Any) -> Optional[bool]:
    return None if value is None else bool(value)


@dataclass(frozen=True)
class MaterialSummary:
    """Selected fields of one Materials Project summary document."""

    material_id: str
    formula: Optional[str] = None
    chemsys: Optional[str] = None
    nsites: Optional[int] = None
    volume: Optional[float] = None
    energy_above_hull: Optional[float] = None
    band_gap: Optional[float] = None
    is_stable: Optional[bool] = None
    theoretical: Optional[bool] = None

    @classmethod
    def from_document(cls, document: Any) -> "MaterialSummary":
        """Build a summary from an ``mp-api`` document or a mapping."""
        material_id = _document_value(document, "material_id")
        if material_id is None:
            raise ValueError("Materials Project document has no material_id")
        return cls(
            material_id=str(material_id),
            formula=_optional_str(_document_value(document, "formula_pretty")),
            chemsys=_optional_str(_document_value(document, "chemsys")),
            nsites=_optional_int(_document_value(document, "nsites")),
            volume=_optional_float(_document_value(document, "volume")),
            energy_above_hull=_optional_float(_document_value(document, "energy_above_hull")),
            band_gap=_optional_float(_document_value(document, "band_gap")),
            is_stable=_optional_bool(_document_value(document, "is_stable")),
            theoretical=_optional_bool(_document_value(document, "theoretical")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a JSON-compatible dictionary."""
        return {
            "material_id": self.material_id,
            "formula": self.formula,
            "chemsys": self.chemsys,
            "nsites": self.nsites,
            "volume": self.volume,
            "energy_above_hull": self.energy_above_hull,
            "band_gap": self.band_gap,
            "is_stable": self.is_stable,
            "theoretical": self.theoretical,
        }


@dataclass(frozen=True)
class MaterialStructure:
    """A downloaded structure together with its summary information."""

    summary: MaterialSummary
    structure: Any

    @property
    def material_id(self) -> str:
        """Return the Materials Project identifier of this structure."""
        return self.summary.material_id

    def to_abacus_structure(self):
        """Convert to an :class:`~abacustools.io.stru.AbacusSTRU`."""
        from abacustools.io.stru import AbacusSTRU

        return AbacusSTRU.from_pymatgen(
            self.structure,
            meta_data={
                "source": "materials-project",
                "material_id": self.material_id,
            },
        )


@contextmanager
def open_rester(
    *,
    api_key: Optional[str] = None,
    client: Any = None,
) -> Iterator[Any]:
    """Yield an ``MPRester`` for one batch of queries.

    Args:
        api_key: Explicit API key; otherwise read from the environment.
        client: An already constructed client, which is yielded unchanged.
            Tests and callers that manage their own connection use this to
            avoid touching the network.
    """
    if client is not None:
        yield client
        return
    rester_type = load_materials_project()
    with rester_type(api_key=resolve_api_key(api_key)) as rester:
        yield rester


def _summary_fields(fields: Optional[Sequence[str]]) -> list[str]:
    """Return the requested summary fields, always including material_id."""
    requested = list(DEFAULT_SUMMARY_FIELDS if fields is None else fields)
    if "material_id" not in requested:
        requested.insert(0, "material_id")
    return requested


def _search_criteria(
    *,
    formula: Optional[str],
    chemsys: Optional[str],
    elements: Optional[Sequence[str]],
    material_ids: Optional[Sequence[str]],
    is_stable: Optional[bool],
    theoretical: Optional[bool],
    fields: Sequence[str],
) -> dict[str, Any]:
    """Translate command-line selectors into ``mp-api`` search keywords."""
    criteria: dict[str, Any] = {"fields": list(fields)}
    if formula:
        criteria["formula"] = formula
    if chemsys:
        criteria["chemsys"] = chemsys
    if elements:
        criteria["elements"] = [str(element) for element in elements]
    if material_ids:
        criteria["material_ids"] = [str(identifier) for identifier in material_ids]
    if is_stable is not None:
        criteria["is_stable"] = bool(is_stable)
    if theoretical is not None:
        criteria["theoretical"] = bool(theoretical)
    if len(criteria) == 1:
        raise ValueError(
            "a search needs at least one selector: formula, chemsys, elements, or material_id"
        )
    return criteria


def search_materials(
    *,
    formula: Optional[str] = None,
    chemsys: Optional[str] = None,
    elements: Optional[Sequence[str]] = None,
    material_ids: Optional[Sequence[str]] = None,
    is_stable: Optional[bool] = None,
    theoretical: Optional[bool] = None,
    limit: int = 20,
    fields: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    client: Any = None,
) -> list[MaterialSummary]:
    """Search the Materials Project summary endpoint.

    Args:
        formula: Chemical formula such as ``Fe2O3``.
        chemsys: Chemical system such as ``Li-Fe-O``.
        elements: Elements that must all be present.
        material_ids: Restrict the search to specific material identifiers.
        is_stable: When given, keep only materials matching this stability.
        theoretical: When given, keep only materials matching this flag.
        limit: Maximum number of summaries to return.
        fields: Summary fields to request; defaults to
            :data:`DEFAULT_SUMMARY_FIELDS`.
        api_key: Explicit API key; otherwise read from the environment.
        client: Pre-constructed client, used to avoid the network in tests.

    Returns:
        The matching summaries, in the order returned by the database.
    """
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    criteria = _search_criteria(
        formula=formula,
        chemsys=chemsys,
        elements=elements,
        material_ids=material_ids,
        is_stable=is_stable,
        theoretical=theoretical,
        fields=_summary_fields(fields),
    )
    with open_rester(api_key=api_key, client=client) as rester:
        documents = list(rester.materials.summary.search(**criteria))
    return [MaterialSummary.from_document(document) for document in documents[:limit]]


def download_material(
    material_id: str,
    *,
    fields: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    client: Any = None,
) -> MaterialStructure:
    """Download one structure and its summary information.

    Args:
        material_id: Materials Project identifier such as ``mp-149``.
        fields: Summary fields to request; defaults to
            :data:`DEFAULT_SUMMARY_FIELDS`.
        api_key: Explicit API key; otherwise read from the environment.
        client: Pre-constructed client, used to avoid the network in tests.

    Returns:
        The downloaded structure and its summary.

    Raises:
        LookupError: If the database has no structure for ``material_id``.
    """
    identifier = str(material_id)
    with open_rester(api_key=api_key, client=client) as rester:
        structure = rester.get_structure_by_material_id(identifier)
        if structure is None:
            raise LookupError(f"no Materials Project structure for {identifier}")
        documents = list(
            rester.materials.summary.search(
                material_ids=[identifier],
                fields=_summary_fields(fields),
            )
        )
    summary = (
        MaterialSummary.from_document(documents[0])
        if documents
        else MaterialSummary(material_id=identifier)
    )
    return MaterialStructure(summary=summary, structure=structure)


def material_directory(
    output: Path,
    material_id: str,
    *,
    fmt: str = "stru",
) -> Path:
    """Return the file that receives one downloaded structure.

    Every material is written below ``output/<material_id>/`` so that a
    download of several structures never collides, and so that the directory
    can be used as an ABACUS job directory directly.

    Args:
        output: Parent directory of the per-material directories.
        material_id: Materials Project identifier such as ``mp-149``.
        fmt: Structure format; one of :data:`STRUCTURE_FILENAMES`.

    Returns:
        The destination path, which need not exist yet.
    """
    normalized = str(fmt).lower().lstrip(".")
    try:
        filename = STRUCTURE_FILENAMES[normalized]
    except KeyError:
        supported = ", ".join(sorted(STRUCTURE_FILENAMES))
        raise ValueError(f"unsupported structure format {fmt!r}; choose from {supported}") from None
    return Path(output) / str(material_id) / filename


def write_material_structure(
    material: MaterialStructure,
    destination: Path,
    *,
    fmt: Optional[str] = None,
) -> Path:
    """Write a downloaded structure to ``destination``.

    Args:
        material: Structure returned by :func:`download_material`.
        destination: Target path; its parent directory is created if needed.
        fmt: Output format; inferred from ``destination`` when omitted.

    Returns:
        The written path.

    Raises:
        IOError: If the structure cannot be written.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not material.to_abacus_structure().write(str(path), fmt=fmt):
        raise IOError(f"failed to write structure file: {path}")
    return path
