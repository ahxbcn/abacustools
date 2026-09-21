"""File layout of downloaded structures.

Downloads are written below one directory per entry so that a batch never
collides and so that each directory can be used as an ABACUS job directory.
The file names are shared with the Materials Project adapter through
:data:`~abacustools.integrations.materials_project.STRUCTURE_FILENAMES`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..materials_project import STRUCTURE_FILENAMES as STRUCTURE_FILENAMES
from .base import DatabaseStructure


def structure_filename(fmt: str = "stru") -> str:
    """Return the file name used for one structure format.

    Args:
        fmt: Structure format; one of the keys of :data:`STRUCTURE_FILENAMES`.

    Returns:
        The file name, such as ``STRU`` or ``POSCAR``.

    Raises:
        ValueError: If the format is not supported.
    """
    normalized = str(fmt).lower().lstrip(".")
    try:
        return STRUCTURE_FILENAMES[normalized]
    except KeyError:
        supported = ", ".join(sorted(STRUCTURE_FILENAMES))
        raise ValueError(f"unsupported structure format {fmt!r}; choose from {supported}") from None


def structure_path(
    output: Path,
    identifier: str,
    *,
    fmt: str = "stru",
    database: Optional[str] = None,
) -> Path:
    """Return the file that receives one downloaded structure.

    Args:
        output: Parent directory of the per-entry directories.
        identifier: Database entry identifier.
        fmt: Structure format; one of the keys of :data:`STRUCTURE_FILENAMES`.
        database: When given, nest the entry directory below the database
            name, which keeps identifiers of different databases apart.

    Returns:
        The destination path, which need not exist yet.
    """
    root = Path(output)
    if database:
        root = root / str(database)
    return root / str(identifier) / structure_filename(fmt)


def write_structure(
    structure: DatabaseStructure,
    destination: Path,
    *,
    fmt: Optional[str] = None,
) -> Path:
    """Write a downloaded structure to ``destination``.

    Args:
        structure: Structure returned by a database's ``fetch``.
        destination: Target path; its parent directory is created if needed.
        fmt: Output format; inferred from ``destination`` when omitted.

    Returns:
        The written path.

    Raises:
        IOError: If the structure cannot be written.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not structure.to_abacus_structure().write(str(path), fmt=fmt):
        raise IOError(f"failed to write structure file: {path}")
    return path
