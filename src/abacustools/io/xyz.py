"""Writing structures as extended XYZ.

The vibration workflow writes the displaced geometries of every mode together
with their velocities, which ASE used to do through ``ase.io.write``.  This
module implements the subset of the extended XYZ format that is needed for
that output, so the mode animations do not depend on ASE.

The layout follows the conventions of ``ase.io.extxyz``: the second line of a
frame holds ``Lattice``, ``Properties``, arbitrary metadata and ``pbc``, and
atomic velocities are stored in the ``momenta`` column as mass-weighted
velocities.  Files written here can therefore be read back with
``ase.io.read`` and ``Atoms.get_velocities()``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


#: Formats of the symbol and the floating point columns, matching
#: ``ase.io.extxyz``.
_SYMBOL_FORMAT = "{:<2s}"
_FLOAT_FORMAT = "{:16.8f}"

#: Comment keys that need no quoting.
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.\-]+$")

#: Per-atom arrays accepted in a frame, with their output name and their number
#: of columns: one to three columns are both allowed for magnetic moments.
_PER_ATOM_COLUMNS: dict[str, tuple[str, tuple[int, ...]]] = {
    "magmoms": ("initial_magmoms", (1, 3)),
    "momenta": ("momenta", (3,)),
}


def _quote(value: str) -> str:
    """Return a quoted comment value with quotes and backslashes escaped."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _comment_key(key: Any) -> str:
    """Return a comment line key, quoting keys that need it."""
    name = str(key)
    return name if _SAFE_KEY.match(name) else _quote(name)


def _comment_value(value: Any) -> Optional[str]:
    """Encode one value of an extended XYZ comment line.

    Numbers and booleans are written as they are, following the ``T``/``F``
    convention of the format for logical values, strings are quoted, and other
    objects are stored as JSON behind the ``_JSON`` marker that
    ``ase.io.extxyz`` uses.  ``None`` encodes to an empty value.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "T" if value else "F"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        return _quote("_JSON " + json.dumps(value))
    except TypeError as error:
        raise ValueError(f"cannot encode {value!r} in an XYZ comment line") from error


def _frame_columns(frame: Mapping[str, Any]) -> list[tuple[str, str, np.ndarray]]:
    """Return the comment name, the column type and the values of every column.

    The returned arrays are two-dimensional, with one row per atom and one
    column per written value.
    """
    try:
        elements = np.asarray([str(symbol) for symbol in frame["elements"]])
        positions = np.asarray(frame["positions"], dtype=float)
    except KeyError as error:
        raise ValueError(f"a frame must define {error.args[0]!r}") from error
    if positions.shape != (elements.size, 3):
        raise ValueError(
            "frame positions must have shape (natoms, 3), "
            f"got {positions.shape} for {elements.size} symbols"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("frame positions must be finite")

    columns = [("species", "S", elements.reshape(-1, 1)), ("pos", "R", positions)]
    for key, (name, sizes) in _PER_ATOM_COLUMNS.items():
        values = frame.get(key)
        if values is None:
            continue
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape[0] != elements.size or array.shape[1] not in sizes:
            expected = " or ".join(str(size) for size in sizes)
            raise ValueError(
                f"frame {key} must have one row per atom and {expected} column(s), "
                f"got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"frame {key} must be finite")
        columns.append((name, "R", array))
    return columns


def _comment_line(
    columns: list[tuple[str, str, np.ndarray]],
    cell: Optional[np.ndarray],
    pbc: Optional[np.ndarray],
    info: Mapping[str, Any],
) -> str:
    """Build the comment line that describes one frame."""
    properties = ":".join(
        f"{name}:{kind}:{values.shape[1]}" for name, kind, values in columns
    )

    parts = []
    if cell is not None:
        lattice = " ".join(repr(float(value)) for value in np.asarray(cell).reshape(-1))
        parts.append(f'Lattice="{lattice}"')
    parts.append(f"Properties={properties}")
    for key, value in info.items():
        encoded = _comment_value(value)
        parts.append(_comment_key(key) if encoded is None else f"{_comment_key(key)}={encoded}")
    if pbc is not None:
        flags = " ".join("T" if flag else "F" for flag in pbc)
        parts.append(f'pbc="{flags}"')
    return " ".join(parts)


def _frame_lines(frame: Mapping[str, Any]) -> list[str]:
    """Return the comment line and the atom lines of one frame."""
    columns = _frame_columns(frame)

    cell = frame.get("cell")
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        if cell.shape != (3, 3):
            raise ValueError("frame cell must have shape (3, 3)")

    pbc = frame.get("pbc")
    if pbc is not None:
        pbc = np.asarray(pbc, dtype=bool)
        if pbc.shape != (3,):
            raise ValueError("frame pbc must contain three flags")

    info = frame.get("info") or {}
    if not isinstance(info, Mapping):
        raise ValueError("frame info must be a mapping")

    natoms = columns[0][2].shape[0]
    comment = _comment_line(columns, cell, pbc, info)
    lines = [f"{natoms}", comment]
    for index in range(natoms):
        row = []
        for _, kind, values in columns:
            row.extend(
                _SYMBOL_FORMAT.format(value) if kind == "S" else _FLOAT_FORMAT.format(value)
                for value in values[index]
            )
        lines.append(" ".join(row))
    return lines


def write_extxyz(
    path: Union[str, Path],
    frames: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]],
) -> int:
    """Write one or more frames as extended XYZ.

    Args:
        path: Output filename; missing parent directories are created.
        frames: One frame, or an iterable of frames.  A frame is a mapping
            with the required keys ``elements`` (element symbols) and
            ``positions`` (an ``(n, 3)`` array in Angstrom), the optional
            per-atom arrays ``momenta`` (the ``(n, 3)`` mass-weighted
            velocities of the frame) and ``magmoms`` (an ``(n,)`` collinear or
            ``(n, 3)`` vector magnetic moment array), and the optional frame
            keys ``cell`` (``(3, 3)`` lattice vectors in Angstrom), ``pbc``
            (three flags) and ``info`` (extra metadata of the comment line).

    Returns:
        int: Number of frames written.

    Raises:
        ValueError: If a frame is missing an array or an array has the wrong
            shape.
    """
    if isinstance(frames, Mapping):
        frame_list: Iterable[Mapping[str, Any]] = [frames]
    else:
        frame_list = frames

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for frame in frame_list:
            for line in _frame_lines(frame):
                handle.write(line + "\n")
            count += 1
    return count
