"""K-point grids and band paths built on top of ABACUS KPT files.

The readers and writers of the KPT format itself live in
:mod:`abacustools.io.abacus`; this module turns them into the operations the
commands need: describing a KPT file, deriving a mesh from a target k-spacing
or from another job, and generating the seekpath band path of a structure in
line mode.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.io.abacus import (
    FormatKpt,
    ReadKpt,
    kspacing2kpt,
)


MESH_MODELS = ("gamma", "mp")
"""KPT models that describe a regular mesh instead of explicit k-points."""


def read_kpt(path: str | Path) -> Tuple[Any, str]:
    """Read a KPT file and return its values and model.

    Args:
        path: KPT file to read.

    Returns:
        The mesh/node values and the canonical model name.

    Raises:
        ValueError: When the file cannot be read as a KPT file.
    """
    parsed = ReadKpt(str(path))
    if parsed is None:
        raise ValueError(f"could not read a KPT file: {path}")
    return parsed


def _kpt_label(value: Any) -> Optional[str]:
    """Return a high-symmetry label without its comment marker."""
    text = str(value).strip()
    for marker in ("//", "#"):
        if text.startswith(marker):
            text = text[len(marker):].strip()
            break
    return text or None


def model_report(values: Any, model: str) -> dict[str, Any]:
    """Describe the mesh or k-point list of one KPT file.

    Args:
        values: KPT values as :func:`read_kpt` returns them.
        model: Canonical KPT model name.

    Returns:
        dict: The model, the mesh with its shifts or the explicit points, the
        number of generated points, and, when the writer rejects the values,
        ``valid=False`` with the reason.
    """
    report: dict[str, Any] = {"model": model}
    try:
        FormatKpt(values, model)
    except ValueError as error:
        report["valid"] = False
        report["error"] = str(error)
    else:
        report["valid"] = True

    if model in MESH_MODELS:
        mesh = [int(value) for value in list(values)[:3]]
        report["mesh"] = mesh
        report["shifts"] = [float(value) for value in list(values)[3:6]]
        report["mesh_points"] = int(np.prod(mesh))
    else:
        nodes = [list(group) for group in values]
        report["nodes"] = len(nodes)
        report["points"] = [
            [float(value) for value in node[:3]] for node in nodes
        ]
        report["point_counts"] = [
            int(node[3]) for node in nodes if len(node) > 3
        ]
        report["labels"] = [
            None if len(node) < 5 else _kpt_label(node[4]) for node in nodes
        ]
        if model in ("direct", "cartesian"):
            report["weights"] = [float(node[3]) for node in nodes if len(node) > 3]
    return report


def _spacing_vector(spacing: float | Sequence[float], name: str = "spacing") -> list[float]:
    """Return one spacing or three spacings as a validated list."""
    if isinstance(spacing, (int, float)) and not isinstance(spacing, bool):
        values = [float(spacing)]
    else:
        try:
            values = [float(value) for value in spacing]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} values must be numbers") from error
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"{name} needs one value or three values")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive finite numbers")
    return values


def _reciprocal_lengths(structure) -> list[float]:
    """Return ``2*pi*|b_i|`` of a structure in 1/Bohr."""
    cell = np.asarray(structure.cell, dtype=float) * ANG_TO_BOHR
    volume = abs(float(np.linalg.det(cell)))
    if volume < 1e-12:
        raise ValueError("the structure needs a periodic cell with a volume")
    return [
        float(np.linalg.norm(np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])) * 2 * np.pi / volume)
        for axis in range(3)
    ]


def mesh_from_spacing(structure, spacing: float | Sequence[float]) -> list[int]:
    """Return the mesh that realizes a target k-spacing.

    The mesh follows the ABACUS rule ``nk_i = max(1, int(|b_i|/k_i) + 1)``,
    computed with the reciprocal vectors in 1/Bohr.

    Args:
        structure: Structure the mesh is built for.
        spacing: Target k-spacing in 1/Angstrom, one value or three.

    Returns:
        list: The three mesh subdivisions.

    Raises:
        ValueError: When the spacing is invalid or the cell has no volume.
    """
    values = _spacing_vector(spacing)
    cell_bohr = np.asarray(structure.cell, dtype=float) * ANG_TO_BOHR
    bohr_spacing = [value * BOHR_TO_ANG for value in values]
    return [int(value) for value in kspacing2kpt(bohr_spacing, cell_bohr)]


def spacing_from_mesh(structure, mesh: Sequence[int]) -> list[Optional[float]]:
    """Return the k-spacing in 1/Angstrom that a mesh realizes.

    Args:
        structure: Structure the mesh belongs to.
        mesh: The three mesh subdivisions.

    Returns:
        list: The k-spacing in 1/Angstrom that reproduces this mesh, or
        ``None`` along a direction that has no subdivision.
    """
    subdivisions = [int(value) for value in mesh]
    if len(subdivisions) != 3 or any(value <= 0 for value in subdivisions):
        raise ValueError("the mesh needs three positive subdivisions")
    lengths = _reciprocal_lengths(structure)
    # The mesh that ABACUS derives from a spacing sits in the interval
    # ``|b|/n < k <= |b|/(n-1)``; the middle of that window is reported so
    # that feeding the value back reproduces the same mesh.
    return [
        None
        if count <= 1
        else float(length * (1.0 / count + 1.0 / (count - 1)) / 2.0 / BOHR_TO_ANG)
        for count, length in zip(subdivisions, lengths)
    ]


def band_path_nodes(
    structure,
    *,
    npoints: int = 20,
    with_time_reversal: bool = True,
    recipe: str = "hpkot",
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
) -> Tuple[list[list[Any]], list[list[str]]]:
    """Return the seekpath band path of a structure as line-mode KPT nodes.

    Args:
        structure: Structure the path is built for.
        npoints: Number of points sampled in every segment.
        with_time_reversal: Passed to seekpath.
        recipe: seekpath recipe, ``hpkot`` by default.
        symprec: Symmetry tolerance in Angstrom.
        angle_tolerance: Angle tolerance in degrees; negative lets seekpath
            estimate it.

    Returns:
        The KPT node list, one ``[x, y, z, count, label]`` group per node, and
        the high-symmetry segments the nodes come from.

    Raises:
        ValueError: When the number of points is invalid or seekpath cannot
            find a path for the structure.
    """
    if not isinstance(npoints, int) or isinstance(npoints, bool) or npoints < 1:
        raise ValueError("npoints must be a positive integer")

    point_coords, segments = structure.get_kline(
        with_time_reversal=with_time_reversal,
        recipe=recipe,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if not segments:
        raise ValueError("seekpath did not return a band path for this structure")

    chain: list[str] = [segments[0][0]]
    for _, end in segments:
        chain.append(end)
    nodes: list[list[Any]] = []
    for label in chain:
        if nodes and nodes[-1][4] == label:
            continue
        nodes.append([*point_coords[label], npoints, label])
    if len(nodes) < 2:
        raise ValueError("the seekpath band path has fewer than two nodes")
    nodes[-1][3] = 1
    return nodes, [[start, end] for start, end in segments]
