"""K-point grids and band paths built on top of ABACUS KPT files.

The readers and writers of the KPT format itself live in
:mod:`abacustools.io.abacus`; this module turns them into the operations the
commands need: describing a KPT file, deriving a mesh from a target k-spacing
or from another job, and generating the seekpath band path of a structure in
line mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from abacustools.core.constant import ANG_TO_BOHR, BOHR_TO_ANG
from abacustools.data.dimensionality import DIMENSIONALITY_LABELS, classify_dimensionality
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


def mesh_from_job(job: str | Path, inputs: dict[str, Any], structure) -> Tuple[list[float], str]:
    """Return the regular k-point mesh an ABACUS job uses.

    Args:
        job: Job directory holding the ``INPUT`` and any explicit ``KPT`` file.
        inputs: Parsed ``INPUT`` of the job.
        structure: Structure the mesh belongs to.

    Returns:
        The six mesh values and the KPT model, ``"gamma"`` or ``"mp"``.

    Raises:
        ValueError: When neither an input mesh nor a valid KPT file is found.
    """
    job = Path(job)
    try:
        if float(inputs.get("gamma_only", 0)) > 0:
            return [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], "gamma"
    except (TypeError, ValueError):
        pass

    kspacing = inputs.get("kspacing")
    if kspacing not in (None, 0, "0", "0.0"):
        cell_bohr = np.asarray(structure.cell, dtype=float) * ANG_TO_BOHR
        mesh = kspacing2kpt(kspacing, cell_bohr)
        return [float(value) for value in mesh] + [0.0, 0.0, 0.0], "gamma"

    parsed = ReadKpt(str(job))
    if parsed is None:
        raise ValueError(f"could not read a KPT file below {job}")
    kpt_data, model = parsed
    values = [float(value) for value in list(kpt_data)[:6]]
    if model not in MESH_MODELS or len(values) != 6:
        raise ValueError(f"the KPT file below {job} does not define a regular mesh")
    return values, model


#: High-symmetry paths of the 2D Bravais lattices, in fractional coordinates
#: of the two periodic lattice vectors. The hexagonal and square lattices use
#: the conventional paths; the remaining lattices use the loop through the two
#: reciprocal-direction midpoints ``X``/``Y`` and their sum, which covers the
#: irreducible wedge without ever stepping into the vacuum direction.
_TWO_DIMENSIONAL_PATHS: dict[str, list[tuple[str, tuple[float, float]]]] = {
    "hexagonal": [("G", (0.0, 0.0)), ("M", (0.5, 0.0)), ("K", (1.0 / 3.0, 1.0 / 3.0)), ("G", (0.0, 0.0))],
    "square": [("G", (0.0, 0.0)), ("X", (0.5, 0.0)), ("M", (0.5, 0.5)), ("G", (0.0, 0.0))],
    "rectangular": [
        ("G", (0.0, 0.0)),
        ("X", (0.5, 0.0)),
        ("S", (0.5, 0.5)),
        ("Y", (0.0, 0.5)),
        ("G", (0.0, 0.0)),
    ],
    "oblique": [
        ("G", (0.0, 0.0)),
        ("X", (0.5, 0.0)),
        ("M", (0.5, 0.5)),
        ("Y", (0.0, 0.5)),
        ("G", (0.0, 0.0)),
    ],
}

#: Label of the zone boundary along each lattice direction.
_AXIS_BOUNDARY_LABELS = {"a": "X", "b": "Y", "c": "Z"}

_DIRECTION_INDEX = {"a": 0, "b": 1, "c": 2}

_PATH_MODES = ("auto", "bulk", "slab", "wire")


@dataclass(frozen=True)
class BandPath:
    """A band path chosen for the dimensionality of a structure.

    Attributes:
        nodes: Line-mode KPT nodes, one ``[x, y, z, count, label]`` group each.
        segments: High-symmetry segments the nodes follow.
        dimensionality: ``bulk``, ``slab`` or ``wire``.
        label: Human-readable dimensionality such as ``2D slab``.
        method: How the path was chosen, such as ``seekpath`` or ``2D rectangular``.
        periodic_directions: Lattice directions the path samples.
    """

    nodes: list[list[Any]]
    segments: list[list[str]]
    dimensionality: str
    label: str
    method: str
    periodic_directions: list[str]

    @property
    def labels(self) -> list[str]:
        """Return the high-symmetry labels of the nodes."""
        return [str(node[4]) for node in self.nodes]

    @property
    def points(self) -> int:
        """Return the number of k-points the path samples."""
        return int(sum(int(node[3]) for node in self.nodes))


def _three_dimensional_path(
    structure,
    *,
    npoints: int,
    with_time_reversal: bool,
    recipe: str,
    symprec: float,
    angle_tolerance: float,
) -> tuple[list[list[Any]], list[list[str]]]:
    """Return the seekpath path of a bulk structure."""
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


def _two_dimensional_lattice(
    length_a: float,
    length_b: float,
    gamma: float,
    *,
    symprec: float,
    angle_tolerance: float,
) -> str:
    """Classify the in-plane lattice of a slab."""
    tolerance = angle_tolerance if angle_tolerance and angle_tolerance > 0 else 1.0
    equal_lengths = abs(length_a - length_b) <= symprec
    if equal_lengths and min(abs(gamma - 120.0), abs(gamma - 60.0)) <= tolerance:
        return "hexagonal"
    if equal_lengths and abs(gamma - 90.0) <= tolerance:
        return "square"
    if abs(gamma - 90.0) <= tolerance:
        return "rectangular"
    return "oblique"


def _two_dimensional_path(
    structure,
    periodic_directions: list[str],
    *,
    npoints: int,
    symprec: float,
    angle_tolerance: float,
) -> tuple[list[list[Any]], list[list[str]], str]:
    """Return the in-plane path of a slab with the vacuum direction at k=0."""
    axes = [_DIRECTION_INDEX[direction] for direction in periodic_directions]
    cell = np.asarray(structure.cell, dtype=float)
    vector_a, vector_b = cell[axes[0]], cell[axes[1]]
    length_a = float(np.linalg.norm(vector_a))
    length_b = float(np.linalg.norm(vector_b))
    cosine = float(np.dot(vector_a, vector_b) / (length_a * length_b))
    gamma = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    lattice = _two_dimensional_lattice(
        length_a, length_b, gamma, symprec=symprec, angle_tolerance=angle_tolerance
    )

    nodes: list[list[Any]] = []
    for label, (first, second) in _TWO_DIMENSIONAL_PATHS[lattice]:
        fractional = [0.0, 0.0, 0.0]
        fractional[axes[0]] = first
        fractional[axes[1]] = second
        nodes.append([*fractional, npoints, label])
    nodes[-1][3] = 1
    segments = [
        [nodes[index][4], nodes[index + 1][4]] for index in range(len(nodes) - 1)
    ]
    return nodes, segments, f"2D {lattice}"


def _one_dimensional_path(
    periodic_directions: list[str], npoints: int
) -> tuple[list[list[Any]], list[list[str]], str]:
    """Return the path along the periodic axis of a wire."""
    axis = _DIRECTION_INDEX[periodic_directions[0]]
    boundary = [0.0, 0.0, 0.0]
    boundary[axis] = 0.5
    label = _AXIS_BOUNDARY_LABELS[periodic_directions[0]]
    nodes = [[0.0, 0.0, 0.0, npoints, "G"], [*boundary, 1, label]]
    return nodes, [["G", label]], f"1D along {periodic_directions[0]}"


def band_path(
    structure,
    *,
    npoints: int = 20,
    min_vacuum: float = 5.0,
    path_mode: str = "auto",
    with_time_reversal: bool = True,
    recipe: str = "hpkot",
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
) -> BandPath:
    """Choose the band path that fits the dimensionality of a structure.

    The vacuum analysis of :mod:`abacustools.data.dimensionality` decides how
    many lattice directions are still periodic: a bulk keeps the seekpath path,
    a slab samples only the two periodic directions with the vacuum direction
    pinned to ``k = 0``, and a wire samples the single periodic direction. A
    zero-dimensional structure has no path and is rejected, because its bands
    are flat and a Gamma-point calculation describes it completely.

    Args:
        structure: Structure the path is built for.
        npoints: Number of points sampled in every segment.
        min_vacuum: Empty span in Angstrom that counts as vacuum.
        path_mode: ``auto`` follows the detected dimensionality; ``bulk``,
            ``slab`` or ``wire`` forces one.
        with_time_reversal: Passed to seekpath for a bulk.
        recipe: seekpath recipe, ``hpkot`` by default.
        symprec: Symmetry tolerance in Angstrom, which also decides whether two
            in-plane lattice vectors count as equally long.
        angle_tolerance: Angle tolerance in degrees; a negative value lets
            seekpath estimate it and a default of 1 degree classify a slab.

    Returns:
        BandPath: The nodes, the segments and the dimensionality they follow.

    Raises:
        ValueError: When the number of points or the mode is invalid, when a
            forced mode does not match the structure, or when the structure is
            zero-dimensional.
    """
    if not isinstance(npoints, int) or isinstance(npoints, bool) or npoints < 1:
        raise ValueError("npoints must be a positive integer")
    if path_mode not in _PATH_MODES:
        raise ValueError(f"unknown path mode: {path_mode!r}; use one of {list(_PATH_MODES)}")

    dimension = classify_dimensionality(structure, min_vacuum=min_vacuum)
    periodic = list(dimension["periodic_directions"])
    name = dimension["dimensionality"] if path_mode == "auto" else path_mode

    if name == "bulk":
        nodes, segments = _three_dimensional_path(
            structure,
            npoints=npoints,
            with_time_reversal=with_time_reversal,
            recipe=recipe,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        method = "seekpath"
    elif name == "slab":
        if len(periodic) != 2:
            raise ValueError(
                "path mode slab needs exactly two periodic directions, "
                f"but the structure has {len(periodic)}"
            )
        nodes, segments, method = _two_dimensional_path(
            structure,
            periodic,
            npoints=npoints,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
    elif name == "wire":
        if len(periodic) != 1:
            raise ValueError(
                "path mode wire needs exactly one periodic direction, "
                f"but the structure has {len(periodic)}"
            )
        nodes, segments, method = _one_dimensional_path(periodic, npoints)
    else:
        raise ValueError(
            f"a {DIMENSIONALITY_LABELS[name]} has no band path; its bands are flat, "
            "so calculate the Gamma point or a DOS mesh instead"
        )

    return BandPath(
        nodes=nodes,
        segments=segments,
        dimensionality=name,
        label=DIMENSIONALITY_LABELS[name],
        method=method,
        periodic_directions=periodic,
    )


def band_path_nodes(
    structure,
    **kwargs,
) -> Tuple[list[list[Any]], list[list[str]]]:
    """Return the nodes and segments of :func:`band_path`.

    Args:
        structure: Structure the path is built for.
        **kwargs: Forwarded to :func:`band_path`.

    Returns:
        The KPT node list and the high-symmetry segments.
    """
    path = band_path(structure, **kwargs)
    return path.nodes, path.segments
