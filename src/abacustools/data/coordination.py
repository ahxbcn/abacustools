"""Coordination numbers and coordination environments of a structure.

The dimensionality of the structure decides which analysis runs by default.
ChemEnv describes the coordination polyhedra of a bulk crystal, but its Voronoi
tessellation is undefined in the vacuum of a slab, a wire or a molecule, where
the nearest-neighbour algorithms are used instead.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from abacustools.data.dimensionality import classify_dimensionality
from abacustools.io.stru import AbacusSTRU


METHODS = ("auto", "crystalnn", "chemenv", "voronoi", "minimum-distance")
DEFAULT_METHODS = {
    "bulk": "chemenv",
    "slab": "crystalnn",
    "wire": "crystalnn",
    "molecule": "crystalnn",
}


def _site(index: int, atom, **values: Any) -> dict[str, Any]:
    """Build one per-site entry with the keys every method fills in."""
    entry = {
        "index": index + 1,
        "label": atom.label,
        "element": atom.element or atom.label,
        "coordination_number": None,
        "neighbors": None,
        "geometry": None,
        "csm": None,
        "fraction": None,
    }
    entry.update(values)
    return entry


def _number(value: Any) -> Any:
    """Return an integral coordination number as an int."""
    number = float(value)
    return int(number) if number.is_integer() else number


def _neighbors(structure: AbacusSTRU, method: str) -> list[dict[str, Any]]:
    """Analyse the structure with a nearest-neighbour algorithm."""
    from pymatgen.analysis.local_env import CrystalNN, MinimumDistanceNN, VoronoiNN

    strategies = {
        "crystalnn": CrystalNN,
        "voronoi": VoronoiNN,
        "minimum-distance": MinimumDistanceNN,
    }
    strategy = strategies[method]()
    pymatgen_structure = structure.to("pymatgen")
    sites = []
    for index, atom in enumerate(structure.atoms):
        neighbors = strategy.get_cn_dict(pymatgen_structure, index)
        sites.append(
            _site(
                index,
                atom,
                coordination_number=_number(strategy.get_cn(pymatgen_structure, index)),
                neighbors={str(element): int(count) for element, count in neighbors.items()},
            )
        )
    return sites


def _environment_number(symbol: Optional[str]) -> Optional[int]:
    """Return the coordination number encoded in a ChemEnv symbol."""
    if not symbol or ":" not in str(symbol):
        return None
    try:
        return int(str(symbol).rsplit(":", 1)[1])
    except ValueError:
        return None


def _environments(
    structure: AbacusSTRU,
    maximum_distance_factor: float,
) -> list[dict[str, Any]]:
    """Analyse the structure with ChemEnv."""
    from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
        SimplestChemenvStrategy,
    )
    from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
        LocalGeometryFinder,
    )
    from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
        LightStructureEnvironments,
    )

    finder = LocalGeometryFinder()
    finder.setup_structure(structure.to("pymatgen"))
    structure_environments = finder.compute_structure_environments(
        maximum_distance_factor=maximum_distance_factor,
    )
    light = LightStructureEnvironments.from_structure_environments(
        strategy=SimplestChemenvStrategy(),
        structure_environments=structure_environments,
    )
    sites = []
    for index, atom in enumerate(structure.atoms):
        environments = light.coordination_environments[index]
        if not environments:
            sites.append(_site(index, atom))
            continue
        best = environments[0]
        symbol = best.get("ce_symbol")
        sites.append(
            _site(
                index,
                atom,
                coordination_number=_environment_number(symbol),
                geometry=symbol,
                csm=None if best.get("csm") is None else float(best["csm"]),
                fraction=None if best.get("ce_fraction") is None else float(best["ce_fraction"]),
            )
        )
    return sites


def _analyse(
    structure: AbacusSTRU,
    method: str,
    maximum_distance_factor: float,
) -> list[dict[str, Any]]:
    """Run one coordination analysis on the structure."""
    if method == "chemenv":
        return _environments(structure, maximum_distance_factor)
    return _neighbors(structure, method)


def _run(
    structure: AbacusSTRU,
    method: str,
    maximum_distance_factor: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run an analysis and turn the warnings it raises into notes."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        sites = _analyse(structure, method, maximum_distance_factor)
    messages = {str(item.message) for item in captured}
    notes = []
    if method == "crystalnn" and any("No oxidation states" in text for text in messages):
        notes.append(
            "crystalnn falls back to covalent or atomic radii because the structure "
            "has no oxidation states"
        )
    return sites, notes


def coordination_analysis(
    structure: AbacusSTRU,
    *,
    method: str = "auto",
    min_vacuum: float = 5.0,
    maximum_distance_factor: float = 1.4,
) -> dict[str, Any]:
    """Return the coordination number and environment of every atom.

    Args:
        structure: Structure to analyse.
        method: ``auto``, ``crystalnn``, ``chemenv``, ``voronoi`` or
            ``minimum-distance``.  ``auto`` follows the dimensionality of the
            structure: ChemEnv for a bulk, CrystalNN for a slab, a wire or a
            molecule.
        min_vacuum: Empty span in Angstrom that counts as vacuum.
        maximum_distance_factor: ChemEnv search range around the nearest
            neighbour distance.

    Returns:
        dict: The method that was used, the dimensionality it was chosen from,
        any note about the choice, and one entry per atom with its coordination
        number, its neighbour species, and for ChemEnv the coordination
        geometry symbol and its continuous symmetry measure.
    """
    if method not in METHODS:
        raise ValueError(f"unknown coordination method: {method}")
    dimension = classify_dimensionality(structure, min_vacuum=min_vacuum)
    if not dimension["periodic"]:
        return {
            "available": False,
            "error": "a non-zero three-dimensional periodic cell is required",
            "requested_method": method,
            "method": None,
            "dimensionality": dimension,
            "notes": [],
            "sites": [],
        }
    chosen = DEFAULT_METHODS[dimension["dimensionality"]] if method == "auto" else method
    result = {
        "available": True,
        "error": None,
        "requested_method": method,
        "method": chosen,
        "dimensionality": dimension,
        "notes": [],
        "sites": [],
    }
    if method == "auto" and chosen != "chemenv":
        result["notes"].append(
            f"{dimension['label']}: chemenv needs a three-dimensional Voronoi "
            f"tessellation, so {chosen} is used instead"
        )
    try:
        result["sites"], notes = _run(structure, chosen, maximum_distance_factor)
        result["notes"] += notes
        return result
    except Exception as error:
        if method != "auto":
            result["available"] = False
            result["error"] = f"{chosen} analysis failed: {error}"
            return result
        result["notes"].append(f"chemenv failed ({error}); crystalnn is used instead")
    result["method"] = "crystalnn"
    try:
        result["sites"], notes = _run(structure, "crystalnn", maximum_distance_factor)
        result["notes"] += notes
    except Exception as error:
        result["available"] = False
        result["error"] = f"crystalnn analysis failed: {error}"
    return result


__all__ = [
    "DEFAULT_METHODS",
    "METHODS",
    "coordination_analysis",
]
