"""Build lattice-matched heterojunction interfaces from two structures.

The heavy lifting is pymatgen's coherent interface builder: the Zur-McGuire
lattice matching (ZSL) searches supercells of the two surfaces whose in-plane
vectors agree within a strain tolerance, the surfaces are cut along the
requested Miller indices, and the film is strained onto the substrate and
stacked on top of it with an interlayer gap and vacuum above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from abacustools.core.input_prep import resolve_library_resource
from abacustools.io.stru import AbacusSTRU


class InterfaceError(RuntimeError):
    """Raised when an interface cannot be built."""


@dataclass(frozen=True)
class InterfaceCandidate:
    """One lattice-matched combination of film and substrate supercells.

    Attributes:
        index: Position in the match list, best first.
        area: Supercell area in Angstrom squared.
        length_strain: Relative length mismatch of the two in-plane vectors.
        angle_mismatch: Angle mismatch of the two vectors in degrees.
        film_cells: Number of film surface cells in the supercell.
        substrate_cells: Number of substrate surface cells in the supercell.
    """

    index: int
    area: float
    length_strain: list[float]
    angle_mismatch: float
    film_cells: int
    substrate_cells: int


@dataclass(frozen=True)
class InterfaceResult:
    """A built interface structure and the choices that produced it.

    Attributes:
        structure: The interface, stacked along ``c`` with vacuum above the film.
        candidate: The lattice match that was used.
        matches: Number of matches the search found.
        termination: Termination pair of the film and substrate surfaces.
        termination_index: Index of that termination in the builder list.
        terminations: Number of terminations the surfaces offer.
        film_miller: Miller indices of the film surface.
        substrate_miller: Miller indices of the substrate surface.
        gap: Interlayer distance in Angstrom.
        vacuum: Vacuum above the film in Angstrom.
        resources: ``{element: (pp, orb, reason)}`` of the written structure.
    """

    structure: AbacusSTRU
    candidate: InterfaceCandidate
    matches: int
    termination: tuple[str, str]
    termination_index: int
    terminations: int
    film_miller: tuple[int, int, int]
    substrate_miller: tuple[int, int, int]
    gap: float
    vacuum: float
    resources: dict[str, tuple[Optional[str], Optional[str], str]]


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    """Validate a positive number."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InterfaceError(f"{name} must be a number, got {value!r}") from error
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise InterfaceError(f"{name} must be a {qualifier} number, got {value!r}")
    return number


def _miller(value: Sequence[int], name: str) -> tuple[int, int, int]:
    """Validate three Miller indices."""
    indices = list(value)
    if len(indices) != 3:
        raise InterfaceError(f"{name} needs three indices")
    try:
        numbers = tuple(int(index) for index in indices)
    except (TypeError, ValueError) as error:
        raise InterfaceError(f"{name} must be three integers") from error
    if numbers == (0, 0, 0):
        raise InterfaceError(f"{name} must not be all zero")
    return numbers


def _pymatgen(structure: AbacusSTRU):
    """Return the pymatgen structure of a structure."""
    return structure.to("pymatgen")


def _candidate(index: int, match: Any) -> InterfaceCandidate:
    """Summarise one ZSL match."""
    film = np.asarray(match.film_sl_vectors, dtype=float)
    substrate = np.asarray(match.substrate_sl_vectors, dtype=float)
    strain = []
    for film_vector, substrate_vector in zip(film, substrate):
        substrate_length = float(np.linalg.norm(substrate_vector))
        strain.append(float(np.linalg.norm(film_vector) / substrate_length - 1.0))

    def _angle(vectors) -> float:
        first, second = vectors[0], vectors[1]
        cosine = float(
            np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        )
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    return InterfaceCandidate(
        index=index,
        area=float(match.match_area),
        length_strain=strain,
        angle_mismatch=float(_angle(film) - _angle(substrate)),
        film_cells=abs(int(round(np.linalg.det(np.asarray(match.film_transformation, dtype=float))))),
        substrate_cells=abs(
            int(round(np.linalg.det(np.asarray(match.substrate_transformation, dtype=float))))
        ),
    )


def _builder(
    film: AbacusSTRU,
    substrate: AbacusSTRU,
    *,
    film_miller: Sequence[int],
    substrate_miller: Sequence[int],
    max_strain: float,
    max_angle: float,
    max_area: float,
):
    """Return the pymatgen coherent interface builder of two structures."""
    from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator

    generator = ZSLGenerator(
        max_area=max_area,
        max_length_tol=max_strain,
        max_angle_tol=math.radians(max_angle),
    )
    return CoherentInterfaceBuilder(
        _pymatgen(substrate),
        _pymatgen(film),
        _miller(film_miller, "film miller"),
        _miller(substrate_miller, "substrate miller"),
        generator,
    )


def interface_matches(
    film: AbacusSTRU,
    substrate: AbacusSTRU,
    *,
    film_miller: Sequence[int] = (0, 0, 1),
    substrate_miller: Sequence[int] = (0, 0, 1),
    max_strain: float = 0.03,
    max_angle: float = 0.6,
    max_area: float = 200.0,
) -> list[InterfaceCandidate]:
    """Return the lattice matches of two surfaces, best first.

    Args:
        film: Structure the film is cut from.
        substrate: Structure the substrate is cut from.
        film_miller: Miller indices of the film surface.
        substrate_miller: Miller indices of the substrate surface.
        max_strain: Largest relative length mismatch that is accepted.
        max_angle: Largest angle mismatch in degrees that is accepted.
        max_area: Largest supercell area in Angstrom squared.

    Returns:
        list: One candidate per match, ordered by pymatgen from best to worst.

    Raises:
        InterfaceError: When the tolerances are invalid.
    """
    _positive(max_strain, "max_strain")
    _positive(max_angle, "max_angle")
    _positive(max_area, "max_area")
    builder = _builder(
        film,
        substrate,
        film_miller=film_miller,
        substrate_miller=substrate_miller,
        max_strain=max_strain,
        max_angle=max_angle,
        max_area=max_area,
    )
    return [_candidate(index, match) for index, match in enumerate(builder.zsl_matches)]


def _element_resources(
    sources: Sequence[tuple[str, AbacusSTRU]], element: str
) -> tuple[Optional[str], Optional[str], str]:
    """Return the pseudopotential, orbital and reason of one element.

    The element is looked up in the structures the interface is built from, so
    a heterojunction inherits the files of its two parents. An element that
    neither structure describes falls back to the configured library.
    """
    for name, structure in sources:
        atoms = [atom for atom in structure.atoms if (atom.element or atom.label) == element]
        pseudo = next((atom.pp for atom in atoms if atom.pp), None)
        orbital = next((atom.orb for atom in atoms if atom.orb), None)
        if pseudo or orbital:
            return pseudo, orbital, f"reused from the {name} structure"
    pseudo_resource = resolve_library_resource(element, "pp")
    orbital_resource = resolve_library_resource(element, "orb")
    reason = f"taken from the configured resource library {pseudo_resource.library!r}"
    return pseudo_resource.filename, orbital_resource.filename, reason


def build_interface(
    film: AbacusSTRU,
    substrate: AbacusSTRU,
    *,
    film_miller: Sequence[int] = (0, 0, 1),
    substrate_miller: Sequence[int] = (0, 0, 1),
    film_thickness: float = 1.0,
    substrate_thickness: float = 1.0,
    in_layers: bool = True,
    gap: float = 2.0,
    vacuum: float = 15.0,
    termination: int = 0,
    max_strain: float = 0.03,
    max_angle: float = 0.6,
    max_area: float = 200.0,
    max_atoms: Optional[int] = None,
) -> InterfaceResult:
    """Build the best lattice-matched interface of two structures.

    Args:
        film: Structure the film is cut from.
        substrate: Structure the substrate is cut from.
        film_miller: Miller indices of the film surface.
        substrate_miller: Miller indices of the substrate surface.
        film_thickness: Film thickness, in layers unless ``in_layers`` is False.
        substrate_thickness: Substrate thickness, in layers unless ``in_layers``
            is False.
        in_layers: Interpret the thicknesses as layer counts instead of Angstrom.
        gap: Interlayer distance in Angstrom.
        vacuum: Vacuum above the film in Angstrom.
        termination: Index of the termination pair to use.
        max_strain: Largest relative length mismatch that is accepted.
        max_angle: Largest angle mismatch in degrees that is accepted.
        max_area: Largest supercell area in Angstrom squared.
        max_atoms: Skip matches whose interface has more atoms.

    Returns:
        InterfaceResult: The interface structure, its lattice match and the
        resource choice of every element.

    Raises:
        InterfaceError: When the tolerances are invalid, the surfaces offer no
            match, or no match satisfies ``max_atoms``.
    """
    gap = _positive(gap, "gap")
    vacuum = _positive(vacuum, "vacuum")
    film_thickness = _positive(film_thickness, "film thickness")
    substrate_thickness = _positive(substrate_thickness, "substrate thickness")
    max_strain = _positive(max_strain, "max_strain")
    max_angle = _positive(max_angle, "max_angle")
    max_area = _positive(max_area, "max_area")
    if max_atoms is not None:
        max_atoms = int(_positive(max_atoms, "max_atoms"))

    builder = _builder(
        film,
        substrate,
        film_miller=film_miller,
        substrate_miller=substrate_miller,
        max_strain=max_strain,
        max_angle=max_angle,
        max_area=max_area,
    )
    terminations = list(builder.terminations)
    if not terminations:
        raise InterfaceError(
            "the surfaces offer no termination; check the Miller indices and the "
            "symmetry of the two structures"
        )
    if termination < 0 or termination >= len(terminations):
        raise InterfaceError(
            f"termination {termination} is outside 0..{len(terminations) - 1}"
        )
    chosen = terminations[termination]

    matches = builder.zsl_matches
    if not matches:
        raise InterfaceError(
            "no lattice match was found; raise --max-strain, --max-angle or --max-area, "
            "or change the Miller indices"
        )

    candidate = None
    interface = None
    interfaces = builder.get_interfaces(
        chosen,
        gap=gap,
        vacuum_over_film=vacuum,
        film_thickness=film_thickness,
        substrate_thickness=substrate_thickness,
        in_layers=in_layers,
    )
    for index, generated in enumerate(interfaces):
        if index >= len(matches):
            break
        if max_atoms is not None and len(generated) > max_atoms:
            continue
        candidate = _candidate(index, matches[index])
        interface = generated
        break
    if interface is None or candidate is None:
        if max_atoms is None:
            raise InterfaceError("the interface builder returned no structure")
        raise InterfaceError(
            f"every lattice match of the {len(matches)} found needs more than "
            f"{max_atoms} atoms; raise --max-atoms or tighten --max-area"
        )

    structure = AbacusSTRU.from_pymatgen(interface)
    resources: dict[str, tuple[Optional[str], Optional[str], str]] = {}
    sources = (("film", film), ("substrate", substrate))
    for element in dict.fromkeys(structure.elements):
        pseudo, orbital, reason = _element_resources(sources, element)
        resources[element] = (pseudo, orbital, reason)
    for atom in structure.atoms:
        element = atom.element or atom.label
        atom.pp, atom.orb, _ = resources[element]

    return InterfaceResult(
        structure=structure,
        candidate=candidate,
        matches=len(matches),
        termination=chosen,
        termination_index=termination,
        terminations=len(terminations),
        film_miller=_miller(film_miller, "film miller"),
        substrate_miller=_miller(substrate_miller, "substrate miller"),
        gap=gap,
        vacuum=vacuum,
        resources=resources,
    )
