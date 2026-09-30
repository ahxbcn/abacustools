"""Substitute atoms of a structure and resolve the resources of the dopant.

A substitution replaces one or more atoms by another element. The
pseudopotential and orbital of the new element are taken from the input
structure when it already contains that element, because those files are then
known to belong to the same calculation. An element that is not present in the
structure gets its files from the configured resource library, exactly as
``job prepare`` reads it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from abacustools.core.input_prep import resolve_library_resource
from abacustools.io.stru import MASS_DICT, AbacusSTRU


class SubstitutionError(RuntimeError):
    """Raised when a substitution cannot be applied."""


#: Where a resolved resource came from.
STRUCTURE_SOURCE = "structure"
LIBRARY_SOURCE = "library"
COMMAND_SOURCE = "command line"
UNUSED_SOURCE = "unused"


@dataclass(frozen=True)
class DopantResources:
    """The pseudopotential and orbital chosen for a dopant element.

    Attributes:
        element: Element symbol that is substituted in.
        pp: Pseudopotential file name, or ``None``.
        orb: Orbital file name, or ``None``.
        pp_source: ``structure``, ``library``, ``command line`` or ``unused``.
        orb_source: ``structure``, ``library``, ``command line`` or ``unused``.
        library: Configured library the files came from, when they did.
        variant: Orbital variant that was honoured, when one applies.
        reason: One sentence explaining both choices.
    """

    element: str
    pp: Optional[str]
    orb: Optional[str]
    pp_source: str
    orb_source: str
    pp_reason: str
    orb_reason: str
    library: Optional[str] = None
    variant: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        """Return the choice as a JSON-compatible mapping."""
        return {
            "dopant": self.element,
            "pseudopotential": self.pp,
            "orbital": self.orb,
            "pseudopotential_source": self.pp_source,
            "orbital_source": self.orb_source,
            "pseudopotential_reason": self.pp_reason,
            "orbital_reason": self.orb_reason,
            "library": self.library,
            "variant": self.variant,
        }


def _element_resources(
    structure: AbacusSTRU, element: str
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Return the pseudopotential, orbital and labels an element already uses."""
    atoms = [atom for atom in structure.atoms if (atom.element or atom.label) == element]
    labels = sorted({atom.label for atom in atoms})
    for attribute in ("pp", "orb"):
        values = sorted({value for atom in atoms if (value := getattr(atom, attribute))})
        if len(values) > 1:
            raise SubstitutionError(
                f"{element} appears with several {attribute} files in the input "
                f"structure ({', '.join(values)}); pass --{attribute} to choose one"
            )
    pseudo = next((atom.pp for atom in atoms if atom.pp), None)
    orbital = next((atom.orb for atom in atoms if atom.orb), None)
    return pseudo, orbital, labels


def _choose(
    symbol: str,
    kind: str,
    explicit: Optional[str],
    present: Optional[str],
    *,
    orbital: bool,
    library: Optional[str],
    variant: Optional[str],
    pp_path,
    orb_path,
) -> tuple[Optional[str], str, Optional[str], Optional[str]]:
    """Choose one resource and report where it comes from."""
    if kind == "orb" and not orbital:
        return None, UNUSED_SOURCE, None, None
    if explicit is not None:
        return str(explicit), COMMAND_SOURCE, None, None
    if present is not None:
        return present, STRUCTURE_SOURCE, None, None
    resolved = resolve_library_resource(
        symbol,
        kind,
        library=library,
        variant=variant,
        pp_path=pp_path,
        orb_path=orb_path,
    )
    return resolved.filename, LIBRARY_SOURCE, resolved.library, resolved.variant


def _reason(
    value: Optional[str],
    source: str,
    *,
    symbol: str,
    present: bool,
    library: Optional[str],
    variant: Optional[str],
) -> str:
    """Explain why one resource was chosen."""
    if source == UNUSED_SOURCE:
        return "no orbital is used for a plane-wave basis"
    presence = (
        f"{symbol} is already present in the input structure"
        if present
        else f"{symbol} is not present in the input structure"
    )
    if source == STRUCTURE_SOURCE:
        return f"{presence}, so {value} is reused from it"
    if source == COMMAND_SOURCE:
        return f"{value} was given on the command line"
    text = f"{presence}, so {value} is taken from the configured resource library {library!r}"
    if variant:
        text += f" with the orbital variant {variant}"
    return text


def resolve_dopant_resources(
    structure: AbacusSTRU,
    element: str,
    *,
    pp: Optional[str] = None,
    orb: Optional[str] = None,
    library: Optional[str] = None,
    variant: Optional[str] = None,
    orbital: bool = True,
    pp_path=None,
    orb_path=None,
) -> DopantResources:
    """Return the pseudopotential and orbital to use for a dopant element.

    An element that the structure already contains keeps the files of that
    element, which keeps a doped cell consistent with the host. Any other
    element is resolved from the configured resource library. Explicit
    ``pp``/``orb`` arguments override both.

    Args:
        structure: Structure that is going to be doped.
        element: Element symbol that is substituted in.
        pp: Pseudopotential file name to use instead of the resolved one.
        orb: Orbital file name to use instead of the resolved one.
        library: Configured resource library; the configured default when omitted.
        variant: Orbital variant such as ``DZP``.
        orbital: Resolve an orbital at all; ``False`` for a plane-wave structure.
        pp_path: Explicit pseudopotential directory for the library lookup.
        orb_path: Explicit orbital directory for the library lookup.

    Returns:
        DopantResources: The two files, their provenance and the reason.

    Raises:
        SubstitutionError: When the structure uses several files for the element.
    """
    symbol = str(element).strip().capitalize()
    if not symbol:
        raise SubstitutionError("a dopant element is required")
    present = symbol in set(structure.elements)
    present_pp, present_orb, _ = _element_resources(structure, symbol)

    pseudo, pp_source, pp_library, pp_variant = _choose(
        symbol, "pp", pp, present_pp,
        orbital=orbital, library=library, variant=variant,
        pp_path=pp_path, orb_path=orb_path,
    )
    orbital_file, orb_source, orb_library, orb_variant = _choose(
        symbol, "orb", orb, present_orb,
        orbital=orbital, library=library, variant=variant,
        pp_path=pp_path, orb_path=orb_path,
    )
    return DopantResources(
        element=symbol,
        pp=pseudo,
        orb=orbital_file,
        pp_source=pp_source,
        orb_source=orb_source,
        pp_reason=_reason(
            pseudo, pp_source,
            symbol=symbol, present=present, library=pp_library, variant=pp_variant,
        ),
        orb_reason=_reason(
            orbital_file, orb_source,
            symbol=symbol, present=present, library=orb_library, variant=orb_variant,
        ),
        library=pp_library or orb_library,
        variant=orb_variant or pp_variant,
    )


def substitute_atoms(
    structure: AbacusSTRU,
    *,
    element: str,
    indices: Iterable[int],
    resources: DopantResources,
    label: Optional[str] = None,
    keep_moments: bool = False,
) -> AbacusSTRU:
    """Return a copy with the selected atoms replaced by the dopant element.

    The cell, the atom order and every attribute that does not belong to the
    replaced element are kept. The substituted atoms take the mass, the
    pseudopotential and the orbital of the dopant; their PAW file is dropped and
    their magnetic moment is cleared unless ``keep_moments`` is set, because a
    moment of the old element rarely fits the new one.

    Args:
        structure: Structure to dope.
        element: Element symbol that is substituted in.
        indices: Zero-based indices of the atoms to replace.
        resources: Files chosen by :func:`resolve_dopant_resources`.
        label: Label of the substituted atoms; the element symbol by default.
        keep_moments: Keep the magnetic moment and its angles of the atoms.

    Returns:
        AbacusSTRU: A new structure with the substituted atoms.

    Raises:
        SubstitutionError: When no atom is selected, an index is out of range,
            or the label already belongs to another element.
    """
    symbol = str(element).strip().capitalize()
    selected = sorted({int(index) for index in indices})
    if not selected:
        raise SubstitutionError("no atoms were selected for substitution")
    for index in selected:
        if index < 0 or index >= structure.natoms:
            raise SubstitutionError(
                f"atom index {index + 1} is outside 1..{structure.natoms}"
            )
    if resources.element != symbol:
        raise SubstitutionError(
            f"the resources were resolved for {resources.element}, not {symbol}"
        )

    new_label = symbol if label is None else str(label)
    other_labels = {
        atom.label for atom in structure.atoms if (atom.element or atom.label) != symbol
    }
    if new_label in other_labels:
        raise SubstitutionError(
            f"label {new_label!r} already belongs to another element; "
            "pass a new label or the label of the dopant element"
        )

    edited = copy.deepcopy(structure)
    for index in selected:
        atom = edited.atoms[index]
        atom.label = new_label
        atom.element = symbol
        atom.mass = MASS_DICT.get(symbol, atom.mass)
        atom.pp = resources.pp
        atom.orb = resources.orb
        atom.paw = None
        if not keep_moments:
            atom.mag = None
            atom.angle1 = None
            atom.angle2 = None
            atom.type_mag = 0.0
    return edited
