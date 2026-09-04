"""Unified Pseudopotential Format (UPF) file handling.

The parser supports XML-based UPF 2 files and the older sectioned UPF 1.x
format used by ABACUS and Quantum ESPRESSO. Numerical values are returned as
:class:`numpy.ndarray` objects so that the radial data can be used directly in
numerical work.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


PathLike = Union[str, Path]
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _tag_name(element: ET.Element) -> str:
    """Return an XML tag without an optional namespace."""
    return element.tag.rsplit("}", 1)[-1]


def _child(parent: Optional[ET.Element], tag: str) -> Optional[ET.Element]:
    """Find a direct child by tag, accepting namespace-qualified XML."""
    if parent is None:
        return None
    return next((item for item in parent if _tag_name(item) == tag), None)


def _as_float(value: str, field: str, filename: Path) -> float:
    """Parse a UPF floating-point attribute with a useful error message."""
    try:
        return float(value.strip().replace("D", "E").replace("d", "e"))
    except (AttributeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field!r} value {value!r} in UPF file {filename}"
        ) from error


def _as_int(value: str, field: str, filename: Path) -> int:
    """Parse an integer UPF attribute, accepting integral float notation."""
    number = _as_float(value, field, filename)
    if not number.is_integer():
        raise ValueError(
            f"Expected an integer {field!r} value in UPF file {filename}, "
            f"got {value!r}"
        )
    return int(number)


def _array_from_element(element: ET.Element, filename: Path) -> np.ndarray:
    """Parse the whitespace-separated numerical payload of a UPF element."""
    text = element.text or ""
    tokens = text.split()
    try:
        values = np.fromiter(
            (float(token.replace("D", "E").replace("d", "e")) for token in tokens),
            dtype=float,
            count=len(tokens),
        )
    except ValueError as error:
        raise ValueError(
            f"Invalid numerical data in <{_tag_name(element)}> of UPF file {filename}"
        ) from error

    size = element.attrib.get("size")
    if size is not None:
        expected_size = _as_int(size, f"{_tag_name(element)} size", filename)
        if values.size != expected_size:
            raise ValueError(
                f"<{_tag_name(element)}> in UPF file {filename} declares size "
                f"{expected_size}, but contains {values.size} values"
            )
    return values


def _attribute(
    element: ET.Element, name: str, filename: Path, *, required: bool = True
) -> Optional[str]:
    """Get a stripped XML attribute and report missing required attributes."""
    value = element.attrib.get(name)
    if value is None:
        if required:
            raise ValueError(
                f"Missing required {name!r} attribute in <{_tag_name(element)}> "
                f"of UPF file {filename}"
            )
        return None
    return value.strip()


def _numbered_tag_index(element: ET.Element, filename: Path) -> int:
    """Read an UPF ``index`` attribute, falling back to ``PP_*.<number>``.

    A few otherwise valid ONCV UPF files use ``index=\"*\"`` for their final
    ``PP_BETA`` tag. The suffix is unambiguous and is used as a compatibility
    fallback for those files.
    """
    value = _attribute(element, "index", filename)
    try:
        return _as_int(value, f"{_tag_name(element)} index", filename)
    except ValueError as error:
        if value != "*":
            raise error
        match = re.search(r"\.(\d+)$", _tag_name(element))
        if match is None:
            raise error
        return int(match.group(1))


def _parse_section(element: ET.Element, filename: Path) -> Dict[str, Any]:
    """Recursively preserve a UPF XML section and parse its numerical payload.

    ``sections`` uses this representation for every child of the ``UPF`` root.
    A list is used for ``children`` to retain the XML order and to support
    repeated tags. Numerical leaf nodes marked as ``type=\"real\"`` are exposed
    as NumPy arrays; other leaf text remains available unchanged as ``text``.
    """
    section: Dict[str, Any] = {
        "tag": _tag_name(element),
        "attributes": dict(element.attrib),
    }
    children = list(element)
    text = (element.text or "").strip()
    if children:
        if text:
            section["text"] = text
        section["children"] = [_parse_section(child, filename) for child in children]
    elif element.attrib.get("type", "").strip().lower() == "real":
        section["data"] = _array_from_element(element, filename)
    elif text:
        section["text"] = text
    return section


def _legacy_section(text: str, tag: str, occurrence: int = 0) -> Optional[str]:
    """Return one section body from a sectioned UPF 1.x file."""
    pattern = re.compile(
        rf"<{re.escape(tag)}(?:\s[^>]*)?>(.*?)</{re.escape(tag)}\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(text)
    if occurrence >= len(matches):
        return None
    return matches[occurrence]


def _legacy_numbers(line: str) -> List[float]:
    """Parse numbers from one legacy UPF text line."""
    return [float(value.replace("D", "E").replace("d", "e")) for value in re.findall(_FLOAT, line)]


def _legacy_array(text: str, filename: Path, section: str) -> np.ndarray:
    """Parse a numerical body from a legacy UPF section."""
    tokens = text.split()
    try:
        return np.fromiter(
            (float(token.replace("D", "E").replace("d", "e")) for token in tokens),
            dtype=float,
            count=len(tokens),
        )
    except ValueError as error:
        raise ValueError(
            f"Invalid numerical data in <{section}> of UPF file {filename}"
        ) from error


def _legacy_header_value(
    lines: List[str], marker: str, filename: Path, count: int = 1
) -> List[float]:
    """Read numeric values from the legacy header line containing a marker."""
    for line in lines:
        if marker.lower() in line.lower():
            values = _legacy_numbers(line)
            if len(values) >= count:
                return values[:count]
    raise ValueError(f"Could not find {marker!r} in UPF file {filename}")


def _legacy_reference_configurations(info: str, filename: Path) -> Dict[tuple, Dict[str, Any]]:
    """Read orbital reference metadata from the legacy ``PP_INFO`` section."""
    references: Dict[tuple, Dict[str, Any]] = {}
    for line in info.splitlines():
        fields = line.split()
        if not fields or re.fullmatch(r"\d+[A-Za-z]", fields[0]) is None:
            continue
        if len(fields) < 7:
            continue
        try:
            angular_momentum = int(fields[2])
            occupation = _as_float(fields[3], "wavefunction occupation", filename)
            cutoff_radius = _as_float(fields[5], "wavefunction cutoff_radius", filename)
            pseudo_energy = _as_float(fields[6], "wavefunction pseudo_energy", filename)
        except (IndexError, ValueError):
            continue
        references.setdefault(
            (fields[0], angular_momentum),
            {
                "occupation": occupation,
                "cutoff_radius": cutoff_radius,
                "pseudo_energy": pseudo_energy,
            },
        )
    return references


def _read_legacy_upf(text: str, filename: Path) -> "UPF":
    """Read a sectioned, non-XML UPF 1.x file."""
    header_text = _legacy_section(text, "PP_HEADER")
    if header_text is None:
        raise ValueError(f"Could not find <PP_HEADER> in UPF file {filename}")
    header_lines = [line for line in header_text.splitlines() if line.strip()]

    element_match = next(
        (
            match
            for match in (
                re.match(r"^\s*([A-Za-z]{1,3})\s+Element", line, re.IGNORECASE)
                for line in header_lines
            )
            if match is not None
        ),
        None,
    )
    if element_match is None:
        raise ValueError(f"Could not find element in UPF file {filename}")
    element = element_match.group(1)
    version_number = header_lines[0].split()[0]
    valence = _legacy_header_value(header_lines, "Z valence", filename)[0]
    lmax = int(_legacy_header_value(header_lines, "Max angular momentum", filename)[0])
    mesh_size = int(_legacy_header_value(header_lines, "Number of points in mesh", filename)[0])
    number_of_wfc, number_of_proj = [
        int(value)
        for value in _legacy_header_value(
            header_lines, "Number of Wavefunctions, Number of Projectors", filename, 2
        )
    ]

    mesh_text = _legacy_section(text, "PP_MESH")
    if mesh_text is None:
        raise ValueError(f"Could not find <PP_MESH> in UPF file {filename}")
    r_text = _legacy_section(mesh_text, "PP_R")
    rab_text = _legacy_section(mesh_text, "PP_RAB")
    if r_text is None or rab_text is None:
        raise ValueError(f"Could not find <PP_R> and <PP_RAB> in UPF file {filename}")
    r = _legacy_array(r_text, filename, "PP_R")
    rab = _legacy_array(rab_text, filename, "PP_RAB")
    if r.size != mesh_size:
        raise ValueError(
            f"PP_HEADER in UPF file {filename} declares mesh size {mesh_size}, "
            f"but <PP_R> contains {r.size} values"
        )

    local_text = _legacy_section(text, "PP_LOCAL")
    rhoatom_text = _legacy_section(text, "PP_RHOATOM")
    if local_text is None:
        raise ValueError(f"Could not find <PP_LOCAL> in UPF file {filename}")
    if rhoatom_text is None:
        raise ValueError(f"Could not find <PP_RHOATOM> in UPF file {filename}")
    local_potential = _legacy_array(local_text, filename, "PP_LOCAL")
    rhoatom = _legacy_array(rhoatom_text, filename, "PP_RHOATOM")
    nlcc_text = _legacy_section(text, "PP_NLCC")
    nlcc = _legacy_array(nlcc_text, filename, "PP_NLCC") if nlcc_text is not None else None

    info_text = _legacy_section(text, "PP_INFO") or ""
    references = _legacy_reference_configurations(info_text, filename)
    cutoff_by_l = {
        angular_momentum: max(
            reference["cutoff_radius"]
            for (label, angular_momentum), reference in references.items()
        )
        for angular_momentum in {angular_momentum for _, angular_momentum in references}
    }

    nonlocal_text = _legacy_section(text, "PP_NONLOCAL")
    beta_bodies = []
    if nonlocal_text is not None:
        index = 0
        while True:
            beta_text = _legacy_section(nonlocal_text, "PP_BETA", index)
            if beta_text is None:
                break
            beta_bodies.append(beta_text)
            index += 1

    projectors: List[Dict[str, Any]] = []
    for beta_text in beta_bodies:
        lines = [line.strip() for line in beta_text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError(f"Invalid <PP_BETA> section in UPF file {filename}")
        descriptor = lines[0].split()
        if len(descriptor) < 2:
            raise ValueError(f"Invalid <PP_BETA> header in UPF file {filename}")
        beta_index = _as_int(descriptor[0], "projector index", filename)
        angular_momentum = _as_int(descriptor[1], "projector angular_momentum", filename)
        declared_size = int(_legacy_numbers(lines[1])[0])
        data = _legacy_array("\n".join(lines[2:]), filename, "PP_BETA")
        if data.size != declared_size:
            raise ValueError(
                f"<PP_BETA> in UPF file {filename} declares size {declared_size}, "
                f"but contains {data.size} values"
            )
        projectors.append(
            {
                "index": beta_index,
                "l": angular_momentum,
                "cutoff_radius": cutoff_by_l.get(angular_momentum, 0.0),
                "cutoff_radius_index": None,
                "data": data,
            }
        )

    if len(projectors) != number_of_proj:
        raise ValueError(
            f"UPF file {filename} declares {number_of_proj} projectors, "
            f"but contains {len(projectors)} <PP_BETA> sections"
        )

    dij = np.array([])
    dij_text = _legacy_section(nonlocal_text or "", "PP_DIJ")
    if dij_text is not None:
        dij_lines = [line.strip() for line in dij_text.splitlines() if line.strip()]
        if not dij_lines:
            raise ValueError(f"Empty <PP_DIJ> section in UPF file {filename}")
        declared_entries = int(_legacy_numbers(dij_lines[0])[0])
        entries = [_legacy_numbers(line) for line in dij_lines[1:]]
        if len(entries) != declared_entries or any(len(entry) < 3 for entry in entries):
            raise ValueError(f"Invalid <PP_DIJ> section in UPF file {filename}")
        dij_matrix = np.zeros((number_of_proj, number_of_proj), dtype=float)
        for entry in entries:
            i, j = int(entry[0]), int(entry[1])
            if not (1 <= i <= number_of_proj and 1 <= j <= number_of_proj):
                raise ValueError(f"Invalid projector index in <PP_DIJ> of UPF file {filename}")
            dij_matrix[i - 1, j - 1] = entry[2]
            dij_matrix[j - 1, i - 1] = entry[2]
        dij = dij_matrix.ravel()

    pseudo_wavefunctions: List[Dict[str, Any]] = []
    pswfc_text = _legacy_section(text, "PP_PSWFC")
    if pswfc_text is not None:
        chi_sections = []
        index = 0
        while True:
            chi_text = _legacy_section(pswfc_text, "PP_CHI", index)
            if chi_text is None:
                break
            chi_sections.append((index + 1, chi_text))
            index += 1

        if chi_sections:
            wavefunction_blocks = chi_sections
        else:
            pswfc_lines = pswfc_text.splitlines()
            starts = [
                line_index
                for line_index, line in enumerate(pswfc_lines)
                if re.search(r"\bWavefunction\b", line, re.IGNORECASE)
            ]
            wavefunction_blocks = [
                (block_index + 1, "\n".join(pswfc_lines[start:end]))
                for block_index, (start, end) in enumerate(
                    zip(starts, starts[1:] + [len(pswfc_lines)])
                )
            ]

        for index, wavefunction_text in wavefunction_blocks:
            lines = [
                line.strip() for line in wavefunction_text.splitlines() if line.strip()
            ]
            if len(lines) < 2:
                raise ValueError(
                    f"Invalid wavefunction section in <PP_PSWFC> of UPF file {filename}"
                )
            descriptor = lines[0].split()
            if len(descriptor) < 3:
                raise ValueError(f"Invalid wavefunction header in UPF file {filename}")
            label = descriptor[0]
            angular_momentum = _as_int(descriptor[1], "wavefunction l", filename)
            occupation = _as_float(descriptor[2], "wavefunction occupation", filename)
            data = _legacy_array("\n".join(lines[1:]), filename, "PP_PSWFC")
            reference = references.get((label, angular_momentum), {})
            pseudo_wavefunctions.append(
                {
                    "index": index,
                    "l": angular_momentum,
                    "label": label,
                    "occupation": occupation,
                    "pseudo_energy": reference.get("pseudo_energy"),
                    "data": data,
                }
            )

    if len(pseudo_wavefunctions) > number_of_wfc:
        raise ValueError(
            f"UPF file {filename} declares {number_of_wfc} wavefunctions, "
            f"but contains {len(pseudo_wavefunctions)} wavefunction sections"
        )

    top_level_tags = [
        "PP_INFO",
        "PP_HEADER",
        "PP_MESH",
        "PP_NLCC",
        "PP_LOCAL",
        "PP_NONLOCAL",
        "PP_PSWFC",
        "PP_RHOATOM",
    ]
    sections = [
        {"tag": tag, "attributes": {}, "text": _legacy_section(text, tag).strip()}
        for tag in top_level_tags
        if _legacy_section(text, tag) is not None
    ]
    header = {
        "version_number": version_number,
        "element": element,
        "z_valence": str(valence),
        "l_max": str(lmax),
        "mesh_size": str(mesh_size),
        "number_of_wfc": str(number_of_wfc),
        "number_of_proj": str(number_of_proj),
    }
    return UPF(
        element=element,
        valence=valence,
        cutoff_radius=max(
            (projector["cutoff_radius"] for projector in projectors), default=0.0
        ),
        lmax=lmax,
        version=version_number,
        header=header,
        r=r,
        rab=rab,
        local_potential=local_potential,
        projectors=projectors,
        dij=dij,
        rhoatom=rhoatom,
        nlcc=nlcc,
        pseudo_wavefunctions=pseudo_wavefunctions,
        sections=sections,
        info=info_text.strip(),
        filename=filename,
    )


class UPF:
    """Representation of an UPF 1.x or UPF 2 pseudopotential.

    Attributes:
        element: Chemical symbol.
        valence: Valence charge (``z_valence`` in ``PP_HEADER``).
        cutoff_radius: Largest non-local projector cutoff in Bohr, or ``0.0``
            when the file has no projectors.
        lmax: Maximum angular momentum from the UPF header.
        mesh: Number of radial grid points.
        r: Radial grid in Bohr.
        rab: Radial-grid integration increments.
        local_potential: Local potential on ``r``.
        projectors: Non-local projectors, including their metadata and data.
        dij: Flattened non-local coupling matrix from ``PP_DIJ``.
        rhoatom: Pseudo-atomic charge density.
        nlcc: Non-linear core correction, if present.
        pseudo_wavefunctions: Pseudo-atomic wavefunctions from ``PP_PSWFC``.
        input_file: Generator input retained from ``PP_INPUTFILE``, if present.
        header: Original ``PP_HEADER`` attributes, retained as strings.
        spin_orbit: Relativistic metadata from ``PP_SPIN_ORB``, if present.
        sections: Complete XML section hierarchy, including sections without a
            dedicated convenience attribute.
    """

    def __init__(
        self,
        element: str,
        valence: float,
        cutoff_radius: float,
        lmax: int,
        *,
        version: Optional[str] = None,
        header: Optional[Dict[str, str]] = None,
        r: Optional[np.ndarray] = None,
        rab: Optional[np.ndarray] = None,
        local_potential: Optional[np.ndarray] = None,
        projectors: Optional[List[Dict[str, Any]]] = None,
        dij: Optional[np.ndarray] = None,
        rhoatom: Optional[np.ndarray] = None,
        nlcc: Optional[np.ndarray] = None,
        pseudo_wavefunctions: Optional[List[Dict[str, Any]]] = None,
        spin_orbit: Optional[List[Dict[str, Any]]] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        info: str = "",
        input_file: str = "",
        filename: Optional[PathLike] = None,
    ) -> None:
        self.element = element
        self.valence = valence
        self.z_valence = valence
        self.cutoff_radius = cutoff_radius
        self.lmax = lmax
        self.version = version
        self.header = dict(header) if header is not None else {}
        self.r = np.asarray(r, dtype=float) if r is not None else np.array([])
        self.rab = np.asarray(rab, dtype=float) if rab is not None else np.array([])
        self.local_potential = (
            np.asarray(local_potential, dtype=float)
            if local_potential is not None
            else np.array([])
        )
        self.projectors = list(projectors) if projectors is not None else []
        self.dij = np.asarray(dij, dtype=float) if dij is not None else np.array([])
        self.rhoatom = (
            np.asarray(rhoatom, dtype=float) if rhoatom is not None else np.array([])
        )
        self.nlcc = np.asarray(nlcc, dtype=float) if nlcc is not None else None
        self.pseudo_wavefunctions = (
            list(pseudo_wavefunctions) if pseudo_wavefunctions is not None else []
        )
        self.spin_orbit = list(spin_orbit) if spin_orbit is not None else []
        self.sections = list(sections) if sections is not None else []
        self.info = info
        self.input_file = input_file
        self.filename = Path(filename) if filename is not None else None

        self.mesh = self.r.size
        self._validate_meshes()

    def _validate_meshes(self) -> None:
        """Ensure radial sections use the mesh declared by the radial grid."""
        if self.mesh == 0:
            return
        radial_data = {
            name: data
            for name, data in {
                "rab": self.rab,
                "local_potential": self.local_potential,
                "rhoatom": self.rhoatom,
            }.items()
            if data.size
        }
        if self.nlcc is not None:
            radial_data["nlcc"] = self.nlcc
        inconsistent = [name for name, data in radial_data.items() if data.size != self.mesh]
        if inconsistent:
            raise ValueError(
                f"Inconsistent radial mesh size for {', '.join(inconsistent)}; "
                f"expected {self.mesh} values"
            )

    @staticmethod
    def read_from_file(upf_file: PathLike) -> "UPF":
        """Read a UPF 1.x or UPF 2 file.

        Args:
            upf_file: Path to the UPF file.

        Returns:
            A populated :class:`UPF` instance.

        Raises:
            ValueError: If the file is not a valid, complete UPF file.
        """
        filename = Path(upf_file)
        try:
            root = ET.parse(filename).getroot()
        except ET.ParseError as error:
            try:
                text = filename.read_text(encoding="utf-8", errors="replace")
            except OSError as file_error:
                raise ValueError(
                    f"Could not parse UPF file {filename}: {file_error}"
                ) from file_error
            if (
                re.search(r"<PP_HEADER(?:\s|>)", text, re.IGNORECASE)
                and not re.search(r"<UPF(?:\s|>)", text, re.IGNORECASE)
            ):
                return _read_legacy_upf(text, filename)
            raise ValueError(f"Could not parse UPF file {filename}: {error}") from error
        except OSError as error:
            raise ValueError(f"Could not parse UPF file {filename}: {error}") from error

        if _tag_name(root) != "UPF":
            raise ValueError(f"Root element of {filename} is not <UPF>")

        header_element = _child(root, "PP_HEADER")
        if header_element is None:
            raise ValueError(f"Could not find <PP_HEADER> in UPF file {filename}")

        header = dict(header_element.attrib)
        element = _attribute(header_element, "element", filename)
        z_valence = _as_float(
            _attribute(header_element, "z_valence", filename), "z_valence", filename
        )
        lmax = _as_int(_attribute(header_element, "l_max", filename), "l_max", filename)
        declared_mesh = _as_int(
            _attribute(header_element, "mesh_size", filename), "mesh_size", filename
        )

        mesh = _child(root, "PP_MESH")
        r_element = _child(mesh, "PP_R")
        rab_element = _child(mesh, "PP_RAB")
        if r_element is None or rab_element is None:
            raise ValueError(f"Could not find <PP_R> and <PP_RAB> in UPF file {filename}")
        r = _array_from_element(r_element, filename)
        rab = _array_from_element(rab_element, filename)
        if r.size != declared_mesh:
            raise ValueError(
                f"PP_HEADER in UPF file {filename} declares mesh_size {declared_mesh}, "
                f"but <PP_R> contains {r.size} values"
            )

        local_element = _child(root, "PP_LOCAL")
        if local_element is None:
            raise ValueError(f"Could not find <PP_LOCAL> in UPF file {filename}")
        local_potential = _array_from_element(local_element, filename)

        projectors: List[Dict[str, Any]] = []
        nonlocal_element = _child(root, "PP_NONLOCAL")
        if nonlocal_element is not None:
            for child in nonlocal_element:
                if not _tag_name(child).startswith("PP_BETA"):
                    continue
                index = _numbered_tag_index(child, filename)
                angular_momentum = _as_int(
                    _attribute(child, "angular_momentum", filename),
                    "projector angular_momentum",
                    filename,
                )
                cutoff_radius = _as_float(
                    _attribute(child, "cutoff_radius", filename),
                    "projector cutoff_radius",
                    filename,
                )
                cutoff_radius_index = _attribute(
                    child, "cutoff_radius_index", filename, required=False
                )
                projectors.append(
                    {
                        "index": index,
                        "l": angular_momentum,
                        "cutoff_radius": cutoff_radius,
                        "cutoff_radius_index": (
                            _as_int(cutoff_radius_index, "projector cutoff_radius_index", filename)
                            if cutoff_radius_index is not None
                            else None
                        ),
                        "data": _array_from_element(child, filename),
                    }
                )

        dij_element = _child(nonlocal_element, "PP_DIJ")
        dij = (
            _array_from_element(dij_element, filename)
            if dij_element is not None
            else np.array([])
        )
        expected_dij_size = len(projectors) ** 2
        if dij.size not in (0, expected_dij_size):
            raise ValueError(
                f"<PP_DIJ> in UPF file {filename} contains {dij.size} values, "
                f"but {len(projectors)} projectors require {expected_dij_size}"
            )

        pseudo_wavefunctions: List[Dict[str, Any]] = []
        pswfc_element = _child(root, "PP_PSWFC")
        if pswfc_element is not None:
            for child in pswfc_element:
                if not _tag_name(child).startswith("PP_CHI"):
                    continue
                pseudo_wavefunctions.append(
                    {
                        "index": _numbered_tag_index(child, filename),
                        "l": _as_int(
                            _attribute(child, "l", filename), "wavefunction l", filename
                        ),
                        "label": _attribute(child, "label", filename),
                        "occupation": _as_float(
                            _attribute(child, "occupation", filename),
                            "wavefunction occupation",
                            filename,
                        ),
                        "pseudo_energy": _as_float(
                            _attribute(child, "pseudo_energy", filename),
                            "wavefunction pseudo_energy",
                            filename,
                        ),
                        "data": _array_from_element(child, filename),
                    }
                )

        rhoatom_element = _child(root, "PP_RHOATOM")
        if rhoatom_element is None:
            raise ValueError(f"Could not find <PP_RHOATOM> in UPF file {filename}")
        rhoatom = _array_from_element(rhoatom_element, filename)
        nlcc_element = _child(root, "PP_NLCC")
        nlcc = _array_from_element(nlcc_element, filename) if nlcc_element is not None else None

        spin_orbit: List[Dict[str, Any]] = []
        spin_orbit_element = _child(root, "PP_SPIN_ORB")
        if spin_orbit_element is not None:
            for child in spin_orbit_element:
                attributes: Dict[str, Any] = {"tag": _tag_name(child)}
                for name, value in child.attrib.items():
                    try:
                        attributes[name] = _as_float(value, name, filename)
                    except ValueError:
                        attributes[name] = value.strip()
                spin_orbit.append(attributes)

        info_element = _child(root, "PP_INFO")
        input_file_element = _child(info_element, "PP_INPUTFILE")
        cutoff_radius = max(
            (projector["cutoff_radius"] for projector in projectors), default=0.0
        )
        return UPF(
            element=element,
            valence=z_valence,
            cutoff_radius=cutoff_radius,
            lmax=lmax,
            version=root.attrib.get("version"),
            header=header,
            r=r,
            rab=rab,
            local_potential=local_potential,
            projectors=projectors,
            dij=dij,
            rhoatom=rhoatom,
            nlcc=nlcc,
            pseudo_wavefunctions=pseudo_wavefunctions,
            spin_orbit=spin_orbit,
            sections=[_parse_section(child, filename) for child in root],
            info=(info_element.text or "").strip() if info_element is not None else "",
            input_file=(input_file_element.text or "").strip()
            if input_file_element is not None
            else "",
            filename=filename,
        )

    @property
    def dij_matrix(self) -> np.ndarray:
        """Return ``PP_DIJ`` as a square matrix, or an empty ``(0, 0)`` array."""
        if self.dij.size == 0:
            return np.empty((0, 0))
        return self.dij.reshape((len(self.projectors), len(self.projectors)))

    def __repr__(self) -> str:
        return (
            f"UPF(element={self.element!r}, valence={self.valence}, "
            f"lmax={self.lmax}, mesh={self.mesh}, "
            f"projectors={len(self.projectors)}, "
            f"pseudo_wavefunctions={len(self.pseudo_wavefunctions)})"
        )
