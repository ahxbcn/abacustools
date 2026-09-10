"""Prepare complete, self-contained ABACUS input directories."""

from __future__ import annotations

import json
import os
import re
import shutil
import warnings
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from abacustools.core.config import CONFIG
from abacustools.io.abacus import (
    FormatKpt,
    IsEnabled,
    NormalizeKptModel,
    ReadInput,
    WriteInput,
    WriteKpt,
)
from abacustools.io.pseudo import UPF
from abacustools.io.stru import MASS_DICT, AbacusSTRU


PathLike = Union[str, Path]

_MAGNETIC_D_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "La", "Ac",
    "Th",
}
_MAGNETIC_F_ELEMENTS = {
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
}
_ORBITAL_INDEX = {"p": 1, "d": 2, "f": 3}
_ORBITAL_CUTOFF = re.compile(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*Ry", re.IGNORECASE)


class InputPreparationError(RuntimeError):
    """Raised when a complete ABACUS input directory cannot be prepared."""


@dataclass(frozen=True)
class PreparedJob:
    """Description of one generated ABACUS input directory."""

    path: Path
    source: Path


def _unique(values: Iterable[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def parse_input_value(value: str) -> Any:
    """Parse a command-line parameter value like ``ReadInput`` does."""
    values = value.split()
    if len(values) > 1:
        parsed = [parse_input_value(item) for item in values]
        if all(isinstance(item, (int, float)) for item in parsed):
            return parsed
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _normalize_kpt(kpt: Sequence[Any], model: str) -> list:
    """Group a flat or nested KPT argument for the requested model.

    The gamma/mp models take a single mesh group, while the explicit and line
    models take one group per k-point or node.  A flat list is treated as one
    group so that ``kpt=[2, 2, 2]`` keeps working for gamma/mp.
    """
    name = NormalizeKptModel(model)
    values = list(kpt)
    if not values:
        raise ValueError("kpt must not be empty")
    nested = all(isinstance(value, (list, tuple)) for value in values)
    if name in ("gamma", "mp"):
        if nested:
            if len(values) != 1:
                raise ValueError("gamma/mp kpt accepts a single mesh group")
            values = list(values[0])
        return values
    return [list(value) for value in values] if nested else [values]


def _element_from_filename(filename: str) -> Optional[str]:
    match = re.match(r"^([A-Z][a-z]?)(?:[^A-Za-z]|$)", Path(filename).name)
    if match is None:
        return None
    element = match.group(1)
    return element if element in MASS_DICT else None


def _orbital_cutoff(filename: str) -> Optional[float]:
    """Return the plane-wave cutoff a numerical orbital was generated with.

    Orbital file names encode it, such as ``Si_gga_8au_100Ry_2s2p1d.orb``.

    Args:
        filename: Orbital file name or path.

    Returns:
        The cutoff energy in Ry, or ``None`` when the name does not carry one.
    """
    match = _ORBITAL_CUTOFF.search(Path(filename).name)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _spin_orbit_support(path: Path) -> Optional[bool]:
    """Return whether a pseudopotential supports spin-orbit calculations.

    Fully relativistic (spinor) pseudopotentials either declare
    ``relativistic="full"`` in ``PP_HEADER`` or carry tabulated spin-orbit
    projectors (``has_so``).  Files that cannot be read as UPF return ``None``.

    Args:
        path: Pseudopotential file.

    Returns:
        ``True`` when spin-orbit support is declared, ``False`` when the file
        declares another flavour, and ``None`` when it cannot be interpreted.
    """
    try:
        header = UPF.read_from_file(path).header
    except (OSError, ValueError):
        return None
    attributes = {
        str(key).lower(): str(value).strip().lower() for key, value in header.items()
    }
    if attributes.get("relativistic") == "full":
        return True
    return attributes.get("has_so") in {"t", "true", "1"}


_RESOURCE_DIRECTORY_PREFIXES = {
    "pp": ("pseudopotential",),
    "orb": ("orbital",),
    "paw": ("paw",),
}


def _resolve_library_path(path: Path, resource_type: Optional[str]) -> Path:
    """Return a resource path, recovering from a reorganisation of the library.

    Upstream libraries are occasionally reorganised, so a configured
    ``.../Orbitals`` can become ``.../Orbitals_v2.0``.  When the configured path
    is missing, a uniquely matching sibling directory of the same resource type
    is used instead of failing.
    """
    if path.exists():
        return path
    prefixes = _RESOURCE_DIRECTORY_PREFIXES.get(resource_type or "", ())
    candidates = []
    if prefixes and path.parent.is_dir():
        candidates = sorted(
            child
            for child in path.parent.iterdir()
            if child.is_dir()
            and any(child.name.lower().startswith(prefix) for prefix in prefixes)
        )
    if len(candidates) == 1:
        warnings.warn(
            f"resource path {path} does not exist; using {candidates[0]} instead",
            stacklevel=4,
        )
        return candidates[0]
    message = f"resource path does not exist: {path}"
    if candidates:
        message += "; candidate directories: " + ", ".join(str(item) for item in candidates)
    raise InputPreparationError(message)


def _standard_rcut_index(root: Path, variant: Optional[str]) -> dict[str, float]:
    """Read the standard orbital cutoffs published next to an orbital library.

    Upstream libraries ship ``<orbital directory>_<VARIANT>_..._StandardRcut.json``
    files that name the recommended cutoff radius of every element, with an
    ``Others`` fallback.  The index is only read when a variant is requested,
    because the file name identifies the variant it belongs to.
    """
    if not variant:
        return {}
    for directory in (root, root.parent):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            name = path.name.lower()
            if "standardrcut" not in name or variant.lower() not in name:
                continue
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise InputPreparationError(
                    f"invalid standard cutoff index: {path}"
                ) from error
            if not isinstance(values, Mapping):
                raise InputPreparationError(f"invalid standard cutoff index: {path}")
            index = {}
            for element, value in values.items():
                try:
                    index[str(element)] = float(value)
                except (TypeError, ValueError):
                    continue
            return index
    return {}


def _variant_token(path: Path, root: Path, element: str) -> Optional[str]:
    """Return the SZ/DZP/TZDP-style token of a resource path, if it has one.

    Resource libraries such as SG15 and Dojo store one directory per element
    and variant, named like ``Si_DZP``.
    """
    directories = path.relative_to(root).parts[:-1]
    if not directories:
        return None
    name = Path(directories[0]).name
    for separator in ("_", "-", "."):
        prefix = f"{element}{separator}"
        if name.lower().startswith(prefix.lower()):
            return name[len(prefix) :] or None
    return None


def _matches_variant(path: Path, root: Path, element: str, variant: str) -> bool:
    """Return whether a resource path belongs to the requested variant."""
    token = _variant_token(path, root, element)
    if token is not None:
        return token.lower() == variant.lower()
    directories = path.relative_to(root).parts[:-1]
    return bool(directories) and Path(directories[0]).name.lower() == variant.lower()


def _select_resource(
    element: str,
    resources: Sequence[Path],
    root: Path,
    variant: Optional[str],
    cutoffs: Optional[Mapping[str, float]] = None,
    variant_required: bool = False,
) -> Path:
    """Pick one resource file for an element, honouring the orbital variant.

    A library without variant directories is unaffected.  A library that only
    provides other variants raises instead of silently using a different basis.
    Within the selected variant the cutoff radius recommended by the library
    index is preferred, falling back to the ``Others`` entry.

    Args:
        element: Element symbol the resources belong to.
        resources: Candidate files of this element, best first.
        root: Resource directory the candidates were collected from.
        variant: Requested variant such as ``DZP`` or ``precision``.
        cutoffs: Recommended cutoff radii published by the library.
        variant_required: Warn when an explicitly requested variant cannot be
            honoured, instead of silently picking the best candidate.
    """
    candidates = list(resources)
    matched = False
    if variant:
        matching = [
            path for path in candidates if _matches_variant(path, root, element, variant)
        ]
        if matching:
            candidates = matching
            matched = True
        else:
            available = sorted(
                {
                    token
                    for path in candidates
                    if (token := _variant_token(path, root, element)) is not None
                }
            )
            if available:
                raise InputPreparationError(
                    f"no {variant} resource for {element} in {root}; available variants: "
                    + ", ".join(available)
                )
    if variant_required and not matched:
        warnings.warn(
            f"the configured orbital library {root} has no {variant} variant; "
            "using its default orbital set",
            stacklevel=4,
        )
    if cutoffs:
        cutoff = cutoffs.get(element, cutoffs.get("Others"))
        if cutoff is not None:
            pattern = re.compile(rf"(?<![0-9]){cutoff:g}au(?![0-9])")
            matching = [path for path in candidates if pattern.search(path.name)]
            if matching:
                return matching[0]
    return candidates[0]


def _collect_library(
    path: Optional[PathLike],
    resource_type: Optional[str] = None,
    variant: Optional[str] = None,
    elements: Optional[Iterable[str]] = None,
    variant_required: bool = False,
) -> dict[str, Path]:
    """Collect element-to-file mappings from a resource directory.

    Args:
        path: Library file or directory.
        resource_type: One of ``pp``, ``orb`` or ``paw``.
        variant: Orbital variant such as ``DZP``, see :func:`_select_resource`.
        elements: Elements to resolve; all of them when omitted.  Restricting
            the mapping keeps an unrelated element that lacks the requested
            variant, such as La in Dojo-NC-SR, from failing the whole run.
        variant_required: Warn when the requested variant is unavailable.
    """
    if path is None:
        return {}
    root = Path(path).expanduser()
    root = _resolve_library_path(root, resource_type)
    if root.is_file():
        element = _element_from_filename(root.name)
        if element is None:
            raise InputPreparationError(f"cannot infer element from resource file: {root}")
        return {element: root.resolve()}

    mapping: dict[str, Path] = {}
    element_file = root / "element.json"
    if element_file.is_file():
        try:
            configured = json.loads(element_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InputPreparationError(f"invalid element.json: {element_file}") from error
        for element, filename in configured.items():
            resource = root / str(filename)
            if not resource.is_file():
                raise InputPreparationError(
                    f"element.json maps {element} to a missing file: {resource}"
                )
            mapping[str(element).capitalize()] = resource.resolve()
        return mapping

    preferred_suffixes = {
        "pp": {".upf"},
        "orb": {".orb"},
        "paw": {".paw"},
    }.get(resource_type)
    candidates = []
    for resource in root.rglob("*"):
        if not resource.is_file():
            continue
        element = _element_from_filename(resource.name)
        if element is None:
            continue
        priority = 0 if preferred_suffixes and resource.suffix.lower() in preferred_suffixes else 1
        candidates.append((priority, resource.name, element, resource.resolve()))

    by_element: dict[str, list[Path]] = {}
    for _, _, element, resource in sorted(candidates):
        by_element.setdefault(element, []).append(resource)
    cutoffs = _standard_rcut_index(root, variant) if resource_type == "orb" else {}
    wanted = None if elements is None else {str(element) for element in elements}
    for element, resources in by_element.items():
        if wanted is not None and element not in wanted:
            continue
        mapping[element] = _select_resource(
            element, resources, root, variant, cutoffs, variant_required
        )
    return mapping


def _recommended_cutoffs(path: Optional[PathLike]) -> dict[str, float]:
    """Read optional recommended plane-wave cutoffs from a resource library."""
    if path is None:
        return {}
    root = Path(path).expanduser()
    if not root.is_dir():
        return {}
    cutoff_file = root / "ecutwfc.json"
    if not cutoff_file.is_file():
        return {}
    try:
        values = json.loads(cutoff_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputPreparationError(f"invalid ecutwfc.json: {cutoff_file}") from error
    result = {}
    for element, value in values.items():
        try:
            result[str(element).capitalize()] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _resolve_resource(
    filename: str,
    source_dir: Path,
    library: Mapping[str, Path],
) -> Optional[Path]:
    path = Path(filename).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((source_dir / path, library.get(path.name, Path())))
        candidates.extend(resource for resource in library.values() if resource.name == path.name)
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    return None


def _resource_assignments(
    structure: AbacusSTRU,
    source_dir: Path,
    library: Mapping[str, Path],
    attribute: str,
    required: bool,
    configured_path: Optional[PathLike] = None,
    library_name: Optional[str] = None,
) -> dict[Path, str]:
    """Resolve one STRU resource type and return its source files."""
    assignments: dict[str, tuple[str, Path]] = {}
    missing = []
    for element in _unique(structure.elements):
        atoms = [atom for atom in structure.atoms if atom.element == element]
        filename = next((getattr(atom, attribute) for atom in atoms if getattr(atom, attribute)), None)
        resource = _resolve_resource(filename, source_dir, library) if filename else None
        if resource is None:
            resource = library.get(element)
        if resource is None:
            if required:
                missing.append(element)
            continue
        assignments[element] = (resource.name, resource)

    if missing:
        resource_name = {
            "pp": "pseudopotential",
            "orb": "orbital",
            "paw": "PAW",
        }[attribute]
        message = f"missing {resource_name} for element(s): {', '.join(missing)}"
        if configured_path is None:
            if library_name is not None:
                message += (
                    f"; configure resources.libraries.{library_name}.{attribute} "
                    "in ~/.abacustools/config.yaml"
                )
        else:
            message += f"; no matching file in {configured_path}"
        raise InputPreparationError(message)

    setter = {"pp": structure.set_pp, "orb": structure.set_orb, "paw": structure.set_paw}[attribute]
    if assignments and (required or len(assignments) == len(_unique(structure.elements))):
        setter({element: filename for element, (filename, _) in assignments.items()})
    return {resource: filename for filename, resource in assignments.values()}


def _install_resources(
    resources: Mapping[Path, str], destination: Path, copy_resources: bool
) -> None:
    for source, filename in resources.items():
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if copy_resources:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source)


def _folder_name(source: Path, index: int, syntax: Optional[str]) -> str:
    if syntax is None:
        return f"{index:06d}"
    try:
        # Evaluate the configured f-string against the source name and index
        # only, so a folder-syntax value cannot reach any other name.
        value = eval("f" + repr(syntax), {"__builtins__": {}}, {"x": source.name, "i": index})
    except Exception as error:
        raise InputPreparationError(f"invalid folder syntax: {syntax}") from error
    folder = Path(str(value))
    if not str(value) or folder.is_absolute() or ".." in folder.parts:
        raise InputPreparationError(f"folder syntax escapes output directory: {syntax}")
    return str(folder)


def available_job_types() -> tuple[str, ...]:
    templates = CONFIG.get("input_templates", {})
    return tuple(templates)


def available_resource_libraries() -> tuple[str, ...]:
    """Return the configured pseudopotential/orbital library names."""
    libraries = CONFIG.get("resources", {}).get("libraries", {})
    return tuple(sorted(libraries))


class InputPreparer:
    """Generate complete ABACUS input directories from structure files.

    Resource libraries are selected from ``CONFIG``. The explicit ``pp_path``
    and ``orb_path`` arguments remain available for programmatic compatibility.
    """

    def __init__(
        self,
        files: Union[PathLike, Sequence[PathLike]],
        *,
        output_dir: PathLike = ".",
        filetype: Optional[str] = None,
        job_type: str = "scf",
        library: Optional[str] = None,
        pp_path: Optional[PathLike] = None,
        orb_path: Optional[PathLike] = None,
        orb_variant: Optional[str] = None,
        paw_path: Optional[PathLike] = None,
        input_template: Optional[PathLike] = None,
        kpt: Optional[Sequence[int]] = None,
        kpt_model: str = "gamma",
        basis: Optional[str] = None,
        nspin: int = 1,
        soc: bool = False,
        dftu: bool = False,
        dftu_param: Optional[Mapping[str, Any]] = None,
        init_mag: Optional[Mapping[str, float]] = None,
        afm: bool = False,
        set_params: Optional[Mapping[str, Any]] = None,
        copy_resources: bool = False,
        folder_syntax: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        self.files = [files] if isinstance(files, (str, Path)) else list(files)
        self.output_dir = Path(output_dir).expanduser().absolute()
        self.filetype = filetype
        self.job_type = job_type
        self.library = library or CONFIG.get("resources", {}).get("default")
        configured_resources = self._resource_library(self.library)
        legacy_pp = os.environ.get("ABACUS_PP_PATH") if library is None else None
        legacy_orb = os.environ.get("ABACUS_ORB_PATH") if library is None else None
        self.pp_path = (
            pp_path
            if pp_path is not None
            else configured_resources.get("pp") or legacy_pp
        )
        self.orb_path = (
            orb_path
            if orb_path is not None
            else configured_resources.get("orb") or legacy_orb
        )
        configured_variant = (
            configured_resources.get("orb_variant")
            or CONFIG.get("resources", {}).get("orb_variant")
        )
        self.orb_variant = (
            orb_variant if orb_variant is not None else configured_variant
        )
        self.orb_variant_explicit = orb_variant is not None
        if self.orb_variant is not None:
            self.orb_variant = str(self.orb_variant)
        configured_variants = configured_resources.get("orb_variants") or {}
        self.orb_variants = (
            {str(name).lower(): path for name, path in configured_variants.items()}
            if isinstance(configured_variants, Mapping)
            else {}
        )
        self.paw_path = paw_path if paw_path is not None else os.environ.get("ABACUS_PAW_PATH")
        self.input_template = input_template
        self.kpt = list(kpt) if kpt is not None else None
        self.kpt_model = kpt_model
        self.basis = basis.lower() if basis else None
        self.nspin = nspin
        self.soc = soc
        self.dftu = dftu
        self.dftu_param = dict(dftu_param) if dftu_param is not None else None
        self.init_mag = dict(init_mag) if init_mag is not None else None
        self.afm = afm
        self.set_params = (
            {str(key).lower(): value for key, value in set_params.items()}
            if set_params is not None
            else {}
        )
        self.copy_resources = copy_resources
        self.folder_syntax = folder_syntax
        self.overwrite = overwrite
        self._validate_options()

    @staticmethod
    def _resource_library(name: Optional[str]) -> Mapping[str, Any]:
        """Return one configured resource library and validate its name."""
        libraries = CONFIG.get("resources", {}).get("libraries", {})
        if name is None:
            return {}
        if name not in libraries:
            available = ", ".join(sorted(libraries)) or "none"
            raise ValueError(
                f"unsupported resource library: {name}; available libraries: {available}"
            )
        configured = libraries[name]
        if not isinstance(configured, Mapping):
            raise ValueError(f"invalid resource library configuration: resources.libraries.{name}")
        return configured

    def _validate_options(self) -> None:
        if self.job_type not in available_job_types():
            raise ValueError(
                f"unsupported job type: {self.job_type}; "
                f"supported types are {list(available_job_types())}"
            )
        configured_bases = CONFIG.get("basis_settings", {})
        if self.basis is not None and self.basis not in configured_bases:
            raise ValueError(f"unsupported basis: {self.basis}")
        requested_basis = self.set_params.get("basis_type")
        if requested_basis is not None:
            requested_basis = str(requested_basis).lower()
            if requested_basis not in configured_bases:
                raise ValueError(
                    f"unsupported basis_type: {requested_basis}; configured bases are "
                    f"{', '.join(sorted(configured_bases)) or 'none'}"
                )
            if self.basis is not None and requested_basis != self.basis:
                raise ValueError(
                    f"--basis {self.basis} conflicts with --set basis_type "
                    f"{requested_basis}; use a single basis"
                )
        if self.nspin not in (1, 2, 4):
            raise ValueError("nspin must be 1, 2, or 4")
        if self.soc and self.nspin != 4:
            self.nspin = 4
        self.kpt_model = NormalizeKptModel(self.kpt_model)
        if self.kpt is not None:
            self.kpt = _normalize_kpt(self.kpt, self.kpt_model)
            # Validate before any directory is created.
            FormatKpt(self.kpt, self.kpt_model)

    def _sources(self) -> list[Path]:
        sources = []
        for pattern in self.files:
            matches = sorted(Path(match) for match in glob(str(Path(pattern).expanduser())))
            if not matches and Path(pattern).is_file():
                matches = [Path(pattern)]
            if not matches:
                raise InputPreparationError(f"structure file does not exist: {pattern}")
            sources.extend(match.resolve() for match in matches if match.is_file())
        if not sources:
            raise InputPreparationError("no structure files were found")
        return sources

    def _base_inputs(self) -> dict[str, Any]:
        inputs = deepcopy(CONFIG["input_templates"][self.job_type])
        explicit_basis = self.basis
        if self.input_template is not None:
            template_path = Path(self.input_template).expanduser()
            if not template_path.is_file():
                raise InputPreparationError(f"INPUT template does not exist: {template_path}")
            template = ReadInput(template_path)
            template.pop("calculation", None)
            inputs.update(template)
        if explicit_basis is not None:
            template_basis = inputs.get("basis_type")
            if template_basis is not None and str(template_basis).lower() != explicit_basis:
                raise InputPreparationError(
                    f"INPUT template requests basis_type {template_basis} but "
                    f"--basis {explicit_basis} was given; use a single basis"
                )
            inputs["basis_type"] = explicit_basis

        basis = explicit_basis or str(
            inputs.get("basis_type", CONFIG["abacus"].get("default_basis", "pw"))
        ).lower()

        inputs.update(self.set_params)
        # The job type always wins over a template or --set calculation value.
        inputs["calculation"] = self.job_type
        basis = str(inputs.get("basis_type", basis)).lower()
        settings = CONFIG.get("basis_settings", {})
        if basis not in settings:
            raise InputPreparationError(
                f"unsupported basis_type: {basis}; configured bases are "
                f"{', '.join(sorted(settings)) or 'none'}"
            )
        for key, value in settings[basis].items():
            inputs.setdefault(key, deepcopy(value))
        inputs["basis_type"] = basis
        if self.nspin == 2:
            inputs.update({"nspin": 2, "mixing_beta": 0.4, "symmetry": 0})
            if basis.startswith("lcao"):
                inputs["out_mul"] = 1
        elif self.nspin == 4:
            inputs.update({"nspin": 4, "noncolin": 1, "mixing_beta": 0.4, "symmetry": -1})
            if self.soc:
                inputs["lspinorb"] = 1
            if basis.startswith("lcao"):
                inputs["out_mul"] = 1
        if basis.startswith("pw"):
            inputs.pop("out_mul", None)
            inputs.pop("onsite_radius", None)
        return inputs

    def _dftu_inputs(self, inputs: dict[str, Any], structure: AbacusSTRU) -> None:
        if not self.dftu:
            return
        elements = _unique(structure.elements)
        corrections = []
        values = []
        for element in elements:
            configured = self.dftu_param.get(element) if self.dftu_param else None
            if isinstance(configured, (list, tuple)):
                orbital = _ORBITAL_INDEX.get(str(configured[0]).lower())
                if orbital is None or len(configured) != 2:
                    raise ValueError(f"invalid DFT+U setting for {element}: {configured}")
                corrections.append(orbital)
                values.append(float(configured[1]))
            elif configured is not None:
                orbital = 2 if element in _MAGNETIC_D_ELEMENTS else 3 if element in _MAGNETIC_F_ELEMENTS else 1
                corrections.append(orbital)
                values.append(float(configured))
            elif self.dftu_param is None and element in _MAGNETIC_D_ELEMENTS:
                corrections.append(2)
                values.append(4.0)
            elif self.dftu_param is None and element in _MAGNETIC_F_ELEMENTS:
                corrections.append(3)
                values.append(6.0)
            else:
                corrections.append(-1)
                values.append(0.0)
        inputs.update({"dft_plus_u": 1, "orbital_corr": corrections, "hubbard_u": values})
        if any(orbital >= 0 for orbital in corrections):
            if inputs.get("basis_type", "pw").startswith("pw"):
                inputs["uramping"] = max(values)
            else:
                inputs["mixing_dmr"] = 1
        inputs["mixing_restart"] = 0.001

    def _set_initial_magnets(self, structure: AbacusSTRU) -> None:
        if not self.init_mag and not self.afm:
            return
        values = []
        configured = self.init_mag or {}
        magnetic_indices: dict[str, list[int]] = {}
        for index, atom in enumerate(structure.atoms):
            value = float(configured.get(atom.element, 0.0))
            if atom.element in configured:
                magnetic_indices.setdefault(atom.element, []).append(index)
            values.append(value)
        if self.afm and not self.init_mag:
            for index, atom in enumerate(structure.atoms):
                if atom.element in _MAGNETIC_D_ELEMENTS or atom.element in _MAGNETIC_F_ELEMENTS:
                    values[index] = 3.0
                    magnetic_indices.setdefault(atom.element, []).append(index)
        if self.afm:
            for indices in magnetic_indices.values():
                for index in indices[::2]:
                    values[index] = -values[index]
        if self.nspin == 4:
            structure.atom_mags = [(0.0, 0.0, value) for value in values]
        else:
            structure.atom_mags = values

    def _prepare_structure(self, source: Path, structure: AbacusSTRU, basis: str) -> dict[Path, str]:
        source_dir = source.parent
        elements = _unique(structure.elements)
        pp_resources = _resource_assignments(
            structure,
            source_dir,
            _collect_library(self.pp_path, "pp", elements=elements),
            "pp",
            required=True,
            configured_path=self.pp_path,
            library_name=self.library,
        )
        if basis.startswith("lcao"):
            orbital_library = self._orbital_library()
            orb_resources = _resource_assignments(
                structure,
                source_dir,
                _collect_library(
                    orbital_library,
                    "orb",
                    variant=self.orb_variant,
                    elements=elements,
                    variant_required=self.orb_variant_explicit
                    and not self._variant_is_mapped(),
                ),
                "orb",
                required=True,
                configured_path=orbital_library,
                library_name=self.library,
            )
        else:
            # A plane-wave job neither ships nor references numerical orbitals.
            structure.set_orb({element: None for element in elements})
            orb_resources = {}
        paw_resources = _resource_assignments(
            structure,
            source_dir,
            _collect_library(self.paw_path, "paw", elements=elements),
            "paw",
            required=False,
        )
        resources = {}
        resources.update(pp_resources)
        resources.update(orb_resources)
        resources.update(paw_resources)
        return resources

    def _orbital_library(self) -> Optional[PathLike]:
        """Return the orbital directory holding the requested variant.

        SG15 and Dojo express variants as subdirectories of one directory,
        while APNS ships a separate directory per variant (efficiency,
        precision).  ``orb_variants`` maps the latter, so the same ``--variant``
        and ``orb_variant`` settings select either kind.
        """
        if self._variant_is_mapped():
            return self.orb_variants[(self.orb_variant or "").lower()]
        return self.orb_path

    def _variant_is_mapped(self) -> bool:
        """Return whether ``orb_variants`` maps the selected variant."""
        variant = (self.orb_variant or "").lower()
        return bool(variant) and variant in self.orb_variants

    def _destination(self, source: Path, index: int) -> Path:
        base = self.output_dir / _folder_name(source, index, self.folder_syntax)
        if self.overwrite and base.exists():
            if base.is_dir() and not base.is_symlink():
                shutil.rmtree(base)
            else:
                base.unlink()
            return base
        candidate = base
        suffix = 1
        while candidate.exists() or candidate.is_symlink():
            candidate = Path(f"{base}.{suffix}")
            suffix += 1
        return candidate

    def _write_kpt(self, inputs: Mapping[str, Any], source: Path, destination: Path) -> None:
        filename = str(inputs.get("kpoint_file", "KPT"))
        (destination / filename).parent.mkdir(parents=True, exist_ok=True)
        if self.kpt is not None:
            WriteKpt(self.kpt, destination / filename, model=self.kpt_model)
            return
        if IsEnabled(inputs.get("gamma_only")) or IsEnabled(inputs.get("kspacing")):
            return
        source_kpt = source.parent / filename
        if source_kpt.is_file():
            if self.copy_resources:
                shutil.copy2(source_kpt, destination / filename)
            else:
                (destination / filename).symlink_to(source_kpt.resolve())
            return
        warnings.warn(
            f"no KPT file found for {source.name}; writing a 1x1x1 Gamma mesh. "
            "Pass --kpt, or set kspacing/gamma_only in INPUT, to choose the mesh.",
            stacklevel=2,
        )
        WriteKpt([1, 1, 1, 0, 0, 0], destination / filename, model="gamma")

    def _apply_orbital_cutoff(
        self, inputs: dict[str, Any], resources: Mapping[Path, str]
    ) -> None:
        """Set the LCAO cutoff energy from the selected numerical orbitals.

        ABACUS needs ``ecutwfc`` to cover the cutoff the orbitals were generated
        with, which their file names encode.  An explicitly requested value is
        kept, but a value below the orbital cutoff is reported.
        """
        cutoffs = [
            cutoff
            for path in resources
            if path.suffix.lower() == ".orb"
            if (cutoff := _orbital_cutoff(path.name)) is not None
        ]
        if not cutoffs:
            return
        required = max(cutoffs)
        current = inputs.get("ecutwfc")
        if current is None:
            inputs["ecutwfc"] = required
            return
        try:
            current_value = float(current)
        except (TypeError, ValueError):
            return
        if current_value < required:
            warnings.warn(
                f"ecutwfc {current_value:g} Ry is below the {required:g} Ry cutoff of the "
                "selected numerical orbitals; ABACUS needs at least the orbital cutoff",
                stacklevel=3,
            )

    def _check_spin_orbit_pseudopotentials(
        self, resources: Mapping[Path, str]
    ) -> None:
        """Warn when a spinor calculation uses pseudopotentials without SO data.

        ``nspin=4`` needs a fully relativistic pseudopotential, but the job is
        still prepared: the warning names the files that do not declare it.
        """
        unsupported = []
        for path in resources:
            if path.suffix.lower() != ".upf":
                continue
            support = _spin_orbit_support(path)
            if support is True:
                continue
            unsupported.append(path.name if support is False else f"{path.name} (unreadable)")
        if unsupported:
            warnings.warn(
                "nspin=4 requires pseudopotentials that explicitly support fully "
                "relativistic (spinor) calculations; these do not declare it: "
                + ", ".join(sorted(unsupported))
                + ". The job was still prepared.",
                stacklevel=3,
            )

    def run(self) -> list[PreparedJob]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        inputs_template = self._base_inputs()
        pp_cutoffs = _recommended_cutoffs(self.pp_path)
        jobs = []
        for index, source in enumerate(self._sources()):
            structure = AbacusSTRU.read(source, fmt=self.filetype)
            if structure is None:
                raise InputPreparationError(f"failed to read structure: {source}")
            inputs = deepcopy(inputs_template)
            basis = str(inputs.get("basis_type", "pw")).lower()
            resources = self._prepare_structure(source, structure, basis)
            self._set_initial_magnets(structure)
            self._dftu_inputs(inputs, structure)
            if basis.startswith("lcao"):
                self._apply_orbital_cutoff(inputs, resources)
            elif "ecutwfc" not in inputs:
                recommendations = [pp_cutoffs[element] for element in _unique(structure.elements) if element in pp_cutoffs]
                if recommendations:
                    inputs["ecutwfc"] = max(recommendations)
            if self.nspin == 4:
                self._check_spin_orbit_pseudopotentials(resources)

            destination = self._destination(source, index)
            destination.mkdir(parents=True, exist_ok=False)
            WriteInput(inputs, destination / "INPUT")
            if not structure.write(destination / "STRU"):
                raise InputPreparationError(f"failed to write structure: {destination / 'STRU'}")
            _install_resources(resources, destination, self.copy_resources)
            self._write_kpt(inputs, source, destination)
            jobs.append(PreparedJob(path=destination, source=source))
        return jobs
