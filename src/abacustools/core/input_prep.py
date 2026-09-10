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


def _collect_library(path: Optional[PathLike], resource_type: Optional[str] = None) -> dict[str, Path]:
    """Collect element-to-file mappings from a resource directory."""
    if path is None:
        return {}
    root = Path(path).expanduser()
    if not root.exists():
        raise InputPreparationError(f"resource path does not exist: {root}")
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
    for _, _, element, resource in sorted(candidates):
        if element not in mapping:
            mapping[element] = resource
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
        if configured_path is None and library_name is not None:
            message += (
                f"; configure resources.libraries.{library_name}.{attribute} "
                "in ~/.abacustools/config.yaml"
            )
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
        self.set_params = dict(set_params) if set_params is not None else {}
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
        if self.basis is not None and self.basis not in CONFIG.get("basis_settings", {}):
            raise ValueError(f"unsupported basis: {self.basis}")
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
        if explicit_basis in CONFIG.get("basis_settings", {}):
            inputs.update(deepcopy(CONFIG["basis_settings"][explicit_basis]))
        if self.input_template is not None:
            template_path = Path(self.input_template).expanduser()
            if not template_path.is_file():
                raise InputPreparationError(f"INPUT template does not exist: {template_path}")
            template = ReadInput(template_path)
            template.pop("calculation", None)
            inputs.update(template)

        basis = explicit_basis or str(
            inputs.get("basis_type", CONFIG["abacus"].get("default_basis", "pw"))
        ).lower()
        for key, value in CONFIG.get("basis_settings", {}).get(basis, {}).items():
            inputs.setdefault(key, deepcopy(value))
        inputs["basis_type"] = basis

        inputs.update(self.set_params)
        # The job type always wins over a template or --set calculation value.
        inputs["calculation"] = self.job_type
        basis = str(inputs.get("basis_type", basis)).lower()
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
        pp_resources = _resource_assignments(
            structure,
            source_dir,
            _collect_library(self.pp_path, "pp"),
            "pp",
            required=True,
            configured_path=self.pp_path,
            library_name=self.library,
        )
        if basis.startswith("lcao"):
            orb_resources = _resource_assignments(
                structure,
                source_dir,
                _collect_library(self.orb_path, "orb"),
                "orb",
                required=True,
                configured_path=self.orb_path,
                library_name=self.library,
            )
        else:
            # A plane-wave job neither ships nor references numerical orbitals.
            structure.set_orb({element: None for element in _unique(structure.elements)})
            orb_resources = {}
        paw_resources = _resource_assignments(
            structure, source_dir, _collect_library(self.paw_path, "paw"), "paw", required=False
        )
        resources = {}
        resources.update(pp_resources)
        resources.update(orb_resources)
        resources.update(paw_resources)
        return resources

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
            if basis.startswith("pw") and "ecutwfc" not in inputs:
                recommendations = [pp_cutoffs[element] for element in _unique(structure.elements) if element in pp_cutoffs]
                if recommendations:
                    inputs["ecutwfc"] = max(recommendations)

            destination = self._destination(source, index)
            destination.mkdir(parents=True, exist_ok=False)
            WriteInput(inputs, destination / "INPUT")
            if not structure.write(destination / "STRU"):
                raise InputPreparationError(f"failed to write structure: {destination / 'STRU'}")
            _install_resources(resources, destination, self.copy_resources)
            self._write_kpt(inputs, source, destination)
            jobs.append(PreparedJob(path=destination, source=source))
        return jobs
