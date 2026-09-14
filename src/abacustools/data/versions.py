"""Version profiles describing how ABACUS branches print their results.

ABACUS does not keep one stable running-log format.  The LTS 3.10 branch and
the develop branch (3.11), for example, report the electronic density error,
the SCF convergence flag and the geometry-optimization progress with different
markers::

    LTS 3.10                              develop (3.11)
    density error = 1.0E-4                Electron density deviation 1.0E-4
    charge density convergence achieved   #SCF IS CONVERGED#
    final etot is -10.0 eV                !FINAL_ETOT_IS -10.0 eV
    STEP OF RELAXATION : 1                RELAX STEP: 1
    Largest gradient in force is 0.5 eV/A Largest force is 0.5 eV/Angstrom
    Relaxation is converged!              end of geometry optimization

:mod:`abacustools.data.abacus_result` therefore reads every marker through the
:class:`VersionProfile` selected here.

The profile of a job is normally detected from the ``ABACUS v...`` banner in
its running log.  The configured version (``abacus.version``) acts as a hint:
an explicit name is used when no log declares its version, and the version
found in the log wins when the two disagree.  Profiles may be extended from
``~/.abacustools/config.yaml`` without touching the code::

    abacus:
      version: develop
      versions:
        develop:
          density_error_keywords: ["electron density deviation", "my marker"]
        my-branch:
          aliases: ["mybranch"]
          version_prefixes: ["6."]
          scf_converged_keywords: ["#SCF DONE#"]
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from abacustools.core.config import CONFIG


PathLike = Union[str, Path]

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_VERSION_BANNER = re.compile(
    r"ABACUS\s+v?(\d+\.\d+(?:\.\d+)?(?:[-_.][0-9A-Za-z.]+)?)"
)
_AUTO_KEY = "auto"

#: Fields that hold marker names or patterns and therefore accept a list of
#: strings in the configuration file.
_MARKER_FIELDS = (
    "aliases",
    "version_prefixes",
    "energy_keywords",
    "final_energy_keywords",
    "density_error_keywords",
    "scf_converged_keywords",
    "fermi_keywords",
    "normal_end_keywords",
    "vdw_keywords",
    "total_mag_keywords",
    "absolute_mag_keywords",
    "orbital_mag_header_keywords",
    "force_header_keywords",
    "stress_header_keywords",
    "scf_step_patterns",
    "ion_step_patterns",
    "md_step_patterns",
    "relax_force_threshold_patterns",
    "relax_stress_threshold_patterns",
    "relax_step_patterns",
    "relax_energy_patterns",
    "relax_force_patterns",
    "relax_stress_patterns",
    "relax_converged_keywords",
)


@dataclass(frozen=True)
class VersionProfile:
    """Log markers used when collecting results from one ABACUS branch.

    Keywords are matched case-insensitively against the running log; patterns
    must expose the captured number as their first group.
    """

    key: str
    aliases: tuple[str, ...] = ()
    version_prefixes: tuple[str, ...] = ()
    energy_keywords: tuple[str, ...] = ("e_kohnsham",)
    final_energy_keywords: tuple[str, ...] = ("final etot",)
    density_error_keywords: tuple[str, ...] = ("density error",)
    scf_converged_keywords: tuple[str, ...] = (
        "charge density convergence is achieved",
    )
    fermi_keywords: tuple[str, ...] = ("efermi",)
    normal_end_keywords: tuple[str, ...] = ("Total  Time  :",)
    vdw_keywords: tuple[str, ...] = ("e_vdw",)
    total_mag_keywords: tuple[str, ...] = ("total magnetism (Bohr mag/cell)",)
    absolute_mag_keywords: tuple[str, ...] = ("absolute magnetism",)
    orbital_mag_header_keywords: tuple[str, ...] = ("orbital charge analysis",)
    force_header_keywords: tuple[str, ...] = ("total-force",)
    stress_header_keywords: tuple[str, ...] = ("total-stress",)
    scf_step_patterns: tuple[str, ...] = (
        r"ion=\s*\+?\d+\s+elec=\s*\+?(\d+)",
        r"#elec\s+iter#\s*\+?(\d+)",
    )
    ion_step_patterns: tuple[str, ...] = (
        r"ion=\s*\+?(\d+)\s+elec=",
        r"#ion\s+move#\s*\+?(\d+)",
    )
    md_step_patterns: tuple[str, ...] = (
        r"step\s+of\s+molecular\s+dynamics\s*:\s*(\d+)",
        r"md\s+step\s*:\s*(\d+)",
    )
    relax_force_threshold_patterns: tuple[str, ...] = (
        rf"threshold\s+is\s+({_FLOAT})\s*eV\s*/",
    )
    relax_stress_threshold_patterns: tuple[str, ...] = (
        rf"threshold\s+is\s+({_FLOAT})\s*kbar",
    )
    relax_step_patterns: tuple[str, ...] = (
        r"step\s+of\s+relaxation\s*:\s*(\d+)",
        r"step\s+of\s+ion\s+relaxation\s*:\s*(\d+)",
    )
    relax_energy_patterns: tuple[str, ...] = (
        rf"(?:final\s+etot\s+is|!final_etot_is)\s+({_FLOAT})",
    )
    relax_force_patterns: tuple[str, ...] = (
        rf"largest\s+gradient\s+in\s+force\s+is\s+({_FLOAT})",
    )
    relax_stress_patterns: tuple[str, ...] = (
        rf"largest\s+gradient\s+in\s+stress\s+is\s+({_FLOAT})",
    )
    relax_converged_keywords: tuple[str, ...] = ("relaxation is converged",)


#: ABACUS LTS 3.8-3.10 log dialect.  This matches the historical behaviour of
#: the result collectors.
_LTS = VersionProfile(
    key="3.10.1LTS",
    aliases=("LTS3.10.1", "3.10.1", "3.10", "3.10.1lts", "lts"),
    version_prefixes=("3.8", "3.9", "3.10"),
)

#: ABACUS develop dialect.  The keyword lists keep the LTS markers where the
#: two dialects do not conflict, so a develop profile also reads older logs.
_DEVELOP = VersionProfile(
    key="develop",
    aliases=("dev", "3.11", "3.12", "beta"),
    version_prefixes=("3.11", "3.12", "4.", "5."),
    final_energy_keywords=("final etot", "!final_etot_is", "#total energy#"),
    density_error_keywords=("density error", "electron density deviation"),
    scf_converged_keywords=(
        "charge density convergence is achieved",
        "#scf is converged#",
    ),
    fermi_keywords=("efermi", "e_fermi"),
    relax_step_patterns=(
        r"step\s+of\s+relaxation\s*:\s*(\d+)",
        r"step\s+of\s+ion\s+relaxation\s*:\s*(\d+)",
        r"relax\s+step\s*:\s*(\d+)",
    ),
    relax_energy_patterns=(
        rf"(?:final\s+etot\s+is|!final_etot_is)\s+({_FLOAT})",
        rf"#\s*total\s+energy\s*#\s+({_FLOAT})",
    ),
    relax_force_patterns=(
        rf"largest\s+gradient\s+in\s+force\s+is\s+({_FLOAT})",
        rf"largest\s+force\s+is\s+({_FLOAT})\s*eV\s*/\s*Angstrom",
    ),
    relax_stress_patterns=(
        rf"largest\s+gradient\s+in\s+stress\s+is\s+({_FLOAT})",
        rf"largest\s+stress\s+is\s+({_FLOAT})\s*kbar",
    ),
    relax_converged_keywords=(
        "relaxation is converged",
        "end of geometry optimization",
    ),
)

_BUILTIN_PROFILES: tuple[VersionProfile, ...] = (_LTS, _DEVELOP)

_REGISTRY_CACHE: Optional[
    tuple[Any, dict[str, VersionProfile], dict[str, str]]
] = None
_WARNED_MISMATCHES: set[tuple[str, str]] = set()


def _warn_mismatch(requested: str, declared: str) -> None:
    """Warn once per process when the configured version disagrees."""
    key = (requested, declared)
    if key in _WARNED_MISMATCHES:
        return
    _WARNED_MISMATCHES.add(key)
    warnings.warn(
        f"ABACUS job reports version {declared!r}, but {requested!r} was "
        f"requested; using {declared!r}.",
        stacklevel=3,
    )


def _normalize(value: str) -> str:
    """Normalize a version or alias for lookups."""
    return value.strip().lower().replace(" ", "").lstrip("v")


def _as_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Convert a configured marker entry into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        items: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise ValueError(f"version profile field {name!r} must be a string or a list")
    return tuple(str(item) for item in items)


def _union(key: str, profiles: Sequence[VersionProfile]) -> VersionProfile:
    """Combine profiles into one that accepts every listed marker."""
    merged: dict[str, tuple[str, ...]] = {}
    for field in fields(VersionProfile):
        if field.name == "key":
            continue
        values: list[str] = []
        for profile in profiles:
            for value in getattr(profile, field.name):
                if value not in values:
                    values.append(value)
        merged[field.name] = tuple(values)
    return VersionProfile(key=key, **merged)


def _configured_profiles() -> Mapping[str, Any]:
    """Return the ``abacus.versions`` mapping of the active configuration."""
    section = CONFIG.get("abacus", {}) if isinstance(CONFIG, Mapping) else {}
    versions = section.get("versions", {}) if isinstance(section, Mapping) else {}
    return versions if isinstance(versions, Mapping) else {}


def _build_registry(
    overrides: Mapping[str, Any],
) -> tuple[dict[str, VersionProfile], dict[str, str]]:
    """Merge configured overrides over the built-in profiles."""
    profiles = {profile.key: profile for profile in _BUILTIN_PROFILES}
    for key, spec in overrides.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"version profile {key!r} must be a mapping")
        base = profiles.get(str(key), VersionProfile(key=str(key)))
        updates = {}
        for name, value in spec.items():
            if name not in _MARKER_FIELDS:
                raise ValueError(
                    f"unknown version profile field {name!r} for {key!r}"
                )
            updates[name] = _as_tuple(value, name)
        profiles[str(key)] = replace(base, **updates)

    profiles[_AUTO_KEY] = _union(_AUTO_KEY, list(profiles.values()))

    aliases: dict[str, str] = {}
    for key, profile in profiles.items():
        if key == _AUTO_KEY:
            continue
        for alias in (key, *profile.aliases):
            aliases[_normalize(alias)] = key
    return profiles, aliases


def _registry() -> tuple[dict[str, VersionProfile], dict[str, str]]:
    """Return the (cached) profile registry for the active configuration."""
    global _REGISTRY_CACHE
    overrides = dict(_configured_profiles())
    if _REGISTRY_CACHE is not None and _REGISTRY_CACHE[0] == overrides:
        return _REGISTRY_CACHE[1], _REGISTRY_CACHE[2]
    profiles, aliases = _build_registry(overrides)
    _REGISTRY_CACHE = (overrides, profiles, aliases)
    return profiles, aliases


def _lookup(
    version: Optional[str],
    profiles: Mapping[str, VersionProfile],
    aliases: Mapping[str, str],
) -> Optional[VersionProfile]:
    """Resolve an explicitly requested version name."""
    if not version:
        return None
    name = _normalize(str(version))
    if not name or name == _AUTO_KEY:
        return None
    key = aliases.get(name)
    if key is None:
        for candidate, profile in profiles.items():
            if candidate == _AUTO_KEY:
                continue
            if any(name.startswith(_normalize(item)) for item in profile.version_prefixes):
                key = candidate
                break
    return None if key is None else profiles[key]


def detect_version_from_text(text: str) -> Optional[VersionProfile]:
    """Detect the ABACUS profile from the banner of a running log."""
    match = _VERSION_BANNER.search(text)
    if match is None:
        return None
    version = match.group(1)
    profiles, _ = _registry()
    for key, profile in profiles.items():
        if key == _AUTO_KEY:
            continue
        if any(version.startswith(prefix) for prefix in profile.version_prefixes):
            return profile
    return None


def detect_version(job_dir: Optional[PathLike]) -> Optional[VersionProfile]:
    """Detect the ABACUS profile from the logs of one job directory."""
    if job_dir is None:
        return None
    job = Path(job_dir)
    candidates = sorted(job.glob("OUT.*/running_*.log"))
    candidates += sorted(job.glob("running_*.log"))
    for log in candidates:
        try:
            with log.open(encoding="utf-8", errors="replace") as handle:
                text = handle.read(8192)
        except OSError:
            continue
        profile = detect_version_from_text(text)
        if profile is not None:
            return profile
    return None


def resolve_version(
    version: Optional[str] = None,
    *,
    job_dir: Optional[PathLike] = None,
    text: Optional[str] = None,
) -> VersionProfile:
    """Select the profile used to read results.

    Args:
        version: Configured or requested ABACUS version.  ``None``/``auto``
            selects the version declared by the output itself.
        job_dir: Job directory whose ``OUT.*/running_*.log`` banners are
            inspected.
        text: Log text inspected instead of ``job_dir``.

    Returns:
        The matching version profile.  The version declared by the output wins
        over an explicitly requested one; a warning is emitted when they
        disagree.  When no version can be detected, a profile accepting every
        built-in marker is returned so that incomplete logs are still read.
    """
    profiles, aliases = _registry()
    requested = _lookup(version, profiles, aliases)
    declared = detect_version_from_text(text) if text else None
    if declared is None:
        declared = detect_version(job_dir)
    if declared is not None:
        if requested is not None and requested.key != declared.key:
            _warn_mismatch(requested.key, declared.key)
        return declared
    if requested is not None:
        return requested
    return profiles[_AUTO_KEY]


def default_version() -> str:
    """Return the configured default ABACUS version name."""
    section = CONFIG.get("abacus", {}) if isinstance(CONFIG, Mapping) else {}
    value = section.get("version", _AUTO_KEY) if isinstance(section, Mapping) else None
    return _AUTO_KEY if value is None else str(value)


__all__ = [
    "VersionProfile",
    "default_version",
    "detect_version",
    "detect_version_from_text",
    "resolve_version",
]
