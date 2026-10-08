"""Tools for accompanying using ABACUS.

The file-handling data types are importable straight from the top level, so a
script can write ``from abacustools import AbacusSTRU`` instead of reaching
into the ``io`` submodule.  The names resolve lazily (PEP 562):
``import abacustools`` stays cheap, and the module that defines a type is
imported only when the name is first used.
"""

from __future__ import annotations

import importlib
from typing import Any

from abacustools.version import __version__

#: Public name -> module that defines it.  Kept in one place so ``__all__`` and
#: the lazy resolver never drift apart.
_LAZY_EXPORTS: dict[str, str] = {
    # Structures
    "AbacusSTRU": "abacustools.io.stru",
    "AbacusATOM": "abacustools.io.stru",
    "AbacusAtomType": "abacustools.io.stru",
    "StructureConversionWarning": "abacustools.io.stru",
    "Unitcell": "abacustools.data.unitcell",
    # INPUT
    "ReadInput": "abacustools.io.abacus",
    "WriteInput": "abacustools.io.abacus",
    # Pseudopotentials and numerical orbitals
    "UPF": "abacustools.io.pseudo",
    "AbacusNAO": "abacustools.io.nao",
    # Molden files
    "MoldenShell": "abacustools.io.molden",
    "MoldenAtom": "abacustools.io.molden",
    "MoldenOrbital": "abacustools.io.molden",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name to the type from its defining module."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public names of the package."""
    return sorted(__all__)


__all__ = ["__version__", *_LAZY_EXPORTS]
