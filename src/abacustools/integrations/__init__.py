"""Optional integrations with external calculation engines."""

from .abacuslite import (
    AbacusLiteUnavailableError,
    abacuslite_available,
    attach_calculator,
    calculator_from_job,
    calculator_from_structure,
    load_abacuslite,
    make_profile,
    result_to_dict,
    structure_to_atoms,
    write_result,
)

__all__ = [
    "AbacusLiteUnavailableError",
    "abacuslite_available",
    "attach_calculator",
    "calculator_from_job",
    "calculator_from_structure",
    "load_abacuslite",
    "make_profile",
    "result_to_dict",
    "structure_to_atoms",
    "write_result",
]
