"""Tests for configured physical constants and unit conversions."""

from abacustools.core.config import CONFIG
from abacustools.core.constant import (
    ANG_TO_BOHR,
    BOHR_TO_ANG,
    BOLTZMANN_CONSTANT_EV_PER_K,
    THZ_TO_K,
)
from abacustools.io.stru import A2BOHR, BOHR2A


def test_unit_conversions_use_configured_constants():
    constants = CONFIG["constants"]
    assert BOHR_TO_ANG == constants["bohr2ang"]
    assert ANG_TO_BOHR == 1.0 / constants["bohr2ang"]
    assert BOHR2A == BOHR_TO_ANG
    assert A2BOHR == ANG_TO_BOHR


def test_derived_physical_conversions_use_configured_constants():
    constants = CONFIG["constants"]
    assert BOLTZMANN_CONSTANT_EV_PER_K == (
        constants["boltzmann_constant"] / constants["elementary_charge"]
    )
    assert THZ_TO_K == (
        constants["planck_constant"] / constants["boltzmann_constant"] * 1.0e12
    )
