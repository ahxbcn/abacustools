import math

from abacustools.core.config import CONFIG

HARTREE_TO_EV = CONFIG['constants']['hartree2ev']
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
RY_TO_EV = HARTREE_TO_EV / 2.0
EV_TO_RY = 1.0 / RY_TO_EV

BOHR_TO_ANG = CONFIG['constants']['bohr2ang']
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

#: Grams per atomic mass unit, used to turn a cell mass in amu and a cell
#: volume in Angstrom^3 into a density in g/cm^3.
AMU_TO_GRAM = 1.66053906660e-24

PLANCK_CONSTANT = CONFIG['constants']['planck_constant']
BOLTZMANN_CONSTANT = CONFIG['constants']['boltzmann_constant']
ELEMENTARY_CHARGE = CONFIG['constants']['elementary_charge']
ELECTRON_MASS = CONFIG['constants']['electron_mass']
VACUUM_PERMITTIVITY = CONFIG['constants']['vacuum_permittivity']

HBAR = PLANCK_CONSTANT / (2.0 * math.pi)
BOLTZMANN_CONSTANT_EV_PER_K = BOLTZMANN_CONSTANT / ELEMENTARY_CHARGE
THZ_TO_K = PLANCK_CONSTANT / BOLTZMANN_CONSTANT * 1.0e12

#: Speed of light in vacuum, in metres per second.
SPEED_OF_LIGHT = 299792458.0

#: Energy of one reciprocal centimetre, in eV, which turns spectroscopic
#: wavenumbers such as vibrational frequencies into photon energies.
INV_CM_TO_EV = PLANCK_CONSTANT * SPEED_OF_LIGHT * 100.0 / ELEMENTARY_CHARGE
