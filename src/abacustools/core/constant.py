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

#: Kilograms per atomic mass unit. Together with :data:`ANGSTROM_TO_METRE` it
#: converts a mass-weighted Hessian into frequencies.
AMU_TO_KG = AMU_TO_GRAM * 1.0e-3

#: Metres per Angstrom.
ANGSTROM_TO_METRE = 1.0e-10

PLANCK_CONSTANT = CONFIG['constants']['planck_constant']
BOLTZMANN_CONSTANT = CONFIG['constants']['boltzmann_constant']
ELEMENTARY_CHARGE = CONFIG['constants']['elementary_charge']
ELECTRON_MASS = CONFIG['constants']['electron_mass']
VACUUM_PERMITTIVITY = CONFIG['constants']['vacuum_permittivity']

HBAR = PLANCK_CONSTANT / (2.0 * math.pi)
BOLTZMANN_CONSTANT_EV_PER_K = BOLTZMANN_CONSTANT / ELEMENTARY_CHARGE
THZ_TO_K = PLANCK_CONSTANT / BOLTZMANN_CONSTANT * 1.0e12

#: Avogadro constant, in reciprocal mole.  Phonopy reports its thermal
#: properties per mole of unit cells, which this turns into a value per cell.
AVOGADRO_CONSTANT = 6.02214076e23

#: Kilojoule per mole, the unit phonopy reports a harmonic free energy in, as
#: electronvolt per cell.
KILOJOULE_PER_MOL_TO_EV = 1000.0 / (ELEMENTARY_CHARGE * AVOGADRO_CONSTANT)

#: Joule per kelvin per mole, the unit phonopy reports an entropy and a heat
#: capacity in, as electronvolt per kelvin per cell.
JOULE_PER_MOL_KELVIN_TO_EV_PER_KELVIN = KILOJOULE_PER_MOL_TO_EV * 1.0e-3

#: Speed of light in vacuum, in metres per second.
SPEED_OF_LIGHT = 299792458.0

#: Energy of one reciprocal centimetre, in eV, which turns spectroscopic
#: wavenumbers such as vibrational frequencies into photon energies.
INV_CM_TO_EV = PLANCK_CONSTANT * SPEED_OF_LIGHT * 100.0 / ELEMENTARY_CHARGE

#: One gigapascal as electronvolt per cubic Angstrom.
GPA_TO_EV_PER_ANGSTROM3 = 1.0 / 160.21766208

#: One kilobar as electronvolt per cubic Angstrom.  ABACUS prints stresses in
#: kBar with the compression-positive convention, while ASE stores the
#: tension-positive stress in eV/Angstrom^3, so the two differ by this factor
#: and a sign.
KBAR_TO_EV_PER_ANGSTROM3 = 0.1 * GPA_TO_EV_PER_ANGSTROM3
