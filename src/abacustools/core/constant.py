from abacustools.core.config import CONFIG

HARTREE_TO_EV = CONFIG['constants']['hartree2ev']
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
RY_TO_EV = HARTREE_TO_EV / 2.0
EV_TO_RY = 1.0 / RY_TO_EV

BOHR_TO_ANG = CONFIG['constants']['bohr2ang']
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
