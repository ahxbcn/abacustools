"""Harmonic vibrational analysis of a finite-difference Hessian.

The vibration workflow used to hand its Hessian to ASE
(:class:`ase.vibrations.data.VibrationsData` and
:class:`ase.thermochemistry.HarmonicThermo`) for the normal mode analysis and
the thermochemistry.  The routines here implement the same mass-weighted
analysis inside the package, so postprocessing is self-contained and the
intermediate quantities of the harmonic model (mass-weighted eigenvalues,
reduced masses, per-mode force constants, per-mode thermal contributions) stay
available to callers.

Conventions, which follow the ASE implementation:

* The Hessian is ``d^2 E / d r_i d r_j`` in eV/Angstrom^2 for the Cartesian
  coordinates of the atoms it is built for.
* Normal modes are obtained from the mass-weighted Hessian
  ``M^(-1/2) H M^(-1/2)``.  Modes are given as Cartesian displacement vectors
  ``u = M^(-1/2) v`` and are normalized to ``sum_i m_i |u_i|^2 = 1``.
* Mode energies are reported in eV and frequencies in cm^-1.  A mode with a
  negative eigenvalue is unstable: its frequency is purely imaginary and it
  contributes neither to the zero-point energy nor to the thermochemistry.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

from abacustools.core.constant import (
    AMU_TO_KG,
    ANGSTROM_TO_METRE,
    BOLTZMANN_CONSTANT_EV_PER_K,
    ELEMENTARY_CHARGE,
    EV_TO_HARTREE,
    HBAR,
    INV_CM_TO_EV,
)

if TYPE_CHECKING:
    from abacustools.io.stru import AbacusSTRU


#: Frequencies whose imaginary part is smaller than this are treated as real,
#: in cm^-1.
IMAGINARY_TOLERANCE = 1.0e-8

#: Factor that turns a mass-weighted Hessian eigenvalue ``lambda`` in
#: eV/(Angstrom^2 amu) into a mode energy ``E = HESSIAN_TO_ENERGY * sqrt(lambda)``
#: in eV.  It is ``hbar / sqrt(Angstrom^2 * amu * elementary charge)``, written
#: with the SI units of the mass-weighted Hessian.
HESSIAN_TO_ENERGY = HBAR / np.sqrt(
    ANGSTROM_TO_METRE ** 2 * AMU_TO_KG * ELEMENTARY_CHARGE
)


def validate_stepsize(stepsize: float) -> None:
    """Validate a finite positive Cartesian displacement.

    Args:
        stepsize: Displacement in Angstrom.

    Raises:
        ValueError: When the step is not a positive finite number.
    """
    if not np.isfinite(stepsize) or stepsize <= 0:
        raise ValueError("stepsize must be a positive finite number")


def selected_atom_indices(selected_atoms: Any, natoms: int) -> list[int]:
    """Validate one-based CLI atom indices and return zero-based indices.

    Args:
        selected_atoms: One-based atom indices from the command line, or None
            to select every atom.
        natoms: Number of atoms in the structure.

    Returns:
        Sorted zero-based atom indices.

    Raises:
        ValueError: When the selection is empty, out of range, or repeated.
    """
    if selected_atoms is None:
        return list(range(natoms))
    if not selected_atoms:
        raise ValueError("selected atom indices must not be empty")
    if any(isinstance(index, bool) for index in selected_atoms):
        raise ValueError("atom indices must be positive integers")
    indices = [int(index) for index in selected_atoms]
    if any(index < 1 or index > natoms for index in indices):
        raise ValueError(f"atom indices must be between 1 and {natoms}")
    if len(set(indices)) != len(indices):
        raise ValueError("atom indices must not contain duplicates")
    return sorted(index - 1 for index in indices)


def frequency_values(frequencies: Sequence[complex]) -> list[float]:
    """Represent imaginary frequencies as negative real values.

    Args:
        frequencies: Complex frequencies, in cm^-1.  A purely imaginary value
            describes an unstable mode.

    Returns:
        list[float]: Real frequencies in cm^-1, in which an imaginary
        frequency is written as the negative of its magnitude.
    """
    values = []
    for frequency in np.asarray(frequencies, dtype=complex):
        if abs(frequency.imag) > IMAGINARY_TOLERANCE:
            if abs(frequency.real) > IMAGINARY_TOLERANCE:
                raise RuntimeError(
                    f"frequency has both real and imaginary parts: {frequency}"
                )
            values.append(-abs(float(frequency.imag)))
        else:
            values.append(float(frequency.real))
    return values


@dataclass(frozen=True)
class HarmonicThermo:
    """Harmonic oscillator thermochemistry of a set of modes at one temperature.

    All energies are in eV and all entropies and heat capacities in eV/K.  The
    internal and the free energy include the zero-point energy, matching
    :class:`ase.thermochemistry.HarmonicThermo` without a potential energy.

    Attributes:
        temperature: Temperature in Kelvin.
        zero_point_energy: Zero-point energy of the stable modes, in eV.
        thermal_energy: Thermal vibrational energy ``dU_v`` above the
            zero-point energy, in eV.
        internal_energy: ``zero_point_energy + thermal_energy``, in eV.
        entropy: Vibrational entropy, in eV/K.
        heat_capacity: Vibrational heat capacity at constant volume, in eV/K.
        free_energy: Helmholtz free energy ``U - T S``, in eV.
        n_imaginary_modes: Number of modes skipped because they are unstable.
    """

    temperature: float
    zero_point_energy: float
    thermal_energy: float
    internal_energy: float
    entropy: float
    heat_capacity: float
    free_energy: float
    n_imaginary_modes: int = 0


def _bose_occupation(x: np.ndarray) -> np.ndarray:
    """Return the Bose-Einstein occupation ``1 / (exp(x) - 1)``.

    The occupation is evaluated through ``expm1`` and the argument is clipped,
    so that low temperatures, where the occupation is zero up to machine
    precision, do not overflow.
    """
    return 1.0 / np.expm1(np.minimum(x, 700.0))


def harmonic_thermo(mode_energies: Sequence[float], temperature: float) -> HarmonicThermo:
    """Calculate the harmonic thermochemistry of one set of stable modes.

    Args:
        mode_energies: Mode energies ``h nu`` in eV.  Unstable modes must be
            removed by the caller; every energy is expected to be positive.
        temperature: Temperature in Kelvin.

    Returns:
        HarmonicThermo: Zero-point energy, thermal energy, entropy, heat
        capacity and free energy of the modes.

    Raises:
        ValueError: If the temperature is not a positive finite number or a
            mode energy is not positive.
    """
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    energies = np.asarray(mode_energies, dtype=float)
    if energies.ndim != 1:
        raise ValueError("mode energies must be a one-dimensional sequence")
    if energies.size and (not np.all(np.isfinite(energies)) or np.any(energies <= 0.0)):
        raise ValueError("mode energies must be positive finite numbers")

    kT = BOLTZMANN_CONSTANT_EV_PER_K * temperature
    x = energies / kT
    occupation = _bose_occupation(x)
    zero_point_energy = 0.5 * float(energies.sum())
    thermal_energy = float(np.sum(energies * occupation))
    internal_energy = zero_point_energy + thermal_energy
    entropy = BOLTZMANN_CONSTANT_EV_PER_K * float(
        np.sum(x * occupation - np.log1p(-np.exp(-x)))
    )
    heat_capacity = BOLTZMANN_CONSTANT_EV_PER_K * float(
        np.sum(x ** 2 * occupation * (occupation + 1.0))
    )
    return HarmonicThermo(
        temperature=float(temperature),
        zero_point_energy=zero_point_energy,
        thermal_energy=thermal_energy,
        internal_energy=internal_energy,
        entropy=entropy,
        heat_capacity=heat_capacity,
        free_energy=internal_energy - temperature * entropy,
    )


class HarmonicVibration:
    """Normal modes of the Hessian of a vibrating set of atoms.

    The Hessian may cover only a subset of the atoms of the system, for
    example the atoms selected for a finite-difference calculation; the indices
    of those atoms are kept so that modes can be expanded to the full
    structure.

    Args:
        hessian: Second derivatives of the energy with respect to Cartesian
            displacements, in eV/Angstrom^2, either as a ``(3n, 3n)`` matrix
            or as an ``(n, 3, n, 3)`` array of the ``n`` atoms of ``masses``.
            The matrix is symmetrized before diagonalization, because central
            finite differences only give a symmetric Hessian up to numerical
            noise.
        masses: Atomic masses in amu, one per Hessian atom.
        indices: Zero-based indices of the Hessian atoms in the full structure,
            used when modes are expanded to all atoms.  Defaults to
            ``0 .. n - 1``.
        natoms: Number of atoms in the full structure.  Defaults to the
            largest value of ``indices`` plus one.

    Raises:
        ValueError: If the Hessian, the masses or the atom indices are
            inconsistent.
    """

    def __init__(
        self,
        hessian: Union[np.ndarray, Sequence],
        masses: Union[np.ndarray, Sequence[float]],
        indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
        natoms: Optional[int] = None,
    ) -> None:
        array = np.asarray(hessian, dtype=float)
        if array.ndim == 4:
            if array.shape[1] != 3 or array.shape[3] != 3 or array.shape[0] != array.shape[2]:
                raise ValueError("hessian must be a (3n, 3n) or (n, 3, n, 3) array")
            array = array.reshape(3 * array.shape[0], 3 * array.shape[0])
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] % 3:
            raise ValueError("hessian must be a (3n, 3n) or (n, 3, n, 3) array")
        if not np.all(np.isfinite(array)):
            raise ValueError("hessian must contain finite values")
        n_atoms = array.shape[0] // 3

        masses_array = np.asarray(masses, dtype=float)
        if masses_array.shape != (n_atoms,):
            raise ValueError(
                f"masses must contain one value per Hessian atom ({n_atoms} expected)"
            )
        if not np.all(np.isfinite(masses_array)) or np.any(masses_array <= 0.0):
            raise ValueError("masses must be positive finite numbers")

        if indices is None:
            indices_array = np.arange(n_atoms, dtype=int)
        else:
            raw_indices = np.asarray(indices)
            if raw_indices.dtype.kind not in "iu" or raw_indices.shape != (n_atoms,):
                raise ValueError(
                    f"indices must contain one integer per Hessian atom ({n_atoms} expected)"
                )
            indices_array = raw_indices.astype(int)
            if np.any(indices_array < 0):
                raise ValueError("indices must not be negative")
            if len(set(indices_array.tolist())) != n_atoms:
                raise ValueError("indices must not contain duplicates")

        if natoms is None:
            natoms = int(indices_array.max()) + 1 if n_atoms else 0
        if int(natoms) < 1 or (n_atoms and int(indices_array.max()) >= int(natoms)):
            raise ValueError("natoms must cover every atom index")

        self._hessian = 0.5 * (array + array.T)
        self._masses = masses_array.copy()
        self._indices = indices_array.copy()
        self._natoms = int(natoms)

    @classmethod
    def from_structure(
        cls,
        structure: "AbacusSTRU",
        hessian: Union[np.ndarray, Sequence],
        indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
        masses: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> "HarmonicVibration":
        """Build the harmonic analysis of a Hessian over an ABACUS structure.

        Args:
            structure: Equilibrium structure the Hessian was calculated for.
            hessian: Second derivatives of the energy in eV/Angstrom^2 over the
                atoms given by ``indices``.
            indices: Zero-based indices of the Hessian atoms in ``structure``.
                Defaults to every atom of the structure.
            masses: Optional masses in amu overriding the masses of the
                structure.

        Returns:
            HarmonicVibration: Analysis over the selected atoms.
        """
        all_masses = np.asarray(
            structure.masses if masses is None else masses, dtype=float
        )
        if all_masses.shape != (len(structure.atoms),):
            raise ValueError("masses must contain one value per structure atom")
        if indices is None:
            selected = np.arange(len(all_masses), dtype=int)
        else:
            raw_indices = np.asarray(indices)
            if raw_indices.dtype.kind not in "iu" or raw_indices.ndim != 1:
                raise ValueError("indices must be a sequence of integers")
            selected = raw_indices.astype(int)
            if np.any(selected < 0) or np.any(selected >= len(all_masses)):
                raise ValueError("indices must refer to atoms of the structure")
        return cls(
            hessian,
            all_masses[selected],
            indices=selected,
            natoms=len(all_masses),
        )

    @property
    def hessian(self) -> np.ndarray:
        """np.ndarray: Symmetrized Hessian over the Hessian atoms, in eV/Angstrom^2."""
        return self._hessian.copy()

    @property
    def masses(self) -> np.ndarray:
        """np.ndarray: Masses of the Hessian atoms in amu."""
        return self._masses.copy()

    @property
    def indices(self) -> np.ndarray:
        """np.ndarray: Indices of the Hessian atoms in the full structure."""
        return self._indices.copy()

    @property
    def n_atoms(self) -> int:
        """int: Number of atoms covered by the Hessian."""
        return int(self._masses.size)

    @property
    def n_modes(self) -> int:
        """int: Number of vibrational modes, that is three per Hessian atom."""
        return 3 * self.n_atoms

    @property
    def natoms(self) -> int:
        """int: Number of atoms in the full structure."""
        return self._natoms

    @cached_property
    def _eigenpairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the mass-weighted eigenvalues and eigenvectors of the Hessian."""
        weights = np.repeat(self._masses ** -0.5, 3)
        weighted = weights[:, np.newaxis] * self._hessian * weights[np.newaxis, :]
        return np.linalg.eigh(weighted)

    @property
    def eigenvalues(self) -> np.ndarray:
        """np.ndarray: Mass-weighted eigenvalues in eV/(Angstrom^2 amu)."""
        return self._eigenpairs[0].copy()

    @property
    def energies(self) -> np.ndarray:
        """np.ndarray: Mode energies ``h nu`` in eV, imaginary when unstable."""
        return HESSIAN_TO_ENERGY * np.sqrt(self._eigenpairs[0].astype(complex))

    @property
    def frequencies(self) -> np.ndarray:
        """np.ndarray: Mode frequencies in cm^-1, imaginary when unstable."""
        return self.energies / INV_CM_TO_EV

    @property
    def signed_frequencies(self) -> list[float]:
        """list[float]: Frequencies in cm^-1, with unstable modes negative."""
        return frequency_values(self.frequencies)

    @property
    def imaginary_modes(self) -> np.ndarray:
        """np.ndarray: Indices of the imaginary, that is unstable, modes."""
        return np.flatnonzero(np.abs(self.frequencies.imag) > IMAGINARY_TOLERANCE)

    @property
    def modes(self) -> np.ndarray:
        """np.ndarray: Cartesian displacement modes of the Hessian atoms.

        The modes are normalized to ``sum_i m_i |u_i|^2 = 1`` and are ordered
        like :attr:`frequencies`, as a ``(3n, n, 3)`` array of
        (mode, atom, direction) entries.
        """
        vectors = self._eigenpairs[1]
        modes = vectors.T.reshape(self.n_modes, self.n_atoms, 3)
        return modes * (self._masses ** -0.5)[np.newaxis, :, np.newaxis]

    def modes_all_atoms(self) -> np.ndarray:
        """Expand the modes to the atoms of the full structure.

        Returns:
            np.ndarray: Displacement modes as a ``(3n, natoms, 3)`` array, in
            which the atoms that the Hessian does not cover are zero.
        """
        modes = np.zeros((self.n_modes, self.natoms, 3), dtype=float)
        modes[:, self._indices, :] = self.modes
        return modes

    @cached_property
    def reduced_masses(self) -> np.ndarray:
        """np.ndarray: Effective mass of every mode in amu.

        The modes use the usual normal mode convention: the Cartesian mode
        vectors are normalized to ``sum_i m_i |u_i|^2 = 1``, so that the
        reduced mass of a mode is ``mu = 1 / sum_i |u_i|^2``.  Together with
        the force constant of the mode it reproduces the frequency through
        ``omega = sqrt(k / mu)``.  A mode that moves light atoms alone, such as
        an X-H stretch, has a small reduced mass, while the translations and
        rotations of a molecule carry its total mass.
        """
        return 1.0 / np.sum(self.modes ** 2, axis=(1, 2))

    @cached_property
    def force_constants(self) -> np.ndarray:
        """np.ndarray: Effective force constant of every mode in eV/Angstrom^2.

        The force constant of a mode is the mass-weighted eigenvalue scaled
        back by its reduced mass, ``k = mu * lambda``, so that the pair of
        :attr:`reduced_masses` and :attr:`force_constants` describes the mode as
        a one-dimensional oscillator.
        """
        return self.eigenvalues * self.reduced_masses

    def zero_point_energy(self, *, stable_only: bool = False) -> float:
        """Return the zero-point energy of the modes in eV.

        Args:
            stable_only: If True, unstable modes are left out of the sum, as
                in the harmonic thermochemistry (and in
                ``ase.vibrations.data.VibrationsData``).  The default keeps the
                historical convention of the vibration workflow, which adds
                the magnitude of an imaginary mode.

        Returns:
            float: Zero-point energy in eV.
        """
        energies = self.energies
        if stable_only:
            return 0.5 * float(np.sum(energies.real))
        return 0.5 * float(np.sum(np.abs(energies)))

    def thermo(
        self,
        temperature: float,
        *,
        ignore_imaginary: bool = True,
    ) -> HarmonicThermo:
        """Calculate the harmonic thermochemistry of the modes at one temperature.

        Args:
            temperature: Temperature in Kelvin.
            ignore_imaginary: If True, unstable modes are skipped and counted
                in :attr:`HarmonicThermo.n_imaginary_modes`.  If False, an
                unstable mode raises a :class:`RuntimeError` instead, because
                the harmonic partition function is not defined for it.

        Returns:
            HarmonicThermo: Harmonic thermochemistry at ``temperature``.

        Raises:
            RuntimeError: If the modes are unstable and ``ignore_imaginary`` is
                False.
        """
        imaginary = self.imaginary_modes
        if imaginary.size and not ignore_imaginary:
            raise RuntimeError(
                "the harmonic thermochemistry requires a stable structure, "
                f"but {imaginary.size} mode(s) are imaginary"
            )
        stable = np.real(self.energies)
        stable = stable[stable > 0.0]
        thermo = harmonic_thermo(stable, temperature)
        return HarmonicThermo(
            temperature=thermo.temperature,
            zero_point_energy=thermo.zero_point_energy,
            thermal_energy=thermo.thermal_energy,
            internal_energy=thermo.internal_energy,
            entropy=thermo.entropy,
            heat_capacity=thermo.heat_capacity,
            free_energy=thermo.free_energy,
            n_imaginary_modes=int(imaginary.size),
        )

    def summary(self) -> list[dict[str, Any]]:
        """Return one dictionary of mode properties per mode.

        Returns:
            list[dict[str, Any]]: One entry per mode with its one-based number,
            signed frequency in cm^-1, mode energy in eV, reduced mass in amu
            and force constant in eV/Angstrom^2.  The energy of an unstable
            mode is reported as a negative value, like its frequency.
        """
        energies = self.energies
        frequencies = self.signed_frequencies
        imaginary = set(self.imaginary_modes.tolist())
        return [
            {
                "mode": index + 1,
                "frequency": frequencies[index],
                "energy": -float(abs(energies[index])) if index in imaginary else float(energies[index].real),
                "reduced_mass": float(self.reduced_masses[index]),
                "force_constant": float(self.force_constants[index]),
            }
            for index in range(self.n_modes)
        ]


#: Speed of light in cm/s, the unit of the Gaussian frequencies.
_SPEED_OF_LIGHT_CM_PER_S = 2.99792458e10

#: Header of the Gaussian harmonic-frequency section.
_GAUSSIAN_FREQUENCY_HEADER = (
    " Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering",
    " activities (A**4/AMU), depolarization ratios for plane and unpolarized",
    " incident light, reduced masses (AMU), force constants (mDyne/A),",
    " and normal coordinates:",
)

#: Separator of the Gaussian orientation and frequency blocks.
_GAUSSIAN_RULE = " " + "-" * 69

#: The Gaussian "Standard orientation" banner, with the trailing spaces the
#: real output carries.
_GAUSSIAN_STANDARD_ORIENTATION = (
    "                         Standard orientation:                         "
)

#: The Gaussian "Input orientation" banner, which a periodic calculation uses
#: because the translation vectors turn the point-group symmetry off.
_GAUSSIAN_INPUT_ORIENTATION = (
    "                          Input orientation:                          "
)

#: Atomic number Gaussian gives the translation-vector pseudo-atoms that carry
#: the unit cell of a periodic structure.
_GAUSSIAN_TRANSLATION_VECTOR_NUMBER = -2

#: Labels of the per-mode lines, each exactly 15 characters wide.
_GAUSSIAN_FREQUENCY_LABEL = " Frequencies --"
_GAUSSIAN_REDUCED_MASS_LABEL = " Red. masses --"
_GAUSSIAN_FORCE_CONSTANT_LABEL = " Frc consts  --"
_GAUSSIAN_IR_LABEL = " IR Inten    --"


def _gaussian_atomic_numbers(elements: Sequence[str]) -> list[int]:
    """Return the atomic numbers of the elements, through the ASE table."""
    from ase.data import atomic_numbers

    numbers = []
    for element in elements:
        symbol = str(element)
        if symbol not in atomic_numbers:
            raise ValueError(f"unknown element symbol: {symbol}")
        numbers.append(int(atomic_numbers[symbol]))
    return numbers


def _cell_lengths_and_angles(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell lengths in Angstrom and the three cell angles in degrees."""
    vectors = np.asarray(cell, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    angles = []
    for first, second in ((1, 2), (0, 2), (0, 1)):
        cosine = float(
            np.dot(vectors[first], vectors[second]) / (lengths[first] * lengths[second])
        )
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return lengths, np.asarray(angles)


def _gaussian_orientation_lines(
    structure: "AbacusSTRU", *, periodic: bool = True
) -> list[str]:
    """Render the Gaussian orientation block of a structure.

    A periodic structure follows the Gaussian PBC convention: the block is an
    ``Input orientation`` whose last three centers are the translation vectors,
    written as pseudo-atoms with atomic number ``-2``, and the lengths and
    angles of those vectors follow the coordinates.  GaussView reads that cell.
    A non-periodic structure keeps the usual ``Standard orientation``.

    The column widths follow a real Gaussian output so that GaussView reads the
    geometry without a special case.
    """
    coords = np.asarray(structure.coords, dtype=float)
    numbers = _gaussian_atomic_numbers([str(element) for element in structure.elements])
    cell = np.asarray(structure.cell, dtype=float) if periodic else None
    if cell is not None and abs(float(np.linalg.det(cell))) < 1.0e-8:
        cell = None
    banner = (
        _GAUSSIAN_INPUT_ORIENTATION if cell is not None else _GAUSSIAN_STANDARD_ORIENTATION
    )
    lines = [
        "",
        banner,
        _GAUSSIAN_RULE,
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        _GAUSSIAN_RULE,
    ]
    for center, (number, xyz) in enumerate(zip(numbers, coords), start=1):
        lines.append(
            f"{center:7d}{number:11d}{0:12d}"
            f"{xyz[0]:16.6f}{xyz[1]:12.6f}{xyz[2]:12.6f}"
        )
    if cell is not None:
        for offset, vector in enumerate(cell, start=len(numbers) + 1):
            lines.append(
                f"{offset:7d}{_GAUSSIAN_TRANSLATION_VECTOR_NUMBER:11d}{0:12d}"
                f"{vector[0]:16.6f}{vector[1]:12.6f}{vector[2]:12.6f}"
            )
    lines.append(_GAUSSIAN_RULE)
    if cell is not None:
        lengths, angles = _cell_lengths_and_angles(cell)
        lines.append(
            " Lengths of translation vectors:"
            + f"{lengths[0]:>14.6f}{lengths[1]:>12.6f}{lengths[2]:>12.6f}"
        )
        lines.append(
            "  Angles of translation vectors:"
            + f"{angles[0]:>14.6f}{angles[1]:>12.6f}{angles[2]:>12.6f}"
        )
        lines.append(_GAUSSIAN_RULE)
    return lines


def _gaussian_value_line(label: str, values: Sequence[float]) -> str:
    """Render one ``Frequencies``/``Red. masses`` style line of up to three modes."""
    line = label
    line += f"{values[0]:>12.4f}"
    for value in values[1:]:
        line += f"{value:>23.4f}"
    return line


def _gaussian_mode_columns(
    frequencies: Sequence[float],
    modes: np.ndarray,
    masses: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the printed displacements, reduced masses and force constants.

    Gaussian prints the Cartesian normal coordinates normalized to
    ``sum_i |u_i|^2 = 1``; the reduced mass is then ``mu = sum_i m_i |u_i|^2``
    and the force constant follows from ``k = mu omega^2``, in mDyne/Angstrom.
    """
    array = np.asarray(modes, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    masses = np.asarray(masses, dtype=float)
    squared = np.sum(array ** 2, axis=(1, 2))
    if np.any(squared <= 0.0):
        raise ValueError("a normal mode has no displacement to print")
    weighted = np.sum(array ** 2 * masses[np.newaxis, :, np.newaxis], axis=(1, 2))
    reduced = weighted / squared
    # omega = 2 pi c nu; the imaginary mode keeps the sign of its frequency.
    omega = 2.0 * np.pi * _SPEED_OF_LIGHT_CM_PER_S * frequencies
    constants = reduced * AMU_TO_KG * omega * np.abs(omega) / 100.0
    displacements = array / np.sqrt(squared)[:, np.newaxis, np.newaxis]
    return displacements, reduced, constants


def _gaussian_frequency_block(
    frequencies: np.ndarray,
    displacements: np.ndarray,
    reduced: np.ndarray,
    constants: np.ndarray,
    numbers: Sequence[int],
) -> list[str]:
    """Render the Gaussian harmonic-frequency block, three modes per group."""
    lines = list(_GAUSSIAN_FREQUENCY_HEADER)
    n_modes = len(frequencies)
    natoms = displacements.shape[1]
    for start in range(0, n_modes, 3):
        block = list(range(start, min(start + 3, n_modes)))
        lines.append("".join(f"{index + 1:>23d}" for index in block))
        lines.append("".join(f"{'A':>23}" for _ in block))
        lines.append(
            _gaussian_value_line(_GAUSSIAN_FREQUENCY_LABEL, [frequencies[i] for i in block])
        )
        lines.append(
            _gaussian_value_line(_GAUSSIAN_REDUCED_MASS_LABEL, [reduced[i] for i in block])
        )
        lines.append(
            _gaussian_value_line(_GAUSSIAN_FORCE_CONSTANT_LABEL, [constants[i] for i in block])
        )
        lines.append(_gaussian_value_line(_GAUSSIAN_IR_LABEL, [0.0 for _ in block]))
        lines.append(
            "  Atom  AN" + "  ".join("".join(f"{axis:>7}" for axis in "XYZ") for _ in block)
        )
        for atom in range(natoms):
            row = f"{atom + 1:6d}{numbers[atom]:4d}  "
            for index in block:
                xyz = displacements[index, atom]
                row += f"{xyz[0]:7.2f}{xyz[1]:7.2f}{xyz[2]:7.2f}  "
            lines.append(row)
    return lines


def _gaussian_thermal_line(label: str, value: float, end: int) -> str:
    """Render one ``label`` + value line of the thermal section."""
    return f"{label}{value:>{end + 1 - len(label)}.6f}"


def _gaussian_thermal_lines(
    temperature: float,
    zero_point_energy: Optional[float],
    electronic_energy: Optional[float],
    thermo: Optional[dict[str, Any]],
) -> list[str]:
    """Render the thermochemistry section, with only the values that exist.

    The workflow reports the vibrational internal and free energies relative to
    the electronic energy, which is exactly what the Gaussian thermal
    corrections are, so no reference shift is applied.
    """
    thermo = thermo or {}
    lines = [
        "",
        f" Temperature   {temperature:7.3f} Kelvin.  Pressure   1.00000 Atm.",
    ]
    zpe = None if zero_point_energy is None else float(zero_point_energy) * EV_TO_HARTREE
    if zpe is not None:
        lines.append(
            _gaussian_thermal_line(" Zero-point correction=", zpe, 57) + " Hartree"
        )
    rt = BOLTZMANN_CONSTANT_EV_PER_K * temperature * EV_TO_HARTREE
    correction_energy = None
    if thermo.get("internal_energy") is not None:
        correction_energy = float(thermo["internal_energy"]) * EV_TO_HARTREE
        lines.append(
            _gaussian_thermal_line(" Thermal correction to Energy=", correction_energy, 57)
        )
        lines.append(
            _gaussian_thermal_line(
                " Thermal correction to Enthalpy=", correction_energy + rt, 57
            )
        )
    correction_gibbs = None
    if thermo.get("free_energy") is not None:
        correction_gibbs = float(thermo["free_energy"]) * EV_TO_HARTREE + rt
        lines.append(
            _gaussian_thermal_line(
                " Thermal correction to Gibbs Free Energy=", correction_gibbs, 57
            )
        )
    energy = None if electronic_energy is None else float(electronic_energy) * EV_TO_HARTREE
    if energy is not None:
        lines.append(_gaussian_thermal_line(" Electronic energy=", energy, 64))
        if zpe is not None:
            lines.append(
                _gaussian_thermal_line(
                    " Sum of electronic and zero-point Energies=", energy + zpe, 64
                )
            )
        if correction_energy is not None:
            lines.append(
                _gaussian_thermal_line(
                    " Sum of electronic and thermal Energies=",
                    energy + correction_energy,
                    64,
                )
            )
            lines.append(
                _gaussian_thermal_line(
                    " Sum of electronic and thermal Enthalpies=",
                    energy + correction_energy + rt,
                    64,
                )
            )
        if correction_gibbs is not None:
            lines.append(
                _gaussian_thermal_line(
                    " Sum of electronic and thermal Free Energies=",
                    energy + correction_gibbs,
                    64,
                )
            )
    return lines


def gaussian_frequency_log(
    structure: "AbacusSTRU",
    frequencies: Sequence[float],
    modes: np.ndarray,
    *,
    masses: Optional[Sequence[float]] = None,
    periodic: bool = True,
    temperature: float = 298.15,
    electronic_energy: Optional[float] = None,
    zero_point_energy: Optional[float] = None,
    thermo: Optional[dict[str, Any]] = None,
) -> str:
    """Render a fake Gaussian harmonic-frequency log of a vibration analysis.

    The text follows the layout of a Gaussian frequency job closely enough that
    GaussView opens it and animates the normal modes, in the spirit of OfakeG
    and CP2KfakeG.  The frequencies are the signed wavenumbers in cm^-1, the
    modes are the Cartesian displacements of every atom of ``structure`` as a
    ``(n_modes, natoms, 3)`` array, and the thermochemistry is optional.

    Args:
        structure: Equilibrium structure the modes belong to.
        frequencies: Signed frequencies in cm^-1, negative for unstable modes.
        modes: Cartesian displacement modes of every atom.
        masses: Atomic masses in amu; defaults to the masses of the structure.
        periodic: Write the cell as Gaussian translation vectors, so that a
            periodic ABACUS structure is shown as a unit cell in GaussView.
        temperature: Temperature of the thermal section in Kelvin.
        electronic_energy: Electronic energy of the reference job in eV.
        zero_point_energy: Zero-point energy in eV.
        thermo: Thermal corrections in eV, with the keys ``internal_energy`` and
            ``free_energy`` as far as they are available.

    Returns:
        The complete fake Gaussian log as one string.
    """
    array = np.asarray(modes, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if array.ndim != 3 or array.shape[1] != structure.natoms or array.shape[2] != 3:
        raise ValueError(
            "modes must be a (n_modes, natoms, 3) array over the structure atoms"
        )
    if array.shape[0] != frequencies.size:
        raise ValueError("frequencies and modes must describe the same number of modes")
    per_atom_masses = (
        np.asarray(structure.masses, dtype=float) if masses is None else np.asarray(masses, dtype=float)
    )
    if per_atom_masses.shape != (structure.natoms,):
        raise ValueError("masses must hold one value per structure atom")

    displacements, reduced, constants = _gaussian_mode_columns(
        frequencies, array, per_atom_masses
    )
    numbers = _gaussian_atomic_numbers([str(element) for element in structure.elements])

    lines = [
        " ! This file was generated by abacustools for viewing in GaussView",
        " ! abacustools workflow vibration postprocess --gaussian-log",
        "",
        " 0 basis functions",
        " 0 alpha electrons",
        " 0 beta electrons",
        "GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad",
        "GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad",
    ]
    lines.extend(_gaussian_orientation_lines(structure, periodic=periodic))
    if electronic_energy is not None:
        lines.append("")
        lines.append(
            " SCF Done:  E(ABACUS) = {:>18.12E} A.U. after    1 cycles".format(
                float(electronic_energy) * EV_TO_HARTREE
            )
        )
    lines.append("")
    lines.extend(_gaussian_frequency_block(frequencies, displacements, reduced, constants, numbers))
    lines.extend(
        _gaussian_thermal_lines(temperature, zero_point_energy, electronic_energy, thermo)
    )
    lines.append("")
    lines.append(" Normal termination of Gaussian")
    return "\n".join(lines) + "\n"


def write_gaussian_frequency_log(
    path: Path,
    structure: "AbacusSTRU",
    frequencies: Sequence[float],
    modes: np.ndarray,
    **options: Any,
) -> Path:
    """Write :func:`gaussian_frequency_log` to ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        gaussian_frequency_log(structure, frequencies, modes, **options),
        encoding="utf-8",
    )
    return path
