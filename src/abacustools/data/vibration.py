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
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

from abacustools.core.constant import (
    AMU_TO_KG,
    ANGSTROM_TO_METRE,
    BOLTZMANN_CONSTANT_EV_PER_K,
    ELEMENTARY_CHARGE,
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
