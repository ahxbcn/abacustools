"""Adapter for the ASE harmonic vibration interface.

``abacustools workflow vibration postprocess`` uses ASE by default: the
Hessian of central force differences is handed to
:class:`ase.vibrations.data.VibrationsData` and the thermochemistry to
:class:`ase.thermochemistry.HarmonicThermo`.  This module collects that
interface, so the workflow can switch between it and the built-in analysis of
:mod:`abacustools.data.vibration`, which exists for extensions that ASE does
not provide.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms
    from ase.thermochemistry import HarmonicThermo
    from ase.vibrations.data import VibrationsData

    from abacustools.io.stru import AbacusSTRU


class AseVibrationData:
    """Harmonic analysis of a Hessian through the ASE vibration classes.

    The class mirrors the small part of
    :class:`abacustools.data.vibration.HarmonicVibration` that the vibration
    workflow uses, so that the two backends are interchangeable.  Frequencies
    and thermochemistry follow ASE exactly, including the ASE atomic masses of
    the converted structure.

    Args:
        structure: Equilibrium ABACUS structure the Hessian was calculated for.
        hessian: Second derivatives of the energy with respect to Cartesian
            displacements, in eV/Angstrom^2, as a ``(3n, 3n)`` matrix or as an
            ``(n, 3, n, 3)`` array over the atoms of ``indices``.
        indices: Zero-based indices of the Hessian atoms in the structure.
            Defaults to every atom of the structure.
        masses: Optional masses in amu, one per atom of the structure, which
            replace the masses ASE would otherwise use.  Defaults to the masses
            of the structure, that is the relative atomic masses declared in
            the ``ATOMIC_SPECIES`` block of the STRU file.

    Raises:
        ValueError: If the Hessian does not match the number of atoms or the
            masses do not match the structure.
    """

    def __init__(
        self,
        structure: "AbacusSTRU",
        hessian: Union[np.ndarray, Sequence],
        indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
        masses: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> None:
        from ase.vibrations.data import VibrationsData

        self.atoms: Atoms = structure.to("ase")
        all_masses = np.asarray(
            structure.masses if masses is None else masses, dtype=float
        )
        if all_masses.shape != (len(self.atoms),):
            raise ValueError(
                "masses must contain one value per structure atom "
                f"({len(self.atoms)} expected, got {all_masses.shape})"
            )
        self.atoms.set_masses(all_masses)
        self.masses = all_masses.copy()
        if indices is None:
            selected = list(range(len(self.atoms)))
        else:
            selected = [int(index) for index in np.asarray(indices).reshape(-1)]
        array = np.asarray(hessian, dtype=float)
        if array.ndim == 4:
            array = array.reshape(array.shape[0] * 3, array.shape[2] * 3)
        if array.shape != (3 * len(selected), 3 * len(selected)):
            raise ValueError(
                "hessian must have the shape of the selected atoms, "
                f"expected {(3 * len(selected), 3 * len(selected))}, got {array.shape}"
            )
        self.indices = selected
        self.data: VibrationsData = VibrationsData(
            self.atoms,
            array.reshape(len(selected), 3, len(selected), 3),
            indices=selected,
        )

    @property
    def energies(self) -> np.ndarray:
        """np.ndarray: Mode energies ``h nu`` in eV, imaginary when unstable."""
        return self.data.get_energies()

    @property
    def frequencies(self) -> np.ndarray:
        """np.ndarray: Mode frequencies in cm^-1, imaginary when unstable."""
        return self.data.get_frequencies()

    @property
    def signed_frequencies(self) -> list:
        """list[float]: Frequencies in cm^-1, with unstable modes negative."""
        from abacustools.data.vibration import frequency_values

        return frequency_values(self.frequencies)

    def modes(self) -> np.ndarray:
        """Return the Cartesian displacement modes of all atoms of the structure."""
        return self.data.get_modes(all_atoms=True)

    def zero_point_energy(self) -> float:
        """Return the zero-point energy in eV.

        The sum keeps the convention of the workflow: the magnitude of an
        imaginary mode is added as well, which makes unstable modes visible in
        the reported zero-point energy.
        """
        return float(sum(abs(energy) for energy in self.energies) / 2.0)

    @cached_property
    def _thermo(self) -> "HarmonicThermo":
        """Return the ASE thermochemistry object of the modes, built once."""
        from ase.thermochemistry import HarmonicThermo

        return HarmonicThermo(self.energies, ignore_imag_modes=True)

    def thermo(self, temperature: float) -> Dict[str, Any]:
        """Return the ASE harmonic thermochemistry of one temperature.

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            dict: The entropy in eV/K and the Helmholtz free energy in eV,
            evaluated from the stable modes.
        """
        return {
            "entropy": float(self._thermo.get_entropy(temperature)),
            "free_energy": float(self._thermo.get_helmholtz_energy(temperature)),
        }
