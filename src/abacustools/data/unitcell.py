from typing import List, Literal
import warnings

import numpy as np


def _coordinate_array(coords: List[List[float]], name: str) -> np.ndarray:
    """Convert coordinates to an array and validate their shape."""
    try:
        array = np.asarray(coords, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc

    if array.size == 0 and array.ndim == 1:
        return array.reshape((0, 3))
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


class Unitcell:
    """Represent a three-dimensional unit cell and its lattice operations.

    Lattice vectors are stored as rows, in the same length unit as provided by
    the caller. Cartesian coordinates are calculated as
    ``fractional @ cell_vector``. Fractional coordinates passed to
    :meth:`frac_to_cart` are wrapped by default.

    Args:
        cell_vector: A nonsingular 3x3 array-like object containing the three
            lattice vectors as rows. A left-handed cell is accepted and emits
            a warning.
    """

    def __init__(self, cell_vector: List[List[float]]):
        """Initialize a unit cell from its three lattice vectors."""
        try:
            cell = np.asarray(cell_vector, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("cell_vector must contain numeric values") from exc

        if cell.shape != (3, 3):
            raise ValueError("cell_vector must have shape (3, 3)")
        if not np.all(np.isfinite(cell)):
            raise ValueError("cell_vector must contain finite values")

        volume = float(np.linalg.det(cell))
        if volume == 0.0:
            raise ValueError("cell_vector must define a non-singular unit cell")

        self.cell_vector = cell
        if volume < 0:
            warnings.warn("The unit cell is left-handed.", UserWarning, stacklevel=2)

    def get_cell_param(self) -> List[float]:
        """Return the six conventional cell parameters.

        Returns:
            A list ``[a, b, c, alpha, beta, gamma]``. The lengths use the same
            unit as the input lattice vectors; the angles are in degrees.
        """
        lengths = np.linalg.norm(self.cell_vector, axis=1)
        a, b, c = lengths

        cos_alpha = np.dot(self.cell_vector[1], self.cell_vector[2]) / (b * c)
        cos_beta = np.dot(self.cell_vector[0], self.cell_vector[2]) / (a * c)
        cos_gamma = np.dot(self.cell_vector[0], self.cell_vector[1]) / (a * b)
        angles = np.degrees(np.arccos(np.clip([cos_alpha, cos_beta, cos_gamma], -1.0, 1.0)))

        return [*lengths.tolist(), *angles.tolist()]

    def get_cell_volume(self) -> float:
        """Return the positive cell volume in the cubed input length unit."""
        return float(abs(np.linalg.det(self.cell_vector)))

    def cart_to_frac(
        self, cart_coords: List[List[float]], wrap: bool = True
    ) -> List[List[float]]:
        """Convert Cartesian coordinates to fractional coordinates.

        Args:
            cart_coords: Cartesian coordinates with shape ``(N, 3)``.
            wrap: If true, map every fractional component to ``[0, 1)``.

        Returns:
            Fractional coordinates with shape ``(N, 3)``.
        """
        cart = _coordinate_array(cart_coords, "cart_coords")
        frac = np.linalg.solve(self.cell_vector.T, cart.T).T
        if wrap:
            frac = frac % 1
        return frac.tolist()

    def frac_to_cart(self, frac_coords: List[List[float]], wrap: bool = True) -> List[List[float]]:
        """Convert fractional coordinates to Cartesian coordinates.

        Args:
            frac_coords: Fractional coordinates with shape ``(N, 3)``.
            wrap: If true, map every fractional component to ``[0, 1)`` before
                converting.

        Returns:
            Cartesian coordinates with shape ``(N, 3)``.
        """
        frac = _coordinate_array(frac_coords, "frac_coords")
        if wrap:
            frac = frac % 1
        return np.dot(frac, self.cell_vector).tolist()

    def reciprocal(self, standard: Literal["abacus", "vasp"] = "abacus") -> "Unitcell":
        """Return the reciprocal unit cell without the ``2*pi`` factor.

        With row-wise lattice vectors, the returned vectors satisfy
        ``reciprocal.cell_vector @ self.cell_vector.T = I``.

        Args:
            standard: Reciprocal-vector convention to use. ``"abacus"`` and
                ``"vasp"`` are currently represented by the same row-wise
                definition.

        Returns:
            A new :class:`Unitcell` containing the reciprocal lattice vectors.
        """
        if standard == "abacus":
            reciprocal_vectors = np.linalg.inv(self.cell_vector).T
        elif standard == "vasp":
            reciprocal_vectors = np.linalg.inv(self.cell_vector).T
        return Unitcell(reciprocal_vectors.tolist())
