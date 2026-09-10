from typing import Optional, Sequence, Tuple, Union, Literal
import os
import struct

import numpy as np

from ase.data import atomic_numbers

from abacustools.core.constant import (
    ANG_TO_BOHR,
    BOHR_TO_ANG,
    ELEMENTARY_CHARGE,
    RY_TO_EV,
    VACUUM_PERMITTIVITY,
)
from abacustools.io.stru import AbacusSTRU


BOHR2A = BOHR_TO_ANG
RY2EV = RY_TO_EV


def _reciprocal_lattice(cell):
    """Calculate reciprocal lattice vectors including the 2*pi factor."""
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), got {cell.shape}")
    volume = np.dot(cell[0], np.cross(cell[1], cell[2]))
    if np.isclose(volume, 0.0):
        raise ValueError("cell must be non-singular")
    return np.asarray(
        [
            2 * np.pi * np.cross(cell[1], cell[2]) / volume,
            2 * np.pi * np.cross(cell[2], cell[0]) / volume,
            2 * np.pi * np.cross(cell[0], cell[1]) / volume,
        ]
    )


def _charge_to_potential(chg, cell):
    """Solve the periodic Poisson equation for a charge density."""
    charge = np.asarray(chg, dtype=float)
    if charge.ndim != 3:
        raise ValueError("charge density must be a three-dimensional array")
    nx, ny, nz = charge.shape
    reciprocal = _reciprocal_lattice(cell)
    grid = np.meshgrid(
        np.fft.fftfreq(nx, d=1 / nx),
        np.fft.fftfreq(ny, d=1 / ny),
        np.fft.fftfreq(nz, d=1 / nz),
        indexing="ij",
    )
    indices = np.vstack([component.ravel() for component in grid]).T
    wavevectors = indices @ reciprocal
    squared = np.sum(wavevectors**2, axis=1).reshape(charge.shape)
    squared[0, 0, 0] = 1.0
    potential = np.fft.ifftn(np.fft.fftn(charge) / squared).real
    return potential / VACUUM_PERMITTIVITY * ELEMENTARY_CHARGE * -1 * 1e10


def _restart_grid_index(miller: np.ndarray, grid_shape) -> np.ndarray:
    """Map Miller indices onto non-negative FFT-array indices (``g mod n``)."""
    shape = np.asarray(grid_shape, dtype=np.int64)
    return np.mod(np.asarray(miller, dtype=np.int64), shape)


def _scatter_rhog(rhog: np.ndarray, miller: np.ndarray, grid_shape) -> np.ndarray:
    full = np.zeros(tuple(int(n) for n in grid_shape), dtype=np.complex128)
    index = _restart_grid_index(miller, grid_shape)
    full[index[:, 0], index[:, 1], index[:, 2]] = rhog
    return full


def _gather_rhog(full: np.ndarray, miller: np.ndarray, grid_shape) -> np.ndarray:
    index = _restart_grid_index(miller, grid_shape)
    return full[index[:, 0], index[:, 1], index[:, 2]]


def miller_indices_within_cutoff(reciprocal_lattice: np.ndarray, cutoff: float) -> np.ndarray:
    """Enumerate the Miller indices of all G-vectors with ``|G|**2 <= cutoff``.

    Args:
        reciprocal_lattice: (3, 3) array whose rows are the reciprocal lattice
            vectors including the ``2*pi`` factor, i.e. as returned by
            :func:`_reciprocal_lattice`.
        cutoff: Maximum squared length of ``G``, in Bohr^-2.

    Returns:
        An ``(ngm, 3)`` integer array of Miller indices, sorted by ``|G|**2``
        and then lexicographically for reproducibility.
    """
    reciprocal = np.asarray(reciprocal_lattice, dtype=float)
    if reciprocal.shape != (3, 3):
        raise ValueError(f"reciprocal_lattice must have shape (3, 3), got {reciprocal.shape}")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    norms = np.linalg.norm(reciprocal, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("reciprocal_lattice must be non-singular")
    bounds = np.ceil(np.sqrt(cutoff) / norms).astype(np.int64) + 1
    axes = [np.arange(-bound, bound + 1) for bound in bounds]
    miller = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    squared = np.sum((miller @ reciprocal) ** 2, axis=1)
    keep = miller[squared <= cutoff * (1.0 + 1e-12)]
    keep_squared = np.sum((keep @ reciprocal) ** 2, axis=1)
    order = np.lexsort((keep[:, 2], keep[:, 1], keep[:, 0], keep_squared))
    return keep[order]


class RestartCharge:
    """Reader/writer and FFT converter for ABACUS ``*-CHARGE-DENSITY.restart``.

    ABACUS stores the charge density in reciprocal space as complex Fourier
    coefficients ``rho(G)`` on the plane-wave basis. The binary layout (native
    little-endian) follows the Quantum ESPRESSO-like format documented in
    ``source/module_io/rhog_io.h`` of ABACUS::

        int32   3
        int32   gamma_only
        int32   ngm_g
        int32   nspin
        int32   3
        int32   9
        float64[9]  GT  (row-major; inv(latvec), latvec in units of lat0, G in 2*pi/lat0)
        int32   9
        int32   ngm_g * 3
        int32[ngm_g*3]  Miller indices (gx, gy, gz) of every G-vector
        int32   ngm_g * 3
        for each spin channel:
            int32             ngm_g
            complex128[ngm_g] rho(G)
            int32             ngm_g

    The FFT convention matches ABACUS ``PW_Basis``::

        rho(G) = (1/N) * sum_r rho(r) * exp(-i G.r)
        rho(r) =         sum_G rho(G) * exp(+i G.r)

    so with numpy ``rho(G) = fftn(rho(r)) / N`` and ``rho(r) = N * ifftn(rho(G))``.

    Note:
        Only ``gamma_only=False`` files can be converted between real and
        reciprocal space. Files with ``gamma_only=True`` can still be read and
        written, but :meth:`to_real` and :meth:`from_real` raise
        :class:`NotImplementedError` because the half-sphere reconstruction
        depends on ABACUS-specific G-vector ordering.
    """

    def __init__(
        self,
        rhog: np.ndarray,
        miller: np.ndarray,
        reciprocal_lattice: np.ndarray,
        gamma_only: bool = False,
    ):
        rhog = np.asarray(rhog, dtype=np.complex128)
        if rhog.ndim == 1:
            rhog = rhog[np.newaxis, :]
        miller = np.asarray(miller, dtype=np.int64)
        reciprocal_lattice = np.asarray(reciprocal_lattice, dtype=float)

        if miller.ndim != 2 or miller.shape[1] != 3:
            raise ValueError(f"miller must have shape (ngm, 3), got {miller.shape}")
        if rhog.ndim != 2 or rhog.shape[1] != miller.shape[0]:
            raise ValueError(
                f"rhog must have shape (nspin, {miller.shape[0]}), got {rhog.shape}"
            )
        if reciprocal_lattice.shape != (3, 3):
            raise ValueError(
                f"reciprocal_lattice must have shape (3, 3), got {reciprocal_lattice.shape}"
            )

        self._rhog = rhog
        self._miller = miller
        self._reciprocal_lattice = reciprocal_lattice
        self._gamma_only = bool(gamma_only)

    @property
    def rhog(self) -> np.ndarray:
        """The complex G-space coefficients, shape ``(nspin, ngm)``."""
        return self._rhog

    @property
    def miller(self) -> np.ndarray:
        """The Miller indices of the stored G-vectors, shape ``(ngm, 3)``."""
        return self._miller

    @property
    def reciprocal_lattice(self) -> np.ndarray:
        """The row-major matrix stored in the file (``inv(latvec)``, latvec in units of lat0)."""
        return self._reciprocal_lattice

    @property
    def gamma_only(self) -> bool:
        """Whether the file was written in the gamma-only representation."""
        return self._gamma_only

    @property
    def nspin(self) -> int:
        """Number of spin channels stored in the file."""
        return self._rhog.shape[0]

    @property
    def ngm(self) -> int:
        """Number of G-vectors stored in the file."""
        return self._miller.shape[0]

    @classmethod
    def read(cls, filename: str) -> "RestartCharge":
        """Read an ABACUS ``*-CHARGE-DENSITY.restart`` file.

        Args:
            filename: Path to the restart file.

        Returns:
            A :class:`RestartCharge` holding ``rho(G)`` and the G-vectors.
        """
        with open(filename, "rb") as handle:
            def read_int32() -> int:
                raw = handle.read(4)
                if len(raw) != 4:
                    raise ValueError(f"unexpected end of restart file {filename}")
                return struct.unpack("<i", raw)[0]

            def read_float64(count: int) -> np.ndarray:
                raw = handle.read(8 * count)
                if len(raw) != 8 * count:
                    raise ValueError(f"unexpected end of restart file {filename}")
                return np.frombuffer(raw, dtype="<f8").copy()

            def read_int32_array(count: int) -> np.ndarray:
                raw = handle.read(4 * count)
                if len(raw) != 4 * count:
                    raise ValueError(f"unexpected end of restart file {filename}")
                return np.frombuffer(raw, dtype="<i4").copy()

            def expect(value: int, expected: int) -> None:
                if value != expected:
                    raise ValueError(
                        f"malformed restart file {filename}: expected marker "
                        f"{expected}, got {value}"
                    )

            expect(read_int32(), 3)
            gamma_only = read_int32()
            ngm = read_int32()
            nspin = read_int32()
            expect(read_int32(), 3)

            expect(read_int32(), 9)
            reciprocal_lattice = read_float64(9).reshape(3, 3)
            expect(read_int32(), 9)

            expect(read_int32(), 3 * ngm)
            miller = read_int32_array(3 * ngm).reshape(ngm, 3)
            expect(read_int32(), 3 * ngm)

            rhog = np.empty((nspin, ngm), dtype=np.complex128)
            for ispin in range(nspin):
                expect(read_int32(), ngm)
                raw = handle.read(16 * ngm)
                if len(raw) != 16 * ngm:
                    raise ValueError(f"unexpected end of restart file {filename}")
                rhog[ispin] = np.frombuffer(raw, dtype="<c16")
                expect(read_int32(), ngm)

        return cls(rhog, miller, reciprocal_lattice, gamma_only=bool(gamma_only))

    def write(self, filename: str) -> None:
        """Write the current data to an ABACUS ``*-CHARGE-DENSITY.restart`` file.

        Args:
            filename: Path of the file to create.
        """
        ngm = self.ngm
        with open(filename, "wb") as handle:
            handle.write(struct.pack("<i", 3))
            handle.write(struct.pack("<i", int(self._gamma_only)))
            handle.write(struct.pack("<i", ngm))
            handle.write(struct.pack("<i", self.nspin))
            handle.write(struct.pack("<i", 3))

            handle.write(struct.pack("<i", 9))
            handle.write(np.ascontiguousarray(self._reciprocal_lattice, dtype="<f8").tobytes())
            handle.write(struct.pack("<i", 9))

            handle.write(struct.pack("<i", 3 * ngm))
            handle.write(np.ascontiguousarray(self._miller, dtype="<i4").tobytes())
            handle.write(struct.pack("<i", 3 * ngm))

            for ispin in range(self.nspin):
                handle.write(struct.pack("<i", ngm))
                handle.write(np.ascontiguousarray(self._rhog[ispin], dtype="<c16").tobytes())
                handle.write(struct.pack("<i", ngm))

    def to_real(self, grid_shape) -> np.ndarray:
        """Inverse FFT of ``rho(G)`` onto a real-space grid.

        Args:
            grid_shape: ``(nx, ny, nz)`` dimensions of the FFT grid. The grid is
                not stored in the restart file, so it must be supplied (for
                example from ``running_scf.log`` or from the STRU plus cutoff).

        Returns:
            A ``(nspin, nx, ny, nz)`` float array of the charge density in
            ABACUS units (e/Bohr^3).
        """
        if self._gamma_only:
            raise NotImplementedError(
                "gamma_only restart files cannot be converted to real space yet"
            )
        shape = tuple(int(n) for n in grid_shape)
        if len(shape) != 3 or any(n <= 0 for n in shape):
            raise ValueError(f"grid_shape must be three positive integers, got {shape}")
        npoints = int(np.prod(shape))
        real = np.empty((self.nspin,) + shape, dtype=float)
        for ispin in range(self.nspin):
            full = _scatter_rhog(self._rhog[ispin], self._miller, shape)
            real[ispin] = (np.fft.ifftn(full) * npoints).real
        return real

    @classmethod
    def from_real(
        cls,
        rho: np.ndarray,
        reciprocal_lattice: np.ndarray,
        miller: np.ndarray,
        gamma_only: bool = False,
    ) -> "RestartCharge":
        """Forward FFT of a real-space density into ``rho(G)``.

        Args:
            rho: Real-space density, either ``(nx, ny, nz)`` or
                ``(nspin, nx, ny, nz)``, in ABACUS units (e/Bohr^3).
            reciprocal_lattice: (3, 3) row-major reciprocal matrix in the ABACUS
                convention (``inv(cell)``, without the ``2*pi`` factor).
            miller: ``(ngm, 3)`` Miller indices of the G-vectors to keep.
            gamma_only: Value stored in the file header. Only ``False`` is
                supported for the FFT conversion.

        Returns:
            A :class:`RestartCharge` with the sampled ``rho(G)``.
        """
        if gamma_only:
            raise NotImplementedError(
                "gamma_only restart files cannot be converted from real space yet"
            )
        rho = np.asarray(rho, dtype=float)
        if rho.ndim == 3:
            rho = rho[np.newaxis, :]
        if rho.ndim != 4:
            raise ValueError(
                f"rho must have shape (nx, ny, nz) or (nspin, nx, ny, nz), got {rho.shape}"
            )
        shape = rho.shape[1:]
        npoints = int(np.prod(shape))
        miller = np.asarray(miller, dtype=np.int64)
        rhog = np.empty((rho.shape[0], miller.shape[0]), dtype=np.complex128)
        for ispin in range(rho.shape[0]):
            full = np.fft.fftn(rho[ispin]) / npoints
            rhog[ispin] = _gather_rhog(full, miller, shape)
        return cls(rhog, miller, reciprocal_lattice, gamma_only=gamma_only)


class Grid:
    """The class for charge density and potential data
    
    Attributes:
        data (np.ndarray): 3D numpy array of the grid data (charge density or potential), shape (Nx, Ny, Nz)
        cell (np.ndarray): 3x3 numpy array representing the lattice vectors
        atom_positions (np.ndarray): Nx3 numpy array of atomic positions in Cartesian coordinates
        atom_types (Sequence[int]): atomic types (atomic numbers)
        atom_charges (Sequence[float]): the charge of each atom
        origin (np.ndarray): 1D numpy array of length 3 representing the origin of the grid in Cartesian coordinates
    """
    def __init__(self, 
                 data: np.ndarray,
                 cell: np.ndarray,
                 atom_positions: Optional[np.ndarray] = None,
                 atom_types: Optional[Union[np.ndarray, Sequence]] = None,
                 atom_charges: Optional[Union[np.ndarray, Sequence]] = None,
                 origin: Optional[np.ndarray] = np.zeros(3),
                 
                 ):
        assert data.ndim == 3, "Data should be a 3D numpy array"
        assert cell.shape == (3,3), "Cell should be a 3x3 numpy array"
        
        if atom_positions is not None:
            assert atom_positions.ndim == 2 and atom_positions.shape[1] == 3, "Atom positions should be a Nx3 numpy array"

            if atom_types is not None:
                assert len(atom_types) == atom_positions.shape[0], "Atom types should be a list of length N"
            else:
                atom_types = [0] * atom_positions.shape[0]  # Default type if not provided
                
            if atom_charges is not None:
                assert len(atom_charges) == atom_positions.shape[0], "Atom charges should be a list of length N"
            else:
                atom_charges = [0.0] * atom_positions.shape[0]  # Default charge if not provided
        else:
            atom_positions = np.empty((0, 3), dtype=float)
            atom_types = []
            atom_charges = []
                
        assert origin.shape == (3,), "Origin should be a 1D numpy array of length 3"
        
        self._data_dict = {
            'data': data,
            'cell': cell,
            'atom_positions': atom_positions,
            'atom_types': atom_types,
            'atom_charges': atom_charges,
            'origin': origin
        }
    
    @property
    def data(self):
        return self._data_dict['data']
    
    @data.setter
    def data(self, value: np.ndarray):
        assert value.shape == self._data_dict['data'].shape, "New data must have the same shape as the original data"
        self._data_dict['data'] = value

    @property
    def data_coord(self):
        """Get the coordinates of the grid points in Cartesian coordinates.
        
        Returns:
            np.ndarray: A (Nx*Ny*Nz, 3) array of grid point coordinates.
        """
        nx, ny, nz = self.data.shape
        x = np.linspace(0, 1, nx, endpoint=False)
        y = np.linspace(0, 1, ny, endpoint=False)
        z = np.linspace(0, 1, nz, endpoint=False)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        fractional_coords = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
        cartesian_coords = fractional_coords @ self.cell
        # reshape to (Nx, Ny, Nz, 3)
        cartesian_coords = cartesian_coords.reshape((nx, ny, nz, 3))
        return cartesian_coords
    
    @property
    def cell(self):
        return self._data_dict['cell']
    
    @property
    def atom_positions(self):
        return self._data_dict['atom_positions']
    
    @property
    def atom_types(self):
        return self._data_dict['atom_types']
    
    @property
    def atom_charges(self):
        return self._data_dict['atom_charges']
    
    @property
    def origin(self):
        return self._data_dict['origin']
    
    @property
    def volume(self):
        return np.abs(np.linalg.det(self.cell))
    
    def _save_cube(self, filename: str,
                  data_factor: float = 1.0,
                  box_factor: float = 1.0):
        """Save the grid data to a cube file.
        
        Args:
            filename (str): Path to the output cube file.
            data_factor (float): Factor to multiply the data values (e.g., to convert units).
            box_factor (float): Factor to multiply the cell vectors (e.g., to convert units
        """
        with open(filename, 'w') as f:
            f.write("CUBE FILE\n")
            f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
            f.write(f"{len(self.atom_types):5d} {self.origin[0] * box_factor:12.6f} {self.origin[1] * box_factor:12.6f} {self.origin[2] * box_factor:12.6f}\n")
            for i in range(3):
                f.write(f"{self.data.shape[i]:5d} {self.cell[i,0] * box_factor/self.data.shape[i]:12.6f} {self.cell[i,1] * box_factor/self.data.shape[i]:12.6f} {self.cell[i,2] * box_factor/self.data.shape[i]:12.6f}\n")
            for i in range(len(self.atom_types)):
                f.write(f"{self.atom_types[i]:5d} {self.atom_charges[i]:12.6f} {self.atom_positions[i,0] * box_factor:12.6f} {self.atom_positions[i,1] * box_factor:12.6f} {self.atom_positions[i,2] * box_factor:12.6f}\n")
            flat_data = self.data.flatten() * data_factor
            for i in range(0, len(flat_data), 6):
                line_data = flat_data[i:i+6] 
                f.write(" ".join(f"{x:17.11e}" for x in line_data) + "\n")
    
    def save_cube(self, filename: str):
        """Save the grid data to a cube file in its original units."""
        self._save_cube(filename, data_factor=1.0, box_factor=1.0)
    
    @staticmethod
    def from_cube(cube_file: str, 
                  data_factor: float = 1.0,
                  box_factor: float = 1.0,
                  ):
        """Create a Cube object from a cube file.
        
        Args:
            cube_file (str): Path to the cube file.
            data_factor (float): Factor to multiply the data values (e.g., to convert units).
            box_factor (float): Factor to multiply the cell vectors (e.g., to convert units).
        """
        assert os.path.isfile(cube_file), f"{cube_file} does not exist"
        
        with open(cube_file, 'r') as f:
            lines = f.readlines()

        natom, x_origin, y_origin, z_origin = map(float, lines[2].split())
        origin =[x_origin, y_origin, z_origin]

        grid_size = []
        cell = []
        for i in range(3):
            sl = lines[3 + i].split()
            grid_size.append(int(sl[0]))
            cell.append([float(x) * int(sl[0]) for x in sl[1:4]])

        atomz, chg, coords = [], [], []
        for line in lines[6:6+int(natom)]:
            atomz.append(int(line.split()[0]))
            chg.append(float(line.split()[1]))
            coords.append(list(map(float, line.split()[2:])))

        data = []
        for line in lines[6+int(natom):]:
            data.extend(list(map(float, line.split())))
        data = np.array(data) 
        data = data.reshape(grid_size)
        
        return Grid(data * data_factor, 
                    np.array(cell) * box_factor, 
                    np.array(coords) * box_factor, 
                    np.array(atomz), 
                    np.array(chg), 
                    np.array(origin) * box_factor)

    def save_npz(self, filename: str):
        """Save the data to a .npz file"""
        np.savez_compressed(filename, **self._data_dict)
    
    @classmethod
    def from_npz(cls, filename: str):
        """Load the grid data from a .npz file"""
        loaded = np.load(filename, allow_pickle=True)
        return cls(loaded['data'], loaded['cell'], loaded['atom_positions'], loaded['atom_types'], loaded['atom_charges'], loaded['origin'])
    
    @classmethod
    def from_stru(cls, stru_file: str, grid_size: Tuple[int, int, int]):
        """Create a Grid object from a STRU file.
        
        Args:
            stru_file (str): Path to the STRU file.
            grid_size (tuple): Grid size along each cell direction (nx, ny, nz).
        """
        stru = AbacusSTRU.read(stru_file)
        if stru is None:
            raise ValueError(f"Failed to read STRU file {stru_file}")
        
        cell = stru.cell
        coord = stru.coords
        element = [atomic_numbers[symbol] for symbol in stru.elements]
        
        return cls(np.zeros(grid_size), np.array(cell), np.array(coord), np.array(element), np.zeros(len(element)), np.zeros(3))

    def supercell(self, sc: Tuple[int, int, int]):
        """Create a supercell of the grid data.
        
        Args:
            sc (tuple): Supercell size in each dimension (sx, sy, sz).
        
        Returns:
            Grid: A new Grid object representing the supercell.
        """
        assert len(sc) == 3 and all(isinstance(x, int) and x > 0 for x in sc), "Supercell size should be a tuple of three positive integers"
        
        new_data = np.tile(self.data, sc)
        new_cell = self.cell * np.array(sc)[:, None]
        
        if self.atom_positions is not None and len(self.atom_positions) > 0:
            new_atom_positions = []
            new_atom_types = []
            new_atom_charges = []
            for i in range(sc[0]):
                for j in range(sc[1]):
                    for k in range(sc[2]):
                        shift = i * self.cell[0] + j * self.cell[1] + k * self.cell[2]
                        new_atom_positions.append(self.atom_positions + shift)
                        new_atom_types.extend(self.atom_types)
                        new_atom_charges.extend(self.atom_charges)
            new_atom_positions = np.vstack(new_atom_positions)
        else:
            new_atom_positions = np.zeros(3)
            new_atom_types = np.array([0])
            new_atom_charges = np.array([0.0])
        
        new_origin = self.origin
        
        return Grid(new_data, new_cell, new_atom_positions, new_atom_types, new_atom_charges, new_origin)
    
    def profile1d(self, axis: Literal['a', 'b', 'c'] = 'c', average: bool=False, cartesian: bool=False):
        """Integrate the 3D cube data to 2D plane.
        Args:
            axis (str): the axis to be integrated. 'a' means integrate bc plane, 'b' means ac plane, 'c' means ab plane.
            average (bool): whether to take the average of the integrated values. If False, the integrated values will be summed.
            cartesian (bool): whether to return the coordinates of profile in cartesian coordinates. If False, the profile will be returned in direct coordinates.
        
        Returns:
            tuple: A tuple containing the integrated values and the coordinates of the profile.
        """
        import numpy as np

        func = np.mean if average else np.sum
        if axis == "a":
            val = func(self.data, axis=2) # integrate along c
            val = func(val, axis=1) # integrate along b (axis 1 in integrated val)
        elif axis == "b":
            val = func(self.data, axis=0) # integrate along a
            val = func(val, axis=1) # integrate along c (axis 1 in integrated val)
        elif axis == "c":
            val = func(self.data, axis=0) # integrate along a
            val = func(val, axis=0) # integrate along b (axis 0 in integrated val)
        else:
            raise ValueError(f"Invalid axis: {axis} is not a, b or c")
        
        ngrid = self.data.shape[0] if axis == "a" else self.data.shape[1] if axis == "b" else self.data.shape[2]
        if cartesian:
            vec_length = np.linalg.norm(self.cell[0]) if axis == "a" else np.linalg.norm(self.cell[1]) if axis == "b" else np.linalg.norm(self.cell[2])
            coord = np.linspace(0, vec_length, ngrid)
        else:
            coord = np.linspace(0, 1, ngrid)

        return val, coord

class Charge(Grid):
    """Subclass for charge density data.
    
    By using from_cube method, the unit of charge density will be converted to e/Ang^3,
    and the unit of cell and positions will be converted to Angstrom.
    
    For example:
    >>> chg = Charge.from_cube("charge.cube")
    >>> print(chg.data.shape)
    (51, 51, 51)
    >>> print(chg.cell)
    [[ 5.00000000e+01  0.00000000e+00  0.00000000e+00]
     [ 0.00000000e+00  5.00000000e+01  0.00000000e+00]
     [ 0.00000000e+00  0.00000000e+00  5.00000000e+01]]
    >>> supercell = chg.supercell((2, 2, 2)) # create a supercell of size (2, 2, 2)
    >>> print(supercell.data.shape)
    (102, 102, 102)
    >>> supercell.save_cube("supercell.cube") # save the supercell to a cube file

    """
    def __init__(self, 
                 data: np.ndarray,
                 cell: np.ndarray,
                 atom_positions: Optional[np.ndarray] = None,
                 atom_types: Optional[Union[np.ndarray, Sequence]] = None,
                 atom_charges: Optional[Union[np.ndarray, Sequence]] = None,
                 origin: np.ndarray = np.zeros(3)
                 ):
        super().__init__(data, cell, atom_positions, atom_types, atom_charges, origin)

    @staticmethod        
    def from_cube(cube_file: str, format: str = "abacus"):
        """Load charge density from a cube file"""
        
        if format == "abacus":
            data_factor = 1 / BOHR2A**3  # ABACUS charge density is in e/Bohr^3, convert to e/Ang^3
            box_factor = BOHR2A  # ABACUS cell is in Bohr, convert to Angstrom
        else:
            raise ValueError(f"Unsupported format {format}. Supported formats are 'abacus'.")
            
        cube = Grid.from_cube(cube_file, data_factor, box_factor)
        return Charge(cube.data, cube.cell, cube.atom_positions, cube.atom_types, cube.atom_charges, cube.origin)
    
    def save_cube(self, filename, format: str = "abacus"):
        """Save the charge density to a cube file"""
        if format == "abacus":
            data_factor = BOHR2A**3  # Convert back to e/Bohr^3
            box_factor = 1 / BOHR2A  # Convert back to Bohr
            return self._save_cube(filename, data_factor, box_factor)
        else:   
            raise ValueError(f"Unsupported format {format}. Supported formats is 'abacus'.")
    
    def to_pot(self):
        """Solve the Poisson equation to get the electrostatic potential from the charge density.
        Returns:
            Potential: The electrostatic potential object.
        """
        pot = _charge_to_potential(self.data, self.cell)
        return Potential(pot, self.cell, self.atom_positions, self.atom_types, self.atom_charges, self.origin)  

    @classmethod
    def from_restart(
        cls,
        restart_file: str,
        grid_shape: Tuple[int, int, int],
        cell: Optional[np.ndarray] = None,
        atom_positions: Optional[np.ndarray] = None,
        atom_types: Optional[Union[np.ndarray, Sequence]] = None,
        atom_charges: Optional[Union[np.ndarray, Sequence]] = None,
        origin: Optional[np.ndarray] = None,
        spin: int = 0,
        lat0: float = ANG_TO_BOHR,
    ) -> "Charge":
        """Build a Charge from an ABACUS ``*-CHARGE-DENSITY.restart`` file.

        The real-space density is recovered with an inverse FFT and converted
        from ABACUS units (e/Bohr^3) to e/Ang^3. When ``cell`` is omitted it is
        reconstructed from the reciprocal matrix stored in the file; atomic
        information is not part of the restart format and stays empty unless
        supplied. ``lat0`` is the ABACUS ``LATTICE_CONSTANT`` in Bohr and is only
        needed for the reconstruction (the common 1.889726 Bohr makes the STRU
        vectors Angstrom).
        """
        restart = RestartCharge.read(restart_file)
        if spin < 0 or spin >= restart.nspin:
            raise IndexError(f"spin {spin} out of range for nspin={restart.nspin}")
        data = restart.to_real(grid_shape)[spin] / BOHR2A**3
        if cell is None:
            cell = np.linalg.inv(restart.reciprocal_lattice) * lat0 * BOHR2A
        if origin is None:
            origin = np.zeros(3)
        return cls(
            data,
            np.asarray(cell, dtype=float),
            atom_positions,
            None if atom_types is None else np.asarray(atom_types),
            None if atom_charges is None else np.asarray(atom_charges),
            origin,
        )

    def save_restart(
        self,
        filename: str,
        miller: np.ndarray,
        gamma_only: bool = False,
        lat0: float = ANG_TO_BOHR,
    ) -> None:
        """Write the charge density to an ABACUS ``*-CHARGE-DENSITY.restart`` file.

        Args:
            filename: Path of the file to create.
            miller: ``(ngm, 3)`` Miller indices of the plane waves to keep.
            gamma_only: Value stored in the file header. Only ``False`` supports
                the FFT conversion.
            lat0: ABACUS ``LATTICE_CONSTANT`` in Bohr used to build the stored
                reciprocal matrix (default matches a STRU with Angstrom vectors).
        """
        cell_bohr = np.asarray(self.cell, dtype=float) * ANG_TO_BOHR
        reciprocal_lattice = lat0 * np.linalg.inv(cell_bohr)
        rho_bohr = self.data * BOHR2A**3
        restart = RestartCharge.from_real(rho_bohr, reciprocal_lattice, miller, gamma_only=gamma_only)
        restart.write(filename)

    def supercell(self, sc: Tuple[int, int, int]):
        """Create a supercell of the charge density data.
        
        Args:
            sc (tuple): Supercell size in each dimension (sx, sy, sz).
        
        Returns:
            Charge: A new Charge object representing the supercell.
        """
        grid = super().supercell(sc)
        return Charge(grid.data, grid.cell, grid.atom_positions, grid.atom_types, grid.atom_charges, grid.origin)

class Potential(Grid):
    """Subclass for electrostatic potential data.
    
    By using from_cube method, the unit of potential will be converted to eV,
    and the unit of cell and positions will be converted to Angstrom.
    
    For example:
    >>> pot = Potential.from_cube("potential.cube")
    >>> print(pot.data.shape)
    (51, 51, 51)
    >>> supercell = pot.supercell((2, 2, 2))
    >>> print(supercell.data.shape)
    (102, 102, 102)
    >>> supercell.save_cube("supercell.cube")
    """

    def __init__(
        self,
        data: np.ndarray,
        cell: np.ndarray,
        atom_positions: Optional[np.ndarray] = None,
        atom_types: Optional[Union[np.ndarray, Sequence]] = None,
        atom_charges: Optional[Union[np.ndarray, Sequence]] = None,
        origin: np.ndarray = np.zeros(3),
    ):
        super().__init__(data, cell, atom_positions, atom_types, atom_charges, origin)

    @staticmethod
    def from_cube(cube_file: str, format: str = "abacus"):
        """Load potential from a cube file"""

        if format == "abacus":
            data_factor = (
                -1 * RY2EV
            )  # ABACUS potential is in Ry, convert to eV. In abacus the electron is positive, so we need a negative sign here.
            box_factor = BOHR2A  # ABACUS cell is in Bohr, convert to Angstrom
        else:
            raise ValueError(f"Unsupported format {format}. Supported formats is 'abacus'.")
        
        cube = Grid.from_cube(cube_file, data_factor, box_factor)
        return Potential(
            cube.data,
            cube.cell,
            cube.atom_positions,
            cube.atom_types,
            cube.atom_charges,
            cube.origin,
        )

    @staticmethod
    def from_locpot(locpot_file: str):
        """Read the local potential from a LOCPOT file written by VASP"""
        from ase.io import read
        import uuid

        with open(locpot_file, "r") as f:
            lines = [line.strip() for line in f if line.strip() != "" or line == "\n"]

        # Get total number of atoms
        atom_nums = [int(x) for x in lines[6].split()]
        total_atom_nums = sum(atom_nums)

        # Dump POSCAR file in the head of LOCPOT
        end_line_idx = total_atom_nums + 7
        if lines[7].lower == "selective dynamics":
            end_line_idx += 1

        poscar_dump = "_dumped_POSCAR"
        while os.path.exists(poscar_dump):
            poscar_dump = "_dumped_POSCAR_" + str(uuid.uuid4())[:8]

        with open(poscar_dump, "w") as f:
            for line_idx in range(end_line_idx + 1):
                f.write(lines[line_idx] + "\n")

        pos = read(poscar_dump, format="vasp")
        os.unlink(poscar_dump)

        # Read size of gird data
        line_idx = end_line_idx + 1
        grid = tuple(int(x) for x in lines[line_idx].split())
        nx, ny, nz = grid
        total_grid_points = nx * ny * nz
        line_idx += 1

        def read_grid_data(start_idx: int, n_points: int) -> np.ndarray:
            values = []
            idx = start_idx
            while len(values) < n_points:
                if idx >= len(lines):
                    raise RuntimeError("No sufficient data in LOCPOT file")
                values.extend([float(x) for x in lines[idx].split()])
                idx += 1
            if len(values) != n_points:
                raise RuntimeError( f"expected {n_points} data points, read {len(values)} data points")
            # reshape data - VASP stores data in z, y, x order (fastest to slowest: x, y, z)
            return (np.array(values).reshape((nz, ny, nx)).transpose(2, 1, 0))  # Convert to (nx, ny, nz)

        data_first = read_grid_data(line_idx, total_grid_points)
        line_idx += int(np.ceil(total_grid_points / 5))

        # check if there are any remaining lines
        remaining_lines = len(lines) - line_idx
        min_spin_lines = (1 + int(np.ceil(total_atom_nums / 5)) + int(np.ceil(total_grid_points / 5)))
        is_spin_polarized = remaining_lines >= min_spin_lines

        if is_spin_polarized:
            # skip atom count lines of "1" (5 per line)
            spin_marker_lines = int(np.ceil(total_atom_nums / 5))
            line_idx += spin_marker_lines

            grid2 = tuple(int(x) for x in lines[line_idx].split())
            if grid2 != grid:
                raise Warning(f"Mesh size of second data is {grid2}, not same with first data ({grid})")
            line_idx += 1

            data_second = read_grid_data(line_idx, total_grid_points)
            data = data_first + data_second
        else:
            data = data_first

        # Convert to potential of electrons
        data *= -1

        # Create atom_charges (default to 0) and origin (default to [0, 0, 0])
        atom_charges = np.array([0.0] * total_atom_nums)
        origin = np.zeros(3)

        return Potential(data, np.array(pos.get_cell()), pos.get_positions(), pos.get_atomic_numbers(), atom_charges, origin)

    def save_cube(self, filename, format: str = "abacus"):
        """Save the potential to a cube file"""
        if format == "abacus":
            data_factor = -1 / RY2EV  # Convert back to Ry
            box_factor = 1 / BOHR2A  # Convert back to Bohr
            return self._save_cube(filename, data_factor, box_factor)
        else:   
            raise ValueError(f"Unsupported format {format}. Supported formats is 'abacus'.")
        
    def supercell(self, sc: Tuple[int, int, int]):
        """Create a supercell of the potential data.
        
        Args:
            sc (tuple): Supercell size in each dimension (sx, sy, sz).
        
        Returns:
            Potential: A new Potential object representing the supercell.
        """
        grid = super().supercell(sc)
        return Potential(grid.data, grid.cell, grid.atom_positions, grid.atom_types, grid.atom_charges, grid.origin)
