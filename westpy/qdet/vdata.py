import re
import numpy as np
from six import string_types
from copy import deepcopy
from pyscf.fci import cistring
from westpy import Angstrom


class Atom(object):
    """An atom in a specific cell.

    An atom can be initialized by and exported to an ASE Atom object.

    All internal quantities below are in atomic unit.

    Attributes:
        cell(Cell): the cell where the atom lives.
        symbol(str): chemical symbol of the atom.
        abs_coord(float array): absolute coordinate.
        cry_coord(float array): crystal coordinate.

    Extra attributes are welcomed to be attached to an atom.
    """

    _extra_attr_to_print = [
        "velocity",
        "force",
        "freezed",
        "ghost",
        "Aiso",
        "Adip",
        "V",
    ]

    def __init__(
        self, cell, ase_atom=None, symbol=None, cry_coord=None, abs_coord=None, **kwargs
    ):
        # assert isinstance(cell, Cell)
        self.cell = cell

        if ase_atom is not None:
            from ase import Atom as ASEAtom

            assert isinstance(ase_atom, ASEAtom)
            self.symbol = ase_atom.symbol
            self.abs_coord = ase_atom.position * Angstrom
        else:
            assert isinstance(symbol, string_types)
            assert bool(cry_coord is None) != bool(abs_coord is None)
            self.symbol = symbol
            if cry_coord is not None:
                self.cry_coord = np.array(cry_coord)
            else:
                self.abs_coord = np.array(abs_coord)

        if self.cell.isperiodic:
            for i in range(3):
                if not 0 <= self.cry_coord[i] < 1:
                    self.cry_coord[i] = self.cry_coord[i] % 1

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def cry_coord(self):
        if self.cell.isperiodic:
            return self.abs_coord @ self.cell.G.T / (2 * np.pi)

    @cry_coord.setter
    def cry_coord(self, cry_coord):
        if self.cell.isperiodic:
            self.abs_coord = cry_coord @ self.cell.R
        else:
            raise ValueError("Crystal coordinate not defined for non-periodic system.")


class Cell(object):
    """A cell that is defined by lattice constants and contains a list of atoms.

    A cell can be initialized by and exported to an ASE Atoms object.
    A cell can also be exported to several useful formats such as Qbox and Quantum Espresso formats.

    All internal quantities below are in atomic unit.

    Attributes:
        R (float 3x3 array): primitive lattice vectors (each row is one vector).
        G (float 3x3 array): reciprocal lattice vectors (each row is one vector).
        omega(float): cell volume.
        atoms(list of Atoms): list of atoms.
    """

    def __init__(self, ase_cell=None, R=None):
        """
        Args:
            ase_cell (ASE Atoms or str): input ASE cell, or name of a (ASE-supported) file.
            R (float array): real space lattice constants, used to construct an empty cell.
        """

        if ase_cell is None:
            self.update_lattice(R)
            self._atoms = list()
        else:
            if isinstance(ase_cell, string_types):
                from ase.io import read

                ase_cell = read(ase_cell)
            else:
                from ase import Atoms

                assert isinstance(ase_cell, Atoms)

            lattice = ase_cell.get_cell()
            if np.all(lattice == np.zeros([3, 3])):
                self.update_lattice(None)
            else:
                self.update_lattice(lattice * Angstrom)

            self._atoms = list(Atom(cell=self, ase_atom=atom) for atom in ase_cell)

        self.distance_matrix = None

    def update_lattice(self, R):
        if R is None:
            self._R = self._G = self._omega = None
        else:
            if isinstance(R, int) or isinstance(R, float):
                self._R = np.eye(3) * R
            else:
                assert R.shape == (3, 3)
                self._R = R.copy()
            self._G = 2 * np.pi * np.linalg.inv(self._R).T
            assert np.all(np.isclose(np.dot(self._R, self._G.T), 2 * np.pi * np.eye(3)))
            self._omega = np.linalg.det(self._R)

    @property
    def isperiodic(self):
        return not bool(self.R is None)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        if self.isperiodic:
            cry_coords = [a.cry_coord for a in self.atoms]

        self.update_lattice(R)

        if self.isperiodic:
            for i, atom in enumerate(self.atoms):
                atom.cry_coord = cry_coords[i]

    @property
    def G(self):
        return self._G

    @property
    def omega(self):
        return self._omega

    @property
    def atoms(self):
        return self._atoms

    @property
    def natoms(self):
        return len(self.atoms)

    @property
    def species(self):
        return sorted(set([atom.symbol for atom in self.atoms]))

    @property
    def nspecies(self):
        return len(self.species)


class VData:
    def __init__(
        self,
        filename=None,
        cell=None,
        nx=None,
        ny=None,
        nz=None,
        data=None,
        cmplx=False,
        funct=None,
        normalize=False,
    ):

        self.name = "vdata"
        self.comments = None

        if filename is not None:
            from ase.io.cube import read_cube_data

            # read from file (currently only cube file format is supported)
            assert isinstance(filename, string_types)

            self.name = filename
            self.data, ase_cell = read_cube_data(filename)
            self.cell = Cell(ase_cell)
            self.nx, self.ny, self.nz = self.data.shape

            content = open(filename, "r").readlines()
            self.comments = content[0].strip() + " // " + content[1].strip()

            self.dx = float(content[3].split()[1]) / Angstrom
            self.dy = float(content[4].split()[2]) / Angstrom
            self.dz = float(content[5].split()[3]) / Angstrom

            # currently only positive values (bohr) are supported
            assert all(d > 0 for d in [self.dx, self.dy, self.dz])

        elif cell is not None:
            # build VData from scratch
            assert isinstance(cell, Cell)
            self.cell = deepcopy(cell)

            if data is not None:
                # numerical data is given (nx*ny*nz array)
                assert isinstance(data, np.ndarray) and data.ndim == 3
                if np.iscomplexobj(data):
                    if cmplx:
                        self.data = data.copy()
                    else:
                        print("only real part is loaded")
                        self.data = np.real(data)
                else:
                    self.data = data.copy()
                self.nx, self.ny, self.nz = self.data.shape

                self.dx = self.cell.R[(0, 0)] / self.nx / Angstrom
                self.dy = self.cell.R[(0, 0)] / self.ny / Angstrom
                self.dz = self.cell.R[(0, 0)] / self.nz / Angstrom

            elif funct is not None:
                # analytical expression is given as function
                # nx, ny and nz need to be specified, VData will be computed by function
                assert callable(funct)
                assert all(isinstance(n, int) for n in [nx, ny, nz])
                self.nx = nx
                self.ny = ny
                self.nz = nz
                self.data = np.zeros([nx, ny, nz])

                for ix in range(self.nx):
                    for iy in range(self.ny):
                        for iz in range(self.nz):
                            rx = ix * self.dx
                            ry = iy * self.dy
                            rz = iz * self.dz
                            self.data[(nx, ny, nz)] = funct(rx, ry, rz)

            else:
                raise ValueError

        else:
            raise ValueError

        self.nxyz = self.nx * self.ny * self.nz

        self.normalize(normalize)

    def normalize(self, do_normalize):
        if do_normalize == "sqrt":
            self.data = np.sign(self.data) * np.sqrt(np.abs(self.data))
        if do_normalize:
            self.data /= np.sqrt(self.cell.omega * np.sum(self.data**2) / self.nxyz)
