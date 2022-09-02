import numpy as np
from six import string_types
from westpy import Angstrom, Atom


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
