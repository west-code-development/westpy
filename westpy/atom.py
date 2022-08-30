import numpy as np
from six import string_types
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

    Extra attributes can be attached to an atom.
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
        self,
        cell=None,
        ase_atom=None,
        symbol=None,
        cry_coord=None,
        abs_coord=None,
        **kwargs
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
        if self.cell is not None:
            if self.cell.isperiodic:
                for i in range(3):
                    if not 0 <= self.cry_coord[i] < 1:
                        self.cry_coord[i] = self.cry_coord[i] % 1

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def cry_coord(self):
        if self.cell is not None:
            if self.cell.isperiodic:
                return self.abs_coord @ self.cell.G.T / (2 * np.pi)

    @cry_coord.setter
    def cry_coord(self, cry_coord):
        if sel.cell is not None:
            if self.cell.isperiodic:
                self.abs_coord = cry_coord @ self.cell.R
            else:
                raise ValueError(
                    "Crystal coordinate not defined for non-periodic system."
                )
