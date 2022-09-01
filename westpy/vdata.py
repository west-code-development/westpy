import numpy as np
from six import string_types
from copy import deepcopy
from pyscf.fci import cistring
from westpy import Angstrom, Cell


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
