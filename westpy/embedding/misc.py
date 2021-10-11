import re
import numpy as np
from six import string_types
from copy import deepcopy


ev_to_hartree = 3.67494E-02
ev_to_rydberg = 7.34989E-02
hartree_to_ev = 2.72113E+01
rydberg_to_hartree = 5.00000E-01
angstrom_to_bohr = 1.88973E+00
bohr_to_angstrom = 5.29177E-01


def find_indices(string, content, start=0, case_sensitive=True):
  """Find indices (line #s) of given string that appeared in content

  :param string: string to be found
  :param content: a list of string (i.e. from readlines())
  :param start: starting index (line #)
  :param case_sensitive: self-explanatory
  :return: a list of integers
  """

  indices = []
  for i in range(start, len(content)):
    if case_sensitive:
      if string in content[i]:
        indices.append(i)
    else:
      if string.lower() in content[i].lower():
        indices.append(i)
  return indices


def find_index(string, content, start=0, case_sensitive=True):
  """Find the first index of given string that appeared in content

  :param string: string to be found
  :param content: a list of string (i.e. from readlines())
  :param start: starting index (line #)
  :param case_sensitive: self-explanatory
  :return: first occurrence of string
  """

  indices = find_indices(string, content, start, case_sensitive)
  if indices:
    return indices[0]
  else:
    return None


def regex(dtype):
  """Returns the regular expression required by re package

  :param dtype: int, float or str
  :return: string of regular expression
  """

  if dtype is int:
    return r"-*\d+"
  elif dtype is float:
    return r"-*\d+\.\d*[DeEe]*[+-]*\d*"
  elif dtype is str:
    return r".*"
  else:
    raise ValueError("unsupported type")


def parse_one_value(dtype, content, index=0):
  """Parse one value of type dtype from content

  :param dtype: type of value wanted
  :param content: a string to be parsed
  :param index: index of parsed value
  :return: first (if index not specified) value found in content
  """

  results = re.findall(regex(dtype), content)
  if results:
    return dtype(results[index])

def parse_many_values(n, dtype, content):
  """Parse n values of type dtype from content

  :param n: # of values wanted
  :param dtype: type of values wanted
  :param content: a string or a list of strings,
      it is assumed that n values exist in continues
      lines of content starting from the first line
  :return: a list of n values
  """

  if isinstance(content, string_types) or isinstance(content, np.string_):
    results = re.findall(regex(dtype), content)
    return [dtype(value) for value in results[0:n]]

  results = list()
  started = False
  for i in range(len(content)):
    found = re.findall(regex(dtype), content[i])
    if found:
      started = True
    else:
      if started:
        raise ValueError("cannot parse {} {} variables in content {}".format(
          n, dtype, content
        ))
    results.extend(found)
    assert len(results) <= n
    if len(results) == n:
      return [dtype(result) for result in results]

# Some functions in the following classes require ASE to execute. The import commands are moved within the functions
# to avoid making ASE a global dependency.

class Atom(object):
  """ An atom in a specific cell.

  An atom can be initialized by and exported to an ASE Atom object.

  All internal quantities below are in atomic unit.

  Attributes:
      cell(Cell): the cell where the atom lives.
      symbol(str): chemical symbol of the atom.
      abs_coord(float array): absolute coordinate.
      cry_coord(float array): crystal coordinate.

  Extra attributes are welcomed to be attached to an atom.
  """

  _extra_attr_to_print = ["velocity", "force", "freezed", "ghost",
                          "Aiso", "Adip", "V"]

  def __init__(self, cell, ase_atom=None, symbol=None, cry_coord=None, abs_coord=None, **kwargs):
    # assert isinstance(cell, Cell)
    self.cell = cell

    if ase_atom is not None:
      from ase import Atom as ASEAtom
      assert isinstance(ase_atom, ASEAtom)
      self.symbol = ase_atom.symbol
      self.abs_coord = ase_atom.position * angstrom_to_bohr
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
  """ A cell that is defined by lattice constants and contains a list of atoms.

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
        self.update_lattice(lattice * angstrom_to_bohr)

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
    def __init__(self, filename=None, cell=None, nx=None, ny=None, nz=None, data=None,
                 cmplx=False, funct=None, normalize=False):

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

            self.dx = float(content[3].split()[1]) * bohr_to_angstrom
            self.dy = float(content[4].split()[2]) * bohr_to_angstrom
            self.dz = float(content[5].split()[3]) * bohr_to_angstrom

            # currently only positive values (bohr) are supported
            assert (all(d > 0 for d in [self.dx, self.dy, self.dz]))

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

                self.dx = self.cell.R[(0, 0)] / self.nx * bohr_to_angstrom
                self.dy = self.cell.R[(0, 0)] / self.ny * bohr_to_angstrom
                self.dz = self.cell.R[(0, 0)] / self.nz * bohr_to_angstrom

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
            self.data /= np.sqrt(self.cell.omega * np.sum(self.data ** 2) / self.nxyz)

class Wavefunction:
    def __init__(self, file, volume, norm = 'qe', ordering = 'xyz'):
        self.filename = file
        self.file = open(file, 'r').readlines()
        self.volume = volume
        self.norm = norm
        self.ordering = ordering
        self.natoms = int(self.file[2].split()[0])
        
        self.npoints = []
        self.delta = np.zeros((3,3))
        # npoints store the number of grid points along each crystal axis
        # delta stores the incremental vectors that define a volume element
        for i in range(3):
            self.npoints.append(int(self.file[3+i].split()[0]))
            self.delta[:,i] = [float(entry) for entry in self.file[3+i].split()[1:]]
        self.npoints = np.asarray(self.npoints)
        
        self.read_data(volume, ordering, norm)
    
    def __copy__(self):
      return type(self)(self.filename, self.volume, self.norm, self.ordering)
        
    def read_data(self, volume, ordering='xyz', norm='qe'):
        offset = self.natoms+6
        self.data = np.zeros(self.npoints)
        # generate 1D list of data-points, irrespective of how the cube file is
        # structure
        data_ = []
        for line in range(self.natoms+6,len(self.file)):
            for entry in self.file[line].strip().split():
                data_.append(float(entry))
        # sort 1D data to 3D array, assuming x-y-z ordering
        index = 0
        if ordering == 'xyz':
            for i in range(self.npoints[0]):
                for j in range(self.npoints[1]):
                    for k in range(self.npoints[2]):
                        self.data[i,j,k] = np.sqrt(np.abs(data_[index]))*np.sign(data_[index])
                        index += 1
        elif ordering == 'zyx':
            for k in range(self.npoints[0]):
                for j in range(self.npoints[1]):
                    for i in range(self.npoints[2]):
                        self.data[i,j,k] = np.sqrt(np.abs(data_[index]))*np.sign(data_[index])
                        index += 1
        else:
            print("Unknown ordering:", ordering)
        # set prefactor depending on what code produced the data:
        if norm == 'qe':
            self.data[:,:,:] = self.data[:,:,:] * np.sqrt( volume/(self.npoints[0]* self.npoints[1]*self.npoints[2]))
        elif norm == 'west':
            self.data[:,:,:] = self.data[:,:,:] * np.sqrt( 1.0 /(self.npoints[0]* self.npoints[1]*self.npoints[2]))

    def integrate(self, box=None):
        if box == None:
        # integrate the wavefunction over the whole unit cell (should be 1)
            return np.dot(self.data.flatten(), self.data.flatten())
        else:
        # integrate over specified box
            data_= []
            count = 0
            # generate 3D grid in the unit cell from self.npoints
            grid=[np.linspace(0, 1, num = self.npoints[0], endpoint = False),
             np.linspace(0, 1, num = self.npoints[1], endpoint = False),
             np.linspace(0, 1, num = self.npoints[2], endpoint = False)]
            
            boundary = np.zeros((3,2),dtype= int)
            # loop over 3 dimensions
            for comp in range(3):
                count = 0
                # find the first grid point that lies within the box 
                for i in range(self.npoints[comp]):
                    if grid[comp][i] >= box[comp][0]:
                        break
                    count += 1
                boundary[comp,0] = count
                # find the last grid point that lies within the box 
                for i in range(boundary[comp,0], self.npoints[comp]):
                    if grid[comp][i] > box[comp][1]:
                        break
                    count += 1
                boundary[comp,1] = count
            # compute integral
            slice_ = self.data[boundary[0,0]:boundary[0,1],boundary[1,0]:boundary[1,1],boundary[2,0]:boundary[2,1]]
            integral_ = np.dot(slice_.flatten(), slice_.flatten())

            return integral_

    def ipr(self):
        # generate absolute square of wavefunction
        inter = np.array([entry**2 for entry in self.data.flatten()])
        ipr = np.dot( inter, inter)

        return ipr
