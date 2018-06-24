from __future__ import print_function

"""Westpy uses Hartree atomic units."""

class Units(dict):
    #
    """Dictionary for units that supports .attribute access."""
    def __init__(self, *args, **kwargs):
        super(Units, self).__init__(*args, **kwargs)
        self.__dict__ = self

def set_units() : 
   #
   """Sets Rydberg atomic units.

   Available units are: 
      - Bohr
      - Angstrom, Ang
      - Rydberg, Ry
      - Hartree, Ha
      - eV
   
   .. note:: **westpy** operates in Rydberg atomic units.
   """   
   #
   AU = Units()
   #
   import scipy.constants as sp
   #
   #sp.c           # speed of light in vacuum
   #sp.mu_0        # the magnetic constant μ0
   #sp.epsilon_0   # the electric constant (vacuum permittivity), ϵ0
   #sp.h           # the Planck constant h
   #sp.hbar        # ℏ=h/(2π)
   #sp.G           # Newtonian constant of gravitation
   #sp.g           # standard acceleration of gravity
   #sp.e           # elementary charge
   #sp.R           # molar gas constant
   #sp.alpha       # fine-structure constant
   #sp.N_A         # Avogadro constant
   #sp.k           # Boltzmann constant
   #sp.sigma       # Stefan-Boltzmann constant σ
   #sp.Wien        # Wien displacement law constant
   #sp.Rydberg     # Rydberg constant
   #sp.m_e         # electron mass
   #sp.m_p         # proton mass
   #sp.m_n         # neutron mass
   #
   AU['Bohr']      = 1.0 
   AU['Angstrom']  = 1e-10 / 4.0 / sp.pi / sp.epsilon_0 / sp.hbar**2 * sp.m_e * sp.e**2
   AU['Ang']        = AU['Angstrom']
   AU['Rydberg']   = 1.0
   AU['Ry']        = AU['Rydberg']
   AU['Hartree']   = 2.0 * AU['Rydberg']
   AU['Ha']        = AU['Hartree']
   AU['eV']        = sp.e / sp.m_e / sp.e**4 * ( 4.0 * sp.pi * sp.epsilon_0 * sp.hbar )**2 / 2.0  
   #
   return AU

globals().update(set_units()) 
