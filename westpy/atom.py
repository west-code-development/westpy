from __future__ import print_function

class Atom(object):
   """Class for representing a single atom.

   :param symbol: chemical symbol 
   :type symbol: string
   :param position: position
   :type position: 3-dim tuple 
   :param units: Units, optional  
   :type units: "Bohr" or "Angstrom" 
   
   :Example:

   >>> from westpy import * 
   >>> atom = Atom("Si",(0.,0.,0.))

   .. note:: Positions are set in a.u. by default. If you set units=Angstrom a coversion to a.u. will be made.  
 
   """
   from westpy.units import Bohr 
   #
   def __init__(self, symbol='X', position=(0, 0, 0), units=Bohr) :
      #
      from mendeleev import element
      import numpy as np
      #
      el = element(symbol)
      #
      self.symbol   = el.symbol 
      self.position = np.array(position, float) * units
