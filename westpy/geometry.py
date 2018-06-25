from __future__ import print_function

class Geometry(object) :
   """Class for representing a set of atoms in a periodic cell.
   
   :Example:

   >>> from westpy import * 
   >>> geom = Geometry()
   >>> geom.setCell( (1,0,0), (0,1,0), (0,0,1) ) 
   >>> geom.addAtom( "Si", (0,0,0) ) 
   >>> geom.addSpecies( "Si", "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.1.upf" ) 

   .. note:: Vectors are set in a.u. by default. If you set units=Angstrom a coversion to a.u. will be made.  
 
   """
   #
   from westpy import Bohr 
   #
   def __init__(self,cell=None) : 
      self.cell = {}
      self.atoms = []
      self.species = {}
      self.isSet = {}
      self.isSet["cell"] = False
      self.isSet["atoms"] = False
      self.isSet["species"] = False
      self.pseudoFormat = None 
   #
   def __atomsMatchSpecies(self) :
      """Checks if atoms match species"""
      matches = self.isSet["species"] and self.isSet["atoms"]  
      for atom in self.atoms :
         matches = matches and atom.symbol in self.species.keys()
      return matches
   #
   def isValid(self) :
      """Checks if geometry is valid

      The method checks that: 
         - the cell is set
         - at least one atom has been added
         - the pseudopotentials of all species are defined 
         - the pseudopotentials do not contain a mix of upf and xml formats
      """
      isValid = True
      for key in self.isSet.keys() : 
         isValid = isValid and self.isSet[key]
         if( not self.isSet[key] ) : 
            print("ERR: set "+key)
      isValid = isValid and self.__atomsMatchSpecies()
      isValid = isValid and self.pseudoFormat in ["upf","xml"]
      return isValid
   #
   def addSpecies(self, symbol, url) : 
      """Adds a species.
   
      :param symbol: chemical symbol 
      :type symbol: string
      :param url: url 
      :type url: string
   
      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.addSpecies( "Si", "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.1.upf" ) 
      
      .. note:: You can use this method to add either upf or xml pseudopotentials. However it is forbidded to mix them.  
      """
      from westpy import extractFileNamefromUrl
      fname = extractFileNamefromUrl(url)
      this_pseudo_format = None
      if( fname.endswith("upf") or fname.endswith("UPF")) : 
         this_pseudo_format = "upf"
      if( fname.endswith("xml") or fname.endswith("XML")) : 
         this_pseudo_format = "xml"
      assert( this_pseudo_format in ["upf","xml"] )
      from mendeleev import element
      el = element(symbol)
      self.species[symbol] = {}
      self.species[symbol]["fname"] = fname
      self.species[symbol]["url"] = url
      self.species[symbol]["symbol"] = el.symbol
      self.species[symbol]["atomic_number"] = el.atomic_number
      self.species[symbol]["name"] = el.name
      self.species[symbol]["mass"] = el.mass
      self.species[symbol]["format"] = this_pseudo_format
      self.__updatePseudoFormat()
      self.isSet["species"] = True
   #
   def __updatePseudoFormat(self) : 
      """pseudo format is either upf, xml or mixed.
      """
      f = []
      for key in self.species.keys() :
         f.append(self.species[key]["format"])
      self.pseudoFormat = 'mixed'
      if( all(form == 'upf' for form in f) ) : 
         self.pseudoFormat = 'upf'
      if( all(form == 'xml' for form in f) ) : 
         self.pseudoFormat = 'xml'
   #
   def setCell(self, a1=(0, 0, 0), a2=(0, 0, 0), a3=(0, 0, 0), units=Bohr ) : 
      """Sets cell, given the three vectors :math:`a_1`, :math:`a_2`, :math:`a_3`.
   
      :param a1: :math:`a_1`
      :type a1: 3-dim tuple
      :param a2: :math:`a_2`
      :type a2: 3-dim tuple
      :param a3: :math:`a_3`
      :type a3: 3-dim tuple
      :param units: Units, optional  
      :type units: "Bohr" or "Angstrom"
   
      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.setCell( (1,0,0), (0,1,0), (0,0,1) ) 
      """
      import numpy as np
      # 
      self.cell["a1"] = np.array(a1, float) * units 
      self.cell["a2"] = np.array(a2, float) * units 
      self.cell["a3"] = np.array(a3, float) * units 
      self.cell["volume"] = np.dot(self.cell["a1"], np.cross(self.cell["a2"], self.cell["a3"]))
      self.cell["b1"] = 2.0 * np.pi * np.cross(self.cell["a2"], self.cell["a3"]) / self.cell["volume"]
      self.cell["b2"] = 2.0 * np.pi * np.cross(self.cell["a3"], self.cell["a1"]) / self.cell["volume"]
      self.cell["b3"] = 2.0 * np.pi * np.cross(self.cell["a1"], self.cell["a2"]) / self.cell["volume"]
      self.isSet["cell"] = True
   #
   def addAtom(self, symbol, position, units=Bohr) : 
      """Adds a single atom.
   
      :param symbol: chemical symbol 
      :type symbol: string
      :param position: position
      :type position: 3-dim tuple 
      :param units: Units, optional  
      :type units: "Bohr" or "Angstrom" 
   
      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.addAtom( "Si", (0,0,0) ) 
      """
      from westpy import Atom 
      self.atoms.append( Atom(symbol, position, units) )
      self.isSet["atoms"] = True
   #
   def __addAtomsFromXYZLines(self, lines ) : 
      """Adds atoms from XYZ lines.
   
      :param lines: lines read from XYZ file (only one image)
      :type lines: list of string
      """
      #
      from westpy import Angstrom 
      natoms = int(lines[0])
      for line in lines[2:2+natoms] : 
         symbol, x, y, z = line.split()[:4]
         self.addAtom( symbol.decode("utf-8"), (float(x), float(y), float(z)), units=Angstrom )
   #
   def addAtomsFromXYZFile(self, fname ) : 
      """Adds atoms from XYZ file (only one image).
   
      :param fname: file name
      :type fname: string
   
      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.addAtomFromXYZFile( "CH4.xyz" ) 
      """
      # 
      with open(fname,'r') as file: 
         lines = file.readlines()
      self.__addAtomsFromXYZLines(lines) 
   #
   def addAtomsFromOnlineXYZ(self, url ) : 
      """Adds atoms from XYZ file (only one image) located at url.
   
      :param url: url
      :type url: string

      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.addAtomsFromOnlineXYZ( "http://www.west-code.org/database/gw100/xyz/CH4.xyz" ) 
      """
      # 
      import urllib.request
      with urllib.request.urlopen(url) as response :
         lines = response.readlines()
      self.__addAtomsFromXYZLines(lines)
   #
   def getNumberOfAtoms(self) : 
      """Returns number of atoms.
   
      :returns: number of atoms 
      :rtype: int

      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.addAtomsFromOnlineXYZ( "http://www.west-code.org/database/gw100/xyz/CH4.xyz" )
      >>> nat = geom.getNumberOfAtoms()
      >>> print( nat ) 
      5 
      """
      nat = len(self.atoms)
      return nat
   #
   def getNumberOfSpecies(self) : 
      """Returns number of species.
   
      :returns: number of species
      :rtype: int

      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.addAtomsFromOnlineXYZ( "http://www.west-code.org/database/gw100/xyz/CH4.xyz" )
      >>> ntyp = geom.getNumberOfSpecies()
      >>> print( ntyp ) 
      2 
      """
      sp = []
      for atom in self.atoms : 
         if atom.symbol not in sp : 
            sp.append(atom.symbol)
      ntyp = len(sp)
      return ntyp
   #
   def getNumberOfElectrons(self) : 
      """Returns number of electrons.
   
      :returns: number of electrons 
      :rtype: int

      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.addAtomsFromOnlineXYZ( "http://www.west-code.org/database/gw100/xyz/CH4.xyz" )
      >>> geom.addSpecies( "C", "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/C_ONCV_PBE-1.0.xml")
      >>> geom.addSpecies( "H", "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/H_ONCV_PBE-1.0.xml")
      >>> nelec = geom.getNumberOfElectrons()
      >>> print( nelec ) 
      8
      """
      assert( self.__atomsMatchSpecies() )
      #
      nelec = 0 
      if self.pseudoFormat in ["upf"] : 
         from westpy import listLinesWithKeyfromOnlineText
         for atom in self.atoms : 
             symbol = atom.symbol
             url = self.species[symbol]["url"]
             resp = listLinesWithKeyfromOnlineText(url,"z_valence")[0]
             this_valence = float(resp.decode("utf-8").split('"')[1]) 
             nelec += this_valence
      if self.pseudoFormat in ["xml"] :
         from westpy import listValuesWithKeyFromOnlineXML
         for atom in self.atoms : 
             symbol = atom.symbol
             url = self.species[symbol]["url"]
             this_valence = float(listValuesWithKeyFromOnlineXML(url,"valence_charge")[0])
             nelec += this_valence
      return int(nelec)
   #
   def downloadPseudopotentials(self) : 
      """Download Pseudopotentials.
   
      :Example:

      >>> from westpy import * 
      >>> geom = Geometry()
      >>> geom.addAtomsFromOnlineXYZ( "http://www.west-code.org/database/gw100/xyz/CH4.xyz" )
      >>> geom.addSpecies( "C", "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/C_ONCV_PBE-1.0.xml")
      >>> geom.addSpecies( "H", "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/H_ONCV_PBE-1.0.xml")
      >>> geom.downloadPseudopotentials()
     
      .. note:: Pseudopotential files will be downloaded in the current directory. 
      """
      assert( self.__atomsMatchSpecies() )
      #
      from westpy import download 
      for key in self.species.keys() :
         download( self.species[key]["url"], fname=self.species[key]["fname"] )
