from __future__ import print_function
import requests
import json
from ast import literal_eval

class GroundState() :
   """Class for representing a ground state calculation with DFT.

   :param geom: geometry (cell, atoms, species)
   :type geom: Class(Geometry)
   :param xc: exchange-correlation functional 
   :type xc: string 
   :param ecut: energy cutoff for the wavefunction (in Rydberg units)
   :type ecut: float 
   
   :Example:

   >>> from westpy import * 
   >>> geom = Geometry() 
   >>> geom.setCell( (1,0,0), (0,1,0), (0,0,1) ) 
   >>> geom.addAtom( "Si", (0,0,0) ) 
   >>> geom.addSpecies( "Si", "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.1.upf" ) 
   >>> gs = GroundState(geom,"PBE",30.0)

   .. note:: Vectors are set in a.u. by default. If you set units=Angstrom a coversion to a.u. will be made.  
 
   """
   #
   def __init__(self,geom,xc,ecut) :
      from westpy import Geometry
      assert( isinstance(geom,Geometry) )
      assert( geom.isValid() )
      self.geom = geom
      self.xc = xc
      self.ecut = ecut
      self.nempty = 0
      self.kmesh = "gamma"
      self.isolated = False
      self.spin = {}
   #
   def setNempty(self,nempty) : 
      """Sets the number of empty bands. 
   
      :param nempty: number of empty bands
      :type nempty: int
   
      :Example:

      >>> gs.setNempty(10) 
      """
      self.nempty = nempty 
   #
   def setKmesh(self,kmesh) : 
      """Sets the uniform grid for k-points sampling. 
   
      :param kmesh: kmesh
      :type kmesh: 3-dim tuple of int 
   
      :Example:

      >>> gs.setKmesh((2,2,2)) 
      """
      self.kmesh = kmesh
   #
   def setIsolated(self) : 
      """Sets isolated system. Valid only for QuantumEspresso calculations.  
   
      :Example:

      >>> gs.setIsolated() 
      """
      self.isolated = True
   #
   def setCollinearSpin(self,tot_magnetization=0.) :
      """Sets collinear spin.
  
      :param tot_magnetization: Total majority spin charge - minority spin charge, optional 
      :type tot_magnetization: float      
   
      :Example:

      >>> gs.setCollinearSpin() 
      """
      self.spin = {}
      self.spin["nspin"] = 2  
      self.spin["tot_magnetization"] = tot_magnetization
   #
   def setNonCollinearSpin(self,lspinorb=False) :
      """Sets non collinear spin. Requires fully relativistic pseudopotentials. Valid only for QuantumEspresso calculations. 

      Optionally spin-orbit can be turned on. 
  
      :param lspinorb: spin-orbit, optional 
      :type lspinorb: boolean      
   
      :Example:

      >>> gs.setNonCollinearSpin() 
      """
      self.spin = {}
      self.spin["nspin"] = 4
      self.spin["lspinorb"] = lspinorb
      if self.kmesh == "gamma" : 
         self.setKmesh((1,1,1))
   # 
   def generateInputPW(self,fname="pw.in") :
      """Generates input file for pwscf. Valid only for QuantumEspresso calculations. 
  
      :param fname: fname, optional
      :type fname: string      
   
      :Example:

      >>> gs.generateInputPW("pw.in") 
      """
      #
      if( self.geom.pseudoFormat in ["upf"]) : 
         #
         with open(fname, "w") as file :
            file.write("&CONTROL\n")
            file.write("calculation       = 'scf'\n")
            file.write("restart_mode      = 'from_scratch'\n")
            file.write("pseudo_dir        = './'\n")
            file.write("outdir            = './'\n")
            file.write("prefix            = 'calc'\n")
            file.write("wf_collect        = .TRUE.\n")
            file.write("/\n")
            file.write("&SYSTEM\n")
            file.write("ibrav             = 0\n")
            file.write("nat               = " + str(self.geom.getNumberOfAtoms()) + "\n" )
            file.write("ntyp              = " + str(self.geom.getNumberOfSpecies()) + "\n" )
            file.write("ecutwfc           = " + str(self.ecut) + "\n" )
            file.write("nbnd              = " + str(self.geom.getNumberOfElectrons() + self.nempty) + "\n" )
            file.write("input_dft         = '" + str(self.xc) +"'\n" )
            file.write("nosym             = .TRUE.\n" )
            file.write("noinv             = .TRUE.\n" )
            if( "nspin" in self.spin.keys() ) : 
               if( self.spin["nspin"] == 2 ) : 
                  file.write("nspin             = 2\n" )  
                  file.write("tot_magnetization = " + str(self.spin["tot_magnetization"]) + "\n" )  
               if( self.spin["nspin"] == 4 ) : 
                  file.write("noncolin          = .TRUE.\n" )  
                  from westpy import logical2str
                  file.write("lspinorb          = " + logical2str(self.spin["lspinorb"]) +"\n" )  
            if( self.isolated ) : 
               file.write("assume_isolated   = 'mp'\n")
            file.write("/\n")
            #
            file.write("&ELECTRONS\n")
            file.write("diago_full_acc = .TRUE.\n")
            file.write("conv_thr       = 1.d-8\n")
            file.write("/\n")
            #
            file.write("ATOMIC_SPECIES\n")
            sp = [] 
            for atom in self.geom.atoms : 
               if( atom.symbol not in sp ) :
                  sp.append(atom.symbol)
                  file.write(atom.symbol + " " + str(self.geom.species[atom.symbol]["mass"]) + " " +  self.geom.species[atom.symbol]["fname"] + "\n")
            #
            file.write("ATOMIC_POSITIONS {bohr}\n")
            for atom in self.geom.atoms :
               file.write(atom.symbol + " " + str(atom.position[0]) + " " + str(atom.position[1]) + " " + str(atom.position[2]) + "\n")
            # 
            if ( self.kmesh in ["gamma"] ) :
               file.write("K_POINTS {gamma}\n")
            else : 
               file.write("K_POINTS {automatic}\n")
               file.write(str(self.kmesh[0]) + " " + str(self.kmesh[1]) + " " + str(self.kmesh[2]) + " 0 0 0\n")
            #
            file.write("CELL_PARAMETERS {bohr}\n")
            a1 = self.geom.cell["a1"]
            a2 = self.geom.cell["a2"]
            a3 = self.geom.cell["a3"]
            file.write(str(a1[0]) + " " + str(a1[1]) + " " + str(a1[2]) + "\n" )
            file.write(str(a2[0]) + " " + str(a2[1]) + " " + str(a2[2]) + "\n" )
            file.write(str(a3[0]) + " " + str(a3[1]) + " " + str(a3[2]) + "\n" )
            #
            print("")
            print("Generated file: ", fname )
            # 
      else : 
         print("Cannot generate input for QuantumEspresso, pseudo are not upf.") 
   # 
   def generateInputQbox(self,fname="qbox.in") : 
      """Generates input file for qbox. Valid only for Qbox calculations. 
  
      :param fname: fname, optional
      :type fname: string      
   
      :Example:

      >>> gs.generateInputQbox("qbox.in") 
      """
      #
      if( self.geom.pseudoFormat in ["xml"] ) :  
         #
         with open(fname, "w") as file :
            # cell 
            a1 = self.geom.cell["a1"]
            a2 = self.geom.cell["a2"]
            a3 = self.geom.cell["a3"]
            file.write("set cell " + str(a1[0]) + " " + str(a1[1]) + " " + str(a1[2]) + " " + str(a2[0]) + " " + str(a2[1]) + " " + str(a2[2]) + " " + str(a3[0]) + " " + str(a3[1]) + " " + str(a3[2]) + "\n" )
            # species
            sp = [] 
            for atom in self.geom.atoms : 
               if( atom.symbol not in sp ) :
                  sp.append(atom.symbol)
                  file.write("species " + self.geom.species[atom.symbol]["name"] + " " +  self.geom.species[atom.symbol]["url"] + "\n")
            # atom
            i = 0
            for atom in self.geom.atoms :
               i+=1
               file.write("atom " + atom.symbol+str(i) + " " + self.geom.species[atom.symbol]["name"] + " " + str(atom.position[0]) + " " + str(atom.position[1]) + " " + str(atom.position[2]) + "\n")
               #
            file.write("set ecut " + str(self.ecut) +"\n")
            if( self.nempty > 0 ) :  
               file.write("set nempty " + self.nempty + "\n")
            file.write("set wf_dyn JD\n")
            file.write("set xc " + self.xc + "\n")
            file.write("set scf_tol 1.e-8\n")
            if( "nspin" in self.spin.keys() ) : 
               if( self.spin["nspin"] == 2 ) : 
                  file.write("set nspin 2\n" )
                  delta_spin = (self.spin["tot_magnetization"] - (self.geom.getNumberOfElectrons() % 2) ) / 2 
                  file.write("set delta_spin" + str(self.spin["tot_magnetization"]) + "\n" )  
               else :  
                  print("ERR: non supported") 
            file.write("randomize_wf\n")
            file.write("run -atomic_density 0 100 5\n")
            file.write("save gs.xml\n")
            #
            print("")
            print("Generated file: ", fname )
            #
      else : 
         print("Cannot generate input for Qbox, pseudo are not xml.") 
   #
   def downloadPseudopotentials(self) : 
      """Download Pseudopotentials.
   
      :Example:

      >>> gs.downloadPseudopotentials()
     
      .. note:: Pseudopotential files will be downloaded in the current directory. 
      """
      self.geom.downloadPseudopotentials()
   #
   def updateSpecies(self,symbol,url) :
      """Update a species.
   
      :param symbol: chemical symbol 
      :type symbol: string
      :param url: url 
      :type url: string
   
      :Example:

      >>> geom.addSpecies( "Si", "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.1.upf" ) 
      
      .. note:: You can use this method to add either upf or xml pseudopotentials. However it is forbidded to mix them.  
      """
      self.geom.addSpecies(symbol,url)
