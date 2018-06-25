from __future__ import print_function

class ElectronicStructure(object) :
   """Class for representing an electronic structure calculation.
   
   :Example:

   >>> from westpy import * 
   >>> es = ElectronicStructure()
 
   """
   #
   def __init__(self) :
      self.info = {}
      self.data = []
   #
   def loadFromJsonFile(self,fname) : 
      """Loads an electronic structure from file (Json format).
   
      :param fname: file name
      :type fname: string
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.loadFromJsonFile("es.json") 
      """
      from westpy import readJsonFile
      data = readJsonFile(fname)
      self.info = data["info"]
      self.data = data["data"]
   #
   def dumpToJsonFile(self,fname) : 
      """Dumps an electronic structure to file (Json format).
   
      :param fname: file name
      :type fname: string
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.dumpToJsonFile("es.json") 
      """
      from westpy import writeJsonFile
      data = {}
      data["info"] = self.info
      data["data"] = self.data
      writeJsonFile(fname,data)
   #
   def addKey(self,key,value) : 
      """Sets key-value in metadata.
   
      :param key: key 
      :type key: string
      :param value: value 
      :type value: string
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKey("eks","Kohn-Sham") 
      """
      if "keys" not in self.info.keys() : 
         self.info["keys"] = {}
      self.info["keys"][key] = value
   #
   def removeKey(self,key) : 
      """Removes key from metadata.
   
      :param key: key 
      :type key: string
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.removeKey("eks") 
      """
      if "keys" in self.info.keys() : 
         if key in self.info["keys"] : 
            self.info["keys"].pop(key,None)
         else : 
            print("key not recognized")
      else : 
         print("No keys")
   #
   #
   def showKeys(self) : 
      """Shows keys in metadata.
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.showKeys() 
      """
      if "keys" in self.info.keys() : 
         for key in self.info["keys"] : 
            print( key, self.info["keys"][key]) 
      else : 
         print("No keys") 
   #
   def addKpoint(self,kpoint,coord) : 
      """Adds kpoint info to metadata.
   
      :param kpoint: k-point integer label
      :type kpoint: int
      :param coord: crystal coordinates of the k-point
      :type coord: 3-dim list of int
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKpoint(1,[0.,0.,0.]) 
      """
      if "k" not in self.info.keys() : 
         self.info["k"] = {}
      self.info["k"][kpoint] = [float(coord[0]), float(coord[1]), float(coord[2])]
   #
   def __addSpins(self,spins) : 
      """Adds spin labels to metadata.
   
      :param spin: list of spin integer label
      :type spin: list of int
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addSpins([1,2])
      """
      if "s" not in self.info.keys() : 
         self.info["s"] = []
      for spin in spins : 
         self.info["s"].append(spin)
   #
   def __addBands(self,bands) : 
      """Adds band label to metadata.
   
      :param band: list of band integer label
      :type band: list of int
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addBands([1,2,3])
      """
      if "b" not in self.info.keys() : 
         self.info["b"] = []
      for band in bands : 
         self.info["b"].append(band)
   #
   def addDataPoint(self,ksb,key,what) : 
      """Adds datapoint to data.
   
      :param ksb: triplet of k-point, spin, band (integer labels)
      :type ksb: 3-dim int
      :param key: key
      :type key: string 
      :param what: content attached to key 
      :type what: *
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKey("eks","Kohn-Sham energy in eV")
      >>> es.addKpoint(1,[0.,0.,0.])
      >>> es.addDataPoint([1,1,1],"eks",-4.6789)
      """
      lk = False
      lkey = False
      if "k" in self.info.keys() :
         lk = ( ksb[0] in self.info["k"] )
      else : 
         print("No k")
      if "keys" in self.info.keys() : 
         lkey = ( key in self.info["keys"] ) 
      else : 
         print("No keys")
      if( lk and lkey ) :
         lfound = False 
         for x in self.data :
            if x["ksb"] == ksb : 
               x[key] = what
               lfound = True
               break 
         if not lfound : 
            d = {}
            d["ksb"] = ksb 
            d[key] = what
            self.data.append(d) 
      else : 
         if not lk : 
            print(ksb[0], "not in k, add it first") 
         if not lkey : 
            print(key, "not in keys, add it first") 
     
   #
   def plotDOS(self,k=1,s=1,energyKeys=["eks"],sigma=0.1,weight=1.,energyRange=[-20.,0.,0.01],fname="dos.png") : 
      """Plots desnity of states (DOS).
   
      :param k: k-point integer label 
      :type k: int
      :param s: spin integer label 
      :type s: int 
      :param energyKeys: energy keys
      :type energyKeys: list of string (needs to match the available keys) 
      :param sigma: standard deviation of gaussian (eV), optional 
      :type sigma: float .OR. list of string (needs to match the available keys)   
      :param weight: weight, optional 
      :type weight: float .OR. list of string (needs to match the available keys)  
      :param energyRange: energy range = min, max, step (eV), optional 
      :type energyRange: 3-dim float
      :param fname: output file name 
      :type fname: string 
      
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKey("eks","Kohn-Sham energy in eV")
      >>> es.addKpoint(1,[0.,0.,0.])
      >>> es.addDataPoint([1,1,1],"eks",-4.6789)
      >>> es.plotDOS(k=1,s=1,energyKeys=["eks"],energyRange=[-5.,-3,0.01]) 
      """
      #
      if(all(x in self.info["keys"] for x in energyKeys)) :  
         #
         import numpy as np
         import scipy as sp
         import matplotlib as mpl
         from westpy import gaussian 
         #
         npte = int((energyRange[1]-energyRange[0])/energyRange[2]) + 1
         energyAxis = np.linspace(energyRange[0], energyRange[1], npte, endpoint=True)
         #
         dosAxis = {}
         emin = []
         emax = []
         ymax = []
         for energyKey in energyKeys : 
            #
            dosAxis[energyKey] = np.zeros ( (npte) )
            #
            for dataPoint in self.data :
               #
               if energyKey in dataPoint.keys() and dataPoint["ksb"][0] == k and dataPoint["ksb"][1] == s : 
                  mu = dataPoint[energyKey] 
                  emin.append( mu )
                  emax.append( mu )
                  if isinstance(sigma, str) : 
                     si = dataPoint[sigma]
                  else : 
                     si = sigma 
                  if isinstance(weight, str) : 
                     we = dataPoint[weight]
                  else : 
                     we = weight 
                  #
                  for ix in range(npte) : 
                     dosAxis[energyKey][ix] += gaussian( energyAxis[ix], mu, si ) * we
            #
            ymax.append( np.max(dosAxis[energyKey][:]) )
         #
         print("Requested (emin,emax) : ", energyRange[0],energyRange[1])
         print("Detected  (emin,emax) : ", np.min(emin), np.max(emax))
         #
         import matplotlib.pyplot as plt
         fig = plt.figure()
         ax = fig.add_subplot(1,1,1)
         for energyKey in energyKeys : 
            dosPlot = ax.plot(energyAxis,dosAxis[energyKey],label=energyKey+" @ (k="+str(k)+",s="+str(s)+")")
         #
         plt.xlim([energyRange[0],energyRange[1]])
         plt.ylim([0,np.max(ymax[:])])
         plt.xlabel('energy (eV)')
         plt.ylabel('DOS')
         plt.savefig(fname)
         plt.legend()
         print("output written in : ", fname)
         print("waiting for user to close image preview...")
         plt.show()
         fig.clear()
      else: 
         for energyKey in energyKeys : 
            if energyKey not in self.info["keys"] : 
               print("Unrecognized energyKey:", energyKey)
