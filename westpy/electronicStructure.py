from __future__ import print_function

class ElectronicStructure() :
   """Class for representing an electronic structure calculation.
   
   :Example:

   >>> from westpy import * 
   >>> es = ElectronicStructure()
 
   """
   #
   def __init__(self) :
      from westpy import DataContainer
      self.dc = DataContainer()
      self.dc.upsertKey("k","k-point")
      self.dc.upsertKey("s","spin")
      self.dc.upsertKey("b","band")
   #
   def addKey(self,key,description) : 
      """Describes metadata key.
    
      :param key: key 
      :type key: string
      :param description: description
      :type description: * (hashable object)
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKey("eks","Kohn-Sham") 
      """
      self.dc.upsertKey( key, description ) 
   #
   def removeKey(self,key) : 
      """Removes key from metadata.
   
      :param key: key 
      :type key: string
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKey("eks","Kohn-Sham") 
      >>> es.removeKey("eks") 
      """
      self.dc.removeKey( key ) 
   #
   #
   def showKeys(self) : 
      """Shows keys in metadata.
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.showKeys() 
      """  
      l = self.dc.checkKeys(printSummary=True)
   #
   #
   def addDataPoint(self,ksb,key,what) : 
      """Adds datapoint to data.
   
      :param ksb: triplet of integers: k-point, spin, band (integer labels)
      :type ksb: 3-dim int
      :param key: key
      :type key: string 
      :param what: content attached to key 
      :type what: * (hashable object)
   
      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKey("eks","Kohn-Sham energy in eV")
      >>> es.addDataPoint([1,1,1],"eks",-4.6789)
      """
      self.dc.upsertPoint( { "k" : ksb[0],  "s" : ksb[1],  "b" : ksb[2] }, { key : what } ) 
     
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
      >>> es.addDataPoint([1,1,1],"eks",-4.6789)
      >>> es.plotDOS(k=1,s=1,energyKeys=["eks"],energyRange=[-5.,-3,0.01]) 
      """
      #
      if(all(x in self.dc.info.keys() for x in energyKeys)) :  
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
            for dataPoint in self.dc.coll :
               #
               if energyKey in dataPoint["d"].keys() and dataPoint["i"]["k"] == k and dataPoint["i"]["s"] == s : 
                  mu = dataPoint["d"][energyKey] 
                  emin.append( mu )
                  emax.append( mu )
                  if isinstance(sigma, str) : 
                     si = dataPoint["d"][sigma]
                  else : 
                     si = sigma 
                  if isinstance(weight, str) : 
                     we = dataPoint["d"][weight]
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
            if energyKey not in self.dc.info.keys() : 
               print("Unrecognized energyKey:", energyKey)
