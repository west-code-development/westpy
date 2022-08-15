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
   def showKeys(self) :
      """Shows keys in metadata.

      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.showKeys()
      """
      l = self.dc.checkKeys(printSummary=True)
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
   def plotDOS(self,kk=[1],ss=[1],energyKeys=["eks"],sigma=0.1,weight=1.,energyRange=[-20.,0.,0.01],fname="dos.png") :
      """Plots desnity of states (DOS).

      :param kk: list of k-points
      :type kk: list of int
      :param ss: list of spin channels
      :type ss: list of int (must be [1], [2], or [1,2])
      :param energyKeys: energy keys
      :type energyKeys: list of string (needs to match the available keys)
      :param sigma: standard deviation of gaussian (eV), optional
      :type sigma: float .OR. string (needs to match the available keys)
      :param weight: weight, optional
      :type weight: float .OR. string (needs to match the available keys)
      :param energyRange: energy range = min, max, step (eV), optional
      :type energyRange: 3-dim float
      :param fname: output file name
      :type fname: string

      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKey("eks","Kohn-Sham energy in eV")
      >>> es.addDataPoint([1,1,1],"eks",-4.6789)
      >>> es.addDataPoint([1,1,2],"eks",-4.3456)
      >>> es.addDataPoint([1,2,1],"eks",-4.4567)
      >>> es.addDataPoint([1,2,2],"eks",-4.0123)
      >>> es.plotDOS(kk=[1],ss=[1,2],energyKeys=["eks"],energyRange=[-5.,-3,0.01])
      """
      #
      if(all(x in self.dc.info.keys() for x in energyKeys)) :
         #
         import numpy as np
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
            dosAxis[energyKey] = np.zeros ( (len(ss),npte) )
            #
            for dataPoint in self.dc.coll :
               for s in ss :
                  for k in kk :
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
                           dosAxis[energyKey][s-1][ix] += gaussian( energyAxis[ix], mu, si ) * we
            #
            ymax.append( np.max(dosAxis[energyKey][:]) )
         #
         print("Requested (emin,emax) : ", energyRange[0],energyRange[1])
         print("Detected  (emin,emax) : ", np.min(emin), np.max(emax))
         #
         import matplotlib.pyplot as plt
         #
         fig = plt.figure()
         ax = fig.add_subplot(1,1,1)
         for energyKey in energyKeys :
            for s in ss :
               dosPlot = ax.plot(energyAxis,dosAxis[energyKey][s-1],label=f"{energyKey} @ (s={s})")
         #
         plt.xlim([energyRange[0],energyRange[1]])
         plt.ylim([0,np.max(ymax[:])])
         plt.xlabel("energy (eV)")
         plt.ylabel("DOS")
         plt.savefig(fname,dpi=300)
         plt.legend()
         print("output written in : ", fname)
         print("waiting for user to close image preview...")
         plt.show()
         fig.clear()
      else:
         for energyKey in energyKeys :
            if energyKey not in self.dc.info.keys() :
               print("Unrecognized energyKey:", energyKey)
   #
   def plotLDOS(self,kk=[1],ss=[1],energyKeys=["eks"],sigma=0.1,weight=1.,energyRange=[-20.,0.,0.01],wfcKey="wfcFile",fname="ldos.png") :
      """Plots desnity of states (DOS).

      :param kk: list of k-points
      :type kk: list of int
      :param ss: list of spin channels
      :type ss: list of int
      :param energyKeys: energy keys
      :type energyKeys: list of string (needs to match the available keys)
      :param sigma: standard deviation of gaussian (eV), optional
      :type sigma: float .OR. string (needs to match the available keys)
      :param weight: weight, optional
      :type weight: float .OR. string (needs to match the available keys)
      :param energyRange: energy range = min, max, step (eV), optional
      :type energyRange: 3-dim float
      :param wfcKey: wavefunction file
      :type wfcKey: string (needs to match the available keys)
      :param fname: output file name
      :type fname: string

      :Example:

      >>> from westpy import *
      >>> es = ElectronicStructure()
      >>> es.addKey("eks","Kohn-Sham energy in eV")
      >>> es.addDataPoint([1,1,1],"eks",-4.6789)
      >>> es.addDataPoint([1,1,2],"eks",-4.3456)
      >>> es.addDataPoint([1,2,1],"eks",-4.4567)
      >>> es.addDataPoint([1,2,2],"eks",-4.0123)
      >>> es.plotLDOS(kk=[1],ss=[1,2],energyKeys=["eks"],energyRange=[-5.,-3,0.01])
      """
      #
      if(all(x in self.dc.info.keys() for x in energyKeys)) :
         #
         import numpy as np
         from scipy import interpolate
         from westpy import gaussian
         #
         npte = int((energyRange[1]-energyRange[0])/energyRange[2]) + 1
         energyAxis = np.linspace(energyRange[0], energyRange[1], npte, endpoint=True)
         #
         # Read x axis from file
         #
         for dataPoint in self.dc.coll :
            #
            energyKey = energyKeys[0]
            k = kk[0]
            s = ss[0]
            #
            if energyKey in dataPoint["d"].keys() and dataPoint["i"]["k"] == k and dataPoint["i"]["s"] == s :
               wfcFile = dataPoint["d"][wfcKey]
               #
               with open(wfcFile, "r") as f :
                  lines = f.readlines()
                  nptx = 0
                  for line in lines :
                     if not line.startswith("#"):
                        nptx += 1
               #
               xAxisFromFile = np.zeros(nptx)
               #
               with open(wfcFile, "r") as f :
                  lines = f.readlines()
                  iptx = 0
                  for line in lines :
                     if not line.startswith("#"):
                        xAxisFromFile[iptx] = float(line.split()[0])
                        iptx += 1
               #
               break
         #
         ntimes = 3
         xAxis = np.linspace(0., np.max(xAxisFromFile[:]), ntimes*nptx, endpoint=True)
         #
         emin = []
         emax = []
         xmap = np.zeros((ntimes*nptx, npte))
         ymap = np.zeros((ntimes*nptx, npte))
         cmap = {}
         #
         for i in range(ntimes*nptx) :
             for j in range(npte) :
                 xmap[i,j] = xAxis[i]
                 ymap[i,j] = energyAxis[j]
         #
         for energyKey in energyKeys :
            #
            cmap[energyKey] = np.zeros((ntimes*nptx-1, npte-1))
            #
            for dataPoint in self.dc.coll :
               for s in ss :
                  for k in kk :
                     #
                     if energyKey in dataPoint["d"].keys() and dataPoint["i"]["k"] == k and dataPoint["i"]["s"] == s :
                        mu = dataPoint["d"][energyKey]
                        emin.append(mu)
                        emax.append(mu)
                        if isinstance(sigma, str) :
                           si = dataPoint["d"][sigma]
                        else :
                           si = sigma
                        if isinstance(weight, str) :
                           we = dataPoint["d"][weight]
                        else :
                           we = weight
                        #
                        # Read wavefunction from file
                        #
                        wfcFile = dataPoint["d"][wfcKey]
                        wfc = np.zeros(nptx)
                        #
                        with open(wfcFile, "r") as f :
                           lines = f.readlines()
                           ipt = 0
                           for line in lines :
                              if not line.startswith("#"):
                                 wfc[ipt] = float(line.split()[1])
                                 ipt += 1
                        #
                        part = gaussian(energyAxis[0:npte-1], mu, si) * we
                        tck = interpolate.splrep(xAxisFromFile[:], wfc[:], s=0, per=True)
                        f   = interpolate.splev(xAxis, tck)
                        for i in range(ntimes*nptx-1) :
                           cmap[energyKey][i,:] += f[i] * part[:]
         #
         print("Requested (emin,emax) : ", energyRange[0], energyRange[1])
         print("Detected  (emin,emax) : ", np.min(emin), np.max(emax))
         #
         # Plot
         #
         import matplotlib.pyplot as plt
         from matplotlib.colors import ListedColormap,LinearSegmentedColormap
         #
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         #
         ncolors = 256
         red_color_array = plt.get_cmap(ListedColormap(['red']))(range(ncolors))
         blue_color_array = plt.get_cmap(ListedColormap(['blue']))(range(ncolors))
         red_color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
         blue_color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
         red_map_object = LinearSegmentedColormap.from_list("mycmap", colors=red_color_array)
         blue_map_object = LinearSegmentedColormap.from_list("mycmap", colors=blue_color_array)
         #
         iKey = 0
         for energyKey in energyKeys :
            if iKey == 1 :
               cmapUse = red_map_object
            elif iKey == 2 :
               cmapUse = blue_map_object
            else :
               cmapUse = "gray_r"
            heatmap = ax.pcolormesh(xmap, ymap, cmap[energyKey], cmap=cmapUse)
            #
            iKey += 1
         #
         plt.xlim([0, np.max(xAxis[:])])
         plt.ylim([energyRange[0], energyRange[1]])
         plt.xlabel("avg. dir. (a.u.)")
         plt.ylabel("energy (eV)")
         plt.savefig(fname,dpi=300)
         #plt.legend()
         print("output written in : ", fname)
         print("waiting for user to close image preview...")
         plt.show()
         fig.clear()
      else:
         for energyKey in energyKeys :
            if energyKey not in self.dc.info.keys() :
               print("Unrecognized energyKey:", energyKey)
