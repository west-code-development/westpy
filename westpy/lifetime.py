def radiative_lifetime(westpp_file,ispin,band1,band2,n,e_zpl):
   """Computes radiative lifetime.

   :param westpp_file: The JSON output file of Westpp
   :type westpp_file: string
   :param ispin: spin index
   :type ispin: int
   :param band1: band index (transition from band1 to band2 is computed)
   :type band1: int
   :param band2: band index (transition from band1 to band2 is computed)
   :type band2: int
   :param n: refractive index
   :type n: float
   :param e_zpl: zero phonon line (ZPL) energy in Rydberg
   :type e_zpl: float

   :Example:

   >>> from westpy import *
   >>> tau = radiative_lifetime("westpp.json",2,101,102,2.0,1.25)
   """
   #
   import json
   import numpy as np
   import scipy.constants as sc
   from westpy.units import Angstrom, Joule
   #
   assert (ispin == 1 or ispin == 2)
   assert n > 0.
   assert e_zpl > 0.
   #
   # read westpp
   with open(westpp_file,'r') as f:
       westpp_json = json.load(f)
   #
   nkstot = westpp_json['system']['electron']['nkstot']
   lsda = westpp_json['system']['electron']['lsda']
   nkpt = int(nkstot/2) if lsda else nkstot
   ikpt0 = 1+nkpt*(ispin-1)
   ikpt1 = ikpt0+nkpt
   #
   westpp_range = westpp_json['input']['westpp_control']['westpp_range']
   nband = westpp_range[1]-westpp_range[0]+1
   itrans = (band2-westpp_range[0])*nband+(band1-westpp_range[0])
   #
   rr = np.zeros(3)
   for ikpt in range(ikpt0,ikpt1):
      label_k = 'K'+'{:05d}'.format(ikpt)
      #
      assert westpp_json['output']['D'][label_k]['dipole'][itrans]['trans'] == [band1,band2]
      #
      eig1 = westpp_json['output']['D'][label_k]['energies'][band1-1]
      eig2 = westpp_json['output']['D'][label_k]['energies'][band2-1]
      e_diff = eig2-eig1
      #
      assert e_diff > 1.e-8
      #
      wk = westpp_json['output']['D'][label_k]['weight']
      if not lsda:
         wk /= 2.
      #
      re = westpp_json['output']['D'][label_k]['dipole'][itrans]['re']
      im = westpp_json['output']['D'][label_k]['dipole'][itrans]['im']
      #
      for i in range(3):
         rr[i] += np.sqrt(re[i]**2 + im[i]**2) * wk * / e_diff
   #
   rr_sq = sum(rr**2)
   #
   # Bohr to m
   Meter = Angstrom * 1.e10
   rr_sq /= Meter**2
   # Ry to J
   e_zpl /= Joule
   #
   # compute radiative lifetime using SI units
   tau = (3 * sc.epsilon_0 * sc.pi * (sc.c**3) * (sc.hbar**4)) / (n * (e_zpl**3) * (sc.e**2) * rr_sq)
   #
   return tau
