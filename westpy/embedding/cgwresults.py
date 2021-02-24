from typing import Dict, List, Tuple, Optional, Union
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy
from lxml import etree
from sunyata.parsers.text import parse_one_value, find_index
from sunyata.structure.volumetric import VData
from sunyata.aux.units import rydberg_to_hartree, ev_to_hartree, hartree_to_ev
from sunyata.heff.heff import Heff
from sunyata.heff.symm import PointGroup, PointGroupRep
from sunyata.parsers.west import WstatOutput


class CGWResults:

    ev_thr = 0.001 * ev_to_hartree  # threshold for determining degenerate states
    occ_thr = 0.001  # threshold for determining equal occupations

    # labels for all possible expressions of W
    Ws_all = [
        "Wrp_rpa", "Wrp_tc", "Wrp_te", "Wrp_vte", "Wrp_epstev",
        "Wp_rpa", "Wp_tc", "Wp_te", "Wp_vte", "Wp_epstev", "Wp_zero"
    ]

    def __init__(self,
                 path: str,
                 occ: Optional[np.ndarray] = None,
                 fxc: Optional[np.ndarray] = None,
                 eps_infty: Optional[float] = None,
                 overlap_basis: Optional[Union[Dict, List[int]]] = None,
                 point_group: Optional[PointGroup] = None,
                 verbose: Optional[bool] = True):
        """ Parser of constrained GW calculations.

        Args:
            path: directory where pw, wstat and wfreq calculations are performed.
            occ: occupation numbers. Read from pw results if not defined.
            fxc: user-defined fxc matrix on PDEP basis. Read from wstat save folder if not defined.
            eps_infty: user-defined epsilon infinity.
            point_group: point group of the system.
            verbose: if True, print a summary.
        """
        self.path = path
        self.eps_infty = eps_infty
        self.point_group = point_group

        # general info
        self.js = json.load(open(f"{self.path}/west.wfreq.save/wfreq.json"))
        self.nspin = self.js["system"]["electron"]["nspin"]
        self.npdep = self.js["input"]["wfreq_control"]["n_pdep_eigen_to_use"]
        self.omega = self.js["system"]["cell"]["omega"]

        self.cgw_calculation = self.js["input"]["cgw_control"]["cgw_calculation"]
        self.projector_type = self.js["input"]["cgw_control"]["projector_type"]
        self.nproj = 0
        self.range = False
        if self.projector_type in ('B','R','M'):
            self.local_projectors = np.array(self.js['input']['cgw_control']['local_projectors'], dtype=int)
            self.nproj += len(self.local_projectors)
        if self.projector_type in ('K','M'):
            self.nbndstart, self.nbndend = np.array(self.js['input']['cgw_control']['ks_projector_range'], dtype=int)
            self.ks_projectors_ = np.arange(self.nbndstart, self.nbndend + 1)
            if self.nbndstart != 0:
                self.range = True
            else: 
                self.ks_projectors_ = np.array(self.js['input']['cgw_control']['ks_projectors'], dtype=int)
            if self.projector_type == 'K' and self.cgw_calculation == 'G':
                tmp = int(np.sqrt(len(np.fromfile(f"{path}/west.wfreq.save/overlap.dat", dtype=float))/self.nspin/(self.npdep+3)))
                if tmp != len(self.ks_projectors_):
                    # print(len(overlap_basis),tmp,len(self.ks_projectors_))
                    assert len(overlap_basis) == tmp and tmp > len(self.ks_projectors_)
                    self.ks_projectors_sigma = self.ks_projectors_
                    self.nproj_sigma = len(self.ks_projectors_sigma)
                    self.ks_projectors_ = np.array(overlap_basis)
                    # print(self.ks_projectors_sigma)
                        
            self.nbndstart, self.nbndend = self.ks_projectors[0], self.ks_projectors[1]
            # print(self.ks_projectors)
            self.nproj += len(self.ks_projectors)
        # print(self.ks_projectors)
        if self.projector_type != 'K':
            self.point_group = None

        self.npair, self.pijmap, self.ijpmap = self.make_index_map(self.nproj)

        wfoutput = open(f"{self.path}/wfreq.out").readlines()
        i = find_index("Divergence =", wfoutput)
        self.div = parse_one_value(float, wfoutput[i])
        i = find_index("n_spectralf", wfoutput)
        self.n_spectralf = parse_one_value(int, wfoutput[i])
        i = find_index("PBE0", wfoutput)
        if i != None:
            self.xc = 'ddh'
        else:
            self.xc = 'pbe'

        # occupation numbers and eigenvalues
        if self.projector_type in ('B','R','M'):
            pass
        else:
            self.occ = np.zeros([self.nspin, self.nproj])
            self.egvs = np.zeros([self.nspin, self.nproj])
            if occ is None:
                # read occupation from pw xml file
                for ispin in range(self.nspin):
                    if self.nspin == 1:
                        xmlroot = etree.parse(f"{path}/pwscf.save/K00001/eigenval.xml")
                    else:
                        xmlroot = etree.parse(f"{path}/pwscf.save/K00001/eigenval{ispin+1}.xml")
                    egvsleaf = xmlroot.find("./EIGENVALUES")
                    nbnd = int(egvsleaf.attrib["size"])
                    egvs = np.fromstring(egvsleaf.text, sep=" ")
                    occleaf = xmlroot.find("./OCCUPATIONS")
                    occ = np.fromstring(occleaf.text, sep=" ")
                    assert egvs.shape == (nbnd,) and occ.shape == (nbnd,)
                    if self.nspin == 1:
                        occ *= 2
                    self.egvs[ispin, :] = egvs[self.ks_projectors - 1]  # -1 because band index start from 1
                    self.occ[ispin, :] = occ[self.ks_projectors - 1]
            else:
                print("Warning: user-defined occupation!")
                self.occ[...] = occ

        # 1e density matrix
        if self.projector_type in ('B','R','M'): 
            if self.nspin != 1:
                print('Not implemented!')
            else:
                self.dm = 2*np.fromfile(f"./west.wfreq.save/rdm.dat", dtype=float).reshape((1, self.nproj, self.nproj),order='F')
                occ = np.diagonal(self.dm[0,:,:])
                self.occ = occ.reshape(1,len(occ))
                # print(self.occ)
                self.nel = int(round(np.sum(self.occ)))
                # print(self.occ,self.nel)
        else:
            self.dm = np.zeros([self.nspin, self.nproj, self.nproj])
            for ispin in range(self.nspin):
                self.dm[ispin, ...] = np.diag(self.occ[ispin])
            self.nel = int(np.sum(self.dm))

        # 1 electron Hamiltonian elements
        self.parse_h1e()

        # Vc
        self.Vc = self.parse_eri("/west.wfreq.save/vc.dat")

        # fxc
        self.fxc = np.zeros([self.npdep, self.npdep])
        if fxc is None:
            try:
                self.parse_fxc(f"{self.path}/west.wstat.save/FXC.dat")
            except:
                print("Warning: error reading fxc file, fxc is set to zero!")
        else:
            print("Warning: user-defined fxc matrix!")
            if isinstance(fxc, np.ndarray):
                assert fxc.shape == (self.npdep, self.npdep)
                self.fxc[...] = fxc
            else:
                self.parse_fxc(fxc)
        self.I = np.eye(self.npdep)
        self.fhxc = self.fxc + self.I

        # braket and overlap
        if self.projector_type == 'K':
            self.overlap = np.fromfile(f"{path}/west.wfreq.save/overlap.dat", dtype=float).reshape(
                self.nspin, self.nproj, self.nproj, self.npdep + 3
            )
        braket = np.fromfile(f"{path}/west.wfreq.save/braket.dat", dtype=complex).reshape(
            self.npdep, self.nspin, self.npair).real
        self.braket_pair = np.einsum("nsi->sin", braket)
        self.braket = np.zeros([self.nspin, self.nproj, self.nproj, self.npdep])
        for ispin, i, j, n in np.ndindex(self.nspin, self.nproj, self.nproj, self.npdep):
            self.braket[ispin, i, j, n] = self.braket_pair[ispin, self.ijpmap[i, j], n]

        # Read chi0, compute p, chi
        # note: F -> C order
        self.chi0 = np.fromfile(
            "{}/west.wfreq.save/chi0.dat".format(path), dtype=complex
        ).reshape(self.npdep + 3, self.npdep + 3).T

        # Read chi0a and other constrained quantities, FOR REFERNCE ONLY
        self.chi0a_ref = np.fromfile(
            "{}/west.wfreq.save/chi0a.dat".format(path), dtype=complex
        ).reshape(self.npdep + 3, self.npdep + 3).T
        self.p_ref = np.fromfile(
            "{}/west.wfreq.save/p.dat".format(path), dtype=complex
        ).reshape(self.npdep + 3, self.npdep + 3).T
        # self.pa_ref = np.fromfile(
        #     "{}/west.wfreq.save/pa.dat".format(path), dtype=complex
        # ).reshape(self.npdep + 3, self.npdep + 3).T
        # self.pr_ref = np.fromfile(
        #     "{}/west.wfreq.save/pr.dat".format(path), dtype=complex
        # ).reshape(self.npdep + 3, self.npdep + 3).T
        # chi_ref = np.fromfile("{}/west.wfreq.save/chi.dat".format(path), dtype=complex)
        # self.chi_ref = (chi_ref[0], chi_ref[1:].reshape([self.npdep, self.npdep]).T)

        if verbose:
            self.print_summary()

    def parse_fxc(self, path: str):
        """ Read fxc matrix from FXC.dat.

        Args:
            path: directory containing FXC.dat.

        Returns:
            fxc matrix.
        """
        self.fxc[...] = 0
        fxc = WstatOutput.parse_fxc(path)
        npdep_ = fxc.shape[0]
        n = min(self.npdep, npdep_)
        if n < self.npdep:
            print("Warning: fxc dimension < npdep, extra elements are set to zero")
        self.fxc[:n, :n] = fxc[:n, :n]

    @property
    def ks_projectors(self):
        """ Band indices of KS projectors (orbitals defining the active space). """
        return self.ks_projectors_

    def print_summary(self):
        """ Print a summary after parsing CGW results. """
        print("---------------------------------------------------------------")
        print("CGW Results General Info")
        print(f"path: {self.path}")
        print(f"nspin = {self.nspin}, nel = {self.nel}, nproj = {self.nproj}, npdep = {self.npdep}")
        if self.point_group is not None:
            print(f"point group: {self.point_group.name}")
        if self.eps_infty is not None:
            print(f"eps_infinity from input: {self.eps_infty}")
        if self.projector_type in ('M','B','R'):
            print(f"local_projectors: {self.local_projectors}")
        if self.projector_type in ('K','M'):
            print(f"ks_projectors: {self.ks_projectors}")
        
        print("occupations:")
        print(self.occ)
        # print(f"max|p - p_ref| = {np.max(np.abs(self.p - self.p_ref)):.3f}")
        # chi_ref = self.solve_dyson_with_identity_kernel(self.p_ref)
        # print("max|dyson(p) - dyson(p_ref)| = {:.3f}, {:.3f}".format(
        #     np.abs(self.chi[0] - chi_ref[0]), np.max(np.abs(self.chi[1] - chi_ref[1]))
        # ))
        if self.projector_type == 'K':
            # print(self.ks_projectors)
            print("max|chi0a - chi0a_ref| = {:.3f}".format(
                np.max(np.abs(self.compute_chi0a(basis=self.ks_projectors) - self.chi0a_ref))
                # np.max(np.abs(self.compute_chi0a(basis=None) - self.chi0a_ref))
            ))
        else:
            print(f'projector_type: {self.projector_type}')
        print("---------------------------------------------------------------")

    def print_egvs(self, e0: float = 0.0):
        """ Print KS eigenvalues.

        Args:
            e0: print KS eigenvalues shifted by e0 (in Hartree).
        """
        for ispin in range(self.nspin):
            print(f"band#  ev  occ (spin {ispin})")
            for i, (ev, occ) in enumerate(zip(self.egvs[ispin], self.occ[ispin])):
                print(f"{i + self.nbndstart}, {ev * hartree_to_ev - e0:.2f}, {occ:.2f}")

    def compute_chi0a(self, basis: List[int] = None, npdep_to_use: int = None) -> np.ndarray:
        """ Compute chi0^a (chi0 projected into active space).

        Args:
            basis: list of band indices for the orbitals defining the active space.
            npdep_to_use: # of PDEP basis to use.

        Returns:
            chi0a defined on PDEP basis.
        """

        assert self.projector_type == 'K'

        if self.range == True:
            if basis is None:
                basis_ = self.ks_projectors - self.nbndstart
            else:
                basis_ = np.array(basis) - self.nbndstart
            if npdep_to_use is None:
                npdep_to_use = self.npdep
        else:
            if basis is None:
                basis_ = np.array(range(len(self.ks_projectors)))
            else:
                basis_ = []
                for i in basis:
                    basis_ += np.argwhere(self.ks_projectors == i)[0].tolist()
                basis_ = np.array(basis_)
        # print(basis_)
        # if basis is None:
        #     basis_ = self.ks_projectors - self.nbndstart
        # else:
        #     basis_ = np.array(basis) - self.nbndstart

        if npdep_to_use is None:
            npdep_to_use = self.npdep
        overlap = self.overlap[..., list(range(npdep_to_use)) + [-3, -2, -1]]

        # print(overlap[:,basis_,:,:][:,:,basis_,:])

        # Summation over state (SOS) / Adler-Wiser expression
        chi0a = np.zeros([npdep_to_use + 3, npdep_to_use + 3])
        # print(self.egvs.shape)
        # print(basis_)
        for ispin in range(self.nspin):
            for i in basis_:
                for j in basis_:
                    if i >= j:
                        continue
                    # print(self.egvs)
                    ei, ej = self.egvs[ispin, i], self.egvs[ispin, j]
                    fi, fj = self.occ[ispin, i], self.occ[ispin, j]
                    if abs(ei - ej) < self.ev_thr:
                        assert (fi - fj) < self.occ_thr
                        continue

                    prefactor = 2 * (fi - fj) / (ei - ej)
                    # print(prefactor)
                    # raise
                    chi0a += prefactor * np.einsum(
                        "m,n->mn", overlap[ispin,i,j,:], overlap[ispin,i,j,:]
                    )
        # Note: at this point overlap is in Rydberg unit, eigenvalues is in Hartree unit
        chi0a *= rydberg_to_hartree / self.omega

        return chi0a

    @staticmethod
    def extract(m, npdep_to_use: int) -> np.ndarray:
        """ Helper function to handle macroscopic part of chi0. """
        s = list(range(npdep_to_use)) + [-3, -2, -1]
        return m[s, :][:, s]

    def compute_Ws(self, basis: List[int] = None, chi0a_fortran: bool = False, npdep_to_use: int = None) -> Dict[str, np.ndarray]:
        """ Compute unconstrained Wp (independent of active space) and
        constrained Wrp of a certain active space.

        Args:
            basis: list of band indices for the orbitals defining the active space.
            npdep_to_use: # of PDEP basis to use.

        Returns:
            a dictionary of W (4-index array)
        """
        if npdep_to_use is None:
            npdep_to_use = self.npdep

        chi0 = self.extract(self.chi0, npdep_to_use)
        fxc = self.fxc[:npdep_to_use, :npdep_to_use]
        fhxc = self.fhxc[:npdep_to_use, :npdep_to_use]

        chirpa = self.solve_dyson_with_identity_kernel(chi0)
        p = self.solve_dyson_with_body_only_kernel(chi0, fxc)
        chi = self.solve_dyson_with_identity_kernel(p)

        # Unconstrained W
        chi_head, chi_body = chi
        wps = {
            "wp_rpa": chirpa,
            "wp_zero": (0, np.zeros([npdep_to_use, npdep_to_use])),
            "wp_tc": chi,
            "wp_te": (chi_head, fxc + fhxc @ chi_body @ fhxc),
            "wp_vte": (chi_head, fhxc @ chi_body @ fhxc),
            "wp_epstev": (chi_head, fhxc @ chi_body),
        }

        # Constrained W
        if self.projector_type == 'K' and chi0a_fortran == False:
            chi0a = self.compute_chi0a(basis=basis, npdep_to_use=npdep_to_use)
        else:
            chi0a = self.extract(self.chi0a_ref, npdep_to_use=npdep_to_use)

        # RPA
        chi0r = chi0 - chi0a
        chirrpa = self.solve_dyson_with_identity_kernel(chi0r)
        # Test charge
        pa_tc = self.solve_dyson_with_body_only_kernel(chi0a, fxc)
        pr_tc = p - pa_tc
        chir_tc = self.solve_dyson_with_identity_kernel(pr_tc)
        # Test electron
        pr_te = self.solve_dyson_with_body_only_kernel(chi0r, fxc)
        chir_te = self.solve_dyson_with_identity_kernel(pr_te)

        chir_te_head, chir_te_body = chir_te

        wrps = {
            "wrp_rpa": chirrpa,
            "wrp_tc": chir_tc,
            "wrp_te": (chir_te_head, fxc + fhxc @ chir_te_body @ fhxc),
            "wrp_vte": (chir_te_head, fhxc @ chir_te_body @ fhxc),
            "wrp_epstev": (chir_te_head, fhxc @ chir_te_body),  # eps^-1_Hxc Vc
        }

        # combine various W in PDEP basis and compute their ERI
        ws = {**wps, **wrps}
        Ws = {
            key.capitalize(): self.solve_eri(
                h=chi_head if self.eps_infty is None else (1.0 / self.eps_infty - 1.0),
                B=chi_body, basis=basis, npdep_to_use=npdep_to_use
            )
            for key, (chi_head, chi_body) in ws.items()
        }
        return Ws

    def parse_h1e(self):
        """ Read 1e terms in the KS hamiltonian. """
        self.s1e = np.fromfile(f"{self.path}/west.wfreq.save/s1e.dat", dtype=float).reshape(
            self.nspin, self.nspin, self.nproj, self.nproj
        )
        self.h1e_treatment = self.js["input"]["cgw_control"]["h1e_treatment"]
        checklist = ["kin", "vloc", "vnl", "vh", "vxc", "vxx", "hks"]            
        for h in checklist:
            setattr(self, h, np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj, self.nproj))

        if self.h1e_treatment in ('R','T') or 'self.nproj_sigma' in locals().keys():
            # for vertex in ('n'):
            #     self.parse_sigma(vertex)
            for vertex in ('v','c','n'):
                try: 
                    self.parse_sigma(vertex)
                except:
                    print(f'vertex = {vertex} not found!')

    def parse_sigma(self,vertex):
        try:
            self.nproj_sigma
            self.ks_projectors_sigma
        except:
            self.nproj_sigma = self.nproj
            self.ks_projectors_sigma = self.ks_projectors_

        checklist_cgw = [f"sigmax_{vertex}_a",f"sigmax_{vertex}_e",f"sigmax_{vertex}_f"]
        for h in checklist_cgw:
            tmp = np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj_sigma)
            tmp1 = np.zeros((self.nspin, self.nproj_sigma, self.nproj_sigma))
            for ispin in range(self.nspin):
                for iproj in range(self.nproj_sigma):
                    tmp1[ispin,iproj,iproj] = tmp[ispin,iproj]
            setattr(self, h, tmp1)

        checklist_cgw = [f"qp_energy_{vertex}"]
        for h in checklist_cgw:
            setattr(self, h, np.diag(np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            )).reshape(self.nspin, self.nproj_sigma, self.nproj_sigma))

        checklist_cgw = [f"sigmac_eigen_{vertex}_a",f"sigmac_eigen_{vertex}_e",f"sigmac_eigen_{vertex}_f"]
        for h in checklist_cgw:
            setattr(self, h, np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj_sigma, self.nproj_sigma))

        checklist_cgw = [f"sigmac_{vertex}_a",f"sigmac_{vertex}_e",f"sigmac_{vertex}_f"]

        for h in checklist_cgw:
            # setattr(self, h, np.fromfile(
            #     f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            # ).reshape(self.nspin, self.nproj, 3, self.n_spectralf))
            tmp = np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj_sigma, 3, self.n_spectralf)
            tmp1 = np.zeros((self.nspin, self.nproj_sigma, self.nproj_sigma, 3, self.n_spectralf))
            for i_spectralf in range(self.n_spectralf):
                for index in range(3):
                    for ispin in range(self.nspin):
                        for iproj in range(self.nproj_sigma):
                            tmp1[ispin,iproj,iproj,index,i_spectralf] = tmp[ispin,iproj,index,i_spectralf]
            setattr(self, h, tmp1*ev_to_hartree) 

        if vertex == 'n':
            data = getattr(self,'sigmac_'+vertex+'_e') 
            assert self.nspin == 1
            legend = [f'$\Sigma^E$-{iproj}' for iproj in list(self.ks_projectors_sigma)]
            for iproj in range(self.nproj_sigma):
                x = data[0,iproj,iproj,0,:]
                res = data[0,iproj,iproj,1,:] + getattr(self,'sigmax_'+vertex+'_e')[0,iproj,iproj]
                plt.plot(x * hartree_to_ev,res * hartree_to_ev)
                #
            #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
            #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')

            plt.title(f'{self.xc.upper()}-Re$\Sigma$')
            plt.xlabel('$\omega$ (eV)')
            plt.ylabel('E (eV)')
            plt.legend(legend)
            plt.savefig(f'{self.xc}-re.png',bbox_inches='tight')
            plt.show()

            for iproj in range(self.nproj_sigma):
                x = data[0,iproj,iproj,0,:]
                res = data[0,iproj,iproj,2,:]
                plt.plot(x * hartree_to_ev,res * hartree_to_ev)
                #
            #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
            #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')

            plt.title(f'{self.xc.upper()}-Im$\Sigma$')
            plt.xlabel('$\omega$ (eV)')
            plt.ylabel('E (eV)')
            plt.legend(legend)
            plt.savefig(f'{self.xc}-im.png',bbox_inches='tight')
            plt.show()

        basis_ = []
        for i in self.ks_projectors_sigma:
            basis_ += np.argwhere(self.ks_projectors == i)[0].tolist()
        basis_ = np.array(basis_)
        # if self.cgw_calculation == 'G' and self.projector_type == 'K':
        #     basis__sigma = []
        #     for i in basis:
        #         basis__sigma += np.argwhere(self.ks_projectors_sigma == i)[0].tolist()
        #     basis__sigma = np.array(basis__sigma)
        # else:
        #     basis__sigma = basis_

        # basis_ = self.ks_projectors_sigma - self.nbndstart
        # print(self.ks_projectors_sigma)
        # print(self.nbndstart)
        # print(basis_)
        # print(self.egvs[0,basis_] - self.vxc[0,basis_,:][:,basis_])
        # print(basis_)
        # print(self.egvs[0,basis_]* hartree_to_ev)
        # print(self.vxc[0,basis_,:][:,basis_]* hartree_to_ev)
        # print(self.sigmac_eigen_n_e[0,:,:][:,:]* hartree_to_ev + self.sigmac_eigen_n_a[0,:,:][:,:]* hartree_to_ev)
        # print(self.sigmax_n_f[0,:,:][:,:]* hartree_to_ev)
        # print(self.qp_energy_n[0,:,:][:,:]* hartree_to_ev)

        # print(basis_)

        print( (self.hks[0,basis_,:][:,basis_] - self.vxc[0,basis_,:][:,basis_] - self.vxx[0,basis_,:][:,basis_]\
                + self.sigmac_eigen_n_f[0,:,:][:,:] \
                + self.sigmax_n_f[0,:,:][:,:] - self.qp_energy_n[0,:,:][:,:] ) * hartree_to_ev)
        print("max|E - ($\epsilon$ + $\Sigma$ - $V_{{xc}}$)| = {:.3f} eV".format(
                np.max(np.einsum("ii->i", np.abs(self.hks[0,basis_,:][:,basis_] - self.vxc[0,basis_,:][:,basis_]\
                - self.vxx[0,basis_,:][:,basis_] + self.sigmac_eigen_n_f[0,:,:][:,:]\
                + self.sigmax_n_f[0,:,:][:,:] - self.qp_energy_n[0,:,:][:,:]) * hartree_to_ev, optimize=True) )
            ))

        return

    def print_sigma(self, basis: List[int] = None, xc = True):

        assert self.h1e_treatment in ('R','T')
        assert self.nspin == 1
        
        if basis == None:
            basis = self.ks_projectors_sigma
        basis_ = []
        for i in basis:
            basis_ += np.argwhere(self.ks_projectors_sigma == i)[0].tolist()
        basis_ = np.array(basis_)

        for i in range(len(basis_)):
            spaces = ('f','e','a')

            legend = [f'$\Sigma^{space.upper()}$-{basis[i]}' for space in spaces]
            for space in spaces:
                data = getattr(self,'sigmac_n_'+space)
                x = data[0,basis_[i],basis_[i],0,:]
                res = data[0,basis_[i],basis_[i],1,:]
                if xc:
                    res += getattr(self,'sigmax_n_'+space)[0,basis_[i],basis_[i]]
                plt.plot(x * hartree_to_ev,res * hartree_to_ev)
                #
            #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
            #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')
            
            if xc:
                plt.title(f'{self.xc.upper()}-Re$\Sigma$')
            else:
                plt.title(f'{self.xc.upper()}-Re$\Sigma$ (correlation part)')
            plt.xlabel('$\omega$ (eV)')
            plt.ylabel('E (eV)')
            plt.legend(legend)
            plt.savefig(f'{self.xc}-re-{basis[i]}.png',bbox_inches='tight')
            plt.show()

            for space in spaces:
                data = getattr(self,'sigmac_n_'+space)
                x = data[0,basis_[i],basis_[i],0,:]
                res = data[0,basis_[i],basis_[i],2,:]                    
                plt.plot(x * hartree_to_ev,res * hartree_to_ev)
                #
            #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
            #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')

            if xc:
                plt.title(f'{self.xc.upper()}-Im$\Sigma$') 
            else:   
                plt.title(f'{self.xc.upper()}-Im$\Sigma$ (correlation part)')
            plt.xlabel('$\omega$ (eV)')
            plt.ylabel('E (eV)')
            plt.legend(legend)
            plt.savefig(f'{self.xc}-im-{basis[i]}.png',bbox_inches='tight')
            plt.show()
        
        return

    def print_spectral(self, basis: List[int] = None):

        assert self.h1e_treatment in ('R','T')
        assert self.nspin == 1
        
        if basis == None:
            basis = self.ks_projectors_sigma
        basis_ = []
        for i in basis:
            basis_ += np.argwhere(self.ks_projectors_sigma == i)[0].tolist()
        basis_ = np.array(basis_)

        # print(basis_)
        for i in range(len(basis_)):
            spaces = ('f')

            legend = [f'$\Sigma^{space.upper()}$-{basis[i]}' for space in spaces]
            for space in spaces:
                data = getattr(self,'sigmac_n_'+space)
                x = data[0,basis_[i],basis_[i],0,:] 
                tmp1 = np.abs(data[0,basis_[i],basis_[i],2,:])
                tmp2 = data[0,basis_[i],basis_[i],1,:] + getattr(self,'sigmax_n_'+space)[0,basis_[i],basis_[i]]\
                    - np.einsum('ii->i',self.vxc[0,:,:],optimize=True)[basis_[i]]
                res = tmp1 / ( (x - np.einsum('ii->i',self.hks[0,:,:],optimize=True)[basis_[i]] - tmp2)**2 + tmp1**2 )
                
                plt.plot(x * hartree_to_ev,res * hartree_to_ev)
                #
            #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
            #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')
            
            plt.title(f'{self.xc.upper()}-Re$A$')

            plt.xlabel('$\omega$ (eV)')
            plt.ylabel('Abs')
            plt.legend(legend)
            plt.savefig(f'{self.xc}-re-{basis[i]}-A.png',bbox_inches='tight')
            plt.show()

            # for space in spaces:
            #     data = getattr(self,'sigmac_n_'+space)
            #     x = data[0,basis_[i],basis_[i],0,:]
            #     res = data[0,basis_[i],basis_[i],2,:]                    
            #     plt.plot(x * hartree_to_ev,res * hartree_to_ev)
            #     #
            # #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
            # #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')

            # if xc:
            #     plt.title(f'{self.xc.upper()}-Im$\Sigma$') 
            # else:   
            #     plt.title(f'{self.xc.upper()}-Im$\Sigma$ (correlation part)')
            # plt.xlabel('$\omega$ (eV)')
            # plt.ylabel('E (eV)')
            # plt.legend(legend)
            # plt.savefig(f'{self.xc}-im-{basis[i]}.png',bbox_inches='tight')
            # plt.show()
        
        return

    def debug(self, basis: List[int] = None):

        assert self.h1e_treatment in ('R','T') and self.projector_type == 'K'
        assert self.nspin == 1

        if basis == None:
            basis = self.ks_projectors_sigma
        if self.range == True:
            if basis is None:
                basis_ = self.ks_projectors - self.nbndstart
            else:
                basis_ = np.array(basis) - self.nbndstart
            # if npdep_to_use is None:
            #     npdep_to_use = self.npdep
        else:
            if basis is None:
                basis_ = np.array(range(len(self.ks_projectors)))
            else:
                basis_ = []
                for i in basis:
                    basis_ += np.argwhere(self.ks_projectors == i)[0].tolist()
                basis_ = np.array(basis_)
        if self.cgw_calculation == 'G' and self.projector_type == 'K':
            basis__sigma = []
            for i in basis:
                basis__sigma += np.argwhere(self.ks_projectors_sigma == i)[0].tolist()
            basis__sigma = np.array(basis__sigma)
        else:
            basis__sigma = basis_

        # print(self.nbndstart)
        # print(basis_)
        for i in range(len(basis_)):
            spaces = ('f')

            legend = [f'$\Sigma^{space.upper()}$-{basis[i]}' for space in spaces]
            for space in spaces:
                data = getattr(self,'sigmac_n_'+space)
                x = data[0,basis__sigma[i],basis__sigma[i],0,:]
                res = data[0,basis__sigma[i],basis__sigma[i],1,:] - np.einsum('ii->i',self.vxc[0,:,:],optimize=True)[basis_[i]]
                res2 = x - np.einsum('ii->i',self.hks[0,:,:],optimize=True)[basis_[i]]
                res += getattr(self,'sigmax_n_'+space)[0,basis__sigma[i],basis__sigma[i]]
                plt.plot(x * hartree_to_ev,res * hartree_to_ev)
                plt.plot(x * hartree_to_ev,res2 * hartree_to_ev)
                plt.plot(x * hartree_to_ev,(res2-res) * hartree_to_ev)
                for j in range(len(res)):
                    if np.abs(res2[j]-res[j]) * hartree_to_ev < 0.5:
                        print(x[j] * hartree_to_ev)
                #
            #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
            #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')
            
            plt.title(f'{self.xc.upper()}-Re-debug')
            plt.xlabel('$\omega$ (eV)')
            plt.ylabel('E (eV)')
            plt.legend(legend)
            plt.savefig(f'{self.xc}-re-{basis[i]}-debug.png',bbox_inches='tight')
            plt.show()

            # for space in spaces:
            #     data = getattr(self,'sigmac_n_'+space)
            #     x = data[0,basis_[i],basis_[i],0,:]
            #     res = data[0,basis_[i],basis_[i],2,:]                    
            #     plt.plot(x * hartree_to_ev,res * hartree_to_ev)
            #     #
            # #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
            # #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')

            # if xc:
            #     plt.title(f'{self.xc.upper()}-Im$\Sigma$') 
            # else:   
            #     plt.title(f'{self.xc.upper()}-Im$\Sigma$ (correlation part)')
            # plt.xlabel('$\omega$ (eV)')
            # plt.ylabel('E (eV)')
            # plt.legend(legend)
            # plt.savefig(f'{self.xc}-im-{basis[i]}.png',bbox_inches='tight')
            # plt.show()
        
        return

    def print_hopping(self, basis: List[int] = None, npdep_to_use: int = None):

        assert self.h1e_treatment in ('R','T')
        assert self.nspin == 1
        if basis == None:
            basis = self.ks_projectors_sigma
        # print(self.ks_projectors)
        
        if self.projector_type in ('K','M'):
            # print(self.range)
            if self.range == True:
                if basis is None:
                    basis_ = self.ks_projectors - self.nbndstart
                else:
                    basis_ = np.array(basis) - self.nbndstart
                if npdep_to_use is None:
                    npdep_to_use = self.npdep
            else:
                if basis is None:
                    basis_ = np.array(range(len(self.ks_projectors)))
                else:
                    basis_ = []
                    for i in basis:
                        basis_ += np.argwhere(self.ks_projectors == i)[0].tolist()
                    basis_ = np.array(basis_)
            if self.cgw_calculation == 'G' and self.projector_type == 'K':
                basis__sigma = []
                for i in basis:
                    basis__sigma += np.argwhere(self.ks_projectors_sigma == i)[0].tolist()
                basis__sigma = np.array(basis__sigma)
            else:
                basis__sigma = basis_
                        
        # basis_ = []
        # for i in basis:
        #     basis_ += np.argwhere(self.ks_projectors_sigma == i)[0].tolist()
        # basis_ = np.array(basis_)

        # print(basis__sigma)
        # print(self.sigmac_n_e.shape)
        # print(self.ks_projectors_sigma)

        data = getattr(self,'sigmac_n_e')[:,basis__sigma,:,:,:][:,:,basis__sigma,:,:]
        hks = self.hks[:,basis_,:][:,:,basis_]
        assert self.nspin == 1
        legend = [f'$t^E$-{iproj}' for iproj in basis]
        for index in range(len(basis__sigma)):
            x = data[0,index,index,0,:]
            res = self.compute_h1e_from_hks(basis=basis,\
                eri=self.make_heffs(basis=basis,dc='print_hopping'), dc='print_hopping')[0,index,index,0,:]
            plt.plot(x * hartree_to_ev,res * hartree_to_ev)
            #
        #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
        #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')

        plt.title(f'{self.xc.upper()}-Re$t$')
        plt.xlabel('$\omega$ (eV)')
        plt.ylabel('E (eV)')
        plt.legend(legend)
        plt.savefig(f'{self.xc}-re.png',bbox_inches='tight')
        plt.show()

        for index in range(len(basis__sigma)):
            x = data[0,index,index,0,:]
            res = self.compute_h1e_from_hks(basis=basis,\
                eri=self.make_heffs(basis=basis,dc='print_hopping'), dc='print_hopping')[0,index,index,1,:]
            plt.plot(x * hartree_to_ev,res * hartree_to_ev)
            #
        #     legend.append(f'$\Sigma^F$-{orbital[iproj]}')
        #     legend.append(f'$\Sigma^A$-{orbital[iproj]}')

        plt.title(f'{self.xc.upper()}-Im$t$')
        plt.xlabel('$\omega$ (eV)')
        plt.ylabel('E (eV)')
        plt.legend(legend)
        plt.savefig(f'{self.xc}-im.png',bbox_inches='tight')
        plt.show()
        
        return

    @staticmethod
    def make_index_map(n: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """ Helper function to build index maps.

        Args:
            n: # of orbitals.

        Returns:
            (# of pairs, a map from two indices to pair index,
            a map from pair index to two indices)
        """
        npair = int(n * (n + 1) / 2)
        pijmap = np.zeros([npair, 2], dtype=int)
        ijpmap = np.zeros([n, n], dtype=int)
        p = 0
        for i in range(n):
            for j in range(i, n):
                pijmap[p, :] = i, j
                ijpmap[i, j] = ijpmap[j, i] = p
                p += 1
        return npair, pijmap, ijpmap

    def parse_eri(self, fname: str) -> np.ndarray:
        """ Read ERI (electron repulsion integral, aka four-center integral) from file.

        Args:
            fname: filename (in the directory of self.path).

        Returns:
            ERI as a 4-index array.
        """
        fpath = f"{self.path}/{fname}"
        eri_pair = np.fromfile(fpath, dtype=complex).reshape(
            self.nspin, self.nspin, self.npair, self.npair).real
        return self.unfold_eri(eri_pair, self.nproj)

    def solve_eri(self,
                  h: float,
                  B: np.ndarray,
                  basis: List[int] = None,
                  npdep_to_use: int = None) -> np.ndarray:
        """ Compute ERI of given W (defined by head h and body B) on a basis of KS orbitals.

        Args:
            h: head.
            B: body.
            basis: list of band indices for the orbitals defining the active space.
            npdep_to_use: # of PDEP basis to use.

        Returns:
            W, represented as a 4-index array.
        """
        nspin = self.nspin
        if self.projector_type in ('K','M'):
            if self.range == True:
                if basis is None:
                    basis_ = self.ks_projectors - self.nbndstart
                else:
                    basis_ = np.array(basis) - self.nbndstart
                if npdep_to_use is None:
                    npdep_to_use = self.npdep
            else:
                if basis is None:
                    basis_ = np.array(range(len(self.ks_projectors)))
                else:
                    basis_ = []
                    for i in basis:
                        basis_ += np.argwhere(self.ks_projectors == i)[0].tolist()
                    basis_ = np.array(basis_)
        if self.projector_type in ('B','R','M'):
            local_basis_ = np.array(range(len(self.local_projectors)))
            if 'basis_' in dir():
                basis_ = np.append(local_basis_,basis_+len(local_basis_))
            else:
                basis_ = local_basis_


        neff = len(basis_)
        npair, pijmap, ijpmap = self.make_index_map(neff)
        assert npair <= self.npair

        braket_pair_eff = np.zeros([self.nspin, npair, npdep_to_use])
        for ispin in range(self.nspin):
            for p, (i, j) in enumerate(pijmap):
                iproj, jproj, = basis_[[i, j]]
                braket_pair_eff[ispin, p, :] = self.braket[ispin, iproj, jproj, :npdep_to_use]
        eri_pair = (1/self.omega) * np.einsum(
            "sip,pq,tjq->stij", braket_pair_eff, B.real, braket_pair_eff, optimize=True
        )

        eri = self.unfold_eri(eri_pair, neff)
        for s1, s2, i, j in np.ndindex(nspin, nspin, neff, neff):
            eri[s1, s2, i, i, j, j] += h.real * self.div

        eri *= rydberg_to_hartree

        return eri

    def unfold_eri(self, eri_pair: np.ndarray, n: int) -> np.ndarray:
        """ Helper function to unfold an ERI defined on orbital paris to 4-index ERI. """
        nspin = self.nspin
        npair, _, ijpmap = self.make_index_map(n)
        assert eri_pair.shape == (nspin, nspin, npair, npair)

        eri = np.zeros([nspin, nspin, n, n, n, n])
        for s1, s2, i, j, k, l in np.ndindex(nspin, nspin, n, n, n, n):
            eri[s1, s2, i, j, k, l] = eri_pair[s1, s2, ijpmap[i, j], ijpmap[k, l]]
        return eri

    @staticmethod
    def solve_dyson_with_body_only_kernel(bare: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ Solve Dyson equation between bare and screened quantity.

        scr = bare + bare @ kernel @ scr
        bare is Npdep + 3 by Npdep + 3 matrix
        kernel is Npdep + Npdep (neglecting head and wings)

        Args:
            bare: bare quantity.
            kernel: kernel in Dyson equation.

        Returns:
            screened (dressed) quantity.
        """

        npdep = bare.shape[0] - 3
        assert bare.shape == (npdep + 3, npdep + 3)
        assert kernel.shape == (npdep, npdep), "currently only support headless wingless kernel"

        I = np.eye(npdep)

        # extract body, head and wings of bare quantity
        B = bare[:npdep, :npdep]
        h = bare[npdep:, npdep:]
        W1 = bare[:npdep, npdep:]
        W2 = bare[npdep:, :npdep]

        # compute body, head and wings of screened quantity
        scr00 = h + W2 @ kernel @ np.linalg.inv((I - B @ kernel)) @ W1
        scr01 = W2 @ np.linalg.inv((I - kernel @ B))
        scr10 = scr01.T
        scr11 = np.linalg.inv((I - B @ kernel)) @ B

        return np.array(np.bmat([[scr11, scr10], [scr01, scr00]]))

    @staticmethod
    def solve_dyson_with_identity_kernel(bare: np.ndarray) -> Tuple[float, np.ndarray]:
        """ Solve Dyson equation between bare and screened quantity.

        scr = bare + bare @ scr
        bare is Npdep + 3 by Npdep + 3 matrix
        kernel is assumed to be identity.

        Args:
            bare: bare quantity.

        Returns:
            (head, body) of the screened/dressed quantity.
        """

        npdep = bare.shape[0] - 3
        assert bare.shape == (npdep + 3, npdep + 3)

        I = np.eye(npdep)

        # WEST Eq 42-43
        B = bare[:npdep, :npdep]
        h = bare[npdep:, npdep:]
        W1 = bare[:npdep, npdep:]
        W2 = bare[npdep:, :npdep]

        f = 1 / 3 * np.trace(h)
        mu = 1 / 3 * np.einsum("ia,aj->ij", W1, W2)
        k = 1 - f - np.trace(mu @ np.linalg.inv(I - B))

        # Lambda: scr11, Theta: Tr(scr00)
        Lambda = np.linalg.inv(I - B) @ B + 1 / k * np.linalg.inv(I - B) @ mu @ np.linalg.inv(I - B)
        Theta = (1 - k) / k

        return Theta, Lambda

    def compute_vh_from_eri(self, basis_: List[int], eri: np.ndarray) -> np.ndarray:
        """ Compute VHartree from ERI.
        
        Args:
            basis_: list of indices (in ks_projectors, NOT absolute band indices) for orbitals in the active space.
            eri: ERI.

        Returns:
            VHartree matrix in active space.
        """
        # print(basis_)
        # print(self.dm[:, basis_, :][:, :, basis_])
        return np.einsum("stijkl,tkl->sij", eri, self.dm[:, basis_, :][:, :, basis_], optimize=True)

    def compute_vxx_from_eri(self, basis_: List[int], eri: np.ndarray) -> np.ndarray:
        """ Compute VEXX from ERI.

        Args:
            basis_: list of indices (in ks_projectors, NOT absolute band indices) for orbitals in the active space.
            eri: ERI.

        Returns:
            VEXX matrix in active space.
        """
        return - 0.5 * self.nspin * np.einsum("ssikjl,skl->sij", eri, self.dm[:, basis_, :][:, :, basis_], optimize=True)

    def compute_h1e_from_hks(self,
                             basis: List[int],
                             eri: np.ndarray,
                             dc: str = "hf",
                             npdep_to_use: int = None,
                             mu: float = 0) -> np.ndarray:
        """ Compute 1e term of effective Hamiltonian from KS Hamiltonian.

        Args:
            basis: list of band indices for orbitals in the active space.
            eri: ERI.
            dc: scheme for computing double counting.
            npdep_to_use: # of PDEP basis to use.
            mu: chemical potential, used to shift the spectrum of resulting effective Hamiltonian.

        Returns:
            1e part of effective Hamiltonian.
        """
        # if basis is None:
        #     basis_ = self.ks_projectors - self.nbndstart
        # else:
        #     basis_ = np.array(basis) - self.nbndstart

        if self.projector_type in ('K','M'):
            if self.range == True:
                if basis is None:
                    basis_ = self.ks_projectors - self.nbndstart
                else:
                    basis_ = np.array(basis) - self.nbndstart
                if npdep_to_use is None:
                    npdep_to_use = self.npdep
            else:
                if basis is None:
                    basis_ = np.array(range(len(self.ks_projectors)))
                else:
                    basis_ = []
                    for i in basis:
                        basis_ += np.argwhere(self.ks_projectors == i)[0].tolist()
                    basis_ = np.array(basis_)
            if self.cgw_calculation == 'G' and self.projector_type == 'K':
                if 'sigma' in dc or 'hopping' in dc or 'test' in dc:
                    basis__sigma = []
                    for i in basis:
                        basis__sigma += np.argwhere(self.ks_projectors_sigma == i)[0].tolist()
                    basis__sigma = np.array(basis__sigma)
            else:
                basis__sigma = basis_
        # print(basis__sigma)

        if self.projector_type in ('B','R','M'):
            # print(basis_,np.array(range(len(self.local_projectors))))
            local_basis_ = np.array(range(len(self.local_projectors)))
            if 'basis_' in dir():
                basis_ = np.append(local_basis_,basis_+len(local_basis_))
            else:
                basis_ = local_basis_

        if npdep_to_use is None:
            npdep_to_use = self.npdep

        # print(basis_)
        # print(basis__sigma)

        braket = self.braket[:, basis_, :, :][:, :, basis_, :]
        # print(braket)
        occ = self.occ[:, basis_]
        if dc == "hf":
            hdc = self.compute_vh_from_eri(basis_, eri) + self.compute_vxx_from_eri(basis_, eri)
        elif dc == "hfxc":
            nA = np.einsum("siin,si->n", braket, occ)[:npdep_to_use]
            hdc = (self.compute_vh_from_eri(basis_, eri) +
                   (1/self.omega) * np.einsum(
                       "spqm,mn,n->spq", braket, self.fxc[:npdep_to_use,:npdep_to_use], nA
                   ))
        elif dc == "fhxc":
            nA = np.einsum("siin,si->n", braket, occ)[:npdep_to_use]
            hdc = (1 / self.omega) * np.einsum(
                "spqm,mn,n->spq", braket, self.fhxc[:npdep_to_use,:npdep_to_use], nA
            )
        # elif dc in ("sigmar","sigmar_0"):
        #     hdc = self.vxc[:, basis_, :][:, :, basis_] \
        #     + self.vxx[:, basis_, :][:, :, basis_] - self.sigmarx - self.sigmarc_0
        # elif dc == "sigmar_f":
        #     hdc = self.vxc[:, basis_, :][:, :, basis_] \
        #     + self.vxx[:, basis_, :][:, :, basis_] - self.sigmarx - self.sigmarc_f
        elif dc.split('_')[0] == 'sigma':
            # dc == 'sigma_e':
            # print('sigmax_e+sigmac_e ===================')
            # print(self.sigmax_e + self.sigmac_e[:, :, :, 1, int((self.n_spectralf+1)/2)])
            hdc = self.compute_vh_from_eri(basis_, eri) \
                + self.vxc[:, basis_, :][:, :, basis_] + self.vxx[:, basis_, :][:, :, basis_]\
                - getattr(self,'sigmax_'+dc.split('_')[1]+'_e')[:, basis__sigma, :][:, :, basis__sigma]\
                - getattr(self,'sigmac_eigen_'+dc.split('_')[1]+'_e')[:, basis__sigma, :][:, :, basis__sigma]
        elif dc == 'test':
            hdc = self.compute_vh_from_eri(basis_, eri) \
                + 0.5 * ( self.vxc[:, basis_, :][:, :, basis_] + self.vxx[:, basis_, :][:, :, basis_]\
                - getattr(self,'sigmax_n_e')[:, basis__sigma, :][:, :, basis__sigma]
                - getattr(self,'sigmac_eigen_n_e')[:, basis__sigma, :][:, :, basis__sigma] )

            print(self.vxc[:, basis_, :][:, :, basis_] + self.vxx[:, basis_, :][:, :, basis_]\
                - getattr(self,'sigmax_n_e')[:, basis__sigma, :][:, :, basis__sigma]\
                - getattr(self,'sigmac_eigen_n_e')[:, basis__sigma, :][:, :, basis__sigma])
                # - getattr(self,'sigmac_eigen_'+dc.split('_')[1]+'_e')[:, basis__sigma, :][:, :, basis__sigma]
                # + Z * (self.vxc[:, basis_, :][:, :, basis_] + self.vxx[:, basis_, :][:, :, basis_]\
                # - getattr(self,'sigmax_n_e')[:, basis__sigma, :][:, :, basis__sigma]\
                # - getattr(self,'sigmac_eigen_n_e')[:, basis__sigma, :][:, :, basis__sigma])
                # + self.z_e * ( self.vxc[:, basis_, :][:, :, basis_] + self.vxx[:, basis_, :][:, :, basis_]\
                # - self.sigmax_e - self.sigmac_eigen_e )
                # + self.vxc[:, basis_, :][:, :, basis_] + self.vxx[:, basis_, :][:, :, basis_]\
                # - self.sigmax_e - self.sigmac_e[:, :, :, 1, int((self.n_spectralf+1)/2)]
        elif dc == 'print_hopping':
            # print(basis__sigma)
            hdc = np.zeros((self.nspin,len(basis__sigma),len(basis__sigma),2,self.n_spectralf))
            tmp = self.compute_vh_from_eri(basis_, eri) \
                + self.vxc[:, basis_, :][:, :, basis_] + self.vxx[:, basis_, :][:, :, basis_]\
                - getattr(self,'sigmax_n_e')[:, basis__sigma, :][:, :, basis__sigma]

            for i in range(2):
                if i == 0:
                    for i_spectralf in range(self.n_spectralf):
                        hdc[:,:,:,i,i_spectralf] = tmp
            hdc -= getattr(self,'sigmac_n_e')[:, basis__sigma, :,1:,:][:, :, basis__sigma,:,:]
        else:
            raise ValueError("Unknown double counting scheme")
        
        # print(self.compute_vh_from_eri(basis_, eri)[:, basis_, :][:, :, basis_], self.compute_vxx_from_eri(basis_, eri)[:, basis_, :][:, :, basis_])
        
        # print('vh_eri ===================')
        # print(self.compute_vh_from_eri(basis_, eri))
        # print('vxx_eri ===================')
        # print(self.compute_vxx_from_eri(basis_, eri))
        # print('vxc+vxx ===================')
        # print(self.vxc[:, basis_, :][:, :, basis_] + self.vxx[:, basis_, :][:, :, basis_])

        # if dc == 'sigmar':
        #     print('sigmarx+sigmarc_0 ===================')
        #     print(self.sigmarx + self.sigmarc_0)
        #     h1e = self.hks[:, basis_, :][:, :, basis_] - hdc - self.compute_vh_from_eri(basis_, eri) 
        # elif dc == 'sigmar_0':
        #     print('sigmarx+sigmarc_0 ===================')
        #     print(self.sigmarx + self.sigmarc_0)
        #     print('z_0 ===================')
        #     print(self.z_0)
        #     h1e = self.z_0 * ( self.hks[:, basis_, :][:, :, basis_] - hdc ) - self.compute_vh_from_eri(basis_, eri)
        # elif dc == 'sigmar_f':
        #     print('sigmarx+sigmarc_f ===================')
        #     print(self.sigmarx + self.sigmarc_f)
        #     print('z_f ===================')
        #     print(self.z_f)
        #     h1e = self.hks[:, basis_, :][:, :, basis_] - self.z_f * hdc - self.compute_vh_from_eri(basis_, eri)

        if dc != 'print_hopping':
            h1e = self.hks[:, basis_, :][:, :, basis_] - hdc
            # h1e = np.diag(np.einsum('ii->i',(self.hks[:, basis_, :][:, :, basis_] - hdc)[0,:,:], optimize=True)).reshape(1, len(basis_), len(basis_))
            for ispin in range(self.nspin):
                for i in range(len(basis_)):
                    h1e[ispin, i, i] -= mu
        else:
            h1e = np.zeros((self.nspin,len(basis__sigma),len(basis__sigma),2,self.n_spectralf))
            for i in range(2):
                if i == 0:
                    for i_spectralf in range(self.n_spectralf):
                        h1e[:,:,:,i,i_spectralf] = self.hks[:, basis_, :][:, :, basis_]
            h1e -= hdc

        # print('h1e ===================')
        # print(self.compute_vh_from_eri(basis_, eri))
        # print(h1e * hartree_to_ev)
        # print(h1e[0,:,:,0,810] * hartree_to_ev )
        # print(self.vxc[:, basis_, :][:, :, basis_] * hartree_to_ev)
        # print(self.sigmac_n_e[0,:,:,1,810] * hartree_to_ev)

        return h1e

    def make_heff_with_eri(self,
                           basis: List[int],
                           eri: np.ndarray,
                           dc: str = "hf",
                           point_group_rep: PointGroupRep = None,
                           npdep_to_use: int = None,
                           nspin: int = 1,
                           symmetrize: Dict[str, bool] = {}) -> Heff:
        """ Construct effective Hamiltonian based on ERI.

        Args:
            basis: list of band indices for orbitals in the active space.
            eri: ERI.
            dc: scheme for computing double counting.
            point_group_rep: representation of
            npdep_to_use: # of PDEP basis to use.
            nspin: # of spin channels.
            symmetrize: arguments for symmetrization function of Heff.

        Returns:
            effective Hamiltonian.
        """
        # print(basis)
        # print(self.occ)
        h1e = self.compute_h1e_from_hks(basis, eri, dc=dc, npdep_to_use=npdep_to_use)

        if self.nspin == 2 and nspin == 1:
            # Rotate spin down orbitals to maximally resemble spin up orbitals
            # Then symmetrize spin up and spin down matrix elements (i.e. apply time reversal symmetry)
            # Orthogonal Procrustes problem as defined on Wikipedia:
            # denote A = psi^{dw}_iG, B = psi^{up}_jG', the rotation matrix for spin down orbitals is
            # R = argmin_Omega || Omega A - B ||_F subject to Omega^T Omega = I
            # let M = B A^T, which is exactly the overlap s1e[0, 1], the SVD of M is
            # M = U Sigma V^T
            # then R is computed as R = U V^T

            basis_ = np.array(basis) - self.nbndstart

            S = self.s1e[0, 1][basis_,:][:,basis_]
            u, s, vt = np.linalg.svd(S)
            R = u @ vt  # R = < psi^old_i | psi^new_j >
            h1e_up = h1e[0]
            h1e_dw = h1e[1]
            h1e_dw_r = np.einsum("pq,pi,qj->ij", h1e_dw, R, R)  # rotate spin down orbitals
            eri_up = eri[0, 0]
            eri_dw = eri[1, 1]
            # rotate spin down orbitals
            eri_dw_r = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_dw, R, R, R, R, optimize=True)
            h1e = (h1e_up + h1e_dw_r) / 2
            eri = (eri_up + eri_dw_r) / 2

        if point_group_rep is None and self.point_group is not None:
            orbitals = [
                VData(f"{self.path}/west.westpp.save/wfcK000001B{i:06d}.cube", normalize="sqrt")
                for i in basis
            ]
            if self.projector_type != 'K':
                orbitals += [
                            VData(f"wannier_{i:05d}.xsf", normalize=True)
                            for i in self.local_projectors
                        ]
            point_group_rep, _ = self.point_group.compute_rep_on_orbitals(orbitals, orthogonalize=True)

        heff = Heff(h1e, eri, point_group_rep=point_group_rep)
        heff.symmetrize(**symmetrize)
        return heff

    def make_heffs(self,
                   basis_name: str = "",
                   basis: Union[Dict, List[int]] = None,
                   Ws: List[str] = None,
                   npdep_to_use: int = None,
                   chi0a_fortran: bool = False,
                   dc: str = "hf",
                   nspin: int = 1,
                   symmetrize: Dict[str, bool] = {},
                   run_fci_inplace: bool = False,
                   nroots: int = 10,
                   verbose: bool = True) -> Union[pd.DataFrame, Dict[str, Heff]]:
        """ Build effective Hamiltonians for given active space.

        The highest level function of CGWResults class. Call self.make_heff to build
        effective Hamiltonians for given set of W. Can run FCI calculations in place.

        Args:
            basis_name: name of basis used to distinguish with calculations using different
                        active spaces. basis_name will be included in the resulting dataframe.
            basis: list of band indices for orbitals in the active space.
            Ws: list of names for W.
            npdep_to_use: # of PDEP basis to use.
            dc: scheme for computing double counting.
            nspin: # of spin channels.
            symmetrize: arguments for symmetrization function of Heff.
            run_fci_inplace: if True, run FCI calculations and return pd.DataFrame that summarize
                             FCI results, otherwise return a dict of Heff.
            nroots: # of roots for FCI calculations.
            verbose: if True, print detailed info for FCI calculations.
        """
        if self.projector_type not in ('B','R'):
            if basis is None:
                # Default: use all ks_projectors
                basis_indices = self.ks_projectors
                basis_labels = [""] * len(basis_indices)
            elif isinstance(basis, dict):
                # basis = {label: indices}
                basis_labels = []
                basis_indices = []
                for label, indices in basis.items():
                    basis_labels.extend([label] * len(indices))
                    basis_indices.extend(indices)
            else:
                # basis = [indices]
                basis_indices = basis
                basis_labels = [""] * len(basis_indices)

            basis_name = basis_name
            if not basis_name:
                idx1, idx2 = basis_indices[0], basis_indices[-1]
                if np.all(basis_indices == np.arange(idx1, idx2+1)):
                    basis_name = f'{idx1}-{idx2}'

        # basis_ = np.array(basis_indices) - self.nbndstart
            basis = basis_indices  # basis_indices here is the "basis" variable for other functions

        # calculate basis_
        if self.projector_type in ('K','M'):
            # print(self.range)
            if self.range == True:
                if basis is None:
                    basis_ = self.ks_projectors - self.nbndstart
                else:
                    basis_ = np.array(basis) - self.nbndstart
                if npdep_to_use is None:
                    npdep_to_use = self.npdep
            else:
                if basis is None:
                    basis_ = np.array(range(len(self.ks_projectors)))
                else:
                    basis_ = []
                    for i in basis:
                        basis_ += np.argwhere(self.ks_projectors == i)[0].tolist()
                    basis_ = np.array(basis_)
            if self.cgw_calculation == 'G' and self.projector_type == 'K':
                if 'sigma' in dc or 'hopping' in dc:
                    basis__sigma = []
                    for i in basis:
                        basis__sigma += np.argwhere(self.ks_projectors_sigma == i)[0].tolist()
                    basis__sigma = np.array(basis__sigma)
            else:
                basis__sigma = basis_

        if self.projector_type in ('B','R','M'):
            local_basis_ = np.array(range(len(self.local_projectors)))
            if 'basis_' in dir():
                basis_ = np.append(local_basis_,basis_+len(local_basis_))
            else:
                basis_ = local_basis_

        if npdep_to_use is None:
            npdep_to_use = self.npdep

        Vc = self.Vc[:,:,basis_,:,:,:][:,:,:,basis_,:,:][:,:,:,:,basis_,:][:,:,:,:,:,basis_]

        Wdict = self.compute_Ws(basis=basis, chi0a_fortran=chi0a_fortran, npdep_to_use=npdep_to_use)

        if dc == 'print_hopping':
            return Vc + Wdict['Wrp_rpa']

        if Ws == ['Bare']:
            pass
        else:
            if Ws is None:
                Ws = self.Ws_all
            assert all(key in Wdict for key in Ws)

        if self.point_group is None:
            point_group_rep = None
        else:
            orbitals = [
                VData(f"{self.path}/west.westpp.save/wfcK000001B{i:06d}.cube", normalize="sqrt")
                for i in basis
            ]
            if self.projector_type != 'K':
                orbitals += [
                            VData(f"wannier_{i:05d}.xsf", normalize=True)
                            for i in self.local_projectors
                        ]
            point_group_rep, orbital_symms = self.point_group.compute_rep_on_orbitals(orbitals, orthogonalize=True)

        if Ws == ['Bare']:
            heffs = {
                W:self.make_heff_with_eri(
                    basis=basis, eri=Vc, dc=dc,
                    npdep_to_use=npdep_to_use, nspin=nspin,
                    symmetrize=symmetrize, point_group_rep=point_group_rep
                )
                for W in Ws
            }
        else:
            heffs = {
                W: self.make_heff_with_eri(
                    basis=basis, eri=Vc + Wdict[W], dc=dc,
                    npdep_to_use=npdep_to_use, nspin=nspin,
                    symmetrize=symmetrize, point_group_rep=point_group_rep
                )
                for W in Ws
            }

        if run_fci_inplace:
            if not verbose:
                # mute all print functions
                stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

            columns = ["basis_name", "basis", "nelec", "nspin", "eri", "dc", "npdep"]
            for i in range(nroots):
                columns.extend([f"ev{i}", f"mult{i}", f"symm{i}"])

            df = pd.DataFrame(columns=columns)

            print("===============================================================")
            print("Building effective Hamiltonian...")
            if basis_name:
                print(f"basis_name: {basis_name}")
            print(f"nspin: {nspin}, double counting: {dc}")
            if self.projector_type in ('M','B','R'):
                print(f"local_projectors: {self.local_projectors}")
            if self.projector_type in ('K','M'):
                print(f"ks_projectors: {basis}")
            # print(f"ks_projectors: {basis}")
            if self.projector_type == 'K':
                print(f"ks_eigenvalues: {self.egvs[:, basis_] * hartree_to_ev}")
            # print(f"ks_occupations: {self.occ[:, basis_]}")
            print(f"occupations: {self.occ[:, basis_]}")
            print(f"npdep_to_use: {npdep_to_use}")
            print("===============================================================")

            nel = np.sum(self.occ[:,basis_])
            # print(nel,int(round(nel))//2,int(round(nel))//2)
            nelec = (int(round(nel))//2, int(round(nel))//2)

            for W, heff in heffs.items():
                data = {
                    "basis_name": basis_name,
                    "basis": basis,
                    "nelec": nelec,
                    "nspin": nspin,
                    "eri": W,
                    "dc": dc,
                    "npdep": npdep_to_use,
                }

                print("-----------------------------------------------------")
                print("FCI calculation using ERI:", W)

                fcires = heff.FCI(nelec=nelec, nroots=nroots)

                # print(fcires['excitations'])

                # print results
                print(f"{'#':>2}  {'ev':>5} {'term':>4} diag[1RDM - 1RDM(GS)]")
                print(f"{'':>15}" + " ".join(f"{b:>4}" for b in basis_))
                ispin = 0
                if self.projector_type == 'K':
                    print(f"{'':>15}" + " ".join(f"{self.egvs[ispin, b]*hartree_to_ev:>4.1f}" for b in basis_))
                if self.point_group is not None:
                    print(f"{'':>15}" + " ".join(f"{s.partition('(')[0]:>4}" for s in orbital_symms))
                if self.projector_type not in ('K','R'):
                    if any(basis_labels):
                        print(f"{'':>15}" + " ".join(f"{label:>4}" for label in basis_labels))
                for i, (ev, mult, symm, ex) in enumerate(zip(
                        fcires["evs"], fcires["mults"], fcires["symms_maxproj"], fcires["excitations"]
                )):
                    symbol = f"{int(round(mult))}{symm.partition('(')[0]}"
                    exstring = " ".join(f"{e:>4.1f}" for e in ex)
                    print(f"{i:>2}  {ev:.3f} {symbol:>4} {exstring}")

                    data.update({
                        f"ev{i}": ev,
                        f"mult{i}": mult,
                        f"symm{i}": symm,
                    })

                print("-----------------------------------------------------")

                df = df.append(data, ignore_index=True)

            if not verbose:
                # mute all print functions
                sys.stdout.close()
                sys.stdout = stdout

            return df
        else:
            return heffs

