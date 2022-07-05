from typing import Dict, List, Tuple, Optional, Union
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy
from lxml import etree
from westpy.qdet.misc import find_indices, parse_one_value, find_index
from westpy.qdet.misc import VData, rydberg_to_hartree, ev_to_hartree, hartree_to_ev
from westpy.qdet.heff import Heff
from westpy.qdet.symm import PointGroup, PointGroupRep
from westpy.qdet.west_output import WstatOutput


class QDETResults:

    ev_thr = 0.001 * ev_to_hartree  # threshold for determining degenerate states
    occ_thr = 0.001  # threshold for determining equal occupations

    def __init__(self,
                 path: str,
                 occ: Optional[np.ndarray] = None,
                 eps_infty: Optional[float] = None,
                 overlap_basis: Optional[Union[Dict, List[int]]] = None,
                 point_group: Optional[PointGroup] = None,
                 verbose: Optional[bool] = True):
        """ Parser of constrained GW calculations.

        Args:
            path: directory where pw, wstat and wfreq calculations are performed.
            occ: occupation numbers. Read from pw results if not defined.
            eps_infty: user-defined epsilon infinity.
            point_group: point group of the system.
            verbose: if True, self.write a summary.
        """

        self.path = path
        self.eps_infty = eps_infty
        self.point_group = point_group

        # read general info from JSON file
        self.__read_wfreq_json(f"{self.path}/west.wfreq.save/wfreq.json")
        
        # read data from wfreq.out
        self.__read_wfreq_out(f"{self.path}/wfreq.out")

        # read occupation numbers and Kohn-Sham eigenvalues
        self.occ = np.zeros([self.nspin, self.nproj])
        self.egvs = np.zeros([self.nspin, self.nproj])
        
        # if occupation is not specified, read occupation from QE pwscf.save
        if occ is None:
            self.__read_pw_xml(f"{path}/pwscf.save/K00001/")
        else:
            self.write("Warning: user-defined occupation!")
            self.occ[...] = occ

        # 1e density matrix
        self.dm = np.zeros([self.nspin, self.nproj, self.nproj])
        for ispin in range(self.nspin):
            self.dm[ispin, ...] = np.diag(self.occ[ispin])
        self.nel = int(np.sum(self.dm))

        # 1 electron Hamiltonian elements
        self.parse_h1e()

        self.qps = np.zeros((self.nspin,self.nproj))
        for ispin in range(self.nspin):
            self.qps[ispin,:] = np.einsum("ii->i", self.qp_energy_n[ispin,:,:], optimize=True)

        # read bare Coulomb potential Vc from file
        self.Vc = self.parse_eri("/west.wfreq.save/vc.dat")

        # braket and overlap
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

        if verbose:
            self.print_summary()

    def print_summary(self):
        """ Print a summary after parsing CGW results. """
        self.write("---------------------------------------------------------------")
        self.write("CGW Results General Info")
        self.write(f"path: {self.path}")
        self.write(f"nspin = {self.nspin}, nel = {self.nel}, nproj = {self.nproj}, npdep = {self.npdep}")
        if self.point_group is not None:
            self.write(f"point group: {self.point_group.name}")
        if self.eps_infty is not None:
            self.write(f"eps_infinity from input: {self.eps_infty}")
        self.write(f"ks_projectors: {self.ks_projectors}")
        
        self.write("occupations:")
        self.write(self.occ)
        self.write("max|chi0a - chi0a_ref| = {:.3f}".format(
            np.max(np.abs(self.compute_chi0a() - self.chi0a_ref))
            ))
        self.write("---------------------------------------------------------------")

    def compute_chi0a(self) -> np.ndarray:
        """ Compute chi0^a (chi0 projected into active space).

        Returns:
            chi0a defined on PDEP basis.
        """

        overlap = self.overlap[..., list(range(self.npdep)) + [-3, -2, -1]]

        # Summation over state (SOS) / Adler-Wiser expression
        chi0a = np.zeros([self.npdep + 3, self.npdep + 3])
        for ispin in range(self.nspin):
            for i in self.basis:
                for j in self.basis:
                    if i >= j:
                        continue
                    ei, ej = self.egvs[ispin, i], self.egvs[ispin, j]
                    fi, fj = self.occ[ispin, i], self.occ[ispin, j]
                    if abs(ei - ej) < self.ev_thr:
                        assert (fi - fj) < self.occ_thr
                        continue

                    prefactor = 2 * (fi - fj) / (ei - ej)
                    
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

    def compute_Ws(self, Ws: str) -> np.ndarray:
        """ Compute unconstrained Wp (independent of active space) and
        constrained Wrp of a certain active space.

        Args:
            basis: list of band indices for the orbitals defining the active space.
            npdep_to_use: # of PDEP basis to use.

        Returns:
            a dictionary of W (4-index array)
        """
        npdep_to_use = self.npdep

        chi0 = self.extract(self.chi0, self.npdep)

        chirpa = self.solve_dyson_with_identity_kernel(chi0)
        
        # fully screened potential
        if Ws.split('_')[0] == 'Wp':
            # random phase approximation
            if Ws.split('_')[1] == 'rpa':
                wps = chirpa
            # unscreened
            elif Ws.split('_')[1] == 'zero':
                wps = (0, np.zeros([npdep_to_use, npdep_to_use]))
        # partially screened potential
        elif Ws.split('_')[0] == 'Wrp':
            chi0a = self.compute_chi0a()
            # TODO: Why are not just extracting?
            #chi0a = self.extract(self.chi0a_ref, npdep_to_use=npdep_to_use)
            chi0r = chi0 - chi0a
            
            # random-phase approximation
            if Ws.split('_')[1] == 'rpa':
                wps = self.solve_dyson_with_identity_kernel(chi0r)
            
        # compute ERI from W in PDEP basis
        Ws = self.solve_eri(h=wps[0] if self.eps_infty is None else (1.0 / self.eps_infty - 1.0),
                            B=wps[1])

        return Ws

    def parse_h1e(self):
        """ Read 1e terms in the KS hamiltonian. """
        self.s1e = np.fromfile(f"{self.path}/west.wfreq.save/s1e.dat", dtype=float).reshape(
            self.nspin, self.nspin, self.nproj, self.nproj
        )
        
        checklist = ["kin", "vloc", "vnl", "vh", "vxc", "vxx", "hks"]            
        for h in checklist:
            setattr(self, h, np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj, self.nproj))

        self.parse_sigma()

    def parse_sigma(self):

        checklist_cgw = [f"sigmax_n_a",f"sigmax_n_e",f"sigmax_n_f"]
        for h in checklist_cgw:
            tmp = np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj, self.nproj)
            setattr(self, h, tmp)
            
        checklist_cgw = [f"qp_energy_n"]
        for h in checklist_cgw:
            tmp1 = np.fromfile(
                    f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj)
            tmp = np.zeros([self.nspin, self.nproj, self.nproj], dtype=float)
            for ispin in range(self.nspin):
                tmp[ispin, ...] = np.diag(tmp1[ispin, ...]) 
            setattr(self, h, tmp)

        checklist_cgw = [f"sigmac_eigen_n_a",f"sigmac_eigen_n_e",f"sigmac_eigen_n_f"]
        for h in checklist_cgw:
            tmp = np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj, self.nproj)
            setattr(self, h, tmp)

        checklist_cgw = [f"sigmac_n_a",f"sigmac_n_e",f"sigmac_n_f"]

        for h in checklist_cgw:
            tmp = np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj, 3, self.n_spectralf)
            tmp1 = np.zeros((self.nspin, self.nproj, self.nproj, 3, self.n_spectralf))
            for i_spectralf in range(self.n_spectralf):
                for index in range(3):
                    for ispin in range(self.nspin):
                        for iproj in range(self.nproj):
                            tmp1[ispin,iproj,iproj,index,i_spectralf] = tmp[ispin,iproj,index,i_spectralf]
            setattr(self, h, tmp1*ev_to_hartree) 

        basis_ = []
        for i in self.ks_projectors:
            basis_ += np.argwhere(self.ks_projectors == i)[0].tolist()
        basis_ = np.array(basis_)

        return

    def write(self, *args):

        data = ''
        for i in args:
            data += str(i)
            data += ' '
        data = data[:-1]
        print(data)

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
                  B: np.ndarray) -> np.ndarray:
        """ Compute ERI of given W (defined by head h and body B) on a basis of KS orbitals.

        Args:
            h: head.
            B: body.

        Returns:
            W, represented as a 4-index array.
        """
        nspin = self.nspin

        neff = len(self.basis)
        npair, pijmap, ijpmap = self.make_index_map(neff)
        assert npair <= self.npair

        braket_pair_eff = np.zeros([self.nspin, npair, self.npdep])
        for ispin in range(self.nspin):
            for p, (i, j) in enumerate(pijmap):
                iproj, jproj, = self.basis[[i, j]]
                braket_pair_eff[ispin, p, :] = self.braket[ispin, iproj, jproj, :self.npdep]
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

    def compute_vh_from_eri(self, eri: np.ndarray) -> np.ndarray:
        """ Compute VHartree from ERI.
        
        Args:
            eri: ERI.

        Returns:
            VHartree matrix in active space.
        """
        return np.einsum("stijkl,tkl->sij", eri, self.dm[:, self.basis, :][:, :, self.basis], optimize=True)

    def compute_vxx_from_eri(self, eri: np.ndarray) -> np.ndarray:
        """ Compute VEXX from ERI.

        Args:
            eri: ERI.

        Returns:
            VEXX matrix in active space.
        """
        return - 0.5 * self.nspin * np.einsum("ssikjl,skl->sij", eri, self.dm[:, self.basis, :][:, :, self.basis], optimize=True)

    def compute_h1e_from_hks(self,
                             eri: np.ndarray,
                             dc: str = "exact",
                             mu: float = 0,
                             sigma: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """ Compute 1e term of effective Hamiltonian from KS Hamiltonian.

        Args:
            basis: list of band indices for orbitals in the active space.
            eri: ERI.
            dc: scheme for computing double counting.
            mu: chemical potential, used to shift the spectrum of resulting effective Hamiltonian.

        Returns:
            1e part of effective Hamiltonian.
        """

        braket = self.braket[:, self.basis, :, :][:, :, self.basis, :]
        occ = self.occ[:, self.basis]
        
        # calculate double-counting term
        if dc == "hf":
            hdc = self.compute_vh_from_eri(eri) + self.compute_vxx_from_eri(eri)
        elif dc == 'exact':
            hdc = self.compute_vh_from_eri(eri) \
                + self.vxc[:, self.basis, :][:, :, self.basis] + self.vxx[:, self.basis, :][:, :, self.basis]\
                - getattr(self,'sigmax_n_e')[:, self.basis, :][:, :, self.basis]\
                - getattr(self,'sigmac_eigen_n_e')[:, self.basis, :][:, :, self.basis]
        else:
            raise ValueError("Unknown double counting scheme")
        
        # subtract double counting from Kohn-Sham eigenvalues
        h1e = self.hks[:, self.basis, :][:, :, self.basis] - hdc
        # subtract chemical potential from diagonal terms
        for ispin in range(self.nspin):
            for i in range(len(self.basis)):
                h1e[ispin, i, i] -= mu

        return h1e

    def make_heffs(self,
                    basis_name: str = "",
                    Ws: str = "Wrp_rpa",
                    dc: str = "exact",
                    nelec: Tuple = None,
                    symmetrize: Dict[str, bool] = {},
                    run_fci_inplace: bool = False,
                    nroots: int = 10,
                    sigma: Optional[List[np.ndarray]] = None,
                    return_fcires: bool = False,
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
            nelec: # of electrons in each spin-channel
            symmetrize: arguments for symmetrization function of Heff.
            run_fci_inplace: if True, run FCI calculations and return pd.DataFrame that summarize
                             FCI results, otherwise return a dict of Heff.
            nroots: # of roots for FCI calculations.
            verbose: if True, self.write detailed info for FCI calculations.
        """
        basis_indices = self.ks_projectors
        basis_labels = [""] * len(basis_indices)
        
        # Set name for basis
        # TODO: unnecessary
        if not basis_name:
            idx1, idx2 = basis_indices[0], basis_indices[-1]
            if np.all(basis_indices == np.arange(idx1, idx2+1)):
                basis_name = f'{idx1}-{idx2}'
        else:
            basis_name = basis_name
        
        npdep_to_use = self.npdep

        Vc = self.Vc[:,:,self.basis,:,:,:][:,:,:,self.basis,:,:][:,:,:,:,self.basis,:][:,:,:,:,:,self.basis]
        # calculate screened electron repulsion integrals
        W = self.compute_Ws(Ws)

        # determine point-group representation
        if self.point_group is None:
            point_group_rep = None
        else:
            orbitals = [
                VData(f"{self.path}/west.westpp.save/wfcK000001B{i:06d}.cube", normalize="sqrt")
                for i in self.ks_projectors
            ]
            point_group_rep, orbital_symms = self.point_group.compute_rep_on_orbitals(orbitals, orthogonalize=True)

        if Ws == 'Bare':
            h1e = self.compute_h1e_from_hks(eri=Vc, dc=dc, sigma=sigma)
            heff = Heff(h1e, eri=Vc, point_group_rep=point_group_rep)
        else:
            h1e = self.compute_h1e_from_hks(eri=Vc + W, dc=dc, sigma=sigma)
            heff = Heff(h1e, eri=Vc + W, point_group_rep=point_group_rep)
            
        heff.symmetrize(**symmetrize)
            
        if run_fci_inplace:
            if not verbose:
                # mute all self.write functions
                stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

            columns = ["basis_name", "basis", "nelec", "nspin", "eri", "dc", "npdep"]
            for i in range(nroots):
                columns.extend([f"ev{i}", f"mult{i}", f"symm{i}"])

            df = pd.DataFrame(columns=columns)

            self.write("===============================================================")
            self.write("Building effective Hamiltonian...")
            if basis_name:
                self.write(f"basis_name: {basis_name}")
            self.write(f"nspin: {self.nspin}, double counting: {dc}")
            self.write(f"ks_eigenvalues: {self.egvs[:, self.basis] * hartree_to_ev}")
            self.write(f"occupations: {self.occ[:, self.basis]}")
            self.write(f"npdep_to_use: {self.npdep}")
            self.write("===============================================================")

            if nelec == None:
                nel = np.sum(self.occ[:,self.basis])
                nelec = (int(round(nel))//2, int(round(nel))//2)

            data = {
                "basis_name": basis_name,
                "basis": self.ks_projectors,
                "nelec": nelec,
                "nspin": self.nspin,
                "eri": Ws,
                "dc": dc,
            }

            self.write("-----------------------------------------------------")
            self.write("FCI calculation using ERI:", Ws)

            fcires = heff.FCI(nelec=nelec, nroots=nroots)

            self.write(f"{'#':>2}  {'ev':>5} {'term':>4} diag[1RDM - 1RDM(GS)]")
            self.write(f"{'':>15}" + " ".join(f"{b:>4}" for b in self.basis))
            ispin = 0
            self.write(f"{'':>15}" + " ".join(f"{self.egvs[ispin,b]*hartree_to_ev:>4.1f}" for b in self.basis))
            if self.point_group is not None:
                self.write(f"{'':>15}" + " ".join(f"{s.partition('(')[0]:>4}" for s in orbital_symms))
            for i, (ev, mult, symm, ex) in enumerate(zip(
                    fcires["evs"], fcires["mults"], fcires["symms_maxproj"], fcires["excitations"]
            )):
                symbol = f"{int(round(mult))}{symm.partition('(')[0]}"
                exstring = " ".join(f"{e:>4.1f}" for e in ex)
                self.write(f"{i:>2}  {ev:.3f} {symbol:>4} {exstring}")

                data.update({
                    f"ev{i}": ev,
                    f"mult{i}": mult,
                    f"symm{i}": symm,
                })

            self.write("-----------------------------------------------------")

            df = df.append(data, ignore_index=True)

            if not verbose:
                # mute all self.write functions
                sys.stdout.close()
                sys.stdout = stdout

            if return_fcires:
                return fcires
            else:
                return df
        else:
            return heff
    
    def __read_wfreq_json(self, filename):
        """The function reads parameters from JSON file and stores them in class
        variables.
        Args:
            filename: filename of the wfreq JSON output file. 
        """
        self.js = json.load(open(filename))
        self.nspin = self.js["system"]["electron"]["nspin"]
        self.npdep = self.js["input"]["wfreq_control"]["n_pdep_eigen_to_use"]
        self.omega = self.js["system"]["cell"]["omega"]
        self.l_enable_lanczos = self.js["input"]["wfreq_control"]["l_enable_lanczos"]

        self.cgw_calculation = self.js["input"]["cgw_control"]["cgw_calculation"]
        self.h1e_treatment = self.js["input"]["cgw_control"]["h1e_treatment"]
        self.nproj = 0
        
        # read data on Kohn-Sham active space
        nbndstart, nbndend = np.array(self.js['input']['cgw_control']['ks_projector_range'], dtype=int)
        if nbndstart != 0:
            self.ks_projectors = np.arange(nbndstart, nbndend + 1)
        else:
            self.ks_projectors = np.array(self.js['input']['cgw_control']['ks_projectors'], dtype=int)
        self.nproj = len(self.ks_projectors)
        
        # generate basis from Kohn-Sham projectors
        self.basis = np.array(range(len(self.ks_projectors)))
        # generate pairs of Kohn-Sham indices and mappings
        self.npair, self.pijmap, self.ijpmap = self.make_index_map(self.nproj)

        return

    def __read_wfreq_out(self, filename):
        """ Reads parameters from wfreq.out and stores them in class variables.
        Args:
            filename: filename of the wfreq.out file
        """

        wfoutput = open(filename).readlines()
        i = find_index("Divergence =", wfoutput)
        self.div = parse_one_value(float, wfoutput[i])
        i = find_index("n_spectralf", wfoutput)
        self.n_spectralf = parse_one_value(int, wfoutput[i])
        i = find_index("n_imfreq", wfoutput)
        self.n_imfreq = parse_one_value(int, wfoutput[i])
        i = find_index("n_refreq", wfoutput)
        self.n_refreq = parse_one_value(int, wfoutput[i])
        i = find_index("n_lanczos", wfoutput)
        self.n_lanczos = parse_one_value(int, wfoutput[i])
        i = find_index("nbnd", wfoutput)
        self.nbnd = parse_one_value(int, wfoutput[i])
        i = find_index("ecut_imfreq", wfoutput)
        self.ecut_imfreq = parse_one_value(float, wfoutput[i])
        i = find_index("ecut_refreq", wfoutput)
        self.ecut_refreq = parse_one_value(float, wfoutput[i])
        i = find_index("nelec", wfoutput)
        self.nelec = parse_one_value(float, wfoutput[i])
        #
        i = find_index("PBE0", wfoutput)
        if i != None:
            self.xc = 'ddh'
        else:
            self.xc = 'pbe'

        return
    def __read_pw_xml(self, path):
        """ Read Kohn-Sham eigenvalues and occupations from Quantum Espresso
        XML files and store it in class variables.
        Args:
            path: directory that contains the eigenval.xml or set of
            eigenval*.xml files.
        """

        for ispin in range(self.nspin):
            if self.nspin == 1:
                xmlroot = etree.parse(path+"eigenval.xml")
            else:
                xmlroot = etree.parse(path+"eigenval{ispin+1}.xml")
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
