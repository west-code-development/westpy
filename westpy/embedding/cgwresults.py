from typing import Dict, List, Tuple, Optional, Union
import sys
import os
import numpy as np
import pandas as pd
import json
from lxml import etree
from .misc import parse_one_value, find_index, VData, rydberg_to_hartree, ev_to_hartree, hartree_to_ev
from .heff import Heff
from .symm import PointGroup, PointGroupRep
from .west_output import WstatOutput


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

        self.nbndstart, self.nbndend = np.array(self.js['input']['cgw_control']['ks_projector_range'], dtype=int)
        self.ks_projectors_ = np.arange(self.nbndstart, self.nbndend + 1)
        self.nproj = len(self.ks_projectors)
        self.npair, self.pijmap, self.ijpmap = self.make_index_map(self.nproj)

        wfoutput = open(f"{self.path}/wfreq.out").readlines()
        i = find_index("Divergence =", wfoutput)
        self.div = parse_one_value(float, wfoutput[i])

        # occupation numbers and eigenvalues
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
        print(f"ks_projectors: {self.ks_projectors}")
        print("occupations:")
        print(self.occ)
        # print(f"max|p - p_ref| = {np.max(np.abs(self.p - self.p_ref)):.3f}")
        # chi_ref = self.solve_dyson_with_identity_kernel(self.p_ref)
        # print("max|dyson(p) - dyson(p_ref)| = {:.3f}, {:.3f}".format(
        #     np.abs(self.chi[0] - chi_ref[0]), np.max(np.abs(self.chi[1] - chi_ref[1]))
        # ))
        print("max|chi0a - chi0a_ref| = {:.3f}".format(
            np.max(np.abs(self.compute_chi0a(basis=self.ks_projectors) - self.chi0a_ref))
        ))
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
        if basis is None:
            basis_ = self.ks_projectors - self.nbndstart
        else:
            basis_ = np.array(basis) - self.nbndstart

        if npdep_to_use is None:
            npdep_to_use = self.npdep
        overlap = self.overlap[..., list(range(npdep_to_use)) + [-3, -2, -1]]

        # Summation over state (SOS) / Adler-Wiser expression
        chi0a = np.zeros([npdep_to_use + 3, npdep_to_use + 3])
        for ispin in range(self.nspin):
            for i in basis_:
                for j in basis_:
                    if i >= j:
                        continue
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

    def compute_Ws(self, basis: List[int] = None, npdep_to_use: int = None) -> Dict[str, np.ndarray]:
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
        chi0a = self.compute_chi0a(basis=basis, npdep_to_use=npdep_to_use)

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
        for h in ["kin", "vloc", "vnl", "vh", "vxc", "vxx", "hks"]:
            setattr(self, h, np.fromfile(
                f"{self.path}/west.wfreq.save/{h}.dat", dtype=float
            ).reshape(self.nspin, self.nproj, self.nproj))

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
        if basis is None:
            basis_ = self.ks_projectors - self.nbndstart
        else:
            basis_ = np.array(basis) - self.nbndstart
        if npdep_to_use is None:
            npdep_to_use = self.npdep

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
        if basis is None:
            basis_ = self.ks_projectors - self.nbndstart
        else:
            basis_ = np.array(basis) - self.nbndstart
        if npdep_to_use is None:
            npdep_to_use = self.npdep

        braket = self.braket[:, basis_, :, :][:, :, basis_, :]
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
        else:
            raise ValueError("Unknown double counting scheme")

        h1e = self.hks[:, basis_, :][:, :, basis_] - hdc

        for ispin in range(self.nspin):
            for i in range(len(basis_)):
                h1e[ispin, i, i] -= mu

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
            point_group_rep, _ = self.point_group.compute_rep_on_orbitals(orbitals, orthogonalize=True)

        heff = Heff(h1e, eri, point_group_rep=point_group_rep)
        heff.symmetrize(**symmetrize)
        return heff

    def make_heffs(self,
                   basis_name: str = "",
                   basis: Union[Dict, List[int]] = None,
                   Ws: List[str] = None,
                   npdep_to_use: int = None,
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

        basis_ = np.array(basis_indices) - self.nbndstart
        basis = basis_indices  # basis_indices here is the "basis" variable for other functions

        if npdep_to_use is None:
            npdep_to_use = self.npdep

        Vc = self.Vc[:,:,basis_,:,:,:][:,:,:,basis_,:,:][:,:,:,:,basis_,:][:,:,:,:,:,basis_]

        Wdict = self.compute_Ws(basis=basis, npdep_to_use=npdep_to_use)

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
            point_group_rep, orbital_symms = self.point_group.compute_rep_on_orbitals(orbitals, orthogonalize=True)

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
            print(f"ks_projectors: {basis}")
            print(f"ks_eigenvalues: {self.egvs[:, basis_] * hartree_to_ev}")
            print(f"ks_occupations: {self.occ[:, basis_]}")
            print(f"npdep_to_use: {npdep_to_use}")
            print("===============================================================")

            nel = np.sum(self.occ[:,basis_])
            nelec = (int(nel//2), int(nel//2))

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

                # print results
                print(f"{'#':>2}  {'ev':>5} {'term':>4} diag[1RDM - 1RDM(GS)]")
                print(f"{'':>15}" + " ".join(f"{b:>4}" for b in basis_))
                ispin = 0
                print(f"{'':>15}" + " ".join(f"{self.egvs[ispin, b]*hartree_to_ev:>4.1f}" for b in basis_))
                if self.point_group is not None:
                    print(f"{'':>15}" + " ".join(f"{s.partition('(')[0]:>4}" for s in orbital_symms))
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
