import numpy as np
import pandas as pd
from IPython.display import display

from pyscf.fci.cistring import make_strings, num_strings
from pyscf.fci.addons import transform_ci_for_orbital_rotation
from pyscf.fci import direct_uhf
from typing import Tuple

from westpy import eV, Hartree
from .json_parser import (
    read_parameters,
    read_occupation,
    read_matrix_elements,
    read_qp_energies,
    read_overlap,
)
from .misc import spin_square_spin_polarized


class eBSEResult:
    def __init__(self, filename: str, spin_flip_: bool = False):
        """Parser for embedded Bethe-Salpeter Equation (eBSE) calculations.

        Args:
            filename (str): name of the JSON file that contains the output of the
                WEST calculation.
            spin_flip (boolean): trigger for spin-conserving (False) or
                spin-flip (True) calculation.
        """

        self.filename = filename
        # read QDET active space from file
        self.nspin, self.npair, self.basis = read_parameters(filename)
        assert self.nspin == 2
        # read QP energies and occupation from file
        self.qp_energies = read_qp_energies(self.filename)
        self.occupation = read_occupation(self.filename)

        # make global arrays
        self.v = read_matrix_elements(self.filename, string="eri_w")[1]
        self.w = read_matrix_elements(self.filename, string="eri_w_full")[1]

        self.spin_flip = spin_flip_

        # get size of single-particle space
        self.norb = self.basis.shape[0]
        # get number of electrons
        self.nelec = [int(np.sum(self.occupation[0])), int(np.sum(self.occupation[1]))]

        # read overlap matrix from file
        self.ovlpab = read_overlap(filename)

        # create mapping between transitions and single-particle indices
        self.smap = self._get_smap()
        self.n_tr = self.smap.shape[0]

        # create mapping between transitions and FCI vectors
        self.cmap, self.jwstring = self._get_map_transitions_to_cistrings()

    def _write(self, *args):
        data = ""
        for i in args:
            data += str(i)
            data += " "
        data = data[:-1]
        print(data)

    def _get_smap(self):
        """Creates a map between the transition index s and the combination of
        KS indices (v,c) of the valence state v and conduction state c and the
        spin index m. The format is smap[s] = (v,c,m).
        """

        smap_ = []

        if not self.spin_flip:
            smap_.append([0, 0, 0])  # ground state, represented by no transition
            # loop over spin
            for m in range(2):
                # loop over occupied states
                for v in range(self.occupation.shape[1]):
                    if self.occupation[m][v] == 1.0:
                        # loop over conduction states
                        for c in range(self.occupation.shape[1]):
                            if self.occupation[m][c] == 0.0:
                                smap_.append([v, c, m])
        else:
            # for spin-flip BSE only transitions from spin-up to spin-down are
            # considered
            # loop over occupied states
            for v in range(self.occupation.shape[1]):
                if self.occupation[0][v] == 1.0:
                    # loop over conduction states
                    for c in range(self.occupation.shape[1]):
                        if self.occupation[1][c] == 0.0:
                            smap_.append([v, c, 0])

        return np.asarray(smap_)

    def solve(self, verbose=True):
        """Constructs and diagonalizes the embedded BSE Hamiltonian.

        Args:
            verbose: if True, the output is written to screen.
        """

        # initialize dictionary for results
        res = {}
        # allocate BSE Hamiltonian
        bse_hamiltonian = np.zeros((self.n_tr, self.n_tr))

        # add diagonal term
        for s in range(self.n_tr):
            v, c, m = self.smap[s][:]
            if not self.spin_flip:
                bse_hamiltonian[s, s] += self.qp_energies[c, m] - self.qp_energies[v, m]
            else:
                m_prime = 1 - m
                bse_hamiltonian[s, s] += (
                    self.qp_energies[c, m_prime] - self.qp_energies[v, m]
                )

            # add direct and exchange terms
            for s2 in range(self.n_tr):
                v2, c2, m2 = self.smap[s2][:]
                # -------------------------------
                # spin-conserving BSE calculation
                # -------------------------------
                if not self.spin_flip:
                    if m == m2:
                        bse_hamiltonian[s, s2] += (
                            self.v[m, m, v, c, v2, c2] - self.w[m, m, v, v2, c, c2]
                        )
                    else:
                        bse_hamiltonian[s, s2] += self.v[m, m2, v, c, v2, c2]
                # -------------------------------
                # spin-flip BSE calculation
                # -------------------------------
                elif self.spin_flip:
                    if m == m2:
                        bse_hamiltonian[s, s2] += -self.w[m, 1 - m, v, v2, c, c2]

        res["hamiltonian"] = bse_hamiltonian[:, :]
        # diagonalize Hamiltonian
        evs_, evcs_ = np.linalg.eigh(bse_hamiltonian)
        # bring eigenvectors in the same format as the QDET ones,
        # such that res['evcs'][i] yields the i-th eigenstate
        evcs_ = evcs_.T

        if not self.spin_flip:
            nelec_ = self.nelec
        else:
            nelec_ = (self.nelec[0] - 1, self.nelec[1] + 1)
        res["nelec"] = nelec_
        res["norb"] = self.norb
        res["evs_au"] = evs_ * eV / Hartree
        res["evs"] = evs_ - evs_[0]
        res["evcs"] = evcs_

        # store density matrix and multiplicty
        res["rdm1s"] = self._get_1rdm(evcs_)
        res["mults"] = np.array([self._get_spin(evcs_[i])[1] for i in range(len(evs_))])
        # get occupation difference relative to the groundstate
        res["excitations"] = np.array(
            [np.diag(res["rdm1s"][i] - res["rdm1s"][0]) for i in range(len(evs_))]
        )

        # write summary to screen
        if verbose:
            self._write(
                "==============================================================="
            )
            self._write("Diagonalizing eBSE Hamiltonian...")
            self._write(f"nspin: {self.nspin}")
            self._write(f"occupations: {self.occupation[:]}")
            self._write(
                "==============================================================="
            )
            # header
            header_ = [("", "E [eV]"), ("", "char")]
            for b in self.basis:
                header_.append(("diag[1RDM - 1RDM(GS)]", f"{b}"))
            df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(header_))
            # formatting float
            pd.options.display.float_format = "{:,.3f}".format
            # data
            for ie, energy in enumerate(res["evs"]):
                row = [energy]
                row.append(res["mults"][ie])
                for ib, b in enumerate(self.basis):
                    row.append(res["excitations"][ie, ib])
                df.loc[ie] = row
            # display
            display(df)

        return res

    def _get_cistring(self, s):
        """For a given transition s, the function returns the cistring (in
        pyscf notation) of the excited spin-up and spin-down Slater determinant
        of the final state of the transition.

        Args:
            s: eBSE transition index.
        """
        v, c, m = self.smap[s][:]
        # adjust occupation for a given transition
        occ_ = np.copy(self.occupation)

        if not self.spin_flip:
            occ_[m, v] = 0.0
            occ_[m, c] = 1.0
        else:
            occ_[m, v] = 0.0
            occ_[1 - m, c] = 1.0

        # turn occupation to binary number
        cistring_ = []
        for m in range(2):
            binary = 0
            for i in range(occ_.shape[1]):
                binary += occ_[m, i] * 2**i
            cistring_.append(binary)

        return np.asarray(cistring_)

    def _get_map_transitions_to_cistrings(self):
        """returns a map that associates each transition s of the transition
        space to a pair of fci-vector indices in pyscf.fci
        additionally: stores Jordan-Wigner string for each product of up and
        down Slater determinant.
        """
        # generate all possible cistrings
        if not self.spin_flip:
            cistring_ = [
                make_strings(range(self.norb), self.nelec[0]),
                make_strings(range(self.norb), self.nelec[1]),
            ]
        else:
            cistring_ = [
                make_strings(range(self.norb), self.nelec[0] - 1),
                make_strings(range(self.norb), self.nelec[1] + 1),
            ]

        # allocate map
        cmap_ = np.zeros((self.n_tr, 2), dtype=np.int32)

        # allocate Jordan-Wigner strings
        jwstring_ = np.zeros(self.n_tr, dtype=np.int32)

        # loop over all transitions
        for s in range(self.n_tr):
            # determine cistrings for transition
            transition_strings = self._get_cistring(s)

            # loop over spin
            for m in range(2):
                # loop over cistrings
                for i in range(cistring_[m].shape[0]):
                    if cistring_[m][i] == transition_strings[m]:
                        cmap_[s, m] = i
            if self.spin_flip:
                jwstring_[s] = (-1) ** (self.nelec[1] + self.smap[s][0])
            else:
                m = self.smap[s][-1]
                jwstring_[s] = (-1) ** (self.nelec[m] + self.smap[s][0] - 1)

        return cmap_, jwstring_

    def transform_transition_to_fci(self, evcs_):
        """Transforms an eBSE eigenstate into an FCI state in
        second quantization. The format of the FCI state follows that of
        pyscf.fci

        Args:

            evcs_: eBSE eigenstate
        """

        # allocate FCI vector in the size of the correct Fock space
        if not self.spin_flip:
            fci_ = np.zeros(
                (
                    num_strings(self.norb, self.nelec[0]),
                    num_strings(self.norb, self.nelec[1]),
                )
            )
        else:
            fci_ = np.zeros(
                (
                    num_strings(self.norb, self.nelec[0] - 1),
                    num_strings(self.norb, self.nelec[1] + 1),
                )
            )

        # loop over all transitions
        for s in range(evcs_.shape[0]):
            # find the indices of the corresponding FCI vectors
            c1 = self.cmap[s, 0]
            c2 = self.cmap[s, 1]

            # assign value
            fci_[c1, c2] = evcs_[s] * self.jwstring[s]

        return fci_

    def _get_spin(self, evcs_):
        """Calculates the expectation value of the total spin $\langle
        \hat{S}^2 \rangle$ and spin multiplicity $M_S$ for a given eBSE
        eigenstate.

        Args:
            evcs_: eBSE eigenstate
        """
        fci_ = self.transform_transition_to_fci(evcs_)

        # account for different occupation in excited state in spin-flip BSE
        if not self.spin_flip:
            nelec_ = self.nelec
        else:
            nelec_ = (self.nelec[0] - 1, self.nelec[1] + 1)

        return spin_square_spin_polarized(
            solver="FCI", fcievc=fci_, norb=self.norb, nelec=nelec_, ovlpab=self.ovlpab
        )

    def get_transition_symmetry(self, vector, point_group_rep):
        """Determines the character of an eBSE eigenstate for a given point
        group representation. The function mimicks the corresponding
        functionality in the qdetresult object.

        Args:
            vector: eBSE eigenstate
            point_group_rep: point group representation on the active space
                orbitals.
        """

        fcivec = self.transform_transition_to_fci(vector)

        # get <S^2> and multiplicity for state
        ss, ms = self._get_spin(vector)
        # generate best-guess integer multiplicity
        ms = int(np.rint(ms))

        if not self.spin_flip:
            nelec_ = self.nelec
        else:
            nelec_ = (self.nelec[0] - 1, self.nelec[1] + 1)

        h_ = point_group_rep.point_group.h
        ctable_ = point_group_rep.point_group.ctable

        irprojs = []
        irreps = []

        for irrep, chis in ctable_.items():
            l = chis[0]
            pfcivec = np.zeros_like(fcivec)

            for chi, U in zip(chis, point_group_rep.rep_matrices.values()):
                pfcivec += chi * transform_ci_for_orbital_rotation(
                    ci=fcivec, norb=self.norb, nelec=nelec_, u=U.T
                )

            irprojs.append(l / h_ * np.sum(fcivec * pfcivec))
            irreps.append(irrep)

        # find maximal symmetry
        imax = np.argmax(irprojs)
        # symm = f"{irreps[imax]}({irprojs[imax]:.2f})"

        return str(ms) + str(irreps[imax])

    def _get_1rdm(self, evcs_):
        """generates density matrix for all eBSE eigenstates.
        Args:
            evcs_: list of eBSE eigenstates
        """

        if self.spin_flip:
            nelec_ = (self.nelec[0] - 1, self.nelec[1] + 1)
        else:
            nelec_ = (self.nelec[0], self.nelec[1])

        rdm1s = []
        for evc_ in evcs_:
            fci_ = self.transform_transition_to_fci(evc_)
            rdm1s.append(
                np.sum(
                    direct_uhf.make_rdm1s(
                        fcivec=fci_, norb=len(self.basis), nelec=nelec_
                    ),
                    axis=0,
                )
            )

        return np.array(rdm1s)
