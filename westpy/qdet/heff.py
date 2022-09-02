from __future__ import annotations
from typing import Dict, Tuple, Union
import numpy as np
from westpy import Hartree, eV
from .symm import PointGroupRep

# Note that some functions in this file requires PySCF or Qiskit to execute. The imports commands are written
# within functions to avoiding making them global dependencies.


class Heff:
    def __init__(
        self,
        h1e: np.ndarray,
        eri: np.ndarray,
        point_group_rep: PointGroupRep = None,
        verbose: bool = False,
    ):
        """Effective Hamiltonian, defined with 1e component h1e and 2e component eri.

        Args:
            h1e: 1e component
            eri: 2e component
            point_group_rep: point group matrix representation on the Hilbert space which Heff is defined on
            verbose: if True, plot a summary of Heff
        """
        if h1e.ndim == 3 and eri.ndim == 6:
            self.nspin = h1e.shape[0]
            self.norb = h1e.shape[1]
            assert h1e.shape == (self.nspin, self.norb, self.norb)
            assert eri.shape == (
                self.nspin,
                self.nspin,
                self.norb,
                self.norb,
                self.norb,
                self.norb,
            )
            if self.nspin == 1:
                self.h1e = h1e[0]
                self.eri = eri[0, 0]
            elif self.nspin == 2:
                self.h1e = h1e
                self.eri = eri
        elif h1e.ndim == 2 and eri.ndim == 4:
            self.nspin = 1
            self.norb = h1e.shape[0]
            assert h1e.shape == (self.norb, self.norb)
            assert eri.shape == (self.norb, self.norb, self.norb, self.norb)
            self.h1e = h1e
            self.eri = eri
        else:
            raise TypeError

        self.point_group_rep = point_group_rep
        if self.point_group_rep is not None:
            assert isinstance(point_group_rep, PointGroupRep)
            assert point_group_rep.norb == self.norb

        self.verbose = verbose
        if self.verbose:
            self.print_symm_info()

    def print_symm_info(self):
        """Print a summary of Heff."""
        print(f"Heff: nspin = {self.nspin}, norb = {self.norb}")
        print(
            "H1e permutation symmetry: {}".format(
                np.all(
                    np.isclose(
                        self.h1e,
                        self.apply_permutation_symm_to_h1e(self.h1e),
                        atol=1e-05,
                    )
                )
            )
        )
        print(
            "ERI permutation symmetry: {}".format(
                np.all(
                    np.isclose(
                        self.eri,
                        self.apply_permutation_symm_to_eri(self.eri),
                        atol=1e-05,
                    )
                )
            )
        )
        if self.has_symm:
            print(
                "H1e point group symmetry: {}".format(
                    np.all(
                        np.isclose(
                            self.h1e,
                            self.apply_point_group_symm_to_h1e(
                                self.h1e, self.point_group_rep
                            ),
                            atol=1e-05,
                        )
                    )
                )
            )
            print(
                "ERI point group symmetry: {}".format(
                    np.all(
                        np.isclose(
                            self.eri,
                            self.apply_point_group_symm_to_eri(
                                self.eri, self.point_group_rep
                            ),
                            atol=1e-05,
                        )
                    )
                )
            )

    @property
    def has_symm(self):
        return self.point_group_rep is not None

    def FCI(
        self,
        nelec: Union[int, Tuple[int, int]],
        nroots: int = 10,
        verbose: bool = False,
    ) -> Dict:
        """Perform FCI calculations using pyscf package.

        Args:
            nelec: # of electrons. If a 2-tuple is given, interpret as spin up and spin down electrons.
            nroots: # of eigenstates to compute.
            verbose: if True, print more info.

        Returns:
            a dict of FCI results
        """

        import pyscf.fci

        if verbose:
            self.print_symm_info()

        if self.nspin == 1:
            self.fcisolver = pyscf.fci.direct_spin1.FCISolver()
            evs, evcs = self.fcisolver.kernel(
                h1e=self.h1e, eri=self.eri, norb=self.norb, nelec=nelec, nroots=nroots
            )
        else:
            self.fcisolver = pyscf.fci.direct_uhf.FCISolver()
            evs, evcs = self.fcisolver.kernel(
                h1e=(self.h1e[0], self.h1e[1]),
                eri=(self.eri[0, 0], self.eri[0, 1], self.eri[1, 1]),
                norb=self.norb,
                nelec=nelec,
                nroots=nroots,
            )
        nstates = len(evs)  # can be smaller than nroots if Hilbert space is too small

        res = {
            "nstates": nstates,
            "evs_au": evs,
            "evs": (evs - evs[0]) * (eV ** (-1) / Hartree),
            "evcs": evcs,
            "mults": np.array(
                [
                    self.fcisolver.spin_square(fcivec=evc, norb=self.norb, nelec=nelec)[
                        1
                    ]
                    for evc in evcs
                ]
            ),
        }

        if self.nspin == 1:
            res["rdm1s"] = np.array(
                [
                    self.fcisolver.make_rdm1(fcivec=evc, norb=self.norb, nelec=nelec)
                    for evc in evcs
                ]
            )
        else:
            res["rdm1s"] = np.array(
                [
                    np.average(
                        self.fcisolver.make_rdm1s(
                            fcivec=evc, norb=self.norb, nelec=nelec
                        ),
                        axis=0,  # average over spin index
                    )
                    for evc in evcs
                ]
            )

        # analyze excitations based on 1rdm
        res["excitations"] = np.array(
            [np.diag(res["rdm1s"][i] - res["rdm1s"][0]) for i in range(nstates)]
        )

        # analyze point group symmetries
        if self.has_symm:
            print(
                f"Solutions are projected onto irreps of {self.point_group_rep.point_group.name} group"
            )

            h = self.point_group_rep.point_group.h
            ctable = self.point_group_rep.point_group.ctable

            from pyscf.fci.addons import transform_ci_for_orbital_rotation

            res["symms_maxproj"] = []
            res["symms_full"] = []

            for fcivec in evcs:
                irprojs = {}
                for irrep, chis in ctable.items():
                    l = chis[0]
                    pfcivec = np.zeros_like(fcivec)
                    for chi, U in zip(chis, self.point_group_rep.rep_matrices.values()):
                        pfcivec += chi * transform_ci_for_orbital_rotation(
                            ci=fcivec, norb=self.norb, nelec=nelec, u=U.T
                        )
                    irproj = l / h * np.sum(fcivec * pfcivec)
                    irprojs[irrep] = irproj  # "{:.2f}".format()
                res["symms_full"].append(
                    {ir: "{:.2f}".format(irproj) for ir, irproj in irprojs.items()}
                )

                irreps = list(irprojs.keys())
                irproj_values = list(irprojs.values())
                imax = np.argmax(irproj_values)
                res["symms_maxproj"].append(
                    f"{irreps[imax]}({irproj_values[imax]:.2f})"
                )
        else:
            res["symms_maxproj"] = ["-"] * nstates
            res["symms_full"] = ["-"] * nstates

        print("-----------------------------------------------------")

        return res

    def symmetrize(
        self, permutation: bool = False, point_group: bool = False, inplace: bool = True
    ) -> Union[None, Heff]:
        """Symmetrize the effective Hamiltonian.

        Args:
            permutation: if True, enforce permutation symmetry.
            point_group: if True, enforce point group symmetry.
            inplace: if True, symmetrize in place.

        Returns:
            if inplace == False, return symmetrized Heff.
        """
        h1e = self.h1e.copy()
        eri = self.eri.copy()
        if permutation:
            h1e = self.apply_permutation_symm_to_h1e(h1e)
            eri = self.apply_permutation_symm_to_eri(eri)
        if point_group:
            h1e = self.apply_point_group_symm_to_h1e(h1e, self.point_group_rep)
            eri = self.apply_point_group_symm_to_eri(eri, self.point_group_rep)
        if inplace:
            self.h1e, self.eri = h1e, eri
        else:
            return Heff(h1e, eri, point_group_rep=self.point_group_rep)

    @staticmethod
    def apply_permutation_symm_to_h1e(h1e: np.ndarray) -> np.ndarray:
        """Symmetrize h1e w/ permutation.

        Args:
            h1e: input h1e.

        Returns:
            symmetrized h1e.
        """
        nspin = 1 if h1e.ndim == 2 else 2
        h1e_symm = np.zeros_like(h1e)
        if nspin == 1:
            p0 = "ij"
            permutations = ["ij", "ji"]
        else:
            p0 = "sij"
            permutations = ["sij", "sji"]
        for p in permutations:
            h1e_symm += np.einsum(f"{p0}->{p}", h1e, optimize=True)
        h1e_symm /= 2
        return h1e_symm

    @staticmethod
    def apply_permutation_symm_to_eri(eri: np.ndarray) -> np.ndarray:
        """Symmetrize eri w/ permutation.

        Args:
            eri: input eri.

        Returns:
            symmetrized eri.
        """
        nspin = 1 if eri.ndim == 4 else 2
        eri_symm = np.zeros_like(eri)
        if nspin == 1:
            p0 = "ijkl"
            permutations = [
                "ijkl",
                "ijlk",
                "jikl",
                "jilk",
                "klij",
                "klji",
                "lkij",
                "lkji",
            ]
        else:
            p0 = "stijkl"
            permutations = [
                "stijkl",
                "stijlk",
                "stjikl",
                "stjilk",
                "tsklij",
                "tsklji",
                "tslkij",
                "tslkji",
            ]
        for p in permutations:
            eri_symm += np.einsum(f"{p0}->{p}", eri, optimize=True)
        eri_symm /= 8
        return eri_symm

    @staticmethod
    def apply_point_group_symm_to_h1e(
        h1e: np.ndarray, rep: PointGroupRep
    ) -> np.ndarray:
        """Symmetrize h1e w/ point group operations.

        Args:
            h1e: input h1e.
            rep: point group matrix representation.

        Returns:
            symmetrized h1e.
        """
        nspin = 1 if h1e.ndim == 2 else 2
        assert isinstance(rep, PointGroupRep)
        h = rep.point_group.h
        h1e_symm = np.zeros_like(h1e)
        if nspin == 1:
            transform = "ip,jq,pq->ij"
        else:
            transform = "ip,jq,spq->sij"
        for D in rep.rep_matrices.values():
            h1e_symm += np.einsum(transform, D, D, h1e, optimize=True)
        h1e_symm /= h
        return h1e_symm

    @staticmethod
    def apply_point_group_symm_to_eri(
        eri: np.ndarray, rep: PointGroupRep
    ) -> np.ndarray:
        """Symmetrize eri w/ point group operations.

        Args:
            eri: input eri.
            rep: point group matrix representation.

        Returns:
            symmetrized eri.
        """
        nspin = 1 if eri.ndim == 4 else 2
        assert isinstance(rep, PointGroupRep)
        h = rep.point_group.h
        eri_symm = np.zeros_like(eri)
        if nspin == 1:
            transform = "ip,jq,kr,ls,pqrs->ijkl"
        else:
            transform = "ip,jq,kr,ls,stpqrs->stijkl"
        for D in rep.rep_matrices.values():
            eri_symm += np.einsum(transform, D, D, D, D, eri, optimize=True)
        eri_symm /= h
        return eri_symm

    def make_fermionic_operator(self, mu: float = 0.0):
        """Construct Qiskit FermionicOperator from Heff.

        Args:
            mu: chemical potential, used to shift the spectrum of Heff.

        Returns:
            Qiskit FermionicOperator instance
        """
        from qiskit_nature.properties.second_quantization.electronic import (
            ElectronicEnergy,
        )
        from qiskit_nature.properties.second_quantization.electronic.bases import (
            ElectronicBasis,
        )
        from qiskit_nature.properties.second_quantization.electronic.integrals import (
            TwoBodyElectronicIntegrals,
        )

        if self.nspin == 1:
            h1e = self.h1e.copy()
            for i in range(self.norb):
                h1e[i, i] += mu
            h1 = np.bmat([[h1e, np.zeros_like(h1e)], [np.zeros_like(h1e), h1e]])
            # worked for older versions of qiskit
            # h2 = QMolecule.twoe_to_spin(np.einsum('ijkl->ljik', self.eri))
            # h2 = QMolecule.twoe_to_spin(self.eri)
            h2 = TwoBodyElectronicIntegrals(
                ElectronicBasis.MO,
                (self.eri, None, None, None),
            ).to_spin()
            return ElectronicEnergy.from_raw_integrals(ElectronicBasis.SO, h1, h2)
        else:
            raise NotImplementedError
