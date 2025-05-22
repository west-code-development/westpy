from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from IPython.display import display
from westpy import eV, Hartree
from westpy import VData
from westpy.qdet.heff import Heff
from westpy.qdet.symm import PointGroup, PointGroupRep
from westpy.qdet.json_parser import (
    read_parameters,
    read_occupation,
    read_matrix_elements,
    read_overlap,
)


class QDETResult(object):
    def __init__(
        self,
        filename: str,
        point_group: Optional[PointGroup] = None,
        wfct_filenames: Optional[list] = None,
        symmetrize: Dict[str, bool] = {},
    ):
        """Parser for Quantum Defect Embedding Theory (QDET) calculations.

        Args:
            filename: name of JSON file that contains the output of WEST
                calculation.
            point_group: point group of the system.
            symmetrize: arguments for symmetrization function of Heff.
        """
        self.filename = filename

        # read basic parameters from JSON file
        self.nspin, self.npair, self.basis = read_parameters(filename)

        # read occupation from file
        self.occupation = read_occupation(filename)

        # read one- and two-body terms from JSON file
        self.h1e, self.eri = read_matrix_elements(filename)

        # read overlap matrix from file
        if self.nspin == 2:
            self.ovlpab = read_overlap(filename)
        elif self.nspin == 1:
            self.ovlpab = None

        # determine point-group representation

        self.point_group = point_group
        if self.point_group is None:
            self.point_group_rep = None
        else:
            orbitals = [VData(entry, normalize="sqrt") for entry in wfct_filenames]
            (
                self.point_group_rep,
                self.orbital_symms,
            ) = self.point_group.compute_rep_on_orbitals(orbitals, orthogonalize=True)

        self.h1e = self.h1e * eV / Hartree
        self.eri = self.eri * eV / Hartree

        # generate effective Hamiltonian
        self.heff = Heff(
            self.h1e, self.eri, self.ovlpab, point_group_rep=self.point_group_rep
        )

        self.heff.symmetrize(**symmetrize)

    def __str__(self):
        """Print a summary of QDET calculation."""
        string = "---------------------------------------------------------------\n"
        string += "QDET Results General Info\n"
        string += f"path: {self.path}\n"
        string += f"nspin = {self.nspin}, nel = {self.nel}, nproj = {len(self.bases)}\n"
        if self.point_group is not None:
            string += f"point group: {self.point_group.name}\n"
        string += f"ks_projectors: {self.basis}\n"

        string += "occupations:" + str(self.occupation) + "\n"
        string += "---------------------------------------------------------------\n"

        return string

    def _write(self, *args):
        data = ""
        for i in args:
            data += str(i)
            data += " "
        data = data[:-1]
        print(data)

    def solve(
        self, nelec: Tuple = None, nroots: int = 10, verbose: bool = True
    ) -> Dict:
        """Build and diagonalize effective Hamiltonians for given active space.

        Args:
            nelec (2-dim tuple of int): Number of electrons in each spin channel
            nroots (int): Number of roots for FCI calculations
            verbose (boolean): If True, write detailed info for FCI calculations
        """
        basis_indices = self.basis
        basis_labels = [""] * len(basis_indices)

        # determine number of electrons from occupations
        if nelec == None:
            if self.nspin == 1:
                nel = np.sum(self.occupation)
                nelec = (int(round(nel)) // 2, int(round(nel)) // 2)
            else:
                nel1 = np.sum(self.occupation[0, :])
                nel2 = np.sum(self.occupation[1, :])
                nelec = (int(nel1), int(nel2))

        # diagonalize effective Hamiltonian
        fcires = self.heff.FCI(nelec=nelec, nroots=nroots)

        if verbose:
            self._write(
                "==============================================================="
            )
            self._write("Building effective Hamiltonian...")
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
            for ie, energy in enumerate(fcires["evs"]):
                row = [energy]
                row.append(
                    f"{int(round(fcires['mults'][ie]))}{fcires['symms_maxproj'][ie].partition('(')[0]}"
                )
                for ib, b in enumerate(self.basis):
                    row.append(fcires["excitations"][ie, ib])
                df.loc[ie] = row
            # display
            display(df)
            self._write("-----------------------------------------------------")

            # remove keys that are confusing to the user and are no longer
            # needed
            fcires.pop("excitations", None)
            if self.point_group is None:
                fcires.pop("symms_maxproj", None)
                fcires.pop("symms_full", None)

        return fcires
