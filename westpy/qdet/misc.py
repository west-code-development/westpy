import re
import numpy as np
from six import string_types
from copy import deepcopy
from pyscf.fci import cistring


def visualize_correlated_state(evcs, norb, nelec, cutoff=10 ** (-3)):
    """Visualizes the Slater determinants that contribute to a given many-body
    states

    :param evcs: FCI eigenstate provided py PYSCF
    :param norb: number of orbitals that form the active space
    :param nelec: a two-entry list containing the number of spin-up and spin-down
    electrons in the active space
    :return: string representing the many-body state
    """
    # constrcut N-particle Fock space
    string_fock = []
    for i in range(2):
        determinants = cistring.make_strings(range(norb), nelec[i])
        string_fock.append(
            ["|" + format(entry, "0" + str(norb) + "b") + ">" for entry in determinants]
        )

    # string for many-body state
    string = ""
    for i in range(evcs.shape[0]):
        for j in range(evcs.shape[1]):
            if np.abs(evcs[i, j]) >= cutoff:
                string = (
                    string
                    + format(evcs[i, j], "+4.3f")
                    + ""
                    + string_fock[0][i]
                    + string_fock[1][j]
                )

    return string
