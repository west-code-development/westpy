import numpy as np
from pyscf.fci import cistring


def visualize_correlated_state(evcs, norb, nelec, cutoff=1e-3):
    """Visualizes the Slater determinants that contribute to a given many-body state

    :param evcs: FCI eigenstates provided by PySCF
    :param norb: number of orbitals that form the active space
    :param nelec: a two-entry list containing the number of spin-up and spin-down
    electrons in the active space
    :return: string representing the many-body state
    """
    # constrcut N-particle Fock space
    string_fock = [[], []]
    for ispin in range(2):
        determinants = cistring.make_strings(range(norb), nelec[ispin])
        for entry in determinants:
            if norb < 64:
                string = "|" + format(entry, "0" + str(norb) + "b") + ">"
            else:
                string = "|"
                for ib in range(norb - 1, -1, -1):
                    if ib in entry:
                        string += "1"
                    else:
                        string += "0"
                string += ">"
            string_fock[ispin].append(string)

    # string for many-body state
    string = ""
    for ib in range(evcs.shape[0]):
        for jb in range(evcs.shape[1]):
            if np.abs(evcs[ib, jb]) >= cutoff:
                string = (
                    string
                    + format(evcs[ib, jb], "+4.3f")
                    + ""
                    + string_fock[0][ib]
                    + string_fock[1][jb]
                )

    return string
