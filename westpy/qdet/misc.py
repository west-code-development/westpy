import numpy as np
from pyscf.fci import cistring
from pyscf.fci import direct_uhf
from typing import Tuple


def visualize_correlated_state(
    fcievc: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    cutoff: float = 1e-3,
) -> str:
    """Visualizes the Slater determinants that contribute to a given many-body state

    Args:
        fcievc: FCI eigenvector
        norb: Number of orbitals in the active space
        nelec: A 2-dim tuple containing the number of spin-up and spin-down
            electrons in the active space
        cutoff: Coefficient smaller than cutoff will not be displayed

    Returns:
        String representing the many-body state
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
    for ib in range(fcievc.shape[0]):
        for jb in range(fcievc.shape[1]):
            if np.abs(fcievc[ib, jb]) >= cutoff:
                string = (
                    string
                    + format(fcievc[ib, jb], "+4.3f")
                    + ""
                    + string_fock[0][ib]
                    + string_fock[1][jb]
                )

    return string


def spin_square_spin_polarized(
    fcievc: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    ovlpab: np.ndarray = None,
) -> Tuple[float, float]:
    """Compute the spin multiplicity for spin polarized calculations. Modified from pyscf spin_square_general().

    Args:
        fcievc: FCI eigenvector.
        norb: Number of orbitals in the active space
        nelec: A 2-dim tuple containing the number of spin-up and spin-down
            electrons in the active space
        ovlpab: overlap matrix between orbitals in spin up and spin down channels

    Returns:
        Tuple[spin_square, spin_multiplicity].
    """

    # compute the density matrices
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = direct_uhf.make_rdm12s(
        fcievc, norb=norb, nelec=nelec
    )

    ovlpaa = np.eye(norb)
    ovlpbb = np.eye(norb)
    if ovlpab is None:
        ovlpab = np.eye(norb)
        ovlpba = np.eye(norb)
    else:
        ovlpba = ovlpab.T

    # if ovlp=1, ssz = (neleca-nelecb)**2 * .25
    ssz = (
        np.einsum("ijkl,ij,kl->", dm2aa, ovlpaa, ovlpaa)
        - np.einsum("ijkl,ij,kl->", dm2ab, ovlpaa, ovlpbb)
        + np.einsum("ijkl,ij,kl->", dm2bb, ovlpbb, ovlpbb)
        - np.einsum("ijkl,ij,kl->", dm2ab, ovlpaa, ovlpbb)
    ) * 0.25
    ssz += (
        np.einsum("ji,ij->", dm1a, ovlpaa) + np.einsum("ji,ij->", dm1b, ovlpbb)
    ) * 0.25

    dm2abba = -dm2ab.transpose(0, 3, 2, 1)  # alpha^+ beta^+ alpha beta
    dm2baab = -dm2ab.transpose(2, 1, 0, 3)  # beta^+ alpha^+ beta alpha
    ssxy = (
        np.einsum("ijkl,ij,kl->", dm2baab, ovlpba, ovlpab)
        + np.einsum("ijkl,ij,kl->", dm2abba, ovlpab, ovlpba)
        + np.einsum("ji,ij->", dm1a, ovlpaa)
        + np.einsum("ji,ij->", dm1b, ovlpbb)
    ) * 0.5
    ss = ssxy + ssz

    s = np.sqrt(ss + 0.25) - 0.5
    multip = s * 2 + 1

    return ss, multip
