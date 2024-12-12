from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .phys_const import muBcm, ge


def compute_magnetic_moment_matrix(sMat, lMat, in_bohr_magnetons: bool = True):
    """
    Computes the magnetic moment from the given spin and angular momentum
    Args:
    ---------------------
    sMat -- spin matrix
    lmat -- angular momentum matrix
    in_bohr_magnetons -- Whether the result shall be given in multiples of the Bohr magneton

    Returns:
    ---------------------
    The matrix of magnetic moment
    """
    muMat = -(ge * sMat + lMat)

    if not in_bohr_magnetons:
        muMat *= muBcm

    return muMat


def compute_A_matrix(mu_x, mu_y, mu_z, num_states: int) -> NDArray[np.float_]:
    """
    Computes the A matrix to calculate g-Tensor afterwards
    Procedure taken from: L. F. Chibotaru, L. Ungur; Ab initio calculation of anisotropic magnetic properties of complexes. I. Unique definition of
    pseudospin Hamiltonians and their derivation. J. Chem. Phys. 14 August 2012; 137 (6): 064112. https://doi.org/10.1063/1.4739763
    Args:
    ---------------------
    mu_x,y,z -- matrices of magnetic moment in SO basis
    num_states -- Number of states that shall be considered when constructing the A matrix

    Returns:
    ---------------------
    The A matrix
    """
    mu_x_n = mu_x[:num_states, :num_states]
    mu_y_n = mu_y[:num_states, :num_states]
    mu_z_n = mu_z[:num_states, :num_states]

    mu = [mu_x_n, mu_y_n, mu_z_n]

    Amat = np.zeros((3, 3))

    for alpha in range(3):
        for beta in range(3):
            prod = np.matmul(mu[alpha], mu[beta])
            trace = np.matrix.trace(prod)
            assert np.abs(trace.imag) < 1e-6
            Amat[alpha, beta] = 0.5 * trace.real

    return Amat


def compute_g_tensor(Amat, S: float, in_bohr_magnetons: bool = True) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Computes the g-tensor as well as the main magnetic axes of the system.
    Procedure taken from: L. F. Chibotaru, L. Ungur; Ab initio calculation of anisotropic magnetic properties of complexes. I. Unique definition of
    pseudospin Hamiltonians and their derivation. J. Chem. Phys. 14 August 2012; 137 (6): 064112. https://doi.org/10.1063/1.4739763
    Args:
    ----------------------
    Amat -- The A matrix (cmp. compute_A_matrix)
    S -- The (pseudo)spin quantum number of the system
    in_bohr_magnetons -- Whether the given A matrix is given in multiples of the Bohr magneton

    Returns:
    ----------------------
    A tuple containing the eigenvalues of the g-tensor and the main magneetic axes (as columns in a 3x3 matrix)
    """
    assert int(2 * S) == 2 * S

    eigvals, main_axes = np.linalg.eigh(Amat)

    mB = 1 if in_bohr_magnetons else muBcm

    g_diag = np.array([6 * np.sqrt(A) / (mB * S * (S + 1) * (2 * S + 1)) for A in eigvals])

    return (g_diag, main_axes)
