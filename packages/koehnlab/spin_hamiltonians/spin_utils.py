import numpy as np
from .phys_const import muBcm

""" A set of routines for setting up spin Hamiltonians  """


def spinMat(S, cmp):
    """return matrix elemts of operator S_cmp for spin quantum number S"""
    d = int(2 * S + 1)
    Smat = np.zeros((d, d), dtype=complex)
    if cmp == "Z" or cmp == "z":
        Ms = float(S)
        for ii in range(d):
            Smat[ii, ii] = Ms
            Ms -= 1.0
    elif cmp == "X" or cmp == "x":
        Ms = float(S)
        SS1 = float(S) * (float(S) + 1.0)
        for ii in range(d - 1):  # we run over column
            val = np.sqrt(SS1 - Ms * (Ms - 1))
            Smat[ii + 1, ii] = 0.5 * val
            Smat[ii, ii + 1] = 0.5 * val
            Ms -= 1.0
    elif cmp == "Y" or cmp == "y":
        Ms = float(S)
        SS1 = float(S) * (float(S) + 1.0)
        for ii in range(d - 1):  # we run over column
            val = np.sqrt(SS1 - Ms * (Ms - 1))
            Smat[ii + 1, ii] = 0.5j * val
            Smat[ii, ii + 1] = -0.5j * val
            Ms -= 1.0
    else:
        print("unknown cmp: ", cmp)

    return Smat


def unit(nd):
    """return a (complex type) unit matrix of dimension nd"""
    return np.identity(nd, dtype=complex)



def tprod(A, B):
    """return the tensor product of matrices A and B"""
    """ our own definition of a tensor product """
    # obviously this already existed (keep this wrapper for compatibility):
    return np.kron(A,B)

# this is an explicit code of what kron does for 2D matrices:
#    ldij = B.shape[0]
#    ldkl = B.shape[1]
#    AB = np.zeros((A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]), dtype=A.dtype)
#    for idx in range(A.shape[0]):
#        for jdx in range(B.shape[0]):
#            ijdx = ldij * idx + jdx
#            for kdx in range(A.shape[1]):
#                for ldx in range(B.shape[1]):
#                    kldx = ldkl * kdx + ldx
#                    AB[ijdx, kldx] = A[idx, kdx] * B[jdx, ldx]
#    return AB


def diagonalizeSpinHamiltonian(Hmat, Mmat=None, Bfield=None):
    """diagonalize Spin Hamiltonian for Bfield and compute expectation values"""
    """ Hmat in cm-1, Mmat in Bohr magnetons, Bfield (x,y,z) in Tesla  """

    HmatD = np.array(Hmat)

    if Mmat is not None and Bfield is not None:
        for ii in range(3):
            HmatD += Bfield[ii] * Mmat[ii] * muBcm

    En, U = np.linalg.eigh(HmatD)

    if Mmat is not None:
        # loop across energies; for degenerate tuples diagonalize Mmat[2] expectation values in that block
        idxst = 0
        idxnd = 0
        Mtraf = np.matmul(np.conj(U.T),np.matmul(Mmat[2],U))
        ndim = En.shape[0]
        while idxst < ndim:
            ndim_block = 0
            while (
                idxst + ndim_block < ndim
                and np.abs(En[idxst + ndim_block] - En[idxst]) < 1e-3
            ):
                ndim_block += 1
            idxnd = idxst + ndim_block
            # print("current block: ",idxst,idxnd-1)
            submat = np.array(Mtraf[idxst:idxnd, idxst:idxnd])
            # printMatC(submat)
            # print()
            # diagonalize submatrix
            _, Usub = np.linalg.eigh(submat)
            # apply this to total eigenvectors: transform and insert
            Unew = np.matmul(U[0:ndim, idxst:idxnd], Usub)
            U[0:ndim, idxst:idxnd] = Unew
            idxst = idxnd

    return En, U
