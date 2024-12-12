from collections.abc import Iterable

import numpy as np

from .phys_const import muBcm
from .coordinate import Coordinate3D

""" A set of routines for setting up spin Hamiltonians  """

def spinMat(S: float, cmp: Coordinate3D | str):
    """Return matrix elements of operator S_i (with i == cmp) for spin quantum number S
    (in multiples of hbar)"""

    if isinstance(cmp, str):
        cmp = Coordinate3D[cmp] # type: ignore

    assert type(cmp) is Coordinate3D

    multiplicity = int(2 * S + 1)

    assert S - int(S) in [0, 0.5], "Spin can only take on half-integer numbers"

    if cmp == Coordinate3D.Z:
        # The S_z matrix is a diagonal matrix that contains the possible Ms values
        # on its diagonal (in descending order)
        return np.diag(np.linspace(start=+S, stop=-S, num=multiplicity))
    elif cmp == Coordinate3D.X:
        # S_x is expressed in terms of ladder operators:
        # S_x = 0.5 * (S_+ + S_-)
        # Because of the ladder operators, the entries appear on the off-diagonal
        Smat = np.zeros((multiplicity, multiplicity), dtype=complex)
        Ms = S
        SS1 = S * (S + 1)
        for ii in range(multiplicity - 1):  # we run over column
            val = 0.5 * np.sqrt(SS1 - Ms * (Ms - 1))
            Smat[ii + 1, ii] = val
            Smat[ii, ii + 1] = val
            Ms -= 1.0

        return Smat
    elif cmp == Coordinate3D.Y:
        # S_y is expressed in terms of ladder operators:
        # S_y = -i/2 * (S_+ - S_-)
        # Because of the ladder operators, the entries appear on the off-diagonal
        Smat = np.zeros((multiplicity, multiplicity), dtype=complex)
        Ms = S
        SS1 = S * (S + 1)
        for ii in range(multiplicity - 1):  # we run over column
            val = np.sqrt(SS1 - Ms * (Ms - 1))
            Smat[ii + 1, ii] = 0.5j * val
            Smat[ii, ii + 1] = -0.5j * val
            Ms -= 1.0

        return Smat

def spin_mat(spin_qns: float | Iterable[float], component: Coordinate3D):
    """Computes the combined spin matrix for the given spin quantum numbers (block-diagonal)"""
    if type(spin_qns) is float:
        spin_qns = [spin_qns]

    multiplicities = [int(2 * S) + 1 for S in spin_qns] # type: ignore
    full_dim = sum(multiplicities)

    mat = np.zeros(shape=(full_dim, full_dim), dtype=complex)

    offset = 0
    for mult, S in zip(multiplicities, spin_qns): # type: ignore
        # S is expected to be integer or half-integer valued
        assert int(2 * S) == 2 * S

        mat[offset : offset + mult, offset : offset + mult] = spinMat(S, component)

        offset += mult

    return mat





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
            # H_Zeeman = - M B
            HmatD -= Bfield[ii] * Mmat[ii] * muBcm

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
            # use negative submat to get positive Mu first
            submat = -np.array(Mtraf[idxst:idxnd, idxst:idxnd])
            # printMatC(submat)
            # print()
            # diagonalize submatrix
            _, Usub = np.linalg.eigh(submat)
            # apply this to total eigenvectors: transform and insert
            Unew = np.matmul(U[0:ndim, idxst:idxnd], Usub)
            U[0:ndim, idxst:idxnd] = Unew
            idxst = idxnd

    return En, U

def getMagneticAxes(mu):
    """ mu[3,dim,dim] contains the dim x dim representation of the 
        magnetic moment operator in a given basis, the first dimension
        runs over X, Y, Z
        From this, the Gerloch McMeeking tensor is computed
        on exit, return the main magnetic axes the eigenvalues of the 
        Gerloch McMeeking tensor """

    Amat = np.zeros((3,3))
    dim = mu.shape[1]

    for ii in range(3):
        for jj in range(3):
            for kk in range(dim):
                for ll in range(dim):
                    Amat[ii,jj]=Amat[ii,jj]+0.5*(mu[ii,kk,ll]*mu[jj,ll,kk]).real

    Adia,Rmat = np.linalg.eigh(Amat)

    # evalues are positive and in ascending order
    if Adia[2]-Adia[1] < Adia[1]-Adia[0]:
        # oblate case
        idx0=1
        idx1=2
        idx2=0
    else:
        # prolate case
        idx0=0
        idx1=1
        idx2=2

    Adia = np.array([Adia[idx0],Adia[idx1],Adia[idx2]])
    Rmat = np.array([Rmat[:,idx0],Rmat[:,idx1],Rmat[:,idx2]]).T

    # ensure right-handedness:
    if (np.linalg.det(Rmat)<0.):
        Rmat = -Rmat

    return Adia,Rmat


def A_to_g(Adia,S):
    """ turn the eigenvalues of the Gerloch McMeeking tensor into g matrix values """
    """ S is the spin of the (pseudo)spin system """
    fac = 6./(S*(S+1.)*(2.*S+1.))
    gdia = np.sqrt(fac*Adia)
    return gdia


def rotateVectorOpMatrix(vec,Rmat):
    """ vec[3,dim,dim] is a matrix rep of a vector operator 
        Rmat is a rotation matrix 
        compute the transformed matrix rep of the vector operator"""

    vecT = np.zeros(vec.shape,dtype=vec.dtype)
    for ii in range(3):
        for jj in range(3):
            vecT[ii] += Rmat[jj,ii]*vec[jj]
    return vecT

