import numpy as np
import h5py
from enum import Enum
from .basis_util import get_multispinmat_prod,get_multipropmat_prod,transform_so





class Basis(Enum):
    WF0 = "Zeroth order wavefunction"
    PROD = "Spin-spatial Productbasis"
    SO = "Spin-orbit coupled basis"

def get_2dmat(matrix_3d):
        '''
        reshape
        Transforming a three dimensional matrix containing complex values in the third dimension (first index) into two dimensional complex matrix

        Args:
        ------------------------
        matrix_3d -- 3 dimensional matrix containing the complex values as third dimension

        Returns:
        ------------------------
        2d_matrix -- two dimensional complex matrix
        '''
        matrix = np.zeros((np.shape(matrix_3d)[1],np.shape(matrix_3d)[2]),dtype = complex)
        for i in range(np.shape(matrix_3d)[1]):
                for j in range(np.shape(matrix_3d)[2]):
                        matrix[i,j] = complex(matrix_3d[0,i,j] , matrix_3d[1,i,j])
        return matrix


def get_property_matrix(path:str,prop:str,basis: Basis,coord:str):
    '''
    Returns the matrix of a given operator in a given basis which is stored
    in a HDF5 File in the given path.

    Args:
    ----------------------
    path -- The path the HDF5 file is stored
    prop -- the operator of which the matrix is extracted from Molpro. 
            (DMX,DMY,DMZ,LX(),LY(),LZ(),SO)
    basis -- the basis in which the property matrix should be returned. 
  	     Choose one of the three options of Basis Enum
    coord -- the coordinate the spin_matrix in productbasis of spin and spatial states should be returned: x,y,z
    Returns:
    ---------------------
    PropMat -- The property matrix of given operator in the given basis

    '''
    with h5py.File(path,'r') as h5file:
        PropMat = h5file['prop'][:] # type: ignore
        SO = h5file['so'][:] # type: ignore
        SpinStates = h5file['SpinStates'][:] # type: ignore
        SpatStates = h5file['SpatStates'][:] # type: ignore
    SO_2d = get_2dmat(SO)
    if basis == Basis.WF0:
        return PropMat
    if basis == Basis.PROD:
        if prop == 'so':
            PropMat = get_multispinmat_prod(SpinStates,SpatStates,coord)
        else:
            PropMat = get_multipropmat_prod(PropMat,SpinStates,SpatStates)
        return PropMat
    if basis == Basis.SO:
        eigvecsh,eigvalsh = np.linalg.eigh(SO_2d)
        PropMat = transform_so(PropMat,eigvecsh)
        return PropMat
    raise Exception("an error occurred","unexpected value for basis")
