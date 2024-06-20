import numpy as np
import h5py
from enum import Enum
from koehnlab.io import basis_util
from koehnlab.spin_hamiltonians import spin_utils

class Basis(Enum):
    WF0 = "WF0"
    PROD = "PROD"
    SO = "SO"

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


def get_property_matrix(path:str,prop:str,basis:str):
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
    Returns:
    ---------------------
    PropMat -- The property matrix of given operator in the given basis

    '''
    with h5py.File(path,'r') as h5file:
        if (prop == 'SO'):
            SO = h5file['xh5d_dataset_multidim_so.hdf5'][:] # type: ignore
            SO_2d = get_2dmat(SO)
            return SO_2d
        PropMat = h5file[prop][:] # type: ignore
        SpinStates = h5file['xh5d_dataset_spin_states.hdf5'][:] # type: ignore
        SpatStates = h5file['xh5d_dataset_spat_states.hdf5'][:] # type: ignore
        SpinStates = np.array(SpinStates)
        SpatStates = np.array(SpatStates)
    if (prop[0] == 'L'):
        PropMat = -1j*PropMat
    if basis == 'WF0':
        return PropMat
    if basis == 'PROD':
            PropMat = basis_util.get_multipropmat_prod(PropMat,SpinStates,SpatStates)
            return PropMat
    if basis == 'SO':
        PropMat = basis_util.get_multipropmat_prod(PropMat,SpinStates,SpatStates)
        print(np.shape(PropMat))
        eigvalsh,eigvecsh = np.linalg.eigh(SO_2d)
        PropMat = basis_util.transform_so(PropMat,eigvecsh)
        return PropMat
    raise Exception("an error occurred","unexpected value for basis")



def get_spin_spat_states(path:str):
    with h5py.File(path,'r') as h5file:
        SpinStates = h5file['xh5d_dataset_spin_states.hdf5'][:] # type: ignore
        SpatStates = h5file['xh5d_dataset_spat_states.hdf5'][:] # type: ignore
        SpinStates = np.array(SpinStates)
        SpatStates = np.array(SpatStates)
    return SpinStates, SpatStates

