import numpy as np
import h5py
from .basis_util import get_multispinmat_prod,get_multipropmat_prod,transform_so


def extract_data(path,prop,basis):
    '''
    Returns the matrix of a given operator in a given basis which is stored
    in a HDF5 File in the given path.

    Args:
    ----------------------
    path -- The path the HDF5 file is stored
    prop -- the operator of which the matrix is extracted from Molpro. 
            (DMX,DMY,DMZ,LX(),LY(),LZ(),SO)
    basis -- the basis in which the property matrix should be returned.
            (wf0,prod,so)
    Returns:
    ---------------------
    PropMat -- The property matrix of given operator in the given basis

    '''
    with h5py.File(path,'r') as h5file:
        PropMat = h5file['prop'][:]
        SO = h5file['so'][:]
        SpinStates = h5file['SpinStates'][:]
        SpatStates = h5file['SpatStates'][:]
    SO_2d = get_2dmat(SO)
    if basis == 'wf0':
        return PropMat
    if basis == 'prod':
        if prop == 'so':
            PropMat = get_multispinmat_prod(PropMat,SpinStates,SpatStates)
        else:
            PropMat = get_multipropmat_prod(PropMat,SpinStates,SpatStates)
        return PropMat
    if basis == 'so':
        eigvecsh,eigvalsh = np.linalg.eigh(SO_2d)
        PropMat = transform_so(PropMat,eigvecsh)
        return PropMat

