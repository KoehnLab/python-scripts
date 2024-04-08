import numpy as np
import h5py
from enum import Enum
from .basis_util import get_multispinmat_prod,get_multipropmat_prod,transform_so

class Basis(Enum):
    WF0 = "Zeroth order wavefunction"
    PROD = "Spin-spatial Productbasis"
    SO = "Spin-orbit coupled basis"

def get_property_matrix(path:str,prop:str,basis: Basis):
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
        PropMat = h5file['prop'][:]
        SO = h5file['so'][:]
        SpinStates = h5file['SpinStates'][:]
        SpatStates = h5file['SpatStates'][:]
    SO_2d = get_2dmat(SO)
    if basis == Basis.WF0:
        return PropMat
    if basis == Basis.PROD:
        if prop == 'so':
            PropMat = get_multispinmat_prod(PropMat,SpinStates,SpatStates)
        else:
            PropMat = get_multipropmat_prod(PropMat,SpinStates,SpatStates)
        return PropMat
    if basis == Basis.SO:
        eigvecsh,eigvalsh = np.linalg.eigh(SO_2d)
        PropMat = transform_so(PropMat,eigvecsh)
        return PropMat
    raise Exception("an error occurred","unexpected value for basis")
