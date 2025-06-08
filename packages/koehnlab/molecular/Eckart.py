from .Molecule import Molecule
from .Atom import Atom
from .Element import Element

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as rot

# internal function:
def _get_Rot_alignment(coord1: np.ndarray, coord2: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Get rotation matrix for fulfilling optimal roational Eckart conditions;
       coord1 and coord2 must be in the same order and have a common COM"""

    # we use the quaternion-based formulation as described in
    #   Krasnoshchekov, Isayeva, Stepanov, J. Chem. Phys. 2014, 140, 154104.
    #   https://doi.org/10.1063/1.4870936.

    assert coord1.shape == coord2.shape
    assert coord1.shape[1] == 3
    assert coord1.shape[0] == mass.shape[0]

    natoms = mass.shape[0]

    cp = coord1 + coord2
    cm = coord1 - coord2

    Cmat = np.zeros((4,4))

    # implements eq. 24
    for idx in range(natoms):
        ma = mass[idx]
        xpa = cp[idx,0]; ypa = cp[idx,1]; zpa = cp[idx,2]
        xma = cm[idx,0]; yma = cm[idx,1]; zma = cm[idx,2]
        Cmat[0,0] += ma * (xma**2 + yma**2 + zma**2)
        Cmat[0,1] += ma * (ypa*zma - yma*zpa)
        Cmat[0,2] += ma * (xma*zpa - xpa*zma)
        Cmat[0,3] += ma * (xpa*yma - xma*ypa)
        Cmat[1,1] += ma * (xma**2 + ypa**2 + zpa**2)
        Cmat[1,2] += ma * (xma*yma - xpa*ypa)
        Cmat[1,3] += ma * (xma*zma - xpa*zpa)
        Cmat[2,2] += ma * (xpa**2 + yma**2 + zpa**2)
        Cmat[2,3] += ma * (yma*zma - ypa*zpa)
        Cmat[3,3] += ma * (xpa**2 + ypa**2 + zma**2)

    Cmat[1,0] = Cmat[0,1]
    Cmat[2,0] = Cmat[0,2]
    Cmat[3,0] = Cmat[0,3]
    Cmat[2,1] = Cmat[1,2]
    Cmat[3,1] = Cmat[1,3]
    Cmat[3,2] = Cmat[2,3]

    qv,Q = np.linalg.eigh(Cmat)

    # generate rotation matrix from lowest eigenvalue eigenvector, interpreted as quaternion
    # older versions of scipy only know the "scalar last" format of quaternions, so we need
    # to reshuffle
    Rmat = rot.from_quat([Q[1,0],Q[2,0],Q[3,0],Q[0,0]]).as_matrix()

    return Rmat


def Eckart_alignment(mol_A: Molecule, mol_B: Molecule, verbosity: int = 0) -> None:
    """ Molecule B is translated and rotated to fulfill the Eckart conditions wrt to molecule A.
        On output, molecule B is updated
        At present, both molecules must have the same atom ordering """

    assert mol_A.nAtoms() == mol_B.nAtoms()
    
    masses = mol_A.masses()

    assert all(masses == mol_B.masses())

    com_A = mol_A.centerOfMass()
    com_B = mol_B.centerOfMass()

    # shift both molecules' com to the origin
    mol_A.translate(-com_A) 
    mol_B.translate(-com_B) 

    coord_A = mol_A.coordinates()
    coord_B = mol_B.coordinates()

    # get rotation of B to match Eckart conditions
    Rmat = _get_Rot_alignment(coord_A,coord_B,masses)

    mol_B.transform(Rmat.T)

    mol_A.translate(com_A)
    mol_B.translate(com_A)

    
def check_Eckart(mol_A: Molecule, mol_B: Molecule) -> Tuple[np.ndarray,np.ndarray]:
    """ return the translational and rotational Eckart conditions (zero vectors, if fulfilled) """

    assert mol_A.nAtoms() == mol_B.nAtoms()                                             
    
    nAtoms = mol_A.nAtoms()
    masses = mol_A.masses()           
                                              
    assert all(masses == mol_B.masses())

    coord_A = mol_A.coordinates()
    coord_B = mol_B.coordinates()

    eckart_T = np.zeros((3))
    eckart_R = np.zeros((3))

    for ii in range(nAtoms):
        eckart_T += masses[ii] * (coord_B[ii] - coord_A[ii])
        eckart_R += masses[ii] * np.cross(coord_A[ii],coord_B[ii])

    return eckart_T, eckart_R

