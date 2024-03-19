import numpy as np
from .phys_const import muBcm,g_e


def get_magnetic_moment_matrix(s,l,unit: bool):
        """
        Computes the magnetic moment of the system in the basis of given spin and angular momentum matrix
        Procedure taken from: L. F. Chibotaru, L. Ungur; Ab initio calculation of anisotropic magnetic properties of complexes. I. Unique definition of
        pseudospin Hamiltonians and their derivation. J. Chem. Phys. 14 August 2012; 137 (6): 064112. https://doi.org/10.1063/1.4739763
        Args:
        ---------------------
        s -- spin matrix in Js
        l -- angular momentum matrix in (kg*m^2)/s
        unit -- Boolean if in multiples of bohr magneton

        Returns:
        ---------------------
        muMat -- matrix of magnetic moment in J/T
        """
        if unit:
            muMat = -muBcm*(g_e*s+l)
        else: 
            muMat = -1*(g_e*s+l)
        return muMat

def get_A_matrix(mu_x,mu_y,mu_z,n:int):
        """
        Computes the A matrix to calculate g-Tensor afterwards
        Procedure taken from: L. F. Chibotaru, L. Ungur; Ab initio calculation of anisotropic magnetic properties of complexes. I. Unique definition of
        pseudospin Hamiltonians and their derivation. J. Chem. Phys. 14 August 2012; 137 (6): 064112. https://doi.org/10.1063/1.4739763
        Args:
        ---------------------
        mu_x,y,z -- matrices of magnetic moment in SO basis in every dimension in J/T
        n -- number of states that are crucial, only needed if pseudospin is used so not all states are included in calculation

        Returns:
        ---------------------
        Amat -- helper matrix for g-Tensor
        """
        mu = [mu_x[:n,:n],mu_y[:n,:n],mu_z[:n,:n]]
        n = len(mu)
        Amat = np.zeros((n,n))
        for alpha in range(n):
                for beta in range(n):
                        prod = np.matmul(mu[alpha],mu[beta])
                        trace = np.matrix.trace(prod)
                        assert np.abs(trace.imag) < 1E-6
                        Amat[alpha,beta] = 0.5*trace.real
        return Amat

def get_g_tensor(A,multiplicity:int):
        """
        returns the g-Tensorin cartesian coordinates and his main values  of given helpermatrix A and given multiplicity of the system,
        also returns the rotation matrix between main axes and main magnetic axes. 
        Procedure taken from: L. F. Chibotaru, L. Ungur; Ab initio calculation of anisotropic magnetic properties of complexes. I. Unique definition of
        pseudospin Hamiltonians and their derivation. J. Chem. Phys. 14 August 2012; 137 (6): 064112. https://doi.org/10.1063/1.4739763
        Args:
        ----------------------
        A -- helper matrix
        multiplicity -- multiplicity of the system

        Returns:
        ----------------------
        g_diag -- eigenvalues of g-Tensor on diagonal
        Rmat -- rotation matrix
        """
        S = 0.5*(multiplicity-1)
        g_diag = np.zeros(np.shape(A))
        eigvalA,Rmat = np.linalg.eigh(A)
        for i in range(len(eigvalA)):
                g_diag[i,i] =np.sqrt(6*eigvalA[i]/(S*(S+1)*(2*S+1)))
        return g_diag,Rmat
