import numpy as np
from .phys_const import muBcm,g_e


def get_mu(s,l):
        """
        Computes the magnetic moment of the system in the basis of given spin and angular momentum matrix

        Args:
        ---------------------
        s -- spin matrix
        l -- angular momentum matrix

        Returns:
        ---------------------
        mu -- matrix of magnetic moment
        """
        mu = -muBcm*(g_e*s+l)
        return mu

def get_A_matrix(mu_x,mu_y,mu_z,n):
        """
        Computes the A matrix to calculate g-Tensor afterwards

        Args:
        ---------------------
        mu_x,y,z -- matrices of magnetic moment in SO basis in every dimension

        Returns:
        ---------------------
        A -- helper matrix for g-Tensor
        """
        mu = [mu_x[:n,:n],mu_y[:n,:n],mu_z[:n,:n]]
        n = len(mu)
        A = np.zeros((n,n))
        for alpha in range(n):
                for beta in range(n):
                        prod = np.matmul(mu[alpha],mu[beta])
                        trace = np.matrix.trace(prod)
                        assert np.abs(trace.imag) < 1E-6
                        A[alpha,beta] = 0.5*trace.real
        return A

def get_g_tensor(A,multiplicity):
        """
        returns the g-Tensorin cartesian coordinates and his main values  of given helpermatrix A and given multiplicity of the system, also returns the ro$        Args:
        ----------------------
        A -- helper matrix
        multiplicity -- multiplicity of the system
        Returns:
        ----------------------
        g -- g-Tensor
        g_eig -- main value of g-Tensor
        R -- rotation matrix
        """
        mu_B = 1
        S = 0.5*(multiplicity-1)
        g_diag = np.zeros(np.shape(A))
        eigvalA,R = np.linalg.eigh(A)
        for i in range(len(eigvalA)):
                g_diag[i,i] =np.sqrt(6*eigvalA[i]/(S*(S+1)*(2*S+1)))
        g = np.dot(np.dot(R,g_diag),np.linalg.inv(R))
        g_eig = np.diag(g_diag)
        if np.linalg.det(R) == -1:
                R = -1*R                                                                                                                                           
        return g,g_eig,R
