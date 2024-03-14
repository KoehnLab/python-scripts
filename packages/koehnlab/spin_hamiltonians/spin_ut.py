import numpy as np


def get_spinmat(multiplicity):
        """
        Returns the matrix elements of the spin operator in every dimension with given multiplicity.
        This functions calculates the spin matrix elements with ladder operators.

        Args:
        -------------------------
        multiplicity -- Multiplicity of the system

        Returns:
        -------------------------
        s_x,s_y,s_z -- spin matrix in x,y,z - dimension
        """
        S = 0.5*(multiplicity-1)
        s_z = np.zeros((multiplicity,multiplicity),dtype = complex)
        s_plus = np.zeros((multiplicity,multiplicity),dtype = complex)
        s_minus = np.zeros((multiplicity,multiplicity),dtype = complex)
        for i in range(multiplicity):
                ms = S-i
                s_z[i,i] = ms
                s_plus[i-1,i] = np.sqrt(S*(S+1)-ms*(ms+1)) #defining ladder operators
                s_minus[i,i-1] = np.sqrt(S*(S+1)-ms*(ms+1))
        s_x = 0.5*(s_plus+s_minus)
        s_y = -0.5j*(s_plus-s_minus)                                                                                                                               
        return s_x,s_y,s_z



