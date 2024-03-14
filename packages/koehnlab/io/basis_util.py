import numpy as np

def get_multispinmat_prod(s,spat_nums,coord):
        """
        Returns the spin matrix of all given Spins S in the productbasis of all spins and all spatial coordinates

        Args:
        --------------------------
        s -- array of multiple spin states, sorted in descending order
        spat_nums -- array of numbers of spatial states corresponding to the spin states
        coord -- 0,1,2 for x,y,z

        Returns:
        --------------------------
        S -- spin matrix in Productbasis of given spin and spatial states
        """
        multiplicities = []
        for i in range(len(s)):
                multiplicities.append(int(2*s[i]+1))
        dim = int(np.sum(multiplicities*spat_nums))
        S = np.zeros((dim,dim),dtype = complex)
        lower_bound = 0
        upper_bound = 0
        for mult in multiplicities:
                i = multiplicities.index(mult)
                spinmat_x,spinmat_y,spinmat_z = get_spinmat(mult)
                spinmat = [spinmat_x,spinmat_y,spinmat_z]
                spinmat_prod = get_spinmat_prod(spinmat[coord],mult,spat_nums[i])
                upper_bound += int(mult*spat_nums[i])
                assert upper_bound-lower_bound == np.shape(spinmat_prod)[0]
                assert upper_bound-lower_bound == np.shape(spinmat_prod)[1]
                S[lower_bound:upper_bound,lower_bound:upper_bound] = spinmat_prod
                lower_bound = upper_bound
        return S

def get_multipropmat_prod(l,s,spat_nums):
        """
        Calculates the l matrix in the productbasis for multiple spins

        Args:
        --------------------------
        l -- matrix of angular momentum that needs to be transformed in the basis for multiple spins
        s -- array of multiple spin states, sorted in ascending order
        spat_nums -- array of numbers of spatial states corresponding to the spin states

        Returns:
        --------------------------
        L -- matrix of angular momentum in Productbasis of given spin and spatial states
        """
        multiplicities = []
        for i in range(len(s)):
                multiplicities.append(int(2*s[i]+1))
        dim = int(np.sum(multiplicities*spat_nums))
        L = np.zeros((dim,dim),dtype = complex)
        lower_bound = 0
        upper_bound = 0
        for mult in multiplicities:
                i = multiplicities.index(mult)
                propmat_prod = get_propmat_prod(l,mult,spat_nums[i])
                upper_bound += int(mult*spat_nums[i])
                assert upper_bound-lower_bound == np.shape(propmat_prod)[0]
                assert upper_bound-lower_bound == np.shape(propmat_prod)[1]
                L[lower_bound:upper_bound,lower_bound:upper_bound] = propmat_prod
                lower_bound = upper_bound
        return L

def transform_so(m,eigvecs):
        """
        Transforms a given matrix m into the SO basis

        Args:
        ---------------------
        m -- matrix which is transformed to SO basis
        eigvecsh -- eigenvectors of the SO matrix to perform the transformation

        Returns:
        --------------------
        M -- matrix m transformed into SO basis
        """
        M = np.matmul(np.conj(eigvecs.T),np.matmul(m,eigvecs))
        return M




