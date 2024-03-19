import numpy as np
from spin_hamiltonians import spin_utils

def get_propmat_prod(l,multiplicity:int,spat_num:int):
        """
        Returns the given l matrix in productbasis (spin-spatial basis).
        Args:
        ---------------------
        l -- matrix of angular momentum in (kg*m^2)/s
        multiplicity -- multiplicity of the system
        spat_num - number of spatial states
        Returns:
        ---------------------
        Lmat -- Matrix of angular momentum in productbasis in (kg*m^2)/s
        """
        dim = multiplicity*spat_num
        Lmat = np.zeros((dim,dim),dtype = complex)
        for x in range(dim):
                for y in range(dim):
                        ms = m.floor(x/spat_num)
                        ms_strich = m.floor(y/spat_num)
                        i = x%spat_num
                        j = y%spat_num
                        if ms == ms_strich:
                            Lmat[x,y] = l[i,j]
        return Lmat

def get_spinmat_prod(s,multiplicity:int,spat_num:int)
        """
        Computes the spin matrix elements of the given spin matrix in the productbasis (spin-spatial basis)
        Blocked over ms number.
        Args:
        ------------------------
        s -- spin matrix in spin basis in Js
        multiplicity -- Multiplicity of the system
        spat_num -- number of spatial states

        Returns:
        ------------------------
        SmatX,SmatY,SmatZ -- spin matrix in the productbasis in Js
        """
        dim = multiplicity*spat_num
        spin = 0.5*(multiplicity-1)
        Smat = np.zeros((dim,dim),dtype = complex)
        if spin == 0:
                return Smat
        else:
                for x in range(dim):
                        for y in range(dim):
                                ms = m.floor(x/spat_num)
                                ms_strich = m.floor(y/spat_num)
                                i = x%spat_num
                                j = y%spat_num
                                if i==j:
                                        Smat[x,y] = s[ms,ms_strich]
                return Smat

def get_multispinmat_prod(spins,spat_nums,coord:str):
        """
        Returns the spin matrix of all given Spins S in the productbasis of all spins and all spatial coordinates

        Args:
        --------------------------
        s -- array of multiple spin states, sorted in descending order
        spat_nums -- array of numbers of spatial states corresponding to the spin states
        coord -- x,y,z

        Returns:
        --------------------------
        Smat -- spin matrix in Productbasis of given spin and spatial states in Js
        """
        assert len(spin) == len(spat_nums)
        multiplicities = [2*S+1 for S in spins]
        dim = int(np.sum(multiplicities*spat_nums))
        Smat = np.zeros((dim,dim),dtype = complex)
        lower_bound = 0
        upper_bound = 0
        for i,mult in enumerate(multiplicities):
            spinmat_coord = spin_utils.spinMat(mult,coord)
            spinmat_prod = get_spinmat_prod(spinmat_coord,mult,spat_nums[i])
            upper_bound += int(mult*spat_nums[i])
            assert upper_bound-lower_bound == np.shape(spinmat_prod)[0]
            assert upper_bound-lower_bound == np.shape(spinmat_prod)[1]
            Smat[lower_bound:upper_bound,lower_bound:upper_bound] = spinmat_prod
            lower_bound = upper_bound
        return Smat

def get_multipropmat_prod(l,spins,spat_nums):
        """
        Calculates the l matrix in the productbasis for multiple spins

        Args:
        --------------------------
        l -- matrix of angular momentum that needs to be transformed in the basis for multiple spins in (kg*m^2)/s
        s -- array of multiple spin states, sorted in ascending order 
        spat_nums -- array of numbers of spatial states corresponding to the spin states

        Returns:
        --------------------------
        Lmat -- matrix of angular momentum in Productbasis of given spin and spatial states in (kg*m^2)/s
        """
        assert len(spins) == len(spat_nums)
        multiplicities = [2*S+1 for S in spins]
        dim = int(np.sum(multiplicities*spat_nums))
        Lmat = np.zeros((dim,dim),dtype = complex)
        lower_bound = 0
        upper_bound = 0
        for i,mult in enumerate(multiplicities):
            propmat_prod = get_propmat_prod(l,mult,spat_nums[i])
            upper_bound += int(mult*spat_nums[i])
            assert upper_bound-lower_bound == np.shape(propmat_prod)[0]
            assert upper_bound-lower_bound == np.shape(propmat_prod)[1]
            Lmat[lower_bound:upper_bound,lower_bound:upper_bound] = propmat_prod
            lower_bound = upper_bound
        return Lmat

def transform_so(m,eigvecs):
        """
        Transforms a given matrix m into the SO basis

        Args:
        ---------------------
        m -- matrix which is transformed to SO basis
        eigvecsh -- eigenvectors of the SO matrix to perform the transformation

        Returns:
        --------------------
        Mmat -- matrix m transformed into SO basis
        """
        Mmat = np.matmul(np.conj(eigvecs.T),np.matmul(m,eigvecs))
        return Mmat




