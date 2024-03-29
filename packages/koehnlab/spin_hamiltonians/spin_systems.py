from enum import Enum

import numpy as np 

from .phys_const import ge, muNbohr
from .spin_utils import spinMat

class SpinType(Enum):
    Electronic = 1
    Nuclear = 2
    # aliases for people who are fond of shortcuts (like me)
    El = 1
    Nuc = 2

numThr = 1e-8
maxDim = 1000 

class Spin:
    """ This class defines a spin object  
        Usage: Define an electronic spin system with, say, S=3/2 by 
          my_spin = Spin(1.5)
        or
          my_spin = Spin(1.5,SpinType.Elec)
        Define a nuclear spin with, say, S=3 and gnuc=0.72 by
          my_nuc_spin = Spin(3,SpinType.Nuc,0.72)
    """

    def __init__(self,S,type=SpinType.Electronic,gnuc=None):
        self.S = S
        self.type = type
        self.axes = np.diag([1.,1.,1.])
        if type==SpinType.Electronic:
            self.g = np.diag([ge,ge,ge])
        elif type==SpinType.Nuclear:
            if gnuc is None:
                raise Exception("Must provide g factor for nucleus")
            # the electron g factor is taken here positive, so there is minus sign when
            # the magnetic moments are built; for the nuclei, we thus use a negative sign
            # here by default; note that nuclei can have both pos and neg. nuclear g factors
            self.g = -np.diag([gnuc,gnuc,gnuc])*muNbohr
        else:
            raise Exception(f"Unknown type: {type}")
        self.ZFaxes = np.diag([1.,1.,1.])
        self.ZFaxial = 0.
        self.ZFrhombic = 0.
        self.ZFten = np.diag([0.,0.,0.])


    def set_g(self,g):
        """ set the diagonal values of the g matrix (will be rotated to the defined axes) """
        self.g = np.matmul(self.axes,np.matmul(np.diag(g),self.axes.T))


    def set_axes(self,axes):
        """ set the axes for the spin system; must be orthogonal and right-handed
            automatically updates the g matrix """
        axes = np.array(axes)
        check = np.matmul(axes.T,axes)-np.diag([1.,1.,1.])
        if np.vdot(check,check) > numThr*numThr:
            raise Exception("Non-orthogonal axes on input")
        if np.linalg.det(axes) < 0.:
            raise Exception("Axes are not defining right-handed system")
        self.axes = axes
        self.g = np.matmul(axes,np.matmul(self.g,axes.T))


    def set_ZF(self,ZFaxial=0.,ZFrhombic=0.,ZFaxes=None):
        """ set the zero-field splitting (axiality D, rhombicity E, 
            and the anisotropy axes (if different from magnetic axes) """
        if ZFaxes is not None:
            self.ZFaxes = np.array(ZFaxes)
            check = np.matmul(self.ZFaxes.T,self.ZFaxes)-np.diag([1.,1.,1.])
            if np.vdot(check,check) > numThr*numThr:
                raise Exception("set_ZF: Non-orthogonal axes on input")
            if np.linalg.det(ZFaxes) < 0.:
                raise Exception("set_ZF: Axes are not defining right-handed system")
        else:
            self.ZFaxes = self.axes
        self.ZFaxial = ZFaxial
        self.ZFrhombic = ZFrhombic
     
        self.ZFten = np.matmul(self.ZFaxes,
                     np.matmul(np.diag([-self.ZFaxial/3.+self.ZFrhombic,-self.ZFaxial/3.-self.ZFrhombic,2*self.ZFaxial/3.]),
                               self.ZFaxes.T))


    def get_spin_mat(self):
        """ return spin matrix elements 
            <i|S_c|j>  c=x,y,z
        x,y,z are the reference basis """

        dim = int(2*self.S)+1
        Mat = np.zeros((3,dim,dim),dtype=complex)

        Mat[0] += spinMat(self.S,'x')
        Mat[1] += spinMat(self.S,'y')
        Mat[2] += spinMat(self.S,'z')
       
        return Mat


    def get_M_mat(self):
        """ return magnetic moment matrix elements
            note the minus sign!
            <i|M_c|j> = -g <i|S_c|j>  c=x,y,z
        x,y,z are the reference basis; unit is Bohr magnetons """

        dim = int(2*self.S)+1
        Mat = np.zeros((3,dim,dim),dtype=complex)

        # g is already in the correct basis
        # it also contains the (possible) sign reversal for nuclear spins

        k = -1
        for comp in ['x','y','z']:
            k += 1
            tmpMat = spinMat(self.S,comp)
            Mat[0] -= self.g[0,k]*tmpMat
            Mat[1] -= self.g[1,k]*tmpMat
            Mat[2] -= self.g[2,k]*tmpMat

        return Mat


    def get_ZF_mat(self):
        """ get the zero-field contributions from this center; unit is cm-1 """

        dim = int(2*self.S)+1
        Mat = np.zeros((dim,dim),dtype=complex)

        # transform to magnetic axes
        ZFtenT = np.matmul(self.axes,np.matmul(self.ZFten,self.axes.T))

        Smat = np.array([spinMat(self.S,'x'),spinMat(self.S,'y'),spinMat(self.S,'z')])

        for k in range(3):
            for l in range(k):
                Mat += ZFtenT[k,l]*(np.matmul(Smat[k],Smat[l])+np.matmul(Smat[l],Smat[k]))
            Mat += ZFtenT[k,k]*np.matmul(Smat[k],Smat[k])

        return Mat


class SpinSystem:
    """ This class defines a collection of Spin objects
        Usage: define an object and add spins by
            sp_sys = SpinSystem()
            sp_sys.add(Spin("spin1",0.5))
            sp_sys.add(Spin("spin2",1.5))  """

    def __init__(self):
        self.spins = {}
        self.order = []
        self.interaction = []
        self.dimension = 0


    def add(self,label,spin):
        """ add a new object of type Spin and name it by a label """
        self.spins[label] = spin
        self.order.append(label)
        dim = int(2.*spin.S)+1
        if self.dimension == 0:
            self.dimension = dim
        else:
            self.dimension *= dim


    def set_order(self,labels):
        """ call this routine with a list of (existing!) labels to change the sequence
            has impact on the way the Hamiltonian and property matrices are ordered """
        new_order = []
        for label in labels:
            if label in self.spins.keys():
                new_order.append(label)
            else:
                raise Exception(f"Undefined label: {label}")
        if len(new_order) != len(self.order):
            raise Exception("Label list incomplete")
        self.order = new_order


    def set_interaction(self,label1,label2,Jiso,Jax=0.,Jrh=0.):
        """ Add an interaction between the spin centers with label1 and label
            Jiso is the isotropic part, Jax the axial part (zz), and Jrh the rhombic part (xx - yy) """
        if label1 not in self.spins.keys():
            raise Exception(f"Undefined label: {label1}")
        if label2 not in self.spins.keys():
            raise Exception(f"Undefined label: {label2}")
        Jmat = np.diag([Jiso-Jax/3.+Jrh,Jiso-Jax/3.+Jrh,Jiso+2*Jax/3.])
        # set the actual matrices later to avoid mismatches due to changed axes
        # JmatT = np.matmul(spin1.axes,np.matmul(Jmat,spin2.axes))
        
        self.interaction.append([label1,label2,Jmat])


    def get_spin_mat(self):
        """ get the (pseudo) spin matrix of the total system """

        dim_before=1
        dim_after=self.dimension

        if self.dimension > maxDim:
            raise Exception(f"Dimension too large: {self.dimension}")

        Mat = np.zeros((3,self.dimension,self.dimension),dtype=complex)

        for label in self.order:
            SMat = self.spins[label].get_spin_mat()
            cdim = SMat.shape[1]

            dim_after //= cdim
            if dim_before == 1 and dim_after == 1:
                Mat += SMat
            elif dim_before == 1:
                Mat[0] += np.kron(SMat[0],np.identity(dim_after,dtype=complex))
                Mat[1] += np.kron(SMat[1],np.identity(dim_after,dtype=complex))
                Mat[2] += np.kron(SMat[2],np.identity(dim_after,dtype=complex))
            elif dim_after == 1:
                Mat[0] += np.kron(np.identity(dim_before,dtype=complex),SMat[0])
                Mat[1] += np.kron(np.identity(dim_before,dtype=complex),SMat[1])
                Mat[2] += np.kron(np.identity(dim_before,dtype=complex),SMat[2])
            else:
                Mat[0] += np.kron(np.identity(dim_before,dtype=complex),np.kron(SMat[0],np.identity(dim_after,dtype=complex)))
                Mat[1] += np.kron(np.identity(dim_before,dtype=complex),np.kron(SMat[1],np.identity(dim_after,dtype=complex)))
                Mat[2] += np.kron(np.identity(dim_before,dtype=complex),np.kron(SMat[2],np.identity(dim_after,dtype=complex)))

            dim_before *= cdim

        return Mat


    def get_M_mat(self):
        """ get the magnetic moment matrix elements of the total system """

        dim_before=1
        dim_after=self.dimension

        if self.dimension > maxDim:
            raise Exception(f"Dimension too large: {self.dimension}")

        Mat = np.zeros((3,self.dimension,self.dimension),dtype=complex)

        for label in self.order:
            MMat = self.spins[label].get_M_mat()
            cdim = MMat.shape[1]

            dim_after //= cdim
            if dim_before == 1 and dim_after == 1:
                Mat += MMat
            elif dim_before == 1:
                Mat[0] += np.kron(MMat[0],np.identity(dim_after,dtype=complex))
                Mat[1] += np.kron(MMat[1],np.identity(dim_after,dtype=complex))
                Mat[2] += np.kron(MMat[2],np.identity(dim_after,dtype=complex))
            elif dim_after == 1:
                Mat[0] += np.kron(np.identity(dim_before,dtype=complex),MMat[0])
                Mat[1] += np.kron(np.identity(dim_before,dtype=complex),MMat[1])
                Mat[2] += np.kron(np.identity(dim_before,dtype=complex),MMat[2])
            else:
                Mat[0] += np.kron(np.identity(dim_before,dtype=complex),np.kron(MMat[0],np.identity(dim_after,dtype=complex)))
                Mat[1] += np.kron(np.identity(dim_before,dtype=complex),np.kron(MMat[1],np.identity(dim_after,dtype=complex)))
                Mat[2] += np.kron(np.identity(dim_before,dtype=complex),np.kron(MMat[2],np.identity(dim_after,dtype=complex)))

            dim_before *= cdim

        return Mat


    def get_H_mat(self):
        """ get the (B field free) Hamiltonian matrix of the system """
        
        dim_before=1
        dim_after=self.dimension

        if self.dimension > maxDim:
            raise Exception(f"Dimension too large: {self.dimension}")

        Mat = np.zeros((self.dimension,self.dimension),dtype=complex)

        # start with one-center terms:
        # collect dimensions:
        dims = []
        for label in self.order:
            MMat = self.spins[label].get_ZF_mat()
            cdim = MMat.shape[1]

            dims.append(cdim)

            # skip over zero contribution
            if np.vdot(MMat,MMat) < numThr*numThr:
                continue

            dim_after //= cdim
            if dim_before == 1 and dim_after == 1:
                Mat += MMat
            elif dim_before == 1:
                Mat += np.kron(MMat,np.identity(dim_after,dtype=complex))
            elif dim_after == 1:
                Mat += np.kron(np.identity(dim_before,dtype=complex),MMat)
            else:
                Mat += np.kron(np.identity(dim_before,dtype=complex),np.kron(MMat,np.identity(dim_after,dtype=complex)))

            dim_before *= cdim

        

        for interact in self.interaction:
            label1 = interact[0]
            label2 = interact[1]
            Jmat = interact[2]
            dim_before = 1
            idx = -1
            switch = False
            for label in self.order:
                idx += 1
                if label1 == label:
                    break
                if label2 == label:
                    switch = True
                    break
                dim_before *= dims[idx]

            dim_between = 1
            jdx = idx
            for label in self.order[idx+1:]:
                jdx += 1
                if label1 == label or label2 == label:
                    break
                dim_between *= dims[jdx]

            dim_after = 1
            for kdx in range(jdx+1,len(dims)):
                dim_after *= dims[kdx]

            if not switch:
                spin1 = self.spins[label1]
                spin2 = self.spins[label2]
            else:
                spin1 = self.spins[label2]
                spin2 = self.spins[label1]

            Smat1 = spin1.get_spin_mat()
            Smat2 = spin2.get_spin_mat()

            # relative rotation:
            relaxes = np.matmul(spin1.axes,spin2.axes.T)

            # OK as long as Jmat is symmetric:
            JmatT = np.matmul(relaxes,np.matmul(Jmat,relaxes))

            for k in range(3):
                for l in range(3):
                    if np.abs(JmatT[k,l]) < numThr:
                        continue
                    Mat += JmatT[k,l]*np.kron(np.kron(np.kron(np.kron(
                                              np.identity(dim_before,dtype=complex),
                                              Smat1[k]),
                                              np.identity(dim_between,dtype=complex)),
                                              Smat2[l]),
                                              np.identity(dim_after,dtype=complex)
                                              )

        return Mat


