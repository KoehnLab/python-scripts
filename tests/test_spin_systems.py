#!/usr/bin/env python3

import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from koehnlab.spin_hamiltonians import spin, spin_system


class TestSpin(unittest.TestCase):
    def test_setSpin1(self):
        sp1 = spin(0.5)
        sp2 = spin(1.0)
        sp3 = spin(2.5)

        Mmat1 = sp1.getSpinMat()
        Mmat2 = sp2.getSpinMat()
        Mmat3 = sp3.getSpinMat()

        assert_array_almost_equal(
            Mmat1[0],
            np.array(
                [[0.,0.5],[0.5,0.]], dtype=complex
            ),
        )

        isq2 = np.sqrt(0.5)
        assert_array_almost_equal(
            Mmat2[0],
            np.array(
                [[0.0, isq2, 0.0], [isq2, 0.0, isq2], [0.0, isq2, 0.0]], dtype=complex
            ),
        )
        assert_array_almost_equal(
            Mmat2[1],
            np.array(
                [
                    [0.0, -isq2 * 1j, 0.0],
                    [isq2 * 1j, 0.0, -isq2 * 1j],
                    [0.0, isq2 * 1j, 0.0],
                ],
                dtype=complex,
            ),
        )
        sq2j = np.sqrt(2.0) * 1j
        sq5hj = np.sqrt(5.0) / 2.0 * 1j
        assert_array_almost_equal(
            Mmat3[1],
            np.array(
                [
                    [0.0, -sq5hj, 0.0, 0.0, 0.0, 0.0],
                    [sq5hj, 0.0, -sq2j, 0.0, 0.0, 0.0],
                    [0.0, sq2j, 0.0, -1.5j, 0.0, 0.0],
                    [0.0, 0.0, 1.5j, 0.0, -sq2j, 0.0],
                    [0.0, 0.0, 0.0, sq2j, 0.0, -sq5hj],
                    [0.0, 0.0, 0.0, 0.0, sq5hj, 0.0],
                ],
                dtype=complex,
            ),
        )
        assert_array_almost_equal(
            Mmat3[2], np.diag([5 / 2, 3 / 2, 1 / 2, -1 / 2, -3 / 2, -5 / 2])
        )


    def test_setSpin2(self):
        
        sp1 = spin(1.0)
        sp1.set_g([1.5,2.0,2.5])

        Mmat = sp1.getMMat()

        isq2 = np.sqrt(0.5)

        SX3 = np.array([
                [0.0, isq2, 0.0], [isq2, 0.0, isq2], [0.0, isq2, 0.0],
                ], dtype=complex
            )
        SY3 = np.array([
                [0.0, -isq2*1j, 0.0], [isq2*1j, 0.0, -isq2*1j], [0.0, isq2*1j, 0.0],
                ], dtype=complex
            )
        SZ3 = np.diag([1.0,0.0,-1.0])

        assert_array_almost_equal( Mmat[0], -1.5*SX3 )
        assert_array_almost_equal( Mmat[1], -2.0*SY3 )
        assert_array_almost_equal( Mmat[2], -2.5*SZ3 )

        # rotate axes
        sp1.set_axes([[isq2,isq2,0.],[0.,0.,1.],[isq2,-isq2,0.]])
        Mmat = sp1.getMMat()

        assert_array_almost_equal( Mmat[0], -1.75*SX3 + 0.25*SZ3 )
        assert_array_almost_equal( Mmat[1], -2.5 *SY3 )
        assert_array_almost_equal( Mmat[2],  0.25*SX3 - 1.75*SZ3 )


    def test_setSpin3(self):
 
         sp1 = spin(1.5)

         sp1.set_ZF(ZFaxial=-30,ZFrhombic=10)
         ZFmat = sp1.getZFmat()

         tensq3 = 10.*np.sqrt(3)
         ftnsq3 = 15.*np.sqrt(3)
         assert_array_almost_equal(
            ZFmat,
            np.array([[-30.,0.,tensq3,0.],[0.,30.,0.,tensq3],[tensq3,0.,30.,0.],[0.,tensq3,0.,-30.]],dtype=complex)
         )

         isq2 = np.sqrt(0.5)
         sp1.set_axes([[1.,0.,0.],[0.,isq2,isq2],[0.,-isq2,isq2]])
         ZFmat = sp1.getZFmat()
         assert_array_almost_equal(
            ZFmat,
            np.array([[-15.,tensq3*1j,ftnsq3,0.],[-tensq3*1j,15.,0.,ftnsq3],[ftnsq3,0.,15.,-tensq3*1j],[0.,ftnsq3,tensq3*1j,-15.]],dtype=complex)
         )

         sp2 = spin(1.5)

         sp2.set_ZF(ZFaxial=-30,ZFrhombic=10,ZFaxes=[[1.,0.,0.],[0.,isq2,isq2],[0.,-isq2,isq2]])
         ZFmat2 = sp2.getZFmat()
         assert_array_almost_equal(ZFmat,ZFmat2)


    def test_SpinInteraction1(self):

        sp1 = spin(0.5)
        sp2 = spin(0.5)

        sp1.set_g([2.0,2.0,3.2])
        sp2.set_g([2.0,2.0,3.2])

        sys = spin_system()

        sys.add("Cu1",sp1)
        sys.add("Cu2",sp2)
        sys.set_interaction("Cu1","Cu2",10)

        SMat = sys.getSpinMat()
        MMat = sys.getMMat()
        HMat = sys.getHMat()

        assert_array_almost_equal(
            SMat[1],
            1j*np.array(
                [[0.0,-.5,-.5,0.0],
                 [0.5,0.0,0.0,-.5],
                 [0.5,0.0,0.0,-.5],
                 [0.0,0.5,0.5,0.0]],dtype=complex)
        )

        assert_array_almost_equal(
            MMat[0],
            -1.*np.array(
                [[0.0,1.0,1.0,0.0],
                 [1.0,0.0,0.0,1.0],
                 [1.0,0.0,0.0,1.0],
                 [0.0,1.0,1.0,0.0]],dtype=complex)
        )

        assert_array_almost_equal(
            HMat,
            np.array(
                [[2.5, 0.0, 0.0, 0.0],
                 [0.0,-2.5, 5.0, 0.0],
                 [0.0, 5.0,-2.5, 0.0],
                 [0.0, 0.0, 0.0, 2.5]],dtype=complex)
        )

    def test_SpinInteraction2(self):

        sp1 = spin(0.5)
        sp2 = spin(0.5)

        sp1.set_g([2.0,2.0,3.2])
        sp2.set_g([2.0,2.0,3.2])

        # now with rotated axes
        c = 0.5
        s = 0.5*np.sqrt(3)
        sp1.set_axes([[c,0,s],[0,1.,0],[-s,0,c]])
        sp2.set_axes([[c,0,-s],[0,1.,0],[s,0,c]])

        sys = spin_system()

        sys.add("Cu1",sp1)
        sys.add("Cu2",sp2)
        sys.set_interaction("Cu1","Cu2",10)

        SMat = sys.getSpinMat()
        MMat = sys.getMMat()
        HMat = sys.getHMat()

        assert_array_almost_equal(
            SMat[1],
            1j*np.array(
                [[0.0,-.5,-.5,0.0],
                 [0.5,0.0,0.0,-.5],
                 [0.5,0.0,0.0,-.5],
                 [0.0,0.5,0.5,0.0]],dtype=complex)
        )

        x03s3 = 0.3*np.sqrt(3)
        assert_array_almost_equal(
            MMat[0],
            np.array(
                [[ 0.0 , -1.45, -1.45, 0.0 ],
                 [-1.45,-x03s3,  0.0 ,-1.45],
                 [-1.45,   0.0, x03s3,-1.45],
                 [ 0.0 , -1.45, -1.45, 0.0 ]],dtype=complex)
        )

        x125s3 = 1.25*np.sqrt(3.)
        assert_array_almost_equal(
            HMat,
            np.array(
                [[ -1.25, x125s3,-x125s3, -3.75],
                 [ x125s3,  1.25,  1.25, x125s3],
                 [-x125s3,  1.25,  1.25,-x125s3],
                 [ -3.75, x125s3,-x125s3, -1.25]],dtype=complex)
        )

    def test_SpinInteraction3(self):

        # case: hyperfine coupling
        sp1 = spin(0.5)
        sp2 = spin(2.5,"n",0.5)

        sp1.set_g([2.0,2.0,3.2])

        sys = spin_system()

        sys.add("el",sp1)
        sys.add("nuc",sp2)

        Aiso = 0.1
        Aax  = 0.01
        sys.set_interaction("el","nuc",Aiso,Aax)

        # test also this feature:
        sys.set_order(["nuc","el"])

        SMat = sys.getSpinMat()
        MMat = sys.getMMat()
        HMat = sys.getHMat()

        # check a few specific elements
        assert_almost_equal(SMat[0][1,0],0.5)
        assert_almost_equal(SMat[0][2,0],np.sqrt(1.25))
        assert_almost_equal(SMat[1][4,2],np.sqrt(2)*1j)
        assert_almost_equal(SMat[2][1,1],2.0)

        assert_almost_equal(MMat[0][0,0],0.0)
        assert_almost_equal(MMat[0][1,0],-1.0)
        assert_almost_equal(MMat[0][2,0],0.000304450)
        assert_almost_equal(MMat[0][11,10],-1.0)

        assert_almost_equal(MMat[1][4,2],0.000385102j)

        assert_almost_equal(MMat[2][0,0],-1.599319229)
        assert_almost_equal(MMat[2][4,4],-1.599863846)
        assert_almost_equal(MMat[2][8,8],-1.600408463)
        assert_almost_equal(MMat[2][9,9], 1.599591537)

        assert_almost_equal(HMat[0,0],2/15)
        assert_almost_equal(HMat[1,1],-2/15)
        assert_almost_equal(HMat[2,1],0.108076619)
        assert_almost_equal(HMat[1,2],0.108076619)
        assert_almost_equal(HMat[4,4],2/75)
        assert_almost_equal(HMat[4,3],0.136707311)
        assert_almost_equal(HMat[11,11],2/15)

    def test_SpinInteraction4(self):

        # case: hyperfine coupling with nucleus axes rotated
        sp1 = spin(0.5)
        sp2 = spin(2.5,"n",0.5)

        sp1.set_g([2.0,2.0,3.2])
        
        isq2 = np.sqrt(0.5)
        sp2.set_axes([[isq2,-isq2,0],[isq2,isq2,0],[0,0,1]])


        sys = spin_system()

        sys.add("el",sp1)
        sys.add("nuc",sp2)

        Aiso = 0.1
        Aax  = 0.01
        sys.set_interaction("el","nuc",Aiso,Aax)

        # test also this feature:
        sys.set_order(["nuc","el"])

        SMat = sys.getSpinMat()
        MMat = sys.getMMat()
        HMat = sys.getHMat()

        # check a few specific elements
        assert_almost_equal(SMat[0][1,0],0.5)
        assert_almost_equal(SMat[0][2,0],np.sqrt(1.25))
        assert_almost_equal(SMat[1][4,2],np.sqrt(2)*1j)
        assert_almost_equal(SMat[2][1,1],2.0)

        assert_almost_equal(MMat[0][0,0],0.0)
        assert_almost_equal(MMat[0][1,0],-1.0)
        assert_almost_equal(MMat[0][2,0],0.000304450)
        assert_almost_equal(MMat[0][11,10],-1.0)

        assert_almost_equal(MMat[1][4,2],0.000385102j)

        assert_almost_equal(MMat[2][0,0],-1.599319229)
        assert_almost_equal(MMat[2][4,4],-1.599863846)
        assert_almost_equal(MMat[2][8,8],-1.600408463)
        assert_almost_equal(MMat[2][9,9], 1.599591537)

        assert_almost_equal(HMat[0,0],2/15)
        assert_almost_equal(HMat[1,1],-2/15)
        assert_almost_equal(HMat[2,1], 0.108076619j)
        assert_almost_equal(HMat[1,2],-0.108076619j)
        assert_almost_equal(HMat[4,4],2/75)
        assert_almost_equal(HMat[4,3],0.136707311j)
        assert_almost_equal(HMat[11,11],2/15)


if __name__ == "__main__":
    unittest.main()
