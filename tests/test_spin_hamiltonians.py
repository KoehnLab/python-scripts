#!/usr/bin/env python3

import unittest
from koehnlab.spin_hamiltonians.spin_utils import diagonalizeSpinHamiltonian

#from typing import List, Sequence, Tuple

import numpy as np

from koehnlab.spin_hamiltonians import (
    spinMat, tprod, diagonalizeSpinHamiltonian
)


class TestFiniteDifference(unittest.TestCase):

    # compare to numpy arrays to a given precision:
    def assertNumpyArrayAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        self.assertEqual(first.shape, second.shape, msg=msg)

        for x1, x2 in zip(np.nditer(first),np.nditer(second)):
           self.assertAlmostEqual(x1, x2, places=places, msg=msg, delta=delta)


    def test_spinMat(self):
        isq2 = np.sqrt(.5)
        self.assertNumpyArrayAlmostEqual(
                spinMat(1,'x'),
                np.array([[0.,isq2,0.],[isq2,0.,isq2],[0.,isq2,0.]])
                )
        self.assertNumpyArrayAlmostEqual(
                spinMat(1,'y'),
                np.array([[0.,-isq2*1j,0.],[isq2*1j,0.,-isq2*1j],[0.,isq2*1j,0.]])
                )
        sq2j = np.sqrt(2.)*1j
        sq5hj = np.sqrt(5.)/2.*1j
        self.assertNumpyArrayAlmostEqual(
                spinMat(5/2,'y'),
                np.array([[0.   ,-sq5hj,   0.,   0.,    0.,    0.],
                          [sq5hj,    0.,-sq2j,   0.,    0.,    0.],
                          [0.   ,  sq2j,   0.,-1.5j,    0.,    0.],
                          [0.   ,    0., 1.5j,   0., -sq2j,    0.],
                          [0.   ,    0.,   0., sq2j,    0.,-sq5hj],
                          [0.   ,    0.,   0.,   0., sq5hj,    0.]])
                )
        self.assertNumpyArrayAlmostEqual(
               spinMat(5/2,'z'),
               np.diag([5/2,3/2,1/2,-1/2,-3/2,-5/2])
               )
        self.assertNumpyArrayAlmostEqual(
               spinMat(3/2,'X'),
               spinMat(3/2,'x')
               )

    def test_tprod(self):
        A = np.array([[0.,-1.],[1.,2.]])
        B = B = np.diag([1.,2.,3.])
        self.assertNumpyArrayAlmostEqual(
                tprod(A,B),
                np.array([[ 0.,  0.,  0., -1., -0., -0.],
                          [ 0.,  0.,  0., -0., -2., -0.],
                          [ 0.,  0.,  0., -0., -0., -3.],
                          [ 1.,  0.,  0.,  2.,  0.,  0.],
                          [ 0.,  2.,  0.,  0.,  4.,  0.],
                          [ 0.,  0.,  3.,  0.,  0.,  6.]])
                )

    def test_diagonalizeSpinHamiltonian(self):
        sq3t = 10.*np.sqrt(3.)
        sq3  = np.sqrt(3.)
        Hmat = np.array([[-247.5 ,    0.  ,   sq3t ,    0.  ],
                         [   0.  ,  -27.5 ,    0.  ,   sq3t ],
                         [  sq3t ,    0.  ,  -27.5 ,    0.  ],
                         [   0.  ,   sq3t ,    0.  , -247.5 ]],dtype=complex
                         )
        MmatX = np.array([[0. ,sq3, 0., 0.],
                          [sq3, 0., 2., 0.],
                          [0. , 2., 0.,sq3],
                          [0. , 0.,sq3, 0.]],dtype=complex
                        ) 
        MmatY = np.array([[0. ,-sq3, 0., 0.],
                          [sq3, 0., -2., 0.],
                          [0. , 2., 0.,-sq3],
                          [0. , 0.,sq3, 0.]],dtype=complex
                        )*1j
        MmatZ = np.diag([3.,1.,-1.,-3.])
        
        ev1,U = diagonalizeSpinHamiltonian(Hmat)
        self.assertNumpyArrayAlmostEqual(ev1,np.array([-248.85528726, -248.85528726,  -26.14471274,  -26.14471274]))
        # test also standard choice of Kramers pair:
        MZT = np.matmul(np.conj(U.T),np.matmul(MmatZ,U))
        self.assertAlmostEqual(np.abs(MZT[0,0]),2.97565832)
        self.assertAlmostEqual(np.abs(MZT[1,1]),2.97565832)
        self.assertAlmostEqual(np.abs(MZT[2,2]),0.97565832)
        self.assertAlmostEqual(np.abs(MZT[3,3]),0.97565832)
        self.assertAlmostEqual(MZT[0,2],-0.31108551)

        # test with Zeeman terms:
        ev2,_ = diagonalizeSpinHamiltonian(Hmat,np.array([MmatX,MmatY,MmatZ]),np.array([0.1,0.1,5.]))
        self.assertNumpyArrayAlmostEqual(ev2,np.array([-255.80378613, -241.91164761,  -28.42353524,  -23.86103102]))



if __name__ == "__main__":
    unittest.main()

