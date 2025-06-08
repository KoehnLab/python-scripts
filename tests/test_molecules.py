#!/usr/bin/env python3

import unittest
import os

from koehnlab.molecular import Molecule, readMolecule, writeMolecule, Eckart_alignment, check_Eckart

import numpy as np
from numpy.testing import assert_almost_equal

script_dir: str = os.path.dirname(os.path.realpath(__file__))
data_dir: str = os.path.join(script_dir, "molecules")


class TestMolecules(unittest.TestCase):

    def test_align(self):

        mol_A = readMolecule(os.path.join(data_dir, "biphenyl_unaligned.xyz"))

        mol_A.bringToStandardOrientation()

        mol_B = readMolecule(os.path.join(data_dir, "biphenyl_aligned.xyz"))

        assert_almost_equal(mol_A.coordinates(),mol_B.coordinates())


    def test_eckart(self):

        mol_A = readMolecule(os.path.join(data_dir, "biphenyl_aligned.xyz"))
        mol_B = readMolecule(os.path.join(data_dir, "biphenyl_planar.xyz"))

        Eckart_alignment(mol_A,mol_B)

        mol_C = readMolecule(os.path.join(data_dir, "biphenyl_planar_ec.xyz"))

        assert_almost_equal(mol_B.coordinates(),mol_C.coordinates())

        eckT,eckR = check_Eckart(mol_A,mol_B)

        assert_almost_equal(eckT,np.zeros((3)))
        assert_almost_equal(eckR,np.zeros((3)))


if __name__ == "__main__":
    unittest.main()
