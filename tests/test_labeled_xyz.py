#!/usr/bin/env python3

import unittest
import os
import tempfile

from koehnlab.molecular import (
    Element,
    readMolecule,
    readLabeledMolecule,
    writeMolecule,
)

import numpy as np
from numpy.testing import assert_almost_equal


def _writeFile(directory: str, content: str) -> str:
    path = os.path.join(directory, "test.xyz")
    with open(path, "w") as f:
        f.write(content)
    return path


class TestLabeledXYZ(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_labeled(self):
        path = _writeFile(
            self.dir,
            "3\ncomment\nC1 0 0 0\nCl2 1.5 0 0\nH17 0 1 0.5\n",
        )

        mol, labels = readLabeledMolecule(path)

        self.assertEqual(labels, ["C1", "Cl2", "H17"])
        self.assertEqual(
            [a.element for a in mol.atoms], [Element.C, Element.Cl, Element.H]
        )
        assert_almost_equal(mol.coordinates()[2], [0, 1, 0.5])

    def test_unlabeled_gets_counters(self):
        path = _writeFile(
            self.dir, "4\n\nC 0 0 0\nH 1 0 0\nC 0 1 0\nH 0 0 1\n"
        )

        mol, labels = readLabeledMolecule(path)

        self.assertEqual(labels, ["C1", "H1", "C2", "H2"])
        self.assertEqual(mol.nAtoms(), 4)

    def test_errors(self):
        # mixed
        path = _writeFile(self.dir, "2\n\nC1 0 0 0\nH 1 0 0\n")
        with self.assertRaises(RuntimeError):
            readLabeledMolecule(path)

        # duplicate labels
        path = _writeFile(self.dir, "2\n\nC1 0 0 0\nC1 1 0 0\n")
        with self.assertRaises(RuntimeError):
            readLabeledMolecule(path)

        # unknown element
        path = _writeFile(self.dir, "1\n\nXx1 0 0 0\n")
        with self.assertRaises(RuntimeError):
            readLabeledMolecule(path)

    def test_plain_reader_stays_strict(self):
        path = _writeFile(self.dir, "1\n\nC1 0 0 0\n")
        with self.assertRaises(KeyError):
            readMolecule(path)

    def test_label_round_trip(self):
        path = _writeFile(self.dir, "2\n\nC1 0 0 0\nCl2 1.5 0 0.25\n")
        mol, labels = readLabeledMolecule(path)

        out = os.path.join(self.dir, "out.xyz")
        writeMolecule(mol, out, labels=labels)

        mol2, labels2 = readLabeledMolecule(out)
        self.assertEqual(labels, labels2)
        assert_almost_equal(mol.coordinates(), mol2.coordinates())

        # Without labels, plain symbols are written
        writeMolecule(mol, out)
        assert_almost_equal(mol.coordinates(), readMolecule(out).coordinates())

        with self.assertRaises(ValueError):
            writeMolecule(mol, out, labels=["C1"])


if __name__ == "__main__":
    unittest.main()
