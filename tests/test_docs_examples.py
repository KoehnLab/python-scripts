#!/usr/bin/env python3

"""Builds the example files that are used in docs/structure_builder.md"""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

from koehnlab.molecular import readLabeledMolecule

import numpy as np
from numpy.testing import assert_almost_equal

script_dir = os.path.dirname(os.path.realpath(__file__))
examples_dir = os.path.join(script_dir, "..", "docs", "examples")

_spec = importlib.util.spec_from_file_location(
    "build_structure", os.path.join(script_dir, "..", "scripts", "build_structure.py")
)
assert _spec is not None and _spec.loader is not None
build_structure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_structure)


class TestDocsExamples(unittest.TestCase):

    def build(self, name: str, *extra: str):
        """Builds docs/examples/<name>.txt and returns the molecule and labels"""
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "out.xyz")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = build_structure.main(
                    ["-i", os.path.join(examples_dir, name + ".txt"), "-o", output, "--labels", *extra]
                )
            self.assertEqual(code, 0, err.getvalue())
            return readLabeledMolecule(output)

    @staticmethod
    def distance(molecule, labels, a, b):
        return float(
            np.linalg.norm(molecule.coordinates()[labels.index(a)] - molecule.coordinates()[labels.index(b)])
        )

    def test_water(self):
        molecule, labels = self.build("water")
        self.assertAlmostEqual(self.distance(molecule, labels, "O1", "H2"), 0.96, places=6)

    def test_butadiene(self):
        molecule, labels = self.build("butadiene")
        self.assertAlmostEqual(self.distance(molecule, labels, "C3", "C4"), 1.34, places=6)
        assert_almost_equal(molecule.coordinates()[:, 2], np.zeros(molecule.nAtoms()), decimal=10)
        # trans: the terminal carbons are as far apart as possible
        self.assertGreater(self.distance(molecule, labels, "C1", "C4"), 3.5)

    def test_co2(self):
        molecule, labels = self.build("co2")
        assert_almost_equal(molecule.coordinates()[:, :2], np.zeros((3, 2)), decimal=10)
        self.assertAlmostEqual(self.distance(molecule, labels, "O1", "O2"), 2.32, places=6)

    def test_ammonia(self):
        molecule, labels = self.build("ammonia")
        for hydrogen in ("H1", "H2", "H3"):
            self.assertAlmostEqual(self.distance(molecule, labels, "N1", hydrogen), 1.01, places=6)
        self.assertAlmostEqual(
            self.distance(molecule, labels, "H1", "H2"), self.distance(molecule, labels, "H2", "H3"), places=6
        )

    def test_benzyne(self):
        molecule, labels = self.build("benzyne")
        self.assertAlmostEqual(self.distance(molecule, labels, "C4", "C5"), 1.41751, places=6)
        # tied by symmetry: C5-C6 is equivalent to C3-C4 and C6-C1 to C2-C3
        self.assertAlmostEqual(self.distance(molecule, labels, "C5", "C6"), 1.41259, places=6)
        self.assertAlmostEqual(self.distance(molecule, labels, "C6", "C1"), 1.39887, places=6)
        self.assertAlmostEqual(self.distance(molecule, labels, "C5", "H3"), 1.09507, places=6)

    def test_ethane(self):
        molecule, labels = self.build("ethane_d3d", "--guess", os.path.join(examples_dir, "ethane_guess.xyz"))
        for hydrogen in ("H1", "H2", "H3"):
            self.assertAlmostEqual(self.distance(molecule, labels, "C1", hydrogen), 1.09, places=6)
        for hydrogen in ("H4", "H5", "H6"):
            self.assertAlmostEqual(self.distance(molecule, labels, "C2", hydrogen), 1.09, places=6)
        self.assertAlmostEqual(self.distance(molecule, labels, "C1", "C2"), 1.53, places=6)

    def test_methane(self):
        molecule, labels = self.build("methane_td")
        for hydrogen in ("H1", "H2", "H3", "H4"):
            self.assertAlmostEqual(self.distance(molecule, labels, "C1", hydrogen), 1.087, places=6)
        self.assertAlmostEqual(
            self.distance(molecule, labels, "H1", "H2"), self.distance(molecule, labels, "H3", "H4"), places=6
        )

    def test_sf6(self):
        molecule, labels = self.build("sf6_oh")
        self.assertAlmostEqual(self.distance(molecule, labels, "F1", "F2"), 2 * 1.564, places=6)
        self.assertAlmostEqual(self.distance(molecule, labels, "F1", "F3"), 1.564 * 2**0.5, places=6)


if __name__ == "__main__":
    unittest.main()

