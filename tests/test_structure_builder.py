#!/usr/bin/env python3

import math
import unittest

from koehnlab.molecular.StructureParser import parseStructureSpecString
from koehnlab.molecular.StructureBuilder import defaultGuess
from koehnlab.molecular.InternalCoordinates import (
    bondLengths,
    bondAngles,
    dihedralAngles,
    wrapAngle,
    findAngles,
    findDihedrals,
)

import numpy as np
from numpy.testing import assert_almost_equal

ETHANE = """
C1 - C2
C1 - H1
C1 - H2
C1 - H3
C2 - H4
C2 - H5
C2 - H6
"""

BUTADIENE = """
C1 - C2 1.34
C2 - C3 1.46
C3 - C4 1.34
C1 - C2 - C3 123.5
C2 - C3 - C4 123.5
C1 - C2 - C3 - C4 180
C1 - H1
C1 - H2
C2 - H3
C3 - H4
C4 - H5
C4 - H6
"""


def _distanceToMultiple(value: float, period: float) -> float:
    remainder = value % period
    return min(remainder, period - remainder)


def _parse(text):
    return parseStructureSpecString(text)


class TestDefaultGuess(unittest.TestCase):

    def test_sign_convention_of_specified_dihedrals(self):
        # Specified dihedrals that match a tree entry have to be reproduced including their sign
        for angle in (-150, -60, -20, 0, 35, 90, 120, 180):
            spec = _parse(
                "C1 - C2 1.5\nC2 - C3 1.5\nC3 - C4 1.5\nC1 - C2 - C3 110\nC2 - C3 - C4 112\n"
                "C1 - C2 - C3 - C4 %d" % angle
            )
            coords = defaultGuess(spec)
            assert_almost_equal(
                float(dihedralAngles(coords, [(0, 1, 2, 3)])[0]), math.radians(angle)
            )

    def test_water(self):
        spec = _parse("O1 - H1 0.96\nO1 - H2 0.97\nH1 - O1 - H2 104.5")
        coords = defaultGuess(spec)

        assert_almost_equal(bondLengths(coords, [(0, 1), (0, 2)]), [0.96, 0.97])
        assert_almost_equal(
            bondAngles(coords, [(1, 0, 2)]), [math.radians(104.5)]
        )

    def test_ethane_defaults(self):
        spec = _parse(ETHANE)
        coords = defaultGuess(spec)

        # sum of covalent radii (Cordero): C 0.76, H 0.31
        assert_almost_equal(bondLengths(coords, [(0, 1)]), [1.52], decimal=2)
        assert_almost_equal(bondLengths(coords, [(0, 2), (1, 5)]), [1.07, 1.07], decimal=2)

        tetrahedral = math.radians(109.4712206)
        angles = findAngles(spec.bonds, spec.nAtoms())
        assert_almost_equal(bondAngles(coords, angles), np.full(len(angles), tetrahedral), decimal=5)

        # staggered
        for dihedral in dihedralAngles(coords, findDihedrals(spec.bonds, spec.nAtoms())):
            self.assertAlmostEqual(_distanceToMultiple(math.degrees(dihedral), 60), 0, places=4)

    def test_butadiene_specified_values(self):
        spec = _parse(BUTADIENE)
        coords = defaultGuess(spec)

        assert_almost_equal(bondLengths(coords, [(0, 1), (1, 2), (2, 3)]), [1.34, 1.46, 1.34])
        assert_almost_equal(
            bondAngles(coords, [(0, 1, 2), (1, 2, 3)]), np.full(2, math.radians(123.5))
        )
        self.assertAlmostEqual(
            abs(math.degrees(float(dihedralAngles(coords, [(0, 1, 2, 3)])[0]))), 180
        )

        # Unspecified sp2 centers are planar by default
        for dihedral in dihedralAngles(coords, findDihedrals(spec.bonds, spec.nAtoms())):
            self.assertAlmostEqual(_distanceToMultiple(math.degrees(dihedral), 180), 0, places=3)

    def test_no_overlapping_atoms(self):
        coords = defaultGuess(_parse(ETHANE))
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreater(distances.min(), 0.9)

    def test_many_neighbors_and_linear_fragment(self):
        # Octahedral-ish center and a linear fragment with a following torsion must not produce NaNs
        spec = _parse(
            "S1 - F1\nS1 - F2\nS1 - F3\nS1 - F4\nS1 - F5\nS1 - F6\n"
            "C1 - C2 - C3 180\nC1 - C2\nC2 - C3\nC3 - C4\nC2 - C3 - C4 120\nC1 - C2 - C3 - C4 90"
        )
        # not connected: S fragment vs C fragment
        with self.assertRaises(ValueError):
            defaultGuess(spec)

        s = _parse("S1 - F1\nS1 - F2\nS1 - F3\nS1 - F4\nS1 - F5\nS1 - F6")
        self.assertTrue(np.all(np.isfinite(defaultGuess(s))))

        c = _parse(
            "C1 - C2\nC2 - C3\nC3 - C4\nC1 - C2 - C3 180\nC2 - C3 - C4 120\nC1 - C2 - C3 - C4 90"
        )
        coords = defaultGuess(c)
        self.assertTrue(np.all(np.isfinite(coords)))
        assert_almost_equal(bondAngles(coords, [(0, 1, 2)]), [math.pi])

    def test_ring_is_placed(self):
        coords = defaultGuess(_parse("C1 - C2\nC2 - C3\nC3 - C1"))
        self.assertTrue(np.all(np.isfinite(coords)))


if __name__ == "__main__":
    unittest.main()
