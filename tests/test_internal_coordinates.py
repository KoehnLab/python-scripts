#!/usr/bin/env python3

import unittest

from koehnlab.molecular.InternalCoordinates import (
    bondLengths,
    bondAngles,
    dihedralAngles,
    wrapAngle,
    canonicalBond,
    canonicalAngle,
    canonicalDihedral,
    canonicalCoordinate,
    findAngles,
    findDihedrals,
)

import numpy as np
from numpy.testing import assert_almost_equal
from ase import Atoms
from scipy.spatial.transform import Rotation


class TestInternalCoordinates(unittest.TestCase):

    def test_against_ase(self):
        rng = np.random.default_rng(3)
        coords = rng.normal(size=(8, 3))
        atoms = Atoms("C8", positions=coords)

        bonds = [(0, 1), (3, 2), (5, 7)]
        angles = [(0, 1, 2), (4, 3, 5), (7, 0, 6)]
        dihedrals = [(0, 1, 2, 3), (4, 3, 5, 7), (7, 0, 6, 1)]

        assert_almost_equal(
            bondLengths(coords, bonds), [atoms.get_distance(i, j) for i, j in bonds]
        )
        assert_almost_equal(
            np.degrees(bondAngles(coords, angles)),
            [atoms.get_angle(*a) for a in angles],
        )
        # ASE returns values in [0, 360)
        assert_almost_equal(
            np.degrees(dihedralAngles(coords, dihedrals)) % 360,
            [atoms.get_dihedral(*d) for d in dihedrals],
        )

    def test_known_values(self):
        # Cis/trans/gauche arrangements, looking along the central bond
        cis = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
        trans = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0]], dtype=float)
        right = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=float)

        self.assertAlmostEqual(dihedralAngles(cis, [(0, 1, 2, 3)])[0], 0)
        self.assertAlmostEqual(abs(dihedralAngles(trans, [(0, 1, 2, 3)])[0]), np.pi)
        self.assertAlmostEqual(abs(dihedralAngles(right, [(0, 1, 2, 3)])[0]), np.pi / 2)

        # Mirror image changes the sign
        mirror = right * np.array([1, 1, -1])
        assert_almost_equal(
            dihedralAngles(right, [(0, 1, 2, 3)]), -dihedralAngles(mirror, [(0, 1, 2, 3)])
        )

        # Linear arrangement
        line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        self.assertAlmostEqual(bondAngles(line, [(0, 1, 2)])[0], np.pi)

    def test_invariances(self):
        rng = np.random.default_rng(5)
        coords = rng.normal(size=(4, 3))
        d = [(0, 1, 2, 3)]

        # Reversal of the atom order does not change the dihedral angle
        assert_almost_equal(
            dihedralAngles(coords, d), dihedralAngles(coords, [(3, 2, 1, 0)])
        )

        # Rotations and translations don't either
        rotated = coords @ Rotation.from_quat(np.random.default_rng(1).normal(size=4)).as_matrix().T + 3.0
        assert_almost_equal(dihedralAngles(coords, d), dihedralAngles(rotated, d))

    def test_wrap(self):
        assert_almost_equal(
            wrapAngle([0, np.pi, -np.pi, 3 * np.pi / 2, -3 * np.pi / 2, 2 * np.pi]),
            [0, np.pi, np.pi, -np.pi / 2, np.pi / 2, 0],
        )
        # 179 degrees and -179 degrees are only 2 degrees apart
        assert_almost_equal(
            np.degrees(wrapAngle(np.radians(179) - np.radians(-179))), -2
        )

    def test_canonical_forms(self):
        self.assertEqual(canonicalBond(3, 1), (1, 3))
        self.assertEqual(canonicalAngle(5, 2, 1), (1, 2, 5))
        self.assertEqual(canonicalDihedral(4, 3, 2, 1), (1, 2, 3, 4))
        self.assertEqual(canonicalDihedral(1, 2, 3, 4), (1, 2, 3, 4))

        # The coordinate type follows from the number of atoms
        self.assertEqual(canonicalCoordinate((3, 1)), (1, 3))
        self.assertEqual(canonicalCoordinate([5, 2, 1]), (1, 2, 5))
        self.assertEqual(canonicalCoordinate((4, 3, 2, 1)), (1, 2, 3, 4))
        with self.assertRaises(ValueError):
            canonicalCoordinate((1,))

    def test_enumeration(self):
        # Butane-like chain 0-1-2-3 with a branch 4 on atom 1 and a three-membered ring 5-6-7
        bonds = [(0, 1), (1, 2), (2, 3), (1, 4), (5, 6), (6, 7), (7, 5)]

        angles = findAngles(bonds, 8)
        self.assertEqual(
            sorted(angles),
            [(0, 1, 2), (0, 1, 4), (1, 2, 3), (2, 1, 4), (5, 6, 7), (5, 7, 6), (6, 5, 7)],
        )

        dihedrals = findDihedrals(bonds, 8)
        self.assertEqual(sorted(dihedrals), [(0, 1, 2, 3), (3, 2, 1, 4)])


if __name__ == "__main__":
    unittest.main()
