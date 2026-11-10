#!/usr/bin/env python3

import math
import os
import unittest

from koehnlab.molecular.StructureParser import parseStructureSpecString
from koehnlab.molecular import Atom, Molecule, readLabeledMolecule
from koehnlab.molecular.StructureParser import StructureSpecError
from koehnlab.molecular.StructureBuilder import (
    _Problem,
    buildStructure,
    defaultGuess,
    resolveSpec,
)
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


script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, "molecules")


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


def _fullSpecText(molecule, labels, bonds):
    """Spec containing all bonds, angles and dihedrals of the given molecule"""
    coords = molecule.coordinates()
    lines = []
    for i, j in bonds:
        lines.append("%s - %s %.10f" % (labels[i], labels[j], bondLengths(coords, [(i, j)])[0]))
    for a in findAngles(bonds, len(labels)):
        lines.append(
            "%s %.10f" % (" - ".join(labels[k] for k in a), math.degrees(bondAngles(coords, [a])[0]))
        )
    for d in findDihedrals(bonds, len(labels)):
        lines.append(
            "%s %.10f"
            % (" - ".join(labels[k] for k in d), math.degrees(dihedralAngles(coords, [d])[0]))
        )
    return "\n".join(lines)


def _alignedRmsd(a, b):
    """RMSD after optimal translation and proper rotation"""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rotation - b) ** 2, axis=1))))


class TestBuildStructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.biphenyl, cls.labels = readLabeledMolecule(
            os.path.join(data_dir, "biphenyl_aligned.xyz")
        )
        bonds = resolveSpec(_parse("C1 - C2"), cls.biphenyl, cls.labels).bonds
        cls.bonds = bonds
        cls.specText = _fullSpecText(cls.biphenyl, cls.labels, bonds)

    def _byLabel(self, result):
        order = [self.labels.index(label) for label in result.labels]
        return self.biphenyl.coordinates()[order]

    def test_jacobian(self):
        # Compare the blockwise derivatives to brute-force derivatives of the complete residual vector
        spec = _parse(BUTADIENE + "\nC1 - C4 3.0\nH1 - C1 - C2 119")
        start = defaultGuess(spec)
        problem = _Problem(spec, start)
        x = (start + np.random.default_rng(1).normal(scale=0.05, size=start.shape)).ravel()

        for restraints in (True, False):
            analytic = problem.jacobian(x, restraints)
            numeric = np.zeros_like(analytic)
            for k in range(x.size):
                step = np.zeros(x.size)
                step[k] = 1e-6
                numeric[:, k] = (
                    problem.residuals(x + step, restraints) - problem.residuals(x - step, restraints)
                ) / 2e-6

            self.assertEqual(analytic.shape, (len(problem.residuals(x, restraints)), x.size))
            assert_almost_equal(analytic, numeric, decimal=3)

    def test_water(self):
        result = buildStructure(_parse("O1 - H1 0.96\nO1 - H2 0.97\nH1 - O1 - H2 104.5"))

        self.assertTrue(result.converged)
        self.assertEqual(result.labels, ["O1", "H1", "H2"])
        assert_almost_equal(result.molecule.geometricCenter(), np.zeros(3))
        self.assertIn("H1 - O1 - H2", result.format())

    def test_roundtrip_with_perturbed_guess(self):
        rng = np.random.default_rng(0)
        noisy = self.biphenyl.coordinates() + rng.normal(scale=0.15, size=(22, 3))
        guess = Molecule([Atom(a.element, x) for a, x in zip(self.biphenyl.atoms, noisy)])

        result = buildStructure(_parse(self.specText), guess, self.labels)

        self.assertTrue(result.converged)
        self.assertEqual(result.labels, self.labels)
        self.assertLess(_alignedRmsd(result.molecule.coordinates(), self.biphenyl.coordinates()), 1e-5)

    def test_roundtrip_without_guess(self):
        result = buildStructure(_parse(self.specText))

        self.assertTrue(result.converged)
        self.assertLess(_alignedRmsd(result.molecule.coordinates(), self._byLabel(result)), 1e-5)

    def test_underspecified_stays_close_to_guess(self):
        # Twist the bond between the rings (the only C-C bond longer than 1.45 Angstrom); nothing else should change
        coords = self.biphenyl.coordinates()
        isCarbon = [atom.element.symbol() == "C" for atom in self.biphenyl.atoms]
        a, b = [
            bond
            for bond in self.bonds
            if all(isCarbon[i] for i in bond) and bondLengths(coords, [bond])[0] > 1.45
        ][0]

        def carbonNeighbor(atom, exclude):
            return [
                (set(bond) - {atom}).pop()
                for bond in self.bonds
                if atom in bond and isCarbon[(set(bond) - {atom}).pop()] and exclude not in bond
            ][0]

        dihedral = (carbonNeighbor(a, b), a, b, carbonNeighbor(b, a))
        original = math.degrees(float(dihedralAngles(coords, [dihedral])[0]))
        target = original + 20

        spec = _parse("%s %.8f" % (" - ".join(self.labels[i] for i in dihedral), target))
        result = buildStructure(spec, self.biphenyl, self.labels, orient=False)

        self.assertTrue(result.converged)
        self.assertEqual(result.labels, self.labels)

        newCoords = result.molecule.coordinates()
        self.assertAlmostEqual(
            float(wrapAngle(dihedralAngles(newCoords, [dihedral])[0] - math.radians(target))), 0, places=6
        )

        # All bonds and angles are unchanged ...
        assert_almost_equal(
            bondLengths(newCoords, self.bonds), bondLengths(coords, self.bonds), decimal=4
        )
        angles = findAngles(self.bonds, 22)
        assert_almost_equal(bondAngles(newCoords, angles), bondAngles(coords, angles), decimal=4)

        # ... and so are the dihedrals within each ring
        ringDihedrals = [
            d for d in findDihedrals(self.bonds, 22) if a not in d[1:3] or b not in d[1:3]
        ]
        assert_almost_equal(
            wrapAngle(dihedralAngles(newCoords, ringDihedrals) - dihedralAngles(coords, ringDihedrals)),
            np.zeros(len(ringDihedrals)),
            decimal=2,
        )

    def test_inconsistent_parameters_are_reported(self):
        # A triangle with sides 1, 1 and 5 Angstrom does not exist
        result = buildStructure(_parse("C1 - C2 1.0\nC2 - C3 1.0\nC1 - C3 5.0"), maxAttempts=2)

        self.assertFalse(result.converged)
        self.assertEqual(result.attempts, 2)
        self.assertIn("not satisfied", result.format())

    def test_reproducible(self):
        spec = _parse("C1 - C2 1.0\nC2 - C3 1.0\nC1 - C3 5.0")
        first = buildStructure(spec, maxAttempts=3, seed=7)
        second = buildStructure(spec, maxAttempts=3, seed=7)

        assert_almost_equal(first.molecule.coordinates(), second.molecule.coordinates())

    def test_resolve_errors(self):
        with self.assertRaises(StructureSpecError):
            buildStructure(_parse("C1 - C99 1.5"), self.biphenyl, self.labels)

        # Label N1 belongs to a carbon atom here
        with self.assertRaises(StructureSpecError):
            buildStructure(_parse("N1 - C2 1.5"), self.biphenyl, ["N1"] + self.labels[1:])

    def test_orient_option(self):
        spec = _parse("C1 - C2 1.5\nC2 - C3 1.5\nC1 - C2 - C3 109")

        oriented = buildStructure(spec).molecule
        raw = buildStructure(spec, orient=False).molecule

        assert_almost_equal(oriented.geometricCenter(), np.zeros(3))
        assert_almost_equal(raw.coordinates()[0], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
