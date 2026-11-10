#!/usr/bin/env python3

import math
import os
import unittest

from koehnlab.molecular.StructureParser import parseStructureSpecString
from koehnlab.molecular import Atom, Element, Molecule, readLabeledMolecule
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

    def test_rings_are_closed_and_planar(self):
        bonds = "\n".join("C%d - C%d" % (i, i % 6 + 1) for i in range(1, 7))
        hydrogens = "\n".join("C%d - H%d" % (i, i) for i in range(1, 7))
        spec = _parse(bonds + "\n" + hydrogens)
        coords = defaultGuess(spec)
        index = {label: i for i, label in enumerate(spec.labels)}

        # All ring bonds have the default length, including the one that closes the ring
        ring = [index["C%d" % i] for i in range(1, 7)]
        for k in range(6):
            self.assertAlmostEqual(
                float(np.linalg.norm(coords[ring[k]] - coords[ring[(k + 1) % 6]])), 1.52, places=6
            )

        # The hydrogens point away from the ring and all atoms are in one plane
        center = coords[ring].mean(axis=0)
        for i in range(1, 7):
            carbon, hydrogen = coords[index["C%d" % i]], coords[index["H%d" % i]]
            self.assertGreater(np.linalg.norm(hydrogen - center), np.linalg.norm(carbon - center) + 0.9)
        self.assertAlmostEqual(float(np.linalg.svd(coords - coords.mean(axis=0))[1][-1]), 0, places=8)

    def test_ring_with_substituent_chain(self):
        # Atoms that are not part of the ring don't disturb it
        spec = _parse(
            "C1 - C2\nC2 - C3\nC3 - C4\nC4 - C5\nC5 - C6\nC6 - C1\nC1 - C7\nC7 - C8\nC3 - C9\nC9 - C10\n"
            "C2 - H1\nC4 - H2\nC5 - H3\nC6 - H4\n"
        )
        coords = defaultGuess(spec)
        index = {label: i for i, label in enumerate(spec.labels)}
        for i in range(1, 7):
            self.assertAlmostEqual(
                float(
                    np.linalg.norm(coords[index["C%d" % i]] - coords[index["C%d" % (i % 6 + 1)]])
                ),
                1.52,
                places=6,
            )

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

    def test_clash_terms(self):
        # Chain C1-C2-C3-C4-C5: C1/C3 are two bonds apart, C1/C4 three
        spec = _parse("C1 - C2 1.5\nC2 - C3 1.5\nC3 - C4 1.5\nC4 - C5 1.5")
        start = defaultGuess(spec)
        clashes = _Problem(spec, start).soft[3]

        pairs = {tuple(row) for row in clashes.indices.tolist()}
        self.assertEqual(pairs, {(0, 2), (1, 3), (2, 4), (0, 3), (1, 4), (0, 4)})

        # Fine in the default geometry, but atoms that are too close repel
        self.assertAlmostEqual(float(np.abs(clashes.residuals(start)).max()), 0.0)
        folded = start.copy()
        folded[2] = folded[0] + 0.1
        self.assertGreater(np.abs(clashes.residuals(folded)).max(), 5)

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


BENZENE = """
@planar
C1 - C2 1.40
C2 - C3 1.40
C3 - C4 1.40
C4 - C5 1.40
C5 - C6 1.40
C6 - C1 1.40
C1 - H1 1.08
C2 - H2 1.08
C3 - H3 1.08
C4 - H4 1.08
C5 - H5 1.08
C6 - H6 1.08
C6 - C1 - C2 120
C1 - C2 - C3 120
C2 - C3 - C4 120
C3 - C4 - C5 120
C4 - C5 - C6 120
C5 - C6 - C1 120
"""


class TestFlatMolecules(unittest.TestCase):

    def test_planar_ring(self):
        result = buildStructure(_parse(BENZENE))

        self.assertTrue(result.converged)
        coords = result.molecule.coordinates()
        assert_almost_equal(coords[:, 2], np.zeros(12), decimal=12)

        # The carbons form a regular hexagon (the hydrogens are not constrained to be symmetric)
        carbons = coords[[result.labels.index("C%d" % i) for i in range(1, 7)]]
        carbons = carbons - carbons.mean(axis=0)
        assert_almost_equal(np.linalg.norm(carbons, axis=1), np.full(6, 1.40), decimal=6)

        for i in range(1, 7):
            carbon = coords[result.labels.index("C%d" % i)]
            hydrogen = coords[result.labels.index("H%d" % i)]
            self.assertAlmostEqual(float(np.linalg.norm(hydrogen - carbon)), 1.08, places=6)

    def test_planar_polyene_with_dihedrals(self):
        text = "@planar\n" + BUTADIENE + "\nC2 - C3 - C4 - H5 0\nH1 - C1 - C2 - C3 180"
        result = buildStructure(_parse(text))

        self.assertTrue(result.converged, result.format())
        coords = result.molecule.coordinates()
        assert_almost_equal(coords[:, 2], np.zeros(len(coords)), decimal=12)

    def test_planar_geometry_argument(self):
        spec = _parse("C1 - C2 1.4\nC2 - C3 1.4\nC3 - C4 1.4\nC1 - C2 - C3 120\nC2 - C3 - C4 120")
        coords = buildStructure(spec, geometry="planar").molecule.coordinates()
        assert_almost_equal(coords[:, 2], np.zeros(4), decimal=12)

        # Without it, the default guess is not planar (torsion of the substituents is general)
        general = buildStructure(spec).molecule.coordinates()
        self.assertEqual(general.shape, (4, 3))

    def test_planar_guess_is_projected(self):
        biphenyl, labels = readLabeledMolecule(os.path.join(data_dir, "biphenyl_aligned.xyz"))
        spec = _parse("@planar\nC1 - C2 1.40")

        with self.assertWarns(UserWarning):
            result = buildStructure(spec, biphenyl, labels)

        coords = result.molecule.coordinates()
        assert_almost_equal(coords[:, 2], np.zeros(22), decimal=10)

        # A flat guess does not trigger a warning
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            buildStructure(spec, result.molecule, labels)

    def test_planar_rejects_general_dihedrals(self):
        with self.assertRaises(StructureSpecError):
            buildStructure(_parse("@planar\nC1 - C2 - C3 - C4 90"))

    def test_linear(self):
        result = buildStructure(
            _parse("@linear\nO1 - C1 1.16\nC1 - O2 1.17\nO1 - C1 - O2 180")
        )

        self.assertTrue(result.converged)
        coords = result.molecule.coordinates()
        assert_almost_equal(coords[:, :2], np.zeros((3, 2)), decimal=12)
        assert_almost_equal(
            np.abs(coords[result.labels.index("O1"), 2] - coords[result.labels.index("O2"), 2]), 2.33
        )

    def test_linear_without_angles(self):
        result = buildStructure(_parse("@linear\nH1 - C1 1.06\nC1 - C2 1.20\nC2 - H2 1.06"))

        self.assertTrue(result.converged)
        coords = result.molecule.coordinates()
        assert_almost_equal(coords[:, :2], np.zeros((4, 2)), decimal=12)
        self.assertAlmostEqual(coords[:, 2].max() - coords[:, 2].min(), 3.32)

    def test_linear_rejects_impossible_input(self):
        invalid = {
            "bent": "@linear\nC1 - C2 1.2\nC2 - C3 1.2\nC1 - C2 - C3 120",
            "dihedral": "@linear\nC1 - C2 1.2\nC2 - C3 1.2\nC3 - C4 1.2\nC1 - C2 - C3 - C4 0",
            "branched": "@linear\nC1 - C2 1.2\nC1 - C3 1.2\nC1 - C4 1.2",
            "ring": "@linear\nC1 - C2 1.2\nC2 - C3 1.2\nC3 - C1 1.2",
        }
        for name, text in invalid.items():
            with self.subTest(name):
                with self.assertRaises(StructureSpecError):
                    buildStructure(_parse(text))

    def test_unknown_geometry(self):
        with self.assertRaises(ValueError):
            buildStructure(_parse("C1 - C2 1.4"), geometry="spherical")


WATER_C2V = """
O1 - H1 0.96
O1 - H2
H1 - O1 - H2 104.5
@symmetry C2(z): H1<->H2
@symmetry sigma(y):
"""

AMMONIA_C3V = """
N1 - H1 1.01
N1 - H2
N1 - H3
H1 - N1 - H2 106.7
@symmetry C3(z): H1->H2->H3
@symmetry sigma(y): H2<->H3
"""

# Planar trans-butadiene (C2h), only half of the parameters are given
BUTADIENE_C2H = """
@planar
C1 - C2 1.34
C2 - C3 1.46
C3 - C4
C1 - C2 - C3 123.5
C1 - C2 - C3 - C4 180
C1 - H1
C1 - H2
C2 - H3
C3 - H4
C4 - H5
C4 - H6
@symmetry C2(z): C1<->C4, C2<->C3, H1<->H5, H2<->H6, H3<->H4
"""


class TestSymmetry(unittest.TestCase):

    @staticmethod
    def positions(result):
        return {label: result.molecule.coordinates()[i] for i, label in enumerate(result.labels)}

    def test_water(self):
        result = buildStructure(_parse(WATER_C2V))
        xyz = self.positions(result)

        self.assertTrue(result.converged)
        # The parameter given for one bond applies to the equivalent one as well
        for label in ("H1", "H2"):
            self.assertAlmostEqual(float(np.linalg.norm(xyz[label] - xyz["O1"])), 0.96, places=6)
        # C2v: the hydrogens are equivalent, so the geometric center lies on the symmetry axes
        assert_almost_equal(np.linalg.norm(xyz["H1"] - xyz["O1"]), np.linalg.norm(xyz["H2"] - xyz["O1"]))

    def test_ammonia(self):
        result = buildStructure(_parse(AMMONIA_C3V))
        xyz = self.positions(result)

        self.assertTrue(result.converged)
        for h in ("H1", "H2", "H3"):
            self.assertAlmostEqual(float(np.linalg.norm(xyz[h] - xyz["N1"])), 1.01, places=6)

        # The angle was only specified for one pair of hydrogens
        for a, b in (("H1", "H2"), ("H2", "H3"), ("H1", "H3")):
            u, v = xyz[a] - xyz["N1"], xyz[b] - xyz["N1"]
            angle = math.degrees(math.acos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
            self.assertAlmostEqual(angle, 106.7, places=4)

        # N is on the unique axis, which is the z axis in the standard orientation of a symmetric top
        assert_almost_equal(xyz["N1"][:2], np.zeros(2), decimal=8)

    def test_butadiene_ties_parameters(self):
        result = buildStructure(_parse(BUTADIENE_C2H))
        xyz = self.positions(result)
        distance = lambda a, b: float(np.linalg.norm(xyz[a] - xyz[b]))

        self.assertTrue(result.converged, result.format())
        self.assertAlmostEqual(distance("C3", "C4"), 1.34, places=6)
        self.assertAlmostEqual(distance("C1", "C2"), 1.34, places=6)
        assert_almost_equal(
            bondAngles(result.molecule.coordinates(), [(result.labels.index("C2"), result.labels.index("C3"), result.labels.index("C4"))]),
            [math.radians(123.5)],
        )

        # Inversion centre (C2 axis perpendicular to the molecular plane): partners are mirrored at the centre
        center = result.molecule.coordinates().mean(axis=0)
        for a, b in (("C1", "C4"), ("C2", "C3"), ("H1", "H5"), ("H2", "H6"), ("H3", "H4")):
            assert_almost_equal(xyz[a] + xyz[b], 2 * center, decimal=8)
        assert_almost_equal(result.molecule.coordinates()[:, 2], np.zeros(10), decimal=10)

    def test_conflicting_parameters_are_rejected(self):
        with self.assertRaisesRegex(StructureSpecError, "equivalent by symmetry"):
            buildStructure(_parse(BUTADIENE_C2H + "\nC3 - C4 1.35"))

    def test_dihedral_signs_under_improper_operations(self):
        chain = "C1 - C2 1.5\nC2 - C3 1.5\nC3 - C4 1.5\nC1 - C2 - C3 110\n@symmetry i: C1<->C4, C2<->C3\n"

        # Inversion turns a dihedral into its negative, so it can only be 0 or 180 degrees
        with self.assertRaisesRegex(StructureSpecError, "equivalent by symmetry"):
            buildStructure(_parse(chain + "C1 - C2 - C3 - C4 60"))

        result = buildStructure(_parse(chain + "C1 - C2 - C3 - C4 180"))
        self.assertTrue(result.converged)

        # A rotation keeps the sign
        rotation = chain.replace("@symmetry i:", "@symmetry C2(z):")
        self.assertTrue(buildStructure(_parse(rotation + "C1 - C2 - C3 - C4 60")).converged)

    def test_invalid_operations(self):
        invalid = {
            "bonds not preserved": "C1 - C2 1.5\nC1 - H1 1.1\n@symmetry C2(z): C1<->C2",
            "elements exchanged": "C1 - H1 1.1\n@symmetry C2(z): C1<->H1",
            "atoms exchanged without moving": "C1 - C2 1.5\n@symmetry C1(z): C1<->C2",
            "infinite group": "C1 - C2 1.4\n@symmetry C2(1,0,0):\n@symmetry C2(0.5403023,0.841471,0):",
        }
        for name, text in invalid.items():
            with self.subTest(name):
                with self.assertRaises(StructureSpecError):
                    buildStructure(_parse(text))

    def test_orientation_is_mandatory(self):
        with self.assertRaises(ValueError):
            buildStructure(_parse(WATER_C2V), orient=False)

    def test_starting_geometry(self):
        guess = Molecule(
            [
                Atom(Element.O, [0.01, 0.0, 0.0]),
                Atom(Element.H, [0.77, 0.0, -0.52]),
                Atom(Element.H, [-0.74, 0.0, -0.50]),
            ]
        )

        result = buildStructure(_parse(WATER_C2V), guess, ["O1", "H1", "H2"])
        xyz = self.positions(result)

        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(np.linalg.norm(xyz["H2"] - xyz["O1"])), 0.96, places=6)

        # A geometry that is far from the symmetry triggers a warning
        asymmetric = Molecule(
            [
                Atom(Element.O, [0.0, 0.0, 0.0]),
                Atom(Element.H, [0.9, 0.0, 0.0]),
                Atom(Element.H, [0.0, 0.0, 0.5]),
            ]
        )
        with self.assertWarns(UserWarning):
            self.assertTrue(buildStructure(_parse(WATER_C2V), asymmetric, ["O1", "H1", "H2"]).converged)


WATER_PARTIAL = """
O1 - H1 0.96
O1 - H2
H1 - O1 - H2 104.5
@group C2v
"""

# Staggered ethane in the canonical frame of D3d: C3 axis along z and C2 axes along x. Atoms: C1, C2, H1-H3 (on C1), H4-H6 (on C2)
def _ethane() -> Molecule:
    def hydrogen(angle, z):
        return Atom(Element.H, [1.02 * math.cos(math.radians(angle)), 1.02 * math.sin(math.radians(angle)), z])

    return Molecule(
        [Atom(Element.C, [0, 0, 0.765]), Atom(Element.C, [0, 0, -0.765])]
        + [hydrogen(a, 1.16) for a in (30, 150, 270)]
        + [hydrogen(a, -1.16) for a in (90, 210, 330)]
    )


class TestSymmetryInference(unittest.TestCase):

    def test_group_with_anchors(self):
        # sigma(y) maps the molecule in the xz plane onto itself: the hydrogens stay where they are
        text = WATER_PARTIAL + "@symmetry C2(z): H1<->H2\n@symmetry sigma(y): H1, ...\n"
        result = buildStructure(_parse(text))

        self.assertTrue(result.converged)
        coords = result.molecule.coordinates()
        o, h1, h2 = (coords[result.labels.index(label)] for label in ("O1", "H1", "H2"))
        self.assertAlmostEqual(float(np.linalg.norm(h2 - o)), 0.96, places=6)

    def test_ambiguity_is_reported(self):
        # Without further information, the mirror plane could exchange the hydrogens or leave them in place
        with self.assertRaisesRegex(StructureSpecError, "ambiguous"):
            buildStructure(_parse(WATER_PARTIAL + "@symmetry C2(z): H1<->H2\n"))

        with self.assertRaisesRegex(StructureSpecError, "ambiguous"):
            buildStructure(_parse(WATER_PARTIAL))

    def test_group_with_starting_geometry(self):
        guess = Molecule(
            [
                Atom(Element.O, [0.0, 0.0, 0.0]),
                Atom(Element.H, [0.76, 0.0, -0.52]),
                Atom(Element.H, [-0.76, 0.0, -0.52]),
            ]
        )

        result = buildStructure(_parse(WATER_PARTIAL), guess, ["O1", "H1", "H2"])

        self.assertTrue(result.converged)
        coords = result.molecule.coordinates()
        self.assertAlmostEqual(float(np.linalg.norm(coords[2] - coords[0])), 0.96, places=6)

    def test_partial_list_with_ellipsis(self):
        # The cycle H1->H2 is completed to H1->H2->H3, a swap would not be compatible with a rotation by 120 degrees
        text = AMMONIA_C3V.replace("H1->H2->H3", "H1->H2, ...")
        result = buildStructure(_parse(text))
        self.assertTrue(result.converged)

        # An exchange of two hydrogens is compatible with a mirror plane but a fixed H1 does not tell what happens to the others
        ambiguous = AMMONIA_C3V.replace("sigma(y): H2<->H3", "sigma(y): H1, ...")
        with self.assertRaisesRegex(StructureSpecError, "ambiguous"):
            buildStructure(_parse(ambiguous))

    def test_inconsistent_anchors(self):
        with self.assertRaisesRegex(StructureSpecError, "different element"):
            buildStructure(_parse(AMMONIA_C3V.replace("H1->H2->H3", "H1->N1, ...")))

        # A rotation by 120 degrees cannot exchange two atoms and leave the rest alone
        with self.assertRaisesRegex(StructureSpecError, "No atom mapping"):
            buildStructure(_parse(AMMONIA_C3V.replace("H1->H2->H3", "H1<->H2, ...")))

    def test_ethane_d3d(self):
        guess = _ethane()
        spec = _parse("C1 - C2 1.53\nC1 - H1 1.09\nH1 - C1 - C2 111\n@group D3d")

        result = buildStructure(spec, guess)
        self.assertTrue(result.converged, result.format())

        coords = result.molecule.coordinates()
        index = {label: i for i, label in enumerate(result.labels)}
        distance = lambda a, b: float(np.linalg.norm(coords[index[a]] - coords[index[b]]))

        # Everything follows from the three given parameters
        for carbon, hydrogens in (("C1", ("H1", "H2", "H3")), ("C2", ("H4", "H5", "H6"))):
            for h in hydrogens:
                self.assertAlmostEqual(distance(carbon, h), 1.09, places=6)
        self.assertAlmostEqual(distance("H1", "H2"), distance("H2", "H3"), places=6)
        self.assertAlmostEqual(distance("H1", "H2"), distance("H5", "H6"), places=6)
        # staggered: all H-C-C-H dihedrals are 60, 180 or -60 degrees
        for dihedral in dihedralAngles(coords, [(index["H1"], 0, 1, index["H4"]), (index["H1"], 0, 1, index["H5"])]):
            self.assertAlmostEqual(_distanceToMultiple(math.degrees(float(dihedral)), 60), 0, places=4)

        # Without a geometry the mapping of the hydrogens is not determined by the graph
        with self.assertRaisesRegex(StructureSpecError, "ambiguous"):
            buildStructure(_parse("C1 - C2 1.53\nC1 - H1 1.09\nC1 - H2\nC1 - H3\nC2 - H4\nC2 - H5\nC2 - H6\n@group D3d"))


# Benzyne (planar, C2v): C1 and C2 are the dehydro carbons. The molecular plane is the xy plane, the C2 axis (through the middle of
# C1-C2 and C4-C5) is the y axis and the mirror plane perpendicular to the molecule is the yz plane
BENZYNE = """
C1 - C2 1.27067
C2 - C3 1.39887
C3 - C4 1.41259
C4 - C5 1.41751
C5 - C6
C6 - C1
C3 - H1 1.09239
C4 - H2 1.09507
C5 - H3
C6 - H4
C1 - C2 - C3 126.41
C2 - C3 - H1 126.37
C3 - C4 - H2 118.85
@planar
@symmetry C2(y): C1<->C2, C3<->C6, ...
@symmetry sigma(x): C1<->C2, C3<->C6, ...
"""


class TestRings(unittest.TestCase):

    def test_benzyne(self):
        result = buildStructure(_parse(BENZYNE))
        coords = result.molecule.coordinates()
        index = {label: i for i, label in enumerate(result.labels)}

        self.assertTrue(result.converged, result.format())
        assert_almost_equal(coords[:, 2], np.zeros(len(coords)), decimal=10)

        # No folded ring: The closest atoms are bonded atoms, and the ring angles add up to a hexagon. The angles that were not
        # given follow from the other parameters (sum of 720 degrees and the symmetry)
        distances = np.linalg.norm(coords[:, None] - coords[None], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreater(distances.min(), 1.05)

        ring = [index["C%d" % i] for i in range(1, 7)]
        angles = bondAngles(coords, [(ring[k - 1], ring[k], ring[(k + 1) % 6]) for k in range(6)])
        assert_almost_equal(
            np.degrees(angles), [126.41, 126.41, 111.2, 122.4, 122.4, 111.2], decimal=0
        )
        self.assertAlmostEqual(float(np.degrees(angles).sum()), 720.0, places=6)

        # C2v symmetry: C3/C6 and C4/C5 are equivalent
        for a, b in (("C3", "C6"), ("C4", "C5"), ("H1", "H4"), ("H2", "H3")):
            assert_almost_equal(coords[index[a]] * [-1, 1, 1], coords[index[b]], decimal=6)


if __name__ == "__main__":
    unittest.main()
