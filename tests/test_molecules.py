#!/usr/bin/env python3

import unittest
import os

from koehnlab.molecular import Atom, Element, Molecule, readMolecule, writeMolecule, Eckart_alignment, check_Eckart

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.spatial.transform import Rotation

script_dir: str = os.path.dirname(os.path.realpath(__file__))
data_dir: str = os.path.join(script_dir, "molecules")


def _makeMolecule(elements, positions) -> Molecule:
    return Molecule([Atom(Element[el], pos) for el, pos in zip(elements, positions)])


class TestMolecules(unittest.TestCase):

    def test_align(self):

        mol_A = readMolecule(os.path.join(data_dir, "biphenyl_unaligned.xyz"))

        mol_A.bringToStandardOrientation(center="mass")

        # Note: biphenyl_aligned.xyz is not used here as it is the mirror image of the correct orientation
        # (it was generated with an improper rotation). It only serves as input for the Eckart test.
        mol_B = readMolecule(os.path.join(data_dir, "biphenyl_aligned_mass.xyz"))

        assert_almost_equal(mol_A.coordinates(),mol_B.coordinates())

    def test_align_geometric(self):

        mol_A = readMolecule(os.path.join(data_dir, "biphenyl_unaligned.xyz"))

        mol_A.bringToStandardOrientation()

        assert_almost_equal(mol_A.geometricCenter(), np.zeros(3))

        mol_B = readMolecule(os.path.join(data_dir, "biphenyl_aligned_geom.xyz"))

        assert_almost_equal(mol_A.coordinates(), mol_B.coordinates())

        # The inertia tensor must be diagonal with ascending moments
        tensor = mol_A.inertiaTensor()
        assert_almost_equal(tensor, np.diag(np.diag(tensor)))
        self.assertTrue(np.all(np.diff(np.diag(tensor)) >= 0))

    def test_inertia_tensor_modes(self):

        mol = readMolecule(os.path.join(data_dir, "biphenyl_unaligned.xyz"))

        # The geometric tensor must not depend on the masses (here: replace all H by the heavier He)
        tensor = mol.inertiaTensor()
        for atom in mol.atoms:
            if atom.element == Element.H:
                atom.element = Element.He
        assert_almost_equal(mol.inertiaTensor(), tensor)

        with self.assertRaises(ValueError):
            mol.inertiaTensor(center="invalid")

    def test_align_preserves_chirality(self):

        # CHFClBr: swapping two substituents gives the other enantiomer
        positions = [[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
        elements = ["C", "H", "F", "Cl", "Br"]

        def signedVolume(coord: np.ndarray) -> float:
            return float(np.linalg.det(coord[2:5] - coord[1]))

        for swap in (False, True):
            order = [0, 1, 3, 2, 4] if swap else [0, 1, 2, 3, 4]
            mol = _makeMolecule(elements, [positions[i] for i in order])
            before = signedVolume(mol.coordinates())

            for center in ("geometric", "mass"):
                aligned = _makeMolecule(elements, mol.coordinates())
                aligned.bringToStandardOrientation(center=center)
                self.assertGreater(before * signedVolume(aligned.coordinates()), 0)

    def test_align_is_deterministic(self):

        # For all kinds of rotors, the result must not depend on the initial orientation, and aligning twice must not change
        # anything. The latter two are the interesting cases for degenerate moments of inertia.
        angles = np.arange(6) * np.pi / 3
        ring = [[1.4 * np.cos(t), 1.4 * np.sin(t), 0] for t in angles]
        ringH = [[2.5 * np.cos(t), 2.5 * np.sin(t), 0] for t in angles]
        triangle = [[0.94 * np.cos(t), 0.94 * np.sin(t), -0.3] for t in (0, 2 * np.pi / 3, 4 * np.pi / 3)]

        systems = {
            "asymmetric": (["O", "H", "H"], [[0, 0, 0.1], [0.75, 0, -0.5], [-0.75, 0, -0.5]]),
            "planar symmetric": (["C"] * 6 + ["H"] * 6, ring + ringH),
            "pyramidal symmetric": (["N", "H", "H", "H"], [[0, 0, 0.1]] + triangle),
            "spherical": (
                ["C", "H", "H", "H", "H"],
                [[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
            ),
            "linear": (["O", "C", "O"], [[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]]),
            "diatomic": (["H", "Cl"], [[0, 0, 0], [0, 0, 1.27]]),
            "single atom": (["C"], [[1, 2, 3]]),
        }

        rng = np.random.default_rng(42)

        for name, (elements, positions) in systems.items():
            with self.subTest(system=name):
                reference = _makeMolecule(elements, positions)
                reference.bringToStandardOrientation()
                refCoords = reference.coordinates()

                for _ in range(10):
                    mol = _makeMolecule(elements, positions)
                    mol.transform(Rotation.random(random_state=rng.integers(2**31)).as_matrix())
                    mol.translate(rng.normal(size=3))
                    mol.bringToStandardOrientation()

                    assert_almost_equal(mol.coordinates(), refCoords)

                reference.bringToStandardOrientation()
                assert_almost_equal(reference.coordinates(), refCoords)

    def test_align_symmetric_conventions(self):

        # Symmetric tops: z is the unique axis and the first atom off that axis lies in the xz plane (x > 0)
        angles = np.arange(6) * np.pi / 3
        benzene = _makeMolecule(
            ["C"] * 6, [[1.4 * np.cos(t), 1.4 * np.sin(t), 0] for t in angles]
        )
        benzene.bringToStandardOrientation()
        coords = benzene.coordinates()
        assert_almost_equal(coords[:, 2], np.zeros(6))
        assert_almost_equal(coords[0], [1.4, 0, 0])

        # Linear molecules: along z, first off-center atom at positive z
        linear = _makeMolecule(["O", "C", "O"], [[0, 1.16, 0], [0, 0, 0], [0, -1.16, 0]])
        linear.bringToStandardOrientation()
        assert_almost_equal(linear.coordinates(), [[0, 0, 1.16], [0, 0, 0], [0, 0, -1.16]])

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
