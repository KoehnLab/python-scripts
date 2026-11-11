#!/usr/bin/env python3

import math
import numpy as np
import os
import tempfile
import unittest

from koehnlab.molecular import Element
from koehnlab.molecular.StructureParser import (
    parseSymmetryOperation,
    Constraint,
    StructureSpecError,
    parseStructureSpec,
    parseStructureSpecString,
)

from numpy.testing import assert_almost_equal


class TestStructureParser(unittest.TestCase):

    def test_valid(self):
        spec = parseStructureSpecString(
            """
            # Butadiene fragment
            C1 - C2 1.34     # a double bond
            C2-C3 1.46
            C1 - H1
            C1 - C2 - C3 123.5
            C1 - C2 - C3 - C4 180   # dihedral
            C3 - C4
            """
        )

        self.assertEqual(spec.labels, ["C1", "C2", "C3", "H1", "C4"])
        self.assertEqual(spec.elements, [Element.C] * 3 + [Element.H, Element.C])
        self.assertEqual(spec.bonds, [(0, 1), (0, 3), (1, 2), (2, 4)])
        self.assertEqual(spec.indexOf("H1"), 3)

        self.assertEqual(
            [(c.kind(), c.indices) for c in spec.constraints],
            [
                ("bond", (0, 1)),
                ("bond", (1, 2)),
                ("angle", (0, 1, 2)),
                ("dihedral", (0, 1, 2, 4)),
            ],
        )
        self.assertAlmostEqual(spec.constraints[0].value, 1.34)
        self.assertAlmostEqual(spec.constraints[2].value, math.radians(123.5))
        self.assertAlmostEqual(spec.constraints[3].value, math.pi)

    def test_weights_and_wrapping(self):
        spec = parseStructureSpecString("C1 - C2 - C3 - C4 -270 w=2.5\nC1 - C2 1.5 w = 0.5")

        self.assertAlmostEqual(spec.constraints[0].value, math.pi / 2)
        self.assertEqual(spec.constraints[0].weight, 2.5)
        self.assertEqual(spec.constraints[1].weight, 0.5)

    def test_canonicalization(self):
        spec = parseStructureSpecString(
            "C1 - C2 - C3 - C4 60\nC4 - C3 - C2 - C1 60\nC3 - C2 - C1 100\nC1 - C2 - C3 100\nC2 - C1 1.5\nC1 - C2 1.5"
        )

        self.assertEqual(len(spec.constraints), 3)
        self.assertEqual(spec.constraints[0].indices, (0, 1, 2, 3))
        self.assertEqual(spec.constraints[1].indices, (0, 1, 2))
        self.assertEqual(spec.constraints[2].indices, (0, 1))

    def test_directives(self):
        self.assertEqual(parseStructureSpecString("C1 - C2 1.4").geometry, "general")
        self.assertEqual(parseStructureSpecString("@planar\nC1 - C2 1.4").geometry, "planar")
        self.assertEqual(parseStructureSpecString("C1 - C2 1.4\n  @linear  # comment").geometry, "linear")
        self.assertEqual(parseStructureSpecString("@planar\n@planar\nC1 - C2 1.4").geometry, "planar")

        for text in ("@planar\n@linear\nC1 - C2", "@flat\nC1 - C2", "@\nC1 - C2"):
            with self.subTest(text):
                with self.assertRaises(StructureSpecError):
                    parseStructureSpecString(text)

    def test_symmetry_operations(self):
        assert_almost_equal(parseSymmetryOperation("i"), -np.eye(3))
        assert_almost_equal(parseSymmetryOperation("C2(z)"), np.diag([-1, -1, 1]))
        assert_almost_equal(parseSymmetryOperation("sigma(y)"), np.diag([1, -1, 1]))
        assert_almost_equal(parseSymmetryOperation("sigma(0,0,2)"), np.diag([1, 1, -1]))
        assert_almost_equal(parseSymmetryOperation("S2(z)"), -np.eye(3))
        assert_almost_equal(parseSymmetryOperation("S4(z)") @ [1, 0, 0], [0, 1, 0])
        assert_almost_equal(parseSymmetryOperation("S4(z)") @ [0, 0, 1], [0, 0, -1])

        c3 = parseSymmetryOperation("C3(z)")
        assert_almost_equal(c3 @ [1, 0, 0], [math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3), 0])
        assert_almost_equal(np.linalg.matrix_power(c3, 3), np.eye(3))
        # C2 about the diagonal exchanges x and y
        assert_almost_equal(parseSymmetryOperation("C2(1,1,0)") @ [1, 0, 0], [0, 1, 0])

        for invalid in ("", "C2", "C2(w)", "C0(z)", "sigma(0,0,0)", "sigma(1,2)", "T(z)", "C2(z"):
            with self.subTest(invalid):
                with self.assertRaises(ValueError):
                    parseSymmetryOperation(invalid)

    def test_symmetry_directive(self):
        spec = parseStructureSpecString(
            """
            C1 - C2 1.4
            @symmetry C2(z): C1<->C2, H1<->H2   # H atoms are introduced here
            @symmetry sigma(x): C1->C2 , H3
            @symmetry i:
            """
        )

        self.assertEqual(spec.labels, ["C1", "C2", "H1", "H2", "H3"])
        self.assertEqual([op.name for op in spec.symmetry], ["C2(z)", "sigma(x)", "i"])
        self.assertEqual(spec.symmetry[0].cycles, ((0, 1), (2, 3)))
        self.assertEqual(spec.symmetry[1].cycles, ((0, 1),))
        self.assertEqual(spec.symmetry[2].cycles, ())
        assert_almost_equal(spec.symmetry[0].matrix, np.diag([-1, -1, 1]))

        # cycles of any length
        spec = parseStructureSpecString("@symmetry C3(z): C1->C2->C3, H1->H2->H3")
        self.assertEqual(spec.symmetry[0].cycles, ((0, 1, 2), (3, 4, 5)))

    def test_symmetry_errors(self):
        invalid = {
            "unknown operation": "@symmetry flip(z): C1<->C2",
            "missing colon": "@symmetry C2(z) C1<->C2",
            "repeated atom": "@symmetry C2(z): C1<->C2, C1<->C3",
            "repeated in cycle": "@symmetry C3(z): C1->C2->C1",
            "bad exchange": "@symmetry C2(z): C1<->C2<->C3",
            "bad label": "@symmetry C2(z): C1<->foo",
            "bad element": "@symmetry C2(z): C1<->Xx2",
        }
        for name, text in invalid.items():
            with self.subTest(name):
                with self.assertRaises(StructureSpecError):
                    parseStructureSpecString(text)

        with self.assertRaisesRegex(StructureSpecError, r":2:"):
            parseStructureSpecString("C1 - C2 1.0\n@symmetry C2(q): C1<->C2")

    def test_only_bonds_may_omit_the_value(self):
        for text in ("C1 - C2 1.0\nC2 - C1 - C3\n", "C1 - C2 - C3 - C4\n"):
            with self.assertRaisesRegex(StructureSpecError, "only bond records"):
                parseStructureSpecString(text)

    def test_partial_and_fixed(self):
        spec = parseStructureSpecString(
            "@symmetry C2(z): C1<->C2, H1, ...\n@symmetry sigma(y): C1<->C2\n"
        )

        first, second = spec.symmetry
        self.assertTrue(first.partial)
        self.assertEqual(first.cycles, ((0, 1),))
        self.assertEqual(first.fixed, (2,))
        self.assertFalse(second.partial)
        self.assertEqual(second.fixed, ())

        # Exchanges are closed cycles, chains are only closed if the operation is complete
        spec = parseStructureSpecString("@symmetry C3(z): C1->C2, C3<->C4, ...")
        self.assertEqual(spec.symmetry[0].cycles, ((0, 1), (2, 3)))
        self.assertEqual(spec.symmetry[0].chains, ((0, 1),))

    def test_groups(self):
        spec = parseStructureSpecString(
            "C1 - H1 1.0\n@group C2v\n@symmetry C2(z): H1<->H2\nC1 - H2\n"
        )

        self.assertEqual(spec.groups, ["C2v"])
        names = [op.name for op in spec.symmetry]
        # The explicit operation is kept (and becomes partial), the other generator is added
        self.assertEqual(names, ["C2(z)", "sigma(y)"])
        self.assertTrue(all(op.partial for op in spec.symmetry))
        self.assertEqual(spec.symmetry[0].cycles, ((1, 2),))
        self.assertEqual(spec.symmetry[1].cycles, ())

        # Matches by matrix, not by spelling
        spec = parseStructureSpecString("@group C2v\n@symmetry sigma(0,1,0): H1<->H2\nH1 - H2")
        self.assertEqual([op.name for op in spec.symmetry], ["sigma(0,1,0)", "C2(z)"])

        with self.assertRaises(StructureSpecError):
            parseStructureSpecString("@group Xyz\nC1 - C2")

    def test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spec.txt")
            with open(path, "w") as f:
                f.write("O1 - H1 0.96\nO1 - H2 0.96\nH1 - O1 - H2 104.5\n")

            spec = parseStructureSpec(path)

        self.assertEqual(spec.labels, ["O1", "H1", "H2"])
        self.assertEqual(len(spec.constraints), 3)

    def test_errors(self):
        invalid = {
            "unknown element": "Xx1 - C1 1.0",
            "no labels": "1.5 - 2.5",
            "garbage": "C1 - C2 1.0 banana",
            "bad number": "C1 - C2 abc",
            "second number": "C1 - C2 1.0 2.0",
            "too many atoms": "C1 - C2 - C3 - C4 - C5 1.0",
            "repeated atom": "C1 - C2 - C1 100",
            "missing angle": "C1 - C2 - C3",
            "missing dihedral": "C1 - C2 - C3 - C4",
            "negative bond": "C1 - C2 -1.0",
            "angle range": "C1 - C2 - C3 200",
            "bad weight": "C1 - C2 1.0 w=0",
            "conflict": "C1 - C2 1.0\nC2 - C1 1.1",
            "dihedral conflict": "C1 - C2 - C3 - C4 90\nC4 - C3 - C2 - C1 -90",
            "empty": "# nothing here",
            "no digits": "C - H 1.0",
        }

        for name, text in invalid.items():
            with self.subTest(name):
                with self.assertRaises(StructureSpecError):
                    parseStructureSpecString(text)

    def test_error_message_has_line_number(self):
        with self.assertRaisesRegex(StructureSpecError, r"mine\.txt:3:"):
            parseStructureSpecString("C1 - C2 1.0\n\nC1 - C3 abc", "mine.txt")


    def test_group_frame_adapts_to_given_operation(self):
        base = "O1 - H1 0.96\nO1 - H2 0.96\nH1 - O1 - H2 104.5\n@group C2v\n"
        for axis in "xyz":
            spec = parseStructureSpecString(base + "@symmetry C2(%s): H1<->H2, ...\n" % axis)
            self.assertEqual(len(spec.symmetry), 2)
            assert_almost_equal(spec.symmetry[0].matrix, parseSymmetryOperation("C2(%s)" % axis))
            self.assertEqual(spec.symmetry[0].name, "C2(%s)" % axis)

    def test_group_frame_is_canonical_without_operations(self):
        spec = parseStructureSpecString("O1 - H1 0.96\nO1 - H2 0.96\nH1 - O1 - H2 104.5\n@group C2v\n")
        self.assertEqual([op.name for op in spec.symmetry], ["C2(z)", "sigma(y)"])

    def test_group_frame_adapts_to_mirror_plane(self):
        spec = parseStructureSpecString("O1 - H1 0.96\nO1 - H2 0.96\nH1 - O1 - H2 104.5\n@group Cs\n@symmetry sigma(x): O1, H1, H2\n")
        self.assertEqual(len(spec.symmetry), 1)
        assert_almost_equal(spec.symmetry[0].matrix, parseSymmetryOperation("sigma(x)"))

    def test_group_frame_of_planar_molecule(self):
        # The molecular plane (xy) has to be a mirror plane of the group, so the C2 axis of a planar C2v molecule lies in the plane
        base = "O1 - H1 0.96\nO1 - H2 0.96\nH1 - O1 - H2 104.5\n@group C2v\n"
        for spec in (
            parseStructureSpecString("@planar\n" + base),
            parseStructureSpecString(base, geometry="planar"),
        ):
            names = [op.name for op in spec.symmetry]
            self.assertEqual(spec.geometry, "planar")
            self.assertIn(names[0], ("C2(x)", "C2(y)"))
            self.assertNotIn("C2(z)", names)

        # without planarity the canonical frame is kept
        self.assertEqual([op.name for op in parseStructureSpecString(base).symmetry], ["C2(z)", "sigma(y)"])

        with self.assertRaises(ValueError):
            parseStructureSpecString(base, geometry="cubic")


if __name__ == "__main__":
    unittest.main()
