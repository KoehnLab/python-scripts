#!/usr/bin/env python3

import math
import os
import tempfile
import unittest

from koehnlab.molecular import Element
from koehnlab.molecular.StructureParser import (
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


if __name__ == "__main__":
    unittest.main()
