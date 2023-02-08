#!/usr/bin/env python3

import unittest

from koehnlab.finite_differences import central_difference, forward_difference


class TestFiniteDifference(unittest.TestCase):
    def test_forward_difference(self):
        self.assertAlmostEqual(forward_difference([1, 2], 0.5), 2)

    def test_central_difference(self):
        # Assume straight line with y = x
        self.assertAlmostEqual(central_difference([1, 2], 0.5), 1)
        self.assertAlmostEqual(central_difference([0, 0.5, 1.5, 2], 0.5), 1)

        # Assume y = x + 5
        self.assertAlmostEqual(central_difference([6, 7], 0.5), 1)
        self.assertAlmostEqual(central_difference([5, 5.5, 6.5, 7], 0.5), 1)

        # Assume y = x**2 + 1
        # Evaluate slope at origin
        self.assertAlmostEqual(central_difference([1.25, 1.25], 0.5), 0)
        self.assertAlmostEqual(central_difference([2, 1.25, 1.25, 2], 0.5), 0)

        # Evaluate slope at x = 1
        self.assertAlmostEqual(central_difference([1.25, 3.25], 0.5), 2)
        self.assertAlmostEqual(central_difference([1, 1.25, 3.25, 5], 0.5), 2)


if __name__ == "__main__":
    unittest.main()
