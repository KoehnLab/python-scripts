#!/usr/bin/env python3

import unittest

from typing import Tuple

from koehnlab.finite_differences import (
    central_difference,
    forward_difference,
    generate_finite_difference_coefficients,
)


class TestFiniteDifference(unittest.TestCase):
    def assertSequenceAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        self.assertEqual(len(first), len(second), msg=msg)

        for i in range(len(first)):
            self.assertAlmostEqual(
                first[i], second[i], places=places, msg=msg, delta=delta
            )

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

    def test_generate_finite_difference_coefficients(self):
        # The test data is taken from the tables in
        # Fornberg, B. (1988). Math. Comp., 51(184), 699–706.
        test_data: list[Tuple[float, int, list[float], list[float]]] = []
        # Data entries have format [x0, order, x-values, expected coefficients]

        # Approximations at x = 0 (Table 1)
        test_data.append((0, 0, [0], [1]))

        test_data.append((0, 1, [-1, 0, 1], [-1 / 2, 0, 1 / 2]))
        test_data.append((0, 1, [-2, -1, 0, 1, 2], [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]))
        test_data.append(
            (
                0,
                1,
                [-3, -2, -1, 0, 1, 2, 3],
                [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60],
            )
        )
        test_data.append(
            (
                0,
                1,
                [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280],
            )
        )

        test_data.append((0, 2, [-1, 0, 1], [1, -2, 1]))
        test_data.append(
            (0, 2, [-2, -1, 0, 1, 2], [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
        )
        test_data.append(
            (
                0,
                2,
                [-3, -2, -1, 0, 1, 2, 3],
                [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
            )
        )
        test_data.append(
            (
                0,
                2,
                [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                [
                    -1 / 560,
                    8 / 315,
                    -1 / 5,
                    8 / 5,
                    -205 / 72,
                    8 / 5,
                    -1 / 5,
                    8 / 315,
                    -1 / 560,
                ],
            )
        )

        # Centered approximations at "half-way" points at x = 0 (Table 2)
        test_data.append((0, 0, [-1 / 2, 1 / 2], [1 / 2, 1 / 2]))
        test_data.append(
            (
                0,
                0,
                [-5 / 2, -3 / 2, -1 / 2, 1 / 2, 3 / 2, 5 / 2],
                [3 / 256, -25 / 256, 75 / 128, 75 / 128, -25 / 256, 3 / 256],
            )
        )

        test_data.append((0, 1, [-1 / 2, 1 / 2], [-1, 1]))
        test_data.append(
            (
                0,
                1,
                [-5 / 2, -3 / 2, -1 / 2, 1 / 2, 3 / 2, 5 / 2],
                [-3 / 640, 25 / 384, -75 / 64, 75 / 64, -25 / 384, 3 / 640],
            )
        )

        test_data.append(
            (
                0,
                4,
                [-5 / 2, -3 / 2, -1 / 2, 1 / 2, 3 / 2, 5 / 2],
                [1 / 2, -3 / 2, 1, 1, -3 / 2, 1 / 2],
            )
        )

        # One-sided approximation at x = 0 (Table 3)
        test_data.append((0, 0, [0], [1]))

        test_data.append((0, 1, [0, 1, 2, 3], [-11 / 6, 3, -3 / 2, 1 / 3]))
        test_data.append(
            (0, 1, [0, 1, 2, 3, 4, 5], [-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5])
        )

        test_data.append(
            (
                0,
                3,
                [0, 1, 2, 3, 4, 5],
                [-17 / 4, 71 / 4, -59 / 2, 49 / 2, -41 / 4, 7 / 4],
            )
        )

        for x0, order, x_values, expected_coefficients in test_data:
            assert len(x_values) == len(expected_coefficients)

            with self.subTest(x0=x0, order=order, x_values=x_values):
                self.assertSequenceAlmostEqual(
                    generate_finite_difference_coefficients(
                        x0=x0, x_values=x_values, order=order
                    ),
                    expected_coefficients,
                )


if __name__ == "__main__":
    unittest.main()
