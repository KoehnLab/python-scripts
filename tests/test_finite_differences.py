#!/usr/bin/env python3

import unittest

from typing import List, Sequence, Tuple

from koehnlab.finite_differences import (
    approximate_derivative,
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

    def straigh_line(
        self, locations: Sequence[float] = [0, 1], offset: float = 0
    ) -> List[float]:
        points: list[float] = []
        for x in locations:
            points.append(0.5 * x + offset)

        return points

    def straight_line_derivative(self, order: int = 1) -> float:
        return 0.5 if order == 1 else 0

    def parabola(
        self, locations: Sequence[float] = [-1, 0, 1], offset: float = 0
    ) -> List[float]:
        points: list[float] = []
        for x in locations:
            points.append(-1.75 * (x**2) + 0.5 * x + offset)

        return points

    def parabola_derivative(self, at: float = 0, order: int = 1) -> float:
        if order == 1:
            return 2 * -1.75 * at + 0.5
        elif order == 2:
            return 2 * -1.75
        else:
            return 0

    def test_forward_difference(self):
        self.assertAlmostEqual(
            forward_difference(self.straigh_line(locations=[1, 1.5]), 0.5),
            self.straight_line_derivative(),
        )

        self.assertAlmostEqual(
            forward_difference(
                self.straigh_line(locations=[-12, -11.5], offset=3), 0.5
            ),
            self.straight_line_derivative(),
        )

    def test_central_difference(self):
        # Straight line
        self.assertAlmostEqual(
            central_difference(self.straigh_line(locations=[-12, -11], offset=3), 0.5),
            self.straight_line_derivative(),
        )
        self.assertAlmostEqual(
            central_difference(self.straigh_line(locations=[1, 2], offset=-7), 0.5),
            self.straight_line_derivative(),
        )
        self.assertAlmostEqual(
            central_difference(
                self.straigh_line(locations=[0.5, 1, 2, 2.5], offset=0.1), 0.5
            ),
            self.straight_line_derivative(),
        )

        # Parabola
        self.assertAlmostEqual(
            central_difference(self.parabola(locations=[1, 2], offset=0.1), 0.5),
            self.parabola_derivative(at=1.5),
        )
        self.assertAlmostEqual(
            central_difference(
                self.parabola(locations=[0.5, 1, 2, 2.5], offset=-42), 0.5
            ),
            self.parabola_derivative(at=1.5),
        )

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

    def test_approximate_derivative(self):
        for order in [1, 2, 3]:
            for location in [-4, 0, 12]:
                with self.subTest(order=order, location=location):
                    x_values = [-4, -3.8, -3.4, -3.1, -2.5]

                    # Straight line
                    y_values = self.straigh_line(locations=x_values, offset=-8)

                    derivative = approximate_derivative(
                        x_values=x_values, y_values=y_values, x0=location, order=order
                    )
                    self.assertAlmostEqual(
                        derivative, self.straight_line_derivative(order=order)
                    )

                    # Parabola
                    y_values = self.parabola(locations=x_values, offset=12)

                    derivative = approximate_derivative(
                        x_values=x_values, y_values=y_values, x0=location, order=order
                    )
                    self.assertAlmostEqual(
                        derivative, self.parabola_derivative(at=location, order=order)
                    )


if __name__ == "__main__":
    unittest.main()
