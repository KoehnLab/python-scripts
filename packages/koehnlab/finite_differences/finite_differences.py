from typing import Any, Optional, Tuple

from collections.abc import Sequence
import copy


class OffsetList:
    """This class represents a list structure that may use a different offset system than 0-based.
    The provided indexing offset will be subtracted from any provided offset used to index this list
    in order to arrive at the final offset with which the actually stored list is accessed.
    E.g. indexing_offset=1 means that a 1-based system is used where the first element is accessed
    via object[1] instead of object[0]"""

    def __init__(
        self,
        initial_size: int = 0,
        indexing_offset: int = 0,
        default_value: Any = 0,
    ):
        self.data = []
        for _ in range(initial_size):
            self.data.append(copy.deepcopy(default_value))

        self.offset: int = indexing_offset

    def __getitem__(self, index):
        assert index >= 0 and index < len(self.data)
        return self.data[index - self.offset]

    def __setitem__(self, index, value):
        assert index >= 0 and index < len(self.data)
        self.data[index - self.offset] = value

    def __str__(self):
        str_rep = "[ "
        for current in self.data:
            str_rep += str(current) + ", "

        if len(self.data) > 0:
            # Remove trailing ", "
            str_rep = str_rep[: -len(", ")]

        str_rep += " ]"

        return str_rep


class CoefficientHolder:
    """A class meant to hold the coefficients for the algorithm generating the coefficients for finite
    differences formulas according to B. Fornberg. Thus, this represents a 3D array"""

    def __init__(self, N: int, M: int):
        self.coefficients = OffsetList(
            initial_size=M + 1,
            indexing_offset=0,
            default_value=OffsetList(
                initial_size=N + 1,
                indexing_offset=1,
                default_value=OffsetList(initial_size=N + 1, default_value=0),
            ),
        )

    def __getitem__(self, index):
        return self.coefficients[index]

    def __str__(self):
        return str(self.coefficients)


def generate_finite_difference_coefficients(
    x0: float, x_values: Sequence[float], order=1
) -> Sequence[float]:
    """Generates the weights that can be used to approximate the derivative of the given order
    at the given location x0 from the function values at the provided x-values. The result is
    a sequence of coefficients (same amount as the provided x values) that can then be element-wise
    multiplied with the list of corresponding function values and then summed to arrive at the
    approximated derivative. The provided x values must not contain any duplicates.
    The used algorithm is described and derived in Fornberg, B. (1988). Math. Comp., 51(184), 699–706.
    """

    assert len(x_values) > 0
    N = len(x_values) - 1
    M = order

    coefficients = CoefficientHolder(N, M)

    coefficients[0][0][0] = 1

    c1: float = 1
    for n in range(1, N + 1):
        c2: float = 1
        for v in range(n):
            c3: float = x_values[n] - x_values[v]
            c2 *= c3

            # Our array is already zero-initialized, so we don't need this
            # if n <= M:
            #    coefficients[n][n-1][v] = 0

            for m in range(min(n + 1, M + 1)):
                first: float = (x_values[n] - x0) * coefficients[m][n - 1][v]
                second: float = m * coefficients[m - 1][n - 1][v] if m > 0 else 0

                coefficients[m][n][v] = (first - second) / c3

        for m in range(min(n + 1, M + 1)):
            first: float = m * coefficients[m - 1][n - 1][n - 1] if m > 0 else 0
            second: float = (x_values[n - 1] - x0) * coefficients[m][n - 1][n - 1]

            coefficients[m][n][n] = (c1 / c2) * (first - second)

        c1 = c2

    return coefficients[order][N].data


def forward_difference(values: Sequence[float], delta: float) -> float:
    """Calculates the forward difference of the given two values that are separated by the given delta.
    The provided points are expected to be [f(x), f(x + delta)], where the forward difference is
    evaluated at x"""
    assert len(values) == 2

    return (values[1] - values[0]) / delta


def central_difference(values: Sequence[float], delta: float) -> float:
    """Calculates the central difference of the given values that are separated by the given delta.
    Depending on the amount of values provided, either the two-point or the four-point formula will
    be used.
    The provided values are expected to be either
    [f(x - delta), f(x + delta)] or
    [f(x - 2*delta), f(x - delta), f(x + delta), f(x + 2*delta)]
    where the central difference is evaluated at x"""
    assert len(values) in [2, 4]

    if len(values) == 2:
        # 2-point formula
        return (values[1] - values[0]) / (2 * delta)
    else:
        # 4-point formula
        return (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * delta)


def approximate_derivative(
    points: Optional[Sequence[Tuple[float, float]]] = None,
    x_values: Optional[Sequence[float]] = None,
    y_values: Optional[Sequence[float]] = None,
    order: int = 1,
    x0: float = 0,
) -> float:
    """Approximates the derivative of given order at the given position x0 by means of the finite differences
    method. The provided grid points may be spaced arbitrarily and are free to either contain a value for x0
    or not. An order of zero corresponds to an interpolation (or extraplolation) of the function to the
    provided location"""
    if not points is None:
        if not x_values is None or not y_values is None:
            raise RuntimeError(
                "Arguments 'points' and ('x_values' & 'y_values') are mutually exclusive"
            )

        x_values = []
        y_values = []
        for x, y in points:
            x_values.append(x)
            y_values.append(y)

    if x_values is None or y_values is None:
        raise RuntimeError(
            "Argument 'points' or ('x_values' and 'y_values') are mandatory"
        )

    if len(x_values) != len(y_values):
        raise RuntimeError(
            "The size of the provided 'x_values' and 'y_values' sequences must be equal"
        )

    weights = generate_finite_difference_coefficients(
        x0=x0, x_values=x_values, order=order
    )

    assert len(weights) == len(y_values)

    derivative = 0
    for i in range(len(y_values)):
        derivative += weights[i] * y_values[i]

    return derivative
