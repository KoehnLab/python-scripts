from collections.abc import Sequence


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
