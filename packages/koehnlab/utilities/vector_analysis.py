from typing import Optional
from collections.abc import Sequence
from numpy.typing import ArrayLike

import numpy as np

from .sorting import sortedIndices


def getMainElementIndices(
    array: ArrayLike,
    threshold: Optional[float] = None,
    sumPercentage: Optional[float] = None,
    useNorm: bool = False,
) -> Sequence:
    """Gets the elements to the main entries in the provided array. The main elements are the biggest
    ones and the returned indices refer to elements that either are above the provided threshold or
    cover the provided percentage of the sum of all elements. The useNorm parameter can be used to
    consider the norm of the data entries instead of the entries themselves"""

    dataSource = np.abs(np.asarray(array)) if useNorm else np.asarray(array)
    targetSum = 0

    if threshold is None:
        if sumPercentage is None:
            # Default to 95%
            sumPercentage = 0.95
        else:
            assert sumPercentage >= 0 and sumPercentage <= 1

        total = np.sum(dataSource)
        targetSum = sumPercentage * total

    if threshold == 0:
        return []

    indices = sortedIndices(dataSource)

    if not threshold is None:
        indices = []

        for index, element in np.ndenumerate(dataSource):
            if element >= threshold:
                # numpy uses tuples as iterators, but for 1D arrays, we want to return integers
                assert len(index) > 0
                indices.append(index if len(index) > 1 else index[0])

        return indices
    else:
        total: float = 0
        for i in reversed(range(len(indices))):
            total += dataSource[indices[i]]  # type: ignore

            if total >= targetSum:
                return list(reversed(indices[i:]))

        return indices


def getMainElements(
    array: Sequence,
    threshold: Optional[float] = None,
    sumPercentage: Optional[float] = None,
    useNorm: bool = False,
) -> Sequence:
    return [
        array[idx]  # type: ignore
        for idx in getMainElementIndices(array, threshold, sumPercentage, useNorm)
    ]
