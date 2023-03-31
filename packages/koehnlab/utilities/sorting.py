from numpy.typing import ArrayLike
from collections.abc import Sequence

import numpy as np
import itertools
from functools import cmp_to_key


def proxySort(indices: Sequence, data) -> Sequence:
    """Returns a permutation of the given index sequence that when indexing the data
    object by the returned indices (in that order) the data's elements will be accessed in a sorted manner"""
    proxyIndices = list(range(len(indices)))

    def comparator(lhs: int, rhs: int):
        lhs = data[indices[lhs]]
        rhs = data[indices[rhs]]
        if lhs == rhs:
            return 0
        elif lhs < rhs:
            return -1
        else:
            return 1

    proxyIndices = sorted(proxyIndices, key=cmp_to_key(comparator))

    return [indices[x] for x in proxyIndices]


def sortedIndices(sequence: ArrayLike) -> Sequence:
    """Returns a sequence of indices into the provided object such that access will be in sorted order
    as long as the indices are used in the returned order"""
    array = np.asarray(sequence)

    if len(array) == 0:
        return []

    if len(array.shape) == 1:
        indices = [i for i in range(len(array))]
    else:
        indices = list(itertools.product(*[range(x) for x in array.shape]))

    return proxySort(indices=indices, data=array)
