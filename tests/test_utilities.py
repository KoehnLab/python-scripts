#!/usr/bin/env python3

import unittest

from koehnlab.utilities import (
    sortedIndices,
    proxySort,
    getMainElements,
    compositeSort,
)

import numpy as np


class TestUtilities(unittest.TestCase):
    def test_compositeSort(self):
        mainList = [3, 1, 2]
        lst1 = ["a", "b", "c"]
        lst2 = [None, float, int]

        self.assertEqual([1, 2, 3], compositeSort(mainList))
        self.assertEqual(
            [[1, 2, 3], ["b", "c", "a"]], list(compositeSort(mainList, lst1))
        )
        self.assertEqual(
            [[1, 2, 3], ["b", "c", "a"], [float, int, None]],
            list(compositeSort(mainList, lst1, lst2)),
        )

    def test_proxySort(self):
        self.assertEqual([1, 2, 0], proxySort(indices=[0, 1, 2], data=[43, 2, 21]))

        array = np.asarray([[5, 3, 7], [0, -42, 12]])

        self.assertEqual(
            [(1, 1), (1, 0), (0, 1), (0, 0), (0, 2), (1, 2)],
            proxySort(
                indices=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], data=array
            ),
        )

    def test_sortedIndices(self):
        self.assertEqual([1, 2, 0], sortedIndices([42, 1, 17]))

        array = np.asarray([[5, 3], [4, -42], [0, 7]])

        self.assertEqual(
            [(1, 1), (2, 0), (0, 1), (1, 0), (0, 0), (2, 1)], sortedIndices(array)
        )

    def test_getMainElements(self):
        data = [-0.6, 1, 0, 0.5]
        self.assertEqual([1, 0.5], getMainElements(data, threshold=0.1))
        self.assertEqual([1], getMainElements(data, sumPercentage=0.8))
        self.assertEqual(
            [1, -0.6], getMainElements(data, sumPercentage=0.6, useNorm=True)
        )

        data = [1 + 0j, 0 + 2j, 0.1 + 0.1j]
        self.assertEqual(
            [0 + 2j, 1 + 0j], getMainElements(data, sumPercentage=0.95, useNorm=True)
        )


if __name__ == "__main__":
    unittest.main()
