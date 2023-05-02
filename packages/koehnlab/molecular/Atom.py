from typing import Any

from .Element import Element

import numpy as np


class Atom:
    """Representation of an atom"""

    def __init__(self, element: Element, coordinates: Any = [0, 0, 0]):
        self.element: Element = element
        self.coordinates: np.ndarray = np.asarray(coordinates, dtype=float)

        assert len(coordinates) == 3, "Expected atoms to exist in 3D space"
