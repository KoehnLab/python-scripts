from typing import List

from copy import deepcopy

from .Element import Element
from .Atom import Atom

import numpy as np
from numpy.typing import ArrayLike


class Molecule:
    """Class representing a molecule as a list of Atoms"""

    def __init__(self, atoms: List[Atom] = []):
        self.atoms: List[Atom] = deepcopy(atoms)

    def translate(self, delta: ArrayLike) -> None:
        """Translates the entire molecule by the given delta"""
        for currentAtom in self.atoms:
            currentAtom.coordinates += delta

    def transform(self, transformation: np.ndarray) -> None:
        """Applies the given transformation to this molecule"""
        for currentAtom in self.atoms:
            currentAtom.coordinates = np.matmul(transformation, currentAtom.coordinates)

    def centerOfMass(self) -> np.ndarray:
        """Computes the center of mass of this molecule"""
        center = np.zeros(3)

        totalMass: float = 0

        for currentAtom in self.atoms:
            totalMass += currentAtom.element.mass()
            center += currentAtom.coordinates * currentAtom.element.mass()

        center *= 1 / totalMass

        return center

    def inertiaTensor(self) -> np.ndarray:
        """Computes the inertia tensor for this molecule"""
        tensor = np.zeros(shape=(3, 3))

        center = self.centerOfMass()

        # T_{ij} = sum_k m_k * (|r_k|^2 * delta_{ij} - r_k[i] * r_k[j])
        # where m_k is the k-th element's mass, r_k its position and r_k[i] is the i-th coordinate of that position
        # See also https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        for i in range(3):
            for j in range(3):
                for currentAtom in self.atoms:
                    relCoordinates = currentAtom.coordinates - center

                    if i == j:
                        tensor[i, j] += (
                            currentAtom.element.mass()
                            * np.linalg.norm(relCoordinates) ** 2
                        )


                    tensor[i, j] -= (
                        currentAtom.element.mass()
                        * relCoordinates[i]
                        * relCoordinates[j]
                    )

        return tensor

    def bringToStandardOrientation(self) -> None:
        """Translates the molecule such that its center of mass is located at (0,0,0) and rotates it
        such that the molecule's principle axes are aligned with the cartesian coordinate axes
        """
        # Bring center of mass to the origin
        self.translate(-self.centerOfMass())

        inertiaTensor = self.inertiaTensor()

        # Make use of the fact that the inertia tensor is symmetric
        eigvals, eigvectors = np.linalg.eigh(inertiaTensor)

        assert len(eigvals) == 3
        assert len(eigvals) == len(eigvectors)

        # Rotate the molecule to its principle axes
        self.transform(np.linalg.inv(eigvectors))
