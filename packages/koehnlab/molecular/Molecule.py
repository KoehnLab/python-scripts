from typing import List, Tuple

from copy import deepcopy

from .Element import Element
from .Atom import Atom
from .Orientation import principalAxes

import numpy as np
from numpy.typing import ArrayLike


class Molecule:
    """Class representing a molecule as a list of Atoms"""

    def __init__(self, atoms: List[Atom] = []):
        self.atoms: List[Atom] = deepcopy(atoms)

    def nAtoms(self) -> int:
        """Return number of atoms"""
        return len(self.atoms)

    def coordinates(self) -> np.ndarray:
        """Return coordinates as array"""
        coord = []
        for currentAtom in self.atoms:
            coord.append(currentAtom.coordinates)
        return np.array(coord)

    def masses(self) -> np.ndarray:
        """Return masses of all atoms"""
        mass = []
        for currentAtom in self.atoms:
            mass.append(currentAtom.element.mass())
        return np.array(mass)

    def massVec(self) -> np.ndarray:
        """Return masses of all atoms as mass vector (same mass for x,y,z)"""
        massV = []
        for currentAtom in self.atoms:
            mass = currentAtom.element.mass()
            massV.append([mass,mass,mass])
        return np.array(massV)

    def addAtom(self, atom: Atom) -> None:
        """Add one atom to the current molecule"""
        self.atoms.append(atom)

    def addAtomList(self, atoms: List[Atom]) -> None:
        """Add a list of atoms to the current molecule"""
        self.atoms.extend(atoms)

    def setCoordinates(self, coord: np.ndarray) -> None:
        """Put (externally modified) coordinates into object, number of atoms must not change"""
        assert coord.shape[0] == len(self.atoms)
        assert coord.shape[1] == 3
        nAtoms = len(self.atoms)
        for idx in range(nAtoms):
            self.atoms[idx].coordinates = coord[idx]

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

    def geometricCenter(self) -> np.ndarray:
        """Computes the geometric center (average of all atomic positions) of this molecule"""
        return np.mean(self.coordinates(), axis=0)

    def _centerAndWeights(self, center: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the origin and the per-atom weights belonging to the given centering mode:
        "geometric": geometric center, all atoms weighted equally
        "mass": center of mass, atoms weighted by their mass
        """
        if center == "geometric":
            return self.geometricCenter(), np.ones(self.nAtoms())
        if center == "mass":
            return self.centerOfMass(), self.masses()

        raise ValueError(
            "Unknown centering mode '%s' (expected 'geometric' or 'mass')" % center
        )

    def inertiaTensor(self, center: str = "geometric") -> np.ndarray:
        """Computes the inertia tensor for this molecule.

        With center="geometric" (default), the tensor is evaluated with respect to the geometric center
        and all atoms have unit weight, so the result only depends on the geometry. With center="mass",
        the usual mass-weighted tensor with respect to the center of mass is returned.
        """
        origin, weights = self._centerAndWeights(center)

        relCoordinates = self.coordinates() - origin

        # T_{ij} = sum_k w_k * (|r_k|^2 * delta_{ij} - r_k[i] * r_k[j])
        # where w_k is the k-th atom's weight and r_k its position relative to the origin
        # See also https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        squaredNorms = np.sum(relCoordinates**2, axis=1)
        tensor = np.sum(weights * squaredNorms) * np.eye(3)
        tensor -= np.einsum("k,ki,kj->ij", weights, relCoordinates, relCoordinates)

        return tensor

    def bringToStandardOrientation(self, center: str = "geometric") -> None:
        """Translates the molecule such that its center is located at (0,0,0) and rotates it
        such that the molecule's principle axes are aligned with the cartesian coordinate axes.

        By default, the geometric center is used as the origin and the principle axes are determined
        from the unweighted inertia tensor. With center="mass", the center of mass and the mass-weighted
        inertia tensor are used instead.
        """
        # Bring the center to the origin
        if self.nAtoms() == 0:
            return

        origin, _ = self._centerAndWeights(center)
        self.translate(-origin)

        inertiaTensor = self.inertiaTensor(center)

        # Rotate the molecule to its principle axes. These are chosen such that the result is deterministic
        # and a proper rotation (no mirroring), even if the principal moments are degenerate
        axes = principalAxes(self.coordinates(), inertiaTensor)
        self.transform(axes.T)
