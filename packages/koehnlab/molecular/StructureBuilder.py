from typing import Dict, List, Optional, Tuple

from collections import deque

import math

import numpy as np
from ase.data import covalent_radii

from .InternalCoordinates import canonicalAngle, canonicalBond, canonicalDihedral
from .StructureParser import StructureSpec

# Default angles (in degrees) at an atom depending only on the number of its neighbors. This is deliberately simple:
# The default geometry is only meant as a rough starting point for the optimization and users who care about the
# unspecified parts of the geometry should provide a starting geometry themselves. Linear arrangements are avoided as
# default since the torsion about a linear fragment is not defined.
_DEFAULT_ANGLES: Dict[int, float] = {2: 120.0, 3: 120.0, 4: 109.4712206}
_DEFAULT_ANGLE_MANY_NEIGHBORS = 90.0


def _defaultAngle(nNeighbors: int) -> float:
    return math.radians(_DEFAULT_ANGLES.get(nNeighbors, _DEFAULT_ANGLE_MANY_NEIGHBORS))


def _neighborLists(spec: StructureSpec) -> List[List[int]]:
    neighbors: List[List[int]] = [[] for _ in range(spec.nAtoms())]
    for i, j in spec.bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    return [sorted(n) for n in neighbors]


def _arbitraryPerpendicular(axis: np.ndarray) -> np.ndarray:
    reference = np.zeros(3)
    reference[np.argmin(np.abs(axis))] = 1.0
    perpendicular = reference - (reference @ axis) * axis
    return perpendicular / np.linalg.norm(perpendicular)


def _placeAtom(
    coordinates: np.ndarray,
    center: int,
    axisAtom: int,
    reference: Optional[int],
    distance: float,
    angle: float,
    torsion: float,
) -> np.ndarray:
    """Computes the position of a new atom that is bonded to center with the given distance. angle is the angle between
    the new atom, the center and the axisAtom. torsion is the dihedral angle new atom - center - axisAtom - reference
    (if there is no reference, a deterministic arbitrary direction is used for torsion=0)."""
    origin = coordinates[center]
    axis = coordinates[axisAtom] - origin
    axis = axis / np.linalg.norm(axis)

    e1: Optional[np.ndarray] = None
    if reference is not None:
        v = coordinates[reference] - origin
        v = v - (v @ axis) * axis
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            e1 = v / norm
    if e1 is None:
        e1 = _arbitraryPerpendicular(axis)

    e2 = np.cross(e1, axis)

    direction = math.cos(angle) * axis + math.sin(angle) * (
        math.cos(torsion) * e1 + math.sin(torsion) * e2
    )

    return origin + distance * direction


def defaultGuess(spec: StructureSpec) -> np.ndarray:
    """Generates a rough initial geometry (in Angstrom) for the given specification solely from the connectivity.

    The atoms are placed one after another along a spanning tree of the bond graph. For every atom, bond length, angle
    and torsion are taken from the specification, if it contains the corresponding parameter. Otherwise defaults are used:
    The sum of the covalent radii as bond length, an angle that only depends on the number of neighbors of the central atom
    (120 degrees for 2 or 3, 109.47 degrees for 4 and 90 degrees for more neighbors) and torsions that distribute the
    substituents of an atom evenly (staggered/planar). Bonds that close rings are ignored, so rings are in general not
    closed in the result. Atoms are in the same order as in the specification.
    """
    nAtoms = spec.nAtoms()
    neighbors = _neighborLists(spec)

    bondValues: Dict[Tuple[int, ...], float] = {}
    angleValues: Dict[Tuple[int, ...], float] = {}
    dihedralValues: Dict[Tuple[int, ...], float] = {}
    for constraint in spec.constraints:
        {2: bondValues, 3: angleValues, 4: dihedralValues}[len(constraint.indices)][
            constraint.indices
        ] = constraint.value

    radii = []
    for element in spec.elements:
        z = element.atomicNumber()
        if z >= len(covalent_radii):
            raise ValueError("No covalent radius known for element %s" % element.symbol())
        radii.append(float(covalent_radii[z]))

    coordinates = np.zeros((nAtoms, 3))
    placed = [False] * nAtoms

    # Breadth-first traversal of the bond graph starting from the first atom
    placed[0] = True
    queue = deque([0])
    order: List[Tuple[int, int]] = []  # (atom, parent)
    visited = [False] * nAtoms
    visited[0] = True
    while queue:
        current = queue.popleft()
        for neighbor in neighbors[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                order.append((neighbor, current))
                queue.append(neighbor)

    if not all(visited):
        missing = [spec.labels[i] for i in range(nAtoms) if not visited[i]]
        raise ValueError(
            "The atoms are not all connected via bonds, unreachable from %s: %s"
            % (spec.labels[0], ", ".join(missing))
        )

    for atom, center in order:
        distance = bondValues.get(
            canonicalBond(atom, center), radii[atom] + radii[center]
        )

        # Already placed neighbors of the center, apart from the new atom itself
        others = [n for n in neighbors[center] if placed[n] and n != atom]

        if not others:
            coordinates[atom] = coordinates[center] + distance * np.array([0.0, 0.0, 1.0])
            placed[atom] = True
            continue

        def references(axisAtom: int) -> List[int]:
            """Atoms, bonded to axisAtom, which can be used to measure the torsion"""
            return [
                n
                for n in neighbors[axisAtom]
                if placed[n] and n != center and n != atom
            ]

        # Choose the atoms that define the angle and torsion, preferring those for which the specification
        # contains the corresponding parameter
        axisAtom: Optional[int] = None
        reference: Optional[int] = None
        for candidate in others:
            for ref in references(candidate):
                if canonicalDihedral(atom, center, candidate, ref) in dihedralValues:
                    axisAtom, reference = candidate, ref
                    break
            if axisAtom is not None:
                break
        if axisAtom is None:
            for candidate in others:
                if canonicalAngle(atom, center, candidate) in angleValues:
                    axisAtom = candidate
                    break
        if axisAtom is None:
            axisAtom = others[0]
        if reference is None:
            candidates = references(axisAtom)
            reference = candidates[0] if candidates else None

        angle = angleValues.get(
            canonicalAngle(atom, center, axisAtom), _defaultAngle(len(neighbors[center]))
        )

        key = (
            canonicalDihedral(atom, center, axisAtom, reference)
            if reference is not None
            else None
        )
        if key is not None and key in dihedralValues:
            torsion = dihedralValues[key]
        else:
            # Distribute the substituents of the center evenly around the axis
            step = 2 * math.pi / (len(neighbors[center]) - 1)
            torsion = (math.pi if reference is not None else 0.0) + (len(others) - 1) * step

        coordinates[atom] = _placeAtom(
            coordinates, center, axisAtom, reference, distance, angle, torsion
        )
        placed[atom] = True

    return coordinates
