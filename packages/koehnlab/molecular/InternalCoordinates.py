from typing import Dict, List, Sequence, Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

# Atoms are always referred to by their (zero-based) index in the molecule.
# - A bond is a pair (i, j)
# - An angle is a triple (i, j, k), where j is the atom at the apex
# - A dihedral angle is a quadruple (i, j, k, l) where j-k is the central bond. It is positive, if looking along j->k,
#   the bond i-j has to be rotated clockwise to eclipse the bond k-l (IUPAC convention)
#
# All angles are in radians, bond lengths in the unit of the coordinates. Reversing the order of the atoms of any of
# these coordinates describes the same coordinate, which is why we also provide canonical forms.

Bond = Tuple[int, int]
Angle = Tuple[int, int, int]
Dihedral = Tuple[int, int, int, int]

# Lists of coordinates can also be given as integer arrays with one row per coordinate
Bonds = Union[Sequence[Bond], np.ndarray]
Angles = Union[Sequence[Angle], np.ndarray]
Dihedrals = Union[Sequence[Dihedral], np.ndarray]


def canonicalBond(i: int, j: int) -> Bond:
    """Canonical form of a bond: Smaller index first"""
    return (i, j) if i <= j else (j, i)


def canonicalAngle(i: int, j: int, k: int) -> Angle:
    """Canonical form of an angle: Smaller index of the two outer atoms first"""
    return (i, j, k) if i <= k else (k, j, i)


def canonicalDihedral(i: int, j: int, k: int, l: int) -> Dihedral:
    """Canonical form of a dihedral angle: The orientation (i,j,k,l) vs (l,k,j,i) leading to the lexicographically
    smaller tuple is used (the angle is the same in both directions)"""
    return (i, j, k, l) if (i, j, k, l) <= (l, k, j, i) else (l, k, j, i)


def canonicalCoordinate(indices: Sequence[int]) -> Tuple[int, ...]:
    """Canonical form of a bond (2 atoms), angle (3 atoms) or dihedral angle (4 atoms)"""
    if len(indices) == 2:
        return canonicalBond(indices[0], indices[1])
    if len(indices) == 3:
        return canonicalAngle(indices[0], indices[1], indices[2])
    if len(indices) == 4:
        return canonicalDihedral(indices[0], indices[1], indices[2], indices[3])

    raise ValueError("Expected 2, 3 or 4 atoms but got %d" % len(indices))


def wrapAngle(angle: ArrayLike) -> np.ndarray:
    """Maps the given angle(s) in radians to the interval (-pi, pi]"""
    wrapped = np.mod(np.asarray(angle, dtype=float) + np.pi, 2 * np.pi) - np.pi
    # The modulo operation yields [-pi, pi), so we still have to map -pi to +pi
    return np.where(wrapped <= -np.pi, wrapped + 2 * np.pi, wrapped)


def bondLengths(coordinates: np.ndarray, indices: Bonds) -> np.ndarray:
    """Computes the length of all given bonds"""
    idx = np.asarray(indices, dtype=int).reshape(-1, 2)
    return np.linalg.norm(coordinates[idx[:, 1]] - coordinates[idx[:, 0]], axis=1)


def bondAngles(coordinates: np.ndarray, indices: Angles) -> np.ndarray:
    """Computes all given angles in radians (in the interval [0, pi])"""
    idx = np.asarray(indices, dtype=int).reshape(-1, 3)
    u = coordinates[idx[:, 0]] - coordinates[idx[:, 1]]
    v = coordinates[idx[:, 2]] - coordinates[idx[:, 1]]

    # arctan2 is numerically much better behaved close to 0 and pi than arccos
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), np.sum(u * v, axis=1))


def dihedralAngles(coordinates: np.ndarray, indices: Dihedrals) -> np.ndarray:
    """Computes all given dihedral angles in radians (in the interval (-pi, pi])"""
    idx = np.asarray(indices, dtype=int).reshape(-1, 4)
    b1 = coordinates[idx[:, 1]] - coordinates[idx[:, 0]]
    b2 = coordinates[idx[:, 2]] - coordinates[idx[:, 1]]
    b3 = coordinates[idx[:, 3]] - coordinates[idx[:, 2]]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    y = np.linalg.norm(b2, axis=1) * np.sum(b1 * n2, axis=1)
    x = np.sum(n1 * n2, axis=1)

    return np.arctan2(y, x)


def _neighborLists(bonds: Sequence[Bond], nAtoms: int) -> List[List[int]]:
    neighbors: List[Set[int]] = [set() for _ in range(nAtoms)]
    for i, j in bonds:
        if i == j:
            raise ValueError("Atom %d cannot be bonded to itself" % i)
        neighbors[i].add(j)
        neighbors[j].add(i)

    return [sorted(n) for n in neighbors]


def findAngles(bonds: Sequence[Bond], nAtoms: int) -> List[Angle]:
    """Returns all (canonical) angles in the bond graph, i.e. all pairs of bonds sharing an atom"""
    neighbors = _neighborLists(bonds, nAtoms)

    angles: List[Angle] = []
    for center, nbrs in enumerate(neighbors):
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                angles.append(canonicalAngle(nbrs[a], center, nbrs[b]))

    return angles


def findDihedrals(bonds: Sequence[Bond], nAtoms: int) -> List[Dihedral]:
    """Returns all (canonical) dihedral angles in the bond graph, i.e. all paths of three consecutive bonds that
    involve four different atoms"""
    neighbors = _neighborLists(bonds, nAtoms)

    dihedrals: Dict[Dihedral, None] = {}
    for j in range(nAtoms):
        for k in neighbors[j]:
            if k < j:
                # Every central bond is only considered once
                continue
            for i in neighbors[j]:
                for l in neighbors[k]:
                    if i == k or l == j or i == l:
                        continue
                    dihedrals[canonicalDihedral(i, j, k, l)] = None

    return list(dihedrals)
