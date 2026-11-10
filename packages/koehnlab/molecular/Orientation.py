from typing import Optional

import numpy as np

# Relative tolerance (with respect to the largest principal moment) below which principal moments are considered
# to be equal to each other (or to zero). In these cases the principal axes are not determined by the inertia
# tensor alone and have to be fixed based on the positions of the atoms instead.
DEGENERACY_TOLERANCE: float = 1e-4

# Relative tolerance (with respect to the largest distance of an atom from the center) below which an atom is considered
# to lie on a nodal plane/axis and is therefore not suited for fixing the sign/direction of an axis.
POSITION_TOLERANCE: float = 1e-3


def _signFixed(axis: np.ndarray, coordinates: np.ndarray, tolerance: float) -> Optional[np.ndarray]:
    """Returns the given axis (or its negative) such that the first atom, whose position has a non-negligible
    component along the axis, has a positive coordinate along the axis. Returns None if there is no such atom."""
    for projection in coordinates @ axis:
        if abs(projection) > tolerance:
            return axis if projection > 0 else -axis

    return None


def _perpendicularDirection(
    axis: np.ndarray, coordinates: np.ndarray, tolerance: float
) -> Optional[np.ndarray]:
    """Returns the unit vector pointing from the given axis to the first atom that has a non-negligible distance
    from the axis. Returns None if there is no such atom."""
    for position in coordinates:
        perpendicular = position - (position @ axis) * axis
        norm = np.linalg.norm(perpendicular)
        if norm > tolerance:
            return perpendicular / norm

    return None


def _arbitraryPerpendicular(axis: np.ndarray) -> np.ndarray:
    """Returns some unit vector perpendicular to the given axis (the result is only used if the choice does not
    matter, e.g. for rotations around the axis of a linear molecule)"""
    reference = np.zeros(3)
    reference[np.argmin(np.abs(axis))] = 1.0
    perpendicular = reference - (reference @ axis) * axis
    return perpendicular / np.linalg.norm(perpendicular)


def _linearAxes(axis: np.ndarray, coordinates: np.ndarray, tolerance: float) -> np.ndarray:
    """Axes for a linear molecule: z is the molecular axis"""
    z = _signFixed(axis, coordinates, tolerance)
    z = axis if z is None else z
    x = _arbitraryPerpendicular(z)
    return np.column_stack((x, np.cross(z, x), z))


def _sphericalTopAxes(coordinates: np.ndarray, tolerance: float) -> np.ndarray:
    """Axes for a spherical top: z points to the first atom off the center and the first atom off the z axis defines the
    xz plane"""
    for position in coordinates:
        norm = np.linalg.norm(position)
        if norm > tolerance:
            z = position / norm
            break
    else:
        return np.eye(3)

    x = _perpendicularDirection(z, coordinates, tolerance)
    x = _arbitraryPerpendicular(z) if x is None else x
    return np.column_stack((x, np.cross(z, x), z))


def _symmetricTopAxes(
    uniqueAxis: np.ndarray, coordinates: np.ndarray, tolerance: float
) -> np.ndarray:
    """Axes for a symmetric top: z is the unique axis and the first atom off the z axis defines the xz plane"""
    z = _signFixed(uniqueAxis, coordinates, tolerance)

    if z is not None:
        x = _perpendicularDirection(z, coordinates, tolerance)
        x = _arbitraryPerpendicular(z) if x is None else x
        return np.column_stack((x, np.cross(z, x), z))

    # All atoms lie in the plane perpendicular to the unique axis (e.g. benzene) so the direction of z can't be
    # fixed via atom positions. Instead, we fix the direction of y (which determines z via the right-hand rule)
    x = _perpendicularDirection(uniqueAxis, coordinates, tolerance)
    if x is None:
        return np.eye(3)
    y = np.cross(uniqueAxis, x)
    y = _signFixed(y, coordinates, tolerance)
    y = np.cross(uniqueAxis, x) if y is None else y
    return np.column_stack((x, y, np.cross(x, y)))


def _asymmetricTopAxes(
    eigenvectors: np.ndarray, coordinates: np.ndarray, tolerance: float
) -> np.ndarray:
    """Axes for an asymmetric top: x, y, z are the axes with increasing moment of inertia. The signs of x and y are
    fixed by the atom positions and z follows from the right-hand rule"""
    x = _signFixed(eigenvectors[:, 0], coordinates, tolerance)
    y = _signFixed(eigenvectors[:, 1], coordinates, tolerance)
    x = eigenvectors[:, 0] if x is None else x
    y = eigenvectors[:, 1] if y is None else y
    return np.column_stack((x, y, np.cross(x, y)))


def principalAxes(coordinates: np.ndarray, inertiaTensor: np.ndarray) -> np.ndarray:
    """Determines a unique right-handed set of axes for the molecule with the given coordinates (relative to the origin
    that was used to compute the inertia tensor) and the given inertia tensor.

    The result is a 3x3 matrix whose columns are the new x, y, z axes, so the rotated coordinates are
    given by coordinates @ axes. Unlike the eigenvectors of the inertia tensor, the axes are deterministic
    even if the principal moments are degenerate and never involve a reflection. The orientation is
    only unique if the molecule's atoms are in a fixed order, as the atom order is used to resolve ambiguities:
    - asymmetric top: x, y, z are the axes of increasing moment of inertia, signs are chosen such that the first atom
      that is not on the nodal plane of an axis has a positive coordinate along it
    - symmetric top: z is the unique axis, the first atom off the z axis is placed in the xz plane
    - spherical top: z points to the first atom off the center, the first atom off the z axis is placed in the xz plane
    - linear molecule: z is the molecular axis
    """
    assert coordinates.ndim == 2 and coordinates.shape[1] == 3
    assert inertiaTensor.shape == (3, 3)

    if len(coordinates) == 0:
        return np.eye(3)

    size = np.max(np.linalg.norm(coordinates, axis=1))
    if size < 1e-12:
        # All atoms are located at the origin
        return np.eye(3)

    eigenvalues, eigenvectors = np.linalg.eigh(inertiaTensor)

    degeneracyTolerance = DEGENERACY_TOLERANCE * eigenvalues[2]
    positionTolerance = POSITION_TOLERANCE * size

    if eigenvalues[0] <= degeneracyTolerance:
        axes = _linearAxes(eigenvectors[:, 0], coordinates, positionTolerance)
    elif eigenvalues[2] - eigenvalues[0] <= degeneracyTolerance:
        axes = _sphericalTopAxes(coordinates, positionTolerance)
    elif eigenvalues[1] - eigenvalues[0] <= degeneracyTolerance:
        axes = _symmetricTopAxes(eigenvectors[:, 2], coordinates, positionTolerance)
    elif eigenvalues[2] - eigenvalues[1] <= degeneracyTolerance:
        axes = _symmetricTopAxes(eigenvectors[:, 0], coordinates, positionTolerance)
    else:
        axes = _asymmetricTopAxes(eigenvectors, coordinates, positionTolerance)

    assert abs(np.linalg.det(axes) - 1) < 1e-8, "Orientation must be a proper rotation"

    return axes
