from typing import Callable, Dict, List, Tuple

import math
import re

import numpy as np

# Generators of point groups in a canonical frame: The principal axis is z, the C2 axes perpendicular to it are along x (and
# rotated copies of it), the vertical mirror plane is the xz plane (normal along y) and the horizontal mirror plane is the xy plane
# (normal along z). For the cubic groups the axes x, y, z are the C2 axes (or C4 axes) and (1,1,1) is a C3 axis.

Generator = Tuple[str, np.ndarray]


def rotationMatrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Counterclockwise rotation by the given angle (radians) about the given unit vector"""
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return (
        math.cos(angle) * np.eye(3)
        + math.sin(angle) * cross
        + (1 - math.cos(angle)) * np.outer(axis, axis)
    )


def reflectionMatrix(normal: np.ndarray) -> np.ndarray:
    """Reflection at the plane through the origin with the given unit normal vector"""
    return np.eye(3) - 2 * np.outer(normal, normal)


def _unit(*components: float) -> np.ndarray:
    vector = np.array(components, dtype=float)
    return vector / np.linalg.norm(vector)


_X, _Y, _Z = np.eye(3)


def _C(n: int, axis: np.ndarray, label: str = "z") -> Generator:
    return ("C%d(%s)" % (n, label), rotationMatrix(axis, 2 * math.pi / n))


def _sigma(normal: np.ndarray, label: str) -> Generator:
    return ("sigma(%s)" % label, reflectionMatrix(normal))


_INVERSION: Generator = ("i", -np.eye(3))
_GOLDEN_RATIO = (1 + math.sqrt(5)) / 2


def _cubic(name: str) -> List[Generator]:
    body = _unit(1, 1, 1)
    tetrahedral = [_C(2, _Z), _C(3, body, "1,1,1")]
    octahedral = [_C(4, _Z), _C(3, body, "1,1,1")]
    icosahedral = tetrahedral + [_C(5, _unit(0, 1, _GOLDEN_RATIO), "0,1,phi")]
    groups: Dict[str, Callable[[], List[Generator]]] = {
        "T": lambda: tetrahedral,
        "Td": lambda: tetrahedral + [_sigma(_unit(1, -1, 0), "1,-1,0")],
        "Th": lambda: tetrahedral + [_INVERSION],
        "O": lambda: octahedral,
        "Oh": lambda: octahedral + [_INVERSION],
        "I": lambda: icosahedral,
        "Ih": lambda: icosahedral + [_INVERSION],
    }
    return groups[name]()


def pointGroupGenerators(name: str) -> List[Generator]:
    """Returns generators (name and matrix) of the point group with the given Schoenflies symbol. Supported are Cs, Ci, Cn, Cnv,
    Cnh, Dn, Dnh, Dnd, S2n, T, Td, Th, O, Oh, I and Ih (with n being a number). Raises a ValueError for unknown groups."""
    if name == "Cs":
        return [_sigma(_Z, "z")]
    if name == "Ci":
        return [_INVERSION]
    if name in ("T", "Td", "Th", "O", "Oh", "I", "Ih"):
        return _cubic(name)

    match = re.fullmatch(r"([CDS])(\d+)([vhd]?)", name)
    if match is not None:
        family, n, suffix = match.group(1), int(match.group(2)), match.group(3)

        if family == "C" and n >= 1 and suffix == "":
            return [_C(n, _Z)] if n > 1 else []
        if family == "C" and n >= 2 and suffix == "v":
            return [_C(n, _Z), _sigma(_Y, "y")]
        if family == "C" and n >= 2 and suffix == "h":
            return [_C(n, _Z), _sigma(_Z, "z")]
        if family == "D" and n >= 2 and suffix == "":
            return [_C(n, _Z), _C(2, _X, "x")]
        if family == "D" and n >= 2 and suffix == "h":
            return [_C(n, _Z), _C(2, _X, "x"), _sigma(_Z, "z")]
        if family == "D" and n >= 2 and suffix == "d":
            # The dihedral mirror planes contain the z axis and bisect the C2 axes
            normal = _unit(-math.sin(math.pi / (2 * n)), math.cos(math.pi / (2 * n)), 0)
            return [_C(n, _Z), _C(2, _X, "x"), ("sigma(dihedral)", reflectionMatrix(normal))]
        if family == "S" and n >= 4 and n % 2 == 0 and suffix == "":
            return [
                (
                    "S%d(z)" % n,
                    reflectionMatrix(_Z) @ rotationMatrix(_Z, 2 * math.pi / n),
                )
            ]

    raise ValueError(
        "Unknown point group '%s' (supported: Cs, Ci, Cn, Cnv, Cnh, Dn, Dnh, Dnd, S2n, T, Td, Th, O, Oh, I, Ih)" % name
    )
