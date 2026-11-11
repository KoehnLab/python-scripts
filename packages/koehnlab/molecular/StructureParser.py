from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field

import math
import re

import numpy as np

from .Element import Element
from .PointGroups import adaptedPointGroupGenerators, pointGroupGenerators, reflectionMatrix, rotationMatrix
from .InternalCoordinates import (
    canonicalAngle,
    canonicalBond,
    canonicalDihedral,
    wrapAngle,
)


class StructureSpecError(ValueError):
    """Raised for invalid structure specifications"""


@dataclass(frozen=True)
class Constraint:
    """A single structure parameter: a bond length (2 atoms, in Angstrom), a bond angle (3 atoms, in radians) or a
    dihedral angle (4 atoms, in radians). The atoms are given by their (canonical) indices"""

    indices: Tuple[int, ...]
    value: float
    weight: float = 1.0

    def kind(self) -> str:
        return {2: "bond", 3: "angle", 4: "dihedral"}[len(self.indices)]


@dataclass(eq=False)
class SymmetryOperation:
    """A symmetry operation of the molecule: A 3x3 orthogonal matrix, acting on the position vectors, together with the
    permutation of the atoms it causes. The operation moves the position of every atom in a cycle to the position of the next
    atom in the cycle (the last one to the first). Atoms that don't appear in any cycle stay where they are."""

    name: str
    matrix: np.ndarray
    cycles: Tuple[Tuple[int, ...], ...]
    # Atoms that are explicitly stated to stay where they are
    fixed: Tuple[int, ...] = ()
    # Whether only some of the atoms are given and the remaining ones have to be inferred from the bonds
    partial: bool = False
    # The cycles that were given as chains ('A->B->C'). Unless the operation is partial, they are closed (C moves to A),
    # otherwise they only state where each atom of the chain moves to
    chains: Tuple[Tuple[int, ...], ...] = ()


@dataclass
class StructureSpec:
    """A set of structure parameters for a molecule, whose atoms are identified by unique labels (e.g. "C1")"""

    labels: List[str] = field(default_factory=list)
    elements: List[Element] = field(default_factory=list)
    # Connectivity as a sorted list of canonical bonds
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    # Dimensionality of the molecule: "general" (3D), "planar" (all atoms in one plane) or "linear" (all atoms on one line)
    geometry: str = "general"
    # Generators of the symmetry group of the molecule (all operations are defined in the same frame, whose origin is a point
    # that is invariant under all operations)
    symmetry: List[SymmetryOperation] = field(default_factory=list)
    # Point groups whose generators have been added to the symmetry operations (as partial operations)
    groups: List[str] = field(default_factory=list)

    def nAtoms(self) -> int:
        return len(self.labels)

    def indexOf(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError("Unknown atom label '%s'" % label)


_LABEL = r"[A-Za-z]+\d+"
_LABEL_CHAIN = re.compile(r"^\s*(%s(?:\s*-\s*%s)*)(.*)$" % (_LABEL, _LABEL))
_LABEL_SPLIT = re.compile(r"([A-Za-z]+)(\d+)")
_SYMMETRY = re.compile(r"^@symmetry\s+([^:]+?)\s*:(.*)$")
_GROUP = re.compile(r"^@group\s+(\w+)$")
_DIRECTIVE = re.compile(r"^@(\w+)$")
GEOMETRIES = {"planar": "planar", "linear": "linear"}
_WEIGHT = re.compile(r"(?<!\S)w\s*=\s*(\S+)")

# Parameters that differ by less than this are considered to be identical
_DUPLICATE_TOLERANCE = 1e-8


def _parseAxis(text: str) -> np.ndarray:
    """Parses an axis given as x, y, z or as vector a,b,c"""
    text = text.strip()
    if text in ("x", "y", "z"):
        return np.eye(3)["xyz".index(text)]

    try:
        axis = np.array([float(component) for component in text.split(",")])
    except ValueError:
        axis = np.zeros(0)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-8:
        raise ValueError("Invalid axis '%s' (expected x, y, z or three comma-separated numbers)" % text)

    return axis / np.linalg.norm(axis)


def parseSymmetryOperation(text: str) -> np.ndarray:
    """Returns the matrix of a symmetry operation given by its name: i (inversion), sigma(axis) (mirror plane perpendicular
    to the axis), Cn(axis) (rotation by 360/n degrees about the axis, counterclockwise when looking against the axis) and
    Sn(axis) (rotation by 360/n degrees followed by a reflection at the plane perpendicular to the axis). The axis is x, y, z or
    a vector a,b,c."""
    text = text.strip()
    if text == "i":
        return -np.eye(3)

    match = re.fullmatch(r"(sigma|[CS]\d+)\((.*)\)", text)
    if match is None:
        raise ValueError(
            "Unknown symmetry operation '%s' (expected i, sigma(axis), Cn(axis) or Sn(axis))" % text
        )

    kind, axis = match.group(1), _parseAxis(match.group(2))
    if kind == "sigma":
        return reflectionMatrix(axis)

    order = int(kind[1:])
    if order < 1:
        raise ValueError("The order of '%s' must be positive" % text)
    rotation = rotationMatrix(axis, 2 * math.pi / order)

    return rotation if kind[0] == "C" else reflectionMatrix(axis) @ rotation


def parseStructureSpec(path: str, geometry: Optional[str] = None) -> StructureSpec:
    """Reads a structure specification from the given file. See parseStructureSpecString for the format"""
    with open(path, "r") as inputFile:
        return parseStructureSpecString(inputFile.read(), path, geometry)


def parseStructureSpecString(text: str, source: str = "<string>", geometry: Optional[str] = None) -> StructureSpec:
    """Parses a structure specification: One parameter per line, atoms are identified by labels consisting of an element
    symbol and a number (e.g. C1 or Cl2). Text after a '#' is ignored. The number of labels determines the type:

    C1 - C2 1.38              bond length in Angstrom (also declares that the atoms are bonded)
    C1 - H1                   no value: only declares that the atoms are bonded
    C1 - C2 - C3 122.0        angle in degrees (the atom in the middle is the apex)
    C1 - C2 - C3 - C4 180.0   dihedral angle in degrees

    Each line may end with 'w=<weight>' to weight the parameter relative to others (default: 1).
    The atoms are ordered by their first appearance.

    The directives '@planar' and '@linear' state that all atoms of the molecule are located in one plane or on one line.

    The line '@symmetry <operation>: <atoms>' declares a symmetry operation of the molecule, e.g.
    '@symmetry C2(z): C1<->C3, C2<->C4' (see parseSymmetryOperation for the operations). The atoms are given by exchanges
    ('C1<->C3') or cycles ('C1->C2->C3': the operation moves C1 to the position of C2, C2 to the position of C3 and C3 to the
    position of C1). Atoms that are not listed are not moved by the operation, i.e. they are located on the symmetry element.
    All operations refer to the same coordinate frame and its origin must be on every symmetry element.

    The atoms of an operation can be incomplete if the list ends with '...', e.g. '@symmetry C2(z): C1<->C4, ...'. The
    remaining atoms are then inferred from the bonds (a chain like 'C1->C2' then only states that C1 moves to C2, not that C2
    moves to C1). A single label stands for an atom that stays where it is
    ('@symmetry sigma(y): H1, ...').

    The line '@group C2v' adds the generators of the point group (in a canonical frame that is rotated to fit the other operations
    and the mirror plane of a planar molecule, see adaptedPointGroupGenerators) as operations whose atoms are inferred. The atoms of such an operation can be given by a '@symmetry' line with the same operation (it
    is then treated as incomplete list).

    The geometry ('planar' or 'linear') can also be given as an argument, which is the same as the directive.
    """
    if geometry is not None and geometry not in GEOMETRIES.values():
        raise ValueError("Unknown geometry '%s' (expected one of: %s)" % (geometry, ", ".join(GEOMETRIES.values())))
    spec = StructureSpec(geometry=geometry or "general")
    labelIndex: Dict[str, int] = {}
    seen: Dict[Tuple[int, ...], Tuple[float, int]] = {}

    def fail(lineNumber: int, message: str) -> StructureSpecError:
        return StructureSpecError("%s:%d: %s" % (source, lineNumber, message))

    def atomIndex(label: str, lineNumber: int) -> int:
        if label in labelIndex:
            return labelIndex[label]

        labelMatch = _LABEL_SPLIT.fullmatch(label)
        assert labelMatch is not None, "labels are validated by the line pattern"
        symbol = labelMatch.group(1)
        try:
            element = Element[symbol]
        except KeyError:
            raise fail(lineNumber, "Unknown element '%s' in label '%s'" % (symbol, label))

        labelIndex[label] = len(spec.labels)
        spec.labels.append(label)
        spec.elements.append(element)
        return labelIndex[label]

    bonds = set()

    for lineNumber, rawLine in enumerate(text.splitlines(), start=1):
        line = rawLine.split("#", 1)[0].strip()
        if not line:
            continue

        symmetry = _SYMMETRY.match(line)
        if symmetry is not None:
            try:
                matrix = parseSymmetryOperation(symmetry.group(1))
            except ValueError as error:
                raise fail(lineNumber, str(error))

            cycles: List[Tuple[int, ...]] = []
            chains: List[Tuple[int, ...]] = []
            fixed: List[int] = []
            partial = False
            used: Dict[str, None] = {}
            for token in symmetry.group(2).split(","):
                token = token.strip()
                if not token:
                    continue
                if token == "...":
                    partial = True
                    continue

                if "<->" in token:
                    labels = [label.strip() for label in token.split("<->")]
                    if len(labels) != 2:
                        raise fail(lineNumber, "'<->' exchanges exactly two atoms: '%s'" % token)
                else:
                    labels = [label.strip() for label in token.split("->")]

                for label in labels:
                    if re.fullmatch(_LABEL, label) is None:
                        raise fail(lineNumber, "Expected an atom label but found '%s'" % label)
                    if label in used:
                        raise fail(lineNumber, "Atom %s appears more than once in the symmetry operation" % label)
                    used[label] = None

                indices = tuple(atomIndex(label, lineNumber) for label in labels)
                if len(indices) > 1:
                    cycles.append(indices)
                    if "<->" not in token:
                        chains.append(indices)
                else:
                    fixed.extend(indices)

            spec.symmetry.append(
                SymmetryOperation(
                    symmetry.group(1).strip(),
                    matrix,
                    tuple(cycles),
                    tuple(fixed),
                    partial,
                    tuple(chains),
                )
            )
            continue

        group = _GROUP.match(line)
        if group is not None:
            try:
                pointGroupGenerators(group.group(1))
            except ValueError as error:
                raise fail(lineNumber, str(error))
            spec.groups.append(group.group(1))
            continue

        directive = _DIRECTIVE.match(line)
        if directive is not None:
            name = directive.group(1)
            if name not in GEOMETRIES:
                raise fail(
                    lineNumber,
                    "Unknown directive '@%s' (expected one of: %s)"
                    % (name, ", ".join("@" + g for g in GEOMETRIES)),
                )
            if spec.geometry != "general" and spec.geometry != GEOMETRIES[name]:
                raise fail(lineNumber, "Conflicting directives '@planar' and '@linear'")
            spec.geometry = GEOMETRIES[name]
            continue

        match = _LABEL_CHAIN.match(line)
        if match is None:
            raise fail(lineNumber, "Expected atom labels like 'C1 - C2' but found '%s'" % line)

        labels = [label.strip() for label in match.group(1).split("-")]
        remainder = match.group(2)

        if len(labels) > 4:
            raise fail(lineNumber, "At most 4 atoms (dihedral angle) are supported")
        if len(set(labels)) != len(labels):
            raise fail(lineNumber, "An atom appears more than once in '%s'" % match.group(1))

        def toNumber(token: str) -> float:
            try:
                return float(token)
            except ValueError:
                raise fail(lineNumber, "Invalid number '%s'" % token)

        weight = 1.0
        weightMatch = _WEIGHT.search(remainder)
        if weightMatch:
            weight = toNumber(weightMatch.group(1))
            remainder = remainder[: weightMatch.start()] + remainder[weightMatch.end() :]

        value: Optional[float] = None
        tokens = remainder.split()
        if len(tokens) > 1:
            raise fail(lineNumber, "Unexpected '%s'" % " ".join(tokens[1:]))
        if tokens:
            value = toNumber(tokens[0])

        if not math.isfinite(weight) or weight <= 0:
            raise fail(lineNumber, "The weight must be positive")
        if value is not None and not math.isfinite(value):
            raise fail(lineNumber, "The value must be finite")

        indices = tuple(atomIndex(label, lineNumber) for label in labels)

        if len(indices) == 2:
            bonds.add(canonicalBond(*indices))
            if value is None:
                continue
            if value <= 0:
                raise fail(lineNumber, "Bond lengths must be positive")
            key: Tuple[int, ...] = canonicalBond(*indices)
        elif len(indices) == 3:
            if value is None:
                raise fail(lineNumber, "Missing value for the angle")
            if not 0 <= value <= 180:
                raise fail(lineNumber, "Angles must be between 0 and 180 degrees")
            value = math.radians(value)
            key = canonicalAngle(*indices)
        else:
            if value is None:
                raise fail(lineNumber, "Missing value for the dihedral angle")
            value = float(wrapAngle(math.radians(value)))
            key = canonicalDihedral(*indices)

        assert value is not None

        if key in seen:
            previous, previousLine = seen[key]
            difference = value - previous
            if len(key) == 4:
                difference = float(wrapAngle(difference))
            if abs(difference) > _DUPLICATE_TOLERANCE:
                raise fail(
                    lineNumber,
                    "Conflicts with the value specified in line %d for the same parameter"
                    % previousLine,
                )
            # Identical repetition, nothing to do
            continue

        seen[key] = (value, lineNumber)
        spec.constraints.append(Constraint(key, value, weight))

    spec.bonds = sorted(bonds)

    # The generators of point groups are operations, whose atoms have to be inferred, except for those atoms the user gave
    # explicitly for an operation with the same matrix
    # The frame of the group adapts to the operations the user gave (C2(x) in C2v makes x the principal axis)
    userOperations = [op.matrix for op in spec.symmetry]
    if spec.geometry == "planar":
        # The mirror plane of the molecule is the xy plane and must be an operation of the group
        userOperations.append(np.diag([1.0, 1.0, -1.0]))
    for groupName in spec.groups:
        for name, matrix in adaptedPointGroupGenerators(groupName, userOperations):
            existing = [op for op in spec.symmetry if np.allclose(op.matrix, matrix, atol=1e-8)]
            if existing:
                for operation in existing:
                    operation.partial = True
            else:
                spec.symmetry.append(SymmetryOperation(name, matrix, (), (), True))

    if spec.nAtoms() == 0:
        raise StructureSpecError("%s: The specification does not contain any atoms" % source)

    return spec
