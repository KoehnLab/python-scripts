from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field

import math
import re

from .Element import Element
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


@dataclass
class StructureSpec:
    """A set of structure parameters for a molecule, whose atoms are identified by unique labels (e.g. "C1")"""

    labels: List[str] = field(default_factory=list)
    elements: List[Element] = field(default_factory=list)
    # Connectivity as a sorted list of canonical bonds
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)

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
_WEIGHT = re.compile(r"(?<!\S)w\s*=\s*(\S+)")

# Parameters that differ by less than this are considered to be identical
_DUPLICATE_TOLERANCE = 1e-8


def parseStructureSpec(path: str) -> StructureSpec:
    """Reads a structure specification from the given file. See parseStructureSpecString for the format"""
    with open(path, "r") as inputFile:
        return parseStructureSpecString(inputFile.read(), path)


def parseStructureSpecString(text: str, source: str = "<string>") -> StructureSpec:
    """Parses a structure specification: One parameter per line, atoms are identified by labels consisting of an element
    symbol and a number (e.g. C1 or Cl2). Text after a '#' is ignored. The number of labels determines the type:

    C1 - C2 1.38              bond length in Angstrom (also declares that the atoms are bonded)
    C1 - H1                   no value: only declares that the atoms are bonded
    C1 - C2 - C3 122.0        angle in degrees (the atom in the middle is the apex)
    C1 - C2 - C3 - C4 180.0   dihedral angle in degrees

    Each line may end with 'w=<weight>' to weight the parameter relative to others (default: 1).
    The atoms are ordered by their first appearance.
    """
    spec = StructureSpec()
    labelIndex: Dict[str, int] = {}
    seen: Dict[Tuple[int, ...], Tuple[float, int]] = {}

    def fail(lineNumber: int, message: str) -> StructureSpecError:
        return StructureSpecError("%s:%d: %s" % (source, lineNumber, message))

    def atomIndex(label: str, lineNumber: int) -> int:
        if label in labelIndex:
            return labelIndex[label]

        symbol = _LABEL_SPLIT.fullmatch(label).group(1)  # type: ignore[union-attr]
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

    if spec.nAtoms() == 0:
        raise StructureSpecError("%s: The specification does not contain any atoms" % source)

    return spec
