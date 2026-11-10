from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from collections import deque
from dataclasses import dataclass, field

import math
import warnings

import numpy as np
from ase.data import covalent_radii, vdw_radii
from scipy.optimize import least_squares

from .Atom import Atom
from .IO import generateLabels
from .InternalCoordinates import (
    bondAngles,
    bondLengths,
    canonicalAngle,
    canonicalBond,
    canonicalDihedral,
    dihedralAngles,
    findAngles,
    findDihedrals,
    wrapAngle,
)
from .Molecule import Molecule
from .SymmetryInference import completeSymmetry
from .StructureParser import (
    Constraint,
    StructureSpec,
    StructureSpecError,
    SymmetryOperation,
)

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


def _ringBonds(spec: StructureSpec, neighbors: List[List[int]]) -> Set[Tuple[int, int]]:
    """The bonds that are part of a ring: Those whose atoms remain connected if the bond is removed"""
    ringBonds: Set[Tuple[int, int]] = set()
    for i, j in spec.bonds:
        reached = {i}
        stack = [i]
        while stack and j not in reached:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if canonicalBond(current, neighbor) != (i, j) and neighbor not in reached:
                    reached.add(neighbor)
                    stack.append(neighbor)
        if j in reached:
            ringBonds.add((i, j))

    return ringBonds


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
    planar: bool = False,
) -> np.ndarray:
    """Computes the position of a new atom that is bonded to center with the given distance. angle is the angle between
    the new atom, the center and the axisAtom. torsion is the dihedral angle new atom - center - axisAtom - reference
    (if there is no reference, a deterministic arbitrary direction is used for torsion=0; if planar is set, this is a
    direction in the xy plane)."""
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
        if planar:
            e1 = np.cross(np.array([0.0, 0.0, 1.0]), axis)
            e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(e1, axis)

    direction = math.cos(angle) * axis + math.sin(angle) * (
        math.cos(torsion) * e1 + math.sin(torsion) * e2
    )

    return origin + distance * direction


def defaultGuess(spec: StructureSpec, geometry: Optional[str] = None) -> np.ndarray:
    """Generates a rough initial geometry (in Angstrom) for the given specification solely from the connectivity.

    The atoms are placed one after another along a spanning tree of the bond graph. For every atom, bond length, angle
    and torsion are taken from the specification, if it contains the corresponding parameter. Otherwise defaults are used:
    The sum of the covalent radii as bond length, an angle that only depends on the number of neighbors of the central atom
    (120 degrees for 2 or 3, 109.47 degrees for 4 and 90 degrees for more neighbors) and torsions that distribute the
    substituents of an atom evenly (staggered/planar). Along rings, all torsions are cis (0 degrees), so that the ring bends
    in a consistent direction and (for suitable angles) closes, and substituents of ring atoms point away from the ring.
    The bond that finally closes a ring is not used for placing atoms, so rings are in general not exactly closed in the result.
    Atoms are in the same order as in the specification.

    For geometry="planar" (default: the geometry of the specification) all atoms are placed in the xy plane, i.e. all torsions
    are 0 or 180 degrees (alternating, unless the specification contains a dihedral). For geometry="linear", all atoms are
    placed on the z axis.
    """
    geometry = spec.geometry if geometry is None else geometry
    planar = geometry == "planar"
    linear = geometry == "linear"
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

    ringBonds = _ringBonds(spec, neighbors)

    def isRingBond(i: int, j: int) -> bool:
        return canonicalBond(i, j) in ringBonds

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
        # (neighbors along a ring first, since the ring is what the torsions should follow)
        others = [n for n in neighbors[center] if placed[n] and n != atom]
        others.sort(key=lambda n: not isRingBond(n, center))

        if not others:
            direction = np.array([1.0, 0.0, 0.0] if planar else [0.0, 0.0, 1.0])
            coordinates[atom] = coordinates[center] + distance * direction
            placed[atom] = True
            continue

        def references(axisAtom: int) -> List[int]:
            """Atoms, bonded to axisAtom, which can be used to measure the torsion"""
            found = [
                n
                for n in neighbors[axisAtom]
                if placed[n] and n != center and n != atom
            ]
            return sorted(found, key=lambda n: not isRingBond(n, axisAtom))

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
        if linear:
            angle = math.pi

        key = (
            canonicalDihedral(atom, center, axisAtom, reference)
            if reference is not None
            else None
        )
        if key is not None and key in dihedralValues:
            torsion = dihedralValues[key]
            if planar:
                torsion = round(torsion / math.pi) * math.pi
        else:
            step = 2 * math.pi / (len(neighbors[center]) - 1)
            children = [n for n in others if n != axisAtom]

            if (
                reference is not None
                and isRingBond(center, axisAtom)
                and isRingBond(axisAtom, reference)
            ):
                # Following a ring: The first ring bond continues the ring in the same direction (cis), everything else is
                # distributed around the axis (for a center with three neighbors: away from the ring)
                hasRingChild = any(isRingBond(n, center) for n in children)
                if isRingBond(atom, center) and not hasRingChild:
                    torsion = 0.0
                else:
                    torsion = step * (1 + len(children) - (1 if hasRingChild else 0))
            elif planar:
                torsion = (math.pi if reference is not None else 0.0) + (len(others) - 1) * math.pi
            else:
                # Distribute the substituents of the center evenly around the axis
                torsion = (math.pi if reference is not None else 0.0) + (len(others) - 1) * step

            if planar:
                torsion = round(torsion / math.pi) * math.pi

        coordinates[atom] = _placeAtom(
            coordinates, center, axisAtom, reference, distance, angle, torsion, planar
        )
        placed[atom] = True

    return coordinates


# Two atoms of a given starting geometry are considered to be bonded if their distance is less than this factor times
# the sum of their covalent radii
_BOND_DETECTION_FACTOR = 1.25

# Scales of the residuals: A deviation of one scale unit counts as much as the other one
_LENGTH_SCALE = 1e-3  # Angstrom
_ANGLE_SCALE = math.radians(0.1)

# Relative weights of the weak restraints that keep all unspecified bonds/angles/torsions of the starting geometry
# and the weak repulsion between non-bonded atoms, compared to the specified parameters (weight 1)
_RESTRAINT_WEIGHT = 1e-3
_CLASH_WEIGHT = 1e-2
# Non-bonded atoms (separated by at least three bonds) closer than this factor times the sum of van der Waals radii repel
_CLASH_FACTOR = 0.7

# Maximum deviations of bond lengths (Angstrom) and angles (radians) for a structure to count as converged
_LENGTH_TOLERANCE = 1e-4
_ANGLE_TOLERANCE = math.radians(0.01)


@dataclass
class ParameterReport:
    """Comparison of a specified parameter with the value found in the final structure"""

    labels: Tuple[str, ...]
    target: float
    achieved: float
    # Same units as target/achieved: Angstrom for bonds and degrees for angles and dihedrals
    deviation: float
    isLength: bool

    def isSatisfied(self) -> bool:
        tolerance = _LENGTH_TOLERANCE if self.isLength else math.degrees(_ANGLE_TOLERANCE)
        return abs(self.deviation) <= tolerance


@dataclass
class BuildResult:
    """Result of buildStructure"""

    molecule: Molecule
    labels: List[str]
    converged: bool
    attempts: int
    parameters: List[ParameterReport] = field(default_factory=list)

    def format(self) -> str:
        """Human-readable table of all specified parameters and how well they are fulfilled"""
        lines = ["%-28s %12s %12s %12s" % ("parameter", "target", "achieved", "deviation")]
        for parameter in self.parameters:
            lines.append(
                "%-28s %12.6f %12.6f %12.2e%s"
                % (
                    " - ".join(parameter.labels),
                    parameter.target,
                    parameter.achieved,
                    parameter.deviation,
                    "" if parameter.isSatisfied() else "  <-- not satisfied",
                )
            )
        return "\n".join(lines)


def resolveSpec(
    spec: StructureSpec, guess: Molecule, guessLabels: Optional[Sequence[str]] = None
) -> StructureSpec:
    """Maps the specification onto the atoms of a starting geometry: The result contains all atoms of the geometry (in
    its order), the constraints of the specification and as connectivity the union of the bonds of the specification
    and the bonds detected from the geometry (via covalent radii). If no labels are given for the geometry, per-element
    counters are used (C1, C2, ..., H1, ...)."""
    elements = [atom.element for atom in guess.atoms]
    labels = list(guessLabels) if guessLabels is not None else generateLabels(elements)

    if len(labels) != guess.nAtoms() or len(set(labels)) != len(labels):
        raise StructureSpecError("The labels of the starting geometry must be unique and one per atom")

    index = {label: i for i, label in enumerate(labels)}
    mapping: List[int] = []
    for label, element in zip(spec.labels, spec.elements):
        if label not in index:
            raise StructureSpecError("Atom '%s' is not part of the starting geometry" % label)
        if elements[index[label]] != element:
            raise StructureSpecError(
                "Atom '%s' is %s in the specification but %s in the starting geometry"
                % (label, element.symbol(), elements[index[label]].symbol())
            )
        mapping.append(index[label])

    coordinates = guess.coordinates()
    radii = np.array([covalent_radii[element.atomicNumber()] for element in elements])
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    bonded = distances < _BOND_DETECTION_FACTOR * (radii[:, None] + radii[None, :])

    bonds = {
        (i, j) for i in range(len(labels)) for j in range(i + 1, len(labels)) if bonded[i, j]
    }
    bonds.update(canonicalBond(mapping[i], mapping[j]) for i, j in spec.bonds)

    constraints: List[Constraint] = []
    for constraint in spec.constraints:
        indices = tuple(mapping[i] for i in constraint.indices)
        canonical = {2: canonicalBond, 3: canonicalAngle, 4: canonicalDihedral}[len(indices)](*indices)  # type: ignore[arg-type]
        constraints.append(Constraint(tuple(canonical), constraint.value, constraint.weight))

    symmetry = [
        SymmetryOperation(
            operation.name,
            operation.matrix,
            tuple(tuple(mapping[i] for i in cycle) for cycle in operation.cycles),
            tuple(mapping[i] for i in operation.fixed),
            operation.partial,
            tuple(tuple(mapping[i] for i in chain) for chain in operation.chains),
        )
        for operation in spec.symmetry
    ]

    return StructureSpec(labels, elements, sorted(bonds), constraints, spec.geometry, symmetry)


# Step for the numerical derivatives of the residuals
_DERIVATIVE_STEP = 1e-6


class _Block:
    """A set of residuals that each only depend on the positions of a few atoms (e.g. all bond lengths). evaluate maps the
    positions of these atoms, an array of shape (number of residuals, number of atoms per residual, 3), to the residuals"""

    def __init__(self, indices: np.ndarray, evaluate: Callable[[np.ndarray], np.ndarray]):
        self.indices = indices
        self.evaluate = evaluate

    def size(self) -> int:
        return len(self.indices)

    def residuals(self, coordinates: np.ndarray) -> np.ndarray:
        return self.evaluate(coordinates[self.indices])

    def addJacobian(self, coordinates: np.ndarray, jacobian: np.ndarray, firstRow: int) -> None:
        """Adds the derivatives of the residuals with respect to all coordinates to the given rows of the jacobian
        (dense matrix with three columns per atom), using central differences."""
        positions = coordinates[self.indices]
        rows = firstRow + np.arange(self.size())
        step = _DERIVATIVE_STEP

        for slot in range(self.indices.shape[1]):
            for axis in range(3):
                forward = positions.copy()
                backward = positions.copy()
                forward[:, slot, axis] += step
                backward[:, slot, axis] -= step
                derivative = (self.evaluate(forward) - self.evaluate(backward)) / (2 * step)
                jacobian[rows, 3 * self.indices[:, slot] + axis] += derivative


def _viaIndices(
    function: Callable[[np.ndarray, Sequence[Tuple[int, ...]]], np.ndarray], atomsPerRow: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Turns a function of coordinates and atom indices into a function of the positions of the atoms of each row"""

    def evaluate(positions: np.ndarray) -> np.ndarray:
        rows = positions.shape[0]
        indices = np.arange(rows * atomsPerRow).reshape(rows, atomsPerRow)
        return function(positions.reshape(-1, 3), indices)  # type: ignore[arg-type]

    return evaluate


class _Problem:
    """Residuals, which are zero if all specified parameters are fulfilled (constraints), and weak terms that keep
    everything else close to a reference geometry (soft terms)"""

    def __init__(
        self,
        spec: StructureSpec,
        reference: np.ndarray,
        generators: Sequence[Tuple[np.ndarray, np.ndarray]] = (),
    ):
        self.nAtoms = spec.nAtoms()

        specified = {2: [], 3: [], 4: []}  # type: Dict[int, List[Constraint]]
        for constraint in spec.constraints:
            specified[len(constraint.indices)].append(constraint)

        def arrays(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            items = specified[n]
            return (
                np.array([c.indices for c in items], dtype=int).reshape(-1, n),
                np.array([c.value for c in items], dtype=float),
                np.array([c.weight for c in items], dtype=float),
            )

        bonds, bondTargets, bondWeights = arrays(2)
        angles, angleTargets, angleWeights = arrays(3)
        dihedrals, dihedralTargets, dihedralWeights = arrays(4)

        self.constraints = [
            _Block(
                bonds,
                lambda P: bondWeights
                * (_viaIndices(bondLengths, 2)(P) - bondTargets)
                / _LENGTH_SCALE,
            ),
            _Block(
                angles,
                lambda P: angleWeights
                * (_viaIndices(bondAngles, 3)(P) - angleTargets)
                / _ANGLE_SCALE,
            ),
            _Block(
                dihedrals,
                lambda P: dihedralWeights
                * wrapAngle(_viaIndices(dihedralAngles, 4)(P) - dihedralTargets)
                / _ANGLE_SCALE,
            ),
        ]

        # Restraints for everything that is not specified
        def unspecified(candidates: Sequence[Tuple[int, ...]], fixed: np.ndarray) -> np.ndarray:
            fixedSet = {tuple(row) for row in fixed.tolist()}
            return np.array([c for c in candidates if c not in fixedSet], dtype=int)

        restraintBonds = unspecified(spec.bonds, bonds).reshape(-1, 2)
        restraintAngles = unspecified(findAngles(spec.bonds, self.nAtoms), angles).reshape(-1, 3)

        # The torsion about a bond is a single degree of freedom, but there are many (redundant) dihedrals around it. So the
        # restraints of all dihedrals about the same bond share the weight of a single one (otherwise, twisting a bond would
        # be penalized more than bending the structure). No restraints are needed for bonds with a specified dihedral
        specifiedBonds = {frozenset(row[1:3]) for row in dihedrals.tolist()}
        torsions = [
            d
            for d in sorted(findDihedrals(spec.bonds, self.nAtoms))
            if frozenset(d[1:3]) not in specifiedBonds
        ]
        restraintDihedrals = np.array(torsions, dtype=int).reshape(-1, 4)
        counts: Dict[frozenset, int] = {}
        for d in torsions:
            counts[frozenset(d[1:3])] = counts.get(frozenset(d[1:3]), 0) + 1
        torsionWeights = np.array([1 / math.sqrt(counts[frozenset(d[1:3])]) for d in torsions])

        bondRef = bondLengths(reference, restraintBonds)
        angleRef = bondAngles(reference, restraintAngles)
        dihedralRef = dihedralAngles(reference, restraintDihedrals)

        # Pairs of atoms that are at least three bonds apart can clash
        graphDistance = np.full((self.nAtoms, self.nAtoms), np.inf)
        np.fill_diagonal(graphDistance, 0)
        for i, j in spec.bonds:
            graphDistance[i, j] = graphDistance[j, i] = 1
        for k in range(self.nAtoms):
            graphDistance = np.minimum(
                graphDistance, graphDistance[:, k : k + 1] + graphDistance[k : k + 1, :]
            )
        vdw = np.array([vdw_radii[e.atomicNumber()] for e in spec.elements])
        vdw = np.where(np.isfinite(vdw), vdw, 2.0)
        i, j = np.triu_indices(self.nAtoms, k=1)
        keep = graphDistance[i, j] >= 3
        clashPairs = np.column_stack((i[keep], j[keep])).reshape(-1, 2)
        clashDistances = _CLASH_FACTOR * (vdw[i[keep]] + vdw[j[keep]])

        self.soft = [
            _Block(
                restraintBonds,
                lambda P: _RESTRAINT_WEIGHT
                * (_viaIndices(bondLengths, 2)(P) - bondRef)
                / _LENGTH_SCALE,
            ),
            _Block(
                restraintAngles,
                lambda P: _RESTRAINT_WEIGHT
                * (_viaIndices(bondAngles, 3)(P) - angleRef)
                / _ANGLE_SCALE,
            ),
            _Block(
                restraintDihedrals,
                lambda P: _RESTRAINT_WEIGHT
                * torsionWeights
                * wrapAngle(_viaIndices(dihedralAngles, 4)(P) - dihedralRef)
                / _ANGLE_SCALE,
            ),
            _Block(
                clashPairs,
                lambda P: _CLASH_WEIGHT
                * np.maximum(0.0, clashDistances - _viaIndices(bondLengths, 2)(P))
                / _LENGTH_SCALE,
            ),
        ]

        # Penalty for violating the symmetry (only used to find a symmetric starting point): The operation has to move
        # every atom to the position of its image
        self.symmetry: List[_Block] = []
        for permutation, matrix in generators:
            indices = np.column_stack((np.arange(self.nAtoms), permutation))
            for axis in range(3):
                self.symmetry.append(
                    _Block(
                        indices,
                        lambda P, matrix=matrix, axis=axis: (P[:, 1, axis] - (P[:, 0] @ matrix.T)[:, axis])
                        / _LENGTH_SCALE,
                    )
                )

    def _blocks(self, restraints: bool, symmetry: bool) -> List[_Block]:
        blocks = self.constraints + (self.soft if restraints else [])
        return blocks + (self.symmetry if symmetry else [])

    def residuals(self, x: np.ndarray, restraints: bool = True, symmetry: bool = False) -> np.ndarray:
        coordinates = x.reshape(-1, 3)
        return np.concatenate(
            [block.residuals(coordinates) for block in self._blocks(restraints, symmetry)]
        )

    def jacobian(self, x: np.ndarray, restraints: bool = True, symmetry: bool = False) -> np.ndarray:
        coordinates = x.reshape(-1, 3)
        blocks = self._blocks(restraints, symmetry)
        jacobian = np.zeros((sum(block.size() for block in blocks), x.size))

        row = 0
        for block in blocks:
            if block.size():
                block.addJacobian(coordinates, jacobian, row)
            row += block.size()

        return jacobian


def _report(spec: StructureSpec, coordinates: np.ndarray) -> List[ParameterReport]:
    reports: List[ParameterReport] = []
    for constraint in spec.constraints:
        indices = constraint.indices
        labels = tuple(spec.labels[i] for i in indices)
        if len(indices) == 2:
            achieved = float(bondLengths(coordinates, [indices])[0])  # type: ignore[list-item]
            deviation = achieved - constraint.value
            reports.append(ParameterReport(labels, constraint.value, achieved, deviation, True))
            continue

        if len(indices) == 3:
            achieved = float(bondAngles(coordinates, [indices])[0])  # type: ignore[list-item]
            deviation = achieved - constraint.value
        else:
            achieved = float(dihedralAngles(coordinates, [indices])[0])  # type: ignore[list-item]
            deviation = float(wrapAngle(achieved - constraint.value))

        reports.append(
            ParameterReport(
                labels,
                math.degrees(constraint.value),
                math.degrees(achieved),
                math.degrees(deviation),
                False,
            )
        )

    return reports


# Tolerance for angles that are supposed to be 0 or 180 degrees in planar/linear molecules
_FLAT_ANGLE_TOLERANCE = math.radians(1e-3)

# Root mean square distance (Angstrom) from the plane/line above which a starting geometry is considered to not be flat/linear
_FLATNESS_WARNING = 0.05


def _checkGeometry(spec: StructureSpec, geometry: str) -> None:
    """Checks that the specification can be fulfilled by a molecule of the given geometry"""
    if geometry not in ("general", "planar", "linear"):
        raise ValueError("Unknown geometry '%s'" % geometry)

    for constraint in spec.constraints:
        labels = " - ".join(spec.labels[i] for i in constraint.indices)
        kind = constraint.kind()

        if geometry == "planar" and kind == "dihedral":
            deviation = abs(constraint.value - round(constraint.value / math.pi) * math.pi)
            if deviation > _FLAT_ANGLE_TOLERANCE:
                raise StructureSpecError(
                    "Dihedral angle %s = %.4f degrees is not possible in a planar molecule (must be 0 or 180)"
                    % (labels, math.degrees(constraint.value))
                )

        if geometry == "linear":
            if kind == "dihedral":
                raise StructureSpecError(
                    "Dihedral angle %s is not defined in a linear molecule" % labels
                )
            if kind == "angle" and abs(constraint.value - math.pi) > _FLAT_ANGLE_TOLERANCE:
                raise StructureSpecError(
                    "Angle %s = %.4f degrees is not possible in a linear molecule (must be 180)"
                    % (labels, math.degrees(constraint.value))
                )

    if geometry == "linear":
        degrees = np.zeros(spec.nAtoms(), dtype=int)
        for i, j in spec.bonds:
            degrees[i] += 1
            degrees[j] += 1
        if np.any(degrees > 2) or len(spec.bonds) != spec.nAtoms() - 1:
            raise StructureSpecError(
                "A linear molecule must be a chain of atoms: every atom can have at most two bonds and there can be no rings"
            )


# Cartesian axes in which the atoms of the different geometries are located
_AXES: Dict[str, List[int]] = {"general": [0, 1, 2], "planar": [0, 1], "linear": [2]}


def _flatten(coordinates: np.ndarray, geometry: str) -> np.ndarray:
    """Projects the coordinates onto their best-fit plane (placed in the xy plane) or line (placed on the z axis)"""
    axes = _AXES[geometry]
    if len(axes) == 3:
        return coordinates

    centered = coordinates - coordinates.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=True)
    components = centered @ directions.T

    # Keep the most extended directions, they are the plane/line
    projected = np.zeros_like(coordinates)
    projected[:, axes] = components[:, : len(axes)]

    discarded = components[:, len(axes) :]
    if np.sqrt(np.mean(np.sum(discarded**2, axis=1))) > _FLATNESS_WARNING:
        warnings.warn(
            "The starting geometry is far from %s, it is projected onto the best-fit %s"
            % (
                "planar" if geometry == "planar" else "linear",
                "plane" if geometry == "planar" else "line",
            ),
            stacklevel=3,
        )

    return projected


# Maximum number of elements of the symmetry group (the group is generated by the user-provided operations)
_MAX_GROUP_SIZE = 192

# Root mean square displacement (Angstrom) of the atoms of a starting geometry by the projection onto the symmetric
# subspace, above which a warning is issued
_SYMMETRY_WARNING = 0.05

Operation = Tuple[np.ndarray, np.ndarray]  # (permutation of the atoms, matrix)


def _permutation(operation: SymmetryOperation, nAtoms: int) -> np.ndarray:
    permutation = np.arange(nAtoms)
    for cycle in operation.cycles:
        for position, atom in enumerate(cycle):
            permutation[atom] = cycle[(position + 1) % len(cycle)]

    return permutation


def _generators(spec: StructureSpec, geometry: str) -> List[Operation]:
    """The symmetry operations of the specification and the ones that express that the molecule is planar/linear (all
    atoms are invariant under mirror planes)"""
    identity = np.arange(spec.nAtoms())
    generators = [(_permutation(op, spec.nAtoms()), op.matrix) for op in spec.symmetry]

    mirrors = {"planar": [2], "linear": [0, 1]}.get(geometry, [])
    for axis in mirrors:
        generators.append((identity, np.diag([1.0 if k != axis else -1.0 for k in range(3)])))

    return generators


def _closure(generators: Sequence[Operation], nAtoms: int) -> List[Operation]:
    """All operations that can be obtained by combining the given ones"""

    def key(operation: Operation) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
        permutation, matrix = operation
        return tuple(permutation.tolist()), tuple((np.round(matrix, 6) + 0.0).ravel().tolist())

    identity: Operation = (np.arange(nAtoms), np.eye(3))
    group = {key(identity): identity}
    queue = [identity]
    while queue:
        current = queue.pop()
        for permutation, matrix in generators:
            # First the current operation, then the generator
            product: Operation = (permutation[current[0]], matrix @ current[1])
            if key(product) not in group:
                group[key(product)] = product
                queue.append(product)
                if len(group) > _MAX_GROUP_SIZE:
                    raise StructureSpecError(
                        "The symmetry operations generate more than %d different operations (is the group finite?)"
                        % _MAX_GROUP_SIZE
                    )

    return list(group.values())


def _checkSymmetry(spec: StructureSpec, generators: Sequence[Operation], group: Sequence[Operation]) -> None:
    """Checks that the symmetry operations are consistent with the atoms, the bonds and the specified parameters"""
    bonds = set(spec.bonds)
    identity = np.arange(spec.nAtoms())

    for permutation, _ in generators:
        for atom, image in enumerate(permutation):
            if spec.elements[atom] != spec.elements[image]:
                raise StructureSpecError(
                    "A symmetry operation moves %s to %s, which is a different element"
                    % (spec.labels[atom], spec.labels[image])
                )
        for i, j in bonds:
            if canonicalBond(int(permutation[i]), int(permutation[j])) not in bonds:
                raise StructureSpecError(
                    "A symmetry operation moves the bond %s - %s to %s - %s, which is not a bond (are atoms missing in the operation?)"
                    % (
                        spec.labels[i],
                        spec.labels[j],
                        spec.labels[permutation[i]],
                        spec.labels[permutation[j]],
                    )
                )

    values = {c.indices: c.value for c in spec.constraints}
    for permutation, matrix in group:
        if not np.array_equal(permutation, identity) and np.allclose(matrix, np.eye(3), atol=1e-6):
            raise StructureSpecError(
                "A combination of the symmetry operations exchanges atoms without moving them"
            )

        # Mirroring changes the sign of dihedral angles
        sign = 1.0 if np.linalg.det(matrix) > 0 else -1.0
        for constraint in spec.constraints:
            image = tuple(int(permutation[i]) for i in constraint.indices)
            canonical = {2: canonicalBond, 3: canonicalAngle, 4: canonicalDihedral}[len(image)](*image)  # type: ignore[arg-type]
            if canonical not in values:
                continue

            expected = sign * constraint.value if len(image) == 4 else constraint.value
            difference = values[canonical] - expected  # type: ignore[index]
            if len(image) == 4:
                difference = float(wrapAngle(difference))
            if abs(difference) > 1e-6:
                raise StructureSpecError(
                    "The parameters %s and %s are equivalent by symmetry but have different values"
                    % (
                        " - ".join(spec.labels[i] for i in constraint.indices),
                        " - ".join(spec.labels[i] for i in canonical),
                    )
                )


def _symmetryBasis(group: Sequence[Operation], nAtoms: int) -> np.ndarray:
    """Orthonormal basis (as columns) of all coordinate vectors (x1,y1,z1,x2,...) that are invariant under the group"""
    projector = np.zeros((3 * nAtoms, 3 * nAtoms))
    for permutation, matrix in group:
        for atom in range(nAtoms):
            image = permutation[atom]
            projector[3 * image : 3 * image + 3, 3 * atom : 3 * atom + 3] += matrix
    projector /= len(group)

    eigenvalues, eigenvectors = np.linalg.eigh((projector + projector.T) / 2)
    basis = eigenvectors[:, eigenvalues > 0.5]
    if basis.shape[1] == 0:
        raise StructureSpecError("The symmetry does not leave any degrees of freedom")

    return basis


def _selectionBasis(nAtoms: int, axes: Sequence[int]) -> np.ndarray:
    """Basis of all coordinate vectors, whose atoms are located on the given Cartesian axes"""
    basis = np.zeros((3 * nAtoms, nAtoms * len(axes)))
    for atom in range(nAtoms):
        for position, axis in enumerate(axes):
            basis[3 * atom + axis, atom * len(axes) + position] = 1.0

    return basis


def _solve(
    problem: _Problem,
    basis: np.ndarray,
    start: np.ndarray,
    restraints: bool,
    symmetry: bool = False,
) -> np.ndarray:
    """Least-squares solution (as array of shape (number of atoms, 3)), starting from the given coordinates and with the
    coordinates restricted to the linear space spanned by the columns of basis. If restraints is set, the weak terms of the
    problem are included, which keep everything unspecified close to the start. The solution is then computed once more
    without them to remove the compromises they cause. Without restraints, only the specified parameters are fulfilled.
    With symmetry, deviations from the symmetry operations of the problem are penalized (in addition)."""

    def solve(q0: np.ndarray, restraints: bool) -> np.ndarray:
        solution = least_squares(
            lambda q: problem.residuals(basis @ q, restraints, symmetry),
            q0,
            jac=lambda q: problem.jacobian(basis @ q, restraints, symmetry) @ basis,
            method="trf",
            max_nfev=1000,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        return solution.x

    solution = basis.T @ start.ravel()
    if restraints:
        solution = solve(solution, True)
    solution = solve(solution, False)

    return (basis @ solution).reshape(-1, 3)


def buildStructure(
    spec: StructureSpec,
    guess: Optional[Molecule] = None,
    guessLabels: Optional[Sequence[str]] = None,
    orient: bool = True,
    maxAttempts: int = 5,
    seed: int = 0,
    geometry: Optional[str] = None,
) -> BuildResult:
    """Computes Cartesian coordinates that fulfill the structure parameters of the given specification.

    All parameters are treated as equations for the coordinates, which are solved by a least-squares optimization starting
    from the given geometry. Without a geometry, a rough one is generated from the connectivity (see defaultGuess).
    The solution is not unique when parameters are missing; the one closest to the starting geometry is returned as all
    other bonds, angles and dihedrals are weakly restrained to their starting values.

    If the parameters are not fulfilled after an attempt, the starting coordinates are randomly displaced (reproducibly,
    depending on the seed) and the optimization is repeated, up to maxAttempts times. The best result is returned and
    BuildResult.converged tells whether all parameters are fulfilled.

    The geometry "planar" or "linear" (default: the one of the specification) restricts all atoms to a plane or a line. This
    is exact, it is not a penalty. A starting geometry is projected onto the plane/line. Note that dihedral angles are
    discrete in a planar molecule (0 or 180 degrees, i.e. cis or trans) and cannot be changed continuously by the
    optimization; they have to be realized by the starting geometry and are reported as not satisfied otherwise.

    Symmetry operations of the specification (see parseStructureSpecString) are enforced exactly: only structures that are
    invariant under the operations are considered. Parameters that are equivalent by symmetry are thereby tied together,
    i.e. it is sufficient to specify one of them (conflicting values are rejected). A starting geometry has to be given in the
    coordinate system of the operations, whose origin is on all symmetry elements. The result is always brought into the
    standard orientation if there is symmetry.

    With orient=True, the result is translated and rotated into the standard orientation (the solution is otherwise
    arbitrary with respect to translation and rotation).
    """
    geometry = spec.geometry if geometry is None else geometry
    _checkGeometry(spec, geometry)

    if spec.symmetry and not orient:
        raise ValueError(
            "Structures with symmetry are always brought into the standard orientation (orient must not be disabled)"
        )

    if guess is not None:
        spec = resolveSpec(spec, guess, guessLabels)

    # Operations with missing atoms are completed with the help of the bonds (and the starting geometry)
    spec = completeSymmetry(spec, guess.coordinates() if guess is not None else None)

    nAtoms = spec.nAtoms()
    generators = _generators(spec, geometry) if spec.symmetry else []
    everything = np.eye(3 * nAtoms)

    if generators:
        group = _closure(generators, nAtoms)
        _checkSymmetry(spec, generators, group)
        basis = _symmetryBasis(group, nAtoms)
    else:
        basis = _selectionBasis(nAtoms, _AXES[geometry])

    if guess is not None:
        if generators:
            projected = (basis @ (basis.T @ guess.coordinates().ravel())).reshape(-1, 3)
            displacement = np.sqrt(np.mean(np.sum((projected - guess.coordinates()) ** 2, axis=1)))
            if displacement > _SYMMETRY_WARNING:
                warnings.warn(
                    "The starting geometry deviates from the symmetry (and planarity/linearity) by %.2f Angstrom (rms), it is "
                    "projected onto the symmetric structures. Note that all symmetry elements must pass through the origin of the "
                    "coordinate system of the starting geometry" % displacement,
                    stacklevel=2,
                )
            start = projected
        else:
            start = _flatten(guess.coordinates(), geometry)
    else:
        start = defaultGuess(spec, geometry)

        if generators:
            # The default guess is not symmetric. Make it so by minimizing the deviation from the symmetry together with the
            # specified parameters before restricting the structure to the symmetric space
            relaxed = _solve(_Problem(spec, start, generators), everything, start, False, True)
            start = (basis @ (basis.T @ relaxed.ravel())).reshape(-1, 3)
        elif geometry == "planar":
            # Rings can only be closed by choosing cis/trans torsions appropriately, which can't be done continuously in a
            # planar molecule. So first fulfill the parameters without restricting the guess to a plane (which twists rings
            # into shape) and use the result, projected onto its best-fit plane, as starting point
            start = _flatten(_solve(_Problem(spec, start), everything, start, False), geometry)

    problem = _Problem(spec, start)

    rng = np.random.default_rng(seed)
    best: Optional[np.ndarray] = None
    bestCost = math.inf
    bestConverged = False
    attempts = 0

    for attempt in range(max(1, maxAttempts)):
        attempts = attempt + 1
        x0 = start.copy()
        if attempt > 0:
            displacement = rng.normal(scale=0.1 * attempt, size=basis.shape[1])
            x0 = x0 + (basis @ displacement).reshape(-1, 3)

        full = _solve(problem, basis, x0, True)
        cost = float(np.sum(problem.residuals(full, restraints=False) ** 2))
        converged = all(r.isSatisfied() for r in _report(spec, full.reshape(-1, 3)))

        if best is None or (converged, -cost) > (bestConverged, -bestCost):
            best, bestCost, bestConverged = full, cost, converged

        if converged:
            break

    assert best is not None

    molecule = Molecule(
        [Atom(element, coordinate) for element, coordinate in zip(spec.elements, best.reshape(-1, 3))]
    )
    if orient:
        molecule.bringToStandardOrientation()

    return BuildResult(
        molecule=molecule,
        labels=list(spec.labels),
        converged=bestConverged,
        attempts=attempts,
        parameters=_report(spec, molecule.coordinates()),
    )
