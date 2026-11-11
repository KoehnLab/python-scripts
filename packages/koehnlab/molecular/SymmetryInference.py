from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from collections import deque
from dataclasses import replace

import itertools
import math

import numpy as np

from .InternalCoordinates import canonicalCoordinate, wrapAngle
from .StructureParser import StructureSpec, StructureSpecError, SymmetryOperation

# Upper limit for the number of candidates that are enumerated for a single operation
_MAX_CANDIDATES = 5000
# Upper limit for the size of the search
_MAX_SEARCH_STEPS = 500000
# A starting geometry selects one candidate only if its root mean square deviation from the symmetry (Angstrom) is below this
# value and the best candidate is clearly better than the second best one
_GEOMETRY_ACCEPTANCE = 0.5
_GEOMETRY_MARGIN = 0.1
# Upper limit for the number of operations of a symmetry group
_MAX_GROUP_SIZE = 192
# Step (degrees) of the rotations of the coordinate frame about the axes of the group that are tried when deciding whether two
# assignments of atoms to the operations are the same up to the orientation of the molecule
_FRAME_STEP = 3


def _matrixOrder(matrix: np.ndarray, maximum: int = 12) -> Optional[int]:
    """Smallest n for which the n-th power of the matrix is the identity (None if there is none up to maximum)"""
    power = np.eye(3)
    for order in range(1, maximum + 1):
        power = matrix @ power
        if np.allclose(power, np.eye(3), atol=1e-6):
            return order

    return None


def _cycles(permutation: Sequence[int]) -> List[Tuple[int, ...]]:
    seen = [False] * len(permutation)
    cycles: List[Tuple[int, ...]] = []
    for start in range(len(permutation)):
        if seen[start]:
            continue
        cycle = [start]
        seen[start] = True
        while permutation[cycle[-1]] != start:
            cycle.append(permutation[cycle[-1]])
            seen[cycle[-1]] = True
        cycles.append(tuple(cycle))

    return cycles


def _candidates(
    operation: SymmetryOperation, spec: StructureSpec
) -> Tuple[List[Tuple[int, ...]], str]:
    """All permutations of the atoms that preserve the elements and bonds, contain the atoms given in the operation and have
    cycles whose lengths are compatible with the order of the operation. If there are none, the second return value explains
    where the search failed."""
    nAtoms = spec.nAtoms()
    neighbors: List[Set[int]] = [set() for _ in range(nAtoms)]
    for i, j in spec.bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    signature = [
        (spec.elements[i].atomicNumber(), len(neighbors[i]), tuple(sorted(spec.elements[j].atomicNumber() for j in neighbors[i])))
        for i in range(nAtoms)
    ]
    labels = spec.labels

    image = [-1] * nAtoms
    preimage = [-1] * nAtoms

    def assign(atom: int, target: int) -> None:
        if image[atom] != -1 and image[atom] != target:
            raise StructureSpecError(
                "Contradicting atoms in the symmetry operation %s: %s is moved to different atoms"
                % (operation.name, labels[atom])
            )
        if signature[atom] != signature[target]:
            raise StructureSpecError(
                "The symmetry operation %s moves %s to %s, which have a different element or bonds"
                % (operation.name, labels[atom], labels[target])
            )
        image[atom] = target
        preimage[target] = atom

    for cycle in operation.cycles:
        isOpen = operation.partial and cycle in operation.chains
        for position, atom in enumerate(cycle):
            if isOpen and position == len(cycle) - 1:
                continue
            assign(atom, cycle[(position + 1) % len(cycle)])
    for atom in operation.fixed:
        assign(atom, atom)

    # Order of the remaining atoms: those bonded to already known ones first, so that the bonds restrict the choices
    known = [i for i in range(nAtoms) if image[i] != -1]
    queue = deque(known)
    queued = set(known)
    order: List[int] = []
    while queue or len(queued) < nAtoms:
        if not queue:
            first = next(i for i in range(nAtoms) if i not in queued)
            queue.append(first)
            queued.add(first)
            order.append(first)
        current = queue.popleft()
        for neighbor in sorted(neighbors[current]):
            if neighbor not in queued:
                queued.add(neighbor)
                queue.append(neighbor)
                order.append(neighbor)
    order = [atom for atom in order if image[atom] == -1]

    def bondsOf(atom: int) -> str:
        count = len(neighbors[atom])
        return "%d bond%s: %s" % (count, "" if count == 1 else "s", ", ".join(labels[n] for n in sorted(neighbors[atom])))

    def _explainFailure(atom: int) -> str:
        """Describes why no image can be found for the atom, in the current state of the search"""
        text = "The mapping breaks down at %s (%s, %s)" % (labels[atom], spec.elements[atom].symbol(), bondsOf(atom))

        anchor = next((b for b in sorted(neighbors[atom]) if image[b] != -1), None)
        if anchor is None:
            return text + ": no unused atom with the same element and number of bonds is available"

        candidates = [
            t
            for t in sorted(neighbors[image[anchor]])
            if spec.elements[t] == spec.elements[atom] and preimage[t] == -1
        ]
        text += ": %s is moved to %s, so %s would have to be moved to an atom of the same element bonded to %s" % (
            labels[anchor],
            labels[image[anchor]],
            labels[atom],
            labels[image[anchor]],
        )
        if not candidates:
            return text + ", but there is none"
        return text + ", but %s %s (%s)" % (
            ", ".join(labels[t] for t in candidates),
            "has different bonds" if len(candidates) == 1 else "have different bonds",
            "; ".join("%s: %s" % (labels[t], bondsOf(t)) for t in candidates),
        )

    order_k = _matrixOrder(operation.matrix)
    solutions: List[Tuple[int, ...]] = []
    steps = 0
    # The atom, at which the search got furthest without finding a possible image (position in the search and explanation)
    stuck: Tuple[int, str] = (-1, "")
    rejectedByOrder = 0

    def search(position: int) -> None:
        nonlocal steps, stuck, rejectedByOrder
        steps += 1
        if len(solutions) >= _MAX_CANDIDATES or steps > _MAX_SEARCH_STEPS:
            return

        if position == len(order):
            # The bond graph must be mapped onto itself
            if any(image[j] not in neighbors[image[i]] for i, j in spec.bonds):
                return
            if order_k is not None and any(order_k % len(c) for c in _cycles(image)):
                rejectedByOrder += 1
                return
            solutions.append(tuple(image))
            return

        atom = order[position]
        found = False
        # Try the atom itself first and then the others, so that the result is deterministic
        for target in [atom] + [t for t in range(nAtoms) if t != atom]:
            if preimage[target] != -1 or signature[target] != signature[atom]:
                continue
            if any(image[b] != -1 and image[b] not in neighbors[target] for b in neighbors[atom]):
                continue
            if any(preimage[d] != -1 and preimage[d] not in neighbors[atom] for d in neighbors[target]):
                continue

            found = True
            image[atom] = target
            preimage[target] = atom
            search(position + 1)
            image[atom] = -1
            preimage[target] = -1

        if not found and position > stuck[0]:
            stuck = (position, _explainFailure(atom))

    search(0)

    if steps > _MAX_SEARCH_STEPS or len(solutions) >= _MAX_CANDIDATES:
        raise StructureSpecError(
            "Too many possible atom mappings for the symmetry operation %s, specify more of its atoms" % operation.name
        )

    explanation = ""
    if not solutions:
        if rejectedByOrder and stuck[0] < 0:
            explanation = (
                "Atom mappings exist, but their cycles are not compatible with the operation (applying it %s times has to restore "
                "all atoms)" % order_k
            )
        elif stuck[0] >= 0:
            explanation = stuck[1] + ". Check for missing or extra bonds around these atoms"

    return solutions, explanation


def _describe(permutation: Sequence[int], labels: Sequence[str], limit: int = 4) -> str:
    parts = ["%s->%s" % (labels[i], labels[t]) for i, t in enumerate(permutation) if i != t]
    return ", ".join(parts[:limit]) + (", ..." if len(parts) > limit else "") if parts else "all atoms fixed"


def completeOperation(
    operation: SymmetryOperation, spec: StructureSpec, coordinates: Optional[np.ndarray] = None
) -> SymmetryOperation:
    """Infers the missing atoms of a partial symmetry operation from the bonds of the specification: The operation has to map
    the bond graph onto itself while preserving the elements, contain the given atoms and have a permutation whose cycles
    are compatible with the order of the operation. If this still leaves several possibilities, the coordinates of a starting
    geometry (in the frame of the operation) can select one by choosing the permutation that the matrix of the operation
    maps best onto the geometry. If it is still ambiguous, an error is raised."""
    candidates, explanation = _candidates(operation, spec)

    if not candidates:
        raise StructureSpecError(
            "No atom mapping exists for the symmetry operation %s that is compatible with the bonds and the given atoms.%s"
            % (operation.name, " " + explanation if explanation else "")
        )

    if len(candidates) > 1:
        if coordinates is not None:
            rotated = coordinates @ operation.matrix.T
            costs = [
                float(np.sqrt(np.mean(np.sum((rotated - coordinates[list(c)]) ** 2, axis=1))))
                for c in candidates
            ]
            ranking = np.argsort(costs, kind="stable")
            best = costs[ranking[0]]
            second = costs[ranking[1]]
            if best < _GEOMETRY_ACCEPTANCE and second - best > _GEOMETRY_MARGIN and second > 2 * best:
                candidates = [candidates[ranking[0]]]

        if len(candidates) > 1:
            raise StructureSpecError(
                "The symmetry operation %s is ambiguous: %d atom mappings are compatible with the bonds and the given atoms "
                "(e.g. '%s' or '%s'). Specify more atoms of the operation%s"
                % (
                    operation.name,
                    len(candidates),
                    _describe(candidates[0], spec.labels),
                    _describe(candidates[1], spec.labels),
                    "." if coordinates is None else " or provide a starting geometry that is closer to the symmetry.",
                )
            )

    permutation = candidates[0]
    return SymmetryOperation(
        operation.name,
        operation.matrix,
        tuple(c for c in _cycles(permutation) if len(c) > 1),
        tuple(i for i, t in enumerate(permutation) if i == t),
        False,
    )


Permutation = Tuple[int, ...]
_Table = Dict[Tuple[float, ...], Tuple[Permutation, np.ndarray]]


def _matrixKey(matrix: np.ndarray) -> Tuple[float, ...]:
    return tuple((np.round(matrix, 6) + 0.0).ravel().tolist())


class _MatrixGroup:
    """The group generated by the matrices of the symmetry operations, with the effect of every generator on every element"""

    def __init__(self, generators: Sequence[np.ndarray]):
        self.matrices: List[np.ndarray] = [np.eye(3)]
        self.keys: List[Tuple[float, ...]] = [_matrixKey(np.eye(3))]
        index = {self.keys[0]: 0}
        # step[g][i]: index of the element that results from the element i followed by the generator g
        self.step: List[List[int]] = [[] for _ in generators]

        position = 0
        while position < len(self.matrices):
            for g, generator in enumerate(generators):
                product = generator @ self.matrices[position]
                key = _matrixKey(product)
                if key not in index:
                    index[key] = len(self.matrices)
                    self.matrices.append(product)
                    self.keys.append(key)
                    if len(self.matrices) > _MAX_GROUP_SIZE:
                        raise StructureSpecError(
                            "The symmetry operations generate more than %d different operations (is the group finite?)"
                            % _MAX_GROUP_SIZE
                        )
                self.step[g].append(index[key])
            position += 1


def _groupTable(
    permutations: Sequence[Tuple[int, Permutation]], group: _MatrixGroup, nAtoms: int
) -> Optional[_Table]:
    """All combinations of the operations (given as pairs of an index of a generator of the matrix group and the permutation of
    the atoms), indexed by the matrix. Returns None if the operations are inconsistent, i.e. if the same matrix would have to
    move the atoms in different ways. The matrices act on the positions of the atoms, the permutation moves the atom a to the
    position of the atom permutation[a]."""
    assigned: Dict[int, Permutation] = {0: tuple(range(nAtoms))}
    queue = [0]
    while queue:
        current = queue.pop()
        for generator, permutation in permutations:
            # First the current operation, then the generator
            product = tuple(permutation[i] for i in assigned[current])
            target = group.step[generator][current]
            if target in assigned:
                if assigned[target] != product:
                    return None
                continue
            assigned[target] = product
            queue.append(target)

    return {group.keys[i]: (permutation, group.matrices[i]) for i, permutation in assigned.items()}


def _confinement(table: _Table, nAtoms: int) -> int:
    """Dimension of the space that the atoms are confined to because some operations leave all atoms where they are (3 if there
    are no such operations; the operations then require, e.g., a planar molecule)"""
    identity = tuple(range(nAtoms))
    rows = [matrix - np.eye(3) for permutation, matrix in table.values() if permutation == identity]
    return 3 - (int(np.linalg.matrix_rank(np.vstack(rows), tol=1e-6)) if rows else 0)


def _forcesCoincidence(table: _Table, nAtoms: int) -> bool:
    """Whether the symmetry forces two atoms to be at the same position, so that no actual structure has this symmetry"""
    projector = np.zeros((3 * nAtoms, 3 * nAtoms))
    for permutation, matrix in table.values():
        for atom in range(nAtoms):
            image = permutation[atom]
            projector[3 * image : 3 * image + 3, 3 * atom : 3 * atom + 3] += matrix
    projector /= len(table)

    generic = (projector @ np.random.default_rng(12345).normal(size=3 * nAtoms)).reshape(nAtoms, 3)
    distances = np.linalg.norm(generic[:, None] - generic[None, :], axis=2) + np.eye(nAtoms) * 10
    return bool(np.min(distances) < 1e-6)


class _Orbits:
    """Classifies the parameters of a molecule by the symmetry group: Parameters in the same orbit have the same value (up to
    the sign of dihedral angles under reflections)"""

    def __init__(self, table: _Table):
        self._elements = [(permutation, 1 if np.linalg.det(matrix) > 0 else -1) for permutation, matrix in table.values()]
        self._cache: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], Set[int]]] = {}

    def key(self, indices: Tuple[int, ...], value: float) -> Tuple[Tuple[int, ...], float]:
        """Identifies the orbit of a parameter together with its value as seen from the representative of the orbit"""
        if indices not in self._cache:
            images: Dict[Tuple[int, ...], Set[int]] = {}
            for permutation, sign in self._elements:
                image = canonicalCoordinate(tuple(permutation[i] for i in indices))
                images.setdefault(image, set()).add(sign if len(indices) == 4 else 1)
            representative = min(images)
            self._cache[indices] = (representative, images[representative])

        representative, signs = self._cache[indices]
        if len(indices) != 4:
            return representative, round(value, 6)
        if len(signs) == 2:
            return representative, round(abs(float(wrapAngle(value))), 6)
        return representative, round(float(wrapAngle(next(iter(signs)) * value)), 6)


def _symmetryConflict(spec: StructureSpec, orbits: _Orbits) -> Optional[str]:
    """Describes two parameters that are equivalent by symmetry but have different values (None if there are none)"""
    seen: Dict[Tuple[int, ...], Tuple[float, Tuple[int, ...]]] = {}
    for constraint in spec.constraints:
        representative, value = orbits.key(constraint.indices, constraint.value)
        if representative in seen and abs(seen[representative][0] - value) > 1e-5:
            return "The parameters %s and %s are equivalent by symmetry but have different values" % (
                " - ".join(spec.labels[i] for i in seen[representative][1]),
                " - ".join(spec.labels[i] for i in constraint.indices),
            )
        seen.setdefault(representative, (value, constraint.indices))

    return None


def _frames(table: _Table) -> List[np.ndarray]:
    """Orientations of the coordinate frame (proper rotations) that map the symmetry group onto itself: Rotating the molecule
    in this way does not change its structure. Different assignments of the atoms to the operations that only differ by such a
    rotation (for example, which of two mirror planes contains the atoms) give the same structure."""
    matrices = [matrix for _, matrix in table.values()]
    keys = {_matrixKey(matrix) for matrix in matrices}

    candidates: List[np.ndarray] = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            frame = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                frame[row, column] = sign
            if np.linalg.det(frame) > 0:
                candidates.append(frame)

    # Rotations about the axes of the rotations in the group
    axes: List[np.ndarray] = []
    for matrix in matrices:
        if np.linalg.det(matrix) < 0 or np.allclose(matrix, np.eye(3), atol=1e-6):
            continue
        values, vectors = np.linalg.eig(matrix)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1)))])
        axis = axis / np.linalg.norm(axis)
        if not any(np.allclose(axis, other, atol=1e-6) or np.allclose(axis, -other, atol=1e-6) for other in axes):
            axes.append(axis)
    for axis in axes:
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        for degrees in range(_FRAME_STEP, 360, _FRAME_STEP):
            angle = math.radians(degrees)
            candidates.append(np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross))

    frames: List[np.ndarray] = [np.eye(3)]
    for frame in candidates:
        if all(_matrixKey(frame @ matrix @ frame.T) in keys for matrix in matrices):
            frames.append(frame)

    return frames


class _Solution:
    """An assignment of permutations to the incomplete operations"""

    def __init__(self, permutations: Tuple[Permutation, ...], table: _Table):
        self.permutations = permutations
        self.table = table
        self.orbits = _Orbits(table)


def _sameStructure(
    first: _Solution,
    second: _Solution,
    generatorKeys: Sequence[Tuple[float, ...]],
    frameKeys: Sequence[Sequence[Tuple[float, ...]]],
    spec: StructureSpec,
    signature: Sequence[object],
    neighbors: Sequence[Set[int]],
) -> bool:
    """Whether the two solutions give the same structure, up to renaming atoms (that are equivalent in the bond graph) and the
    orientation of the molecule: The renaming has to turn the operations of the first solution into the operations of the
    second (in some orientation) and the given parameters have to be tied in the same way."""
    nAtoms = spec.nAtoms()
    firstPermutations = [first.table[key][0] for key in generatorKeys]

    for keys in frameKeys:
        secondPermutations = [second.table[key][0] for key in keys]

        image = [-1] * nAtoms
        used = [False] * nAtoms

        def assign(atom: int, target: int, newly: List[int]) -> bool:
            if used[target] or signature[atom] != signature[target]:
                return False
            image[atom] = target
            used[target] = True
            newly.append(atom)
            return True

        def spread(atom: int, target: int) -> Optional[List[int]]:
            newly: List[int] = []
            ok = assign(atom, target, newly)
            queue = [atom]
            while ok and queue:
                current = queue.pop()
                for moved, moves in zip(firstPermutations, secondPermutations):
                    neighbor, wanted = moved[current], moves[image[current]]
                    if image[neighbor] == -1:
                        ok = assign(neighbor, wanted, newly)
                        queue.append(neighbor)
                        if not ok:
                            break
                    elif image[neighbor] != wanted:
                        ok = False
                        break
            if ok:
                return newly
            for atom_ in newly:
                used[image[atom_]] = False
                image[atom_] = -1
            return None

        def accept() -> bool:
            if any(image[j] not in neighbors[image[i]] for i, j in spec.bonds):
                return False
            tied = {second.orbits.key(c.indices, c.value) for c in spec.constraints}
            mapped = {
                second.orbits.key(canonicalCoordinate(tuple(image[i] for i in c.indices)), c.value)
                for c in spec.constraints
            }
            return tied == mapped

        def search() -> bool:
            atom = next((a for a in range(nAtoms) if image[a] == -1), None)
            if atom is None:
                return accept()
            for target in range(nAtoms):
                newly = spread(atom, target)
                if newly is None:
                    continue
                if search():
                    return True
                for atom_ in newly:
                    used[image[atom_]] = False
                    image[atom_] = -1
            return False

        if search():
            return True

    return False


def _describeSolution(
    solution: _Solution, names: Sequence[str], labels: Sequence[str]
) -> str:
    return "; ".join("%s: %s" % (name, _describe(permutation, labels)) for name, permutation in zip(names, solution.permutations))


# Maximum number of inequivalent completions of the symmetry that are returned
_MAX_ALTERNATIVES = 6


def completeSymmetry(
    spec: StructureSpec, coordinates: Optional[np.ndarray] = None, geometry: Optional[str] = None
) -> StructureSpec:
    """Returns the specification with the atoms of all incomplete symmetry operations inferred (see symmetryAlternatives).
    An error is raised if this is ambiguous."""
    alternatives = symmetryAlternatives(spec, coordinates, geometry)
    if len(alternatives) > 1:
        raise StructureSpecError(describeAmbiguity(spec, alternatives, coordinates))

    return alternatives[0]


def describeAmbiguity(spec: StructureSpec, alternatives: Sequence[StructureSpec], coordinates: Optional[np.ndarray]) -> str:
    partial = [k for k, operation in enumerate(spec.symmetry) if operation.partial]
    names = [spec.symmetry[k].name for k in partial]

    def describe(completed: StructureSpec) -> str:
        parts = []
        for k in partial:
            operation = completed.symmetry[k]
            permutation = list(range(spec.nAtoms()))
            for cycle in operation.cycles:
                for position, atom in enumerate(cycle):
                    permutation[atom] = cycle[(position + 1) % len(cycle)]
            parts.append("%s: %s" % (operation.name, _describe(permutation, spec.labels)))
        return "; ".join(parts)

    return (
        "The symmetry operation%s %s %s ambiguous: several atom mappings are compatible with the bonds and the given atoms "
        "(e.g. '%s' or '%s'). Specify more atoms of the operation%s%s"
        % (
            "s" if len(partial) > 1 else "",
            ", ".join(names),
            "are" if len(partial) > 1 else "is",
            describe(alternatives[0]),
            describe(alternatives[1]),
            "s" if len(partial) > 1 else "",
            "." if coordinates is None else " or provide a starting geometry that is closer to the symmetry.",
        )
    )


def symmetryAlternatives(
    spec: StructureSpec, coordinates: Optional[np.ndarray] = None, geometry: Optional[str] = None
) -> List[StructureSpec]:
    """Returns the specifications with the atoms of all incomplete symmetry operations inferred. There is more than one
    specification only if the symmetry cannot be completed uniquely, in which case they are the possibilities that give
    different structures (at most _MAX_ALTERNATIVES, and none of them is better than the others as far as the symmetry and the
    bonds are concerned; the parameters may favor one).

    The incomplete operations are completed together: The atoms must be moved such that every operation maps the bond graph onto
    itself, contains the atoms that were given and has a cycle structure that fits the order of the operation (see
    completeOperation), and in addition all operations must be consistent with each other, i.e. the combinations of the
    operations must act in the same way on the atoms as the corresponding combinations of their matrices. Several possibilities
    that remain are the same if they only differ by the names of equivalent atoms and the orientation of the molecule. The
    coordinates of a starting geometry (in the frame of the operations) are used to select one of the remaining possibilities."""
    partial = [k for k, operation in enumerate(spec.symmetry) if operation.partial]
    if not partial:
        return [spec]

    nAtoms = spec.nAtoms()
    geometry = spec.geometry if geometry is None else geometry
    identity = tuple(range(nAtoms))

    # Operations that are known: the complete ones and the mirror planes that express planarity/linearity
    known: List[Tuple[Permutation, np.ndarray]] = []
    for k, operation in enumerate(spec.symmetry):
        if not operation.partial:
            permutation = list(identity)
            for cycle in operation.cycles:
                for position, atom in enumerate(cycle):
                    permutation[atom] = cycle[(position + 1) % len(cycle)]
            known.append((tuple(permutation), operation.matrix))
    for axis in {"planar": [2], "linear": [0, 1]}.get(geometry, []):
        known.append((identity, np.diag([1.0 if k != axis else -1.0 for k in range(3)])))

    candidates: List[List[Permutation]] = []
    for k in partial:
        found, explanation = _candidates(spec.symmetry[k], spec)
        if not found:
            raise StructureSpecError(
                "No atom mapping exists for the symmetry operation %s that is compatible with the bonds and the given atoms.%s"
                % (spec.symmetry[k].name, " " + explanation if explanation else "")
            )
        candidates.append(found)

    # Operations that leave all atoms in place confine the molecule to their symmetry element: a nonlinear molecule needs a plane
    minimumDimension = 1 if geometry == "linear" or nAtoms <= 2 else 2

    # All consistent combinations; the operation with the fewest possibilities first
    order = sorted(range(len(partial)), key=lambda position: len(candidates[position]))
    generatorMatrices = [matrix for _, matrix in known] + [spec.symmetry[k].matrix for k in partial]
    group = _MatrixGroup(generatorMatrices)
    solutions: List[_Solution] = []
    chosen: List[Optional[Permutation]] = [None] * len(partial)
    steps = 0

    def search(depth: int) -> None:
        nonlocal steps
        steps += 1
        if len(solutions) >= _MAX_CANDIDATES or steps > _MAX_SEARCH_STEPS:
            return
        operations = [(g, permutation) for g, (permutation, _) in enumerate(known)]
        for position in order[:depth]:
            permutation = chosen[position]
            assert permutation is not None
            operations.append((len(known) + position, permutation))
        table = _groupTable(operations, group, nAtoms)
        if table is None or _confinement(table, nAtoms) < minimumDimension:
            return
        if depth == len(order):
            solutions.append(_Solution(tuple(p for p in chosen if p is not None), table))
            return
        position = order[depth]
        for permutation in candidates[position]:
            chosen[position] = permutation
            search(depth + 1)
        chosen[position] = None

    search(0)
    if steps > _MAX_SEARCH_STEPS or len(solutions) >= _MAX_CANDIDATES:
        raise StructureSpecError(
            "Too many possible atom mappings for the symmetry operations %s, specify more of their atoms"
            % ", ".join(spec.symmetry[k].name for k in partial)
        )

    names = [spec.symmetry[k].name for k in partial]
    if not solutions:
        raise StructureSpecError(
            "The symmetry operations %s cannot be completed consistently: for each of them, atom mappings exist that fit the "
            "bonds and the given atoms, but no choice makes their combinations consistent with each other. Check the atoms "
            "that were given" % ", ".join(names)
        )

    solutions = [solution for solution in solutions if not _forcesCoincidence(solution.table, nAtoms)]
    if not solutions:
        raise StructureSpecError(
            "No atom mapping for the symmetry operations %s gives a structure with different positions of all atoms; check "
            "the given atoms" % ", ".join(names)
        )

    valid = [solution for solution in solutions if _symmetryConflict(spec, solution.orbits) is None]
    if not valid:
        raise StructureSpecError(_symmetryConflict(spec, solutions[0].orbits) or "Inconsistent symmetry")
    solutions = valid

    selected: Optional[_Solution] = None
    if len(solutions) == 1:
        selected = solutions[0]

    if selected is None and coordinates is not None:
        costs = []
        for solution in solutions:
            total = 0.0
            for position, k in enumerate(partial):
                rotated = coordinates @ spec.symmetry[k].matrix.T
                total += float(np.mean(np.sum((rotated - coordinates[list(solution.permutations[position])]) ** 2, axis=1)))
            costs.append(float(np.sqrt(total / len(partial))))
        ranking = np.argsort(costs, kind="stable")
        best, second = costs[ranking[0]], costs[ranking[1]]
        if best < _GEOMETRY_ACCEPTANCE and second - best > _GEOMETRY_MARGIN and second > 2 * best:
            selected = solutions[int(ranking[0])]

    classes: List[_Solution] = [selected] if selected is not None else []
    if selected is None:
        # Solutions that give the same structure are not different alternatives
        generatorKeys = [_matrixKey(matrix) for matrix in generatorMatrices]
        frameKeys = [
            [_matrixKey(frame @ matrix @ frame.T) for matrix in generatorMatrices] for frame in _frames(solutions[0].table)
        ]
        neighbors: List[Set[int]] = [set() for _ in range(nAtoms)]
        for i, j in spec.bonds:
            neighbors[i].add(j)
            neighbors[j].add(i)
        signature = [
            (spec.elements[i].atomicNumber(), len(neighbors[i]), tuple(sorted(spec.elements[j].atomicNumber() for j in neighbors[i])))
            for i in range(nAtoms)
        ]
        for solution in solutions:
            if not any(
                _sameStructure(representative, solution, generatorKeys, frameKeys, spec, signature, neighbors)
                for representative in classes
            ):
                classes.append(solution)
                if len(classes) > _MAX_ALTERNATIVES:
                    raise StructureSpecError(
                        "The symmetry operations %s are ambiguous: more than %d different atom mappings are compatible with the "
                        "bonds and the given atoms. Specify more atoms of the operations" % (", ".join(names), _MAX_ALTERNATIVES)
                    )

    alternatives: List[StructureSpec] = []
    for solution in classes:
        completed = list(spec.symmetry)
        for position, k in enumerate(partial):
            permutation = solution.permutations[position]
            completed[k] = SymmetryOperation(
                spec.symmetry[k].name,
                spec.symmetry[k].matrix,
                tuple(c for c in _cycles(permutation) if len(c) > 1),
                tuple(i for i, t in enumerate(permutation) if i == t),
                False,
            )
        alternatives.append(replace(spec, symmetry=completed))

    return alternatives
