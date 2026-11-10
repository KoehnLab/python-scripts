from typing import Dict, List, Optional, Sequence, Set, Tuple

from collections import deque
from dataclasses import replace

import math

import numpy as np

from .StructureParser import StructureSpec, StructureSpecError, SymmetryOperation

# Upper limit for the number of candidates that are enumerated for a single operation
_MAX_CANDIDATES = 5000
# Upper limit for the size of the search
_MAX_SEARCH_STEPS = 500000
# A starting geometry selects one candidate only if its root mean square deviation from the symmetry (Angstrom) is below this
# value and the best candidate is clearly better than the second best one
_GEOMETRY_ACCEPTANCE = 0.5
_GEOMETRY_MARGIN = 0.1


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
) -> List[Tuple[int, ...]]:
    """All permutations of the atoms that preserve the elements and bonds, contain the atoms given in the operation and have
    cycles whose lengths are compatible with the order of the operation"""
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

    order_k = _matrixOrder(operation.matrix)
    solutions: List[Tuple[int, ...]] = []
    steps = 0

    def search(position: int) -> None:
        nonlocal steps
        steps += 1
        if len(solutions) >= _MAX_CANDIDATES or steps > _MAX_SEARCH_STEPS:
            return

        if position == len(order):
            # The bond graph must be mapped onto itself
            if any(image[j] not in neighbors[image[i]] for i, j in spec.bonds):
                return
            if order_k is not None and any(order_k % len(c) for c in _cycles(image)):
                return
            solutions.append(tuple(image))
            return

        atom = order[position]
        # Try the atom itself first and then the others, so that the result is deterministic
        for target in [atom] + [t for t in range(nAtoms) if t != atom]:
            if preimage[target] != -1 or signature[target] != signature[atom]:
                continue
            if any(image[b] != -1 and image[b] not in neighbors[target] for b in neighbors[atom]):
                continue
            if any(preimage[d] != -1 and preimage[d] not in neighbors[atom] for d in neighbors[target]):
                continue

            image[atom] = target
            preimage[target] = atom
            search(position + 1)
            image[atom] = -1
            preimage[target] = -1

    search(0)

    if steps > _MAX_SEARCH_STEPS or len(solutions) >= _MAX_CANDIDATES:
        raise StructureSpecError(
            "Too many possible atom mappings for the symmetry operation %s, specify more of its atoms" % operation.name
        )

    return solutions


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
    candidates = _candidates(operation, spec)

    if not candidates:
        raise StructureSpecError(
            "No atom mapping exists for the symmetry operation %s that is compatible with the bonds and the given atoms"
            % operation.name
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


def completeSymmetry(spec: StructureSpec, coordinates: Optional[np.ndarray] = None) -> StructureSpec:
    """Returns the specification with the atoms of all partial symmetry operations inferred (see completeOperation)"""
    if not any(operation.partial for operation in spec.symmetry):
        return spec

    return replace(
        spec,
        symmetry=[
            completeOperation(operation, spec, coordinates) if operation.partial else operation
            for operation in spec.symmetry
        ],
    )
