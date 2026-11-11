#!/usr/bin/env python3

"""Generates an XYZ file from a set of structure parameters (bond lengths, angles and dihedral angles).

The parameters are given in a text file with one parameter per line (see the example below). They are treated as equations
for the Cartesian coordinates, which are solved by an optimization. So the parameters can be anything that is convenient,
in particular they don't have to form a Z-matrix. Whatever is not specified stays close to the starting geometry, which
is either provided via --guess or, if not, automatically generated from the connectivity. No two atoms can come closer than
0.6 Angstrom or 0.75 times the sum of their covalent radii (the latter not for pairs with an explicitly specified distance),
which prevents atoms from being placed on top of each other.
"""

from koehnlab.molecular import (
    readLabeledMolecule,
    writeMolecule,
    parseStructureSpec,
    buildStructure,
    StructureSpecError,
)

import argparse
import sys
from typing import Optional

EXAMPLE = """\
Format of the parameter file (lengths in Angstrom, angles in degrees, everything after '#' is ignored):

    C1 - C2 1.34              # bond length; also states that the atoms are bonded
    C1 - H1                   # only states that the atoms are bonded
    C1 - C2 - C3 123.5        # angle (the atom in the middle is the apex)
    C1 - C2 - C3 - C4 180     # dihedral angle
    C1 - C2 1.34 w=2          # parameters can be weighted (default: 1)
    @planar                   # all atoms lie in one plane (or: @linear for atoms on a line)
    @symmetry C2(z): C1<->C3, H1<->H3     # the molecule is invariant under a rotation about z that exchanges these atoms

Symmetry operations are i, sigma(axis), Cn(axis) and Sn(axis) with the axis x, y, z or a vector a,b,c, all in the same
coordinate system whose origin lies on every symmetry element. The atoms are given as exchanges (C1<->C3) or cycles
(C1->C2->C3, the operation moves C1 to the position of C2 and so on); atoms that are not listed stay where they are, i.e.
are located on the symmetry element. Parameters that are equivalent by symmetry are tied, so it is sufficient to specify one
of them. Structures with symmetry are always brought into the standard orientation.

Atoms of an operation can be left out: '@symmetry C2(z): C1<->C4, ...' (the trailing '...') lets the program infer the other atoms
from the bonds, and '@group C2v' adds the generators of a point group (Cs, Ci, Cn, Cnv, Cnh, Dn, Dnh, Dnd, S2n, T, Td, Th, O, Oh, I,
Ih) in a canonical frame (principal axis z, C2 axes along x, vertical mirror plane xz, horizontal mirror plane xy) with all atoms
inferred. A '@symmetry' line with the same operation as one of the group provides atoms for it ('H1' alone: this atom stays where
it is). If the bonds don't determine the atoms uniquely, the starting geometry (in the same frame) selects the best fit, or the
program asks for more atoms.

Atoms are identified by labels consisting of an element symbol and a number. If a starting geometry is given,
its XYZ file may use such labels instead of plain element symbols (C1 x y z); otherwise the atoms are numbered per
element in the order of the file (C1, C2, ..., H1, H2, ...). The bonds of the starting geometry are detected automatically.
"""


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to the file containing the structure parameters",
        metavar="PATH",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to which the resulting XYZ file shall be written",
        metavar="PATH",
        required=True,
    )
    parser.add_argument(
        "--guess",
        "-g",
        help="XYZ file with a starting geometry. Everything the parameters don't specify stays close to it",
        metavar="PATH",
    )
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument(
        "--planar",
        help="Restrict all atoms to one plane (same as the directive @planar in the parameter file)",
        action="store_true",
    )
    geometry.add_argument(
        "--linear",
        help="Restrict all atoms to one line (same as the directive @linear in the parameter file)",
        action="store_true",
    )
    parser.add_argument(
        "--no-orient",
        help="Don't bring the result into the standard orientation (geometric center at the origin, principal axes along x, y, z). Not possible with symmetry",
        action="store_true",
    )
    parser.add_argument(
        "--labels",
        help="Write the atom labels (C1, H3, ...) instead of plain element symbols to the output",
        action="store_true",
    )
    parser.add_argument(
        "--max-attempts",
        help="Number of attempts (with randomly displaced start) to fulfill all parameters (default: %(default)s)",
        type=int,
        default=5,
        metavar="N",
    )
    parser.add_argument(
        "--seed",
        help="Seed for the random displacements in repeated attempts (default: %(default)s)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="Print how well every parameter is fulfilled",
        action="store_true",
    )

    args = parser.parse_args(argv)

    try:
        geometry = "planar" if args.planar else "linear" if args.linear else None
        spec = parseStructureSpec(args.input, geometry)

        guess = None
        guessLabels = None
        if args.guess is not None:
            guess, guessLabels = readLabeledMolecule(args.guess)

        result = buildStructure(
            spec,
            guess,
            guessLabels,
            orient=not args.no_orient,
            maxAttempts=args.max_attempts,
            seed=args.seed,
            geometry=geometry,
        )
    except (StructureSpecError, ValueError, OSError, RuntimeError) as error:
        print("Error: %s" % error, file=sys.stderr)
        return 2

    writeMolecule(
        result.molecule, args.output, labels=result.labels if args.labels else None
    )

    if args.verbose or not result.converged:
        print(result.format(), file=sys.stderr)

    if not result.converged:
        print(
            "Warning: Not all parameters could be fulfilled (see above). The parameters might be inconsistent.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
